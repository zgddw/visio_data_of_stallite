# -*- coding: utf-8 -*-
"""SolO EAS1/EAS2 nominal mode 3-D PSD -> pitch angle, event-scoped downloads.
Uses actual CDF support axes, EAS->SRF orientation, contemporaneous MAG-SRF.
No extrapolation across missing data, no fictitious pitch angles.
"""
from __future__ import annotations
import os, re, time, warnings
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd

_PYCDF = None


def _load_pycdf():
    """SpacePy is initialized only when local CDF data are actually read."""
    global _PYCDF
    if _PYCDF is None:
        began = time.perf_counter()
        print('[PAD] 正在按需加载 SpacePy CDF 读取器 ...', flush=True)
        from spacepy import pycdf as reader
        _PYCDF = reader
        print(f'[PAD] SpacePy 加载完成，耗时 {time.perf_counter()-began:.2f} s', flush=True)
    return _PYCDF

SOLO_READER_VERSION = 'SOLO_NM3D_PSD_20261021_V2_DOC_VERIFIED'


def _parse_time_array(raw):
    """将 SpacePy CDF 时间轴转换为无时区 UTC 时间索引。

    使用本地实现，避免读取 PAD 时再次导入绘图公共模块。
    """
    arr = np.asarray(raw)
    try:
        converted = pd.to_datetime(arr, utc=True, errors='coerce')
        idx = pd.DatetimeIndex(converted)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        if idx.notna().any():
            return idx
    except Exception:
        pass
    values = []
    for value in arr.ravel():
        try:
            ts = pd.to_datetime(value, utc=True, errors='coerce')
            if pd.isna(ts):
                values.append(pd.NaT)
            elif getattr(ts, 'tzinfo', None) is not None:
                values.append(pd.Timestamp(ts).tz_convert(None))
            else:
                values.append(pd.Timestamp(ts))
        except Exception:
            values.append(pd.NaT)
    return pd.DatetimeIndex(values)


def _interval_from_name(path):
    stamps = re.findall(r'20\d{6}[tT]\d{6}', path.name)
    if len(stamps) >= 2:
        a, b = (pd.Timestamp(pd.to_datetime(x,format='%Y%m%dT%H%M%S')) for x in (stamps[0].upper(),stamps[1].upper()))
        return a,b
    # MAG-SRF 以单个 YYYYMMDD 命名，每个文件对应一天。
    daily = re.search(r'_(20\d{6})(?=_v\d+\.cdf$)', path.name, re.IGNORECASE)
    if daily:
        a = pd.Timestamp(pd.to_datetime(daily.group(1), format='%Y%m%d'))
        return a, a + pd.Timedelta(days=1)
    return None


def _local_files(root, prefix, start, end):
    if not Path(root).exists(): return []
    selected=[]
    root=Path(root)
    candidates=list(root.glob('*.cdf'))
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year+1):
        candidates.extend((root/str(year)).glob('*.cdf'))
    for p in candidates:
        if not p.name.lower().startswith(prefix.lower()): continue
        cover = _interval_from_name(p)
        if cover is not None and cover[1]>=start and cover[0]<=end: selected.append(p)
    # Prefer highest version for duplicate physical intervals
    best={}
    for p in sorted(selected):
        ident = re.sub(r'_v\d+\.cdf$', '', p.name.lower())
        if ident not in best or p.name.lower()>best[ident].name.lower(): best[ident]=p
    return sorted(best.values())


def _candidate_urls(result):
    if isinstance(result, list): entries=result
    elif isinstance(result,dict):
        entries=result.get('FileDescription',result.get('files',result.get('File',[])))
        if not isinstance(entries,list): entries=[entries]
    else: entries=[]
    urls=[]
    for x in entries:
        uri = x.get('Name') if isinstance(x,dict) else x
        if isinstance(uri,str) and urlparse(uri).scheme=='https': urls.append(uri)
    return sorted(set(urls))


def _download_one(session,url,path,retries=2):
    part=path.with_name(path.name+'.part')
    for attempt in range(retries):
        try:
            print(f'[PAD下载] {path.name} 第 {attempt+1}/{retries} 次连接',flush=True)
            with session.get(url, stream=True, timeout=(8,30)) as r:
                r.raise_for_status()
                length=int(r.headers.get('Content-Length','0') or 0)
                with open(part,'wb') as f:
                    total=0
                    reported=time.perf_counter()
                    for block in r.iter_content(chunk_size=1024*1024):
                        if block:
                            total+=f.write(block)
                            now=time.perf_counter()
                            if now-reported>=5:
                                print(f'[PAD下载] {path.name}: {total/1048576:.1f} / {length/1048576:.1f} MiB',flush=True)
                                reported=now
                if total<1024 or (length and total!=length):
                    raise IOError(f'下载文件大小不匹配：got={total}, expected={length}')
            part.replace(path)
            print(f'[PAD下载] 完成 {path.name} ({total/1048576:.1f} MiB)',flush=True)
            return
        except Exception as exc:
            print(f'[PAD下载] {path.name} 失败：{type(exc).__name__}: {exc}',flush=True)
            if part.exists(): part.unlink()
            if attempt==retries-1: raise
            time.sleep(min(3*(2**attempt),15))


def _covered_by_local(files,start,end,slack_seconds=90):
    """Whether downloaded filenames cover the requested time continuously."""
    intervals=sorted([v for f in files if (v:=_interval_from_name(f)) is not None])
    if not intervals:return False
    cursor=pd.Timestamp(start)
    slack=pd.Timedelta(seconds=slack_seconds)
    for a,b in intervals:
        if a>cursor+slack:return False
        if b>=cursor:cursor=max(cursor,b)
        if cursor>=pd.Timestamp(end)-slack:return True
    return False


def ensure_eas_files(config,start,end):
    """Request manifest only if no local file overlaps requested times.
    If at least one local file overlaps, do not blindly assume complete coverage: query
    the remote file list once and fetch only missing filenames. Manifest cached by day.
    """
    root=Path(config.get('NM3D_ROOT', r'H:\solo\swa'))
    offline=bool(config.get('PAD_OFFLINE',False))
    notes=[]
    if offline:
        return ['离线模式：直接读取已有 EAS1/EAS2 文件']
    from cdasws import CdasWs
    root.mkdir(parents=True,exist_ok=True)
    cli=CdasWs(timeout=(8, 15))
    # cdasws 实际使用 _session；设置 session 属性不会改变其网络请求。
    cli._session.trust_env=bool(config.get('PAD_USE_SYSTEM_PROXY',True))
    # Get original files per calendar day, cached to avoid many repeated requests.
    cache=config.setdefault('_NM3D_MANIFEST_CACHE',{})
    for sensor in (1,2):
        dataset=f'SOLO_L2_SWA-EAS{sensor}-NM3D-PSD'
        folder=root/dataset
        folder.mkdir(parents=True,exist_ok=True)
        prefix=f'solo_l2_swa-eas{sensor}-nm3d-psd'
        dayseries=pd.date_range(start.floor('D'),end.floor('D'),freq='D')
        for d in dayseries:
            segment_start=max(start,d)
            segment_end=min(end,d+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1))
            local_files=_local_files(folder,prefix,segment_start,segment_end)
            print(f'[PAD检查] EAS{sensor} {d.date()}：本地找到 {len(local_files)} 个文件。', flush=True)
            if _covered_by_local(local_files,segment_start,segment_end):
                continue
            key=(sensor,str(d.date()))
            if key not in cache:
                a=d.strftime('%Y-%m-%dT00:00:00Z')
                b=(d+pd.Timedelta(days=1)-pd.Timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
                print(f'[PAD网络] 正在查询 {dataset} {d.date()} 的 NASA 文件清单 ...', flush=True)
                for attempt in range(2):
                    try:
                        print(f'[PAD网络] {dataset} {d.date()} 第 {attempt+1}/2 次查询', flush=True)
                        status,res=cli.get_original_files(dataset,a,b)
                        if status==200:
                            cache[key]=_candidate_urls(res);break
                        if status in (204,404):
                            cache[key]=[];break
                        raise RuntimeError(f'HTTP {status}: {str(res)[:180]}')
                    except Exception as exc:
                        if attempt==1:
                            notes.append(f'{dataset} {d.date()}: 清单获取失败 {exc}')
                            print(f'[PAD网络] 清单获取失败：{type(exc).__name__}: {exc}', flush=True)
                        else: time.sleep(2**attempt)
            for url in cache.get(key,[]):
                name=Path(urlparse(url).path).name
                if not name.lower().startswith(prefix) or not name.lower().endswith('.cdf'): continue
                # 按文件覆盖时间所属年份保存；读取端兼容直存文件和年份目录。
                cover=_interval_from_name(Path(name))
                year_folder=folder/str(cover[0].year if cover else d.year)
                year_folder.mkdir(parents=True,exist_ok=True)
                dest=year_folder/name
                # Existing nontrivial completed files are never downloaded again.
                existing = [p for p in folder.rglob(name) if p.is_file() and p.stat().st_size > 1024]
                if existing:
                    continue
                try:
                    print(f'[PAD下载] {dataset}: {name} -> {dest}',flush=True)
                    _download_one(cli._session,url,dest)
                    notes.append(f'{dataset}: 已下载 {name}')
                except Exception as exc:
                    notes.append(f'{dataset}: 下载失败 {name}: {exc}')
    return notes


def _key(c, names):
    dic={str(x).lower():x for x in c.keys()}
    for x in names:
        if x.lower() in dic:return dic[x.lower()]
    return None


def _arr(c,names, required=True):
    key=_key(c,names)
    if key is None:
        if required:raise KeyError(f'缺少变量候选 {names}; 实际变量={list(c.keys())}')
        return None
    return np.asarray(c[key][...]),key


def _support(c,head,sensor,kind):
    # Products vary in naming/casing by release; check explicit likely labels first.
    names={
      'energy':[f'SWA_EAS{sensor}_ENERGY',f'SWA_EAS{sensor}_Energy','SWA_EAS_ENERGY',f'{head}_ENERGY','ENERGY'],
      'elevation':[f'SWA_EAS{sensor}_ELEVATION','SWA_EAS_ELEVATION',f'{head}_ELEVATION','ELEVATION'],
      'azimuth':[f'SWA_EAS{sensor}_AZIMUTH','SWA_EAS_AZIMUTH',f'{head}_AZIMUTH','AZIMUTH'],
      'rotation':[f'EAS{sensor}_TO_SRF','EAS_TO_SRF',f'SWA_EAS{sensor}_TO_SRF'],
    }[kind]
    k=_key(c,names)
    if k is None and kind!='rotation':
        # Use dimension metadata if naming differs, avoid grabbing *_delta_* support vars.
        ks=[x for x in c.keys() if kind.upper() in str(x).upper() and not any(y in str(x).upper() for y in ['DELTA','LABL','SPEC'])]
        if len(ks)==1:k=ks[0]
    if k is None:raise KeyError(f'缺少 {kind}; 期望 {names}; 实际变量={list(c.keys())}')
    return np.asarray(c[k][...],float)


def _at_sample(a, idx, n, sample_rank=None):
    a=np.asarray(a)
    # Never treat an invariant (3,3) rotation as time-variable when Nt == 3.
    if sample_rank is not None:
        return a[idx] if a.ndim == sample_rank + 1 and a.shape[0] == n else a
    if a.ndim >= 2 and a.shape[0] == n:return a[idx]
    return a


def _mag_frame(root,start,end):
    files=_local_files(root,'solo_l2_mag-srf-normal',start,end)
    out=[]
    if not files:
        return pd.DataFrame()
    pycdf = _load_pycdf()
    for p in files:
        began=time.perf_counter()
        print(f'[PAD磁场] 读取 {p.name} 的事件时段 ...',flush=True)
        with pycdf.CDF(str(p)) as c:
            tk=_key(c,['EPOCH','Epoch','epoch'])
            bk=_key(c,['B_SRF','B','B_SRF_NORMAL'])
            if tk is None or bk is None:raise KeyError(f'{p.name}: SRF磁场变量不存在，{list(c.keys())}')
            epoch=c[tk]
            if epoch.type()==pycdf.const.CDF_TIME_TT2000.value:
                # MAG-SRF 每天可有数十万条记录。原始 int64 时间定位后只转换事件片段。
                raw=np.asarray(c.raw_var(tk)[...],dtype=np.int64)
                low=pycdf.lib.datetime_to_tt2000((start-pd.Timedelta(minutes=3)).to_pydatetime())
                high=pycdf.lib.datetime_to_tt2000((end+pd.Timedelta(minutes=3)).to_pydatetime())
                i0=int(np.searchsorted(raw,low,side='left'))
                i1=int(np.searchsorted(raw,high,side='right'))
                if i0==i1:
                    print(f'[PAD磁场] 此文件无事件时段记录，耗时 {time.perf_counter()-began:.2f} s',flush=True)
                    continue
                t=pd.DatetimeIndex(_parse_time_array(epoch[i0:i1]))
                b=np.asarray(c[bk][i0:i1],float)
            else:
                t=pd.DatetimeIndex(_parse_time_array(epoch[...]))
                b=np.asarray(c[bk][...],float)
            if b.shape!=(len(t),3):raise ValueError(f'{p.name}: B shape={b.shape}')
            mask=(t>=start-pd.Timedelta(minutes=3))&(t<=end+pd.Timedelta(minutes=3))
            out.extend(zip(t[mask],b[mask]))
        print(f'[PAD磁场] 获得 {int(mask.sum())} 条记录，耗时 {time.perf_counter()-began:.2f} s',flush=True)
    if not out:return pd.DataFrame(columns=['x','y','z'],index=pd.DatetimeIndex([]))
    t=[i[0] for i in out]; b=np.array([i[1] for i in out]); df=pd.DataFrame(b,index=pd.DatetimeIndex(t),columns=['x','y','z']).sort_index()
    return df[~df.index.duplicated()]


def _decode_axes(data, energy, elev, az):
    """Return cube (elevation,energy,azimuth), without assuming physical CDF axis order."""
    import itertools
    if data.ndim!=3:raise ValueError(f'expected 3D sample, got {data.shape}')
    lengths=[len(np.ravel(x)) for x in (elev,energy,az)]
    possible=[p for p in itertools.permutations(range(3)) if tuple(data.shape[i] for i in p)==tuple(lengths)]
    if not possible:raise ValueError(f'Axis mismatch data={data.shape}, elevation/energy/azimuth={lengths}')
    # Prefer native order elevation, energy, azimuth when ambiguous.
    perm=(0,1,2) if (0,1,2) in possible else possible[0]
    return np.transpose(data,perm)


def _sample_direction(elevation, azimuth):
    elev=np.deg2rad(np.asarray(elevation,float))[:,None]
    az=np.deg2rad(np.asarray(azimuth,float))[None,:]
    return np.stack(np.broadcast_arrays(np.cos(elev)*np.cos(az),np.cos(elev)*np.sin(az),np.sin(elev)),axis=-1)


def read_solo_pad(config,start,end):
    started = time.perf_counter()
    print('[PAD] 开始读取 SolO NM3D PSD；时间解析器使用本地实现。', flush=True)
    notes=[]
    try:
        if config.get('PAD_AUTO_DOWNLOAD',True):notes+=ensure_eas_files(config,start,end)
    except Exception as exc:
        notes.append(f'自动下载阶段错误: {type(exc).__name__}: {exc}')
    root=Path(config.get('NM3D_ROOT',r'H:\solo\swa'))
    print('[PAD] 文件检查完成，开始读取 MAG-SRF。',flush=True)
    mf=_mag_frame(Path(config['PAD_SRF_MAG_ROOT']),start,end)
    if mf.empty:
        print(f'[PAD] 找不到同期 MAG-SRF，耗时 {time.perf_counter()-started:.2f} s。',flush=True)
        return dict(available=False,reason='找不到同期 MAG-SRF 数据（只自动下载 EAS1/EAS2，不自动下载 MAG）',notes=notes)
    pycdf = _load_pycdf()
    tolerance=pd.Timedelta(seconds=float(config.get('PAD_MAG_MAX_DELTA_SECONDS',90)))
    emin=float(config.get('PAD_MIN_ENERGY_EV',70.0))
    emax=float(config.get('PAD_MAX_ENERGY_EV',np.inf))
    nb=int(config.get('PAD_PA_BINS',36))
    edges=np.linspace(0,180,nb+1); pitch=(edges[1:]+edges[:-1])/2
    combined={}
    for sensor in (1,2):
        folder=root/f'SOLO_L2_SWA-EAS{sensor}-NM3D-PSD'
        files=_local_files(folder,f'solo_l2_swa-eas{sensor}-nm3d-psd',start,end)
        notes.append(f'EAS{sensor} 可用本地文件 {len(files)} 个')
        for file in files:
            file_started=time.perf_counter()
            print(f'[PAD电子] 读取 {file.name} ...',flush=True)
            try:
                with pycdf.CDF(str(file)) as c:
                    tk=_key(c,['EPOCH','Epoch','epoch'])
                    fk=_key(c,[f'SWA_EAS{sensor}_Data',f'SWA_EAS{sensor}_NM3D_PSD_Data',f'SWA_EAS{sensor}_NM3D_PSD_DATA'])
                    if tk is None or fk is None:raise KeyError(f'未找到 EPOCH / PSD: {list(c.keys())}')
                    tt=pd.DatetimeIndex(_parse_time_array(c[tk][...]))
                    var=c[fk]; shape=var.shape
                    if len(shape)!=4 or shape[0]!=len(tt):raise ValueError(f'PSD shape={shape}')
                    ee=_support(c,'SWA_EAS',sensor,'energy'); el=_support(c,'SWA_EAS',sensor,'elevation'); az=_support(c,'SWA_EAS',sensor,'azimuth')
                    rot=_support(c,'SWA_EAS',sensor,'rotation')
                    qkey=_key(c,['QUALITY_FLAG'])
                    quality=np.asarray(c[qkey][...]) if qkey else None
                    max_quality=int(config.get('PAD_MAX_QUALITY_FLAG',4))
                    mi=mf.index.get_indexer(tt,method='nearest',tolerance=tolerance)
                    choices=np.flatnonzero((tt>=start)&(tt<=end)&(mi>=0))
                    processed=0
                    for k in choices:
                        if quality is not None and (not np.isfinite(quality[k]) or quality[k] > max_quality):
                            continue
                        ener=np.ravel(_at_sample(ee,k,len(tt)))
                        ele=np.ravel(_at_sample(el,k,len(tt)))
                        azi=np.ravel(_at_sample(az,k,len(tt)))
                        matrix=np.asarray(_at_sample(rot,k,len(tt),sample_rank=2),float)
                        if matrix.shape!=(3,3) or not np.isfinite(matrix).all():continue
                        take=np.flatnonzero(np.isfinite(ener)&(ener>=emin)&(ener<=emax)&(ener<1e20))
                        if not take.size:continue
                        raw=np.asarray(var[k],float)
                        cube=_decode_axes(raw,ener,ele,azi)
                        values=cube[:,take,:]
                        # PSD=0 是有效观测值；只将负值、填充值和非有限值视为缺测。
                        values=np.where(np.isfinite(values)&(values>=0)&(values<1e25),values,np.nan)
                        # Owen 2022: sum PSD over E>=threshold separately for each look direction.
                        nvalid=np.sum(np.isfinite(values),axis=1)
                        sums=np.nansum(values,axis=1)
                        sums[nvalid < max(1,int(np.ceil(len(take)*float(config.get('PAD_MIN_ENERGY_COVERAGE',0.7)))))] = np.nan
                        if not np.isfinite(sums).any():continue
                        v=_sample_direction(ele,azi)
                        v=np.einsum('ij,abj->abi',matrix,v)
                        # The NM3D azimuth definition explicitly says 'electron flow direction'.
                        # Invert only when instrument documentation/physical cross-check requires it.
                        if config.get('PAD_LOOK_IS_OPPOSITE_VELOCITY',False):v=-v
                        b=mf.iloc[mi[k]].to_numpy(float)
                        if not np.all(np.isfinite(b)) or np.linalg.norm(b)<1e-10:continue
                        cos=np.einsum('abi,i->ab',v,b)/np.maximum(np.linalg.norm(v,axis=-1)*np.linalg.norm(b),1e-30)
                        pa=np.degrees(np.arccos(np.clip(cos,-1,1)))
                        j=np.clip(np.digitize(pa.ravel(),edges)-1,0,nb-1)
                        vals=sums.ravel(); good=np.isfinite(vals)&(vals>0)
                        key=tt[k].value
                        if key not in combined:combined[key]=[np.zeros(nb),np.zeros(nb,dtype=int)]
                        accum,count=combined[key]
                        np.add.at(accum,j[good],vals[good]);np.add.at(count,j[good],1)
                        processed+=1
                    notes.append(f'{file.name}: 处理 {processed}/{len(choices)} 条记录')
                    print(f'[PAD电子] {file.name}: 处理 {processed}/{len(choices)} 条，耗时 {time.perf_counter()-file_started:.2f} s',flush=True)
            except Exception as exc:
                notes.append(f'{file.name}: {type(exc).__name__}: {str(exc)[:280]}')
                print(f'[PAD电子] {file.name}: {type(exc).__name__}: {exc}',flush=True)
    if not combined:
        print(f'[PAD] 无可用记录，耗时 {time.perf_counter()-started:.2f} s。',flush=True)
        return dict(available=False,reason='无可用 3D PSD 投掷角记录；'+ '; '.join(notes[-6:]),notes=notes)
    keys=sorted(combined)
    data=np.array([np.divide(combined[k][0],combined[k][1],out=np.full(nb,np.nan),where=combined[k][1]>0) for k in keys])
    times=pd.to_datetime(np.asarray(keys,dtype=np.int64),utc=True).tz_convert(None)
    print(f'[PAD] SolO NM3D PSD 读取完成，耗时 {time.perf_counter()-started:.2f} s。', flush=True)
    return dict(available=True,times=pd.DatetimeIndex(times),pitch=pitch,flux=data,
                energy_range_ev=(emin,emax),selected_energy_ev=emin,notes=notes,quantity='PSD')
