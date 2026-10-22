# -*- coding: utf-8 -*-
"""SolO SWA-EAS PAD-DEF to pitch angle using synchronous MAG-SRF.

Input dimensions per attached CDF skeleton:
  flux       (time, elevation=2, energy=64, azimuth=32)
  energy     (time, 64) or (64,)
  elevation  (time, 2) or (2,)
  azimuth    (32,)
  EAS_TO_SRF (time, 3, 3)

Requires spacepy.pycdf. Does NOT use PSP-style PITCHANGLE variables.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from spacepy import pycdf


def _read_magnetic(config, start, end, files):
    from ICME_density_pileup_plot_common import _cdf_key, _parse_time_array
    records = []
    notes = []
    for f in files:
        try:
            with pycdf.CDF(str(f)) as c:
                tkey = _cdf_key(c, ['EPOCH', 'Epoch', 'epoch'])
                bkey = _cdf_key(c, ['B_SRF', 'B', 'B_SRF_NORMAL', 'B_MAG'])
                if not tkey or not bkey:
                    notes.append(f'{f.name}: 找不到 SRF 磁场变量，实际变量={list(c.keys())}')
                    continue
                tt = pd.DatetimeIndex(_parse_time_array(c[tkey][...]))
                bb = np.asarray(c[bkey][...], dtype=float)
                if bb.ndim != 2 or bb.shape != (len(tt), 3):
                    notes.append(f'{f.name}: B 维度不正确 {bb.shape}')
                    continue
                keep = (~tt.isna()) & (tt >= start-pd.Timedelta(minutes=3)) & (tt <= end+pd.Timedelta(minutes=3))
                for ti, bi in zip(tt[keep], bb[keep]):
                    if np.all(np.isfinite(bi)) and np.linalg.norm(bi)>0 and np.max(np.abs(bi))<1e20:
                        records.append((ti, *bi))
        except Exception as exc:
            notes.append(f'{f.name}: {type(exc).__name__}: {exc}')
    if not records:
        return None, notes
    frame = pd.DataFrame(records, columns=['time','x','y','z']).set_index('time').sort_index()
    frame = frame[~frame.index.duplicated(keep='first')]
    return frame, notes


def read_solo_pad(config, start, end):
    from ICME_density_pileup_plot_common import _cdf_files_for_interval, _cdf_key, _parse_time_array
    root = Path(config['PAD_ROOT'])
    mag_root = Path(config['PAD_SRF_MAG_ROOT'])
    files = _cdf_files_for_interval(root, config['PAD_PREFIX'], start, end)
    mags = _cdf_files_for_interval(mag_root, config['PAD_SRF_MAG_PREFIX'], start, end)
    if not files or not mags:
        return dict(available=False, reason=(f'缺少同期文件：PAD={len(files)} (目录 {root})；'
                                              f'MAG-SRF={len(mags)} (目录 {mag_root})'))
    mf, notes = _read_magnetic(config, start, end, mags)
    if mf is None:
        return dict(available=False, reason='同期 MAG-SRF 未能读取到三分量磁场；'+'; '.join(notes[:4]))
    target = float(config.get('PAD_TARGET_ENERGY_EV', 300.0))
    nbins = int(config.get('PAD_PA_BINS', 12))
    edges = np.linspace(0, 180, nbins+1)
    centers = (edges[1:]+edges[:-1])/2
    tolerance = pd.Timedelta(seconds=float(config.get('PAD_MAG_MAX_DELTA_SECONDS',90)))
    times_out, flux_out, energy_out = [], [], []
    for f in files:
        try:
            with pycdf.CDF(str(f)) as c:
                keys = list(c.keys())
                k_t = _cdf_key(c, ['EPOCH','Epoch','epoch'])
                k_f = _cdf_key(c, ['SWA_EAS_PAD_DEF_Data','SWA_EAS_BM_Data'])
                required = ['SWA_EAS_ENERGY','SWA_EAS_ELEVATION','SWA_EAS_AZIMUTH','EAS_TO_SRF']
                missing = [name for name in required if _cdf_key(c, [name]) is None]
                if not k_t or not k_f or missing:
                    notes.append(f'{f.name}: 缺少 {missing}, EPOCH={k_t}, flux={k_f}; 实际变量={keys}')
                    continue
                tt = pd.DatetimeIndex(_parse_time_array(c[k_t][...]))
                raw = np.asarray(c[k_f][...], dtype=float)
                ee = np.asarray(c[_cdf_key(c,['SWA_EAS_ENERGY'])][...],float)
                elev = np.asarray(c[_cdf_key(c,['SWA_EAS_ELEVATION'])][...],float)
                az = np.asarray(c[_cdf_key(c,['SWA_EAS_AZIMUTH'])][...],float).reshape(-1)
                rot = np.asarray(c[_cdf_key(c,['EAS_TO_SRF'])][...],float)
                n = len(tt)
                if raw.shape != (n,2,64,32):
                    notes.append(f'{f.name}: PAD shape={raw.shape} != {(n,2,64,32)}')
                    continue
                if ee.shape == (64,): ee = np.broadcast_to(ee,(n,64))
                if elev.shape == (2,): elev = np.broadcast_to(elev,(n,2))
                if ee.shape != (n,64) or elev.shape != (n,2) or az.shape!=(32,) or rot.shape!=(n,3,3):
                    notes.append(f'{f.name}: support shapes E={ee.shape}, elev={elev.shape}, az={az.shape}, matrix={rot.shape}')
                    continue
                # Magnetic field nearest timestamp, never forward-fill distant gaps.
                nearest = mf.index.get_indexer(tt, method='nearest', tolerance=tolerance)
                sel = np.flatnonzero((~tt.isna()) & (tt>=start) & (tt<=end) & (nearest>=0))
                if not len(sel):
                    notes.append(f'{f.name}: 文件无覆盖或无同期 MAG-SRF 记录（PAD 时间={tt.min()}..{tt.max()}）')
                    continue
                for i in sel:
                    energy = ee[i]
                    allowed = np.isfinite(energy) & (energy>0) & (np.abs(energy)<1e20)
                    if not allowed.any(): continue
                    j = int(np.argmin(np.where(allowed, np.abs(energy-target),np.inf)))
                    values = raw[i,:,j,:].copy()
                    values[~np.isfinite(values) | (values<=0) | (np.abs(values)>1e20)] = np.nan
                    if not np.isfinite(values).any(): continue
                    b = mf.iloc[nearest[i]].to_numpy(dtype=float)
                    a = np.deg2rad(az)[None, :]
                    e = np.deg2rad(elev[i])[:, None]
                    v = np.stack(np.broadcast_arrays(np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)),axis=-1)
                    v = np.einsum('ij,abj->abi',rot[i],v)
                    if config.get('PAD_LOOK_IS_OPPOSITE_VELOCITY', True): v = -v
                    cospa = np.einsum('abi,i->ab',v,b) / (np.linalg.norm(v,axis=-1)*np.linalg.norm(b))
                    pitch = np.rad2deg(np.arccos(np.clip(cospa,-1,1)))
                    bins = np.clip(np.digitize(pitch.ravel(),edges)-1,0,nbins-1)
                    vals = values.ravel()
                    z = np.full(nbins,np.nan)
                    for k in range(nbins):
                        relevant = vals[(bins==k) & np.isfinite(vals)]
                        if relevant.size: z[k]=np.mean(relevant)
                    times_out.append(tt[i]); flux_out.append(z); energy_out.append(float(energy[j]))
                notes.append(f'{f.name}: 支撑维度 E={ee.shape}, elevation={elev.shape}; 同期 PAD 样本={len(sel)}')
        except Exception as exc:
            notes.append(f'{f.name}: {type(exc).__name__}: {exc}')
    if not times_out:
        return dict(available=False, reason='读取完成但未提取到有效 PAD：'+'; '.join(notes[:6]))
    ti = pd.DatetimeIndex(times_out)
    idx = np.argsort(ti.asi8)
    ti, ff, ee = ti[idx], np.asarray(flux_out)[idx], np.asarray(energy_out)[idx]
    unique = ~ti.duplicated()
    ti, ff, ee = ti[unique], ff[unique], ee[unique]
    return dict(available=True, times=ti, pitch=centers, flux=ff,
                energy_range_ev=(target,target), selected_energy_ev=float(np.nanmedian(ee)),
                notes=notes)
