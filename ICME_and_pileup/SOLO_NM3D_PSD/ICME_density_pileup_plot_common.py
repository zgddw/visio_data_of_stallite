# -*- coding: utf-8 -*-
"""通用绘图内核：PSP / Solar Orbiter ICME 密度堆积区综合诊断图。

请将本文件与 PSP_ICME_density_pileup_plot.py、SOLO_ICME_density_pileup_plot.py 放在同一目录。
"""
from __future__ import annotations

import re
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm

try:
    from spacepy import pycdf
except Exception as exc:  # pragma: no cover - machine-specific dependency
    pycdf = None
    _PYCDF_ERROR = exc
else:
    _PYCDF_ERROR = None


# Physical constants, SI unless explicitly named
MU0 = 4.0e-7 * np.pi
K_B = 1.380649e-23
M_P = 1.67262192369e-27
EV_TO_K = 11604.51812
CM3_TO_M3 = 1.0e6
NT_TO_T = 1.0e-9


def _setup_chinese_fonts() -> None:
    # Matplotlib falls back automatically if a candidate font is not installed.
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _normal_name(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _to_utc_naive(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts).tz_convert(None)


def _parse_time_array(raw: Any) -> pd.DatetimeIndex:
    """Convert SpacePy-returned CDF Epoch values to UTC-naive timestamps."""
    arr = np.asarray(raw)
    try:
        converted = pd.to_datetime(arr, utc=True, errors="coerce")
        idx = pd.DatetimeIndex(converted)
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        if idx.notna().any():
            return idx
    except Exception:
        pass
    # SpacePy often gives TT2000 as Python datetime objects, but retain a safe fallback.
    vals = []
    for x in arr.ravel():
        try:
            vals.append(_to_utc_naive(x))
        except Exception:
            vals.append(pd.NaT)
    return pd.DatetimeIndex(vals)


def _safe_array(raw: Any) -> np.ndarray:
    try:
        arr = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        arr = pd.to_numeric(pd.Series(np.asarray(raw).ravel()), errors="coerce").to_numpy(float)
        arr = arr.reshape(np.asarray(raw).shape)
    arr = np.squeeze(arr)
    arr[~np.isfinite(arr)] = np.nan
    # CDF fill values are commonly huge negative or positive sentinels.
    arr[np.abs(arr) > 1.0e30] = np.nan
    return arr


def _cdf_key(cdf: Any, candidates: Sequence[str]) -> Optional[str]:
    keys = {str(k).lower(): str(k) for k in cdf.keys()}
    for candidate in candidates:
        if candidate.lower() in keys:
            return keys[candidate.lower()]
    normalized = {_normal_name(k): k for k in cdf.keys()}
    for candidate in candidates:
        key = normalized.get(_normal_name(candidate))
        if key:
            return key
    return None


def _attr_text(var: Any, key: str, default: str = "") -> str:
    try:
        value = var.attrs[key]
        if isinstance(value, (list, tuple, np.ndarray)):
            value = value[0] if len(value) else default
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        return str(value)
    except Exception:
        return default


def _energy_to_ev(energy: np.ndarray, units: str) -> np.ndarray:
    unit = str(units).strip().lower().replace(" ", "")
    arr = np.asarray(energy, dtype=float)
    if "kev" in unit:
        return arr * 1.0e3
    if "mev" in unit:
        return arr * 1.0e6
    if unit in {"ev", "electronvolt", "electronvolts", "eV".lower()} or "electronvolt" in unit:
        return arr
    # PSP ENERGY_VALS is documented in eV. For unknown EAS variants, assume eV only
    # if values are in the customary electron energy range (<1e5).
    if np.nanmedian(arr) < 1.0e5:
        return arr
    return arr


def _minute_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.floor("min"), end.ceil("min"), freq="1min")


def _frame_times(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if frame is None or frame.empty or "t" not in frame.columns:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(frame["t"], utc=True, errors="coerce")).tz_convert(None)


def _get_df(data: dict, key: str) -> pd.DataFrame:
    frame = data.get(key) if isinstance(data, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty or "t" not in frame.columns:
        raise ValueError(f"cdfdataread 没有返回有效的 {key!r} 数据。请检查文件路径、前缀和变量名。")
    return frame


def _pick_scalar_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    if frame is None or frame.empty:
        raise ValueError("标量数据表为空")
    cols = {str(c).lower(): str(c) for c in frame.columns}
    norm = {_normal_name(c): str(c) for c in frame.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
        if _normal_name(c) in norm:
            return norm[_normal_name(c)]
    others = [str(c) for c in frame.columns if str(c).lower() != "t"]
    if len(others) == 1:
        return others[0]
    # Some cdfdataread versions prefix names or turn vector/scalar CDF variables into
    # similarly named columns. Use a unique partial match, never an arbitrary column.
    hits = [col for col in others if any(_normal_name(c) in _normal_name(col) for c in candidates)]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"找不到标量列。候选={list(candidates)}；实际列={list(frame.columns)}")


def _pick_vector_columns(frame: pd.DataFrame,
                         x_candidates: Sequence[str], y_candidates: Sequence[str],
                         z_candidates: Sequence[str]) -> Tuple[str, str, str]:
    cols = [str(c) for c in frame.columns if str(c).lower() != "t"]
    lower = {c.lower(): c for c in cols}
    norm = {_normal_name(c): c for c in cols}

    def find(cands: Sequence[str], suffixes: Sequence[str]) -> Optional[str]:
        for cand in cands:
            if cand.lower() in lower:
                return lower[cand.lower()]
            if _normal_name(cand) in norm:
                return norm[_normal_name(cand)]
        for c in cols:
            cl = c.lower()
            if any(cl.endswith(s.lower()) for s in suffixes):
                return c
        hits = [c for c in cols if any(_normal_name(cand) in _normal_name(c) for cand in cands)]
        return hits[0] if len(hits) == 1 else None

    x = find(x_candidates, ["_x", "_r"])
    y = find(y_candidates, ["_y", "_t"])
    z = find(z_candidates, ["_z", "_n"])
    if x and y and z and len({x, y, z}) == 3:
        return x, y, z
    if len(cols) == 3:
        # Existing project readers often return vector component columns in coordinate order.
        return cols[0], cols[1], cols[2]
    raise KeyError(f"找不到三个矢量分量。实际列={list(frame.columns)}")


def _resample_scalar(frame: pd.DataFrame, candidates: Sequence[str], grid: pd.DatetimeIndex,
                     transform=None, label: str = "value") -> pd.Series:
    col = _pick_scalar_column(frame, candidates)
    idx = _frame_times(frame)
    values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    n = min(len(idx), len(values))
    idx, values = idx[:n], values[:n]
    if transform is not None:
        values = transform(values)
    good = (~idx.isna()) & np.isfinite(values)
    if not np.any(good):
        return pd.Series(np.nan, index=grid, name=label, dtype=float)
    raw = pd.Series(values[good], index=idx[good], name=label).groupby(level=0).mean().sort_index()
    return raw.resample("1min").mean().reindex(grid)


def _resample_vector(frame: pd.DataFrame, config: dict, grid: pd.DatetimeIndex,
                     kind: str) -> pd.DataFrame:
    if kind == "mag":
        x_cands, y_cands, z_cands = config["B_COMPONENT_CANDIDATES"]
    else:
        x_cands, y_cands, z_cands = config["V_COMPONENT_CANDIDATES"]
    cx, cy, cz = _pick_vector_columns(frame, x_cands, y_cands, z_cands)
    idx = _frame_times(frame)
    comp = np.column_stack([
        pd.to_numeric(frame[cx], errors="coerce").to_numpy(float),
        pd.to_numeric(frame[cy], errors="coerce").to_numpy(float),
        pd.to_numeric(frame[cz], errors="coerce").to_numpy(float),
    ])
    n = min(len(idx), len(comp))
    idx, comp = idx[:n], comp[:n]
    valid_time = ~idx.isna()
    result = {}
    for j, label in enumerate(["R", "T", "N"]):
        vals = comp[:, j]
        good = valid_time & np.isfinite(vals) & (np.abs(vals) < 1e30)
        if not np.any(good):
            result[label] = pd.Series(np.nan, index=grid, dtype=float)
            continue
        raw = pd.Series(vals[good], index=idx[good]).groupby(level=0).mean().sort_index()
        result[label] = raw.resample("1min").mean().reindex(grid)
    if kind == "mag":
        # Match the detection rule: take |B_T| before minute averaging, so opposite signs
        # inside a high-cadence minute do not cancel before the absolute value is computed.
        vals_abs_t = np.abs(comp[:, 1])
        good_abs_t = valid_time & np.isfinite(vals_abs_t) & (vals_abs_t < 1e30)
        if np.any(good_abs_t):
            raw_abs_t = pd.Series(vals_abs_t[good_abs_t], index=idx[good_abs_t]).groupby(level=0).mean().sort_index()
            result["T_abs"] = raw_abs_t.resample("1min").mean().reindex(grid)
        else:
            result["T_abs"] = pd.Series(np.nan, index=grid, dtype=float)
    return pd.DataFrame(result, index=grid)


def _load_physical_series(config: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if config.get("CDFDATAREAD") is None:
        raise ImportError(
            "无法导入 pro_of_read.cdfdataread。请把 pro_of_read.py 放到脚本目录，"
            "或把它所在目录加入 CUSTOM_MODULE_DIRS。原始错误：" + str(config.get("CDF_IMPORT_ERROR"))
        )
    cdfdataread = config["CDFDATAREAD"]
    # One minute of padding allows edge samples to be included, but final plots are clipped
    # exactly to the pileup interval.
    fetch_start = (start - pd.Timedelta(minutes=1)).floor("min").to_pydatetime()
    fetch_end = (end + pd.Timedelta(minutes=1)).ceil("min").to_pydatetime()
    grid = _minute_index(start, end)

    mag_dict = cdfdataread(str(config["MAG_ROOT"]), [config["MAG_VARIABLE"]],
                           fetch_start, fetch_end, config["MAG_PREFIX"])
    plasma_vars = [config["DENSITY_VARIABLE"], config["VELOCITY_VARIABLE"], config["TEMPERATURE_VARIABLE"]]
    try:
        plasma_dict = cdfdataread(str(config["PLASMA_ROOT"]), plasma_vars,
                                  fetch_start, fetch_end, config["PLASMA_PREFIX"])
    except Exception:
        # Graceful per-variable fallback means a missing velocity product does not prevent
        # density, temperature, field and beta panels from being generated.
        plasma_dict = {}
        for variable in plasma_vars:
            try:
                part = cdfdataread(str(config["PLASMA_ROOT"]), [variable],
                                   fetch_start, fetch_end, config["PLASMA_PREFIX"])
                if isinstance(part, dict):
                    plasma_dict.update(part)
            except Exception as exc:
                print(f"  [警告] {variable} 读取失败：{exc}")

    mag_df = _get_df(mag_dict, config["MAG_VARIABLE"])
    b = _resample_vector(mag_df, config, grid, "mag")
    try:
        n_df = _get_df(plasma_dict, config["DENSITY_VARIABLE"])
        n_p = _resample_scalar(n_df, config["DENSITY_CANDIDATES"], grid,
                               transform=lambda x: np.where(x > 0, x, np.nan), label="Np_cm3")
    except Exception as exc:
        print(f"  [警告] 质子密度读取失败：{exc}")
        n_p = pd.Series(np.nan, index=grid, name="Np_cm3")
    try:
        t_df = _get_df(plasma_dict, config["TEMPERATURE_VARIABLE"])
        t_p = _resample_scalar(t_df, config["TEMPERATURE_CANDIDATES"], grid,
                               transform=config["TEMPERATURE_TRANSFORM"], label="Tp_K")
        t_p = t_p.where(t_p > 0)
    except Exception as exc:
        print(f"  [警告] 质子温度读取失败：{exc}")
        t_p = pd.Series(np.nan, index=grid, name="Tp_K")
    try:
        v_df = _get_df(plasma_dict, config["VELOCITY_VARIABLE"])
        v = _resample_vector(v_df, config, grid, "velocity")
        v_mag = np.sqrt(v["R"] ** 2 + v["T"] ** 2 + v["N"] ** 2)
    except Exception as exc:
        print(f"  [警告] 质子速度读取失败：{exc}")
        v = pd.DataFrame({k: np.nan for k in ["R", "T", "N"]}, index=grid)
        v_mag = pd.Series(np.nan, index=grid, name="Vp_km_s")

    b_mag = np.sqrt(b["R"] ** 2 + b["T"] ** 2 + b["N"] ** 2)
    b_t_abs = b["T_abs"] if "T_abs" in b else np.abs(b["T"])
    # Azimuth: signed angle in the RT plane; latitude: angle above/below RT plane.
    phi = np.degrees(np.arctan2(b["T"], b["R"]))
    theta = np.degrees(np.arctan2(b["N"], np.sqrt(b["R"] ** 2 + b["T"] ** 2)))
    b_tesla = b_mag * NT_TO_T
    beta = 2.0 * MU0 * (n_p * CM3_TO_M3) * K_B * t_p / (b_tesla ** 2)
    beta = beta.where(np.isfinite(beta) & (beta > 0))

    return pd.DataFrame({
        "Br_nT": b["R"], "Bt_nT": b["T"], "Bn_nT": b["N"],
        "BT_abs_nT": b_t_abs, "Bmag_nT": b_mag, "Phi_deg": phi, "Theta_deg": theta,
        "Vp_km_s": v_mag, "Tp_K": t_p, "Np_cm3": n_p, "Beta_p": beta,
    }, index=grid)


@lru_cache(maxsize=16)
def _all_cdf_paths(root_string: str) -> Tuple[str, ...]:
    root = Path(root_string)
    if not root.exists():
        return tuple()
    return tuple(str(p) for p in root.rglob("*.cdf"))


def _cdf_files_for_interval(root: Path, prefix: str, start: pd.Timestamp,
                            end: pd.Timestamp) -> List[Path]:
    if not root.exists():
        return []
    paths = [Path(p) for p in _all_cdf_paths(str(root))]
    prefix_lower = prefix.lower()
    dayset = set(pd.date_range(start.floor("D"), end.floor("D"), freq="D").strftime("%Y%m%d"))
    out = []
    for path in paths:
        name = path.name.lower()
        if prefix_lower not in name:
            continue
        dates = re.findall(r"(?<!\d)(20\d{6})(?!\d)", path.name)
        if dates and not any(d in dayset for d in dates):
            continue
        out.append(path)
    return sorted(out)


def _cdf_epoch_name(cdf: Any) -> Optional[str]:
    return _cdf_key(cdf, ["Epoch", "epoch", "EPOCH", "epoch_mag_RTN_1min", "EPOCH_TIME"])


def _read_pad_files(config: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Read the PSP/SOLO PAD product and average flux over the configured energy band.

    The PSP SPE L3 PAD CDF documents EFLUX_VS_PA_E as VAR_TYPE=ignore_data and exposes
    EFLUX_VS_PA_E_atE_byP / EFLUX_VS_PA_E_byE_atP as alternate spectrogram views.  We try
    the configured candidates in order and fall back when a view cannot be read or does
    not contain valid positive flux.  Time is retained as the leading dimension throughout.
    """
    if pycdf is None:
        return {"available": False, "reason": f"未安装 spacepy.pycdf：{_PYCDF_ERROR}"}
    root = Path(config["PAD_ROOT"])
    files = _cdf_files_for_interval(root, config["PAD_PREFIX"], start, end)
    if not files:
        return {"available": False, "reason": f"未找到 PAD CDF 文件：{root}，前缀 {config['PAD_PREFIX']}，查找时段 {start} 至 {end}"}

    all_t: List[pd.Timestamp] = []
    all_flux: List[np.ndarray] = []
    pitch_ref = None
    energy_ref = None
    read_notes: List[str] = []
    flux_candidates = list(config["PAD_FLUX_CANDIDATES"])
    pitch_candidates = config["PAD_PITCH_CANDIDATES"]
    energy_candidates = config["PAD_ENERGY_CANDIDATES"]
    lo, hi = config["PAD_ENERGY_RANGE_EV"]

    for path in files:
        try:
            with pycdf.CDF(str(path)) as cdf:
                epoch_key = _cdf_epoch_name(cdf)
                pitch_key = _cdf_key(cdf, pitch_candidates)
                energy_key = _cdf_key(cdf, energy_candidates)
                if epoch_key is None:
                    read_notes.append(f"{path.name}: 缺少 Epoch 变量")
                    continue
                if pitch_key is None or energy_key is None:
                    read_notes.append(
                        f"{path.name}: 缺少明确的投掷角/能量支撑变量（pitch={pitch_key}, energy={energy_key}）"
                    )
                    continue

                epoch = _parse_time_array(cdf[epoch_key][...])
                pitch = np.asarray(cdf[pitch_key][...], dtype=float).ravel()
                energy_var = cdf[energy_key]
                energy = np.asarray(energy_var[...], dtype=float).ravel()
                energy[~np.isfinite(energy) | (np.abs(energy) > 1.0e30)] = np.nan
                e_units = _attr_text(energy_var, "UNITS", config.get("PAD_DEFAULT_ENERGY_UNITS", "eV"))
                energy_ev = _energy_to_ev(energy, e_units)
                if len(epoch) == 0 or len(pitch) < 3 or len(energy_ev) == 0:
                    read_notes.append(
                        f"{path.name}: 支撑轴无效（nEpoch={len(epoch)}, nPA={len(pitch)}, nEnergy={len(energy_ev)}）"
                    )
                    continue

                ene_mask = np.isfinite(energy_ev) & (energy_ev >= lo) & (energy_ev <= hi)
                if not np.any(ene_mask):
                    read_notes.append(
                        f"{path.name}: ENERGY_VALS 中没有 {lo:g}–{hi:g} eV 通道；能量范围="
                        f"{np.nanmin(energy_ev):g}–{np.nanmax(energy_ev):g} eV"
                    )
                    continue

                # Try the PA-oriented virtual view first, then other compatible views and
                # the original backing variable. A bad first variable must not hide the others.
                chosen_arr = None
                chosen_key = None
                candidate_notes = []
                time_mask = (~epoch.isna()) & (epoch >= start) & (epoch <= end)
                if not np.any(time_mask):
                    read_notes.append(f"{path.name}: 文件时间范围不覆盖请求时段 {start} 至 {end}")
                    continue

                for candidate in flux_candidates:
                    flux_key = _cdf_key(cdf, [candidate])
                    if flux_key is None:
                        continue
                    try:
                        # Do not np.squeeze the full array: that can erase the time axis
                        # for a one-record slice. Only drop non-time singleton dimensions.
                        arr = np.asarray(cdf[flux_key][...], dtype=float)
                        arr[~np.isfinite(arr) | (np.abs(arr) > 1.0e30)] = np.nan
                        if arr.ndim < 3:
                            candidate_notes.append(f"{flux_key}: shape={arr.shape}, 不是 time×PA×energy")
                            continue
                        if arr.shape[0] != len(epoch):
                            time_axes = [i for i, size in enumerate(arr.shape) if size == len(epoch)]
                            if not time_axes:
                                candidate_notes.append(
                                    f"{flux_key}: Epoch 长度 {len(epoch)} 与 shape={arr.shape} 不匹配"
                                )
                                continue
                            arr = np.moveaxis(arr, time_axes[0], 0)
                        while arr.ndim > 3:
                            singleton_axes = [i for i, size in enumerate(arr.shape[1:], 1) if size == 1]
                            if not singleton_axes:
                                break
                            arr = np.squeeze(arr, axis=singleton_axes[0])
                        if arr.ndim != 3:
                            candidate_notes.append(f"{flux_key}: 无法化为三维数组，shape={arr.shape}")
                            continue

                        remaining = list(arr.shape[1:])
                        pa_axes = [i for i, size in enumerate(remaining) if size == len(pitch)]
                        en_axes = [i for i, size in enumerate(remaining) if size == len(energy_ev)]
                        possible = [(pa_axis, en_axis) for pa_axis in pa_axes for en_axis in en_axes
                                    if pa_axis != en_axis]
                        if len(possible) != 1:
                            candidate_notes.append(
                                f"{flux_key}: 角度/能量维度不匹配或不唯一，shape={arr.shape}, "
                                f"nPA={len(pitch)}, nE={len(energy_ev)}"
                            )
                            continue
                        pa_axis, en_axis = possible[0]
                        arr = np.moveaxis(arr, [1 + pa_axis, 1 + en_axis], [1, 2])
                        arr = arr[time_mask]
                        # A virtual view may exist in metadata but not have readable values
                        # in some readers; in that case continue to the next candidate.
                        band = arr[:, :, ene_mask]
                        if not np.any(np.isfinite(band) & (band > 0)):
                            candidate_notes.append(f"{flux_key}: {lo:g}–{hi:g} eV 时段内没有正通量值")
                            continue
                        chosen_arr, chosen_key = arr, flux_key
                        break
                    except Exception as exc:
                        candidate_notes.append(f"{flux_key}: {type(exc).__name__}: {exc}")
                        continue

                if chosen_arr is None:
                    details = "; ".join(candidate_notes[:4])
                    read_notes.append(
                        f"{path.name}: 所有候选通量变量均不可用。候选={flux_candidates}。{details}"
                    )
                    continue

                times = epoch[time_mask]
                chosen_arr[chosen_arr <= 0] = np.nan

                if pitch_ref is None:
                    pitch_ref = np.asarray(pitch, dtype=float)
                    energy_ref = np.asarray(energy_ev, dtype=float)
                else:
                    if len(pitch) != len(pitch_ref) or not np.allclose(pitch, pitch_ref, equal_nan=True):
                        read_notes.append(f"{path.name}: PA 网格与前一文件不同，跳过该文件")
                        continue
                    if len(energy_ev) != len(energy_ref) or not np.allclose(
                        energy_ev, energy_ref, rtol=1e-3, equal_nan=True
                    ):
                        read_notes.append(f"{path.name}: 能量网格与前一文件不同，跳过该文件")
                        continue

                # Average differential energy flux across the requested energy bins.
                selected = chosen_arr[:, :, ene_mask]
                with np.errstate(invalid="ignore"):
                    pa_flux = np.nanmean(selected, axis=2)
                if not np.any(np.isfinite(pa_flux) & (pa_flux > 0)):
                    read_notes.append(f"{path.name}: {chosen_key} 能量积分后没有可绘制的正通量")
                    continue
                all_t.extend(list(times))
                all_flux.extend(list(pa_flux))
                read_notes.append(
                    f"{path.name}: 使用 {chosen_key}，shape={chosen_arr.shape}，" 
                    f"nPA={len(pitch)}, 能量窗={lo:g}–{hi:g} eV，有效时刻={len(times)}"
                )
        except Exception as exc:
            read_notes.append(f"{path.name}: {type(exc).__name__}: {exc}")

    if not all_t or pitch_ref is None:
        reason = (
            f"未能读取到可直接绘制的 time×pitch-angle×energy 产品。目录={root}；"
            f"候选通量变量={flux_candidates}。"
        )
        if read_notes:
            reason += "诊断：" + "；".join(read_notes[:6])
        if config["SATELLITE"].upper() == "SOLO":
            reason += (
                "。当前 EAS PAD 产品若为 elevation × energy × azimuth 四维数据，需要先按 EAS 视线方向、"
                "航天器姿态和同期磁场方向转换到磁场对齐投掷角；本程序不会将传感器方位角误标为投掷角。"
            )
        return {"available": False, "reason": reason, "notes": read_notes}

    times = pd.DatetimeIndex(all_t)
    mat = np.asarray(all_flux, dtype=float)
    order = np.argsort(times.asi8)
    times = times[order]
    mat = mat[order]
    # Collapse duplicate epochs if files overlap; NaN-only duplicates do not erase valid data.
    unique_times, unique_idx = np.unique(times.asi8, return_inverse=True)
    if len(unique_times) != len(times):
        tmp = np.full((len(unique_times), mat.shape[1]), np.nan)
        for i in range(len(unique_times)):
            group = mat[unique_idx == i]
            with np.errstate(invalid="ignore"):
                counts = np.sum(np.isfinite(group), axis=0)
                sums = np.nansum(group, axis=0)
                tmp[i] = np.divide(sums, counts, out=np.full(mat.shape[1], np.nan), where=counts > 0)
        mat = tmp
        times = pd.to_datetime(unique_times).tz_localize(None)
    return {
        "available": True, "times": times, "pitch": pitch_ref,
        "flux": mat, "energy_range_ev": config["PAD_ENERGY_RANGE_EV"],
        "notes": read_notes,
    }


def _bde_mask(pad: dict) -> np.ndarray:
    """Unified 3-region BDE criterion: both directional/background ratios > 3.

    Owen et al. (2022) eqs (2)-(3): beta > 2 => direction/perp > 3.
    Require minimum angular coverage in each zone. Missing angles => unclassified.
    """
    pa=np.asarray(pad['pitch'],float)
    f=np.asarray(pad['flux'],float)
    if f.ndim!=2 or f.shape[1]!=len(pa):return np.zeros(len(f),bool)
    masks=[(pa>=0)&(pa<45),(pa>=65)&(pa<=115),(pa>135)&(pa<=180)]
    means=[]; valids=[]
    for mask in masks:
        values=f[:,mask]
        if values.shape[1]==0:return np.zeros(len(f),bool)
        count=np.sum(np.isfinite(values)&(values>0),axis=1)
        sums=np.nansum(np.where(np.isfinite(values)&(values>0),values,np.nan),axis=1)
        means.append(np.divide(sums,count,out=np.full(len(f),np.nan),where=count>0))
        valids.append(count>=max(1,int(np.ceil(mask.sum()*0.5))))
    zero,mid,back=means
    return np.logical_and.reduce(valids)&np.isfinite(mid)&(mid>0)&(zero>2*mid)&(back>2*mid)


def _draw_boundaries(ax: Any, start: pd.Timestamp, end: pd.Timestamp) -> None:
    ax.axvline(start, color="red", linestyle="--", linewidth=1.4, zorder=8)
    ax.axvline(end, color="red", linestyle="--", linewidth=1.4, zorder=8)


def _plot_timeseries(ax: Any, times: pd.DatetimeIndex, values: Any, label: str,
                     ylabel: str, view_start: pd.Timestamp, view_end: pd.Timestamp,
                     event_start: pd.Timestamp, event_end: pd.Timestamp,
                     color: Optional[str] = None, ylim: Optional[Tuple[float, float]] = None) -> None:
    vals = np.asarray(values, dtype=float)
    if len(times) == len(vals) and np.isfinite(vals).any():
        ax.plot(times, vals, linewidth=0.9, label=label, color=color)
        if label and (label != ylabel):
            ax.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    else:
        ax.text(0.5, 0.5, "该参数无有效数据", transform=ax.transAxes,
                ha="center", va="center", color="dimgray")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(view_start, view_end)
    _draw_boundaries(ax, event_start, event_end)


def _plot_pad(ax_bde: Any, ax_pad: Any, pad: dict,
              view_start: pd.Timestamp, view_end: pd.Timestamp,
              event_start: pd.Timestamp, event_end: pd.Timestamp, config: dict) -> None:
    ax_bde.set_xlim(view_start, view_end)
    ax_pad.set_xlim(view_start, view_end)
    _draw_boundaries(ax_bde, event_start, event_end)
    _draw_boundaries(ax_pad, event_start, event_end)
    ax_bde.set_ylabel("BDE\n标记", fontsize=8)
    ax_bde.set_yticks([])
    ax_bde.grid(False)

    if not pad.get("available"):
        reason = pad.get("reason", "没有可用电子投掷角数据")
        ax_pad.text(0.5, 0.5, reason, transform=ax_pad.transAxes,
                    ha="center", va="center", wrap=True, fontsize=9)
        ax_pad.set_ylabel("投掷角 [°]", fontsize=9)
        ax_bde.text(0.5, 0.5, "未计算（无可用投掷角轴）", transform=ax_bde.transAxes,
                    ha="center", va="center", fontsize=8, color="dimgray")
        return

    times = pd.DatetimeIndex(pad["times"])
    pa = np.asarray(pad["pitch"], dtype=float)
    flux = np.asarray(pad["flux"], dtype=float)
    mask = (~times.isna()) & (times >= view_start) & (times <= view_end)
    times, flux = times[mask], flux[mask]
    if not len(times):
        ax_pad.text(0.5, 0.5, "PAD 数据没有覆盖显示时段", transform=ax_pad.transAxes,
                    ha="center", va="center")
        ax_pad.set_ylabel("电子投掷角 [°]", fontsize=9)
        return
    positive = flux[np.isfinite(flux) & (flux > 0)]
    if not len(positive):
        ax_pad.text(0.5, 0.5, "所选能量通道内无正通量数据", transform=ax_pad.transAxes,
                    ha="center", va="center")
    else:
        vmin = max(float(np.nanpercentile(positive, 2)), np.finfo(float).tiny)
        vmax = max(float(np.nanpercentile(positive, 98)), vmin * 1.01)
        tnum = mdates.date2num(times.to_pydatetime())
        z = np.ma.masked_where(~np.isfinite(flux.T) | (flux.T <= 0), flux.T)
        pcm = ax_pad.pcolormesh(tnum, pa, z, shading="auto", cmap="turbo",
                                norm=LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax_pad, width="1.8%", height="100%", loc="lower left",
                         bbox_to_anchor=(1.015, 0, 1, 1), bbox_transform=ax_pad.transAxes,
                         borderpad=0)
        cb = ax_pad.figure.colorbar(pcm, cax=cax)
        cb.set_label("电子相空间密度求和 (PSD)" if pad.get("quantity")=="PSD" else "电子微分能量通量", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        bde = _bde_mask({"pitch": pa, "flux": flux})
        if bde.any():
            ax_bde.scatter(times[bde], np.zeros(bde.sum()), s=18, marker="o",
                           color="red", edgecolors="darkred", linewidths=0.25, zorder=6)
            ax_bde.set_ylim(-0.8, 0.8)
            ax_bde.set_yticks([])
            ax_bde.text(0.995, 0.5, f"BDE 候选：{int(bde.sum())} 点",
                        transform=ax_bde.transAxes, ha="right", va="center", fontsize=8)
        else:
            ax_bde.set_ylim(-0.8, 0.8)
            ax_bde.text(0.995, 0.5, "未发现双向超热电子束（两端/垂直 PSD 均 >3）",
                        transform=ax_bde.transAxes, ha="right", va="center", fontsize=8, color="dimgray")
    lo, hi = pad.get("energy_range_ev", (config.get("PAD_MIN_ENERGY_EV",70.), config.get("PAD_MAX_ENERGY_EV",float("inf"))))
    ax_pad.set_ylabel("电子投掷角 [°]", fontsize=9)
    ax_pad.set_title((f"电子投掷角 PSD 求和，E ≥ {lo:g} eV" if not np.isfinite(hi) else f"电子投掷角 PSD 求和，{lo:g}–{hi:g} eV") if pad.get("quantity")=="PSD" else f"电子投掷角微分能量通量，{lo:g}–{hi:g} eV", fontsize=10, loc="left")
    ax_pad.set_ylim(0, 180)
    ax_pad.set_yticks(np.arange(0, 181, 30))
    ax_pad.grid(False)
    ax_pad.set_xlim(view_start, view_end)


def _safe_filename(value: Any) -> str:
    text = str(value).strip() or "ICME"
    return re.sub(r'[<>:"/\\|?*\s]+', "_", text)[:120]


def _ask_user_for_table_path(config: dict) -> dict:
    """当入口未配置 DETECTED_TABLE 时，允许用户通过文件对话框或终端自行选择列表。"""
    if config.get("DETECTED_TABLE") or not config.get("PROMPT_FOR_TABLE", True):
        return config

    selected = ""
    # Windows / 桌面环境优先使用文件选择窗口，避免脚本替用户猜测列表位置。
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = filedialog.askopenfilename(
            title=f"选择 {config['SATELLITE']} 密度堆积区检索结果表",
            filetypes=[
                ("CSV / Excel 表格", "*.csv *.xlsx *.xls"),
                ("CSV 文件", "*.csv"),
                ("Excel 文件", "*.xlsx *.xls"),
                ("所有文件", "*.*"),
            ],
        )
        root.destroy()
    except Exception as exc:
        print(f"[提示] 无法打开文件选择窗口：{exc}")

    # 如果对话框被取消或当前环境不支持 Tk，允许用户直接输入绝对路径。
    if not selected and getattr(sys.stdin, "isatty", lambda: False)():
        try:
            selected = input(
                f"请输入 {config['SATELLITE']} 密度堆积区列表的完整路径（直接回车则尝试自动查找）：\n> "
            ).strip().strip('"')
        except (EOFError, KeyboardInterrupt):
            selected = ""

    resolved = dict(config)
    if selected:
        resolved["DETECTED_TABLE"] = Path(selected).expanduser()
        print(f"[用户选择的事件表] {resolved['DETECTED_TABLE']}")
    else:
        resolved["PROMPT_FOR_TABLE"] = False
        print("[提示] 未手动选择列表，将尝试按默认目录自动查找。")
    return resolved


def _read_event_table(config: dict) -> pd.DataFrame:
    table_path = config.get("DETECTED_TABLE")
    if table_path:
        # 用户明确指定了文件时，若路径无效就明确报错，不静默切换到另一张表。
        table = Path(table_path).expanduser()
        if not table.is_file():
            raise FileNotFoundError(f"指定的密度堆积区列表不存在或不是文件：{table}")
    else:
        candidates = [
            Path(config["DETECTION_OUTPUT_DIR"]) / f"{config['SATELLITE']}_ICME_密度堆积区列表.csv",
            Path(__file__).resolve().parent / f"{config['SATELLITE']}_ICME_密度堆积区列表.csv",
            Path(config["DETECTION_OUTPUT_DIR"]) / f"{config['SATELLITE']}_ICME_密度堆积区结果.xlsx",
        ]
        table = next((p for p in candidates if p and p.exists()), None)
        if table is None:
            raise FileNotFoundError(
                "找不到已检出的密度堆积区表。请在弹出的文件选择窗口中选取列表，"
                "或在脚本顶部配置 DETECTED_TABLE。\n默认尝试过：\n" +
                "\n".join(str(p) for p in candidates)
            )
    if table.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(table, sheet_name="密度堆积区列表")
    else:
        df = pd.read_csv(table, encoding="utf-8-sig")
    required = {"Pileup_Start_UTC", "Pileup_End_UTC"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"事件表缺少必要字段 {sorted(missing)}：{table}")
    if "Status" in df:
        df = df.loc[df["Status"].astype(str).eq("已检出")].copy()
    df["Pileup_Start_UTC"] = pd.to_datetime(df["Pileup_Start_UTC"], utc=True, errors="coerce").dt.tz_convert(None)
    df["Pileup_End_UTC"] = pd.to_datetime(df["Pileup_End_UTC"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["Pileup_Start_UTC", "Pileup_End_UTC"])
    df = df.loc[df["Pileup_End_UTC"] > df["Pileup_Start_UTC"]].copy()
    if df.empty:
        raise RuntimeError(f"事件表中没有有效的已检出密度堆积区：{table}")
    print(f"[事件表] {table}；有效堆积区 {len(df)} 个")
    return df


def _solo_bde_occupancy(pad, start, end):
    """BDE occupancy over valid PAD samples; gaps are not classified as non-BDE."""
    result = {"BDE_Time_min": np.nan, "PAD_Valid_Time_min": 0.0,
              "BDE_Ratio_Percent": np.nan, "PAD_Coverage_Percent": 0.0,
              "BDE_Valid_Samples": 0, "BDE_Positive_Samples": 0}
    if not pad.get("available"):
        return result
    t = pd.DatetimeIndex(pad["times"])
    flux = np.asarray(pad["flux"], dtype=float)
    if len(t) == 0 or flux.ndim != 2 or flux.shape[0] != len(t):
        return result
    order = np.argsort(t.asi8)
    t, flux = t[order], flux[order]
    good_unique = ~t.duplicated(keep="first")
    t, flux = t[good_unique], flux[good_unique]
    pitch = np.asarray(pad["pitch"], float)
    regions=[(pitch<45),(pitch>=65)&(pitch<=115),(pitch>135)]
    valid=np.ones(len(t),dtype=bool)
    for zone in regions:
        if not zone.any(): valid[:]=False; break
        count=np.sum(np.isfinite(flux[:,zone])&(flux[:,zone]>0),axis=1)
        valid &= count>=max(1,int(np.ceil(zone.sum()*0.5)))
    bde = _bde_mask({'pitch':pitch,'flux':flux}) & valid
    times = t.asi8.astype(np.int64)
    cadence = np.median(np.diff(times)) if len(times)>1 else 60_000_000_000
    if cadence <= 0: cadence = 60_000_000_000
    # Centered nominal-cadence intervals clipped to the DPR. Never bridge gaps.
    lo, hi = _to_utc_naive(start).value, _to_utc_naive(end).value
    width = np.maximum(0, np.minimum(times+cadence//2, hi)-np.maximum(times-cadence//2, lo))/60e9
    duration = (hi-lo)/60e9
    valid_minutes = float(width[valid].sum())
    bde_minutes = float(width[bde].sum())
    result.update(BDE_Time_min=bde_minutes, PAD_Valid_Time_min=valid_minutes,
                  PAD_Coverage_Percent=100*valid_minutes/duration if duration>0 else 0,
                  BDE_Ratio_Percent=100*bde_minutes/valid_minutes if valid_minutes>0 else np.nan,
                  BDE_Valid_Samples=int(np.count_nonzero(valid & (width>0))),
                  BDE_Positive_Samples=int(np.count_nonzero(bde & (width>0))))
    return result


def _plot_one_region(config: dict, row: pd.Series, output_dir: Path) -> dict:
    start = _to_utc_naive(row["Pileup_Start_UTC"])
    end = _to_utc_naive(row["Pileup_End_UTC"])
    event_id = str(row.get("Event_ID", row.get("ICME_ID", "ICME")))
    grade = str(row.get("Grade", "未知等级"))
    if pd.isna(start) or pd.isna(end) or end <= start:
        raise ValueError(f"事件 {event_id} 的堆积区起止时间无效")

    margin_hours = float(config.get("PLOT_CONTEXT_HOURS", 2.0))
    if margin_hours < 0:
        raise ValueError("PLOT_CONTEXT_HOURS 不能为负数")
    margin = pd.Timedelta(hours=margin_hours)
    view_start, view_end = start - margin, end + margin

    print(f"\n[{config['SATELLITE']}] {event_id} | 堆积区={start} -> {end} | "
          f"显示范围={view_start} -> {view_end} | 等级={grade}")
    physical = _load_physical_series(config, view_start, view_end)
    if config["SATELLITE"].upper() == "SOLO":
        print("  [读取路径] SolO 专用 NM3D PSD(EAS1+EAS2)+MAG-SRF 投掷角计算")
        from solo_nm3d_pad import read_solo_pad
        pad = read_solo_pad(config, view_start, view_end)
    else:
        pad = _read_pad_files(config, view_start, view_end)
    if not pad.get("available"):
        print(f"  [PAD 提示] {pad.get('reason', '')}")
    else:
        print(f"  [PAD] 已读取 {len(pad['times'])} 个时刻，{len(pad['pitch'])} 个投掷角通道；"
              f"最低能量={config.get('PAD_MIN_ENERGY_EV',70.0)} eV")
        for note in pad.get("notes", []):
            if ": 使用 " in note:
                print(f"  [PAD 读取] {note}")

    plot_started = time.perf_counter()
    _setup_chinese_fonts()
    print('  [绘图] 开始生成十面板图。', flush=True)
    fig, axes = plt.subplots(
        10, 1, figsize=(17, 25), sharex=True,
        gridspec_kw={"height_ratios": [0.30, 2.25, 0.85, 1.15, 0.85, 0.85, 0.9, 0.9, 0.9, 0.9]},
        constrained_layout=False,
    )
    ax_bde, ax_pad, ax_bt, ax_components, ax_phi, ax_theta, ax_v, ax_t, ax_n, ax_beta = axes
    _plot_pad(ax_bde, ax_pad, pad, view_start, view_end, start, end, config)

    t = physical.index
    _plot_timeseries(ax_bt, t, physical["BT_abs_nT"], "|B_T|", "|B_T| [nT]",
                     view_start, view_end, start, end)

    colors = {"Br_nT": "tab:blue", "Bt_nT": "tab:orange", "Bn_nT": "tab:green"}
    for key, label in [("Br_nT", r"$B_R$"), ("Bt_nT", r"$B_T$"), ("Bn_nT", r"$B_N$")]:
        if np.isfinite(physical[key].to_numpy(float)).any():
            ax_components.plot(t, physical[key], linewidth=0.9, label=label, color=colors[key])
    ax_components.set_ylabel("B 分量 [nT]", fontsize=9)
    ax_components.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    ax_components.grid(True, alpha=0.22, linewidth=0.5)
    ax_components.set_xlim(view_start, view_end)

    _plot_timeseries(ax_phi, t, physical["Phi_deg"], "方位角 φ", "方位角 φ [°]",
                     view_start, view_end, start, end)
    ax_phi.set_ylim(-180, 180)
    ax_phi.set_yticks([-180, -90, 0, 90, 180])
    _plot_timeseries(ax_theta, t, physical["Theta_deg"], "纬向角 θ", "纬向角 θ [°]",
                     view_start, view_end, start, end)
    ax_theta.set_ylim(-90, 90)
    ax_theta.set_yticks([-90, -45, 0, 45, 90])
    _plot_timeseries(ax_v, t, physical["Vp_km_s"], "|Vp|", "质子速度 [km/s]",
                     view_start, view_end, start, end)
    _plot_timeseries(ax_t, t, physical["Tp_K"] / 1.0e6, "Tp", r"质子温度 [$10^6$ K]",
                     view_start, view_end, start, end)
    _plot_timeseries(ax_n, t, physical["Np_cm3"], "Np", r"质子密度 [cm$^{-3}$]",
                     view_start, view_end, start, end)
    _plot_timeseries(ax_beta, t, physical["Beta_p"], r"β$_p$", r"质子 β$_p$",
                     view_start, view_end, start, end)
    beta_valid = physical["Beta_p"].to_numpy(float)
    if np.any(np.isfinite(beta_valid) & (beta_valid > 0)):
        ax_beta.set_yscale("log")

    for ax in axes:
        _draw_boundaries(ax, start, end)
        ax.set_xlim(view_start, view_end)
    ax_beta.set_xlabel("时间 (UTC)")
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_beta.xaxis.set_major_locator(locator)
    ax_beta.xaxis.set_major_formatter(formatter)
    fig.suptitle(
        f"{config['SATELLITE']} ICME 密度堆积区综合诊断\n{event_id} | {grade}\n"
        f"堆积区：{start:%Y-%m-%d %H:%M:%S} — {end:%Y-%m-%d %H:%M:%S} UTC\n"
        f"显示范围：{view_start:%Y-%m-%d %H:%M} — {view_end:%Y-%m-%d %H:%M} UTC",
        fontsize=13, y=0.995,
    )
    print(f'  [绘图] 数据已绘制，耗时 {time.perf_counter()-plot_started:.2f} s；正在排版。', flush=True)
    fig.tight_layout(rect=[0.025, 0.025, 0.92, 0.965], h_pad=0.45)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(f"{event_id}_{start:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}")
    png_path = output_dir / f"{filename}.png"
    dpi = int(config.get("PLOT_DPI", 120))
    save_started = time.perf_counter()
    print(f'  [保存] 正在写入 PNG（{dpi} dpi）：{png_path}', flush=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] 完成 {png_path}；耗时 {time.perf_counter()-save_started:.2f} s", flush=True)
    return {
        "Satellite": config["SATELLITE"], "Event_ID": event_id,
        "Grade": grade, "Pileup_Start_UTC": start, "Pileup_End_UTC": end,
        "Display_Start_UTC": view_start, "Display_End_UTC": view_end,
        "Context_Hours_Each_Side": margin_hours,
        "Duration_min": (end - start).total_seconds() / 60.0,
        "PAD_Available": bool(pad.get("available")),
        "PAD_Note": pad.get("reason", ""),
        "Image_PNG": str(png_path), "Status": "完成",
        **_solo_bde_occupancy(pad, start, end),
    }


def run_plotting(config: dict) -> None:
    config = _ask_user_for_table_path(config)
    df = _read_event_table(config)
    output_dir = Path(config["PLOT_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    done, errors = [], []
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"\n[{i}/{len(df)}]")
        try:
            done.append(_plot_one_region(config, row, output_dir))
        except Exception as exc:
            event_id = str(row.get("Event_ID", f"event_{i}"))
            print(f"  [错误] {event_id}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            errors.append({
                "Satellite": config["SATELLITE"], "Event_ID": event_id,
                "Pileup_Start_UTC": row.get("Pileup_Start_UTC"),
                "Pileup_End_UTC": row.get("Pileup_End_UTC"),
                "Error": f"{type(exc).__name__}: {exc}",
            })
    if done:
        if config["SATELLITE"].upper() == "SOLO":
            keys = ["Satellite", "Event_ID", "Grade", "Pileup_Start_UTC", "Pileup_End_UTC",
                    "Duration_min", "PAD_Available", "BDE_Time_min", "PAD_Valid_Time_min",
                    "BDE_Ratio_Percent", "PAD_Coverage_Percent", "BDE_Valid_Samples",
                    "BDE_Positive_Samples", "PAD_Note"]
            pd.DataFrame(done).reindex(columns=keys).to_csv(
                output_dir / "SOLO_密度堆积区_BDE时间占比.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(done).to_csv(output_dir / f"{config['SATELLITE']}_综合图像生成清单.csv",
                                  index=False, encoding="utf-8-sig")
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / f"{config['SATELLITE']}_绘图失败清单.csv",
                                    index=False, encoding="utf-8-sig")
    print("\n" + "=" * 80)
    print(f"{config['SATELLITE']} 绘图结束：完成 {len(done)} / {len(df)} 个堆积区")
    print(f"输出目录：{output_dir.resolve()}")
    if errors:
        failure_path = output_dir / (str(config["SATELLITE"]) + "_绘图失败清单.csv")
        print(f"失败清单：{failure_path}")
    print("=" * 80)
