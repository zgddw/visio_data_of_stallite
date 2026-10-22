# -*- coding: utf-8 -*-
"""Read PSP SPAN-E L3 PAD and convert differential energy flux to electron PSD.

The CDF stores EFLUX_VS_PA_E in eV/(cm^2 s sr eV), ENERGY_VALS in eV,
and typically has shapes (time, 12, 32), (time, 32), (time, 12).
The output contract matches the plotter's SolO PAD reader: flux is a
(time, pitch) array, but its values here are summed PSD [s^3/km^6].
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ELECTRON_MASS_KG = 9.1093837139e-31
ELECTRON_VOLT_J = 1.602176634e-19
PSD_SI_TO_KM = 1.0e18
CM2_TO_M2 = 1.0e4
PSP_READER_VERSION = "PSP_SPE_PSD_20261022_V1"


def energy_flux_to_psd(eflux: np.ndarray, energy_ev: np.ndarray) -> np.ndarray:
    """Convert energy flux to nonrelativistic f(v), in s^3/km^6.

    f = m_e^2 * J_E / (2 E^2), with J_E in SI units. The CDF energy-flux
    unit has eV in both numerator and denominator, so its SI multiplier
    is 10^4 for cm^-2 -> m^-2. E is the electron kinetic energy in joules.
    """
    values = np.asarray(eflux, dtype=float)
    energy = np.asarray(energy_ev, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = (values * CM2_TO_M2 * ELECTRON_MASS_KG**2 /
               (2.0 * (energy * ELECTRON_VOLT_J)**2) * PSD_SI_TO_KM)
    return np.where((values > 0) & (energy > 0) & np.isfinite(out), out, np.nan)


def _daily_file(root: Path, prefix: str, day) -> Path | None:
    stamp = day.strftime("%Y%m%d")
    matches = list((root / day.strftime("%Y")).glob(f"{prefix}_{stamp}_v*.cdf"))
    matches += list(root.glob(f"{prefix}_{stamp}_v*.cdf"))
    if not matches:
        return None

    def version(path: Path) -> tuple[int, str]:
        found = re.search(r"_v(\d+)\.cdf$", path.name, re.IGNORECASE)
        return (int(found.group(1)) if found else -1, str(path))

    return max(matches, key=version)


def _clean(values, attrs) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    bad = ~np.isfinite(out)
    for name in ("FILLVAL", "VALIDMIN", "VALIDMAX"):
        raw = attrs.get(name)
        if raw is None:
            continue
        try:
            limit = float(np.asarray(raw).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if not np.isfinite(limit):
            continue
        if name == "FILLVAL":
            bad |= np.isclose(out, limit, rtol=1e-5, atol=1e-8)
        elif name == "VALIDMIN":
            bad |= out < limit
        else:
            bad |= out > limit
    out[bad] = np.nan
    return out


def _time_axis(values: np.ndarray, ntime: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.broadcast_to(arr, (ntime, len(arr))).copy()
    if arr.ndim == 2 and arr.shape[0] == ntime:
        return arr.copy()
    if arr.ndim == 2 and arr.shape[1] == ntime:
        return arr.T.copy()
    raise ValueError(f"{label} 维度不匹配：{arr.shape}，时间记录数={ntime}")


def _units(attrs, default: str) -> str:
    value = attrs.get("UNITS", default)
    if isinstance(value, (tuple, list, np.ndarray)):
        value = np.asarray(value).reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip().lower()


def read_psp_pad(config: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Read local daily CDFs, filter quality, convert each channel, sum PSD."""
    try:
        from spacepy import pycdf
    except Exception as exc:
        return {"available": False, "reason": f"无法导入 spacepy.pycdf: {exc}"}

    lo, hi = map(float, config["PAD_ENERGY_RANGE_EV"])
    if not (0 < lo < hi):
        raise ValueError("PAD_ENERGY_RANGE_EV 必须为两个正数，且下限小于上限")
    root = Path(config["PAD_ROOT"])
    prefix = str(config["PAD_PREFIX"])
    if not root.is_dir():
        return {"available": False, "reason": f"PSP PAD 目录不存在或不可读：{root}"}

    all_times: list[pd.DatetimeIndex] = []
    all_pitch: list[np.ndarray] = []
    all_psd: list[np.ndarray] = []
    notes: list[str] = []
    day = pd.Timestamp(start).date()
    last_day = pd.Timestamp(end).date()
    while day <= last_day:
        path = _daily_file(root, prefix, day)
        if path is None:
            notes.append(f"{day}: 未找到 PAD CDF（检查 {root / str(day.year)}）")
            day += timedelta(days=1)
            continue
        try:
            with pycdf.CDF(str(path)) as cdf:
                required = ("Epoch", "PITCHANGLE", "ENERGY_VALS", "EFLUX_VS_PA_E")
                missing = [key for key in required if key not in cdf]
                if missing:
                    raise ValueError(f"缺少 CDF 变量 {missing}")
                if "cm" not in _units(cdf["EFLUX_VS_PA_E"].attrs,
                                      "eV/(cm^2 s sr eV)"):
                    raise ValueError("EFLUX_VS_PA_E 单位不是预期的 cm^-2；停止 PSD 换算")
                e_units = _units(cdf["ENERGY_VALS"].attrs, "eV")
                if "kev" in e_units:
                    energy_factor = 1000.0
                elif "ev" in e_units:
                    energy_factor = 1.0
                else:
                    raise ValueError(f"未知 ENERGY_VALS 单位 {e_units!r}")

                epoch = pd.DatetimeIndex(pd.to_datetime(np.asarray(cdf["Epoch"][...])))
                if epoch.tz is not None:
                    epoch = epoch.tz_convert("UTC").tz_localize(None)
                chosen = np.flatnonzero((epoch >= start) & (epoch <= end))
                if not len(chosen):
                    notes.append(f"{path.name}: 不覆盖请求时段")
                    day += timedelta(days=1)
                    continue
                # Daily records are chronological. Slice only the required window.
                i0, i1 = int(chosen[0]), int(chosen[-1]) + 1
                times = epoch[i0:i1]
                pitch_var, energy_var = cdf["PITCHANGLE"], cdf["ENERGY_VALS"]
                pitch_raw = pitch_var[i0:i1] if len(pitch_var) == len(epoch) else pitch_var[...]
                energy_raw = energy_var[i0:i1] if len(energy_var) == len(epoch) else energy_var[...]
                pitch = _time_axis(_clean(pitch_raw, pitch_var.attrs), len(times), "PITCHANGLE")
                energy = _time_axis(_clean(energy_raw, energy_var.attrs), len(times), "ENERGY_VALS")
                energy *= energy_factor

                flux_var = cdf["EFLUX_VS_PA_E"]
                eflux = _clean(flux_var[i0:i1], flux_var.attrs)
                if eflux.shape == (len(times), energy.shape[1], pitch.shape[1]):
                    eflux = np.swapaxes(eflux, 1, 2)
                if eflux.shape != (len(times), pitch.shape[1], energy.shape[1]):
                    raise ValueError(f"EFLUX_VS_PA_E 维度 {eflux.shape} 与 PA/E 轴不符")

                if config.get("PAD_APPLY_QUALITY_FILTER", True) and "QUALITY_FLAG" in cdf:
                    quality = np.asarray(cdf["QUALITY_FLAG"][i0:i1], dtype=np.uint32)
                    quality = quality.reshape(len(times), -1)
                    # Default rejects overflow / alternate energy table / spoiler test.
                    # Snapshot (bit 1) and attenuator (bit 4) are cautions, not automatic
                    # invalidation; otherwise some entire real days disappear.
                    bad_bits = int(config.get("PAD_QUALITY_BAD_BITS", 0x000D))
                    bad_rows = np.any((quality & bad_bits) != 0, axis=1)
                    eflux[bad_rows] = np.nan
                    flagged_rows = int(np.any(quality != 0, axis=1).sum())
                else:
                    bad_rows = np.zeros(len(times), dtype=bool)
                    flagged_rows = 0

                in_band = np.isfinite(energy) & (energy >= lo) & (energy <= hi)
                selected = np.where(in_band[:, None, :], eflux, np.nan)
                psd = energy_flux_to_psd(selected, energy[:, None, :])
                valid = np.sum(np.isfinite(psd), axis=2)
                summed = np.where(valid > 0, np.nansum(psd, axis=2), np.nan)
                if not np.any(np.isfinite(summed) & (summed > 0)):
                    notes.append(f"{path.name}: {lo:g}–{hi:g} eV 内无有效 PSD；"
                                 f"能量范围约 {np.nanmin(energy):g}–{np.nanmax(energy):g} eV")
                    day += timedelta(days=1)
                    continue
                all_times.append(times)
                all_pitch.append(pitch)
                all_psd.append(summed)
                notes.append(f"{path.name}: 使用 EFLUX_VS_PA_E→PSD，记录={len(times)}，"
                             f"有效={int(np.any(np.isfinite(summed), axis=1).sum())}，"
                             f"质量剔除={int(bad_rows.sum())}，非零质量标志={flagged_rows}"
                             "（请结合仪器文档核验）")
        except Exception as exc:
            notes.append(f"{path.name}: {type(exc).__name__}: {exc}")
        day += timedelta(days=1)

    if not all_times:
        return {"available": False, "reason": "未读到 PSP SPE PAD：" + "；".join(notes[:8]),
                "notes": notes}
    times = pd.DatetimeIndex(np.concatenate([t.to_numpy() for t in all_times]))
    pitch = np.concatenate(all_pitch)
    values = np.concatenate(all_psd)
    order = np.argsort(times.asi8)
    times, pitch, values = times[order], pitch[order], values[order]
    _, unique = np.unique(times.asi8, return_index=True)
    unique.sort()
    times, pitch, values = times[unique], pitch[unique], values[unique]
    reference_pitch = np.nanmedian(pitch, axis=0)
    if not np.all(np.isfinite(reference_pitch)):
        return {"available": False, "reason": "PITCHANGLE 角度轴无有效值", "notes": notes}
    if np.nanmax(np.abs(pitch - reference_pitch)) > 5:
        notes.append("注意：投掷角通道中心随时间变化超过 5°；图中使用各通道的中位角")
    return {"available": True, "times": times, "pitch": reference_pitch,
            "flux": values, "quantity": "PSD", "psd_units": "s^3/km^6",
            "energy_range_ev": (lo, hi), "notes": notes,
            "reason": "；".join(n for n in notes if "非零质量标志" in n)[:1000]}
