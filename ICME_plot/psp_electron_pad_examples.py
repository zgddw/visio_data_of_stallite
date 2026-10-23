"""PSP SWEAP/SPAN-Electron L3 PAD 事例程序。

本程序与 pro_of_read.py 的 cdfdataread() 使用相同的“按日期寻找 CDF、
按时间裁剪、读取变量属性”的框架，但不把 PAD 三维数组塞入 DataFrame。
这样可保留真实的 (time, pitch angle, energy) 维度；不需要修改原有
cdfdataread()，也不会影响其他数据读取程序。

已按实际 v04 CDF 核验：
    Epoch                 (N,)
    PITCHANGLE            (N, 12)
    ENERGY_VALS           (N, 32)
    EFLUX_VS_PA_E         (N, 12, 32)
    QUALITY_FLAG          (N, 2)

v04 文件没有物理保存虚拟变量 EFLUX_VS_PA_E_atE_byP；它是对
EFLUX_VS_PA_E 的 alternate view。因此本程序读取实际存在的主变量，
固定能量通道后得到 J(time, pitch angle)。
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import spacepy.pycdf


# ============================== 用户配置 ==============================
PAD_ROOT = r"I:\stdata\psp\sweap\spe\PSP_SWP_SPE_SF0_L3_PAD"
FILE_PREFIX = "psp_swp_spe_sf0_l3_pad"

# 事件索引为 0 起始。当前默认只画 2023-03-10：它在本地 v04 文件中有
# 12,360 个记录，覆盖全天。将此项改为 [3, 4] 可同时画 2023-02-19 和
# 2023-03-10；也可以只保留一个索引。
EVENT_INDEX = [1,2,3,4,5,6,7,8,9]

# 每个事件画一个面板；每个面板固定一个目标能量，并自动选择最近的真实
# ENERGY_VALS 通道。可按需要改为 [100.0]、[100.0, 500.0, 1000.0] 等。
TARGET_ENERGIES_EV = [253.29, 314.45]

# 使用 Shock 前/后各多少小时的窗口。数据在跨日时会自动找相邻日期文件。
HOURS_BEFORE_SHOCK = 1.0
HOURS_AFTER_END = 1.0

# QUALITY_FLAG 中 bits 11--16 是保留位；实际 v04 文件普遍含 4096 (bit 13)，
# 因此不能简单要求 QUALITY_FLAG == 0，否则会丢弃全部记录。
# 下列掩码筛掉文档定义的非保留质量位：1--10。
QUALITY_BAD_BITS = 0x03FF
APPLY_QUALITY_FILTER = True

# PNG 输出目录。设为 None 时输出到本脚本所在目录。
OUTPUT_DIR = r"E:\ZhaoWenjin_Reserch\pileup\双向电子流统计\psp"
# =====================================================================


# 从“有激波psp.txt”转写；索引 0--8 与原表行顺序一致。
EVENTS = [
    {"shock": "2019-03-15 09:00:00", "arrival": "2019-03-15 12:11:00", "end": "2019-03-15 17:49:00"},
    {"shock": "2021-09-26 08:55:00", "arrival": "2021-09-26 23:04:00", "end": "2021-09-27 12:36:00"},
    {"shock": "2022-07-02 04:42:00", "arrival": "2022-07-02 20:02:00", "end": "2022-07-04 19:42:00"},
    {"shock": "2023-02-19 16:07:00", "arrival": "2023-02-19 22:54:00", "end": "2023-02-20 06:04:00"},
    {"shock": "2023-03-10 04:58:00", "arrival": "2023-03-10 07:05:00", "end": "2023-03-10 08:49:00"},
    {"shock": "2023-05-11 15:18:00", "arrival": "2023-05-11 22:55:00", "end": "2023-05-12 12:49:00"},
    {"shock": "2024-05-14 18:41:00", "arrival": "2024-05-14 20:34:00", "end": "2024-05-14 23:40:00"},
    {"shock": "2024-08-26 02:34:00", "arrival": "2024-08-26 12:36:00", "end": "2024-08-27 15:45:00"},
    {"shock": "2025-01-22 19:42:00", "arrival": "2025-01-22 22:11:00", "end": "2025-01-23 02:39:00"},
]


def _as_datetime_array(values: Any) -> np.ndarray:
    """转换 spacepy 的 Epoch 输出为 datetime64[ns] 数组。"""
    return pd.to_datetime(np.asarray(values)).to_numpy(dtype="datetime64[ns]")


def _clean_numeric(values: np.ndarray, attrs: Any) -> np.ndarray:
    """根据 CDF 的 FILLVAL/VALIDMIN/VALIDMAX 清除无效数。"""
    result = np.asarray(values, dtype=float).copy()
    invalid = ~np.isfinite(result)
    for attr_name, comparison in (("FILLVAL", "fill"), ("VALIDMIN", "min"), ("VALIDMAX", "max")):
        value = attrs.get(attr_name)
        if value is None:
            continue
        try:
            scalar = float(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(scalar):
            continue
        if comparison == "fill":
            invalid |= np.isclose(result, scalar, rtol=1e-5, atol=1e-8)
        elif comparison == "min":
            invalid |= result < scalar
        else:
            invalid |= result > scalar
    result[invalid] = np.nan
    return result


def _date_range(start: datetime, end: datetime):
    day = start.date()
    while day <= end.date():
        yield day
        day += timedelta(days=1)


def _find_daily_file(root: str, date_obj) -> str | None:
    """兼容 root/YYYY/ 和 root/ 两种用户目录布局，选择最高版本文件。"""
    ymd = date_obj.strftime("%Y%m%d")
    patterns = [
        os.path.join(root, date_obj.strftime("%Y"), f"{FILE_PREFIX}_{ymd}_v*.cdf"),
        os.path.join(root, f"{FILE_PREFIX}_{ymd}_v*.cdf"),
    ]
    matches = sorted({file for pattern in patterns for file in glob.glob(pattern)})
    return matches[-1] if matches else None


def _canonical_pad_shape(flux: np.ndarray, ntime: int, npitch: int, nenergy: int) -> np.ndarray:
    """将 PAD 主变量统一为 (time, pitch, energy)，若维度不符则停止。"""
    flux = np.asarray(flux)
    if flux.shape == (ntime, npitch, nenergy):
        return flux
    if flux.shape == (ntime, nenergy, npitch):
        return np.swapaxes(flux, 1, 2)
    raise ValueError(
        "PAD 维度与 Epoch/PITCHANGLE/ENERGY_VALS 不一致："
        f"flux={flux.shape}, expected=({ntime}, {npitch}, {nenergy})。"
    )


def read_pad_multidimensional(root: str, start: datetime, end: datetime) -> dict[str, np.ndarray]:
    """专门读取 PAD 多维变量；这是对 cdfdataread() 的补充，不替换它。"""
    chunks: dict[str, list[np.ndarray]] = {"time": [], "pitch": [], "energy": [], "flux": [], "quality": []}
    used_files: list[str] = []

    for day in _date_range(start, end):
        filename = _find_daily_file(root, day)
        if filename is None:
            print(f"[WARN] 未找到 {day:%Y-%m-%d} 的 PAD CDF。")
            continue

        with spacepy.pycdf.CDF(filename) as cdf:
            required = ["Epoch", "PITCHANGLE", "ENERGY_VALS", "QUALITY_FLAG", "EFLUX_VS_PA_E"]
            missing = [name for name in required if name not in cdf]
            if missing:
                print(f"[WARN] 跳过 {Path(filename).name}：缺少 {missing}")
                continue

            raw_time = _as_datetime_array(cdf["Epoch"][...])
            select = (raw_time >= np.datetime64(start)) & (raw_time <= np.datetime64(end))
            if not np.any(select):
                continue

            raw_pitch = _clean_numeric(cdf["PITCHANGLE"][...], cdf["PITCHANGLE"].attrs)
            raw_energy = _clean_numeric(cdf["ENERGY_VALS"][...], cdf["ENERGY_VALS"].attrs)
            raw_flux = _clean_numeric(cdf["EFLUX_VS_PA_E"][...], cdf["EFLUX_VS_PA_E"].attrs)
            raw_flux = _canonical_pad_shape(raw_flux, len(raw_time), raw_pitch.shape[-1], raw_energy.shape[-1])
            raw_quality = np.asarray(cdf["QUALITY_FLAG"][...], dtype=np.uint16)

            chunks["time"].append(raw_time[select])
            chunks["pitch"].append(raw_pitch[select])
            chunks["energy"].append(raw_energy[select])
            chunks["flux"].append(raw_flux[select])
            chunks["quality"].append(raw_quality[select])
            used_files.append(filename)

    if not chunks["time"]:
        raise FileNotFoundError(f"{start} 至 {end} 没有可用的 PAD 记录；请检查 PAD_ROOT。")

    result = {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}
    # 如果相邻日文件存在重叠记录，则保留一条，避免 pcolormesh 重复列。
    _, unique_idx = np.unique(result["time"], return_index=True)
    unique_idx.sort()
    for name in result:
        result[name] = result[name][unique_idx]
    result["files"] = np.asarray(used_files, dtype=object)
    return result


def good_quality_mask(quality: np.ndarray) -> np.ndarray:
    """将 QUALITY_FLAG 的两个分量合并；只筛文档已定义的 bits 1--10。"""
    q = np.asarray(quality, dtype=np.uint16)
    if q.ndim == 1:
        q = q[:, None]
    return np.all((q & QUALITY_BAD_BITS) == 0, axis=1)


def _log_norm(values: np.ndarray) -> LogNorm:
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size == 0:
        raise ValueError("质量筛选后没有正的电子通量，无法使用对数色标绘图。")
    # 用分位数减轻极端值对色图的支配；必要时退回数据范围。
    vmin, vmax = np.nanpercentile(valid, [2, 98])
    if not np.isfinite(vmin) or vmin <= 0:
        vmin = np.nanmin(valid)
    if vmax <= vmin:
        vmax = np.nanmax(valid)
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_event(event_index: int, event: dict[str, str], pad: dict[str, np.ndarray], output_dir: Path) -> Path:
    shock = pd.Timestamp(event["shock"]).to_pydatetime()
    arrival = pd.Timestamp(event["arrival"]).to_pydatetime()
    end = pd.Timestamp(event["end"]).to_pydatetime()
    time = pd.to_datetime(pad["time"])
    pitch = pad["pitch"]
    energy = pad["energy"]
    flux = pad["flux"]
    quality_ok = good_quality_mask(pad["quality"])
    if not APPLY_QUALITY_FILTER:
        quality_ok[:] = True

    # 以各记录的能量中位数选择通道。即使能量表随时间变化，仍取同一通道索引。
    energy_reference = np.nanmedian(energy, axis=0)
    selected_indices = [int(np.nanargmin(np.abs(energy_reference - target))) for target in TARGET_ENERGIES_EV]
    nrows = len(selected_indices)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 4.0 * nrows), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, energy_index, target in zip(axes, selected_indices, TARGET_ENERGIES_EV):
        # (N, pitch) -> (pitch, N), with bad quality / fill / non-positive values masked.
        values = flux[:, :, energy_index].copy()
        values[~quality_ok, :] = np.nan
        values[values <= 0] = np.nan
        mesh = ax.pcolormesh(
            time,
            np.nanmedian(pitch, axis=0),
            np.ma.masked_invalid(values.T),
            shading="auto",
            cmap="turbo",
            norm=_log_norm(values),
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        cbar.set_label("Differential Energy Flux\n[eV cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$]")
        ax.axvline(shock, color="white", ls="--", lw=1.3, label="Shock")
        ax.axvline(arrival, color="cyan", ls="--", lw=1.3, label="Arrival")
        ax.axvline(end, color="magenta", ls="--", lw=1.3, label="End")
        actual = energy_reference[energy_index]
        ax.set_ylabel("Pitch angle (deg)")
        ax.set_yticks(np.arange(0, 181, 30))
        ax.set_ylim(0, 180)
        ax.set_title(f"target {target:g} eV; selected channel {actual:.1f} eV")
        ax.legend(loc="upper right", ncol=3, fontsize=9)

    axes[-1].set_xlabel("Time (UTC)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    good_fraction = quality_ok.mean() * 100.0
    fig.suptitle(
        f"PSP SPE electron PAD | event {event_index}: {shock:%Y-%m-%d} | "
        f"quality retained: {good_fraction:.1f}%",
        fontsize=14,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"psp_spe_pad_event_{event_index}_{shock:%Y%m%d}.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_file


def main() -> None:
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else Path(__file__).resolve().parent
    for index in EVENT_INDEX:
        if index < 0 or index >= len(EVENTS):
            raise IndexError(f"EVENT_INDEX={index} 超出范围；可用索引为 0--{len(EVENTS) - 1}。")
        event = EVENTS[index]
        shock = pd.Timestamp(event["shock"]).to_pydatetime()
        end = pd.Timestamp(event["end"]).to_pydatetime()
        start = shock - timedelta(hours=HOURS_BEFORE_SHOCK)
        finish = end + timedelta(hours=HOURS_AFTER_END)
        print(f"[INFO] 读取 event {index}: {start} -- {finish}")
        pad = read_pad_multidimensional(PAD_ROOT, start, finish)
        output_file = plot_event(index, event, pad, output_dir)
        print(f"[OK] 已保存: {output_file}")


if __name__ == "__main__":
    main()
