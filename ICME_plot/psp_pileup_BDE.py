"""
PSP SPE PAD 事件分析程序（修正版）
根据 "有激波psp.txt" 中每个事件的 Mark1-Mark2 时间段，
读取对应 PAD 数据，绘制投掷角-时间色图，并标记通量各向异性区间。
"""

import glob
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import spacepy.pycdf

# ============================== 用户配置 ==============================
EVENT_FILE = r"E:\ZhaoWenjin_Reserch\pileup\有激波psp.txt"
PAD_ROOT = r"I:\stdata\psp\sweap\spe\PSP_SWP_SPE_SF0_L3_PAD"
FILE_PREFIX = "psp_swp_spe_sf0_l3_pad"

TARGET_ENERGY_EV = 500   # 目标能量 (eV)，可改为其他值

QUALITY_BAD_BITS = 0x03FF   # 只筛掉 bits 1–10
APPLY_QUALITY_FILTER = True

OUTPUT_DIR = None           # 图片保存目录，None 表示脚本所在目录
PROCESS_ALL_EVENTS = True   # True 处理全部事件；False 则只处理 EVENT_INDICES
EVENT_INDICES = [1,2,3,4,5,6,7,8,9]         # 当 PROCESS_ALL_EVENTS=False 时生效
# =====================================================================


def _as_datetime_array(values: Any) -> np.ndarray:
    return pd.to_datetime(np.asarray(values)).to_numpy(dtype="datetime64[ns]")


def _clean_numeric(values: np.ndarray, attrs: Any) -> np.ndarray:
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
    ymd = date_obj.strftime("%Y%m%d")
    patterns = [
        os.path.join(root, date_obj.strftime("%Y"), f"{FILE_PREFIX}_{ymd}_v*.cdf"),
        os.path.join(root, f"{FILE_PREFIX}_{ymd}_v*.cdf"),
    ]
    matches = sorted({file for pattern in patterns for file in glob.glob(pattern)})
    return matches[-1] if matches else None


def _canonical_pad_shape(flux: np.ndarray, ntime: int, npitch: int, nenergy: int) -> np.ndarray:
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
    _, unique_idx = np.unique(result["time"], return_index=True)
    unique_idx.sort()
    for name in result:
        result[name] = result[name][unique_idx]
    result["files"] = np.asarray(used_files, dtype=object)
    return result


def good_quality_mask(quality: np.ndarray) -> np.ndarray:
    q = np.asarray(quality, dtype=np.uint16)
    if q.ndim == 1:
        q = q[:, None]
    return np.all((q & QUALITY_BAD_BITS) == 0, axis=1)


def load_events(event_file: str) -> pd.DataFrame:
    df = pd.read_csv(event_file, sep='\t', parse_dates=['Shock', 'Arrival', 'End', 'Mark1', 'Mark2'])
    return df


def find_nearest_index(array: np.ndarray, target: float) -> int:
    """返回 array 中最接近 target 的索引，忽略 NaN。"""
    diff = np.abs(array - target)
    diff[np.isnan(diff)] = np.inf   # 将 NaN 设为无穷大，避免影响 argmin
    return int(np.argmin(diff))


def find_continuous_intervals(condition: np.ndarray) -> List[Tuple[int, int]]:
    intervals = []
    i = 0
    n = len(condition)
    while i < n:
        if condition[i]:
            start = i
            while i < n and condition[i]:
                i += 1
            end = i
            intervals.append((start, end))
        else:
            i += 1
    return intervals


def plot_event(event_row: pd.Series, pad: dict, target_energy: float, output_dir: Path):
    mark1 = event_row['Mark1'].to_pydatetime()
    mark2 = event_row['Mark2'].to_pydatetime()

    time_all = pd.to_datetime(pad["time"])
    pitch_all = pad["pitch"]          # (N, 12)
    energy_all = pad["energy"]        # (N, 32)
    flux_all = pad["flux"]            # (N, 12, 32)

    quality_ok = good_quality_mask(pad["quality"])
    if not APPLY_QUALITY_FILTER:
        quality_ok[:] = True

    # ---- 裁剪到 Mark1-Mark2 时间段 ----
    mask = (time_all >= mark1) & (time_all <= mark2)
    if not np.any(mask):
        print(f"   [WARN] Mark1-Mark2 时间段内无数据点。")
        return None

    time = time_all[mask]
    pitch = pitch_all[mask]
    energy = energy_all[mask]
    flux = flux_all[mask]
    quality = quality_ok[mask]

    # ---- 选择目标能量通道 ----
    energy_reference = np.nanmedian(energy, axis=0)  # (32,)
    energy_index = find_nearest_index(energy_reference, target_energy)
    actual_energy = energy_reference[energy_index]
    print(f"   目标能量 {target_energy:.1f} eV，实际通道能量 {actual_energy:.1f} eV")

    # ---- 提取该能量通道的通量 (N, pitch) ----
    values = flux[:, :, energy_index].copy()  # (N, 12)
    values[~quality, :] = np.nan
    values[values <= 0] = np.nan

    # ---- 检查有效数据 ----
    valid_vals = values[np.isfinite(values) & (values > 0)]
    if valid_vals.size == 0:
        print(f"   [WARN] 没有正的有效通量数据，无法绘图。")
        return None

    # ---- 计算投掷角比值（仅 Mark1-Mark2 内） ----
    pitch_t = pitch          # (M, 12)
    values_t = values        # (M, 12)
    target_angles = [0, 90, 180]
    flux_at_target = np.full((len(values_t), len(target_angles)), np.nan)

    for i, pa in enumerate(pitch_t):
        for j, t_angle in enumerate(target_angles):
            idx = find_nearest_index(pa, t_angle)
            flux_at_target[i, j] = values_t[i, idx]

    f0 = flux_at_target[:, 0]
    f90 = flux_at_target[:, 1]
    f180 = flux_at_target[:, 2]

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio0 = f0 / f90
        ratio180 = f180 / f90

    condition = (ratio0 >= 2.0) & (ratio180 >= 2.0) & np.isfinite(ratio0) & np.isfinite(ratio180)

    intervals = find_continuous_intervals(condition)
    # 修正：DatetimeIndex 直接支持整数索引，使用 time[start] 而非 time.iloc[start]
    intervals_time = [(time[start], time[end-1]) for start, end in intervals]

    total_duration = (mark2 - mark1).total_seconds()
    covered_duration = sum((end - start).total_seconds() for start, end in intervals_time)
    ratio = covered_duration / total_duration if total_duration > 0 else 0
    print(f"   满足条件区间总时长: {covered_duration:.1f} 秒，占 Mark1-Mark2 比例: {ratio*100:.2f}%")

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(14, 6))

    # 确定色图归一化
    if valid_vals.size < 10:
        print(f"   [INFO] 有效数据点较少 ({valid_vals.size})，使用线性色标。")
        norm = None
    else:
        vmin = np.nanpercentile(valid_vals, 2)
        vmax = np.nanpercentile(valid_vals, 98)
        if not np.isfinite(vmin) or vmin <= 0:
            vmin = np.nanmin(valid_vals)
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = np.nanmax(valid_vals)
        if vmax <= vmin:
            vmax = vmin * 10.0
        norm = LogNorm(vmin=vmin, vmax=vmax)

    # 使用 pcolormesh
    mesh = ax.pcolormesh(
        time,
        np.nanmedian(pitch, axis=0),   # 取中位数作为固定投掷角网格（便于显示）
        np.ma.masked_invalid(values.T),
        shading='auto',
        cmap='turbo',
        norm=norm
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    cbar.set_label("Differential Energy Flux\n[eV cm$^{-2}$ s$^{-1}$ sr$^{-1}$ eV$^{-1}$]")

    # 标记 Mark1 和 Mark2
    ax.axvline(mark1, color='white', ls='--', lw=1.2, label='Mark1')
    ax.axvline(mark2, color='white', ls='--', lw=1.2, label='Mark2')

    # 标记满足条件的区间（在图像上方外部）
    for start, end in intervals_time:
        ax.axvspan(start, end, ymin=1.02, ymax=1.08, color='red', alpha=0.5,
                   clip_on=False, transform=ax.get_xaxis_transform())

    ax.set_ylabel('Pitch angle (deg)')
    ax.set_yticks(np.arange(0, 181, 30))
    ax.set_ylim(0, 180)
    ax.set_xlabel('Time (UTC)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    ax.legend(loc='upper right')

    shock = event_row['Shock'].to_pydatetime()
    ax.set_title(f"PSP SPE electron PAD | Event: {shock:%Y-%m-%d} | Energy: {actual_energy:.1f} eV")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"psp_pad_event_{event_row.name}_{shock:%Y%m%d}_Mark1Mark2.png"
    fig.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"   [OK] 图片已保存: {output_file}")

    return output_file


def main():
    df_events = load_events(EVENT_FILE)
    print(f"读取到 {len(df_events)} 个事件。")

    if PROCESS_ALL_EVENTS:
        indices_to_process = list(range(len(df_events)))
    else:
        indices_to_process = EVENT_INDICES

    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else Path(__file__).resolve().parent

    for idx in indices_to_process:
        if idx < 0 or idx >= len(df_events):
            print(f"[WARN] 索引 {idx} 超出范围，跳过。")
            continue
        row = df_events.iloc[idx]
        mark1 = row['Mark1'].to_pydatetime()
        mark2 = row['Mark2'].to_pydatetime()
        shock = row['Shock'].to_pydatetime()

        print(f"\n处理事件 {idx}: Shock {shock:%Y-%m-%d %H:%M}，Mark1 {mark1} -- Mark2 {mark2}")

        try:
            pad = read_pad_multidimensional(PAD_ROOT, mark1, mark2)
        except FileNotFoundError as e:
            print(f"   [ERROR] {e}")
            continue

        plot_event(row, pad, TARGET_ENERGY_EV, output_dir)

    print("\n所有事件处理完毕。")


if __name__ == "__main__":
    main()