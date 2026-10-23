# -*- coding: utf-8 -*-
"""
ICME Superposed Epoch / Normalized-Phase Analysis
==================================================

功能：
1. 读取 PSP 和 Solar Orbiter (SolO) ICME 事件列表；
2. 根据事件 shock / mark1 / mark2 / ICME end 定义三个阶段：
       phase1 = shock -> mark1
       phase2 = mark1 -> mark2
       phase3 = max(icme_start, mark2) -> icme_end
3. 读取磁场、质子密度、质子速度和质子温度；
4. 先在统一的 1 min 时间网格上构造 B、N、V、T；
5. 先逐时刻计算质子热压 Pth 和质子 beta，再进行时间分箱，
   避免使用 <N><T>/<B>^2 代替 <beta> 的统计偏差；
6. 每个事件的每个 phase 都归一化到固定数量的 bin；
7. 再对所有事件逐 bin 进行 superposed epoch 统计：mean / median / std / SEM / N；
8. 输出 CSV 和 PNG 图。

重要说明：
- 事件列表中的时间默认解释为 UTC；
- PSP 的 wp_moment 目前按照“热速度尺度，单位 km/s”的定义计算 T：
      T = m_p * w_p^2 / (2 k_B)
  只有当你的 wp_moment 数据产品定义确实如此时，这个公式才成立。
  若产品文档给出的是其它 thermal speed 定义，只需修改 calculate_psp_temperature_K()。

依赖：
    numpy, pandas, matplotlib, scipy, pro_of_read.cdfdataread
"""

import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants

from pro_of_read import cdfdataread


# ======================================================
# 1. 常量、参数和配置
# ======================================================
M_P = constants.m_p                  # kg
K_B = constants.k                     # J/K
MU0 = constants.mu_0                  # N/A^2
EV_TO_K = 11604.518                    # 1 eV -> K

CM3_TO_M3 = 1.0e6                     # cm^-3 -> m^-3
NPA_TO_PA = 1.0e-9                    # nPa -> Pa

# 图中物理量顺序
PHYS_PARAMS = [
    "B_total",
    "N",
    "V_total",
    "T_K",
    "Pth_nPa",
    "Beta",
]

# 每个阶段采用的归一化 bin 数
N_BINS = {
    "phase1": 65,
    "phase2": 50,
    "phase3": 231,
}

# 统一内部时间网格
COMMON_CADENCE_MIN = 1.0

# 插值时允许跨越的最大数据间隔。
# 设为 None 表示只要求落在原始数据时间范围内，不限制最大间隔。
MAX_INTERP_GAP_MIN = 5.0

# 输出目录
OUTPUT_DIR = "seaana_output"


# ======================================================
# 2. 时间处理工具
# ======================================================
def parse_datetime_utc(text):
    """解析事件表中的时间，并明确按 UTC 处理，返回 naive datetime(UTC)。"""
    text = str(text).strip()
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间: {text}")


def pd_to_utc_naive(series_or_array):
    """将时间数组统一转为 UTC 的 naive pandas.DatetimeIndex。"""
    ts = pd.to_datetime(series_or_array, errors="coerce", utc=True)
    return pd.DatetimeIndex(ts).tz_localize(None)


def datetime_to_seconds(dt):
    """把 UTC naive datetime 转成 Unix 秒，不依赖计算机本地时区。"""
    return pd.Timestamp(dt, tz="UTC").timestamp()


def seconds_to_utc_naive_datetime(seconds):
    """Unix 秒 -> UTC naive datetime。"""
    return pd.Timestamp(seconds, unit="s", tz="UTC").tz_localize(None).to_pydatetime()


# ======================================================
# 3. 数据清洗工具
# ======================================================
def clean_numeric(values):
    """将输入转换为 float 一维数组；无法转换的设为 NaN。"""
    arr = np.asarray(values)
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    try:
        arr = arr.astype(float)
    except (TypeError, ValueError):
        arr = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
    return np.ravel(arr)


def clean_time_value_pair(df, time_col="t", value_col=None):
    """从 DataFrame 中提取时间和值，并进行长度、时间、有限值基础清理。"""
    if df is None or df.empty:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype=float)

    if time_col not in df.columns:
        raise KeyError(f"数据中找不到时间列 {time_col!r}，现有列: {list(df.columns)}")
    if value_col is None or value_col not in df.columns:
        raise KeyError(f"数据中找不到变量列 {value_col!r}，现有列: {list(df.columns)}")

    times = pd_to_utc_naive(df[time_col])
    vals = clean_numeric(df[value_col])

    n = min(len(times), len(vals))
    times = times[:n]
    vals = vals[:n]

    valid_time = ~pd.isna(times)
    times = times[valid_time]
    vals = vals[valid_time]

    # 相同时间戳保留平均值，避免插值函数遇到重复时间。
    tmp = pd.DataFrame({"value": vals}, index=times)
    tmp = tmp.groupby(level=0).mean().sort_index()

    return tmp.index.to_numpy(), tmp["value"].to_numpy(dtype=float)


def get_required_column(df, candidates, description):
    """从候选列名中找出实际存在的一列。"""
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"找不到 {description}。候选列={candidates}；实际列={list(df.columns)}"
    )


# ======================================================
# 4. 事件列表读取
# ======================================================
def parse_event_file(filename, satellite):
    """
    读取事件列表。

    PSP：
        0-Shock, 1-ICME start, 2-End, 3-Mark1, 4-Mark2, ..., 9-Ratio

    SOLO：
        0-Event_Index, 1-Shock, 2-Start, 3-End, 4-Mark1, 5-Mark2, ..., 8-Ratio
    """
    events = []

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"事件文件不存在: {filename}")

    with open(filename, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # 找到标题之后开始读数据；找不到标题则从第一行开始。
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Shock") or line.strip().startswith("#"):
            start = i + 1
            break

    for line_no, line in enumerate(lines[start:], start=start + 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # 优先 tab，其次多个空格，最后任意空白。
        if "\t" in line:
            parts = [x.strip() for x in line.split("\t")]
        else:
            parts = [x.strip() for x in re.split(r"\s{2,}", line)]
        if len(parts) < 10:
            parts = line.split()

        try:
            if satellite.upper() == "PSP":
                if len(parts) < 10:
                    continue
                shock_str = parts[0]
                icme_start_str = parts[1]
                icme_end_str = parts[2]
                mark1_str = parts[3]
                mark2_str = parts[4]
                ratio_str = parts[9]
            else:
                if len(parts) < 9:
                    continue
                shock_str = parts[1]
                icme_start_str = parts[2]
                icme_end_str = parts[3]
                mark1_str = parts[4]
                mark2_str = parts[5]
                ratio_str = parts[8]

            ratio = float(ratio_str)
            if not np.isfinite(ratio) or ratio < 0.5:
                continue

            shock = parse_datetime_utc(shock_str)
            icme_start = parse_datetime_utc(icme_start_str)
            icme_end = parse_datetime_utc(icme_end_str)
            mark1 = parse_datetime_utc(mark1_str)
            mark2 = parse_datetime_utc(mark2_str)

        except (ValueError, TypeError) as exc:
            print(f"⚠ 跳过 {filename} 第 {line_no} 行：{exc}")
            continue

        # 原程序的“无激波”事件筛选逻辑：shock == ICME start 则跳过。
        if shock == icme_start:
            continue

        # 保留原定义：phase3 从 max(icme_start, mark2) 开始。
        icme_start_eff = max(icme_start, mark2)

        # 基础时间顺序检查。
        if not (shock < mark1 <= mark2 and icme_end > icme_start_eff):
            print(
                f"⚠ 跳过时间边界异常事件 {shock} ({satellite}): "
                f"shock={shock}, mark1={mark1}, mark2={mark2}, "
                f"icme_start_eff={icme_start_eff}, end={icme_end}"
            )
            continue

        events.append(
            {
                "satellite": satellite.upper(),
                "shock": shock,
                "icme_start": icme_start,
                "icme_end": icme_end,
                "mark1": mark1,
                "mark2": mark2,
                "icme_start_eff": icme_start_eff,
                "ratio": ratio,
            }
        )

    return events


# ======================================================
# 5. PSP / SolO 温度与数据读取
# ======================================================
def calculate_psp_temperature_K(wp_km_s):
    """
    根据当前程序采用的定义计算 PSP 质子温度。

    假设：wp_moment 是等效热速度，单位 km/s，且
        E_thermal = (1/2) m_p w_p^2 = k_B T

    因此：T = m_p w_p^2 / (2 k_B)
    """
    wp_km_s = clean_numeric(wp_km_s)
    wp_m_s = wp_km_s * 1000.0
    return M_P * wp_m_s**2 / (2.0 * K_B)


def read_event_data(event, config):
    """读取一个事件的数据并返回：{parameter: (time_array, value_array)}。"""
    sat = event["satellite"]
    start_dt = event["shock"] - timedelta(minutes=30)
    end_dt = event["icme_end"] + timedelta(minutes=30)

    data = {}

    if sat == "PSP":
        # --------------------------
        # 磁场
        # --------------------------
        mag_root = config["PSP"]["mag_root"]
        mag_prefix = config["PSP"]["mag_prefix"]
        mag_var = ["psp_fld_l2_mag_RTN_1min"]

        mag_dict = cdfdataread(mag_root, mag_var, start_dt, end_dt, mag_prefix)
        if mag_var[0] not in mag_dict or mag_dict[mag_var[0]] is None:
            print(f"警告: PSP 磁场数据缺失: {event['shock']}")
            return None

        df_mag = mag_dict[mag_var[0]]
        if df_mag.empty:
            print(f"警告: PSP 磁场数据为空: {event['shock']}")
            return None

        bx_col = get_required_column(
            df_mag,
            ["psp_fld_l2_mag_RTN_1min_X", "B_RTN_X", "Bx"],
            "PSP 磁场 X 分量",
        )
        by_col = get_required_column(
            df_mag,
            ["psp_fld_l2_mag_RTN_1min_Y", "B_RTN_Y", "By"],
            "PSP 磁场 Y 分量",
        )
        bz_col = get_required_column(
            df_mag,
            ["psp_fld_l2_mag_RTN_1min_Z", "B_RTN_Z", "Bz"],
            "PSP 磁场 Z 分量",
        )

        t_b, bx = clean_time_value_pair(df_mag, "t", bx_col)
        _, by = clean_time_value_pair(df_mag, "t", by_col)
        _, bz = clean_time_value_pair(df_mag, "t", bz_col)

        # 上面三个数组因为各列缺测情况可能不同，这里直接使用 DataFrame 的原始对齐方式重新计算。
        tmp_mag = pd.DataFrame(
            {
                "bx": clean_numeric(df_mag[bx_col]),
                "by": clean_numeric(df_mag[by_col]),
                "bz": clean_numeric(df_mag[bz_col]),
            },
            index=pd_to_utc_naive(df_mag["t"]),
        ).sort_index()
        tmp_mag["B_total"] = np.sqrt(tmp_mag["bx"]**2 + tmp_mag["by"]**2 + tmp_mag["bz"]**2)
        tmp_mag = tmp_mag[["B_total"]].groupby(level=0).mean()
        tmp_mag = tmp_mag.dropna(subset=["B_total"])
        data["B_total"] = (
            tmp_mag.index.to_numpy(),
            tmp_mag["B_total"].to_numpy(dtype=float),
        )

        # --------------------------
        # 等离子体
        # --------------------------
        plas_root = config["PSP"]["plas_root"]
        plas_prefix = config["PSP"]["plas_prefix"]
        plas_vars = ["np_moment", "vp_moment_RTN", "wp_moment"]

        plas_dict = cdfdataread(plas_root, plas_vars, start_dt, end_dt, plas_prefix)
        missing = [v for v in plas_vars if v not in plas_dict or plas_dict[v] is None]
        if missing:
            print(f"警告: PSP 等离子体数据缺失: {missing}, event={event['shock']}")
            return None

        df_n = plas_dict["np_moment"]
        df_v = plas_dict["vp_moment_RTN"]
        df_w = plas_dict["wp_moment"]

        if df_n.empty or df_v.empty or df_w.empty:
            print(f"警告: PSP 等离子体数据为空: {event['shock']}")
            return None

        n_col = get_required_column(df_n, ["Np", "np_moment", "N"], "PSP 质子密度")
        vx_col = get_required_column(df_v, ["Vp_X", "Vx", "vp_rtn_x"], "PSP Vx")
        vy_col = get_required_column(df_v, ["Vp_Y", "Vy", "vp_rtn_y"], "PSP Vy")
        vz_col = get_required_column(df_v, ["Vp_Z", "Vz", "vp_rtn_z"], "PSP Vz")
        wp_col = get_required_column(df_w, ["wp_moment", "Wp", "wp"], "PSP wp_moment")

        # 密度
        t_n, n = clean_time_value_pair(df_n, "t", n_col)
        data["N"] = (t_n, n)

        # 速度
        idx_v = pd_to_utc_naive(df_v["t"])
        tmp_v = pd.DataFrame(
            {
                "vx": clean_numeric(df_v[vx_col]),
                "vy": clean_numeric(df_v[vy_col]),
                "vz": clean_numeric(df_v[vz_col]),
            },
            index=idx_v,
        ).sort_index()
        tmp_v["V_total"] = np.sqrt(tmp_v["vx"]**2 + tmp_v["vy"]**2 + tmp_v["vz"]**2)
        tmp_v = tmp_v[["V_total"]].groupby(level=0).mean().dropna()
        data["V_total"] = (
            tmp_v.index.to_numpy(),
            tmp_v["V_total"].to_numpy(dtype=float),
        )

        # 温度
        t_w, wp = clean_time_value_pair(df_w, "t", wp_col)
        T_K = calculate_psp_temperature_K(wp)
        data["T_K"] = (t_w, T_K)

    elif sat == "SOLO":
        # --------------------------
        # 磁场
        # --------------------------
        mag_root = config["SOLO"]["mag_root"]
        mag_prefix = config["SOLO"]["mag_prefix"]
        mag_var = ["B_RTN"]

        mag_dict = cdfdataread(mag_root, mag_var, start_dt, end_dt, mag_prefix)
        if mag_var[0] not in mag_dict or mag_dict[mag_var[0]] is None:
            print(f"警告: SOLO 磁场数据缺失: {event['shock']}")
            return None

        df_mag = mag_dict[mag_var[0]]
        if df_mag.empty:
            print(f"警告: SOLO 磁场数据为空: {event['shock']}")
            return None

        bx_col = get_required_column(df_mag, ["B_RTN_X", "Bx"], "SOLO Bx")
        by_col = get_required_column(df_mag, ["B_RTN_Y", "By"], "SOLO By")
        bz_col = get_required_column(df_mag, ["B_RTN_Z", "Bz"], "SOLO Bz")

        tmp_mag = pd.DataFrame(
            {
                "bx": clean_numeric(df_mag[bx_col]),
                "by": clean_numeric(df_mag[by_col]),
                "bz": clean_numeric(df_mag[bz_col]),
            },
            index=pd_to_utc_naive(df_mag["t"]),
        ).sort_index()
        tmp_mag["B_total"] = np.sqrt(tmp_mag["bx"]**2 + tmp_mag["by"]**2 + tmp_mag["bz"]**2)
        tmp_mag = tmp_mag[["B_total"]].groupby(level=0).mean().dropna()
        data["B_total"] = (
            tmp_mag.index.to_numpy(),
            tmp_mag["B_total"].to_numpy(dtype=float),
        )

        # --------------------------
        # 等离子体
        # --------------------------
        plas_root = config["SOLO"]["plas_root"]
        plas_prefix = config["SOLO"]["plas_prefix"]
        plas_vars = ["N", "V_RTN", "T"]

        plas_dict = cdfdataread(plas_root, plas_vars, start_dt, end_dt, plas_prefix)
        missing = [v for v in plas_vars if v not in plas_dict or plas_dict[v] is None]
        if missing:
            print(f"警告: SOLO 等离子体数据缺失: {missing}, event={event['shock']}")
            return None

        df_n = plas_dict["N"]
        df_v = plas_dict["V_RTN"]
        df_T = plas_dict["T"]

        if df_n.empty or df_v.empty or df_T.empty:
            print(f"警告: SOLO 等离子体数据为空: {event['shock']}")
            return None

        n_col = get_required_column(df_n, ["Np", "N", "np"], "SOLO 质子密度")
        vx_col = get_required_column(df_v, ["Vp_X", "Vx", "V_RTN_X"], "SOLO Vx")
        vy_col = get_required_column(df_v, ["Vp_Y", "Vy", "V_RTN_Y"], "SOLO Vy")
        vz_col = get_required_column(df_v, ["Vp_Z", "Vz", "V_RTN_Z"], "SOLO Vz")
        T_col = get_required_column(df_T, ["Tp", "T", "Tp_eV"], "SOLO 质子温度")

        t_n, n = clean_time_value_pair(df_n, "t", n_col)
        data["N"] = (t_n, n)

        idx_v = pd_to_utc_naive(df_v["t"])
        tmp_v = pd.DataFrame(
            {
                "vx": clean_numeric(df_v[vx_col]),
                "vy": clean_numeric(df_v[vy_col]),
                "vz": clean_numeric(df_v[vz_col]),
            },
            index=idx_v,
        ).sort_index()
        tmp_v["V_total"] = np.sqrt(tmp_v["vx"]**2 + tmp_v["vy"]**2 + tmp_v["vz"]**2)
        tmp_v = tmp_v[["V_total"]].groupby(level=0).mean().dropna()
        data["V_total"] = (
            tmp_v.index.to_numpy(),
            tmp_v["V_total"].to_numpy(dtype=float),
        )

        t_T, T_eV = clean_time_value_pair(df_T, "t", T_col)
        T_K = T_eV * EV_TO_K
        data["T_K"] = (t_T, T_K)

    else:
        raise ValueError(f"不支持的卫星: {sat}")

    # 清理：只保留有限数值；温度、密度、B 等明显不合理值后续再在统计阶段处理。
    for key, (times, values) in list(data.items()):
        times = pd_to_utc_naive(times)
        values = clean_numeric(values)
        n = min(len(times), len(values))
        times = times[:n]
        values = values[:n]
        mask = ~pd.isna(times) & np.isfinite(values)
        data[key] = (times[mask].to_numpy(), values[mask])

    return data


# ======================================================
# 6. 将一个物理量插值到共同时间轴
# ======================================================
def interpolate_to_grid(times, values, grid_times, max_gap_min=5.0):
    """
    一维线性插值到共同时间轴。

    为避免大段缺测被错误地“连线”，如果 max_gap_min 不为 None，
    则只允许两个相邻有效原始点之间的间隔 <= max_gap_min。
    """
    src_t = pd.to_datetime(times, utc=True).tz_localize(None)
    dst_t = pd.to_datetime(grid_times, utc=True).tz_localize(None)

    x = src_t.view("int64") / 1.0e9
    y = clean_numeric(values)
    x_new = dst_t.view("int64") / 1.0e9

    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[valid]
    y = np.asarray(y)[valid]

    if len(x) == 0:
        return np.full(len(x_new), np.nan, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]

    out = np.interp(x_new, x, y, left=np.nan, right=np.nan)

    # np.interp 只会在区间外给 NaN，区间内默认全部插值；这里额外限制 gap。
    if max_gap_min is not None and len(x) >= 2:
        pos = np.searchsorted(x, x_new, side="left")

        inside = (pos > 0) & (pos < len(x))
        left_idx = np.clip(pos - 1, 0, len(x) - 1)
        right_idx = np.clip(pos, 0, len(x) - 1)
        gap_sec = x[right_idx] - x[left_idx]
        out[inside & (gap_sec > max_gap_min * 60.0)] = np.nan

    return out


# ======================================================
# 7. 构造事件统一时间轴和瞬时物理量
# ======================================================
def build_common_timeseries(data, event, cadence_min=1.0):
    """把 B/N/V/T 插值到该事件内部的统一时间网格，并先计算 Pth/Beta。"""
    start = event["shock"] - timedelta(minutes=1)
    end = event["icme_end"] + timedelta(minutes=1)

    grid = pd.date_range(
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        freq=pd.to_timedelta(cadence_min, unit="min"),
    ).tz_localize(None)

    ts = pd.DataFrame(index=grid)

    for p in ["B_total", "N", "V_total", "T_K"]:
        if p not in data:
            ts[p] = np.nan
            continue
        src_t, src_v = data[p]
        ts[p] = interpolate_to_grid(
            src_t,
            src_v,
            grid,
            max_gap_min=MAX_INTERP_GAP_MIN,
        )

    # 物理有效性基础筛选
    ts.loc[ts["B_total"] < 0, "B_total"] = np.nan
    ts.loc[ts["N"] <= 0, "N"] = np.nan
    ts.loc[ts["T_K"] <= 0, "T_K"] = np.nan
    ts.loc[ts["V_total"] < 0, "V_total"] = np.nan

    # -------- 先逐时刻计算派生量 --------
    n_m3 = ts["N"] * CM3_TO_M3
    ts["Pth_nPa"] = n_m3 * K_B * ts["T_K"] / NPA_TO_PA

    B_T = ts["B_total"] * 1.0e-9
    PB_nPa = (B_T**2 / (2.0 * MU0)) / NPA_TO_PA
    ts["Beta"] = ts["Pth_nPa"] / PB_nPa

    # B=0 等情况自然变为 inf，再统一剔除
    ts.replace([np.inf, -np.inf], np.nan, inplace=True)

    return ts


# ======================================================
# 8. 归一化 phase 分箱
# ======================================================
def phase_boundaries(event):
    return {
        "phase1": (event["shock"], event["mark1"]),
        "phase2": (event["mark1"], event["mark2"]),
        "phase3": (event["icme_start_eff"], event["icme_end"]),
    }


def normalized_phase_statistics(ts, event):
    results = {}
    boundaries = phase_boundaries(event)
    ts_index_sec = ts.index.view("int64") / 1.0e9

    for phase, (start, end) in boundaries.items():
        nbin = N_BINS[phase]
        if start >= end:
            results[phase] = {
                p: {
                    "mean": np.full(nbin, np.nan),
                    "median": np.full(nbin, np.nan),
                    "count": np.zeros(nbin, dtype=int),
                    "raw": [[] for _ in range(nbin)],
                }
                for p in PHYS_PARAMS
            }
            continue

        start_sec = datetime_to_seconds(start)
        end_sec = datetime_to_seconds(end)
        valid_time = (ts_index_sec >= start_sec) & (ts_index_sec <= end_sec)
        rel = (ts_index_sec[valid_time] - start_sec) / (end_sec - start_sec)
        rel = np.clip(rel, 0.0, 1.0)
        phase_df = ts.loc[valid_time]
        results[phase] = {}
        bin_edges = np.linspace(0.0, 1.0, nbin + 1)

        for p in PHYS_PARAMS:
            values = phase_df[p].to_numpy(dtype=float)
            means = np.full(nbin, np.nan, dtype=float)
            medians = np.full(nbin, np.nan, dtype=float)
            counts = np.zeros(nbin, dtype=int)
            raw_list = [[] for _ in range(nbin)]

            for i in range(nbin):
                if i == nbin - 1:
                    mask = (rel >= bin_edges[i]) & (rel <= bin_edges[i + 1])
                else:
                    mask = (rel >= bin_edges[i]) & (rel < bin_edges[i + 1])
                vals = values[mask]
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    means[i] = np.mean(vals)
                    medians[i] = np.median(vals)
                    counts[i] = len(vals)
                    raw_list[i] = vals.tolist()

            results[phase][p] = {
                "mean": means,
                "median": medians,
                "count": counts,
                "raw": raw_list,
            }

    return results


# ======================================================
# 9. 所有事件的 superposed epoch 统计
# ======================================================
def append_event_to_collection(collection, event_stats):
    for phase in ["phase1", "phase2", "phase3"]:
        for p in PHYS_PARAMS:
            collection[phase][p]["mean"].append(event_stats[phase][p]["mean"])
            collection[phase][p]["median"].append(event_stats[phase][p]["median"])
            collection[phase][p]["count"].append(event_stats[phase][p]["count"])
            collection[phase][p]["raw"].append(event_stats[phase][p]["raw"])


def composite_statistics(collection):
    superposed = {
        phase: {
            p: {
                "mean_avg": [],
                "mean_median": [],
                "std": [],
                "sem": [],
                "n_events": [],
                "event_median_avg": [],
                "event_median_median": [],
                "global_median": [],          # 新增：所有数据的中位数
            }
            for p in PHYS_PARAMS
        }
        for phase in ["phase1", "phase2", "phase3"]
    }

    for phase in ["phase1", "phase2", "phase3"]:
        nbin = N_BINS[phase]

        for p in PHYS_PARAMS:
            event_means = np.asarray(collection[phase][p]["mean"], dtype=float)
            event_medians = np.asarray(collection[phase][p]["median"], dtype=float)
            event_raws = collection[phase][p]["raw"]   # list of lists: each event has list of bin raw values

            if event_means.size == 0:
                event_means = np.empty((0, nbin))
            if event_medians.size == 0:
                event_medians = np.empty((0, nbin))

            for i in range(nbin):
                # 平均值统计（事件均值的平均）
                vals = event_means[:, i] if len(event_means) else np.array([])
                vals = vals[np.isfinite(vals)]
                n_evt = len(vals)
                if n_evt == 0:
                    mean_avg = np.nan
                    mean_median = np.nan
                    std = np.nan
                    sem = np.nan
                else:
                    mean_avg = np.mean(vals)
                    mean_median = np.median(vals)
                    std = np.std(vals, ddof=1) if n_evt >= 2 else 0.0
                    sem = std / np.sqrt(n_evt) if n_evt >= 2 else 0.0

                # 中位数统计（事件中位数的平均/中位）
                med_vals = event_medians[:, i] if len(event_medians) else np.array([])
                med_vals = med_vals[np.isfinite(med_vals)]
                if len(med_vals) == 0:
                    event_median_avg = np.nan
                    event_median_median = np.nan
                else:
                    event_median_avg = np.mean(med_vals)
                    event_median_median = np.median(med_vals)

                # 全局中位数：合并所有事件在该 bin 的原始值
                all_raw = []
                for evt_raws in event_raws:
                    if i < len(evt_raws):
                        all_raw.extend(evt_raws[i])
                all_raw = np.array(all_raw)
                all_raw = all_raw[np.isfinite(all_raw)]
                if len(all_raw) == 0:
                    global_median = np.nan
                else:
                    global_median = np.median(all_raw)

                superposed[phase][p]["mean_avg"].append(mean_avg)
                superposed[phase][p]["mean_median"].append(mean_median)
                superposed[phase][p]["std"].append(std)
                superposed[phase][p]["sem"].append(sem)
                superposed[phase][p]["n_events"].append(n_evt)
                superposed[phase][p]["event_median_avg"].append(event_median_avg)
                superposed[phase][p]["event_median_median"].append(event_median_median)
                superposed[phase][p]["global_median"].append(global_median)   # 新增

    return superposed


# ======================================================
# 10. 输出表格
# ======================================================
def build_output_dataframe(superposed):
    rows = []
    phase_offset = 0
    for phase in ["phase1", "phase2", "phase3"]:
        nbin = N_BINS[phase]
        x_norm = (np.arange(nbin) + 0.5) / nbin
        for i in range(nbin):
            row = {
                "phase": phase,
                "bin": i + 1,
                "x_norm": x_norm[i],
                "x_combined": phase_offset + i + 0.5,
            }
            for p in PHYS_PARAMS:
                row[f"{p}_mean"] = superposed[phase][p]["mean_avg"][i]
                row[f"{p}_median_of_event_means"] = superposed[phase][p]["mean_median"][i]
                row[f"{p}_std_event"] = superposed[phase][p]["std"][i]
                row[f"{p}_sem_event"] = superposed[phase][p]["sem"][i]
                row[f"{p}_n_events"] = superposed[phase][p]["n_events"][i]
                row[f"{p}_mean_of_event_medians"] = superposed[phase][p]["event_median_avg"][i]
                row[f"{p}_median_of_event_medians"] = superposed[phase][p]["event_median_median"][i]
                row[f"{p}_global_median"] = superposed[phase][p]["global_median"][i]   # 新增
            rows.append(row)
        phase_offset += nbin + 5
    return pd.DataFrame(rows)


# ======================================================
# 11. 绘图
# ======================================================
def plot_superposed(superposed, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    n_panels = len(PHYS_PARAMS)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(13, 3.3 * n_panels),
        sharex=True,
        constrained_layout=False,
    )
    plt.rcParams.update({'font.size': 12})
    if n_panels == 1:
        axes = [axes]

    n1 = N_BINS["phase1"]
    n2 = N_BINS["phase2"]
    n3 = N_BINS["phase3"]

    offsets = {"phase1": -n1, "phase2": 0, "phase3": n2}

    labels = {
        "B_total": r"$B$ [nT]",
        "N": r"$N_p$ [cm$^{-3}$]",
        "V_total": r"$V_p$ [km/s]",
        "T_K": r"$T_p$ [K]",
        "Pth_nPa": r"$P_{th}$ [nPa]",
        "Beta": r"$\beta$",
    }

    for ax, p in zip(axes, PHYS_PARAMS):
        for phase in ["phase1", "phase2", "phase3"]:
            nbin = N_BINS[phase]
            x = np.arange(nbin) + offsets[phase]

            y_mean = np.asarray(superposed[phase][p]["mean_avg"], dtype=float)
            y_median = np.asarray(superposed[phase][p]["global_median"], dtype=float)  # 使用全局中位数

            ax.plot(
                x, y_mean,
                color='blue', linestyle='-', linewidth=1.5,
                label='Mean' if phase == "phase1" else ""
            )
            ax.plot(
                x, y_median,
                color='red', linestyle='--', linewidth=1.5,
                label='Median' if phase == "phase1" else ""
            )

        ax.axvline(-0.5, linestyle='--', linewidth=1.0, color='gray')
        ax.axvline(n2 - 0.5, linestyle='--', linewidth=1.0, color='gray')

        ax.set_ylabel(labels.get(p, p),fontsize=18)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best',fontsize=16)
        ax.tick_params(labelsize=15)
    #axes[-1].set_xlabel("Bin index (relative to phase2 start)")
    # fig.suptitle(
    #     "ICME Superposed Epoch Analysis\nMean and global median profiles",
    #     fontsize=14,
    # )
    fig.subplots_adjust(top=0.95, hspace=0.10, bottom=0.06, left=0.10, right=0.98)

    png_path = os.path.join(output_dir, "superposed_epoch.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return png_path


# ======================================================
# 12. 主程序
# ======================================================
def main():
    # --------------------------------------------------
    # 配置：按你的实际目录修改
    # --------------------------------------------------
    config = {
        "PSP": {
            "mag_root": r"I:\stdata\psp\FLD\PSP_FLD_L2_MAG_RTN_1MIN",
            "mag_prefix": "psp_fld_l2_mag_rtn_1min",
            "plas_root": r"I:\stdata\psp\sweap\psp_swp_spc_l3i",
            "plas_prefix": "psp_swp_spc_l3i",
        },
        "SOLO": {
            "mag_root": r"I:\stdata\solo\mag\solo_l2_mag-rtn-normal-1-minute",
            "mag_prefix": "solo_l2_mag-rtn-normal-1-minute",
            "plas_root": r"I:\stdata\solo\swa\solo_l2_swa-pas-grnd-mom",
            "plas_prefix": "solo_l2_swa-pas-grnd-mom",
        },
    }

    psp_event_file = r"E:\ZhaoWenjin_Reserch\pileup\有激波psp.txt"
    solo_event_file = r"E:\ZhaoWenjin_Reserch\pileup\solo列表.txt"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("ICME Superposed Epoch Analysis")
    print("=" * 70)

    # --------------------------------------------------
    # 读取事件
    # --------------------------------------------------
    psp_events = parse_event_file(psp_event_file, "PSP")
    solo_events = parse_event_file(solo_event_file, "SOLO")
    all_events = psp_events + solo_events

    print(f"PSP 有效事件数 : {len(psp_events)}")
    print(f"SOLO 有效事件数: {len(solo_events)}")
    print(f"总事件数       : {len(all_events)}")

    if len(all_events) == 0:
        print("❌ 没有找到符合条件的事件，程序结束。")
        return

    # --------------------------------------------------
    # 存储所有事件的 phase statistics
    # --------------------------------------------------
    collection = {
        phase: {
            p: {"mean": [], "median": [], "count": [], "raw": []}
            for p in PHYS_PARAMS
        }
        for phase in ["phase1", "phase2", "phase3"]
    }

    success_events = []
    failed_events = []

    # --------------------------------------------------
    # 逐事件处理
    # --------------------------------------------------
    for idx, evt in enumerate(all_events, start=1):
        print("\n" + "-" * 70)
        print(
            f"处理事件 {idx}/{len(all_events)} | "
            f"{evt['satellite']} | shock={evt['shock']} | ratio={evt['ratio']:.3f}"
        )

        try:
            data = read_event_data(evt, config)
            if data is None:
                raise RuntimeError("数据读取失败")

            ts = build_common_timeseries(
                data,
                evt,
                cadence_min=COMMON_CADENCE_MIN,
            )

            if ts.empty or ts["B_total"].notna().sum() == 0:
                raise RuntimeError("统一时间网格没有有效磁场数据")

            event_stats = normalized_phase_statistics(ts, evt)
            append_event_to_collection(collection, event_stats)

            success_events.append(evt)
            print("  ✓ 事件处理成功")

            # 打印三个 phase 的大致有效性
            for phase in ["phase1", "phase2", "phase3"]:
                n_b = np.sum(np.isfinite(event_stats[phase]["B_total"]["mean"]))
                n_n = np.sum(np.isfinite(event_stats[phase]["N"]["mean"]))
                n_t = np.sum(np.isfinite(event_stats[phase]["T_K"]["mean"]))
                print(
                    f"    {phase}: B={n_b}/{N_BINS[phase]}, "
                    f"N={n_n}/{N_BINS[phase]}, T={n_t}/{N_BINS[phase]}"
                )

        except Exception as exc:
            failed_events.append((evt, str(exc)))
            print(f"  ❌ 事件处理失败: {exc}")
            continue

    # --------------------------------------------------
    # 成功事件统计
    # --------------------------------------------------
    success_count = len(success_events)
    print("\n" + "=" * 70)
    print(f"成功处理事件数: {success_count}")
    print(f"失败事件数    : {len(failed_events)}")
    print("=" * 70)

    if success_count == 0:
        print("❌ 没有任何事件成功处理！请检查数据路径、变量名和事件表格式。")
        return

    # --------------------------------------------------
    # 事件间 superposed statistics
    # --------------------------------------------------
    superposed = composite_statistics(collection)

    # --------------------------------------------------
    # 输出 CSV
    # --------------------------------------------------
    output_df = build_output_dataframe(superposed)
    csv_path = os.path.join(OUTPUT_DIR, "superposed_epoch_statistics.csv")
    output_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 保存成功事件清单
    event_rows = []
    for i, evt in enumerate(success_events, start=1):
        row = {
            "event_id": i,
            "satellite": evt["satellite"],
            "shock": evt["shock"],
            "icme_start": evt["icme_start"],
            "icme_start_eff": evt["icme_start_eff"],
            "mark1": evt["mark1"],
            "mark2": evt["mark2"],
            "icme_end": evt["icme_end"],
            "ratio": evt["ratio"],
        }
        event_rows.append(row)

    event_df = pd.DataFrame(event_rows)
    event_csv = os.path.join(OUTPUT_DIR, "successful_events.csv")
    event_df.to_csv(event_csv, index=False, encoding="utf-8-sig")

    # 保存失败事件清单
    if failed_events:
        failed_rows = []
        for evt, reason in failed_events:
            failed_rows.append(
                {
                    "satellite": evt["satellite"],
                    "shock": evt["shock"],
                    "reason": reason,
                }
            )
        pd.DataFrame(failed_rows).to_csv(
            os.path.join(OUTPUT_DIR, "failed_events.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------
    # 绘图
    # --------------------------------------------------
    png_path = plot_superposed(superposed, OUTPUT_DIR)

    print("\n输出文件：")
    print(f"  {csv_path}")
    print(f"  {event_csv}")
    print(f"  {png_path}")
    if failed_events:
        print(f"  {os.path.join(OUTPUT_DIR, 'failed_events.csv')}")


if __name__ == "__main__":
    main()
