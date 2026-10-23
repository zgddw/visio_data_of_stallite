# -*- coding: utf-8 -*-
"""
PSP 多参数 ICME 密度堆积区自动检索
========================================

检索规则：
1. 有激波：检索范围 = 激波时刻 + 鞘层时长 D 的可配置比例（默认 2%） -> ICME 本体前 20% 的终点；
   无激波：检索范围 = ICME 本体开始前“本体时长的 20%” -> 本体前 20% 的终点。
   候选区不得延伸到 ICME 本体中段或尾部；前后 2 小时仅用于判据参照数据读取。
2. 候选窗口从整个检索范围的最大长度开始，以 1 分钟逐步缩短至 5 分钟；
   每个窗口长度均以 1 分钟步长扫描。
3. 每个候选区与其前、后连续 2 小时分别比较：
   Np >= 两侧均值各自的 2 倍；Beta >= 两侧均值各自的 2 倍；
   |B_T| <= 两侧均值各自的 60%；Beta 计算仍使用总磁场强度。
4. 分类：三项全满足=强典型；Np 与 Beta/|B_T| 之一同时满足=典型；仅 Np 满足=弱典型；
   Np 不满足但 Beta/|B_T| 至少一项满足=非典型；三项全不满足=非密度堆积区。
   非典型事件仍计为检出事件；三项全不满足的 ICME 单独输出到未检出列表。
5. 首先扫描全部候选窗口，并将重叠候选归为同一结构。
6. 每个 ICME 默认最终最多保留一个密度堆积区：候选之间间隔不超过可配置阈值时，尝试合并为较长区间，且合并后的整体必须重新通过至少一项物理判据。
7. 若候选区形成多个间隔超过阈值的候选簇，只保留距离 ICME 本体前端最近的候选簇；若合并后不再满足判据，则保留最近簇中距离本体前端最近的原始有效区间。
8. 同一距离下优先持续时间较长、等级较高的候选；保留局部核心信息供复核。

依赖：numpy、pandas，以及你已有的 pro_of_read.py / cdfdataread()。
Beta 按质子热压计算且使用总磁场模长：beta_p = 2*mu0*n_p*kB*T_p/|B|^2；第三项检索判据改用 |B_T|。
注意：这表示质子 beta；若你之前 compcalculate.Pandbeta() 使用的定义不同，需统一定义后再做论文统计。
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 让脚本能找到你之前使用的 pro_of_read.py；若该文件在其他目录，请修改此路径。
CUSTOM_MODULE_DIRS = [
    Path(__file__).resolve().parent,
    Path(r"E:\ZhaoWenjin_Reserch\pileup"),
    Path(r"E:\ZhaoWenjin_Reserch\anaysis_prog"),
]
for _p in CUSTOM_MODULE_DIRS:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from pro_of_read import cdfdataread
except Exception as _import_error:
    cdfdataread = None
    _CDF_IMPORT_ERROR = _import_error
else:
    _CDF_IMPORT_ERROR = None


# ============================================================================
# 用户配置：目录沿用之前 seaana.py / SOLO 绘图程序中出现的配置
# ============================================================================
SATELLITE = "PSP"
EVENT_LIST_FILES = []  # 以 I: 的 HELIO4CAST 综合目录为准，不再将同一旧文件重复当作有/无激波列表
# I: HELIO4CAST 的前三个时间列依次是激波、本体开始、本体结束；前两列相同表示无激波。
FALLBACK_EVENT_FILES = [
    Path(r"I:\list\icme\HELIO4CAST_ICMECAT_v23\divided\ICME_PSP.txt"),
    # 随脚本包附带的 HELIO4CAST 事件表也可直接作为后备输入。
    Path(__file__).resolve().parent / "ICME_PSP.txt",
]

MAG_ROOT = Path(r"I:\stdata\psp\FLD\PSP_FLD_L2_MAG_RTN_1MIN")
MAG_PREFIX = "psp_fld_l2_mag_rtn_1min"
PLASMA_ROOT = Path(r"I:\stdata\psp\sweap\psp_swp_spc_l3i")
PLASMA_PREFIX = "psp_swp_spc_l3i"
MAG_VARIABLE = "psp_fld_l2_mag_RTN_1min"
DENSITY_VARIABLE = "np_moment"
TEMPERATURE_VARIABLE = "wp_moment"

OUTPUT_DIR = Path(__file__).resolve().parent / f"结果_PSP_多参数密度堆积区"
MIN_WINDOW_MINUTES = 5
REFERENCE_MINUTES = 120
STEP_MINUTES = 1
MIN_COVERAGE = 0.90            # 候选区和两侧参照区，各参数均需达到 90% 有效覆盖率
MAX_GAP_BETWEEN_REGIONS_MINUTES = 120  # 同一 ICME 内候选区最大合并间隔；可改为 60、90、180 等
MAX_BODY_FRACTION = 0.20      # 检索终点：最多进入 ICME 本体前 20%
SHOCK_MARGIN_FRACTION = 0.1  # 有激波起点：激波时刻 + 该比例 × D；D = 本体开始 - 激波时刻。改这里即可调整 2%
NO_SHOCK_PRE_BODY_FRACTION = 0.20  # 无激波时：从本体开始前“本体时长的 20%”起搜索
WRITE_XLSX = True

# 常数：beta_p = 2*mu0*n_p*kB*T_p / B^2
MU0 = 4.0e-7 * np.pi           # N/A^2
K_B = 1.380649e-23             # J/K
M_P = 1.67262192369e-27         # kg
EV_TO_K = 11604.51812
CM3_TO_M3 = 1.0e6
NT_TO_T = 1.0e-9
MIN_NS = 60 * 1_000_000_000

DATE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:T|\s+)\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
)
NpY = 2.0
betaY = 2.0
BtY = 0.6
GRADE_RANK = {
    "强典型密度堆积区": 5,
    "典型密度堆积区": 4,
    "弱典型密度堆积区": 3,
    "非典型密度堆积区": 2,
    "非密度堆积区": 0,
    "非密度堆积区事件": 0,
    "数据不足/无法判定": -1,
    "读取失败/无法判定": -1,
}


def normalize_datetime(value: Any) -> pd.Timestamp:
    """解析输入时间；所有时间统一转为无时区 UTC 时间，适配旧版 cdfdataread。"""
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return pd.Timestamp(ts).tz_convert(None)


def parse_event_text_file(path: Path, file_hint: Optional[str]) -> List[dict]:
    """读取旧版多时间列 TXT，以及 ICME_PSP.txt 这种 HELIO4CAST 文本目录。

    兼容行内前 3 个时间依次为 shock/start/end 的列表；无激波列表若仍有
    shock 占位列（时间可能与 start 相同），会忽略 shock 并使用 start/end。
    只有 2 个时间的行按无激波 start/end 处理。
    """
    events: List[dict] = []
    if not path.exists():
        return events
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except Exception as exc:
        print(f"[警告] 无法读取事件列表 {path}: {exc}")
        return events

    hint = file_hint
    if hint is None:
        name = path.stem.lower()
        if "无激波" in name or "no_shock" in name or "noshock" in name:
            hint = "no_shock"
        elif "有激波" in name or "with_shock" in name or "shock" in name:
            hint = "with_shock"

    for line_no, line in enumerate(lines, 1):
        raw = line.strip()
        if not raw or raw.startswith("#") or raw.startswith("//"):
            continue
        matches = list(DATE_RE.finditer(raw))
        if len(matches) < 2:
            continue
        timestamps = [normalize_datetime(m.group(0)) for m in matches[:3]]
        timestamps = [t for t in timestamps if not pd.isna(t)]
        if len(timestamps) < 2:
            continue

        prefix = raw[:matches[0].start()].strip()
        id_tokens = prefix.split()
        event_id = id_tokens[0] if id_tokens else f"{path.stem}_{line_no:04d}"

        if len(timestamps) >= 3:
            t0, t1, t2 = timestamps[:3]
            if hint == "no_shock":
                shock, body_start, body_end, has_shock = pd.NaT, t1, t2, False
            else:
                body_start, body_end = t1, t2
                # 分开的“有激波”列表以文件分类为准；综合目录表则以 shock < start 判断。
                if hint == "with_shock":
                    has_shock, shock = True, t0
                else:
                    has_shock = bool(t0 < t1)
                    shock = t0 if has_shock else pd.NaT
        else:
            shock, body_start, body_end, has_shock = pd.NaT, timestamps[0], timestamps[1], False

        if pd.isna(body_start) or pd.isna(body_end) or body_end <= body_start:
            print(f"[警告] 跳过时间边界无效的行：{path.name}:{line_no}")
            continue
        if hint == "with_shock" and not pd.isna(shock):
            has_shock = True
        if hint == "no_shock":
            has_shock, shock = False, pd.NaT

        # 为避免同一事件出现在两个列表时重复扫描，稍后会按时间边界去重。
        events.append({
            "event_id": event_id,
            "satellite": SATELLITE,
            "shock_time": shock,
            "body_start": body_start,
            "body_end": body_end,
            "has_shock": bool(has_shock),
            "source_file": str(path),
            "source_line": line_no,
        })
    return events


def load_event_list_files() -> List[dict]:
    loaded: List[dict] = []
    available_configured = [(p, hint) for p, hint in EVENT_LIST_FILES if p.exists()]
    missing_hints = {hint for p, hint in EVENT_LIST_FILES if not p.exists()}
    fallback = next((p for p in FALLBACK_EVENT_FILES if p.exists()), None)

    for missing_path in [str(p) for p, _ in EVENT_LIST_FILES if not p.exists()]:
        print(f"[提示] 未找到事件列表：{missing_path}")

    for path, hint in available_configured:
        rows = parse_event_text_file(path, hint)
        print(f"[信息] {path.name}: 读取到 {len(rows)} 个事件")
        loaded.extend(rows)

    if not available_configured:
        if fallback is None:
            candidates = "\n".join(str(p) for p, _ in EVENT_LIST_FILES + [(p, None) for p in FALLBACK_EVENT_FILES])
            raise FileNotFoundError(
                "未找到事件列表。请检查脚本顶部 EVENT_LIST_FILES / FALLBACK_EVENT_FILES：\n" + candidates
            )
        rows = parse_event_text_file(fallback, None)
        print(f"[信息] 使用综合事件表 {fallback.name}: 读取到 {len(rows)} 个事件")
        loaded.extend(rows)
    elif missing_hints and fallback is not None:
        # 若用户只提供了有/无激波两个列表中的一个，用综合表补充缺失类别，避免静默漏检。
        fallback_rows = parse_event_text_file(fallback, None)
        supplement = [
            ev for ev in fallback_rows
            if ("with_shock" if ev["has_shock"] else "no_shock") in missing_hints
        ]
        print(f"[信息] 使用 {fallback.name} 补充缺失类别 {sorted(missing_hints)}：{len(supplement)} 个事件")
        loaded.extend(supplement)

    unique: Dict[Tuple[pd.Timestamp, pd.Timestamp], dict] = {}
    for ev in loaded:
        key = (ev["body_start"], ev["body_end"])
        if key not in unique:
            unique[key] = ev
        else:
            # 若两个列表存在相同本体边界，优先保留显式记录了激波的那条。
            if ev["has_shock"] and not unique[key]["has_shock"]:
                unique[key] = ev
    events = sorted(unique.values(), key=lambda e: e["body_start"])
    for i, ev in enumerate(events, 1):
        if not ev["event_id"] or ev["event_id"].startswith("_ "):
            ev["event_id"] = f"{SATELLITE}_{ev['body_start']:%Y%m%d_%H%M}_{i:03d}"
    return events


def _find_column(df: pd.DataFrame, candidates: List[str], suffix: Optional[str] = None) -> str:
    if df is None or df.empty:
        raise ValueError("CDF 读取结果为空")
    by_lower = {str(c).lower(): str(c) for c in df.columns}
    for name in candidates:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    if suffix:
        suffix_l = suffix.lower()
        for col in df.columns:
            if str(col).lower().endswith(suffix_l):
                return str(col)
    non_time = [str(c) for c in df.columns if str(c).lower() != "t"]
    if len(non_time) == 1:
        return non_time[0]
    raise KeyError(f"找不到变量列。候选={candidates}，实际列={list(df.columns)}")


def _get_df(data: dict, key: str) -> pd.DataFrame:
    frame = data.get(key) if isinstance(data, dict) else None
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"cdfdataread 未返回有效的 {key!r} 数据；可检查数据目录和文件前缀")
    if "t" not in frame.columns:
        raise KeyError(f"{key!r} 数据中没有时间列 't'，实际列为 {list(frame.columns)}")
    return frame


def _frame_times(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(frame["t"], utc=True, errors="coerce")).tz_convert(None)


def _resample_scalar(frame: pd.DataFrame, candidates: List[str], grid: pd.DatetimeIndex,
                     transform=None, label: str = "变量") -> pd.Series:
    col = _find_column(frame, candidates)
    idx = _frame_times(frame)
    values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    if transform is not None:
        values = transform(values)
    good = (~idx.isna()) & np.isfinite(values)
    raw = pd.Series(values[good], index=idx[good], name=label)
    if raw.empty:
        return pd.Series(np.nan, index=grid, name=label, dtype=float)
    raw = raw.groupby(level=0).mean().sort_index()
    # 只做一分钟聚合，不对缺测区间插值，以免制造虚假的独立采样点。
    return raw.resample("1min").mean().reindex(grid)


def _resample_vector_magnitude(frame: pd.DataFrame, candidates_x: List[str],
                               candidates_y: List[str], candidates_z: List[str],
                               grid: pd.DatetimeIndex) -> pd.Series:
    cx = _find_column(frame, candidates_x, suffix="_X")
    cy = _find_column(frame, candidates_y, suffix="_Y")
    cz = _find_column(frame, candidates_z, suffix="_Z")
    idx = _frame_times(frame)
    comp = np.column_stack([
        pd.to_numeric(frame[cx], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(frame[cy], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(frame[cz], errors="coerce").to_numpy(dtype=float),
    ])
    magnitude = np.sqrt(np.sum(comp ** 2, axis=1))
    good = (~idx.isna()) & np.all(np.isfinite(comp), axis=1) & np.isfinite(magnitude)
    raw = pd.Series(magnitude[good], index=idx[good], name="B_total_nT")
    if raw.empty:
        return pd.Series(np.nan, index=grid, name="B_total_nT", dtype=float)
    raw = raw.groupby(level=0).mean().sort_index()
    return raw.resample("1min").mean().reindex(grid)


def _resample_vector_component(frame: pd.DataFrame, candidates: List[str],
                                grid: pd.DatetimeIndex, label: str = "B_T_RTN_nT",
                                absolute: bool = False) -> pd.Series:
    """读取 RTN-T 分量并聚合到一分钟；absolute=True 时先取绝对值再聚合。"""
    col = _find_column(frame, candidates, suffix="_Y")
    idx = _frame_times(frame)
    values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    if absolute:
        values = np.abs(values)
    good = (~idx.isna()) & np.isfinite(values)
    raw = pd.Series(values[good], index=idx[good], name=label)
    if raw.empty:
        return pd.Series(np.nan, index=grid, name=label, dtype=float)
    raw = raw.groupby(level=0).mean().sort_index()
    return raw.resample("1min").mean().reindex(grid)


def read_event_minute_series(event: dict, search_start: pd.Timestamp,
                             search_end: pd.Timestamp) -> pd.DataFrame:
    """调用旧项目的 cdfdataread，返回统一到 1 分钟时间轴的 Np、Beta、总磁场模长及 RTN-T 分量。"""
    if cdfdataread is None:
        raise ImportError(
            "导入 pro_of_read.cdfdataread 失败。请把 pro_of_read.py 放到脚本同目录，"
            "或修改 CUSTOM_MODULE_DIRS。原始导入错误：" + repr(_CDF_IMPORT_ERROR)
        )
    fetch_start = (search_start - pd.Timedelta(minutes=REFERENCE_MINUTES)).floor("min").to_pydatetime()
    fetch_end = (search_end + pd.Timedelta(minutes=REFERENCE_MINUTES)).ceil("min").to_pydatetime()
    grid = pd.date_range(fetch_start, fetch_end, freq="1min")

    mag_dict = cdfdataread(str(MAG_ROOT), [MAG_VARIABLE], fetch_start, fetch_end, MAG_PREFIX)
    plasma_dict = cdfdataread(
        str(PLASMA_ROOT), [DENSITY_VARIABLE, TEMPERATURE_VARIABLE],
        fetch_start, fetch_end, PLASMA_PREFIX
    )
    mag_df = _get_df(mag_dict, MAG_VARIABLE)
    n_df = _get_df(plasma_dict, DENSITY_VARIABLE)
    t_df = _get_df(plasma_dict, TEMPERATURE_VARIABLE)

    if SATELLITE == "PSP":
        bx_candidates = [f"psp_fld_l2_mag_RTN_1min_X", "Br", "BX", "B_RTN_X"]
        by_candidates = [f"psp_fld_l2_mag_RTN_1min_Y", "B_RTN_T", "Bt", "B_T", "B_RTN_Y", "BY"]
        bz_candidates = [f"psp_fld_l2_mag_RTN_1min_Z", "Bn", "BZ", "B_RTN_Z"]
        n_candidates = ["Np", "np_moment", "N", "proton_density"]
        t_candidates = ["wp_moment", "wp", "Wp"]
        # 沿用之前 seaana.py 中 wp_moment 的热速度换算：wp [km/s] -> T_p [K]
        temp_transform = lambda wp_kms: M_P * (wp_kms * 1000.0) ** 2 / (2.0 * K_B)
    else:
        bx_candidates = ["B_RTN_X", "B_RTN_R", f"psp_fld_l2_mag_RTN_1min_X", "Br"]
        by_candidates = ["B_RTN_T", "Bt", "B_T", "B_RTN_Y", f"psp_fld_l2_mag_RTN_1min_Y", "BY"]
        bz_candidates = ["B_RTN_Z", "B_RTN_N", f"psp_fld_l2_mag_RTN_1min_Z", "Bn"]
        n_candidates = ["Np", "N", "proton_density"]
        t_candidates = ["Tp", "T", "T_proton"]
        # 沿用之前 seaana.py：SOLO 的 T 变量按 eV 转为 K。
        temp_transform = lambda temp_ev: temp_ev * EV_TO_K

    # 总磁场模长仍用于计算 beta；磁场判据单独使用 RTN-T 分量的绝对值。
    b_total_nT = _resample_vector_magnitude(mag_df, bx_candidates, by_candidates, bz_candidates, grid)
    b_t_signed_nT = _resample_vector_component(
        mag_df, by_candidates, grid, label="B_T_RTN_nT", absolute=False
    )
    # 对每个原始采样先取 |B_T| 再做分钟聚合，避免分钟内正负值抵消。
    b_t_abs_nT = _resample_vector_component(
        mag_df, by_candidates, grid, label="B_T_abs_nT", absolute=True
    )
    n_cm3 = _resample_scalar(n_df, n_candidates, grid, label="Np_cm3")
    t_K = _resample_scalar(t_df, t_candidates, grid, transform=temp_transform, label="Tp_K")

    # 非正密度/温度/总磁场按无效值处理；B_T 分量允许为零或负值，判据使用 |B_T|。
    n_cm3 = n_cm3.where(n_cm3 > 0)
    t_K = t_K.where(t_K > 0)
    b_total_nT = b_total_nT.where(b_total_nT > 0)
    b_total_T = b_total_nT * NT_TO_T
    beta_p = 2.0 * MU0 * (n_cm3 * CM3_TO_M3) * K_B * t_K / (b_total_T ** 2)
    beta_p = beta_p.where(np.isfinite(beta_p) & (beta_p > 0))

    result = pd.DataFrame({
        "Np_cm3": n_cm3,
        "Beta_p": beta_p,
        "B_total_nT": b_total_nT,
        "B_T_RTN_nT": b_t_signed_nT,
        "B_T_abs_nT": b_t_abs_nT,
    }, index=grid)
    result.index.name = "Time_UTC"
    return result


def _prefix(values: np.ndarray, positive_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    if positive_only:
        valid &= x > 0
    sums = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    counts = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    return sums, counts


def _interval_stats(prefix: Tuple[np.ndarray, np.ndarray], starts: np.ndarray,
                    ends: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    sums, counts = prefix
    c = counts[ends] - counts[starts]
    s = sums[ends] - sums[starts]
    means = np.divide(s, c, out=np.full(len(starts), np.nan), where=c > 0)
    coverage = c / float(width)
    return means, coverage


def _classify(cond_n: bool, cond_beta: bool, cond_b: bool) -> str:
    """按质子密度优先级，将三个判据映射为五类事件等级。"""
    if cond_n:
        if cond_beta and cond_b:
            return "强典型密度堆积区"
        if cond_beta or cond_b:
            return "典型密度堆积区"
        return "弱典型密度堆积区"
    if cond_beta or cond_b:
        # Np 未满足，但 Beta 和/或 |B_T| 满足，仍是检出的非典型事件。
        return "非典型密度堆积区"
    return "非密度堆积区"


def _distance_and_overlap(start_ns: np.ndarray, end_ns: np.ndarray,
                          body_start: pd.Timestamp, body_end: pd.Timestamp):
    bs = body_start.value
    be = body_end.value
    distance_ns = np.where(end_ns < bs, bs - end_ns,
                           np.where(start_ns > be, start_ns - be, 0))
    overlap_ns = np.maximum(0, np.minimum(end_ns, be) - np.maximum(start_ns, bs))
    return distance_ns / MIN_NS, overlap_ns / MIN_NS


def _distance_to_icme_front_min(start_ns: np.ndarray, end_ns: np.ndarray,
                                body_start: pd.Timestamp) -> np.ndarray:
    """候选区间到 ICME 本体前端（body_start）的时间距离，单位分钟。

    候选区间在前端之前：测 ICME 前端与候选区后端的距离；
    区间跨越前端：距离为 0；候选区在前端之后：测候选区前端与本体前端的距离。
    """
    bs = body_start.value
    distance_ns = np.where(end_ns <= bs, bs - end_ns,
                           np.where(start_ns >= bs, start_ns - bs, 0))
    return distance_ns / MIN_NS


def _candidate_core_key(candidate: dict) -> tuple:
    # 局部核心优先级：等级高 -> 持续时间长 -> 距 ICME 近 -> Np > beta > B。
    flags = candidate["conditions"]
    return (
        -GRADE_RANK.get(candidate["grade"], -1),
        -candidate["duration_min"],
        candidate["distance_to_icme_min"],
        -int(flags["Np"]), -int(flags["Beta"]), -int(flags["BT"]),
        candidate["region_start"],
    )


def scan_density_pileup(event: dict, series: pd.DataFrame,
                        search_start: pd.Timestamp, search_end: pd.Timestamp) -> Tuple[List[dict], int]:
    """从最大窗口递减至 5 分钟；采用前缀和，避免逐窗重复求均值。"""
    grid = series.index
    grid_ns = grid.asi8
    step_ns = MIN_NS
    search_start = search_start.floor("min")
    search_end = search_end.floor("min")  # 不允许向上取整越过本体前 20% 的终点
    i0 = int(np.searchsorted(grid_ns, search_start.value, side="left"))
    i1 = int(np.searchsorted(grid_ns, search_end.value, side="left"))  # 半开右边界
    max_len = int((search_end - search_start).total_seconds() // 60)
    if max_len < MIN_WINDOW_MINUTES or i1 - i0 < MIN_WINDOW_MINUTES:
        return [], 0

    prefixes = {
        "Np_cm3": _prefix(series["Np_cm3"].to_numpy(dtype=float)),
        "Beta_p": _prefix(series["Beta_p"].to_numpy(dtype=float)),
        # |B_T| 的零值是有效测量，因此不能按正值过滤。
        "B_T_abs_nT": _prefix(series["B_T_abs_nT"].to_numpy(dtype=float), positive_only=False),
    }
    selected: List[dict] = []
    evaluable_windows = 0
    body_start, body_end = event["body_start"], event["body_end"]

    for length in range(max_len, MIN_WINDOW_MINUTES - 1, -STEP_MINUTES):
        if i0 + length > i1:
            continue
        starts = np.arange(i0, i1 - length + 1, STEP_MINUTES, dtype=np.int64)
        if starts.size == 0:
            continue
        ends = starts + length
        before_starts, before_ends = starts - REFERENCE_MINUTES, starts
        after_starts, after_ends = ends, ends + REFERENCE_MINUTES
        if before_starts.min() < 0 or after_ends.max() >= len(grid) + 1:
            continue

        stats = {}
        coverages = []
        for key in ("Np_cm3", "Beta_p", "B_T_abs_nT"):
            w_mean, w_cov = _interval_stats(prefixes[key], starts, ends, length)
            pre_mean, pre_cov = _interval_stats(prefixes[key], before_starts, before_ends, REFERENCE_MINUTES)
            post_mean, post_cov = _interval_stats(prefixes[key], after_starts, after_ends, REFERENCE_MINUTES)
            stats[key] = {"mean": w_mean, "pre": pre_mean, "post": post_mean,
                          "cov": w_cov, "pre_cov": pre_cov, "post_cov": post_cov}
            coverages.extend([w_cov, pre_cov, post_cov])

        coverage_ok = np.ones(starts.size, dtype=bool)
        for cov in coverages:
            coverage_ok &= cov >= MIN_COVERAGE
        evaluable_windows += int(coverage_ok.sum())

        n_mean = stats["Np_cm3"]["mean"]
        n_pre = stats["Np_cm3"]["pre"]
        n_post = stats["Np_cm3"]["post"]
        beta_mean = stats["Beta_p"]["mean"]
        beta_pre = stats["Beta_p"]["pre"]
        beta_post = stats["Beta_p"]["post"]
        bt_mean = stats["B_T_abs_nT"]["mean"]
        bt_pre = stats["B_T_abs_nT"]["pre"]
        bt_post = stats["B_T_abs_nT"]["post"]

        cond_n = (n_mean >= NpY * np.maximum(n_pre, n_post))
        cond_beta = (beta_mean >= betaY * np.maximum(beta_pre, beta_post))
        cond_b = np.isfinite(bt_mean) & (bt_pre > 0) & (bt_post > 0) & (bt_mean <= BtY * np.minimum(bt_pre, bt_post))
        # 数据覆盖率不足的窗口不参加条件判定。
        cond_n &= coverage_ok
        cond_beta &= coverage_ok
        cond_b &= coverage_ok
        condition_count = cond_n.astype(int) + cond_beta.astype(int) + cond_b.astype(int)
        hit_positions = np.flatnonzero(coverage_ok & (condition_count > 0))
        if hit_positions.size == 0:
            continue

        start_ns = grid_ns[starts]
        end_ns = grid_ns[ends]
        distance, overlap = _distance_and_overlap(start_ns, end_ns, body_start, body_end)
        front_distance = _distance_to_icme_front_min(start_ns, end_ns, body_start)
        # 等级分值与 _classify 完全一致：强典型 5、典型 4、弱典型 3、非典型 2、非密度 0。
        grade_rank = np.zeros(starts.size, dtype=int)
        grade_rank[(~cond_n) & (cond_beta | cond_b)] = 2
        grade_rank[cond_n & ~(cond_beta | cond_b)] = 3
        grade_rank[cond_n & (cond_beta ^ cond_b)] = 4
        grade_rank[cond_n & cond_beta & cond_b] = 5
        priority = cond_n.astype(int) * 4 + cond_beta.astype(int) * 2 + cond_b.astype(int)

        # 将重叠的同长度命中窗口视为同一候选结构。断开且不重叠的组保留为不同候选。
        hit_start_values = starts[hit_positions]
        split_at = np.flatnonzero(np.diff(hit_start_values) > length) + 1
        groups = np.split(hit_positions, split_at)

        def selection_key(k: int) -> tuple:
            # 同长度候选优先靠近 ICME 本体前端；再考虑重叠、等级和参数优先级。
            return (front_distance[k], -overlap[k], -grade_rank[k], -priority[k], int(starts[k]))

        def core_key(k: int) -> tuple:
            return (-grade_rank[k], -condition_count[k], -priority[k], distance[k], -overlap[k], int(starts[k]))

        def make_candidate(k: int) -> dict:
            s_idx, e_idx = int(starts[k]), int(ends[k])
            s_time, e_time = grid[s_idx], grid[e_idx]
            flags = {"Np": bool(cond_n[k]), "Beta": bool(cond_beta[k]), "BT": bool(cond_b[k])}
            row = {
                "region_start": s_time,
                "region_end": e_time,
                "duration_min": int(length),
                "distance_to_icme_min": float(distance[k]),
                "distance_to_icme_front_min": float(front_distance[k]),
                "overlap_with_icme_min": float(overlap[k]),
                "grade": _classify(flags["Np"], flags["Beta"], flags["BT"]),
                "conditions": flags,
                "condition_count": int(condition_count[k]),
                "Np_mean_cm3": float(n_mean[k]),
                "Np_pre2h_mean_cm3": float(n_pre[k]),
                "Np_post2h_mean_cm3": float(n_post[k]),
                "Np_ratio_pre": float(n_mean[k] / n_pre[k]) if n_pre[k] > 0 else np.nan,
                "Np_ratio_post": float(n_mean[k] / n_post[k]) if n_post[k] > 0 else np.nan,
                "Beta_mean": float(beta_mean[k]),
                "Beta_pre2h_mean": float(beta_pre[k]),
                "Beta_post2h_mean": float(beta_post[k]),
                "Beta_ratio_pre": float(beta_mean[k] / beta_pre[k]) if beta_pre[k] > 0 else np.nan,
                "Beta_ratio_post": float(beta_mean[k] / beta_post[k]) if beta_post[k] > 0 else np.nan,
                "BT_abs_mean_nT": float(bt_mean[k]),
                "BT_abs_pre2h_mean_nT": float(bt_pre[k]),
                "BT_abs_post2h_mean_nT": float(bt_post[k]),
                "BT_abs_ratio_pre": float(bt_mean[k] / bt_pre[k]) if bt_pre[k] > 0 else np.nan,
                "BT_abs_ratio_post": float(bt_mean[k] / bt_post[k]) if bt_post[k] > 0 else np.nan,
                "minimum_coverage_pct": float(min(
                    stats[p][x][k]
                    for p in stats for x in ("cov", "pre_cov", "post_cov")
                ) * 100.0),
                "_grade_rank": int(grade_rank[k]),
                "_priority": int(priority[k]),
            }
            return row

        for group in groups:
            # 在该重叠组内，选一个最接近 ICME 的主窗口。
            main_pos = min(group.tolist(), key=selection_key)
            main_candidate = make_candidate(main_pos)
            group_start = grid[int(starts[group].min())]
            group_end = grid[int((starts[group] + length).max())]

            overlapped = [s for s in selected if
                          group_start <= s["main"]["region_end"] and group_end >= s["main"]["region_start"]]
            if overlapped:
                # 长区间保留不拆；将包含在长区间内的短窗口用于记录局部最高等级。
                for existing in overlapped:
                    contained_positions = [int(k) for k in group
                        if starts[k] >= grid.get_loc(existing["main"]["region_start"])
                        and (starts[k] + length) <= grid.get_loc(existing["main"]["region_end"])]
                    if not contained_positions:
                        continue
                    best_inner_pos = min(contained_positions, key=core_key)
                    inner = make_candidate(best_inner_pos)
                    if _candidate_core_key(inner) < _candidate_core_key(existing["core"]):
                        existing["core"] = inner
                continue

            # 初始化局部核心为主窗口；同组窗口若完全落在主窗口内，再寻找最高等级局部窗口。
            main_start_i = int(grid.get_loc(main_candidate["region_start"]))
            main_end_i = int(grid.get_loc(main_candidate["region_end"]))
            contained_in_main = [int(k) for k in group
                if starts[k] >= main_start_i and (starts[k] + length) <= main_end_i]
            if contained_in_main:
                core_pos = min(contained_in_main, key=core_key)
                core_candidate = make_candidate(core_pos)
            else:
                core_candidate = main_candidate
            selected.append({"main": main_candidate, "core": core_candidate})

    selected.sort(key=lambda x: x["main"]["region_start"])
    return selected, evaluable_windows


def _evaluate_fixed_interval(event: dict, series: pd.DataFrame,
                             region_start: pd.Timestamp, region_end: pd.Timestamp) -> Optional[dict]:
    """重新验证合并后的完整区间；若覆盖率不足或三个判据均不通过则返回 None。"""
    duration_min = int(round((region_end - region_start).total_seconds() / 60.0))
    if duration_min < MIN_WINDOW_MINUTES:
        return None

    keys = ("Np_cm3", "Beta_p", "B_T_abs_nT")
    ref_delta = pd.Timedelta(minutes=REFERENCE_MINUTES)
    intervals = {
        "window": (region_start, region_end, duration_min),
        "pre": (region_start - ref_delta, region_start, REFERENCE_MINUTES),
        "post": (region_end, region_end + ref_delta, REFERENCE_MINUTES),
    }
    stats = {}
    coverages = []
    idx = series.index
    for key in keys:
        stats[key] = {}
        for part, (left, right, expected) in intervals.items():
            values = pd.to_numeric(series.loc[(idx >= left) & (idx < right), key], errors="coerce").to_numpy(float)
            good = np.isfinite(values)
            if key != "B_T_abs_nT":
                good &= values > 0
            coverage = float(good.sum()) / float(expected) if expected else 0.0
            mean = float(np.mean(values[good])) if good.any() else np.nan
            stats[key][part] = {"mean": mean, "coverage": coverage}
            coverages.append(coverage)

    if not np.all(np.isfinite(coverages)) or min(coverages) < MIN_COVERAGE:
        return None

    n = stats["Np_cm3"]["window"]["mean"]
    n_pre = stats["Np_cm3"]["pre"]["mean"]
    n_post = stats["Np_cm3"]["post"]["mean"]
    beta = stats["Beta_p"]["window"]["mean"]
    beta_pre = stats["Beta_p"]["pre"]["mean"]
    beta_post = stats["Beta_p"]["post"]["mean"]
    bt = stats["B_T_abs_nT"]["window"]["mean"]
    bt_pre = stats["B_T_abs_nT"]["pre"]["mean"]
    bt_post = stats["B_T_abs_nT"]["post"]["mean"]

    cond_n = bool(n >= 2.0 * max(n_pre, n_post))
    cond_beta = bool(beta >= 2.0 * max(beta_pre, beta_post))
    cond_b = bool(np.isfinite(bt) and bt_pre > 0 and bt_post > 0 and bt <= 0.60 * min(bt_pre, bt_post))
    flags = {"Np": cond_n, "Beta": cond_beta, "BT": cond_b}
    count = int(cond_n) + int(cond_beta) + int(cond_b)
    if count == 0:
        return None

    dist, overlap = _distance_and_overlap(
        np.asarray([region_start.value], dtype=np.int64),
        np.asarray([region_end.value], dtype=np.int64),
        event["body_start"], event["body_end"]
    )
    front_distance = _distance_to_icme_front_min(
        np.asarray([region_start.value], dtype=np.int64),
        np.asarray([region_end.value], dtype=np.int64),
        event["body_start"]
    )
    priority = int(cond_n) * 4 + int(cond_beta) * 2 + int(cond_b)
    return {
        "region_start": region_start,
        "region_end": region_end,
        "duration_min": duration_min,
        "distance_to_icme_min": float(dist[0]),
        "distance_to_icme_front_min": float(front_distance[0]),
        "overlap_with_icme_min": float(overlap[0]),
        "grade": _classify(cond_n, cond_beta, cond_b),
        "conditions": flags,
        "condition_count": count,
        "Np_mean_cm3": n,
        "Np_pre2h_mean_cm3": n_pre,
        "Np_post2h_mean_cm3": n_post,
        "Np_ratio_pre": n / n_pre if n_pre > 0 else np.nan,
        "Np_ratio_post": n / n_post if n_post > 0 else np.nan,
        "Beta_mean": beta,
        "Beta_pre2h_mean": beta_pre,
        "Beta_post2h_mean": beta_post,
        "Beta_ratio_pre": beta / beta_pre if beta_pre > 0 else np.nan,
        "Beta_ratio_post": beta / beta_post if beta_post > 0 else np.nan,
        "BT_abs_mean_nT": bt,
        "BT_abs_pre2h_mean_nT": bt_pre,
        "BT_abs_post2h_mean_nT": bt_post,
        "BT_abs_ratio_pre": bt / bt_pre if bt_pre > 0 else np.nan,
        "BT_abs_ratio_post": bt / bt_post if bt_post > 0 else np.nan,
        "minimum_coverage_pct": min(coverages) * 100.0,
        "_grade_rank": GRADE_RANK[_classify(cond_n, cond_beta, cond_b)],
        "_priority": priority,
    }


def _reduce_regions_to_one(event: dict, series: pd.DataFrame,
                           selected: List[dict]) -> Tuple[List[dict], str, int]:
    """每个 ICME 尽量只留一个堆积区；相邻间隔不超过阈值则尝试合并。

    若候选簇之间的间隔超过阈值，只保留距离 ICME 本体前端最近的候选簇。
    合并区间必须重新通过至少一项物理判据，否则不伪造一个不合格的长事件，
    而在最近候选簇中保留距离本体前端最近的原始有效候选区间。
    """
    original_count = len(selected)
    if original_count <= 1:
        if original_count == 1:
            selected[0]["merged_count"] = 1
            return selected, "该 ICME 仅检出 1 个候选区，直接保留。", original_count
        return selected, "未检出通过判据的候选密度堆积区。", original_count

    ordered = sorted(selected, key=lambda x: x["main"]["region_start"])
    clusters: List[List[dict]] = []
    current: List[dict] = []
    current_end: Optional[pd.Timestamp] = None
    for item in ordered:
        cand = item["main"]
        start, end = cand["region_start"], cand["region_end"]
        if not current:
            current = [item]
            current_end = end
            continue
        gap_min = (start - current_end).total_seconds() / 60.0
        if gap_min <= MAX_GAP_BETWEEN_REGIONS_MINUTES:
            current.append(item)
            current_end = max(current_end, end)
        else:
            clusters.append(current)
            current = [item]
            current_end = end
    if current:
        clusters.append(current)

    def near_key(item: dict) -> tuple:
        cand = item["main"]
        distance = cand.get("distance_to_icme_front_min")
        if distance is None or not np.isfinite(distance):
            distance = abs((cand["region_start"] - event["body_start"]).total_seconds()) / 60.0
        # 优先与 ICME 本体前端距离近；距离相同时优先较长、等级较高的有效区。
        return (float(distance), -cand["duration_min"], -GRADE_RANK.get(cand["grade"], -1), cand["region_start"])

    closest_cluster = min(
        clusters,
        key=lambda cluster: min(near_key(item) for item in cluster)
    )
    dropped_clusters = len(clusters) - 1
    cluster_members = len(closest_cluster)
    interval_start = min(item["main"]["region_start"] for item in closest_cluster)
    interval_end = max(item["main"]["region_end"] for item in closest_cluster)

    notes = []
    if dropped_clusters:
        notes.append(
            f"检测到 {len(clusters)} 个相互间隔超过 {MAX_GAP_BETWEEN_REGIONS_MINUTES} 分钟的候选簇；"
            "仅保留距离 ICME 本体前端最近的候选簇，其余较远候选簇已舍弃。"
        )
    if cluster_members > 1:
        merged = _evaluate_fixed_interval(event, series, interval_start, interval_end)
        if merged is not None:
            best_core = min(
                [item["core"] for item in closest_cluster],
                key=_candidate_core_key
            )
            final = [{"main": merged, "core": best_core, "merged_count": cluster_members}]
            notes.append(
                f"最近候选簇内有 {cluster_members} 个候选区，区间间隔均不超过阈值；"
                "合并后整体仍通过至少一项判据，已合并为一个较长事件。"
            )
            return final, "".join(notes), original_count

        nearest_member = min(closest_cluster, key=near_key)
        nearest_member["merged_count"] = 1
        notes.append(
            f"最近候选簇内有 {cluster_members} 个候选区，但合并后的整体区间未通过判据；"
            "为避免输出不符合物理阈值的长区间，保留其中距离 ICME 本体前端最近的原始有效区。"
        )
        return [nearest_member], "".join(notes), original_count

    closest_cluster[0]["merged_count"] = 1
    return [closest_cluster[0]], "".join(notes) or "候选筛选完成，保留距离 ICME 本体前端最近的区间。", original_count


def _format_candidate_row(event: dict, search_start: pd.Timestamp, search_end: pd.Timestamp,
                          candidate: Optional[dict], core: Optional[dict], status: str,
                          region_number: Optional[int] = None, error: str = "",
                          selection_note: str = "", merged_candidate_count: int = 1) -> dict:
    row = {
        "Satellite": SATELLITE,
        "Event_ID": event.get("event_id", ""),
        "Source_File": event.get("source_file", ""),
        "Source_Line": event.get("source_line", np.nan),
        "Has_Shock": bool(event.get("has_shock", False)),
        "Shock_Time_UTC": event.get("shock_time", pd.NaT),
        "ICME_Start_UTC": event["body_start"],
        "ICME_End_UTC": event["body_end"],
        "Search_Start_UTC": search_start,
        "Search_End_UTC": search_end,
        "Region_Number": region_number if region_number is not None else np.nan,
        "Status": status,
        "Error": error,
        "Region_Selection_Note": selection_note,
        "Merged_Candidate_Count": merged_candidate_count if candidate is not None else np.nan,
    }
    if candidate is None:
        # Status 保留“非密度堆积区事件”以便单独分表；Grade 使用正式等级名“非密度堆积区”。
        grade_label = "非密度堆积区" if status == "非密度堆积区事件" else status
        row.update({
            "Pileup_Start_UTC": pd.NaT, "Pileup_End_UTC": pd.NaT, "Duration_min": np.nan,
            "Distance_to_ICME_min": np.nan, "Distance_to_ICME_Front_min": np.nan, "Overlap_with_ICME_min": np.nan,
            "Grade": grade_label, "Np_condition": np.nan, "Beta_condition": np.nan, "BT_condition": np.nan,
            "Condition_Count": np.nan, "Np_mean_cm3": np.nan,
            "Np_pre2h_mean_cm3": np.nan, "Np_post2h_mean_cm3": np.nan,
            "Np_ratio_pre": np.nan, "Np_ratio_post": np.nan,
            "Beta_mean": np.nan, "Beta_pre2h_mean": np.nan, "Beta_post2h_mean": np.nan,
            "Beta_ratio_pre": np.nan, "Beta_ratio_post": np.nan,
            "BT_abs_mean_nT": np.nan, "BT_abs_pre2h_mean_nT": np.nan, "BT_abs_post2h_mean_nT": np.nan,
            "BT_abs_ratio_pre": np.nan, "BT_abs_ratio_post": np.nan, "Minimum_Coverage_pct": np.nan,
            "Local_Core_Start_UTC": pd.NaT, "Local_Core_End_UTC": pd.NaT,
            "Local_Core_Duration_min": np.nan, "Local_Core_Grade": grade_label,
            "Local_Core_Np_condition": np.nan, "Local_Core_Beta_condition": np.nan,
            "Local_Core_BT_condition": np.nan,
        })
        return row

    flags = candidate["conditions"]
    row.update({
        "Pileup_Start_UTC": candidate["region_start"],
        "Pileup_End_UTC": candidate["region_end"],
        "Duration_min": candidate["duration_min"],
        "Distance_to_ICME_min": candidate["distance_to_icme_min"],
        "Distance_to_ICME_Front_min": candidate.get("distance_to_icme_front_min", np.nan),
        "Overlap_with_ICME_min": candidate["overlap_with_icme_min"],
        "Grade": candidate["grade"],
        "Np_condition": flags["Np"], "Beta_condition": flags["Beta"], "BT_condition": flags["BT"],
        "Condition_Count": candidate["condition_count"],
        "Np_mean_cm3": candidate["Np_mean_cm3"],
        "Np_pre2h_mean_cm3": candidate["Np_pre2h_mean_cm3"],
        "Np_post2h_mean_cm3": candidate["Np_post2h_mean_cm3"],
        "Np_ratio_pre": candidate["Np_ratio_pre"], "Np_ratio_post": candidate["Np_ratio_post"],
        "Beta_mean": candidate["Beta_mean"],
        "Beta_pre2h_mean": candidate["Beta_pre2h_mean"],
        "Beta_post2h_mean": candidate["Beta_post2h_mean"],
        "Beta_ratio_pre": candidate["Beta_ratio_pre"], "Beta_ratio_post": candidate["Beta_ratio_post"],
        "BT_abs_mean_nT": candidate["BT_abs_mean_nT"],
        "BT_abs_pre2h_mean_nT": candidate["BT_abs_pre2h_mean_nT"],
        "BT_abs_post2h_mean_nT": candidate["BT_abs_post2h_mean_nT"],
        "BT_abs_ratio_pre": candidate["BT_abs_ratio_pre"], "BT_abs_ratio_post": candidate["BT_abs_ratio_post"],
        "Minimum_Coverage_pct": candidate["minimum_coverage_pct"],
        "Local_Core_Start_UTC": core["region_start"] if core else candidate["region_start"],
        "Local_Core_End_UTC": core["region_end"] if core else candidate["region_end"],
        "Local_Core_Duration_min": core["duration_min"] if core else candidate["duration_min"],
        "Local_Core_Grade": core["grade"] if core else candidate["grade"],
        "Local_Core_Np_condition": core["conditions"]["Np"] if core else flags["Np"],
        "Local_Core_Beta_condition": core["conditions"]["Beta"] if core else flags["Beta"],
        "Local_Core_BT_condition": core["conditions"]["BT"] if core else flags["BT"],
    })
    return row


def compute_search_bounds(event: dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Only scan the sheath/leading edge, never beyond 20% of the ICME body."""
    body_start, body_end = event["body_start"], event["body_end"]
    duration = body_end - body_start
    if duration <= pd.Timedelta(0):
        raise ValueError(f"ICME 本体时间无效：{event['event_id']}")
    if not 0 < MAX_BODY_FRACTION <= 0.20:
        raise ValueError("MAX_BODY_FRACTION 必须在 (0, 0.20] 内")
    if not 0 < NO_SHOCK_PRE_BODY_FRACTION <= 1:
        raise ValueError("NO_SHOCK_PRE_BODY_FRACTION 必须在 (0, 1] 内")
    shock = event.get("shock_time", pd.NaT)
    if event["has_shock"] and pd.notna(shock) and shock < body_start:
        if not 0 <= SHOCK_MARGIN_FRACTION <= 1:
            raise ValueError("SHOCK_MARGIN_FRACTION 必须在 [0, 1] 内")
        # D 是激波到 ICME 本体起始的鞘层时长；向上取整，避免分钟网格重新包含被排除的激波附近数据。
        sheath_duration = body_start - shock
        search_start = (shock + sheath_duration * SHOCK_MARGIN_FRACTION).ceil("min")
        if SHOCK_MARGIN_FRACTION == 0:
            search_start = shock.floor("min")  # 设为 0 时恢复原有激波起点行为
    else:
        search_start = (body_start - duration * NO_SHOCK_PRE_BODY_FRACTION).floor("min")
    # Floor to a complete minute: ceil would allow a candidate beyond the limit.
    search_end = (body_start + duration * MAX_BODY_FRACTION).floor("min")
    return search_start, search_end


def process_one_event(event: dict) -> Tuple[List[dict], dict]:
    search_start, search_end = compute_search_bounds(event)
    print(f"    搜索范围={search_start} -> {search_end}（本体前 {MAX_BODY_FRACTION:.0%} 内）", flush=True)

    try:
        minute_series = read_event_minute_series(event, search_start, search_end)
        selected, valid_count = scan_density_pileup(event, minute_series, search_start, search_end)
        raw_region_count = len(selected)
        selected, selection_note, raw_region_count = _reduce_regions_to_one(event, minute_series, selected)
        for picked in selected:
            if picked["main"]["region_end"] > search_end or picked["main"]["region_start"] < search_start:
                raise RuntimeError("候选区超出前 20% 搜索边界，已阻止输出")
        if selected:
            output_rows = []
            for region_i, picked in enumerate(selected, 1):
                output_rows.append(_format_candidate_row(
                    event, search_start, search_end, picked["main"], picked["core"],
                    "已检出", region_i, selection_note=selection_note,
                    merged_candidate_count=picked.get("merged_count", 1)
                ))
            summary = {
                "Satellite": SATELLITE, "Event_ID": event["event_id"],
                "ICME_Start_UTC": event["body_start"], "ICME_End_UTC": event["body_end"],
                "Has_Shock": event["has_shock"], "Search_Start_UTC": search_start,
                "Search_End_UTC": search_end, "Status": "已检出",
                "Detected_Regions": len(selected),
                "Candidates_Before_Selection": raw_region_count,
                "Suppressed_or_Merged_Candidates": max(0, raw_region_count - len(selected)),
                "Max_Gap_Between_Regions_min": MAX_GAP_BETWEEN_REGIONS_MINUTES,
                "Selection_Note": selection_note,
                "Evaluable_Windows_All_Scales": valid_count,
                "Max_Grade": max((x["main"]["grade"] for x in selected), key=lambda g: GRADE_RANK.get(g, -1)),
                "Error": "",
            }
            return output_rows, summary

        status = "非密度堆积区事件" if valid_count > 0 else "数据不足/无法判定"
        row = _format_candidate_row(
            event, search_start, search_end, None, None, status,
            selection_note="未检出通过判据的候选区。"
        )
        summary = {
            "Satellite": SATELLITE, "Event_ID": event["event_id"],
            "ICME_Start_UTC": event["body_start"], "ICME_End_UTC": event["body_end"],
            "Has_Shock": event["has_shock"], "Search_Start_UTC": search_start,
            "Search_End_UTC": search_end, "Status": status,
            "Detected_Regions": 0, "Candidates_Before_Selection": 0,
            "Suppressed_or_Merged_Candidates": 0,
            "Max_Gap_Between_Regions_min": MAX_GAP_BETWEEN_REGIONS_MINUTES,
            "Selection_Note": "未检出通过判据的候选区。",
            "Evaluable_Windows_All_Scales": valid_count,
            "Max_Grade": ("非密度堆积区" if status == "非密度堆积区事件" else status), "Error": "",
        }
        return [row], summary
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[错误] {event['event_id']} ({event['body_start']}): {error}")
        traceback.print_exc()
        row = _format_candidate_row(
            event, search_start, search_end, None, None, "读取失败/无法判定",
            error=error, selection_note="读取失败，未进行候选区筛选。"
        )
        summary = {
            "Satellite": SATELLITE, "Event_ID": event["event_id"],
            "ICME_Start_UTC": event["body_start"], "ICME_End_UTC": event["body_end"],
            "Has_Shock": event["has_shock"], "Search_Start_UTC": search_start,
            "Search_End_UTC": search_end, "Status": "读取失败/无法判定",
            "Detected_Regions": 0, "Candidates_Before_Selection": 0,
            "Suppressed_or_Merged_Candidates": 0,
            "Max_Gap_Between_Regions_min": MAX_GAP_BETWEEN_REGIONS_MINUTES,
            "Selection_Note": "读取失败，未进行候选区筛选。",
            "Evaluable_Windows_All_Scales": 0,
            "Max_Grade": "读取失败/无法判定", "Error": error,
        }
        return [row], summary


def main() -> None:
    parser = argparse.ArgumentParser(description="PSP ICME 前缘密度堆积区检索（最多进入本体前 20%）")
    parser.add_argument("--event-id", help="只测试指定 Event_ID；省略时检索全部事件")
    parser.add_argument("--output-dir", type=Path, help="结果输出目录；省略时使用脚本旁默认目录")
    args = parser.parse_args()
    if args.event_id and args.output_dir is None:
        parser.error("单事件测试必须同时指定 --output-dir，避免覆盖正式结果表")
    output_dir = args.output_dir.expanduser() if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_event_list_files()
    if args.event_id:
        events = [event for event in events if event["event_id"] == args.event_id]
        if not events:
            raise ValueError(f"事件列表中找不到 Event_ID={args.event_id!r}")
    if not events:
        raise RuntimeError("事件列表文件存在，但没有解析到有效 ICME 事件。请检查每行是否包含至少两个时间戳。")
    print(f"\n[{SATELLITE}] 共 {len(events)} 个 ICME 事件；窗口从最大长度递减到 {MIN_WINDOW_MINUTES} 分钟。")
    print(f"同一 ICME 默认只保留一个候选区；相邻候选间隔阈值={MAX_GAP_BETWEEN_REGIONS_MINUTES} 分钟。")
    print(f"有激波起点：激波 + 鞘层时长 D 的 {SHOCK_MARGIN_FRACTION:.0%}；检索终点：本体开始 + 本体时长的 {MAX_BODY_FRACTION:.0%}；无激波起点：本体开始 - 本体时长的 {NO_SHOCK_PRE_BODY_FRACTION:.0%}。")
    print("注意：Beta 使用总磁场模长计算；第三项磁场判据使用 RTN-T 分量绝对值 |B_T|；数据覆盖率不足的窗口不参与判定。\n")

    all_event_rows: List[dict] = []
    summary_rows: List[dict] = []
    for i, event in enumerate(events, 1):
        print(f"[{SATELLITE} {i}/{len(events)}] {event['event_id']} | ICME={event['body_start']} -> {event['body_end']} | 激波={event['has_shock']}")
        rows, summary = process_one_event(event)
        all_event_rows.extend(rows)
        summary_rows.append(summary)
        print(f"    状态={summary['Status']}；检出区间数={summary['Detected_Regions']}；有效窗口数={summary['Evaluable_Windows_All_Scales']}")

    all_rows_df = pd.DataFrame(all_event_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not all_rows_df.empty and "Status" in all_rows_df:
        result_df = all_rows_df.loc[all_rows_df["Status"] == "已检出"].copy()
        no_pileup_df = all_rows_df.loc[all_rows_df["Status"] == "非密度堆积区事件"].copy()
        indeterminate_df = all_rows_df.loc[
            all_rows_df["Status"].isin(["数据不足/无法判定", "读取失败/无法判定"])
        ].copy()
    else:
        result_df = all_rows_df.copy()
        no_pileup_df = all_rows_df.copy()
        indeterminate_df = all_rows_df.copy()

    list_csv = output_dir / f"{SATELLITE}_ICME_密度堆积区列表.csv"
    no_pileup_csv = output_dir / f"{SATELLITE}_未检出密度堆积区的ICME列表.csv"
    indeterminate_csv = output_dir / f"{SATELLITE}_数据不足或读取失败的ICME列表.csv"
    summary_csv = output_dir / f"{SATELLITE}_ICME_检索摘要.csv"
    result_df.to_csv(list_csv, index=False, encoding="utf-8-sig")
    no_pileup_df.to_csv(no_pileup_csv, index=False, encoding="utf-8-sig")
    indeterminate_df.to_csv(indeterminate_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\n[完成] 已检出密度堆积区列表：{list_csv}（{len(result_df)} 条）")
    print(f"[完成] 未检出密度堆积区的 ICME 列表：{no_pileup_csv}（{len(no_pileup_df)} 条）")
    print(f"[完成] 数据不足/读取失败列表：{indeterminate_csv}（{len(indeterminate_df)} 条）")
    print(f"[完成] ICME 检索摘要：{summary_csv}")

    if WRITE_XLSX:
        xlsx_path = output_dir / f"{SATELLITE}_ICME_密度堆积区结果.xlsx"
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                result_df.to_excel(writer, sheet_name="密度堆积区列表", index=False)
                no_pileup_df.to_excel(writer, sheet_name="未检出堆积区ICME", index=False)
                indeterminate_df.to_excel(writer, sheet_name="数据不足或读取失败", index=False)
                summary_df.to_excel(writer, sheet_name="ICME检索摘要", index=False)
            print(f"[完成] Excel 汇总：{xlsx_path}")
        except Exception as exc:
            print(f"[提示] Excel 导出失败（CSV 已正常保存）：{exc}")

    print("\n已检出密度堆积区等级统计：")
    if not result_df.empty and "Grade" in result_df:
        print(result_df["Grade"].value_counts(dropna=False).to_string())
    else:
        print("没有检出符合判据的密度堆积区事件。")


if __name__ == "__main__":
    main()
