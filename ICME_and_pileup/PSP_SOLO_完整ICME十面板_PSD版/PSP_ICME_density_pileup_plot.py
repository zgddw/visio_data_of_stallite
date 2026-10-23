# -*- coding: utf-8 -*-
"""PSP full-ICME ten-panel plot with electron PSD PAD (PNG only)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROJECT = Path(r"E:\ZhaoWenjin_Reserch\anaysis_prog")
for folder in (PROJECT, BASE):
    if folder.is_dir() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

_CDFDATAREAD = None


def _lazy_cdfdataread(*args, **kwargs):
    """Keep pro_of_read's normal MAG/SPC functionality without startup delay."""
    global _CDFDATAREAD
    if _CDFDATAREAD is None:
        began = time.perf_counter()
        print("[延迟导入] 正在加载 pro_of_read.cdfdataread ...", flush=True)
        from pro_of_read import cdfdataread
        _CDFDATAREAD = cdfdataread
        print(f"[延迟导入] 完成，耗时 {time.perf_counter()-began:.2f} s", flush=True)
    return _CDFDATAREAD(*args, **kwargs)


def _wp_to_kelvin(wp_km_s):
    import numpy as np
    w = np.asarray(wp_km_s, dtype=float)
    return np.where(w > 0, 1.67262192369e-27 * (w * 1000.0)**2 /
                    (2.0 * 1.380649e-23), np.nan)


CONFIG = {
    "SATELLITE": "PSP",
    "CDFDATAREAD": _lazy_cdfdataread,
    "CDF_IMPORT_ERROR": None,
    "DETECTED_TABLE": None,
    "PROMPT_FOR_TABLE": False,
    "DETECTION_OUTPUT_DIR": PROJECT / "ICME_density_pileup_search_PSP_SOLO" /
                            "结果_PSP_多参数密度堆积区",
    "PLOT_OUTPUT_DIR": BASE / "综合图像_PSP",
    "MAG_ROOT": Path(r"I:\stdata\psp\FLD\PSP_FLD_L2_MAG_RTN_1MIN"),
    "MAG_PREFIX": "psp_fld_l2_mag_rtn_1min",
    "MAG_VARIABLE": "psp_fld_l2_mag_RTN_1min",
    "PLASMA_ROOT": Path(r"I:\stdata\psp\sweap\psp_swp_spc_l3i"),
    "PLASMA_PREFIX": "psp_swp_spc_l3i",
    "DENSITY_VARIABLE": "np_moment",
    "VELOCITY_VARIABLE": "vp_moment_RTN",
    "TEMPERATURE_VARIABLE": "wp_moment",
    "B_COMPONENT_CANDIDATES": (
        ["psp_fld_l2_mag_RTN_1min_X", "B_RTN_X", "Br", "BR"],
        ["psp_fld_l2_mag_RTN_1min_Y", "B_RTN_Y", "Bt", "BT"],
        ["psp_fld_l2_mag_RTN_1min_Z", "B_RTN_Z", "Bn", "BN"],
    ),
    "V_COMPONENT_CANDIDATES": (
        ["Vp_X", "vp_moment_RTN_X", "V_RTN_X", "Vr"],
        ["Vp_Y", "vp_moment_RTN_Y", "V_RTN_Y", "Vt"],
        ["Vp_Z", "vp_moment_RTN_Z", "V_RTN_Z", "Vn"],
    ),
    "DENSITY_CANDIDATES": ["Np", "np_moment", "N"],
    "VELOCITY_CANDIDATES": ["vp_moment_RTN", "Vp_RTN", "V_RTN"],
    "TEMPERATURE_CANDIDATES": ["wp_moment", "wp", "Wp"],
    "TEMPERATURE_TRANSFORM": _wp_to_kelvin,
    "PAD_ROOT": Path(r"I:\stdata\psp\sweap\spe\PSP_SWP_SPE_SF0_L3_PAD"),
    "PAD_PREFIX": "psp_swp_spe_sf0_l3_pad",
    "PAD_ENERGY_RANGE_EV": (70.0, 2000.0),  # 默认宽能窗；也可用 --energy-range 修改
    "PAD_APPLY_QUALITY_FILTER": True,
    "PAD_QUALITY_BAD_BITS": 0x000D,  # 默认剔除 overflow/alt table/spoiler；保留 snapshot/attenuator 并记录警告
    "PLOT_CONTEXT_RATIO": 0.15,  # 这里改默认比例：0.10=两端各延长10%，0.20=各延长20%
    "PLOT_CONTEXT_HOURS": None,  # 填数字可固定小时数，并覆盖上述比例
    "PLOT_DPI": 120,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="PSP 完整 ICME 十面板综合图：PAD 为相空间密度")
    parser.add_argument("--table", type=Path, help="事件 CSV/XLSX；省略时使用默认检索结果目录")
    parser.add_argument("--event-id", help="只画一个 Event_ID，适合先做测试")
    parser.add_argument("--output-dir", type=Path, help="PNG 和清单输出目录")
    parser.add_argument("--energy-range", nargs=2, type=float, metavar=("MIN_EV", "MAX_EV"),
                        default=None, help="PAD 能量范围 eV，默认采用 CONFIG 设置的 70 2000")
    parser.add_argument("--context-ratio", type=float, default=None,
                        help="完整事件跨度两端各延长的比例，默认 0.15；例如 0.10 或 0.20")
    parser.add_argument("--context-hours", type=float, default=None,
                        help="可选：固定延长小时数；指定后覆盖 --context-ratio")
    parser.add_argument("--dpi", type=int, default=120, help="PNG 分辨率，72–300，默认 120")
    args = parser.parse_args()
    lo, hi = args.energy_range if args.energy_range is not None else CONFIG["PAD_ENERGY_RANGE_EV"]
    if not (0 < lo < hi):
        parser.error("--energy-range 须给出正数下限和更大的上限")
    if not 72 <= args.dpi <= 300:
        parser.error("--dpi 须在 72–300 之间")
    if args.context_ratio is not None and not 0 <= args.context_ratio <= 1:
        parser.error("--context-ratio 必须在 0–1 之间")
    if args.context_hours is not None and args.context_hours < 0:
        parser.error("--context-hours 不能为负数")

    config = dict(CONFIG)
    config["PAD_ENERGY_RANGE_EV"] = (lo, hi)
    config["PLOT_DPI"] = args.dpi
    if args.context_ratio is not None:
        config["PLOT_CONTEXT_RATIO"] = args.context_ratio
    if args.context_hours is not None:
        config["PLOT_CONTEXT_HOURS"] = args.context_hours
    config["EVENT_ID_FILTER"] = args.event_id
    if args.table:
        config["DETECTED_TABLE"] = args.table.expanduser()
    if args.output_dir:
        config["PLOT_OUTPUT_DIR"] = args.output_dir.expanduser()

    print(f"[启动] Python: {sys.executable}", flush=True)
    print(f"[配置] PSP PAD PSD 能量窗 {lo:g}–{hi:g} eV；只保存 PNG", flush=True)
    began = time.perf_counter()
    import matplotlib
    matplotlib.use("Agg", force=True)
    import ICME_density_pileup_plot_common as common
    import psp_spe_pad
    common.plt.ioff()
    if Path(common.__file__).resolve().parent != BASE or Path(psp_spe_pad.__file__).resolve().parent != BASE:
        raise RuntimeError("绘图模块不在本脚本目录，请把三个 .py 文件放在同一目录")
    if psp_spe_pad.PSP_READER_VERSION != "PSP_SPE_PSD_20261022_V1":
        raise RuntimeError("PSP PAD 读取器版本不匹配")
    print(f"[启动] 绘图模块加载耗时 {time.perf_counter()-began:.2f} s", flush=True)
    common.run_plotting(config)


if __name__ == "__main__":
    main()
