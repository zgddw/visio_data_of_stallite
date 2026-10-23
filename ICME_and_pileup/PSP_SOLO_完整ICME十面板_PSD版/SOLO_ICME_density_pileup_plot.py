# -*- coding: utf-8 -*-
"""Solar Orbiter 密度堆积区综合绘图。

SolO EAS 四维电子数据经 EAS_TO_SRF 与同期 MAG-SRF 计算投掷角。
"""
import sys
import time
import argparse
from pathlib import Path

CUSTOM_MODULE_DIRS = [
    Path(__file__).resolve().parent,
    Path(r"E:\ZhaoWenjin_Reserch\pileup"),
    Path(r"E:\ZhaoWenjin_Reserch\anaysis_prog"),
]
for _path in reversed(CUSTOM_MODULE_DIRS):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# 不在启动阶段导入 pro_of_read。该模块会加载 SpacePy，并可能触发闰秒表初始化；
# 实际需要读取 MAG/等离子体文件时再导入，避免程序看起来“卡在 from pro_of_read”。
_CDFDATAREAD = None
_CDF_IMPORT_ERROR = None


def _load_cdfdataread():
    """按需加载 pro_of_read.cdfdataread，并打印可诊断的耗时日志。"""
    global _CDFDATAREAD, _CDF_IMPORT_ERROR
    if _CDFDATAREAD is not None:
        return _CDFDATAREAD
    started = time.perf_counter()
    print("[延迟导入] 正在加载 pro_of_read.cdfdataread ...", flush=True)
    try:
        from pro_of_read import cdfdataread as reader
    except Exception as exc:
        _CDF_IMPORT_ERROR = exc
        elapsed = time.perf_counter() - started
        print(f"[延迟导入失败] pro_of_read 导入耗时 {elapsed:.2f} s: {exc!r}", flush=True)
        raise
    _CDFDATAREAD = reader
    elapsed = time.perf_counter() - started
    print(f"[延迟导入完成] pro_of_read 导入耗时 {elapsed:.2f} s", flush=True)
    return _CDFDATAREAD


def _lazy_cdfdataread(*args, **kwargs):
    return _load_cdfdataread()(*args, **kwargs)


def _temperature_ev_to_kelvin(temp_ev):
    import numpy as np
    values = np.asarray(temp_ev, dtype=float)
    return np.where(values > 0, values * 11604.51812, np.nan)


def _load_plot_modules():
    """延迟加载绘图模块，并确认使用的是本目录版本。"""
    started = time.perf_counter()
    print("[启动] 正在加载 SolO 绘图模块 ...", flush=True)
    import matplotlib
    matplotlib.use("Agg", force=True)  # 只保存文件，不创建交互绘图窗口
    import ICME_density_pileup_plot_common as plot_common
    plot_common.plt.ioff()
    import solo_nm3d_pad
    from ICME_density_pileup_plot_common import run_plotting
    elapsed = time.perf_counter() - started
    print(f"[启动] 绘图模块加载完成，耗时 {elapsed:.2f} s", flush=True)
    return plot_common, solo_nm3d_pad, run_plotting
BASE = Path(__file__).resolve().parent
RESULT_DIR = Path(r"E:\ZhaoWenjin_Reserch\anaysis_prog\ICME_density_pileup_search_PSP_SOLO\结果_SOLO_多参数密度堆积区")

CONFIG = {
    "SATELLITE": "SOLO",
    "CDFDATAREAD": _lazy_cdfdataread,
    "CDF_IMPORT_ERROR": _CDF_IMPORT_ERROR,
    "DETECTED_TABLE": None,  # 可用 --table 覆盖；默认从 RESULT_DIR 自动查找
    "PROMPT_FOR_TABLE": False,
    "DETECTION_OUTPUT_DIR": RESULT_DIR,
    "PLOT_OUTPUT_DIR": BASE / "综合图像_SOLO",
    "MAG_ROOT": Path(r"I:\stdata\solo\mag\RTN_NORMAL"),
    "MAG_PREFIX": "solo_l2_mag-rtn-normal",
    "MAG_VARIABLE": "B_RTN",
    "PLASMA_ROOT": Path(r"I:\stdata\solo\swa\solo_l2_swa-pas-grnd-mom"),
    "PLASMA_PREFIX": "solo_l2_swa-pas-grnd-mom",
    "DENSITY_VARIABLE": "N",
    "VELOCITY_VARIABLE": "V_RTN",
    "TEMPERATURE_VARIABLE": "T",
    "B_COMPONENT_CANDIDATES": (
        ["B_RTN_X", "B_RTN_R", "Br", "BR"],
        ["B_RTN_Y", "B_RTN_T", "Bt", "BT", "B_T"],
        ["B_RTN_Z", "B_RTN_N", "Bn", "BN"],
    ),
    "V_COMPONENT_CANDIDATES": (
        ["Vp_X", "V_RTN_X", "V_RTN_R", "Vr", "VR"],
        ["Vp_Y", "V_RTN_Y", "V_RTN_T", "Vt", "VT"],
        ["Vp_Z", "V_RTN_Z", "V_RTN_N", "Vn", "VN"],
    ),
    "DENSITY_CANDIDATES": ["Np", "N", "proton_density"],
    "VELOCITY_CANDIDATES": ["V_RTN", "Vp_RTN", "V"],
    "TEMPERATURE_CANDIDATES": ["Tp", "T", "T_proton"],
    "TEMPERATURE_TRANSFORM": _temperature_ev_to_kelvin,
    "PAD_SRF_MAG_ROOT": Path(r"H:\solo\mag\SOLO_L2_MAG-SRF-NORMAL"),
    "PAD_SRF_MAG_PREFIX": "solo_l2_mag-srf-normal",
    "PAD_MIN_ENERGY_EV": 70.0,
    "PAD_MAX_ENERGY_EV": float("inf"),
    "PAD_MIN_ENERGY_COVERAGE": 0.7,
    "PAD_AUTO_DOWNLOAD": True,
    "PAD_OFFLINE": False,
    "PAD_USE_SYSTEM_PROXY": True,
    "NM3D_ROOT": Path(r"H:\solo\swa"),
    "PAD_PA_BINS": 36,
    "PAD_MAG_MAX_DELTA_SECONDS": 90,
    "PAD_LOOK_IS_OPPOSITE_VELOCITY": False,  # NM3D 方位角标注为电子流方向
    "PAD_MAX_QUALITY_FLAG": 4,  # 0-4 为标注范围，255 为填充值；严格过滤可设 0
    "PLOT_CONTEXT_RATIO": 0.15,  # 这里改默认比例：0.10=两端各延长10%，0.20=各延长20%
    "PLOT_CONTEXT_HOURS": None,  # 填数字可固定小时数，并覆盖上述比例
    "PLOT_DPI": 120,
    "PAD_ROOT": Path(r"H:\solo\swa"),
    "PAD_PREFIX": "solo_l2_swa-eas1-nm3d-psd",
    # SolO 没有 PSP 式的独立 PITCHANGLE 变量；下列数据由 solo_nm3d_pad.py 读取。
    # 不再设置 PAD_PITCH_CANDIDATES / PAD_ENERGY_RANGE_EV，避免误导。

}

def main() -> None:
    print(f'[启动] Python: {sys.executable}', flush=True)
    plot_common, solo_nm3d_pad, run_plotting = _load_plot_modules()
    expected_dir = Path(__file__).resolve().parent
    expected_version = "SOLO_NM3D_PSD_20261021_V2_DOC_VERIFIED"
    if getattr(solo_nm3d_pad, "SOLO_READER_VERSION", None) != expected_version:
        raise RuntimeError("solo_nm3d_pad.py 不是本修复包版本，请将三个 Python 文件放在同一目录。")
    if Path(plot_common.__file__).resolve().parent != expected_dir:
        raise RuntimeError(f"载入了错误位置的共用模块：{plot_common.__file__}")
    if Path(solo_nm3d_pad.__file__).resolve().parent != expected_dir:
        raise RuntimeError(f"载入了错误位置的 SolO PAD 模块：{solo_nm3d_pad.__file__}")
    print(f"[版本确认] SolO PAD: {solo_nm3d_pad.SOLO_READER_VERSION}", flush=True)
    print(f"[模块路径] 共用模块: {plot_common.__file__}", flush=True)
    print(f"[模块路径] SolO 专用: {solo_nm3d_pad.__file__}", flush=True)

    parser = argparse.ArgumentParser(
        description="SOLO 完整 ICME 十面板综合图（NM3D 相空间密度）"
    )
    parser.add_argument(
        "--table", type=Path, default=None,
        help="密度堆积区检索结果表完整路径；省略时查找默认目录",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="可选：自定义综合图像输出目录",
    )
    parser.add_argument(
        "--no-table-dialog", action="store_true",
        help="不弹出选表窗口；未提供 --table 时只按默认位置自动查找",
    )
    parser.add_argument("--offline", action="store_true", help="仅用本地NM3D PSD，不查询/下载")
    parser.add_argument("--event-id", help="只绘制一个 Event_ID，便于测试")
    parser.add_argument("--min-energy", type=float, default=None, help="PSD求和最低能量(eV)，默认70")
    parser.add_argument("--max-energy", type=float, default=None, help="PSD求和最高能量(eV)，默认不限")
    parser.add_argument("--context-ratio", type=float, default=None,
                        help="完整事件跨度两端各延长的比例，默认0.15；例如0.10或0.20")
    parser.add_argument("--context-hours", type=float, default=None,
                        help="可选：固定延长小时数；指定后覆盖 --context-ratio")
    parser.add_argument("--dpi", type=int, default=None, help="PNG 分辨率，默认120；需要高清图可设180")
    args = parser.parse_args()
    config = dict(CONFIG)
    if args.table is not None:
        config["DETECTED_TABLE"] = args.table.expanduser()
        config["PROMPT_FOR_TABLE"] = False
    elif args.no_table_dialog:
        config["PROMPT_FOR_TABLE"] = False
    if args.output_dir is not None:
        config["PLOT_OUTPUT_DIR"] = args.output_dir.expanduser()
    if args.offline:
        config["PAD_OFFLINE"] = True
    config["EVENT_ID_FILTER"] = args.event_id
    if args.min_energy is not None:
        if args.min_energy <= 0:
            parser.error("--min-energy 必须大于 0")
        config["PAD_MIN_ENERGY_EV"] = args.min_energy
    if args.max_energy is not None:
        config["PAD_MAX_ENERGY_EV"] = args.max_energy
    if not config["PAD_MAX_ENERGY_EV"] > config["PAD_MIN_ENERGY_EV"]:
        parser.error("--max-energy 必须大于 --min-energy")
    if args.context_ratio is not None:
        if not 0 <= args.context_ratio <= 1:
            parser.error("--context-ratio 必须在 0–1 之间")
        config["PLOT_CONTEXT_RATIO"] = args.context_ratio
    if args.context_hours is not None:
        if args.context_hours < 0:
            parser.error("--context-hours 不能为负数")
        config["PLOT_CONTEXT_HOURS"] = args.context_hours
    if args.dpi is not None:
        if not 72 <= args.dpi <= 300:
            parser.error("--dpi 须在 72 到 300 之间")
        config["PLOT_DPI"] = args.dpi
    run_plotting(config)


if __name__ == "__main__":
    main()
