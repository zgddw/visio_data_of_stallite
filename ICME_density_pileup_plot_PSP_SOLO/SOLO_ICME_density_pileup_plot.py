# -*- coding: utf-8 -*-
"""Solar Orbiter 密度堆积区综合绘图。

重要：若本机 SOLO EAS CDF 只包含 (elevation, energy, azimuth) 四维传感器数据，
且没有已转换的磁场对齐投掷角轴，本程序会在电子 PAD 面板明确提示，而不会伪造 0–180° PA。
其余磁场和质子参数仍可正常绘制。
"""
import sys
import argparse
from pathlib import Path
import numpy as np

CUSTOM_MODULE_DIRS = [
    Path(__file__).resolve().parent,
    Path(r"E:\ZhaoWenjin_Reserch\pileup"),
    Path(r"E:\ZhaoWenjin_Reserch\anaysis_prog"),
]
for _path in CUSTOM_MODULE_DIRS:
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
try:
    from pro_of_read import cdfdataread
except Exception as _exc:
    cdfdataread = None
    _CDF_IMPORT_ERROR = _exc
else:
    _CDF_IMPORT_ERROR = None

from ICME_density_pileup_plot_common import run_plotting

BASE = Path(__file__).resolve().parent
RESULT_DIR = BASE / "结果_SOLO_多参数密度堆积区"

CONFIG = {
    "SATELLITE": "SOLO",
    "CDFDATAREAD": cdfdataread,
    "CDF_IMPORT_ERROR": _CDF_IMPORT_ERROR,
    "DETECTED_TABLE": None,  # 可直接填你自己的 CSV/Excel 完整路径；None 时运行后由你选择
    "PROMPT_FOR_TABLE": True,  # True：运行时弹窗让用户选择列表；也可用 --table 指定
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
    "TEMPERATURE_TRANSFORM": lambda temp_ev: np.where(
        np.asarray(temp_ev, dtype=float) > 0,
        np.asarray(temp_ev, dtype=float) * 11604.51812,
        np.nan,
    ),
    "PAD_SRF_MAG_ROOT": Path(r"H:\solo\mag\SOLO_L2_MAG-SRF-NORMAL"),
    "PAD_SRF_MAG_PREFIX": "solo_l2_mag-srf-normal",
    "PAD_TARGET_ENERGY_EV": 70,
    "PAD_PA_BINS": 12,
    "PAD_MAG_MAX_DELTA_SECONDS": 90,
    "PAD_LOOK_IS_OPPOSITE_VELOCITY": True,
    "PLOT_CONTEXT_HOURS": 2.0,
    "PAD_ROOT": Path(r"I:\stdata\solo\swa\SOLO_L2_SWA-EAS-PAD-DEF"),
    "PAD_PREFIX": "solo_l2_swa-eas-pad-def",
    "PAD_FLUX_CANDIDATES": [
        "SWA_EAS_PAD_DEF_Flux", "SWA_EAS_PAD_DEF_Data", "EFLUX_VS_PA_E", "EFLUX_VS_PA_E_atE_byP",
        "EFLUX_VS_PA", "PitchAngleFlux",
    ],
    "PAD_PITCH_CANDIDATES": [
        "PITCHANGLE", "PITCH_ANGLE", "SWA_EAS_PITCH_ANGLE", "SWA_EAS_PAD_DEF_PitchAngle", "Pitch_Angle", "PA",
    ],
    "PAD_ENERGY_CANDIDATES": ["SWA_EAS_ENERGY", "ENERGY_VALS", "Electrons_ENERGY", "Energy", "ENERGY"],
    "PAD_DEFAULT_ENERGY_UNITS": "eV",
    "PAD_ENERGY_RANGE_EV": (200.0, 500.0),  # 在这里修改能量范围，例如 (200, 1000)
}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SOLO ICME 密度堆积区综合绘图（输入列表路径可由用户自行指定）"
    )
    parser.add_argument(
        "--table", type=Path, default=None,
        help="密度堆积区检索结果表完整路径（CSV/XLSX/XLS）；省略时弹出选择窗口",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="可选：自定义综合图像输出目录",
    )
    parser.add_argument(
        "--no-table-dialog", action="store_true",
        help="不弹出选表窗口；未提供 --table 时只按默认位置自动查找",
    )
    args = parser.parse_args()
    config = dict(CONFIG)
    if args.table is not None:
        config["DETECTED_TABLE"] = args.table.expanduser()
        config["PROMPT_FOR_TABLE"] = False
    elif args.no_table_dialog:
        config["PROMPT_FOR_TABLE"] = False
    if args.output_dir is not None:
        config["PLOT_OUTPUT_DIR"] = args.output_dir.expanduser()
    run_plotting(config)


if __name__ == "__main__":
    main()
