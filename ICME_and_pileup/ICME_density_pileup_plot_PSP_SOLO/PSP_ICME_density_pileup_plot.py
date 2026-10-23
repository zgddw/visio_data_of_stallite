# -*- coding: utf-8 -*-
"""PSP 密度堆积区综合绘图：每个已检出堆积区输出一张完整多面板图。"""
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
RESULT_DIR = BASE / "结果_PSP_多参数密度堆积区"

CONFIG = {
    "SATELLITE": "PSP",
    "CDFDATAREAD": cdfdataread,
    "CDF_IMPORT_ERROR": _CDF_IMPORT_ERROR,
    "DETECTED_TABLE": None,  # 可直接填你自己的 CSV/Excel 完整路径；None 时运行后由你选择
    "PROMPT_FOR_TABLE": True,  # True：运行时弹窗让用户选择列表；也可用 --table 指定
    "DETECTION_OUTPUT_DIR": RESULT_DIR,
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
        ["psp_fld_l2_mag_RTN_1min_Y", "B_RTN_Y", "Bt", "BT", "B_T"],
        ["psp_fld_l2_mag_RTN_1min_Z", "B_RTN_Z", "Bn", "BN"],
    ),
    "V_COMPONENT_CANDIDATES": (
        ["Vp_X", "vp_moment_RTN_X", "V_RTN_X", "Vr", "VR"],
        ["Vp_Y", "vp_moment_RTN_Y", "V_RTN_Y", "Vt", "VT"],
        ["Vp_Z", "vp_moment_RTN_Z", "V_RTN_Z", "Vn", "VN"],
    ),
    "DENSITY_CANDIDATES": ["Np", "np_moment", "N", "proton_density"],
    "VELOCITY_CANDIDATES": ["vp_moment_RTN", "Vp_RTN", "V_RTN"],
    "TEMPERATURE_CANDIDATES": ["wp_moment", "wp", "Wp"],
    "TEMPERATURE_TRANSFORM": lambda wp_km_s: np.where(
        np.asarray(wp_km_s, dtype=float) > 0,
        1.67262192369e-27 * (np.asarray(wp_km_s, dtype=float) * 1000.0) ** 2 / (2.0 * 1.380649e-23),
        np.nan,
    ),
    "PAD_ROOT": Path(r"I:\stdata\psp\sweap\spe\PSP_SWP_SPE_SF0_L3_PAD"),
    "PAD_PREFIX": "psp_swp_spe_sf0_l3_pad",
    "PAD_FLUX_CANDIDATES": ["EFLUX_VS_PA_E_atE_byP", "EFLUX_VS_PA_E_byE_atP", "EFLUX_VS_PA_E", "EFLUX_VS_PA"],
    "PAD_PITCH_CANDIDATES": ["PITCHANGLE", "PITCH_ANGLE", "Pitch_Angle", "PA"],
    "PAD_ENERGY_CANDIDATES": ["ENERGY_VALS", "Electrons_ENERGY", "Energy", "ENERGY"],
    "PAD_DEFAULT_ENERGY_UNITS": "eV",
    "PLOT_CONTEXT_HOURS": 2.0,  # 每个堆积区前后各显示 2 小时背景
    "PAD_TARGET_ENERGY_EV": 300.0,
    "PAD_ENERGY_RANGE_EV": (300.0, 300.0),  # 在这里修改能量范围，例如 (200, 1000)
}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PSP ICME 密度堆积区综合绘图（输入列表路径可由用户自行指定）"
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
