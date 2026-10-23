# -*- coding: utf-8 -*-
"""
PSP MAG + EPI-Hi LET1 He / O / Fe 数据处理与可视化
================================================

数据：
    1. PSP FIELDS MAG RTN 1-min
    2. PSP ISOIS EPI-Hi LET1 60-s Rates

主要输出：
    Br, Bt, Bn, Bmag
    He_flux(E,t), O_flux(E,t), Fe_flux(E,t)
    He_flux_integrated(t), O_flux_integrated(t), Fe_flux_integrated(t)
    He_density(t), O_density(t), Fe_density(t)
    Fe_O_ratio(t)

EPI-Hi LET1 中：
    A_He_Flux / B_He_Flux    -> He_ENERGY
    A_O_Flux / B_O_Flux      -> O_ENERGY
    A_Fe_Flux / B_Fe_Flux    -> Fe_ENERGY

Flux 单位：
    cm^-2 sr^-1 s^-1 (MeV/nuc)^-1

O/Fe density：
    在各向同性近似下，根据 differential directional flux J(E)：

        n = 4*pi * integral[J(E)/v(E) dE]

    EPI-Hi 能量是 MeV/nuc，因此速度使用每核子能量计算：

        v = sqrt(2 E_nuc / m_p)

    能量积分宽度优先直接使用 CDF 中的：
        O_ENERGY_DELTAPLUS / O_ENERGY_DELTAMINUS
        Fe_ENERGY_DELTAPLUS / Fe_ENERGY_DELTAMINUS

重要：
    这里得到的是 EPI-Hi 能量范围内的 energetic-ion density estimate，
    不是 SWEAP/SPC 意义上的热太阳风 O/Fe density。

本版本重点修复：
    1. 不再使用 pycdf.lib.v_datetime；Epoch 统一用 pandas 转换。
    2. 不再从 A_O/A_Fe 原始 rate 变量计算 density，而只使用 *_Flux。
    3. 不再通过猜测能量边界计算 dE，而直接读取官方 DELTAPLUS/MINUS。
    4. A/B 保持原始 0 值；只有 NaN 才视为无效。
    5. 不把二维 flux 强行放进 DataFrame；DataFrame 只保存一维时间序列。
    6. 修复原程序 plot_energy_spectra 中 epihi["O_flux"] / ["Fe_flux"] 不存在的问题。
    7. 增加二维 O/Fe flux spectrogram，便于直接确认通量确实有值。
    8. O/Fe 统一使用 1--128 MeV/nuc：O 第 1--28 道、Fe 第 3--30 道。
    9. differential flux 积分严格使用用户给出的每个能道 Delta E。
   10. Fe_O_ratio 定义为共同能量范围内的 integrated-flux Fe/O。
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm

from spacepy import pycdf


# ============================================================
# 用户配置
# ============================================================

EPIHI_ROOT = r"I:\stdata\psp\isois-epihi\let1"
MAG_ROOT = r"I:\stdata\psp\FLD\PSP_FLD_L2_MAG_RTN_1MIN"

START_TIME = "2023-03-13 00:00:00"
END_TIME = "2023-03-13 23:59:59"

OUTPUT_DIR = r"E:\ZhaoWenjin_Reserch\ICME\结果图像\PSP\线图\少数离子"

# A/B 合并方式：
#   mean -> 两侧平均
#   sum  -> 两侧相加
SIDE_METHOD = "mean"

# ============================================================
# O / Fe 共同能量范围
# ============================================================
# 用户给出的 LET1 能道定义：
#   Fe: 0.7--181.0 MeV/nuc，共 32 个能道
#   O : 1.0--128.0 MeV/nuc，共 28 个能道
#
# O 和 Fe 能量区间完全对应的共同范围为：
#   1.0--128.0 MeV/nuc
#
# 对应原始能道：
#   O : 1--28   -> Python index 0:28
#   Fe: 3--30   -> Python index 2:30
#
# 本程序严格按照下面给出的能量边界计算 Delta E，
# 不使用由能量中心反推的 bin width，也不简单 sum differential flux。
COMMON_E_MIN = 1.0
COMMON_E_MAX = 128.0

# 是否保存二维 flux 数组到 npz。
SAVE_FLUX_NPZ = True

# 是否显示图。
SHOW_FIGURES = False

# ============================================================
# ICME 事件列表
# ============================================================
# 文档每行前 5 列：
#   ICMEid  Satellite  ShockArrival  ICMEStart  ICMEEnd
# 后面的参数暂时全部忽略。
ICME_LIST_FILE = r"I:\list\icme\HELIO4CAST_ICMECAT_v23\divided\ICME_PSP.txt"

# 只处理指定卫星；当前程序的数据源为 PSP。
TARGET_SATELLITE = "PSP"

# 绘图时间窗口。
# 0 表示严格从激波到达时刻开始；
# 如果希望看到激波前背景，可改成例如 120（分钟）。
PLOT_PRE_MINUTES = 6 * 60
PLOT_POST_MINUTES = 6 * 60

# 是否自动批量处理列表中的所有事件。
BATCH_MODE = True

# ============================================================
# 数据处理原则：严格禁止平滑
# ============================================================
# 本程序不使用 rolling / Savitzky-Golay / Gaussian / 插值平滑。
# 不做人为降采样；保留 EPI-Hi LET1 和 MAG 的原始时间分辨率。
NO_SMOOTHING = True


# ============================================================
# EPI-Hi LET1 官方/用户确认的能量区间
# ============================================================
# 单位：MeV/nuc
# 每一项是 [E_low, E_high]。
# differential flux 的积分必须使用：sum(J_i * Delta_E_i)。

FE_E_LOW = np.array([
    0.7, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.4,
    2.8, 3.4, 4.0, 4.8, 5.7, 6.7, 8.0, 9.5,
    11.3, 13.5, 16.0, 19.0, 22.6, 26.9, 32.0,
    38.1, 45.3, 53.8, 64.0, 76.1, 90.5, 107.6,
    128.0, 152.2
], dtype=float)

FE_E_HIGH = np.array([
    0.8, 1.0, 1.2, 1.4, 1.7, 2.0, 2.4, 2.8,
    3.4, 4.0, 4.8, 5.7, 6.7, 8.0, 9.5, 11.3,
    13.5, 16.0, 19.0, 22.6, 26.9, 32.0, 38.1,
    45.3, 53.8, 64.0, 76.1, 90.5, 107.6, 128.0,
    152.2, 181.0
], dtype=float)

O_E_LOW = np.array([
    1.0, 1.2, 1.4, 1.7, 2.0, 2.4, 2.8, 3.4,
    4.0, 4.8, 5.7, 6.7, 8.0, 9.5, 11.3, 13.5,
    16.0, 19.0, 22.6, 26.9, 32.0, 38.1, 45.3,
    53.8, 64.0, 76.1, 90.5, 107.6
], dtype=float)

O_E_HIGH = np.array([
    1.2, 1.4, 1.7, 2.0, 2.4, 2.8, 3.4, 4.0,
    4.8, 5.7, 6.7, 8.0, 9.5, 11.3, 13.5, 16.0,
    19.0, 22.6, 26.9, 32.0, 38.1, 45.3, 53.8,
    64.0, 76.1, 90.5, 107.6, 128.0
], dtype=float)

FE_DE = FE_E_HIGH - FE_E_LOW
O_DE = O_E_HIGH - O_E_LOW

# 用于绘图和速度计算的能量中心。
# 这里采用每个区间的算术中心；积分权重严格使用上面的 Delta E。
FE_E_CENTER = 0.5 * (FE_E_LOW + FE_E_HIGH)
O_E_CENTER = 0.5 * (O_E_LOW + O_E_HIGH)

# 共同范围：Fe 第 3--30 道，O 第 1--28 道。
COMMON_FE_SLICE = slice(2, 30)
COMMON_O_SLICE = slice(0, 28)


def validate_user_energy_bins(epihi):
    """检查 CDF 中的能道数量，并建立用户确认的共同能量网格。"""
    n_o = epihi["O_A"].shape[1]
    n_fe = epihi["Fe_A"].shape[1]

    if n_o != 28:
        raise ValueError(
            f"当前 O flux 有 {n_o} 个能道，但用户给出的 LET1 O 能道应为 28 个。"
        )
    if n_fe != 32:
        raise ValueError(
            f"当前 Fe flux 有 {n_fe} 个能道，但用户给出的 LET1 Fe 能道应为 32 个。"
        )

    # 如果 CDF 里也提供了 energy center，打印出来用于核对，但不拿它们
    # 推导 Delta E；本程序的 Delta E 以用户给出的能量区间为准。
    print("\\n================ USER ENERGY BIN CHECK ================")
    print(f"O : {len(O_E_LOW)} bins, {O_E_LOW[0]:.1f}-{O_E_HIGH[-1]:.1f} MeV/nuc")
    print(f"Fe: {len(FE_E_LOW)} bins, {FE_E_LOW[0]:.1f}-{FE_E_HIGH[-1]:.1f} MeV/nuc")
    print(f"COMMON RANGE: {COMMON_E_MIN:.1f}-{COMMON_E_MAX:.1f} MeV/nuc")
    print("O common bins : 1--28")
    print("Fe common bins: 3--30")
    print("========================================================")

    return {
        "O_energy": O_E_CENTER.copy(),
        "O_dE": O_DE.copy(),
        "Fe_energy": FE_E_CENTER.copy(),
        "Fe_dE": FE_DE.copy(),
    }


def select_common_energy_data(epihi, O_flux, Fe_flux):
    """选择 O/Fe 共同的 1--128 MeV/nuc 能量范围。"""
    grid = validate_user_energy_bins(epihi)

    O_flux_use = np.asarray(O_flux, dtype=float)[:, COMMON_O_SLICE]
    Fe_flux_use = np.asarray(Fe_flux, dtype=float)[:, COMMON_FE_SLICE]

    # 注意：这里返回的 energy / dE 必须与裁剪后的 flux
    # 在第二维严格一一对应。
    # O 原始 28 道，全部属于共同范围 1--128 MeV/nuc。
    # Fe 原始 32 道，必须去掉第 1、2 道（0.7--0.8、0.8--1.0），
    # 因此共同范围只保留第 3--30 道，共 28 道。
    O_energy = grid["O_energy"][COMMON_O_SLICE]
    Fe_energy = grid["Fe_energy"][COMMON_FE_SLICE]
    O_dE = grid["O_dE"][COMMON_O_SLICE]
    Fe_dE = grid["Fe_dE"][COMMON_FE_SLICE]

    # 共同范围内两个物种应有相同的能量边界。
    if not np.allclose(O_E_LOW, FE_E_LOW[2:30]):
        raise RuntimeError("内部能量表错误：O 与 Fe 的共同能量下边界不一致")
    if not np.allclose(O_E_HIGH, FE_E_HIGH[2:30]):
        raise RuntimeError("内部能量表错误：O 与 Fe 的共同能量上边界不一致")

    # 最终维度检查：O/Fe 都应为 28 个共同能道。
    if O_flux_use.shape[1] != len(O_energy) or O_flux_use.shape[1] != len(O_dE):
        raise RuntimeError(
            f"O 共同能量维度错误：flux={O_flux_use.shape}, "
            f"energy={len(O_energy)}, dE={len(O_dE)}"
        )
    if Fe_flux_use.shape[1] != len(Fe_energy) or Fe_flux_use.shape[1] != len(Fe_dE):
        raise RuntimeError(
            f"Fe 共同能量维度错误：flux={Fe_flux_use.shape}, "
            f"energy={len(Fe_energy)}, dE={len(Fe_dE)}"
        )

    return O_energy, O_dE, O_flux_use, Fe_energy, Fe_dE, Fe_flux_use


# ============================================================
# 常数
# ============================================================

FOUR_PI = 4.0 * np.pi
M_PROTON = 1.67262192369e-27       # kg
MEV_TO_J = 1.602176634e-13        # J / MeV
M_TO_CM = 100.0


# ============================================================
# 通用工具
# ============================================================

def as_1d_float(x, name="array"):
    """将数组严格整理成一维 float 数组。"""
    a = np.asarray(x, dtype=float)
    a = np.squeeze(a)
    if a.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，实际 shape={a.shape}")
    return a


def as_time_array(raw_epoch):
    """安全地将 CDF Epoch 转成 tz-naive pandas DatetimeIndex。"""
    try:
        t = pd.to_datetime(raw_epoch, utc=True)
        # utc=True 后一定是 timezone-aware；tz_convert(None) 去掉时区
        if isinstance(t, pd.DatetimeIndex):
            return t.tz_convert(None)
        return pd.DatetimeIndex(t).tz_convert(None)
    except Exception:
        # 兼容少数 CDF 时间对象
        t = pd.to_datetime(np.asarray(raw_epoch))
        if isinstance(t, pd.DatetimeIndex) and t.tz is not None:
            t = t.tz_convert(None)
        return pd.DatetimeIndex(t)


def make_date_list(start_time, end_time):
    start = pd.Timestamp(start_time)
    end = pd.Timestamp(end_time)
    return pd.date_range(start.normalize(), end.normalize(), freq="1D")


def find_cdf_file(root, prefix, date):
    root = Path(root)
    date_string = pd.Timestamp(date).strftime("%Y%m%d")

    patterns = [
        root / f"{prefix}_{date_string}_*.cdf",
        root / f"{prefix}_{date_string}.cdf",
        root / "**" / f"{prefix}_{date_string}_*.cdf",
        root / "**" / f"{prefix}_{date_string}.cdf",
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(str(pattern), recursive=True))

    files = sorted(set(files))
    return files[0] if files else None


# ============================================================
# 通用 CDF 读取
# ============================================================

def cdfdataread(cdf_file, variables=None):
    """通用 CDF 读取函数；保留为项目统一 CDF 读取入口。"""
    cdf_file = Path(cdf_file)
    if not cdf_file.exists():
        raise FileNotFoundError(f"CDF 文件不存在：\n{cdf_file}")

    result = {}
    with pycdf.CDF(str(cdf_file)) as cdf:
        if variables is None:
            variables = list(cdf.keys())

        for var in variables:
            if var not in cdf:
                print(f"[WARNING] {var} 不存在")
                continue
            try:
                result[var] = np.asarray(cdf[var][...])
            except Exception as exc:
                print(f"[WARNING] 读取 {var} 失败：{exc}")

    return result


def read_cdf_variable_clean(cdf, variable_name):
    """读取 CDF 数值变量并按 FILLVAL/VALIDMIN/VALIDMAX 清理。"""
    var = cdf[variable_name]
    data = np.asarray(var[...], dtype=float)
    attrs = var.attrs

    fillval = attrs.get("FILLVAL", None)
    if fillval is not None:
        try:
            fv = float(np.asarray(fillval).squeeze())
            data[data == fv] = np.nan
        except Exception:
            pass

    validmin = attrs.get("VALIDMIN", None)
    if validmin is not None:
        try:
            vm = float(np.asarray(validmin).squeeze())
            data[data < vm] = np.nan
        except Exception:
            pass

    validmax = attrs.get("VALIDMAX", None)
    if validmax is not None:
        try:
            vm = float(np.asarray(validmax).squeeze())
            data[data > vm] = np.nan
        except Exception:
            pass

    data[~np.isfinite(data)] = np.nan
    return data


def inspect_cdf(cdf_file):
    """打印 CDF 变量、shape 和关键属性。"""
    print("\n" + "=" * 90)
    print("CDF INSPECTION")
    print("=" * 90)
    print(cdf_file)

    with pycdf.CDF(str(cdf_file)) as cdf:
        for name in cdf.keys():
            try:
                var = cdf[name]
                print(f"\n{name}")
                print(f"  shape = {var.shape}")
                for attr in [
                    "CATDESC", "FIELDNAM", "UNITS", "DEPEND_0", "DEPEND_1",
                    "FILLVAL", "VALIDMIN", "VALIDMAX",
                    "DELTA_PLUS_VAR", "DELTA_MINUS_VAR", "VAR_NOTES"
                ]:
                    if attr in var.attrs:
                        print(f"  {attr}: {var.attrs[attr]}")
            except Exception as exc:
                print(f"{name}: {exc}")

    print("=" * 90)


# ============================================================
# 数组检查
# ============================================================

def print_array_check(name, arr):
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    nonzero = finite & (arr != 0)

    print(f"\n{name}")
    print(f"  shape      : {arr.shape}")
    print(f"  ndim       : {arr.ndim}")
    print(f"  size       : {arr.size}")
    print(f"  finite     : {finite.sum()} / {arr.size}")
    print(f"  non-zero   : {nonzero.sum()} / {arr.size}")

    if finite.any():
        vals = arr[finite]
        print(f"  min        : {np.min(vals)}")
        print(f"  max        : {np.max(vals)}")
        print(f"  median     : {np.median(vals)}")
        print(f"  first vals : {vals[:10]}")
    else:
        print("  min/max    : no finite values")


def standardize_time_axis(arr, n_time, name):
    """确保二维数据为 time x energy。"""
    arr = np.asarray(arr, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"{name} 必须为二维数组，实际 shape={arr.shape}")

    if arr.shape[0] == n_time:
        return arr

    if arr.shape[1] == n_time:
        return arr.T

    raise ValueError(
        f"{name} 无法识别时间轴：Epoch={n_time}, shape={arr.shape}"
    )


# ============================================================
# EPI-Hi LET1
# ============================================================

def read_epihi_let1(root, start_time, end_time):
    """
    读取 EPI-Hi LET1 60-s：
        Epoch
        A_O_Flux / B_O_Flux
        A_Fe_Flux / B_Fe_Flux
        O_ENERGY / O_ENERGY_DELTAPLUS / O_ENERGY_DELTAMINUS
        Fe_ENERGY / Fe_ENERGY_DELTAPLUS / Fe_ENERGY_DELTAMINUS

    返回字典，二维 flux 永远保持 time x energy。
    """
    dates = make_date_list(start_time, end_time)

    all_epoch = []
    all_he_a = []
    all_he_b = []
    all_o_a = []
    all_o_b = []
    all_fe_a = []
    all_fe_b = []

    He_energy = None
    He_dE_plus = None
    He_dE_minus = None
    O_energy = None
    O_dE_plus = None
    O_dE_minus = None
    Fe_energy = None
    Fe_dE_plus = None
    Fe_dE_minus = None

    first_file = None

    for date in dates:
        date_string = date.strftime("%Y%m%d")
        file = find_cdf_file(
            root,
            "psp_isois-epihi_l2-let1-rates60",
            date
        )

        if file is None:
            print(f"[WARNING] 找不到 EPI-Hi 文件：{date_string}")
            continue

        if first_file is None:
            first_file = file

        print("\n" + "=" * 90)
        print(f"[EPIHI] {file}")
        print("=" * 90)

        with pycdf.CDF(file) as cdf:
            if "Epoch" not in cdf:
                raise KeyError("EPI-Hi CDF 中找不到 Epoch")

            # ----------------------------------------------------
            # Epoch
            # ----------------------------------------------------
            epoch = as_time_array(cdf["Epoch"][...])
            n_time = len(epoch)

            # ----------------------------------------------------
            # 官方 energy 与 energy-bin width 信息
            # ----------------------------------------------------
            required = [
                "O_ENERGY",
                "O_ENERGY_DELTAPLUS",
                "O_ENERGY_DELTAMINUS",
                "Fe_ENERGY",
                "Fe_ENERGY_DELTAPLUS",
                "Fe_ENERGY_DELTAMINUS",
            ]
            missing = [x for x in required if x not in cdf]
            if missing:
                raise KeyError(f"EPI-Hi 缺少官方 energy 变量：{missing}")

            current_He_energy = as_1d_float(cdf["He_ENERGY"][...], "He_ENERGY")
            current_He_plus = as_1d_float(cdf["He_ENERGY_DELTAPLUS"][...], "He_ENERGY_DELTAPLUS")
            current_He_minus = as_1d_float(cdf["He_ENERGY_DELTAMINUS"][...], "He_ENERGY_DELTAMINUS")

            current_O_energy = as_1d_float(cdf["O_ENERGY"][...], "O_ENERGY")
            current_O_plus = as_1d_float(
                cdf["O_ENERGY_DELTAPLUS"][...],
                "O_ENERGY_DELTAPLUS"
            )
            current_O_minus = as_1d_float(
                cdf["O_ENERGY_DELTAMINUS"][...],
                "O_ENERGY_DELTAMINUS"
            )

            current_Fe_energy = as_1d_float(
                cdf["Fe_ENERGY"][...], "Fe_ENERGY"
            )
            current_Fe_plus = as_1d_float(
                cdf["Fe_ENERGY_DELTAPLUS"][...],
                "Fe_ENERGY_DELTAPLUS"
            )
            current_Fe_minus = as_1d_float(
                cdf["Fe_ENERGY_DELTAMINUS"][...],
                "Fe_ENERGY_DELTAMINUS"
            )

            if O_energy is None:
                He_energy = current_He_energy
                He_dE_plus = current_He_plus
                He_dE_minus = current_He_minus
                O_energy = current_O_energy
                O_dE_plus = current_O_plus
                O_dE_minus = current_O_minus
                Fe_energy = current_Fe_energy
                Fe_dE_plus = current_Fe_plus
                Fe_dE_minus = current_Fe_minus
            else:
                # 多日期数据时确认能量网格没有改变
                checks = [
                    (O_energy, current_O_energy, "O_ENERGY"),
                    (O_dE_plus, current_O_plus, "O_ENERGY_DELTAPLUS"),
                    (O_dE_minus, current_O_minus, "O_ENERGY_DELTAMINUS"),
                    (Fe_energy, current_Fe_energy, "Fe_ENERGY"),
                    (Fe_dE_plus, current_Fe_plus, "Fe_ENERGY_DELTAPLUS"),
                    (Fe_dE_minus, current_Fe_minus, "Fe_ENERGY_DELTAMINUS"),
                ]
                for old, new, name in checks:
                    if old.shape != new.shape or not np.allclose(old, new, equal_nan=True):
                        raise ValueError(f"不同日期的 {name} 能量网格发生变化")

            # ----------------------------------------------------
            # Flux：只使用 *_Flux，不使用 A_O / A_Fe rate
            # ----------------------------------------------------
            He_A = read_cdf_variable_clean(cdf, "A_He_Flux")
            He_B = read_cdf_variable_clean(cdf, "B_He_Flux")
            O_A = read_cdf_variable_clean(cdf, "A_O_Flux")
            O_B = read_cdf_variable_clean(cdf, "B_O_Flux")
            Fe_A = read_cdf_variable_clean(cdf, "A_Fe_Flux")
            Fe_B = read_cdf_variable_clean(cdf, "B_Fe_Flux")

            He_A = standardize_time_axis(He_A, n_time, "A_He_Flux")
            He_B = standardize_time_axis(He_B, n_time, "B_He_Flux")
            O_A = standardize_time_axis(O_A, n_time, "A_O_Flux")
            O_B = standardize_time_axis(O_B, n_time, "B_O_Flux")
            Fe_A = standardize_time_axis(Fe_A, n_time, "A_Fe_Flux")
            Fe_B = standardize_time_axis(Fe_B, n_time, "B_Fe_Flux")

            if He_A.shape[1] != len(He_energy) or He_B.shape[1] != len(He_energy):
                raise ValueError(
                    f"He energy/flux 维度不一致：energy={len(He_energy)}, A={He_A.shape}, B={He_B.shape}"
                )

            if O_A.shape[1] != len(O_energy) or O_B.shape[1] != len(O_energy):
                raise ValueError(
                    f"O energy/flux 维度不一致：energy={len(O_energy)}, "
                    f"A={O_A.shape}, B={O_B.shape}"
                )

            if Fe_A.shape[1] != len(Fe_energy) or Fe_B.shape[1] != len(Fe_energy):
                raise ValueError(
                    f"Fe energy/flux 维度不一致：energy={len(Fe_energy)}, "
                    f"A={Fe_A.shape}, B={Fe_B.shape}"
                )

            # ----------------------------------------------------
            # 原始数据检查
            # ----------------------------------------------------
            print_array_check("A_He_Flux", He_A)
            print_array_check("B_He_Flux", He_B)
            print_array_check("A_O_Flux", O_A)
            print_array_check("B_O_Flux", O_B)
            print_array_check("A_Fe_Flux", Fe_A)
            print_array_check("B_Fe_Flux", Fe_B)

            # ----------------------------------------------------
            # 时间筛选
            # ----------------------------------------------------
            start = pd.Timestamp(start_time)
            end = pd.Timestamp(end_time)
            mask = (epoch >= start) & (epoch <= end)

            if not np.any(mask):
                print(f"[WARNING] {date_string} 在指定时间范围内没有数据")
                continue

            epoch = epoch[mask]
            He_A = He_A[mask]
            He_B = He_B[mask]
            O_A = O_A[mask]
            O_B = O_B[mask]
            Fe_A = Fe_A[mask]
            Fe_B = Fe_B[mask]

            all_epoch.append(epoch)
            all_he_a.append(He_A)
            all_he_b.append(He_B)
            all_o_a.append(O_A)
            all_o_b.append(O_B)
            all_fe_a.append(Fe_A)
            all_fe_b.append(Fe_B)

    if not all_epoch:
        raise RuntimeError("没有读取到任何 EPI-Hi LET1 数据")

    epoch = np.concatenate(all_epoch, axis=0)
    He_A = np.concatenate(all_he_a, axis=0)
    He_B = np.concatenate(all_he_b, axis=0)
    O_A = np.concatenate(all_o_a, axis=0)
    O_B = np.concatenate(all_o_b, axis=0)
    Fe_A = np.concatenate(all_fe_a, axis=0)
    Fe_B = np.concatenate(all_fe_b, axis=0)

    # ------------------------------------------------------------
    # 按时间排序
    # ------------------------------------------------------------
    order = np.argsort(np.asarray(epoch, dtype="datetime64[ns]"))
    epoch = epoch[order]
    He_A = He_A[order]
    He_B = He_B[order]
    O_A = O_A[order]
    O_B = O_B[order]
    Fe_A = Fe_A[order]
    Fe_B = Fe_B[order]

    print("\n" + "=" * 90)
    print("EPI-Hi FINAL ARRAY")
    print("=" * 90)
    print("Epoch :", epoch.shape)
    print("He_A  :", He_A.shape)
    print("He_B  :", He_B.shape)
    print("O_A   :", O_A.shape)
    print("O_B   :", O_B.shape)
    print("Fe_A  :", Fe_A.shape)
    print("Fe_B  :", Fe_B.shape)
    print("O energy  :", O_energy.shape, O_energy)
    print("Fe energy :", Fe_energy.shape, Fe_energy)
    print("=" * 90)

    return {
        "Epoch": epoch,
        "He_A": He_A,
        "He_B": He_B,
        "He_energy": He_energy,
        "He_dE_plus": He_dE_plus,
        "He_dE_minus": He_dE_minus,
        "O_A": O_A,
        "O_B": O_B,
        "Fe_A": Fe_A,
        "Fe_B": Fe_B,
        "O_energy": O_energy,
        "O_dE_plus": O_dE_plus,
        "O_dE_minus": O_dE_minus,
        "Fe_energy": Fe_energy,
        "Fe_dE_plus": Fe_dE_plus,
        "Fe_dE_minus": Fe_dE_minus,
        "first_file": first_file,
    }


# ============================================================
# A/B 合并
# ============================================================

def merge_ab_flux(A, B, method="mean"):
    """
    合并 EPI-Hi A/B flux。

    注意：CDF 中 VALIDMIN=0，因此 0 是合法数值，不能当作 NaN。
    这里只把 NaN 视为无效。
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.shape != B.shape:
        raise ValueError(f"A/B shape 不一致：A={A.shape}, B={B.shape}")

    if method not in {"mean", "sum"}:
        raise ValueError("SIDE_METHOD 只能是 'mean' 或 'sum'")

    A_valid = np.isfinite(A)
    B_valid = np.isfinite(B)

    out = np.full(A.shape, np.nan, dtype=float)

    both = A_valid & B_valid
    only_A = A_valid & ~B_valid
    only_B = ~A_valid & B_valid

    if method == "mean":
        out[both] = (A[both] + B[both]) / 2.0
    else:
        out[both] = A[both] + B[both]

    out[only_A] = A[only_A]
    out[only_B] = B[only_B]

    return out


# ============================================================
# 原始/合并 flux 检查
# ============================================================

def check_merged_flux(name, flux):
    print_array_check(name, flux)

    positive = np.isfinite(flux) & (flux > 0)
    print(f"  positive  : {positive.sum()} / {flux.size}")

    if flux.ndim == 2:
        print(f"  time rows with positive flux: {np.any(positive, axis=1).sum()} / {flux.shape[0]}")
        print(f"  energy bins with positive flux: {np.any(positive, axis=0).sum()} / {flux.shape[1]}")


# ============================================================
# 能量范围
# ============================================================

def select_energy_range(energy, dE_plus, dE_minus, flux, energy_range, species):
    energy = as_1d_float(energy, f"{species}_energy")
    dE_plus = as_1d_float(dE_plus, f"{species}_dE_plus")
    dE_minus = as_1d_float(dE_minus, f"{species}_dE_minus")
    flux = np.asarray(flux, dtype=float)

    if not (len(energy) == len(dE_plus) == len(dE_minus) == flux.shape[1]):
        raise ValueError(
            f"{species} energy/dE/flux 维度不一致："
            f"energy={energy.shape}, plus={dE_plus.shape}, "
            f"minus={dE_minus.shape}, flux={flux.shape}"
        )

    if energy_range is None:
        return energy, dE_plus, dE_minus, flux

    e_min, e_max = energy_range
    mask = (energy >= e_min) & (energy <= e_max)

    if not np.any(mask):
        raise ValueError(
            f"{species} 指定能量范围 {energy_range} 内没有能道"
        )

    return (
        energy[mask],
        dE_plus[mask],
        dE_minus[mask],
        flux[:, mask]
    )


# ============================================================
# Flux -> density
# ============================================================

def flux_to_density(flux, energy_mev_nuc, dE_plus, dE_minus):
    """
    differential directional flux -> density。

    J 单位：cm^-2 sr^-1 s^-1 (MeV/nuc)^-1
    E 单位：MeV/nuc
    输出：cm^-3

    n = 4*pi * sum[J(E_i)/v_i * DeltaE_i]
    """
    flux = np.asarray(flux, dtype=float)
    energy = as_1d_float(energy_mev_nuc, "energy")
    dE_plus = as_1d_float(dE_plus, "dE_plus")
    dE_minus = as_1d_float(dE_minus, "dE_minus")

    if flux.ndim != 2:
        raise ValueError(f"flux 必须是二维 time x energy，实际 {flux.shape}")

    if flux.shape[1] != len(energy):
        raise ValueError(
            f"flux 第二维 {flux.shape[1]} != energy 数量 {len(energy)}"
        )

    # 官方 energy-bin 宽度
    dE = dE_plus + dE_minus

    # 防止异常 bin 进入计算
    valid_energy = (
        np.isfinite(energy) &
        np.isfinite(dE) &
        (energy > 0) &
        (dE > 0)
    )

    if not np.any(valid_energy):
        raise ValueError("没有有效的 energy bin")

    # MeV/nuc -> J/nuc
    E_J_per_nuc = energy[valid_energy] * MEV_TO_J

    # E_nuc = 1/2 mp v^2
    velocity_cm_s = (
        np.sqrt(2.0 * E_J_per_nuc / M_PROTON) * M_TO_CM
    )

    flux_use = flux[:, valid_energy]
    dE_use = dE[valid_energy]

    valid_flux = np.isfinite(flux_use) & (flux_use >= 0)

    # 用广播直接计算 J(E)/v(E)，无效 flux 置零
    integrand = np.where(
        valid_flux,
        flux_use / velocity_cm_s[np.newaxis, :],
        0.0
    )

    integrand *= dE_use[np.newaxis, :]

    density = FOUR_PI * np.sum(integrand, axis=1)

    n_valid = np.sum(valid_flux, axis=1)
    density[n_valid == 0] = np.nan

    return density


# ============================================================
# Energy-integrated flux
# ============================================================

def integrate_flux(flux, dE_plus, dE_minus):
    """对 differential flux 做 energy integration。"""
    flux = np.asarray(flux, dtype=float)
    dE = as_1d_float(dE_plus, "dE_plus") + as_1d_float(dE_minus, "dE_minus")

    valid = np.isfinite(flux)
    values = np.where(valid, flux, 0.0) * dE[np.newaxis, :]
    result = np.sum(values, axis=1)

    n_valid = np.sum(valid, axis=1)
    result[n_valid == 0] = np.nan

    return result


# ============================================================
# 原始 flux 时间序列图
# ============================================================

def plot_epihi_raw_flux(epihi, save_path=None):
    """检查原始/合并后的 energy-integrated flux。"""
    epoch = epihi["Epoch"]

    O = merge_ab_flux(epihi["O_A"], epihi["O_B"], SIDE_METHOD)
    Fe = merge_ab_flux(epihi["Fe_A"], epihi["Fe_B"], SIDE_METHOD)

    O_total = integrate_flux(O, epihi["O_dE_plus"], epihi["O_dE_minus"])
    Fe_total = integrate_flux(Fe, epihi["Fe_dE_plus"], epihi["Fe_dE_minus"])

    print("\n================ FLUX PLOT CHECK ================")
    print_array_check("O integrated flux", O_total)
    print_array_check("Fe integrated flux", Fe_total)
    print("=================================================\n")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(epoch, O_total, linewidth=1.0)
    axes[0].set_ylabel("O integrated flux")
    axes[0].set_title("PSP EPI-Hi LET1 Oxygen Energy-integrated Flux")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epoch, Fe_total, linewidth=1.0)
    axes[1].set_ylabel("Fe integrated flux")
    axes[1].set_xlabel("Time")
    axes[1].set_title("PSP EPI-Hi LET1 Iron Energy-integrated Flux")
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE FIG] {save_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


# ============================================================
# 二维 energy-time flux 图
# ============================================================

def _positive_norm_data(flux):
    positive = flux[np.isfinite(flux) & (flux > 0)]
    if positive.size == 0:
        return None, None
    return np.nanpercentile(positive, 5), np.nanpercentile(positive, 99.5)


def plot_epihi_flux_spectrogram(epihi, save_path=None):
    """绘制 O / Fe 二维 energy-time differential flux。"""
    epoch = epihi["Epoch"]
    O = merge_ab_flux(epihi["O_A"], epihi["O_B"], SIDE_METHOD)
    Fe = merge_ab_flux(epihi["Fe_A"], epihi["Fe_B"], SIDE_METHOD)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    panels = [
        (axes[0], O, epihi["O_energy"], "O", "O differential flux"),
        (axes[1], Fe, epihi["Fe_energy"], "Fe", "Fe differential flux"),
    ]

    for ax, flux, energy, species, title in panels:
        positive = np.where(np.isfinite(flux) & (flux > 0), flux, np.nan)
        vmin, vmax = _positive_norm_data(positive)

        if vmin is None or vmax is None or vmax <= vmin:
            ax.text(
                0.5, 0.5,
                f"{species}: no positive flux",
                transform=ax.transAxes,
                ha="center", va="center"
            )
            continue

        # pcolormesh 要求 x/y 为一维，C 为 time x energy
        mesh = ax.pcolormesh(
            epoch,
            energy,
            positive.T,
            shading="auto",
            norm=LogNorm(vmin=vmin, vmax=vmax)
        )
        ax.set_yscale("log")
        ax.set_ylabel("Energy [MeV/nuc]")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(mesh, ax=ax, pad=0.01, label="Flux")

    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE FIG] {save_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


# ============================================================
# 平均能谱
# ============================================================

def plot_energy_spectra(epihi, save_path=None):
    """绘制时间平均 O / Fe differential flux energy spectrum。"""
    O = merge_ab_flux(epihi["O_A"], epihi["O_B"], SIDE_METHOD)
    Fe = merge_ab_flux(epihi["Fe_A"], epihi["Fe_B"], SIDE_METHOD)

    with np.errstate(invalid="ignore"):
        O_mean = np.nanmean(np.where(O > 0, O, np.nan), axis=0)
        Fe_mean = np.nanmean(np.where(Fe > 0, Fe, np.nan), axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))

    O_mask = np.isfinite(O_mean) & (O_mean > 0)
    Fe_mask = np.isfinite(Fe_mean) & (Fe_mean > 0)

    if np.any(O_mask):
        ax.loglog(
            epihi["O_energy"][O_mask], O_mean[O_mask],
            marker="o", markersize=3, linewidth=1, label="O"
        )

    if np.any(Fe_mask):
        ax.loglog(
            epihi["Fe_energy"][Fe_mask], Fe_mean[Fe_mask],
            marker="o", markersize=3, linewidth=1, label="Fe"
        )

    ax.set_xlabel("Energy [MeV/nuc]")
    ax.set_ylabel(
        r"Differential flux [cm$^{-2}$ sr$^{-1}$ s$^{-1}$ "
        r"(MeV/nuc)$^{-1}$]"
    )
    ax.set_title("PSP EPI-Hi LET1 O / Fe Energy Spectra")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE FIG] {save_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# ============================================================
# 共同能量范围的积分 flux 图
# ============================================================

def plot_epihi_raw_flux_common(epoch, O_total, Fe_total, save_path=None):
    """绘制 1--128 MeV/nuc 共同能量范围内的 energy-integrated flux。"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    axes[0].plot(epoch, O_total, linewidth=1.0)
    axes[0].set_ylabel("O integrated flux")
    axes[0].set_title("PSP EPI-Hi LET1 Oxygen Energy-integrated Flux (1--128 MeV/nuc)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epoch, Fe_total, linewidth=1.0)
    axes[1].set_ylabel("Fe integrated flux")
    axes[1].set_xlabel("Time")
    axes[1].set_title("PSP EPI-Hi LET1 Iron Energy-integrated Flux (1--128 MeV/nuc)")
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE FIG] {save_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


# ============================================================
# 共同能量范围的平均 differential flux 能谱
# ============================================================

def plot_energy_spectra_common(
    epoch,
    O_flux,
    O_energy,
    Fe_flux,
    Fe_energy,
    save_path=None
):
    """绘制 1--128 MeV/nuc 内 O / Fe 的时间平均 differential flux。"""
    with np.errstate(invalid="ignore"):
        O_mean = np.nanmean(np.where(O_flux > 0, O_flux, np.nan), axis=0)
        Fe_mean = np.nanmean(np.where(Fe_flux > 0, Fe_flux, np.nan), axis=0)

    fig, ax = plt.subplots(figsize=(9, 6))

    O_mask = np.isfinite(O_mean) & (O_mean > 0)
    Fe_mask = np.isfinite(Fe_mean) & (Fe_mean > 0)

    if np.any(O_mask):
        ax.loglog(
            O_energy[O_mask], O_mean[O_mask],
            marker="o", markersize=3, linewidth=1, label="O"
        )

    if np.any(Fe_mask):
        ax.loglog(
            Fe_energy[Fe_mask], Fe_mean[Fe_mask],
            marker="o", markersize=3, linewidth=1, label="Fe"
        )

    ax.set_xlabel("Energy [MeV/nuc]")
    ax.set_ylabel(
        r"Differential flux [cm$^{-2}$ sr$^{-1}$ s$^{-1}$ "
        r"(MeV/nuc)$^{-1}$]"
    )
    ax.set_title("PSP EPI-Hi LET1 O / Fe Differential Flux (1--128 MeV/nuc)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE FIG] {save_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# ============================================================
# PSP MAG
# ============================================================

def read_psp_mag(root, start_time, end_time):
    dates = make_date_list(start_time, end_time)

    all_epoch = []
    all_B = []

    for date in dates:
        date_string = date.strftime("%Y%m%d")
        file = find_cdf_file(
            root,
            "psp_fld_l2_mag_rtn_1min",
            date
        )

        if file is None:
            print(f"[WARNING] 找不到 MAG 文件：{date_string}")
            continue

        print(f"\n[MAG] {file}")

        with pycdf.CDF(file) as cdf:
            epoch_name = None
            for name in [
                "epoch_mag_RTN_1min",
                "Epoch",
                "epoch",
                "EPOCH"
            ]:
                if name in cdf:
                    epoch_name = name
                    break

            if epoch_name is None:
                raise KeyError("MAG CDF 中没有 Epoch")

            epoch = as_time_array(cdf[epoch_name][...])

            B_name = None
            for name in [
                "psp_fld_l2_mag_RTN_1min",
                "psp_fld_l2_mag_RTN_1MIN"
            ]:
                if name in cdf:
                    B_name = name
                    break

            if B_name is None:
                raise KeyError("MAG CDF 中找不到 RTN 磁场变量")

            B = read_cdf_variable_clean(cdf, B_name)

            if B.ndim != 2:
                raise ValueError(f"MAG 数据维度异常：{B.shape}")

            if B.shape[1] != 3:
                if B.shape[0] == 3:
                    B = B.T
                else:
                    raise ValueError(f"无法识别 RTN 三分量：{B.shape}")

            if len(epoch) != B.shape[0]:
                raise ValueError(
                    f"MAG Epoch/B 长度不一致：Epoch={len(epoch)}, B={B.shape}"
                )

            start = pd.Timestamp(start_time)
            end = pd.Timestamp(end_time)
            mask = (epoch >= start) & (epoch <= end)

            if np.any(mask):
                all_epoch.append(epoch[mask])
                all_B.append(B[mask])

    if not all_epoch:
        raise RuntimeError("没有读取到 PSP MAG 数据")

    epoch = np.concatenate(all_epoch)
    B = np.concatenate(all_B, axis=0)

    order = np.argsort(np.asarray(epoch, dtype="datetime64[ns]"))
    epoch = epoch[order]
    B = B[order]

    Br = B[:, 0]
    Bt = B[:, 1]
    Bn = B[:, 2]
    Bmag = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

    result = pd.DataFrame({
        "Epoch": epoch,
        "Br": Br,
        "Bt": Bt,
        "Bn": Bn,
        "Bmag": Bmag,
    })

    result = (
        result
        .sort_values("Epoch")
        .drop_duplicates("Epoch")
        .reset_index(drop=True)
    )

    return result


# ============================================================
# MAG + EPI-Hi 合并
# ============================================================

def merge_mag_epihi(mag_df, epihi_df):
    mag_df = mag_df.sort_values("Epoch").reset_index(drop=True)
    epihi_df = epihi_df.sort_values("Epoch").reset_index(drop=True)

    result = pd.merge_asof(
        mag_df,
        epihi_df,
        on="Epoch",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=40)
    )

    return result


# ============================================================
# Fe/O
# ============================================================

def calculate_fe_o_ratio(df):
    """计算共同 1--128 MeV/nuc 范围内的 Fe/O。"""
    flux_ratio = (
        df["Fe_flux_integrated"] / df["O_flux_integrated"]
    )
    flux_ratio = flux_ratio.where(
        np.isfinite(flux_ratio) & (flux_ratio > 0),
        np.nan
    )
    df["Fe_O_ratio"] = flux_ratio

    # 同时保留 density ratio，方便后续物理分析。
    density_ratio = (
        df["Fe_density"] / df["O_density"]
    )
    density_ratio = density_ratio.where(
        np.isfinite(density_ratio) & (density_ratio > 0),
        np.nan
    )
    df["Fe_O_density_ratio"] = density_ratio

    return df


# ============================================================
# Data quality check
# ============================================================

def check_data(df):
    print("\n" + "=" * 90)
    print("DATA QUALITY CHECK")
    print("=" * 90)
    print("Time:", df["Epoch"].min(), "→", df["Epoch"].max())
    print("Points:", len(df))

    for col in [
        "Bmag",
        "O_flux_integrated",
        "Fe_flux_integrated",
        "O_density",
        "Fe_density",
        "Fe_O_ratio",
        "Fe_O_density_ratio",
    ]:
        if col not in df:
            continue

        arr = df[col].to_numpy(dtype=float)
        finite = np.isfinite(arr)
        positive = finite & (arr > 0)

        print(f"\n{col}")
        print(f"  finite  : {finite.sum()} / {len(arr)}")
        print(f"  positive: {positive.sum()} / {len(arr)}")
        if finite.any():
            print(f"  min     : {np.nanmin(arr)}")
            print(f"  max     : {np.nanmax(arr)}")
            print(f"  median  : {np.nanmedian(arr)}")

    print("\nMissing values:")
    print(df.isna().sum())
    print("=" * 90)


# ============================================================
# 最终时间序列图
# ============================================================

def plot_psp_data(df, save_path=None):
    fig, axes = plt.subplots(
        5, 1,
        figsize=(15, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2, 1, 1, 1]}
    )
    fontsize=30

    # 1. Bmag
    axes[0].plot(df["Epoch"], df["Bmag"], linewidth=1.0)
    axes[0].set_ylabel("|B| [nT]")
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', labelsize=16)
    axes[0].tick_params(axis='y', labelsize=16)
    # 2. RTN
    axes[1].plot(df["Epoch"], df["Br"], label="Br", linewidth=0.8)
    axes[1].plot(df["Epoch"], df["Bt"], label="Bt", linewidth=0.8)
    axes[1].plot(df["Epoch"], df["Bn"], label="Bn", linewidth=0.8)
    axes[1].set_ylabel("B [nT]", fontsize=fontsize)
    axes[1].legend(loc="upper right",fontsize=fontsize)
    axes[1].grid(True, alpha=0.3)

    # 3. integrated flux
    axes[2].plot(df["Epoch"], df["He_flux_integrated"], label="He", linewidth=0.9)
    axes[2].plot(
        df["Epoch"], df["O_flux_integrated"],
        label="O", linewidth=0.9
    )
    axes[2].plot(
        df["Epoch"], df["Fe_flux_integrated"],
        label="Fe", linewidth=0.9
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Integrated flux",fontsize=fontsize)
    axes[2].legend(loc="upper right", fontsize=fontsize)
    axes[2].grid(True, alpha=0.3)

    # 4. density
    axes[3].plot(df["Epoch"], df["He_density"], label="He", linewidth=0.9)
    axes[3].plot(
        df["Epoch"], df["O_density"],
        label="O", linewidth=0.9
    )
    axes[3].plot(
        df["Epoch"], df["Fe_density"],
        label="Fe", linewidth=0.9
    )
    axes[3].set_yscale("log")
    axes[3].set_ylabel(r"n [cm$^{-3}$]", fontsize=fontsize)
    axes[3].legend(loc="upper right", fontsize=fontsize)
    axes[3].grid(True, alpha=0.3)

    # 5. Fe/O
    axes[4].plot(
        df["Epoch"], df["Fe_O_ratio"],
        linewidth=0.9
    )
    axes[4].set_yscale("log")
    axes[4].set_ylabel("Fe/O (integrated flux)", fontsize=fontsize)
    axes[4].set_xlabel("Time", fontsize=fontsize)
    axes[4].grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%m-%d\n%H:%M"), fontsize=fontsize
    )

    fig.suptitle("PSP MAG + EPI-Hi LET1 O / Fe", fontsize=fontsize)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE FIG] {save_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes



# ============================================================
# ICME 列表读取
# ============================================================

def read_icme_list(list_file, target_satellite="PSP"):
    """
    自动读取 ICME 列表。

    每行前 5 列固定解释为：
        0 ICMEid
        1 Satellite
        2 ShockArrival
        3 ICMEStart
        4 ICMEEnd

    后续列全部忽略。

    文件当前没有表头，因此使用 whitespace 分隔读取。
    """
    list_file = Path(list_file)
    if not list_file.exists():
        raise FileNotFoundError(f"ICME 列表不存在：\n{list_file}")

    records = []
    with list_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                print(f"[WARNING] ICME 列表第 {line_no} 行列数不足，跳过：{line}")
                continue

            icmeid = parts[0]
            satellite = parts[1]
            shock_raw = parts[2]
            body_raw = parts[3]
            end_raw = parts[4]

            if target_satellite and satellite.upper() != target_satellite.upper():
                continue

            try:
                shock = pd.Timestamp(shock_raw)
                body = pd.Timestamp(body_raw)
                end = pd.Timestamp(end_raw)

                # 当前数据读取函数内部使用 tz-naive 时间，因此统一去掉 UTC 时区。
                if shock.tzinfo is not None:
                    shock = shock.tz_convert(None)
                if body.tzinfo is not None:
                    body = body.tz_convert(None)
                if end.tzinfo is not None:
                    end = end.tz_convert(None)
            except Exception as exc:
                print(f"[WARNING] ICME 列表第 {line_no} 行时间解析失败：{exc}")
                continue

            if not (shock <= body <= end):
                print(
                    f"[WARNING] {icmeid} 时间顺序异常："
                    f"shock={shock}, body={body}, end={end}，跳过"
                )
                continue

            records.append({
                "ICMEid": icmeid,
                "Satellite": satellite,
                "ShockArrival": shock,
                "ICMEStart": body,
                "ICMEEnd": end,
                "line_no": line_no,
            })

    if not records:
        raise RuntimeError(
            f"列表 {list_file} 中没有找到卫星 {target_satellite} 的有效 ICME 事件"
        )

    events = pd.DataFrame(records)
    events = events.sort_values("ShockArrival").reset_index(drop=True)

    print("\n" + "=" * 90)
    print("ICME LIST CHECK")
    print("=" * 90)
    print(f"List file       : {list_file}")
    print(f"Target satellite: {target_satellite}")
    print(f"Valid events    : {len(events)}")
    print(events[["ICMEid", "Satellite", "ShockArrival", "ICMEStart", "ICMEEnd"]].to_string(index=False))
    print("=" * 90)

    return events


def _safe_event_name(icmeid):
    """将 ICMEid 转成 Windows 安全的文件名。"""
    bad = '<>:"/\\|?*'
    name = str(icmeid)
    for ch in bad:
        name = name.replace(ch, "_")
    return name


def add_icme_time_markers(axes, shock_time, body_time, end_time):
    """在所有子图上标记 Shock / ICME start / ICME end。"""
    markers = [
        (shock_time, "Shock arrival", "tab:red"),
        (body_time, "ICME start", "tab:green"),
        (end_time, "ICME end", "tab:blue"),
    ]

    # 三个时间可能完全相同，因此逐个绘制；同一时刻重叠是正常的。
    for ax in axes:
        for t, label, color in markers:
            ax.axvline(
                t,
                linestyle="-",
                linewidth=1.2,
                color=color,
                alpha=0.9,
            )

    # 标签只放在第一幅图，避免整张图过于拥挤。
    ax0 = axes[0]
    y0, y1 = ax0.get_ylim()
    positions = [0.92, 0.78, 0.64]
    for (t, label, color), ypos in zip(markers, positions):
        ax0.text(
            t,
            ypos,
            label,
            transform=ax0.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
            color=color,
            backgroundcolor="white",
            alpha=0.8,
        )


def plot_icme_combined(
    df,
    icme_id,
    shock_time,
    body_time,
    end_time,
    save_path=None,
):
    """
    将 PSP MAG 与 EPI-Hi He/O/Fe 数据画在同一张画布。

    面板：
        1. |B|
        2. Br/Bt/Bn
        3. O/Fe energy-integrated flux
        4. O/Fe energetic-ion density estimate
        5. He energetic-ion density estimate  （新增独立子图）
        6. Fe/O integrated-flux ratio

    同时在所有面板标记：
        Shock arrival
        ICME start
        ICME end
    """
    fig, axes = plt.subplots(
        6, 1,
        figsize=(16, 15),                     # 高度稍增
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2, 1, 0.8, 0.8, 1]},
    )
    fontsize=20
    labelsize=14
    # --------------------------------------------------------
    # 1. Magnetic field magnitude
    # --------------------------------------------------------
    axes[0].plot(df["Epoch"], df["Bmag"], linewidth=1.0)
    axes[0].set_ylabel("|B| [nT]",fontsize=fontsize)
    axes[0].set_title(f"{icme_id}   |   PSP MAG + EPI-Hi LET1 He / O / Fe",fontsize=fontsize)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', labelsize=labelsize)
    axes[0].tick_params(axis='y', labelsize=labelsize)
    # --------------------------------------------------------
    # 2. RTN magnetic field
    # --------------------------------------------------------
    axes[1].plot(df["Epoch"], df["Br"], label="Br", linewidth=0.8)
    axes[1].plot(df["Epoch"], df["Bt"], label="Bt", linewidth=0.8)
    axes[1].plot(df["Epoch"], df["Bn"], label="Bn", linewidth=0.8)
    axes[1].set_ylabel("B [nT]",fontsize=fontsize)
    axes[1].legend(loc="upper right", ncol=3,fontsize=fontsize)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', labelsize=labelsize)
    axes[1].tick_params(axis='y', labelsize=labelsize)

    # --------------------------------------------------------
    # 3. O / Fe flux
    # --------------------------------------------------------
    axes[2].plot(
        df["Epoch"], df["O_flux_integrated"],
        label="O", linewidth=0.9
    )
    axes[2].plot(
        df["Epoch"], df["Fe_flux_integrated"],
        label="Fe", linewidth=0.9
    )
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Flux",fontsize=fontsize)
    axes[2].set_title("EPI-Hi LET1 energy-integrated differential flux (1--128 MeV/nuc)",fontsize=fontsize)
    axes[2].legend(loc="upper right",fontsize=fontsize)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='x', labelsize=labelsize)
    axes[2].tick_params(axis='y', labelsize=labelsize)

    # --------------------------------------------------------
    # 4. O / Fe density (不含 He)
    # --------------------------------------------------------
    axes[3].plot(
        df["Epoch"], df["O_density"],
        label="O", linewidth=0.9
    )
    axes[3].plot(
        df["Epoch"], df["Fe_density"],
        label="Fe", linewidth=0.9
    )
    axes[3].set_yscale("log")
    axes[3].set_ylabel(r"n [cm$^{-3}$]",fontsize=fontsize)
    axes[3].set_title("EPI-Hi energetic-ion density estimate: O / Fe",fontsize=fontsize)
    axes[3].legend(loc="upper right", ncol=2,fontsize=fontsize)
    axes[3].grid(True, alpha=0.3)
    axes[3].tick_params(axis='x', labelsize=labelsize)
    axes[3].tick_params(axis='y', labelsize=labelsize)
    ax = axes[3]
    # 取出有效数据（排除 NaN/Inf）
    vals = df["O_density"][np.isfinite(df["O_density"]) & (df["O_density"] > 0)]
    if len(vals) > 0:
        vmin = vals.min() * 0.5
        vmax = vals.max() * 2.0
        ax.set_ylim(vmin, vmax)
    # --------------------------------------------------------
    # 5. He density (独立子图)
    # --------------------------------------------------------
    axes[4].plot(
        df["Epoch"], df["He_density"],
        label="He", linewidth=0.9, color="tab:purple"
    )
    axes[4].set_yscale("log")
    axes[4].set_ylabel(r"n [cm$^{-3}$]",fontsize=fontsize)
    axes[4].set_title("EPI-Hi energetic-ion density estimate: He",fontsize=fontsize)
    axes[4].legend(loc="upper right",fontsize=fontsize)
    axes[4].grid(True, alpha=0.3)
    axes[4].tick_params(axis='x', labelsize=labelsize)
    axes[4].tick_params(axis='y', labelsize=labelsize)
    ax = axes[4]
    # 取出有效数据（排除 NaN/Inf）
    vals = df["He_density"][np.isfinite(df["He_density"]) & (df["He_density"] > 0)]
    if len(vals) > 0:
        vmin = vals.min() * 0.5
        vmax = vals.max() * 2.0
        ax.set_ylim(vmin, vmax)
    # --------------------------------------------------------
    # 6. Fe/O ratio
    # --------------------------------------------------------
    axes[5].plot(
        df["Epoch"], df["Fe_O_ratio"],
        linewidth=0.9
    )
    axes[5].set_yscale("log")
    axes[5].set_ylabel("Fe/O",fontsize=fontsize)
    axes[5].set_xlabel("Time",fontsize=fontsize)
    axes[5].set_title("Fe/O ratio from common 1--128 MeV/nuc integrated flux",fontsize=fontsize)
    axes[5].grid(True, alpha=0.3)
    axes[5].tick_params(axis='x', labelsize=labelsize)
    axes[5].tick_params(axis='y', labelsize=labelsize)
    # --------------------------------------------------------
    # ICME 三个关键时刻
    # --------------------------------------------------------
    add_icme_time_markers(
        axes,
        shock_time,
        body_time,
        end_time,
    )

    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%m-%d\n%H:%M")
    )
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())

    fig.suptitle(
        f"{icme_id}\n"
        f"Shock: {shock_time:%Y-%m-%d %H:%M}   |   "
        f"ICME start: {body_time:%Y-%m-%d %H:%M}   |   "
        f"ICME end: {end_time:%Y-%m-%d %H:%M}",
        fontsize=20,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVE FIG] {save_path}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def process_single_icme_event(event, output_root):
    """处理一个 ICME 事件。"""
    icme_id = event["ICMEid"]
    shock_time = pd.Timestamp(event["ShockArrival"])
    body_time = pd.Timestamp(event["ICMEStart"])
    end_time = pd.Timestamp(event["ICMEEnd"])

    # 数据读取窗口可以在 shock 前/后扩展，但三个事件标记仍使用原始列表时间。
    data_start = shock_time - pd.Timedelta(minutes=PLOT_PRE_MINUTES)
    data_end = end_time + pd.Timedelta(minutes=PLOT_POST_MINUTES)

    event_dir = Path(output_root)
    event_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 90)
    print(f"# PROCESS ICME: {icme_id}")
    print(f"# Satellite   : {event['Satellite']}")
    print(f"# Shock       : {shock_time}")
    print(f"# ICME start  : {body_time}")
    print(f"# ICME end    : {end_time}")
    print(f"# Data window : {data_start} -> {data_end}")
    print("#" * 90)

    # --------------------------------------------------------
    # EPI-Hi
    # --------------------------------------------------------
    epihi = read_epihi_let1(
        EPIHI_ROOT,
        data_start.strftime("%Y-%m-%d %H:%M:%S"),
        data_end.strftime("%Y-%m-%d %H:%M:%S"),
    )

    He_flux_2d = merge_ab_flux(epihi["He_A"], epihi["He_B"], SIDE_METHOD)
    O_flux_2d = merge_ab_flux(epihi["O_A"], epihi["O_B"], SIDE_METHOD)
    Fe_flux_2d = merge_ab_flux(epihi["Fe_A"], epihi["Fe_B"], SIDE_METHOD)

    He_energy, He_dE_plus, He_dE_minus, He_flux_use = select_energy_range(
        epihi["He_energy"], epihi["He_dE_plus"], epihi["He_dE_minus"],
        He_flux_2d, (COMMON_E_MIN, COMMON_E_MAX), "He"
    )

    (
        O_energy, O_dE, O_flux_use,
        Fe_energy, Fe_dE, Fe_flux_use
    ) = select_common_energy_data(
        epihi,
        O_flux_2d,
        Fe_flux_2d,
    )

    He_density = flux_to_density(
        He_flux_use, He_energy, He_dE_plus, He_dE_minus
    )

    He_flux_integrated = integrate_flux(
        He_flux_use, He_dE_plus, He_dE_minus
    )

    O_density = flux_to_density(
        O_flux_use,
        O_energy,
        O_dE,
        np.zeros_like(O_dE),
    )
    Fe_density = flux_to_density(
        Fe_flux_use,
        Fe_energy,
        Fe_dE,
        np.zeros_like(Fe_dE),
    )

    O_flux_integrated = integrate_flux(
        O_flux_use,
        O_dE,
        np.zeros_like(O_dE),
    )
    Fe_flux_integrated = integrate_flux(
        Fe_flux_use,
        Fe_dE,
        np.zeros_like(Fe_dE),
    )

    epihi_df = pd.DataFrame({
        "Epoch": epihi["Epoch"],
        "He_flux_integrated": He_flux_integrated,
        "He_density": He_density,
        "O_flux_integrated": O_flux_integrated,
        "Fe_flux_integrated": Fe_flux_integrated,
        "O_density": O_density,
        "Fe_density": Fe_density,
    })

    # --------------------------------------------------------
    # MAG
    # --------------------------------------------------------
    mag_df = read_psp_mag(
        MAG_ROOT,
        data_start.strftime("%Y-%m-%d %H:%M:%S"),
        data_end.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # --------------------------------------------------------
    # 合并
    # --------------------------------------------------------
    df = merge_mag_epihi(mag_df, epihi_df)
    df = calculate_fe_o_ratio(df)

    # --------------------------------------------------------
    # 输出 CSV
    # --------------------------------------------------------
    csv_path = event_dir / f"{_safe_event_name(icme_id)}_MAG_EPIHi_He_O_Fe.csv"
    saved_csv = save_csv_safely(df, csv_path)

    # --------------------------------------------------------
    # 输出 NPZ：保留二维 EPI-Hi 能谱数据
    # --------------------------------------------------------
    if SAVE_FLUX_NPZ:
        npz_path = event_dir / f"{_safe_event_name(icme_id)}_EPIHi_He_O_Fe_flux.npz"
        np.savez_compressed(
            npz_path,
            Epoch=np.asarray(epihi["Epoch"], dtype="datetime64[ns]"),
            He_flux=He_flux_use,
            He_energy=He_energy,
            He_dE=He_dE_plus + He_dE_minus,
            He_density=He_density,
            He_flux_integrated=He_flux_integrated,
            O_flux=O_flux_use,
            O_energy=O_energy,
            O_dE=O_dE,
            Fe_flux=Fe_flux_use,
            Fe_energy=Fe_energy,
            Fe_dE=Fe_dE,
            O_density=O_density,
            Fe_density=Fe_density,
            O_flux_integrated=O_flux_integrated,
            Fe_flux_integrated=Fe_flux_integrated,
        )
        print(f"[SAVE NPZ] {npz_path}")

    # --------------------------------------------------------
    # 最终合并图：磁场 + 粒子数据 + 三个 ICME 时间点
    # --------------------------------------------------------
    fig_path = event_dir / f"{_safe_event_name(icme_id)}_MAG_EPIHi_He_O_Fe.png"
    plot_icme_combined(
        df,
        icme_id,
        shock_time,
        body_time,
        end_time,
        save_path=fig_path,
    )

    # 事件元数据单独保存，便于后续统计分析。
    meta_path = event_dir / f"{_safe_event_name(icme_id)}_event.txt"
    meta_path.write_text(
        f"ICMEid: {icme_id}\n"
        f"Satellite: {event['Satellite']}\n"
        f"ShockArrival: {shock_time.isoformat()}\n"
        f"ICMEStart: {body_time.isoformat()}\n"
        f"ICMEEnd: {end_time.isoformat()}\n"
        f"DataStart: {data_start.isoformat()}\n"
        f"DataEnd: {data_end.isoformat()}\n"
        f"CommonEnergyRange: {COMMON_E_MIN}-{COMMON_E_MAX} MeV/nuc\n",
        encoding="utf-8",
    )

    return {
        "ICMEid": icme_id,
        "Satellite": event["Satellite"],
        "ShockArrival": shock_time,
        "ICMEStart": body_time,
        "ICMEEnd": end_time,
        "n_points": len(df),
        "csv": str(saved_csv),
        "figure": str(fig_path),
    }


# ============================================================
# 主程序
# ============================================================

def save_csv_safely(df, target_path):
    """保存 CSV；若目标文件被 Excel/其他程序占用，则自动换名保存。"""
    target_path = Path(target_path)

    try:
        df.to_csv(target_path, index=False)
        return target_path
    except PermissionError:
        print("\n[WARNING] 目标 CSV 无法写入，通常是文件正在被 Excel/其他程序占用：")
        print(f"          {target_path}")

        stem = target_path.stem
        suffix = target_path.suffix
        for i in range(1, 1000):
            fallback = target_path.with_name(f"{stem}_new{i}{suffix}")
            try:
                df.to_csv(fallback, index=False)
                print(f"[SAVE CSV - FALLBACK] {fallback}")
                return fallback
            except PermissionError:
                continue

        raise PermissionError(
            "CSV 文件无法保存：目标文件可能被占用，且自动换名也失败。\n"
            "请关闭正在打开该 CSV 的 Excel/编辑器后重新运行。"
        )


def main():
    print("\n" + "#" * 90)
    print("# PSP ICME BATCH PROCESSING")
    print("# MAG + EPI-Hi LET1 O / Fe")
    print("#" * 90)

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 自动读取 ICME 列表
    # --------------------------------------------------------
    events = read_icme_list(
        ICME_LIST_FILE,
        target_satellite=TARGET_SATELLITE,
    )

    # --------------------------------------------------------
    # 批量处理
    # --------------------------------------------------------
    results = []
    failures = []

    for i, event in events.iterrows():
        print(f"\n[{i + 1}/{len(events)}] {event['ICMEid']}")
        try:
            result = process_single_icme_event(
                event,
                output_root,
            )
            results.append(result)
        except Exception as exc:
            print("[ERROR] 事件处理失败")
            print(f"        ICMEid = {event['ICMEid']}")
            print(f"        error  = {type(exc).__name__}: {exc}")
            failures.append({
                "ICMEid": event["ICMEid"],
                "Satellite": event["Satellite"],
                "ShockArrival": event["ShockArrival"],
                "ICMEStart": event["ICMEStart"],
                "ICMEEnd": event["ICMEEnd"],
                "Error": f"{type(exc).__name__}: {exc}",
            })
            # 一个事件失败不影响后续事件。
            continue

    # --------------------------------------------------------
    # 批处理结果汇总
    # --------------------------------------------------------
    result_path = output_root / "PSP_ICME_batch_processing_results.csv"
    failure_path = output_root / "PSP_ICME_batch_processing_failures.csv"

    if results:
        pd.DataFrame(results).to_csv(result_path, index=False)
        print(f"\n[SAVE BATCH RESULT] {result_path}")

    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        print(f"[SAVE FAILURE LIST] {failure_path}")

    print("\n" + "=" * 90)
    print("BATCH PROCESSING FINISHED")
    print("=" * 90)
    print(f"Total events : {len(events)}")
    print(f"Succeeded    : {len(results)}")
    print(f"Failed       : {len(failures)}")
    print("=" * 90)




if __name__ == "__main__":
    main()
