import numpy as np
import pandas as pd


def Pandbeta(b,n,T,tn,tb):
    """
    计算太阳风热压、磁压和等离子体 beta。
    支持磁场与等离子体参数时间轴不一致的情况。
    【已升级：使用 NumPy 底层插值，免疫卫星数据中的重复时间戳报错】
    """
    # ---------- 常量 ----------
    kB = 1.380649e-23
    mu0 = 4 * np.pi * 1e-7
    cm3_to_m3 = 1e6
    nPa_to_Pa = 1e-9

    # ---------- 热压 ----------
    n_m3 = np.asarray(n, float) * cm3_to_m3
    Tp = np.asarray(T, float)

    Pth_Pa = n_m3 * kB * Tp
    Pth_nPa = Pth_Pa / nPa_to_Pa

    # ---------- 磁场插值 (底层强力方案) ----------
    # 1. 将时间转换为底层数字型时间戳 (纳秒级别)，彻底脱离 Pandas 的 Index 限制
    tb_num = pd.to_datetime(tb.flatten()).astype('int64')
    tn_num = pd.to_datetime(tn.flatten()).astype('int64')

    # 2. NumPy 插值要求 X 轴 (tb_num) 必须是严格从小到大排序的，这里做一层安全排序
    sort_idx = np.argsort(tb_num)
    tb_num_sorted = tb_num[sort_idx]
    b_sorted = np.asarray(b, float)[sort_idx]

    # 3. 核心：使用 numpy 进行线性插值。即使 tn_num 里面有无数个重复的时间戳，也能完美插值不报错！
    Bt_interp_vals = np.interp(tn_num, tb_num_sorted, b_sorted)

    Bt_T = Bt_interp_vals * 1e-9  # nT → T

    # ---------- 磁压 ----------
    PB_Pa = Bt_T**2 / (2 * mu0)
    PB_nPa = PB_Pa / nPa_to_Pa

    # ---------- beta ----------
    beta = Pth_nPa / PB_nPa

    return {
        'Pth': Pth_nPa,
        'beta': beta,
        't': tn  # 这里明确返回了 't'，让主程序的 res_beta['t'] 完美对接
    }


def rtnthetaphi(b):
    # 确保输入是 float 类型的 numpy 数组
    b = np.asarray(b,dtype=float)
    
    # 计算总磁场，遇到 nan 会返回 nan 而不会崩溃
    btotal = np.sqrt(np.nansum(b**2, axis=1))
    
    # 防止分母为 0
    btotal_safe = np.where(btotal == 0, np.nan, btotal)
    
    # 计算 theta (-90 到 90)
    # 使用 np.clip 防止因浮点数微小误差导致值超出 [-1, 1] 范围
    theta = np.arcsin(np.clip(b[:,2]/btotal_safe, -1, 1)) / np.pi * 180
    
    # 计算 phi (0 到 360)
    phi = np.arctan2(b[:,1], b[:,0]) / np.pi * 180 % 360
    
    return theta, phi