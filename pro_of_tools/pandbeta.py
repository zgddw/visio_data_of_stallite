import numpy as np
import pandas as pd


def Pandbeta(b,n,T,tn,tb):
    """
    计算太阳风热压、磁压和等离子体 beta。
    支持磁场与等离子体参数时间轴不一致的情况。

    必需：
        data['Np'], data['Tp'] : 同一时间轴
        data['Bt']             : 可不同时间轴
        data['t_np']           : Np/Tp 时间轴 (datetime)
        data['t_bt']           : Bt 时间轴 (datetime)

    返回：
        dict: Pth, beta, t
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
    # ---------- 磁场插值 ----------
    bt_series = pd.Series(
        np.asarray(b, float),
        index=pd.to_datetime(tb.flatten())
    )
    # 在 pandbeta.py 中，找到构建 Series 或 DataFrame 的地方
# 假设代码类似于：df = pd.Series(data, index=index)

# 1. 先检查并打印是否有重复（调试用）
    if bt_series.index.has_duplicates:
        print("Warning: Duplicate time labels detected! Removing duplicates...")

    # 2. 去除重复的索引，只保留第一个出现的值
    bt_series = bt_series[~bt_series.index.duplicated(keep='first')]

    # 3. 然后再执行插值和重索引
    Bt_interp = bt_series.interpolate(method='time').reindex(tn)
    # # 插值到等离子体时间轴
    # Bt_interp = bt_series.reindex(
    #     bt_series.index.union(tn)
    # ).interpolate(method='time').reindex(tn)

    Bt_T = Bt_interp.values * 1e-9  # nT → T

    # ---------- 磁压 ----------
    PB_Pa = Bt_T**2 / (2 * mu0)
    PB_nPa = PB_Pa / nPa_to_Pa

    # ---------- beta ----------
    beta = Pth_nPa / PB_nPa

    return {
        'Pth': Pth_nPa,
        'beta': beta
    }
