import os
import glob
import numpy as np
import pandas as pd
import spacepy.pycdf
from datetime import datetime, timedelta
import bisect

# ======================== 数据读取函数 ========================
def read_psp_mag(start_time, end_time, root_path):
    """
    读取PSP FIELDS MAG RTN 1分钟数据
    返回: times (datetime列表), B (Nx3 numpy数组, 列顺序: R, T, N)
    """
    all_times = []
    all_B = []

    date = start_time.date()
    end_date = end_time.date()
    while date <= end_date:
        year = date.strftime('%Y')
        ymd = date.strftime('%Y%m%d')
        pattern = os.path.join(root_path, year, f"psp_fld_l2_mag_rtn_1min_{ymd}_v*.cdf")
        files = sorted(glob.glob(pattern))
        if not files:
            date += timedelta(days=1)
            continue

        # 取最新版本（最后一个）
        cdf_file = files[-1]
        with spacepy.pycdf.CDF(cdf_file) as cdf:
            # 读取时间（返回 datetime64[ns] 数组）
            epoch = cdf['epoch_mag_RTN_1min'][...]
            # 转换为 pandas DatetimeIndex
            t_datetime = pd.to_datetime(epoch)

            # 读取磁场数据 (RTN)
            B_data = cdf['psp_fld_l2_mag_RTN_1min'][...]  # shape (N,3)

            # --- 处理填充值 (FILLVAL) ---
            fillval = cdf['psp_fld_l2_mag_RTN_1min'].attrs.get('FILLVAL', None)
            if fillval is not None:
                # 转为 float 并替换填充值为 NaN
                B_data = B_data.astype(float)
                # 使用 isclose 检测（容忍度可调）
                mask_fill = np.isclose(B_data, fillval, rtol=1e-5, atol=1e-8)
                B_data[mask_fill] = np.nan

            # 筛选在时间范围内的数据
            mask_time = (t_datetime >= pd.Timestamp(start_time)) & (t_datetime <= pd.Timestamp(end_time))
            if np.any(mask_time):
                # 提取数据并去除含有 NaN 的行（填充值导致）
                t_sel = t_datetime[mask_time]
                B_sel = B_data[mask_time]
                # 去除任何包含 NaN 的行（即填充值替换后的无效数据）
                valid_rows = ~np.isnan(B_sel).any(axis=1)
                if np.any(valid_rows):
                    all_times.extend(t_sel[valid_rows].tolist())
                    all_B.extend(B_sel[valid_rows].tolist())

        date += timedelta(days=1)

    if len(all_times) == 0:
        return np.array([]), np.array([])

    # 按时间排序（跨天合并后可能乱序）
    combined = sorted(zip(all_times, all_B), key=lambda x: x[0])
    times = np.array([x[0] for x in combined])
    B = np.array([x[1] for x in combined])
    return times, B

# ======================== MVA 计算函数 ========================
def compute_mva(B):
    """
    对磁场数据B (Nx3) 进行MVA分析
    返回: lambda2/lambda3, mean(|Bn|/B), 角度跨度(度), 法向向量n, 特征值数组
    """
    N = B.shape[0]
    if N < 3:
        return np.nan, np.nan, np.nan, None, None

    # 去除可能存在的 NaN（安全处理）
    if np.isnan(B).any():
        B = B[~np.isnan(B).any(axis=1)]
        if B.shape[0] < 3:
            return np.nan, np.nan, np.nan, None, None

    # 均值
    B_mean = B.mean(axis=0)
    dB = B - B_mean
    M = (dB.T @ dB) / N

    # 特征分解 (eigh 返回升序)
    eigvals, eigvecs = np.linalg.eigh(M)  # 升序: λ_min, λ_mid, λ_max
    # 对应: 特征向量列
    n = eigvecs[:, 0]          # 最小特征值 -> 法向
    v2 = eigvecs[:, 1]         # 中间
    v1 = eigvecs[:, 2]         # 最大

    # 防止除零
    lambda2_lambda3 = eigvals[1] / eigvals[0] if eigvals[0] > 1e-12 else np.inf

    # 计算 |Bn|/B 平均值
    Bn = B @ n
    B_norm = np.linalg.norm(B, axis=1)
    ratio = np.abs(Bn) / (B_norm + 1e-30)
    mean_ratio = np.mean(ratio)

    # 计算角度跨度 (在v1-v2平面内)
    B_perp = B - np.outer(Bn, n)  # 投影到法向垂直平面
    # 投影到 v1, v2 坐标
    B1 = B_perp @ v1
    B2 = B_perp @ v2
    angles = np.arctan2(B2, B1)  # 范围 [-pi, pi]

    if len(angles) > 1:
        sorted_angles = np.sort(angles)
        diffs = np.diff(sorted_angles)
        # 加上首尾环绕
        diffs = np.append(diffs, sorted_angles[0] + 2*np.pi - sorted_angles[-1])
        max_gap = np.max(diffs)
        span_rad = 2*np.pi - max_gap
        span_deg = np.degrees(span_rad)
    else:
        span_deg = 0.0

    return lambda2_lambda3, mean_ratio, span_deg, n, eigvals

# ======================== 主程序 ========================
def main():
    # 配置路径
    list_file = r"E:\ZhaoWenjin_Reserch\ICME\结果图像\PSP\线图\list.txt"
    psp_root = r"I:\stdata\psp\FLD\PSP_FLD_L2_MAG_RTN_1MIN"
    output_file = r"E:\ZhaoWenjin_Reserch\PMS\pms_results.csv"

    # 读取激波事件列表
    df_events = pd.read_csv(list_file)
    # 确保时间列解析
    df_events['shock_Datetime'] = pd.to_datetime(df_events['shock_Datetime'])
    df_events['Start_Datetime'] = pd.to_datetime(df_events['Start_Datetime'])
    df_events['End_Datetime'] = pd.to_datetime(df_events['End_Datetime'])

    results = []

    for idx, row in df_events.iterrows():
        shock_time = row['shock_Datetime']
        icme_start = row['Start_Datetime']
        icme_end = row['End_Datetime']

        # 如果激波时间和ICME开始时间相同（即不存在激波），跳过
        if abs((shock_time - icme_start).total_seconds()) < 60:
            print(f"事件 {shock_time} 无激波（ICME开始时间相同），跳过")
            continue

        print(f"处理激波: {shock_time}")

        # 下游区域: 激波后15分钟到2小时
        downstream_start = shock_time + timedelta(minutes=15)
        downstream_end = shock_time + timedelta(minutes=120)

        # 读取磁场数据
        try:
            times, B = read_psp_mag(downstream_start, downstream_end, psp_root)
        except Exception as e:
            print(f"  读取数据失败: {e}")
            results.append({
                'shock_time': shock_time,
                'icme_start': icme_start,
                'icme_end': icme_end,
                'pms_start': None,
                'pms_end': None,
                'duration_min': None,
                'lambda2_lambda3': None,
                'Bn_B_mean': None,
                'angle_span_deg': None,
                'status': 'data_error'
            })
            continue

        if len(times) == 0:
            print(f"  下游区间无数据")
            results.append({
                'shock_time': shock_time,
                'icme_start': icme_start,
                'icme_end': icme_end,
                'pms_start': None,
                'pms_end': None,
                'duration_min': None,
                'lambda2_lambda3': None,
                'Bn_B_mean': None,
                'angle_span_deg': None,
                'status': 'no_data'
            })
            continue

        # 窗口扫描参数
        best_duration = -1
        best_avg_ratio = np.inf
        best_info = None

        total_minutes = 120  # 下游区间长度120分钟
        # 注意：数据点可能不是整分钟，我们通过时间范围筛选，并实际检查点数
        for L in range(120, 4, -1):  # 120 down to 5
            for s in range(0, total_minutes - L + 1):
                win_start = downstream_start + timedelta(minutes=s)
                win_end = win_start + timedelta(minutes=L)
                # 筛选数据
                mask = (times >= win_start) & (times <= win_end)
                B_win = B[mask]
                if len(B_win) < 3:
                    continue

                # 执行MVA
                lam2_lam3, mean_ratio, span_deg, n, eigvals = compute_mva(B_win)
                if np.isnan(lam2_lam3) or np.isnan(mean_ratio) or np.isnan(span_deg):
                    continue

                # 判断是否满足PMS条件
                if lam2_lam3 > 2 and span_deg > 90 and mean_ratio < 0.20:
                    if L > best_duration or (L == best_duration and mean_ratio < best_avg_ratio):
                        best_duration = L
                        best_avg_ratio = mean_ratio
                        best_info = {
                            'pms_start': win_start,
                            'pms_end': win_end,
                            'duration_min': L,
                            'lambda2_lambda3': lam2_lam3,
                            'Bn_B_mean': mean_ratio,
                            'angle_span_deg': span_deg,
                            'n_vector': n,
                            'eigvals': eigvals
                        }

        if best_info is not None:
            print(f"  找到PMS: 持续时间 {best_info['duration_min']} min, λ2/λ3={best_info['lambda2_lambda3']:.2f}, |Bn|/B={best_info['Bn_B_mean']:.3f}")
            results.append({
                'shock_time': shock_time,
                'icme_start': icme_start,
                'icme_end': icme_end,
                'pms_start': best_info['pms_start'],
                'pms_end': best_info['pms_end'],
                'duration_min': best_info['duration_min'],
                'lambda2_lambda3': best_info['lambda2_lambda3'],
                'Bn_B_mean': best_info['Bn_B_mean'],
                'angle_span_deg': best_info['angle_span_deg'],
                'status': 'found'
            })
        else:
            print(f"  未找到满足条件的PMS")
            results.append({
                'shock_time': shock_time,
                'icme_start': icme_start,
                'icme_end': icme_end,
                'pms_start': None,
                'pms_end': None,
                'duration_min': None,
                'lambda2_lambda3': None,
                'Bn_B_mean': None,
                'angle_span_deg': None,
                'status': 'not_found'
            })

    # 保存结果到CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)
    print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    main()