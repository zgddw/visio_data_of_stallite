import os
import glob
import numpy as np
import pandas as pd
import spacepy.pycdf
from datetime import datetime, timedelta
import bisect

# ========================== 数据读取函数（来自 pro_of_read.py）==========================
def cdfdataread(
    root_path,
    var_names,
    start_time,
    end_time,
    file_prefix
):
    start_dt = start_time
    end_dt = end_time

    date_list = []
    d = start_dt.date()
    while d <= end_dt.date():
        date_list.append(d)
        d += timedelta(days=1)

    var_data_tmp = {v: {"t": [], "data": [], "attrs": {}} for v in var_names}

    for date_obj in date_list:
        year = date_obj.strftime('%Y')
        day = date_obj.strftime('%Y%m%d')
        pattern = os.path.join(root_path, year, f"{file_prefix}_{day}_v*.cdf")
        files = sorted(glob.glob(pattern))
        if not files:
            continue

        with spacepy.pycdf.CDF(files[-1]) as cdf:
            for v in var_names:
                if v not in cdf:
                    continue

                dep_name = cdf[v].attrs.get('DEPEND_0', 'Epoch')
                if hasattr(dep_name, 'encode'):
                    dep_name = str(dep_name)

                if dep_name not in cdf:
                    continue

                if not var_data_tmp[v]["attrs"]:
                    var_data_tmp[v]["attrs"] = {
                        'FILLVAL': cdf[v].attrs.get('FILLVAL'),
                        'VALIDMIN': cdf[v].attrs.get('VALIDMIN'),
                        'VALIDMAX': cdf[v].attrs.get('VALIDMAX')
                    }

                times = cdf[dep_name][...]

                i0 = bisect.bisect_left(times, start_dt) if date_obj == start_dt.date() else 0
                i1 = bisect.bisect_right(times, end_dt) if date_obj == end_dt.date() else len(times)

                if i0 >= i1:
                    continue

                var_data_tmp[v]["t"].append(times[i0:i1])
                var_data_tmp[v]["data"].append(cdf[v][i0:i1])

    final_output = {}

    for v in var_names:
        if not var_data_tmp[v]["t"]:
            print(f"[WARN] No data found for variable: {v}")
            continue

        t_arr = np.concatenate(var_data_tmp[v]["t"])
        t_arr = np.squeeze(t_arr)

        arr = np.asarray(np.concatenate(var_data_tmp[v]["data"]), dtype=float)

        attrs = var_data_tmp[v]["attrs"]
        fill_val = attrs.get('FILLVAL')
        valid_min = attrs.get('VALIDMIN')
        valid_max = attrs.get('VALIDMAX')

        invalid = ~np.isfinite(arr)

        if fill_val is not None:
            try:
                fv = float(np.asarray(fill_val).item()) if hasattr(fill_val, '__iter__') and not isinstance(fill_val, (str, bytes)) else float(fill_val)
                invalid |= np.isclose(arr, fv, rtol=1e-5, atol=1e-8) | (arr == fv)
            except (ValueError, TypeError):
                pass

        if valid_min is not None:
            try:
                vmin = float(np.asarray(valid_min).item()) if hasattr(valid_min, '__iter__') and not isinstance(valid_min, (str, bytes)) else float(valid_min)
                invalid |= (arr < vmin)
            except (ValueError, TypeError):
                pass

        if valid_max is not None:
            try:
                vmax = float(np.asarray(valid_max).item()) if hasattr(valid_max, '__iter__') and not isinstance(valid_max, (str, bytes)) else float(valid_max)
                invalid |= (arr > vmax)
            except (ValueError, TypeError):
                pass

        invalid |= (np.abs(arr) > 1e25)

        if invalid.any():
            arr[invalid] = np.nan

        df_single = pd.DataFrame()
        df_single['t'] = pd.to_datetime(t_arr)

        if arr.ndim == 2:
            v_lower = v.lower()
            # 识别矢量变量，这里硬编码为 B_RTN
            is_vector = v == 'B_RTN'
            if arr.shape[1] == 3 and is_vector:
                df_single['B_RTN_X'] = arr[:, 0]
                df_single['B_RTN_Y'] = arr[:, 1]
                df_single['B_RTN_Z'] = arr[:, 2]
            else:
                df_single[v] = [row for row in arr]
        elif arr.ndim == 1:
            if len(arr) == len(df_single):
                df_single[v] = arr
            elif len(arr) == 0:
                print(f"[WARN] 变量 {v} 在此时段内无有效数据")
            else:
                print(f"[WARN] 变量 {v} 数据长度 ({len(arr)}) 与时间轴 ({len(df_single)}) 不匹配")
        else:
            df_single[v] = [slice_ for slice_ in arr]

        final_output[v] = df_single

    return final_output

# ========================== PMS 识别核心函数 ==========================
def compute_pms_metrics(df_window):
    """
    对窗口内的磁场数据计算 MVA 并检查所有 PMS 条件。
    返回字典包含 n, ratio, Bn_avg, coverage, 若条件不满足则返回 None。
    """
    # 提取三分量
    B = df_window[['B_RTN_X', 'B_RTN_Y', 'B_RTN_Z']].values
    # 去除包含 NaN 的行
    valid = ~np.isnan(B).any(axis=1)
    B = B[valid]
    if len(B) < 3:
        return None

    # 去均值协方差矩阵
    mean = np.mean(B, axis=0)
    Bc = B - mean
    M = (Bc.T @ Bc) / len(B)

    # 特征值分解（对称矩阵，使用 eigh）
    eigvals, eigvecs = np.linalg.eigh(M)  # 升序排列
    # 转为降序：lambda1 > lambda2 > lambda3
    lambda1, lambda2, lambda3 = eigvals[2], eigvals[1], eigvals[0]
    if lambda3 == 0:
        return None
    ratio = lambda2 / lambda3
    if ratio <= 2:
        return None

    # 法向方向（最小特征值对应的特征向量）
    n = eigvecs[:, 0]  # 对应 lambda3

    # 计算 |Bn|/B 的平均值
    Bn_abs = np.abs(B @ n)
    Bmag = np.linalg.norm(B, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        Bn_ratio = Bn_abs / Bmag
        Bn_ratio = Bn_ratio[~np.isinf(Bn_ratio) & ~np.isnan(Bn_ratio)]
    if len(Bn_ratio) == 0:
        return None
    Bn_avg = np.mean(Bn_ratio)
    if Bn_avg >= 0.20:
        return None

    # 方位角覆盖范围（使用 RTN 坐标下的方位角 phi = atan2(B_t, B_r)）
    Br = B[:, 0]
    Bt = B[:, 1]
    phi = np.arctan2(Bt, Br)  # 范围 [-pi, pi]

    # 计算圆形范围（覆盖的弧度）
    phi_sorted = np.sort(phi)
    gaps = np.diff(phi_sorted)
    gaps = np.append(gaps, 2*np.pi - (phi_sorted[-1] - phi_sorted[0]))
    max_gap = np.max(gaps)
    coverage = 2*np.pi - max_gap
    coverage_deg = np.degrees(coverage)
    if coverage_deg < 90:
        return None

    # 全部条件满足
    return {
        'n': n,
        'ratio': ratio,
        'Bn_avg': Bn_avg,
        'coverage': coverage_deg
    }


def process_event(shock_time, start_icme, end_icme, df):
    """
    处理单个 ICME 事件，在激波下游 2 小时（排除前 15 分钟）内寻找最佳 PMS。
    返回 (pms_start, pms_end, duration_minutes) 或 None。
    """
    # 分析区间：激波后 15 分钟 到 120 分钟
    interval_start = shock_time + timedelta(minutes=15)
    interval_end = shock_time + timedelta(minutes=120)

    # 如果数据框为空，直接返回
    if df.empty:
        return None

    # 筛选出在区间内的数据（按时间）
    mask = (df['t'] >= interval_start) & (df['t'] <= interval_end)
    df_interval = df.loc[mask].copy()
    if df_interval.empty:
        return None

    # 最大窗口长度（分钟）
    max_len = 105  # 120 - 15
    min_len = 5

    best = None  # 存储最佳窗口信息

    # 从最长窗口开始，向下遍历
    for L in range(max_len, min_len - 1, -1):
        # 候选起始时间：从 interval_start 到 interval_end - L 分钟，步长 1 分钟
        start_candidates = []
        cur = interval_start
        end_possible = interval_end - timedelta(minutes=L)
        while cur <= end_possible:
            start_candidates.append(cur)
            cur += timedelta(minutes=1)

        best_in_L = None
        for st in start_candidates:
            et = st + timedelta(minutes=L)
            # 提取窗口内的数据
            window_mask = (df_interval['t'] >= st) & (df_interval['t'] < et)
            df_window = df_interval.loc[window_mask]
            if len(df_window) < 3:
                continue

            metrics = compute_pms_metrics(df_window)
            if metrics is None:
                continue

            # 记录该长度下的最佳窗口（B n 平均最小）
            if best_in_L is None or metrics['Bn_avg'] < best_in_L['Bn_avg']:
                best_in_L = {
                    'start': st,
                    'end': et,
                    'length': L,
                    'Bn_avg': metrics['Bn_avg'],
                    'ratio': metrics['ratio'],
                    'coverage': metrics['coverage']
                }

        # 如果当前长度下找到了有效窗口，由于我们是按长度递减，第一个找到的长度即为最长可能，直接设为全局最佳并跳出循环
        if best_in_L is not None:
            best = best_in_L
            break

    if best is None:
        return None

    return best['start'], best['end'], best['length']


# ========================== 主程序 ==========================
if __name__ == "__main__":
    # 路径配置
    data_root = r"I:\stdata\solo\mag\solo_l2_mag-rtn-normal-1-minute"
    file_prefix = "solo_l2_mag-rtn-normal-1-minute"
    icme_list_path = r"E:\ZhaoWenjin_Reserch\ICME\结果图像\SOLO\list.txt"

    # 读取 ICME 事件列表
    df_icme = pd.read_csv(icme_list_path, parse_dates=['shock_Datetime', 'Start_Datetime', 'End_Datetime'])

    results = []

    for idx, row in df_icme.iterrows():
        shock = row['shock_Datetime']
        start_icme = row['Start_Datetime']
        end_icme = row['End_Datetime']

        # 若 ICME 开始时间等于激波时间，则跳过（表示无激波）
        if start_icme == shock:
            print(f"[INFO] 跳过事件 {shock}，因 ICME 开始与激波时间相同。")
            continue

        # 定义数据读取时间范围（激波后 2 小时）
        data_start = shock
        data_end = shock + timedelta(minutes=120)

        print(f"处理事件 {idx+1}/{len(df_icme)}: 激波 {shock}")

        try:
            # 读取磁场数据
            data_dict = cdfdataread(
                root_path=data_root,
                var_names=['B_RTN'],
                start_time=data_start,
                end_time=data_end,
                file_prefix=file_prefix
            )
        except Exception as e:
            print(f"  读取数据失败: {e}")
            continue

        if 'B_RTN' not in data_dict:
            print("  未找到 B_RTN 变量，跳过。")
            continue

        df = data_dict['B_RTN']
        if df.empty:
            print("  数据为空，跳过。")
            continue

        # 处理事件
        result = process_event(shock, start_icme, end_icme, df)

        if result is None:
            print("  未找到满足条件的 PMS。")
            results.append({
                'shock_time': shock,
                'pms_start': None,
                'pms_end': None,
                'duration_min': None,
                'status': 'no_pms',
                'num':0
            })
        else:
            pms_start, pms_end, duration = result
            print(f"  找到 PMS: {pms_start} -> {pms_end} (持续时间 {duration} 分钟)")
            results.append({
                'shock_time': shock,
                'pms_start': pms_start,
                'pms_end': pms_end,
                'duration_min': duration,
                'status': 'success',
                'num':1
            })

    # 保存结果到 CSV
    df_results = pd.DataFrame(results)
    output_file = "pms_results.csv"
    df_results.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f"\n结果已保存至 {output_file}")