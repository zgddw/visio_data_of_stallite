import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pro_of_read import cdfdataread

# ---------- 可调参数 ----------
RATIO_THRESHOLD = 0.5   # 只处理 Ratio 大于该值的事件

# ---------- 常量 ----------
kB = 1.380649e-23
mu0 = 4 * np.pi * 1e-7
nPa_to_Pa = 1e-9
cm3_to_m3 = 1e6
eV_to_K = 11604.5

def calc_temp_K(T_eV):
    return T_eV * eV_to_K

def calc_pressure_and_beta(n_cm3, T_K, B_nT):
    n_m3 = n_cm3 * cm3_to_m3
    Pth_Pa = n_m3 * kB * T_K
    Pth_nPa = Pth_Pa / nPa_to_Pa
    B_T = B_nT * 1e-9
    PB_Pa = B_T**2 / (2 * mu0)
    PB_nPa = PB_Pa / nPa_to_Pa
    beta = Pth_nPa / PB_nPa if PB_nPa > 0 else np.nan
    return Pth_nPa, PB_nPa, beta

def interpolate_mag_to_ion_times(df_ion, df_mag):
    if 't' not in df_ion.columns:
        raise ValueError("df_ion 没有 't' 列，现有列: " + str(df_ion.columns.tolist()))
    if 't' not in df_mag.columns:
        raise ValueError("df_mag 没有 't' 列，现有列: " + str(df_mag.columns.tolist()))
    t_ion_num = df_ion['t'].astype('int64') // 10**9
    t_mag_num = df_mag['t'].astype('int64') // 10**9
    sort_idx = np.argsort(t_mag_num)
    t_mag_sorted = t_mag_num.iloc[sort_idx].values
    B_R_sorted = df_mag['B_R'].iloc[sort_idx].values
    B_T_sorted = df_mag['B_T'].iloc[sort_idx].values
    B_N_sorted = df_mag['B_N'].iloc[sort_idx].values
    B_R_interp = np.interp(t_ion_num, t_mag_sorted, B_R_sorted)
    B_T_interp = np.interp(t_ion_num, t_mag_sorted, B_T_sorted)
    B_N_interp = np.interp(t_ion_num, t_mag_sorted, B_N_sorted)
    df_ion = df_ion.copy()
    df_ion['B_R'] = B_R_interp
    df_ion['B_T'] = B_T_interp
    df_ion['B_N'] = B_N_interp
    return df_ion

def parse_solo_event_table(file_path):
    """
    读取 SOLO 事件列表，正确处理时间列和有无激波标志。
    使用正则表达式匹配两个以上空格作为分隔符，避免日期时间内部的空格被拆分。
    列顺序：Event_Index, Shock_Datetime, Start_Time, End_Time, Mark1, Mark2, ...
    """
    df = pd.read_csv(file_path, sep='\s{2,}', comment='#', header=None,
                     engine='python',
                     names=['Event_Index', 'Shock_Datetime', 'Start_Time', 'End_Time',
                            'Mark1', 'Mark2', 'Mean_interval', 'Max_background',
                            'Ratio', 'Duration_hours'])
    # 将时间列转换为 datetime 类型
    time_cols = ['Shock_Datetime', 'Start_Time', 'End_Time', 'Mark1', 'Mark2']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col])
    # 判断是否有激波：比较 Shock_Datetime 和 Start_Time
    df['has_shock'] = df['Shock_Datetime'] != df['Start_Time']
    return df

def get_phase_intervals(row):
    shock = row['Shock_Datetime']
    start = row['Start_Time']
    end = row['End_Time']
    mark1 = row['Mark1']
    mark2 = row['Mark2']
    if row['has_shock']:
        phase1 = ('Shock_to_Mark1', shock, mark1)
    else:
        phase1 = ('Background_8h_to_Mark1', mark1 - timedelta(hours=8), mark1)
    phase2 = ('Mark1_to_Mark2', mark1, mark2)
    icme_start = start if mark2 <= start else mark2
    phase3 = ('ICME', icme_start, end)
    return [phase1, phase2, phase3]

def compute_phase_averages(df_ion, phases):
    results = []
    for name, start, end in phases:
        mask = (df_ion['t'] >= start) & (df_ion['t'] <= end)
        subset = df_ion[mask]
        if len(subset) == 0:
            B_mean = n_mean = V_mean = T_mean = Pth_mean = beta_mean = np.nan
        else:
            B_mean = subset['B_total'].mean()
            n_mean = subset['N'].mean()
            V_mean = subset['V_total'].mean()
            T_mean = subset['T_K'].mean()
            Pth_mean = subset['Pth'].mean()
            beta_mean = subset['beta'].mean()
        results.append({
            'phase': name,
            'B_total_mean_nT': B_mean,
            'N_mean_cm-3': n_mean,
            'V_total_mean_km/s': V_mean,
            'T_mean_K': T_mean,
            'Pth_mean_nPa': Pth_mean,
            'beta_mean': beta_mean
        })
    return results

def ensure_time_column(df, df_name="DataFrame"):
    if 't' in df.columns:
        return df
    time_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'epoch' in col_lower or 'time' in col_lower or 'scet' in col_lower:
            time_col = col
            break
    if time_col is None:
        if len(df.columns) == 1 and pd.api.types.is_datetime64_any_dtype(df[df.columns[0]]):
            time_col = df.columns[0]
    if time_col is None:
        print(f"  警告: {df_name} 中未找到时间列，现有列: {df.columns.tolist()}")
        return df
    df.rename(columns={time_col: 't'}, inplace=True)
    return df

def main():
    event_file = r'E:\ZhaoWenjin_Reserch\pileup\solo列表.txt'
    root_mag = r'I:\stdata\solo\mag\solo_l2_mag-rtn-normal-1-minute'
    root_ion = r'I:\stdata\solo\swa\solo_l2_swa-pas-grnd-mom'
    output_excel = 'solo_shock_analysis_results.xlsx'

    # 读取事件列表
    events = parse_solo_event_table(event_file)
    if events.empty:
        print("事件列表为空，程序退出。")
        return

    # ---------- 应用比例阈值筛选 ----------
    original_count = len(events)
    events = events[events['Ratio'] > RATIO_THRESHOLD].reset_index(drop=True)
    print(f"事件总数: {original_count}，Ratio > {RATIO_THRESHOLD} 的事件: {len(events)}")

    if events.empty:
        print("没有满足阈值的事件，程序退出。")
        return

    all_results = []

    for idx, row in events.iterrows():
        mark1 = row['Mark1']
        end = row['End_Time']
        start_global = min(row['Shock_Datetime'], mark1 - timedelta(hours=9))
        end_global = end + timedelta(hours=1)

        print(f"\n处理事件 {row['Event_Index']}: 激波 {row['Shock_Datetime']} | 到达 {row['Start_Time']} ~ {end} (有激波? {row['has_shock']}, Ratio={row['Ratio']:.3f})")

        # ---------- 读取离子数据 ----------
        ion_vars = ['N', 'V_RTN', 'T']
        ion_data = cdfdataread(
            root_path=root_ion,
            var_names=ion_vars,
            start_time=start_global,
            end_time=end_global,
            file_prefix='solo_l2_swa-pas-grnd-mom'
        )
        if not ion_data:
            print("  未读取到离子数据，跳过")
            continue

        df_N = ion_data.get('N')
        df_V = ion_data.get('V_RTN')
        df_T = ion_data.get('T')
        if df_N is None or df_V is None or df_T is None:
            print("  离子数据缺失变量，跳过")
            continue

        # 确保 't' 列存在
        df_N = ensure_time_column(df_N, "N")
        df_V = ensure_time_column(df_V, "V_RTN")
        df_T = ensure_time_column(df_T, "T")
        if 't' not in df_N.columns or 't' not in df_V.columns or 't' not in df_T.columns:
            print("  时间列修复失败，跳过")
            continue

        # 合并
        df_ion = pd.merge(df_N, df_V, on='t', how='inner')
        df_ion = pd.merge(df_ion, df_T, on='t', how='inner')

        if 't' not in df_ion.columns:
            print("  合并后 't' 列丢失，当前列:", df_ion.columns.tolist())
            continue

        # ---- 重命名物理量列（确保不覆盖 't'） ----
        # 密度列
        density_col = None
        for col in df_ion.columns:
            if col == 't':
                continue
            if col.lower() in ['n', 'np', 'n_moment']:
                density_col = col
                break
        if density_col is None:
            if 'N' in df_ion.columns:
                density_col = 'N'
            elif 'Np' in df_ion.columns:
                density_col = 'Np'
        if density_col is None:
            print("  未找到密度列，当前列名:", df_ion.columns.tolist())
            continue

        # 速度分量列
        vx_col = vy_col = vz_col = None
        for col in df_ion.columns:
            if col == 't':
                continue
            if col.endswith('_X') or col.endswith('_R') or col == 'V_R':
                vx_col = col
                break
        for col in df_ion.columns:
            if col == 't':
                continue
            if col.endswith('_Y') or col.endswith('_T') or col == 'V_T':
                vy_col = col
                break
        for col in df_ion.columns:
            if col == 't':
                continue
            if col.endswith('_Z') or col.endswith('_N') or col == 'V_N':
                vz_col = col
                break
        if not (vx_col and vy_col and vz_col):
            vx_col = next((c for c in df_ion.columns if c.startswith('V_RTN_') and c.endswith('_X')), None)
            vy_col = next((c for c in df_ion.columns if c.startswith('V_RTN_') and c.endswith('_Y')), None)
            vz_col = next((c for c in df_ion.columns if c.startswith('V_RTN_') and c.endswith('_Z')), None)
        if not (vx_col and vy_col and vz_col):
            print("  未找到完整的速度分量列，当前列名:", df_ion.columns.tolist())
            continue

        # 温度列（排除 't'）
        temp_col = None
        for col in df_ion.columns:
            if col == 't':
                continue
            if col.lower() in ['temp', 'temperature', 't_eV', 't_ev']:
                temp_col = col
                break
        if temp_col is None:
            for col in df_ion.columns:
                if col == 't':
                    continue
                if col in ['T', 'Tp', 'T_eV']:
                    temp_col = col
                    break
        if temp_col is None:
            print("  未找到温度列，当前列名:", df_ion.columns.tolist())
            continue

        # 执行重命名（不涉及 't'）
        rename_dict = {}
        if density_col != 'N':
            rename_dict[density_col] = 'N'
        if vx_col != 'V_R':
            rename_dict[vx_col] = 'V_R'
        if vy_col != 'V_T':
            rename_dict[vy_col] = 'V_T'
        if vz_col != 'V_N':
            rename_dict[vz_col] = 'V_N'
        if temp_col != 'T_eV':
            rename_dict[temp_col] = 'T_eV'
        if rename_dict:
            df_ion.rename(columns=rename_dict, inplace=True)

        if 't' not in df_ion.columns:
            print("  重命名后 't' 列丢失，当前列:", df_ion.columns.tolist())
            continue

        if len(df_ion) == 0:
            print("  离子数据为空，跳过")
            continue

        # ---------- 读取磁场数据 ----------
        mag_data = cdfdataread(
            root_path=root_mag,
            var_names=['B_RTN'],
            start_time=start_global,
            end_time=end_global,
            file_prefix='solo_l2_mag-rtn-normal-1-minute'
        )
        if not mag_data:
            print("  未读取到磁场数据，跳过")
            continue
        df_mag = mag_data.get('B_RTN')
        if df_mag is None or len(df_mag) == 0:
            print("  磁场数据为空，跳过")
            continue

        df_mag = ensure_time_column(df_mag, "B_RTN")
        if 't' not in df_mag.columns:
            print("  磁场时间列修复失败，跳过")
            continue

        rename_mag = {}
        for col in df_mag.columns:
            if col.startswith('B_RTN_'):
                suffix = col.split('_')[-1]
                if suffix in ['X', 'R']:
                    rename_mag[col] = 'B_R'
                elif suffix in ['Y', 'T']:
                    rename_mag[col] = 'B_T'
                elif suffix in ['Z', 'N']:
                    rename_mag[col] = 'B_N'
        df_mag.rename(columns=rename_mag, inplace=True)
        if not all(c in df_mag.columns for c in ['B_R', 'B_T', 'B_N']):
            print("  磁场重命名失败，当前列名:", df_mag.columns.tolist())
            continue

        # ---------- 插值磁场 ----------
        try:
            df_ion = interpolate_mag_to_ion_times(df_ion, df_mag)
        except ValueError as e:
            print(f"  插值失败: {e}")
            continue

        # ---------- 计算物理量 ----------
        df_ion['B_total'] = np.sqrt(df_ion['B_R']**2 + df_ion['B_T']**2 + df_ion['B_N']**2)
        df_ion['V_total'] = np.sqrt(df_ion['V_R']**2 + df_ion['V_T']**2 + df_ion['V_N']**2)
        df_ion['T_K'] = calc_temp_K(df_ion['T_eV'])

        Pth_list = []
        beta_list = []
        for _, row_ion in df_ion.iterrows():
            Pth, _, beta = calc_pressure_and_beta(row_ion['N'], row_ion['T_K'], row_ion['B_total'])
            Pth_list.append(Pth)
            beta_list.append(beta)
        df_ion['Pth'] = Pth_list
        df_ion['beta'] = beta_list

        # ---------- 阶段划分 ----------
        phases = get_phase_intervals(row)
        phase_avgs = compute_phase_averages(df_ion, phases)

        for ph in phase_avgs:
            all_results.append({
                'Event_Index': row['Event_Index'],
                'Has_Shock': row['has_shock'],
                'Shock_Datetime': row['Shock_Datetime'],
                'Start_Time': row['Start_Time'],
                'End_Time': row['End_Time'],
                'Ratio': row['Ratio'],
                'Phase': ph['phase'],
                'B_total_mean_nT': ph['B_total_mean_nT'],
                'N_mean_cm-3': ph['N_mean_cm-3'],
                'V_total_mean_km/s': ph['V_total_mean_km/s'],
                'T_mean_K': ph['T_mean_K'],
                'Pth_mean_nPa': ph['Pth_mean_nPa'],
                'beta_mean': ph['beta_mean']
            })

    if all_results:
        df_out = pd.DataFrame(all_results)
        df_out.to_excel(output_excel, index=False)
        print(f"\n结果已保存至 {output_excel}")
    else:
        print("没有生成任何结果。")

if __name__ == '__main__':
    main()