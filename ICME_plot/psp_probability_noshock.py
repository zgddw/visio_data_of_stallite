import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pro_of_read import cdfdataread  # 请确保该模块可导入

# ---------- 常量 ----------
kB = 1.380649e-23
mu0 = 4 * np.pi * 1e-7
nPa_to_Pa = 1e-9
cm3_to_m3 = 1e6

def calc_thermal_temp(wp_km_s):
    return 60.51 * wp_km_s**2

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
    """将磁场插值到离子时间"""
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

def parse_event_table(file_path):
    """读取事件表（无激波）"""
    df = pd.read_csv(file_path, sep='\t', parse_dates=[0,1,2,3,4])
    # 无激波情况下，可以不需要筛选Ratio，但根据要求可能所有事件都处理
    # 这里不筛选，全部处理
    return df

def get_phase_intervals_no_shock(row):
    """
    无激波事件的阶段划分：
    第一段：Mark1 - 8h 到 Mark1
    第二段：Mark1 到 Mark2
    第三段：max(Arrival, Mark2) 到 End
    """
    arrival = row['Shock']  # 第一列（与第二列相同）
    mark1 = row['Mark1']
    mark2 = row['Mark2']
    end = row['End']

    # 第三段开始时间
    icme_start = arrival if mark2 <= arrival else mark2

    phases = [
        ('Pre_Density_Enhancement_8h', mark1 - timedelta(hours=8), mark1),
        ('Density_Enhancement', mark1, mark2),
        ('ICME', icme_start, end)
    ]
    return phases

def compute_phase_averages(df_ion, phases):
    """计算各阶段平均值，返回列表字典"""
    results = []
    for name, start, end in phases:
        mask = (df_ion['t'] >= start) & (df_ion['t'] <= end)
        subset = df_ion[mask]
        if len(subset) == 0:
            B_mean = n_mean = V_mean = T_mean = Pth_mean = beta_mean = np.nan
        else:
            B_mean = subset['B_total'].mean()
            n_mean = subset['np_moment'].mean()
            V_mean = subset['V_total'].mean()
            T_mean = subset['T'].mean()
            Pth_mean = subset['Pth'].mean()
            beta_mean = subset['beta'].mean()
        results.append({
            'phase': name,
            'B_total_mean_nT': B_mean,
            'np_mean_cm-3': n_mean,
            'V_total_mean_km/s': V_mean,
            'T_mean_K': T_mean,
            'Pth_mean_nPa': Pth_mean,
            'beta_mean': beta_mean
        })
    return results

def main():
    # ---------- 文件路径 ----------
    event_file = 'E:\ZhaoWenjin_Reserch\pileup\无激波psp.txt'          # 修改为实际路径
    root_mag = r'I:\stdata\psp\FLD\PSP_FLD_L2_MAG_RTN_1MIN'
    root_ion = r'I:\stdata\psp\sweap\psp_swp_spc_l3i'
    output_excel = 'no_shock_analysis_results.xlsx'

    events = parse_event_table(event_file)
    if events.empty:
        print("事件列表为空，程序退出。")
        return

    all_results = []

    for idx, row in events.iterrows():
        # 计算需要读取的数据时间范围（覆盖所有阶段）
        mark1 = row['Mark1']
        end = row['End']
        start_global = mark1 - timedelta(hours=9)  # 多读1小时余量
        end_global = end + timedelta(hours=1)

        print(f"\n处理事件 {idx+1}: 到达 {row['Shock']} ~ {end}")

        # ---------- 读取离子数据 ----------
        ion_vars = ['np_moment', 'vp_moment_RTN', 'wp_moment']
        ion_data = cdfdataread(
            root_path=root_ion,
            var_names=ion_vars,
            start_time=start_global,
            end_time=end_global,
            file_prefix='psp_swp_spc_l3i'
        )
        if not ion_data:
            print("  未读取到离子数据，跳过此事件")
            continue

        df_np = ion_data.get('np_moment')
        df_vp = ion_data.get('vp_moment_RTN')
        df_wp = ion_data.get('wp_moment')
        if df_np is None or df_vp is None or df_wp is None:
            print("  离子数据缺失变量，跳过")
            continue

        # 合并
        df_ion = pd.merge(df_np, df_vp, on='t', how='inner')
        df_ion = pd.merge(df_ion, df_wp, on='t', how='inner')

        # ---- 自动识别列名并重命名 ----
        density_col = None
        for col in df_ion.columns:
            if col.lower() in ['np_moment', 'np', 'np_fit', 'n']:
                density_col = col
                break
        if density_col is None:
            print("  未找到密度列，当前列名:", df_ion.columns.tolist())
            continue

        vx_col = vy_col = vz_col = None
        for col in df_ion.columns:
            if col.endswith('_X') or col.endswith('_R') or col == 'V_R':
                vx_col = col
                break
        for col in df_ion.columns:
            if col.endswith('_Y') or col.endswith('_T') or col == 'V_T':
                vy_col = col
                break
        for col in df_ion.columns:
            if col.endswith('_Z') or col.endswith('_N') or col == 'V_N':
                vz_col = col
                break
        if not (vx_col and vy_col and vz_col):
            print("  未找到完整的速度分量列，当前列名:", df_ion.columns.tolist())
            continue

        wp_col = None
        for col in df_ion.columns:
            if col.lower() in ['wp_moment', 'wp', 'w']:
                wp_col = col
                break
        if wp_col is None:
            print("  未找到 wp 列，当前列名:", df_ion.columns.tolist())
            continue

        df_ion.rename(columns={
            density_col: 'np_moment',
            vx_col: 'V_R',
            vy_col: 'V_T',
            vz_col: 'V_N',
            wp_col: 'wp'
        }, inplace=True)

        if len(df_ion) == 0:
            print("  离子数据为空，跳过")
            continue

        # ---------- 读取磁场数据 ----------
        mag_data = cdfdataread(
            root_path=root_mag,
            var_names=['psp_fld_l2_mag_RTN_1min'],
            start_time=start_global,
            end_time=end_global,
            file_prefix='psp_fld_l2_mag_RTN_1min'
        )
        if not mag_data:
            print("  未读取到磁场数据，跳过")
            continue
        df_mag = mag_data.get('psp_fld_l2_mag_RTN_1min')
        if df_mag is None or len(df_mag) == 0:
            print("  磁场数据为空，跳过")
            continue

        # 重命名磁场分量
        rename_mag = {}
        for col in df_mag.columns:
            if col.startswith('psp_fld_l2_mag_RTN_1min_'):
                suffix = col.split('_')[-1]
                if suffix == 'X':
                    rename_mag[col] = 'B_R'
                elif suffix == 'Y':
                    rename_mag[col] = 'B_T'
                elif suffix == 'Z':
                    rename_mag[col] = 'B_N'
        df_mag.rename(columns=rename_mag, inplace=True)
        if not all(c in df_mag.columns for c in ['B_R', 'B_T', 'B_N']):
            print("  磁场重命名失败，当前列名:", df_mag.columns.tolist())
            continue

        # ---------- 插值磁场 ----------
        df_ion = interpolate_mag_to_ion_times(df_ion, df_mag)

        # ---------- 计算物理量 ----------
        df_ion['B_total'] = np.sqrt(df_ion['B_R']**2 + df_ion['B_T']**2 + df_ion['B_N']**2)
        df_ion['V_total'] = np.sqrt(df_ion['V_R']**2 + df_ion['V_T']**2 + df_ion['V_N']**2)
        df_ion['T'] = calc_thermal_temp(df_ion['wp'])

        Pth_list = []
        beta_list = []
        for _, row_ion in df_ion.iterrows():
            Pth, _, beta = calc_pressure_and_beta(row_ion['np_moment'], row_ion['T'], row_ion['B_total'])
            Pth_list.append(Pth)
            beta_list.append(beta)
        df_ion['Pth'] = Pth_list
        df_ion['beta'] = beta_list

        # ---------- 阶段划分 ----------
        phases = get_phase_intervals_no_shock(row)
        phase_avgs = compute_phase_averages(df_ion, phases)

        for ph in phase_avgs:
            all_results.append({
                'Event_No': idx+1,
                'Arrival_Time': row['Shock'],
                'Phase': ph['phase'],
                'B_total_mean_nT': ph['B_total_mean_nT'],
                'np_mean_cm-3': ph['np_mean_cm-3'],
                'V_total_mean_km/s': ph['V_total_mean_km/s'],
                'T_mean_K': ph['T_mean_K'],
                'Pth_mean_nPa': ph['Pth_mean_nPa'],
                'beta_mean': ph['beta_mean']
            })

    # ---------- 输出 ----------
    if all_results:
        df_out = pd.DataFrame(all_results)
        df_out.to_excel(output_excel, index=False)
        print(f"\n结果已保存至 {output_excel}")
    else:
        print("没有生成任何结果。")

if __name__ == '__main__':
    main()