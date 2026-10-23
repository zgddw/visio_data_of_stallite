import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from pro_of_read import cdfdataread

# ========== 用户配置 ==========
PSP_DATA_ROOT = r"I:\stdata\psp\sweap\spe\PSP_SWP_SPE_SF0_L3_PAD"
EVENT_FILE = r"E:\ZhaoWenjin_Reserch\pileup\有激波psp.txt"
ENERGY_CHANNEL_INDEX = 10
ANGLE_TOLERANCE = 10.0
THRESHOLD = 2.0
SAVE_FIGURES = True                    # 设为 False 则不保存
SAVE_DIR = r"E:\ZhaoWenjin_Reserch\pileup\双向电子流统计\psp"                # 保存目录，可以是绝对路径或相对路径
# ==============================

def read_psp_pad_data(root_path, var_names, start_time, end_time, file_prefix="PSP_SWP_SPE_SF0_L3_PAD"):
    return cdfdataread(root_path, var_names, start_time, end_time, file_prefix)

def get_pitch_angles(df_pitch):
    pitch = df_pitch['PITCHANGLE'].iloc[0]
    return np.array(pitch) if isinstance(pitch, (list, np.ndarray)) else np.array([pitch])

def get_energy_vals(df_energy):
    energy = df_energy['ENERGY_VALS'].iloc[0]
    return np.array(energy) if isinstance(energy, (list, np.ndarray)) else np.array([energy])

def compute_condition_intervals(times, flux_matrix, pitch_angles, threshold=2.0, angle_width=10.0):
    idx_0 = np.where(np.abs(pitch_angles - 0) <= angle_width)[0]
    idx_180 = np.where(np.abs(pitch_angles - 180) <= angle_width)[0]
    idx_90 = np.where(np.abs(pitch_angles - 90) <= angle_width)[0]
    if len(idx_0)==0 or len(idx_180)==0 or len(idx_90)==0:
        print("警告: 角度容差内无匹配，请调整 ANGLE_TOLERANCE")
        return []

    flag_list = []
    for i in range(len(times)):
        flux = flux_matrix[i, :]
        f0 = np.nanmean(flux[idx_0])
        f180 = np.nanmean(flux[idx_180])
        f90 = np.nanmean(flux[idx_90])
        flag_list.append(f0 >= threshold * f90 and f180 >= threshold * f90)

    intervals = []
    i = 0
    while i < len(flag_list):
        if flag_list[i]:
            start = i
            while i < len(flag_list) and flag_list[i]:
                i += 1
            end = i - 1
            intervals.append((times[start], times[end]))
        else:
            i += 1
    return intervals

def is_valid_dt(t):
    if t is None:
        return False
    if isinstance(t, pd.Timestamp):
        if pd.isna(t):
            return False
        return 1 <= t.year <= 9999
    if isinstance(t, datetime):
        return 1 <= t.year <= 9999
    return False

def plot_pad_with_intervals(times, flux_matrix, pitch_angles, intervals, mark1, mark2, event_info, energy_val):
    # ---------- 过滤有效时间 ----------
    valid_pairs = []
    for t, f in zip(times, flux_matrix):
        if is_valid_dt(t):
            try:
                num = mdates.date2num(t)
                if np.isfinite(num) and 1.0 <= num <= 1e6:
                    valid_pairs.append((num, t, f))
                else:
                    print(f"警告: 忽略异常时间值 {t} (数值={num})")
            except Exception:
                print(f"警告: 忽略无法转换的时间 {t}")
    if len(valid_pairs) == 0:
        print("警告: 所有时间均无效，无法绘图")
        return

    valid_pairs.sort(key=lambda x: x[0])
    num_times = np.array([p[0] for p in valid_pairs])
    valid_times = [p[1] for p in valid_pairs]
    flux_clean = np.array([p[2] for p in valid_pairs])

    # ---------- 处理 Mark1 和 Mark2 ----------
    if not is_valid_dt(mark1) or not is_valid_dt(mark2):
        print("警告: Mark1 或 Mark2 无效，使用数据时间范围替代")
        mark1 = valid_times[0]
        mark2 = valid_times[-1]
    try:
        num_mark1 = mdates.date2num(mark1)
        num_mark2 = mdates.date2num(mark2)
        if not (1.0 <= num_mark1 <= 1e6 and 1.0 <= num_mark2 <= 1e6):
            raise ValueError("Mark 数值超出范围")
    except Exception:
        print("警告: Mark 时间转换异常，使用数据范围替代")
        mark1 = valid_times[0]
        mark2 = valid_times[-1]
        num_mark1 = num_times[0]
        num_mark2 = num_times[-1]

    if num_mark1 > num_mark2:
        num_mark1, num_mark2 = num_mark2, num_mark1

    # ---------- 构建时间网格边界 ----------
    if len(num_times) == 1:
        dt_avg = 1.0
        time_bounds = np.array([num_times[0] - 0.5, num_times[0] + 0.5])
    else:
        dt = np.diff(num_times)
        dt_avg = np.mean(dt) if len(dt) > 0 else 1.0
        time_bounds = np.concatenate([
            [num_times[0] - dt_avg/2],
            (num_times[:-1] + num_times[1:]) / 2,
            [num_times[-1] + dt_avg/2]
        ])
    time_bounds = np.clip(time_bounds, 1.0, 1e6)

    # ---------- 角度边界 ----------
    pitch_bounds = np.concatenate(([pitch_angles[0]-0.5],
                                   (pitch_angles[:-1]+pitch_angles[1:])/2,
                                   [pitch_angles[-1]+0.5]))

    # ---------- 创建图形，使用 GridSpec 精确控制 ----------
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[1, 10],
                           hspace=0.0, wspace=0.05,
                           left=0.07, right=0.92, bottom=0.12, top=0.92)
    top_ax = plt.subplot(gs[0, 0])
    main_ax = plt.subplot(gs[1, 0], sharex=top_ax)
    cbar_ax = plt.subplot(gs[:, 1])

    # ---------- 配置顶部轴 ----------
    top_ax.set_ylim(0, 1)
    top_ax.set_yticks([])
    for spine in top_ax.spines.values():
        spine.set_visible(False)
    top_ax.tick_params(bottom=False, labelbottom=False)
    top_ax.set_facecolor('none')

    # ---------- 绘制主图 ----------
    mesh = main_ax.pcolormesh(time_bounds, pitch_bounds, flux_clean.T,
                              shading='auto', cmap='jet', norm='log')
    main_ax.set_ylim(0, 180)
    main_ax.set_xlabel('Time (UTC)', fontsize=12)
    main_ax.set_ylabel('Pitch Angle (deg)', fontsize=12)
    main_ax.set_title(f"Electron Pitch Angle Distribution\n{event_info}", fontsize=14)
    main_ax.grid(True, linestyle='--', alpha=0.3)

    # 颜色条放在右侧外部
    cbar = fig.colorbar(mesh, cax=cbar_ax, label='Flux (eV/cm²·s·sr·eV) [log]')
    cbar.ax.tick_params(labelsize=10)

    # 标记 Mark1 和 Mark2
    main_ax.axvline(num_mark1, color='red', linestyle='--', linewidth=1.5, label='Mark1')
    main_ax.axvline(num_mark2, color='orange', linestyle='--', linewidth=1.5, label='Mark2')
    main_ax.legend(loc='upper right', fontsize=10)

    # ---------- 在顶部轴绘制区间线段 ----------
    for start, end in intervals:
        if is_valid_dt(start) and is_valid_dt(end):
            try:
                ns = mdates.date2num(start)
                ne = mdates.date2num(end)
                if 1.0 <= ns <= 1e6 and 1.0 <= ne <= 1e6:
                    top_ax.plot([ns, ne], [0.5, 0.5], color='blue', linewidth=5,
                                solid_capstyle='butt', alpha=0.8)
            except Exception:
                continue

    # 设置 x 轴日期格式（主图自动同步）
    main_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    main_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    # 添加条件说明
    main_ax.text(0.02, 0.95, 'Conditions: F(0°) ≥ 2·F(90°) and F(180°) ≥ 2·F(90°)',
                 transform=main_ax.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # ========== 保存图片 ==========
    if SAVE_FIGURES:
        os.makedirs(SAVE_DIR, exist_ok=True)
        # 从 event_info 中提取事件标识，例如 "Event 2022-07-02_0420 (E ~ 253.3 eV)"
        # 用下划线替换空格，去除特殊字符
        safe_event = event_info.replace(' ', '_').replace('(', '').replace(')', '').replace('~', '')
        safe_event = safe_event.replace('·', '').replace('eV', 'eV')
        # 保留字母数字和下划线
        safe_event = ''.join(c for c in safe_event if c.isalnum() or c in '_-.')
        filename = f"{safe_event}.png"
        filepath = os.path.join(SAVE_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {filepath}")

    plt.show()

def main():
    df_events = pd.read_csv(EVENT_FILE, sep='\t', parse_dates=['Shock','Arrival','End','Mark1','Mark2'])
    print(f"共读取 {len(df_events)} 个激波事件")

    for idx, row in df_events.iterrows():
        mark1 = row['Mark1']
        mark2 = row['Mark2']
        event_id = f"{row['Shock'].date()}_{row['Shock'].hour:02d}{row['Shock'].minute:02d}"
        print(f"\n处理事件 {idx+1}: {event_id}, Mark1={mark1}, Mark2={mark2}")

        start_dt = mark1 - timedelta(minutes=5)
        end_dt = mark2 + timedelta(minutes=5)
        var_names = ['PITCHANGLE', 'ENERGY_VALS', 'EFLUX_VS_PA_E']
        data = read_psp_pad_data(PSP_DATA_ROOT, var_names, start_dt, end_dt)
        if not data:
            print("警告: 未读取到数据，跳过")
            continue

        df_pitch = data.get('PITCHANGLE')
        df_energy = data.get('ENERGY_VALS')
        df_eflux = data.get('EFLUX_VS_PA_E')
        if any(df is None for df in [df_pitch, df_energy, df_eflux]):
            print("警告: 缺少必要变量，跳过")
            continue

        pitch_angles = get_pitch_angles(df_pitch)
        energy_vals = get_energy_vals(df_energy)
        print(f"投掷角数: {len(pitch_angles)}, 能量通道数: {len(energy_vals)}")

        if ENERGY_CHANNEL_INDEX < len(energy_vals):
            e_idx = ENERGY_CHANNEL_INDEX
        else:
            e_idx = len(energy_vals) // 2
            print(f"通道索引超界，自动使用 {e_idx}")
        print(f"使用能量通道 {e_idx}, 能量 = {energy_vals[e_idx]:.2f} eV")

        # 清洗时间数据
        df_eflux = df_eflux.dropna(subset=['t']).reset_index(drop=True)
        if len(df_eflux) == 0:
            print("警告: 去除无效时间后无数据，跳过")
            continue

        times_all = pd.to_datetime(df_eflux['t']).values
        valid_mask = ~pd.isna(times_all)
        if not np.any(valid_mask):
            print("警告: 所有时间均为 NaT，跳过")
            continue
        times_all = times_all[valid_mask]
        flux_all_raw = [row[:, e_idx] for row in df_eflux['EFLUX_VS_PA_E']]
        flux_all = np.array(flux_all_raw)[valid_mask, :]

        # 筛选 Mark1-Mark2
        times_all_dt = np.array(times_all, dtype='datetime64[ns]')
        mark1_dt = np.datetime64(mark1)
        mark2_dt = np.datetime64(mark2)
        mask = (times_all_dt >= mark1_dt) & (times_all_dt <= mark2_dt)
        times_in = times_all_dt[mask]
        flux_in = flux_all[mask, :]

        if len(times_in) == 0:
            print("警告: Mark1-Mark2 内无有效数据，跳过")
            continue

        times = [pd.Timestamp(t) for t in times_in if is_valid_dt(pd.Timestamp(t))]
        if len(times) == 0:
            print("警告: 转换后无有效时间，跳过")
            continue

        # 计算区间
        intervals = compute_condition_intervals(times, flux_in, pitch_angles,
                                                THRESHOLD, ANGLE_TOLERANCE)

        total_duration = (mark2 - mark1).total_seconds()
        interval_duration = sum((end - start).total_seconds() for start, end in intervals if is_valid_dt(start) and is_valid_dt(end))
        ratio = interval_duration / total_duration if total_duration > 0 else 0
        print(f"满足区间总时长: {interval_duration/3600:.2f} h, 占比: {ratio*100:.1f}%")

        # 构造 event_info 字符串
        event_info = f"Event {event_id} (E ~ {energy_vals[e_idx]:.1f} eV)"
        plot_pad_with_intervals(times, flux_in, pitch_angles, intervals,
                                mark1, mark2, event_info, energy_vals[e_idx])

if __name__ == "__main__":
    main()