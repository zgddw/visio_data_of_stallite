import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button
from datetime import datetime, timedelta
from pro_of_read import cdfdataread  # 确保该文件可导入
plt.rcParams['font.sans-serif'] = [
    'SimHei',               # Windows 黑体
    'Microsoft YaHei',      # Windows 微软雅黑
    'WenQuanYi Micro Hei',  # Linux 文泉驿微米黑
    'Noto Sans CJK SC',     # Linux/Windows Noto 简体中文
    'DejaVu Sans'           # 通用后备字体（不支持中文但避免报错）
]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# ========== 配置 ==========
CDF_ROOT = r"I:\stdata\psp\sweap\psp_swp_spc_l3i"
FILE_PREFIX = "psp_swp_spc_l3i"
VAR_NAME = "np_moment"

# 默认列表路径（可根据需要修改）
DEFAULT_LIST_SHOCK = r"E:\ZhaoWenjin_Reserch\ICME\结果图像\PSP\能谱\有激波\存在密度增强的\list.txt"
DEFAULT_LIST_NOSHOCK = r"E:\ZhaoWenjin_Reserch\ICME\结果图像\PSP\能谱\无激波\存在密度增强的\list.txt"  # 请根据实际路径修改

# ========== 选择模式 ==========
print("请选择要处理的事件类型：")
print("1. 有激波事件")
print("2. 无激波事件")
print("3. 自定义列表文件路径")
choice = input("请输入数字 (1/2/3): ").strip()

if choice == '1':
    list_path = DEFAULT_LIST_SHOCK
    out_suffix = "shock"
elif choice == '2':
    list_path = DEFAULT_LIST_NOSHOCK
    out_suffix = "noshock"
elif choice == '3':
    list_path = input("请输入列表文件的完整路径: ").strip()
    out_suffix = "custom"
else:
    print("无效选择，退出。")
    exit()

# 如果路径不存在，提示并退出
if not os.path.isfile(list_path):
    print(f"错误：文件不存在 - {list_path}")
    exit()

OUTPUT_FILE = f"annotation_results_{out_suffix}.txt"

print(f"使用列表文件: {list_path}")
print(f"结果将保存至: {OUTPUT_FILE}")

# ========== 读取事件列表 ==========
events = []
with open(list_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) != 3:
            print(f"跳过格式错误行: {line}")
            continue
        try:
            shock = datetime.strptime(parts[0].strip(), '%Y-%m-%d %H:%M:%S')
            arrival = datetime.strptime(parts[1].strip(), '%Y-%m-%d %H:%M:%S')
            end = datetime.strptime(parts[2].strip(), '%Y-%m-%d %H:%M:%S')
            events.append((shock, arrival, end))
        except Exception as e:
            print(f"解析时间失败: {line}, 错误: {e}")

if not events:
    print("未读取到任何事件，请检查文件格式。")
    exit()

# ========== 处理单个事件的函数（完全复用之前的逻辑） ==========
def process_event(event_idx, shock, arrival, end):
    # 读取数据（前后各扩展6小时）
    start_read = shock - timedelta(hours=6)
    end_read = end + timedelta(hours=6)
    print(f"事件 {event_idx+1}: 读取数据从 {start_read} 到 {end_read}")
    
    try:
        data_dict = cdfdataread(
            root_path=CDF_ROOT,
            var_names=[VAR_NAME],
            start_time=start_read,
            end_time=end_read,
            file_prefix=FILE_PREFIX
        )
    except Exception as e:
        print(f"读取CDF失败: {e}")
        return None, None, None

    if VAR_NAME not in data_dict or data_dict[VAR_NAME].empty:
        print(f"事件 {event_idx+1}: 未找到变量 {VAR_NAME} 或数据为空，跳过")
        return None, None, None

    df = data_dict[VAR_NAME]
    # 确定密度列名
    if VAR_NAME not in df.columns:
        cols = [c for c in df.columns if c != 't']
        if cols:
            col_name = cols[0]
            print(f"使用列 '{col_name}' 作为密度数据")
        else:
            print("无有效数据列，跳过")
            return None, None, None
    else:
        col_name = VAR_NAME

    times = pd.to_datetime(df['t'])
    density = df[col_name].values
    mask = np.isfinite(density)
    t_plot = times[mask]
    d_plot = density[mask]

    if len(t_plot) == 0:
        print(f"事件 {event_idx+1}: 无有效密度数据，跳过")
        return None, None, None

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t_plot, d_plot, 'b-', linewidth=0.8, label='np_moment')
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Proton Density (cm$^{-3}$)')
    ax.grid(True, alpha=0.3)
    title = f"Event {event_idx+1}: Shock={shock}, Arrival={arrival}, End={end}\n"
    title += "Click to mark two time points (red dashed lines). Use Undo to remove last."
    ax.set_title(title)

    # 绘制原始三个时间竖线
    ax.axvline(shock, color='green', linestyle='--', linewidth=1.5, label='Shock')
    ax.axvline(arrival, color='orange', linestyle='--', linewidth=1.5, label='Arrival')
    ax.axvline(end, color='red', linestyle='--', linewidth=1.5, label='End')
    ax.legend()

    mark_lines = []  # 存储竖线对象
    mark_times = []  # 存储标注时间

    def onclick(event):
        if event.inaxes != ax:
            return
        if len(mark_times) >= 2:
            print("已标注两个点，请使用撤销或点击下一张。")
            return
        t = mdates.num2date(event.xdata).replace(tzinfo=None)
        if t < times.min() or t > times.max():
            print("点击时间超出数据范围，请重新点击。")
            return
        line = ax.axvline(t, color='red', linestyle='--', linewidth=1.5)
        mark_lines.append(line)
        mark_times.append(t)
        ax.set_title(f"{title}\n已标记 {len(mark_times)} / 2 个点")
        fig.canvas.draw()
        print(f"标记点 {len(mark_times)}: {t}")

    def undo(event):
        if mark_lines:
            line = mark_lines.pop()
            line.remove()
            mark_times.pop()
            ax.set_title(f"{title}\n已标记 {len(mark_times)} / 2 个点")
            fig.canvas.draw()
            print("撤销最后一个标记")
        else:
            print("没有可撤销的标记")

    def next_event(event):
        if len(mark_times) != 2:
            print("请先标注两个时间点！")
            return
        plt.close(fig)

    # 添加按钮
    ax_undo = plt.axes([0.7, 0.02, 0.1, 0.05])
    btn_undo = Button(ax_undo, '撤销')
    btn_undo.on_clicked(undo)

    ax_next = plt.axes([0.85, 0.02, 0.1, 0.05])
    btn_next = Button(ax_next, '下一张')
    btn_next.on_clicked(next_event)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(mark_times) == 2:
        return mark_times[0], mark_times[1], df, col_name
    else:
        return None, None, None, None

# ========== 主循环 ==========
results = []
for idx, (shock, arrival, end) in enumerate(events):
    print(f"\n========== 处理事件 {idx+1}/{len(events)} ==========")
    t1, t2, df, col = process_event(idx, shock, arrival, end)
    if t1 is None or t2 is None:
        print(f"事件 {idx+1} 未完成标注，跳过")
        continue

    if t1 > t2:
        t1, t2 = t2, t1

    # 计算内部平均
    mask_inside = (df['t'] >= t1) & (df['t'] <= t2)
    density_inside = df.loc[mask_inside, col].values
    mean_inside = np.nanmean(density_inside) if len(density_inside) > 0 else np.nan

    # 前两小时
    before_start = t1 - timedelta(hours=2)
    mask_before = (df['t'] >= before_start) & (df['t'] < t1)
    density_before = df.loc[mask_before, col].values
    mean_before = np.nanmean(density_before) if len(density_before) > 0 else np.nan

    # 后两小时
    after_end = t2 + timedelta(hours=2)
    mask_after = (df['t'] > t2) & (df['t'] <= after_end)
    density_after = df.loc[mask_after, col].values
    mean_after = np.nanmean(density_after) if len(density_after) > 0 else np.nan

    max_outside = max(mean_before, mean_after) if not np.isnan(mean_before) or not np.isnan(mean_after) else np.nan
    if np.isnan(max_outside) or max_outside == 0:
        ratio = np.nan
    else:
        ratio = (mean_inside - max_outside) / max_outside

    duration = (t2 - t1).total_seconds() / 3600.0

    results.append({
        'shock': shock,
        'arrival': arrival,
        'end': end,
        't1': t1,
        't2': t2,
        'mean_inside': mean_inside,
        'mean_before': mean_before,
        'mean_after': mean_after,
        'max_outside': max_outside,
        'ratio': ratio,
        'duration_hours': duration
    })

# ========== 写入输出文件 ==========
with open(OUTPUT_FILE, 'w') as f:
    header = ["Shock", "Arrival", "End", "Mark1", "Mark2",
              "Mean_Inside", "Mean_Before", "Mean_After",
              "Max_Outside", "Ratio", "Duration_h"]
    f.write("\t".join(header) + "\n")
    for r in results:
        line = [
            r['shock'].strftime('%Y-%m-%d %H:%M:%S'),
            r['arrival'].strftime('%Y-%m-%d %H:%M:%S'),
            r['end'].strftime('%Y-%m-%d %H:%M:%S'),
            r['t1'].strftime('%Y-%m-%d %H:%M:%S'),
            r['t2'].strftime('%Y-%m-%d %H:%M:%S'),
            f"{r['mean_inside']:.4f}",
            f"{r['mean_before']:.4f}",
            f"{r['mean_after']:.4f}",
            f"{r['max_outside']:.4f}",
            f"{r['ratio']:.4f}",
            f"{r['duration_hours']:.2f}"
        ]
        f.write("\t".join(line) + "\n")

print(f"\n所有事件处理完成，结果已保存至 {OUTPUT_FILE}")