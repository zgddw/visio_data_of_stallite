import os
import glob
import bisect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import spacepy.pycdf
from matplotlib.widgets import Button
plt.rcParams['font.sans-serif'] = [
    'SimHei',               # Windows 黑体
    'Microsoft YaHei',      # Windows 微软雅黑
    'WenQuanYi Micro Hei',  # Linux 文泉驿微米黑
    'Noto Sans CJK SC',     # Linux/Windows Noto 简体中文
    'DejaVu Sans'           # 通用后备字体（不支持中文但避免报错）
]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# ---------- 自定义 ALIASES（如有需要可在此定义） ----------
ALIASES = {}

# ---------- cdfdataread 函数（从原文件复制，已修正导入） ----------
def cdfdataread(root_path, var_names, start_time, end_time, file_prefix):
    start_dt = start_time
    end_dt   = end_time

    date_list = []
    d = start_dt.date()
    while d <= end_dt.date():
        date_list.append(d)
        d += timedelta(days=1)

    var_data_tmp = {v: {"t": [], "data": [], "attrs": {}} for v in var_names}

    for date_obj in date_list:
        year = date_obj.strftime('%Y')
        day  = date_obj.strftime('%Y%m%d')
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
                i1 = bisect.bisect_right(times, end_dt)   if date_obj == end_dt.date()   else len(times)
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

        std_name = ALIASES.get(v, v)

        df_single = pd.DataFrame()
        df_single['t'] = pd.to_datetime(t_arr)

        if arr.ndim == 2:
            df_single[std_name] = arr[:, 0] if arr.shape[1] > 0 else np.full(len(df_single), np.nan)
        elif arr.ndim == 1:
            if len(arr) == len(df_single):
                df_single[std_name] = arr
            else:
                print(f"[WARN] 变量 {std_name} 数据长度 ({len(arr)}) 与时间轴 ({len(df_single)}) 不匹配，已跳过。")
                continue
        else:
            df_single[std_name] = [slice_ for slice_ in arr]

        final_output[v] = df_single

    return final_output

# ---------- 交互式标注应用（使用按钮） ----------
class ICMEAnnotationApp:
    def __init__(self, events, data_root, file_prefix, output_file="annotations.txt"):
        self.events = events
        self.data_root = data_root
        self.file_prefix = file_prefix
        self.output_file = output_file
        self.idx = 0
        self.marks = []
        self.density_data = None
        self.fig = None
        self.ax = None
        self.lines = []

        # 初始化输出文件
        with open(self.output_file, 'w') as f:
            f.write("# Event_Index  Start_Time  End_Time  Mark1  Mark2  Mean_interval  Max_background  Ratio  Duration_hours\n")

        # 进入交互模式
        plt.ion()   # 打开交互模式
        self.plot_current_event()
        plt.show(block=True)  # 进入主循环，等待按钮事件

    def plot_current_event(self):
        """绘制当前事件，并创建按钮"""
        # 如果已有图形，先关闭
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

        # 检查是否已处理完所有事件
        if self.idx >= len(self.events):
            print("所有事件已处理完毕。")
            plt.close('all')
            return

        ev = self.events[self.idx]
        start_dt = ev['start']
        end_dt = ev['end']
        shock_dt = ev['shock']

        # 读取数据（前后各扩展3小时）
        read_start = start_dt - timedelta(hours=6)
        read_end = end_dt + timedelta(hours=6)
        if shock_dt is not None and shock_dt < read_start:
            read_start = shock_dt - timedelta(hours=6)

        var_names = ['N']
        try:
            data_dict = cdfdataread(self.data_root, var_names, read_start, read_end, self.file_prefix)
        except Exception as e:
            print(f"读取数据失败: {e}")
            self.idx += 1
            self.plot_current_event()
            return

        if 'N' not in data_dict or data_dict['N'].empty:
            print(f"事件 {self.idx+1} 无有效密度数据，跳过。")
            self.idx += 1
            self.plot_current_event()
            return

        self.density_data = data_dict['N']
        self.marks = []

        # 创建图形和轴
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.fig.subplots_adjust(bottom=0.2)  # 为按钮留出空间

        # 绘制密度曲线
        self.ax.plot(self.density_data['t'], self.density_data['N'], 'b-', linewidth=1, label='Density (N)')
        self.ax.set_xlabel('Time (UTC)')
        self.ax.set_ylabel('Density (cm$^{-3}$)')
        self.ax.grid(True, linestyle='--', alpha=0.5)

        ymin, ymax = self.ax.get_ylim()
        if shock_dt is not None:
            self.ax.axvline(shock_dt, color='red', linestyle='--', linewidth=1.5, label='Shock')
        self.ax.axvline(start_dt, color='green', linestyle='--', linewidth=1.5, label='Start')
        self.ax.axvline(end_dt, color='orange', linestyle='--', linewidth=1.5, label='End')

        self.ax.set_title(f"Event {self.idx+1}/{len(self.events)}  (Click to mark two points)")
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.fig.autofmt_xdate()
        self.ax.legend(loc='upper right')

        # 连接鼠标点击事件（用于添加标记）
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # ---------- 创建按钮 ----------
        # 按钮位置：底部四个按钮
        ax_undo = plt.axes([0.10, 0.05, 0.15, 0.075])      # 撤销
        ax_next = plt.axes([0.30, 0.05, 0.15, 0.075])      # 保存并下一事件
        ax_skip = plt.axes([0.50, 0.05, 0.15, 0.075])      # 跳过事件
        ax_quit = plt.axes([0.70, 0.05, 0.15, 0.075])      # 退出

        btn_undo = Button(ax_undo, 'Undo')
        btn_next = Button(ax_next, 'Save & Next')
        btn_skip = Button(ax_skip, 'Skip')
        btn_quit = Button(ax_quit, 'Quit')

        # 绑定回调函数
        btn_undo.on_clicked(self.undo_callback)
        btn_next.on_clicked(self.next_callback)
        btn_skip.on_clicked(self.skip_callback)
        btn_quit.on_clicked(self.quit_callback)

        # 保存按钮引用以防止被垃圾回收
        self.buttons = (btn_undo, btn_next, btn_skip, btn_quit)

        # 清空标记线
        self.lines = []
        self.fig.canvas.draw()

    def on_click(self, event):
        """鼠标点击添加标记"""
        if event.inaxes != self.ax:
            return
        if len(self.marks) >= 2:
            print("已标记两个点，不能再添加。")
            return
        x = event.xdata
        if x is None:
            return
        t_mark = mdates.num2date(x).replace(tzinfo=None)
        if t_mark < self.density_data['t'].min() or t_mark > self.density_data['t'].max():
            print("点击位置超出数据时间范围，请重新选择。")
            return
        self.marks.append(t_mark)
        self.update_plot_marks()

    def update_plot_marks(self):
        """更新图形上的标记线/点"""
        for line in self.lines:
            line.remove()
        self.lines.clear()
        for t in self.marks:
            line = self.ax.axvline(t, color='magenta', linestyle='-', linewidth=2, alpha=0.7)
            self.lines.append(line)
            idx = (self.density_data['t'] - t).abs().idxmin()
            y_val = self.density_data.loc[idx, 'N']
            point = self.ax.plot(t, y_val, 'mo', markersize=8)[0]
            self.lines.append(point)
        self.fig.canvas.draw()

    # ---------- 按钮回调函数 ----------
    def undo_callback(self, event):
        """撤销最后一个标记"""
        if self.marks:
            self.marks.pop()
            self.update_plot_marks()
            print(f"已撤销，当前标记数: {len(self.marks)}")
        else:
            print("没有可撤销的标记。")

    def next_callback(self, event):
        """保存标注并进入下一事件"""
        if len(self.marks) < 2:
            print("请先标记两个时间点，或使用 'Skip' 跳过。")
            return
        self.save_current()
        self.idx += 1
        self.plot_current_event()

    def skip_callback(self, event):
        """跳过当前事件，不保存标注"""
        print(f"跳过事件 {self.idx+1}")
        self.idx += 1
        self.plot_current_event()

    def quit_callback(self, event):
        """退出程序"""
        print("退出程序。")
        plt.close('all')
        exit()

    def save_current(self):
        """计算并保存当前标注结果"""
        ev = self.events[self.idx]
        t1, t2 = sorted(self.marks)
        duration = (t2 - t1).total_seconds() / 3600.0

        df = self.density_data
        mask = (df['t'] >= t1) & (df['t'] <= t2)
        if mask.sum() == 0:
            print("区间内无数据点，跳过保存。")
            return
        mean_interval = df.loc[mask, 'N'].mean()

        t1_before = t1 - timedelta(hours=2)
        mask_before = (df['t'] >= t1_before) & (df['t'] <= t1)
        mean_before = df.loc[mask_before, 'N'].mean() if mask_before.sum() > 0 else np.nan

        t2_after = t2 + timedelta(hours=2)
        mask_after = (df['t'] >= t2) & (df['t'] <= t2_after)
        mean_after = df.loc[mask_after, 'N'].mean() if mask_after.sum() > 0 else np.nan

        valid_bg = [m for m in [mean_before, mean_after] if not np.isnan(m)]
        if not valid_bg:
            print("前后背景平均值无效，无法计算比值。")
            ratio = np.nan
            max_bg = np.nan
        else:
            max_bg = max(valid_bg)
            ratio = (mean_interval - max_bg) / max_bg

        with open(self.output_file, 'a') as f:
            line = f"{self.idx+1}  {ev['start']}  {ev['end']}  {t1}  {t2}  {mean_interval:.4f}  {max_bg:.4f}  {ratio:.4f}  {duration:.2f}\n"
            f.write(line)

        print(f"事件 {self.idx+1} 结果已保存。")

# ---------- 主程序 ----------
def main():
    list_file = r"E:\ZhaoWenjin_Reserch\ICME\结果图像\SOLO\list.txt"
    if not os.path.exists(list_file):
        print(f"错误：找不到 {list_file}")
        return

    df_list = pd.read_csv(list_file, parse_dates=['shock_Datetime', 'Start_Datetime', 'End_Datetime'])
    events = []
    for _, row in df_list.iterrows():
        shock = row['shock_Datetime']
        start = row['Start_Datetime']
        end = row['End_Datetime']
        if shock == start:
            shock = None
        events.append({'shock': shock, 'start': start, 'end': end})

    print(f"共读取到 {len(events)} 个事件。")

    data_root = r"I:\stdata\solo\swa\solo_l2_swa-pas-grnd-mom"
    file_prefix = "solo_l2_swa-pas-grnd-mom"
    output_file = "annotations_results.txt"

    app = ICMEAnnotationApp(events, data_root, file_prefix, output_file)

if __name__ == "__main__":
    main()