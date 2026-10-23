from datetime import timedelta
from pro_of_read import read_gic_data
def extract_gic_events(times,values,threshold=3.0,merge_gap_minutes=30):
    """
    从 GIC 时间序列中提取并合并事件

    Returns
    -------
    events : list of dict
        每个 dict 包含：
        {
            'start': datetime,
            'end': datetime,
            'peak': float
        }
    """

    events = []
    in_event = False

    for t, v in zip(times, values):
        if abs(v) > threshold:
            if not in_event:
                # 事件开始
                start_time = t
                peak = abs(v)
                in_event = True
            else:
                peak = max(peak, abs(v))
            end_time = t
        else:
            if in_event:
                events.append({
                    "start": start_time,
                    "end": end_time,
                    "peak": peak
                })
                in_event = False

    # 如果数据在事件中结束
    if in_event:
        events.append({
            "start": start_time,
            "end": end_time,
            "peak": peak
        })

    # ===== 合并相邻事件 =====
    merged_events = []
    merge_gap = timedelta(minutes=merge_gap_minutes)

    for ev in events:
        if not merged_events:
            merged_events.append(ev)
            continue

        last = merged_events[-1]
        if ev["start"] - last["end"] < merge_gap:
            # 合并
            last["end"] = ev["end"]
            last["peak"] = max(last["peak"], ev["peak"])
        else:
            merged_events.append(ev)

    return merged_events

def save_events_to_txt(events, output_file):
    """
    将事件列表保存为 txt 文件
    """

    with open(output_file, "w") as f:
        f.write("# Start_Time (UTC)    End_Time (UTC)      Duration(min)   Peak_|GIC|\n")

        for ev in events:
            duration = (ev["end"] - ev["start"]).total_seconds() / 60.0

            f.write(
                f"{ev['start']}   "
                f"{ev['end']}   "
                f"{duration:8.2f}   "
                f"{ev['peak']:8.2f}\n"
            )

def filter_events_by_peak(events, peak_threshold=10.0):
    """
    在事件列表中筛选峰值大于等于指定阈值的事件

    Parameters
    ----------
    events : list of dict
        extract_gic_events 返回的事件列表
    peak_threshold : float
        峰值阈值（A）

    Returns
    -------
    filtered_events : list of dict
    """

    filtered_events = [
        ev for ev in events if ev["peak"] >= peak_threshold
    ]

    return filtered_events

from datetime import datetime

# 1. 读取数据
result = read_gic_data(
    data_root=r"I:\GIC_data",
    start_time=datetime(1999, 1, 1, 0, 0, 0),
    end_time=datetime(2020, 12, 31, 23, 59, 59)
)
times = result['t']
values = result['GIC']

# 2. 提取 |GIC| > 3 A 的事件并合并
events_gt_3 = extract_gic_events(
    times,
    values,
    threshold=3.0,
    merge_gap_minutes=60
)

# 3. 在上述事件中筛选峰值 ≥ 10 A
events_peak_ge_10 = filter_events_by_peak(
    events_gt_3,
    peak_threshold=10.0
)

# 4. 保存结果
save_events_to_txt(
    events_peak_ge_10,
    "GIC_events_abs_gt_3_peak_ge_10_60min.txt"
)