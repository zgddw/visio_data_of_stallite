import os
import datetime
import numpy as np
from spacepy import datamodel as dm
from pro_of_read import read_omni_hro_1min


def identify_and_save_storms(times, values, threshold, output_file, merge_threshold_hours=24):
    """识别磁暴、合并近距离事件并保存"""
    # 1. 基础清洗
    values = values.astype(float)
    values[values > 2000] = np.nan

    # 2. 识别原始磁暴索引段 (SYM-H < threshold)
    mask = (values < threshold) & (~np.isnan(values))
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    if mask[0]: starts = np.insert(starts, 0, 0)
    if mask[-1]: ends = np.append(ends, len(mask) - 1)

    if len(starts) == 0:
        print("未发现满足条件的磁暴。")
        return

    # 3. 转化为时间段列表 [(start_idx, end_idx), ...]
    initial_intervals = list(zip(starts, ends))

    # 4. 合并逻辑：如果两个段之间间隔 < 24小时，合并为一个段
    merged_intervals = []
    if initial_intervals:
        curr_s, curr_e = initial_intervals[0]

        for i in range(1, len(initial_intervals)):
            next_s, next_e = initial_intervals[i]

            # 计算前一个结束到下一个开始的时间差
            gap_duration = times[next_s] - times[curr_e]

            if gap_duration < datetime.timedelta(hours=merge_threshold_hours):
                # 合并：保持当前的开始，更新结束位置为下一个的结束
                curr_e = next_e
                # 注：合并后，两个段之间的那段 SYM-H > -30 的数据也会被包含在内统计
            else:
                # 间隔足够大，保存当前段，开始新段
                merged_intervals.append((curr_s, curr_e))
                curr_s, curr_e = next_s, next_e

        merged_intervals.append((curr_s, curr_e))

    # 5. 计算统计量并保存
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"磁暴筛选报告 (阈值: {threshold} nT, 合并间隔: {merge_threshold_hours}h)\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'开始时间':<20} | {'结束时间':<20} | {'最小值(nT)':<12} | {'平均值(nT)':<12} | {'持续(h)':<8}\n")
        f.write("-" * 90 + "\n")

        for s, e in merged_intervals:
            storm_times = times[s:e + 1]
            storm_vals = values[s:e + 1]

            duration = (storm_times[-1] - storm_times[0]).total_seconds() / 3600
            min_val = np.nanmin(storm_vals)
            avg_val = np.nanmean(storm_vals)

            start_str = storm_times[0].strftime('%Y-%m-%d %H:%M')
            end_str = storm_times[-1].strftime('%Y-%m-%d %H:%M')

            f.write(f"{start_str:<20} | {end_str:<20} | {min_val:<12.2f} | {avg_val:<12.2f} | {duration:<8.1f}\n")

        f.write("-" * 90 + "\n")
        f.write(f"处理完成。共计导出 {len(merged_intervals)} 个合并后的磁暴事件。\n")

    print(f"合并筛选完成！结果已存至: {output_file}")


# --- 执行 ---
if __name__ == "__main__":
    data_root = r'I:\stdata\omni\hro_1min'
    # 建议查询较长的时间范围以测试合并效果
    start_dt = datetime.datetime(1994, 1, 1)
    end_dt = datetime.datetime(2024, 12, 31)

    # 获取数据
    times, values = read_omni_hro_1min.get_omni_data_structured(start_dt, end_dt, 'SYM_H', data_root)

    if len(times) > 0:
        # threshold=-30, merge_threshold_hours=24
        identify_and_save_storms(times, values, -30, 'storm_list_merged.txt', 24)
    else:
        print("未读取到数据。")