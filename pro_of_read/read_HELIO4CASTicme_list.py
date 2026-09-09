import pandas as pd
from datetime import datetime

def get_all_icme_times(file_path):
    """
    读取 ICME 列表，返回包含 (激波时间, 本体开始时间, 结束时间) 的列表。
    如果没有激波，激波时间返回 None。
    """
    results = []

    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None)

        for _, row in df.iterrows():
            shock_str = row[2]  # 第三列：激波时间
            start_str = row[3]  # 第四列：本体开始时间
            end_str = row[4]    # 第五列：结束时间

            t_start = pd.to_datetime(start_str)
            t_end = pd.to_datetime(end_str)

            if shock_str == start_str:
                t_shock = None
            else:
                t_shock = pd.to_datetime(shock_str)

            icmeid = f"{t_start.year}.{t_start.month}.{t_start.day}"
            eventname = "ICME"

            # ✅ 追加到列表
            results.append({
                "id": icmeid,
                "name": eventname,
                "start": t_start,
                "end": t_end,
                "shock": t_shock,
            })

        return results

    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return []


# file_name = r'E:\Reserch\CME&ICMErelation\事例筛选\psp&wind\wind.txt'
# all_events = get_all_icme_times(file_name)
#
# print(f"成功读取事件总数: {len(all_events)}")
# print("\n前 5 个事件的时间数据 (None 表示无激波):")
# print("-" * 60)
# for i, event in enumerate(all_events[:5], 1):
#     print(f"事件 {i}: Shock={event['shock']}, Start={event['start']}, End={event['end']}")
