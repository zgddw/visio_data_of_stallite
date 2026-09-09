import os
from datetime import datetime, timedelta

def read_gic_data(data_root,start_time,end_time,station_prefix="man"):
    """
    读取指定时间段内的 GIC 数据

    Parameters
    ----------
    data_root : str
        数据根目录（按年份存储）
    start_time : datetime
        起始时间（UTC）
    end_time : datetime
        结束时间（UTC）
    station_prefix : str
        站点文件名前缀，例如 "man"

    Returns
    -------
    times : list of datetime
        时间序列
    values : list of float
        对应的 GIC 数值
    """

    times = []
    values = []

    current_date = start_time.date()
    end_date = end_time.date()

    while current_date <= end_date:
        year = current_date.year
        ymd = current_date.strftime("%Y%m%d")

        file_path = os.path.join(
            data_root,
            f"man{str(year)}",
            f"{station_prefix}{ymd}.txt"
        )

        if not os.path.isfile(file_path):
            current_date += timedelta(days=1)
            continue

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()

                # 跳过注释和空行
                if not line or line.startswith("%"):
                    continue

                parts = line.split()

                # 跳过第一行 YYYY MM DD 0
                if len(parts) == 4 and parts[3] == "0":
                    file_year = int(parts[0])
                    file_month = int(parts[1])
                    file_day = int(parts[2])
                    continue

                # 正常数据行
                if len(parts) == 4:
                    hh, mm, ss = map(int, parts[:3])
                    value = float(parts[3])

                    t = datetime(
                        file_year, file_month, file_day,
                        hh, mm, ss
                    )

                    if start_time <= t <= end_time:
                        times.append(t)
                        values.append(value)

        current_date += timedelta(days=1)
        result = {
            't':times,
            'GIC':values
        }
    return result

# data_root = r"I:\GIC_data"

# start_time = datetime(1999, 1, 1, 0, 10, 0)
# end_time   = datetime(1999, 2, 1, 1, 0, 0)

# times, values = read_gic_data(
#     data_root,
#     start_time,
#     end_time
# )

# print(times[:5])
# print(values[:5])
