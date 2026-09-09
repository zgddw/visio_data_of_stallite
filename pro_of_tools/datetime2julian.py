import numpy as np
from datetime import datetime, timedelta
import math


def datetime_to_julian(dt):
    """
    将datetime对象转换为儒略日

    参数:
    dt: datetime - 要转换的datetime对象

    返回:
    float - 儒略日
    """
    # 确保传入的是datetime对象
    if not isinstance(dt, datetime):
        raise ValueError("输入必须是datetime对象")

    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second
    microsecond = dt.microsecond

    # 将时间转换为一天中的小数部分
    fractional_day = (hour + minute / 60.0 + second / 3600.0 + microsecond / 3600000000.0) / 24.0

    # 处理月份和年份
    if month <= 2:
        year -= 1
        month += 12

    a = year // 100
    b = 2 - a + a // 4

    # 儒略日计算公式
    jd = (int(365.25 * (year + 4716)) +
          int(30.6001 * (month + 1)) +
          day + fractional_day + b - 1524.5)

    return jd


def epoch_to_julian(epoch_data):
    """
    将Epoch时间数组转换为儒略日数组

    参数:
    epoch_data: numpy array - Epoch时间数组，包含datetime对象

    返回:
    numpy array - 儒略日数组
    """
    if not isinstance(epoch_data, np.ndarray) or len(epoch_data) == 0:
        raise ValueError("epoch_data必须是非空的numpy数组")

    # 检查第一个元素是否是datetime对象
    if len(epoch_data) > 0 and not isinstance(epoch_data[0], datetime):
        # 尝试转换第一个元素
        try:
            # 如果是numpy数组，尝试提取第一个元素
            if isinstance(epoch_data[0], np.ndarray) and len(epoch_data[0]) > 0:
                first_element = epoch_data[0][0]
            else:
                first_element = epoch_data[0]

            if not isinstance(first_element, datetime):
                raise ValueError("epoch_data必须包含datetime对象或可以转换为datetime的对象")
        except:
            raise ValueError("epoch_data必须包含datetime对象")

    # 使用向量化操作转换所有时间点
    jd_array = np.array([datetime_to_julian(dt) for dt in epoch_data])

    return jd_array


def julian_to_datetime(jd):
    """
    将儒略日转换回datetime对象

    参数:
    jd: float - 儒略日

    返回:
    datetime - 对应的datetime对象
    """
    # 确保导入了正确的datetime类
    from datetime import datetime

    jd += 0.5
    z = int(jd)
    f = jd - z

    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - int(alpha / 4)

    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)

    day = b - d - int(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715

    # 提取小数部分作为时间
    fractional_day = day - int(day)
    hours = fractional_day * 24
    hour = int(hours)
    minutes = (hours - hour) * 60
    minute = int(minutes)
    seconds = (minutes - minute) * 60
    second = int(seconds)
    microseconds = int((seconds - second) * 1e6)

    # 确保day是整数
    day_int = int(day)

    return datetime(year, month, day_int, hour, minute, second, microseconds)


def modified_julian_date(jd):
    """
    计算简化儒略日(MJD)

    参数:
    jd: float - 儒略日

    返回:
    float - 简化儒略日
    """
    # 简化儒略日 = 儒略日 - 2400000.5
    return jd - 2400000.5


def add_julian_to_data(merged_data, epoch_key='Epoch', jd_key='JulianDay', mjd_key='ModifiedJulianDay'):
    """
    将儒略日添加到数据字典中

    参数:
    merged_data: dict - 合并后的数据字典
    epoch_key: str - Epoch数据的键名
    jd_key: str - 儒略日数据的键名
    mjd_key: str - 简化儒略日数据的键名

    返回:
    dict - 添加了儒略日数据的新字典
    """
    if epoch_key not in merged_data:
        raise KeyError(f"在数据中找不到Epoch键: {epoch_key}")

    # 创建数据副本，避免修改原始数据
    result = merged_data.copy()

    # 转换Epoch为儒略日
    epoch_data = result[epoch_key]

    # 检查epoch_data的类型并处理
    if len(epoch_data) > 0:
        # 如果第一个元素是numpy数组，尝试提取datetime对象
        if isinstance(epoch_data[0], np.ndarray):
            print("警告: Epoch数据包含嵌套数组，尝试提取时间数据")
            # 尝试提取时间数据
            try:
                # 假设每个子数组的第一个元素是时间数据
                time_data = np.array([arr[0] for arr in epoch_data])
                jd_data = epoch_to_julian(time_data)
            except Exception as e:
                print(f"无法提取时间数据: {e}")
                # 如果无法提取，返回原始数据
                return result
        else:
            jd_data = epoch_to_julian(epoch_data)
    else:
        jd_data = np.array([])

    result[jd_key] = jd_data

    # 计算简化儒略日
    if len(jd_data) > 0:
        result[mjd_key] = modified_julian_date(jd_data)
    else:
        result[mjd_key] = np.array([])

    return result


def julian_century(jd):
    """
    计算儒略世纪

    参数:
    jd: float - 儒略日

    返回:
    float - 儒略世纪
    """
    # 儒略世纪是从J2000.0起算的儒略日数除以36525
    # J2000.0 = JD 2451545.0
    return (jd - 2451545.0) / 36525.0


def julian_millennium(jd):
    """
    计算儒略千年

    参数:
    jd: float - 儒略日

    返回:
    float - 儒略千年
    """
    # 儒略千年是从J2000.0起算的儒略日数除以365250
    return (jd - 2451545.0) / 365250.0


# 使用示例
if __name__ == "__main__":
    # 示例1: 单个datetime转换
    dt = datetime(2023, 5, 15, 12, 30, 45)
    jd = datetime_to_julian(dt)
    print(f"日期: {dt}")
    print(f"儒略日: {jd}")

    # 转换回datetime验证
    dt_back = julian_to_datetime(jd)
    print(f"转换回的日期: {dt_back}")

    # 示例2: 数组转换
    # 创建一些测试日期
    test_dates = np.array([
        datetime(2023, 1, 1, 0, 0, 0),
        datetime(2023, 1, 1, 12, 0, 0),
        datetime(2023, 1, 2, 0, 0, 0)
    ])

    jd_array = epoch_to_julian(test_dates)
    print(f"日期数组: {test_dates}")
    print(f"儒略日数组: {jd_array}")

    # 示例3: 添加到数据字典
    # 假设我们有合并后的数据
    sample_data = {
        'Epoch': test_dates,
        'BGSE': np.random.rand(3, 3),
        'BGSM': np.random.rand(3, 3)
    }

    enhanced_data = add_julian_to_data(sample_data)
    print(f"增强后的数据键: {list(enhanced_data.keys())}")
    print(f"儒略日数据: {enhanced_data['JulianDay']}")
    print(f"简化儒略日数据: {enhanced_data['ModifiedJulianDay']}")

    # 示例4: 计算儒略世纪和儒略千年
    jd_example = 2460000.0
    jc = julian_century(jd_example)
    jm = julian_millennium(jd_example)
    print(f"儒略日 {jd_example} 对应的儒略世纪: {jc}")
    print(f"儒略日 {jd_example} 对应的儒略千年: {jm}")