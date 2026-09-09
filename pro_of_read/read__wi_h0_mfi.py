import os
import glob
import numpy as np
import spacepy.pycdf
from datetime import datetime
import bisect


def read_wind_data_spacepy(root_path, var_names, start_time_str, end_time_str):
    """
    使用 spacepy 读取 Wind 卫星 CDF 数据。

    参数:
        root_path (str): 数据保存的根目录 (目录下应包含按年份命名的文件夹，如 '1995').
        var_names (list): 需要读取的变量名称列表，例如 ['BF1', 'B3GSM'].
        start_time_str (str): 开始时间，格式 'YYYY-MM-DD HH:MM:SS'.
        end_time_str (str): 结束时间，格式 'YYYY-MM-DD HH:MM:SS'.

    返回:
        dict: 按时间分辨率分类的数据字典。
    """

    # 1. 解析时间字符串为 datetime 对象
    try:
        start_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        print("错误：时间格式必须为 'YYYY-MM-DD HH:MM:SS'")
        return None

    # 2. 定位文件路径
    # 根据数据说明，文件按年份存储
    year_str = start_dt.strftime('%Y')
    date_str = start_dt.strftime('%Y%m%d')

    # 构建搜索路径: root_path/YYYY/wi_h0_mfi_YYYYMMDD_v*.cdf
    search_pattern = os.path.join(root_path, year_str, f"wi_h0_mfi_{date_str}_v*.cdf")
    found_files = glob.glob(search_pattern)

    if not found_files:
        print(f"错误：在路径 {os.path.join(root_path, year_str)} 下未找到日期 {date_str} 的数据文件。")
        return None

    # 获取对应日期的文件（如果有多个版本，默认取最后一个，通常是最新版本）
    file_path = sorted(found_files)[-1]
    print(f"正在读取文件: {file_path}")

    results = {}

    # 3. 使用 spacepy 打开 CDF 文件
    try:
        with spacepy.pycdf.CDF(file_path) as cdf:
            # 遍历用户请求的每个变量
            for var_name in var_names:
                if var_name not in cdf:
                    print(f"警告: 变量 {var_name} 不在文件中，已跳过。")
                    continue

                # 获取变量对象
                var_obj = cdf[var_name]

                # --- 获取元数据信息 ---
                # 根据数据说明，读取 DEPEND_0 (依赖的时间轴) 和 TIME_RES (时间分辨率)
                # 例如 BF1 依赖 Epoch ，分辨率为 1 min
                try:
                    # attrs.get 返回的可能是 byte string，视文件版本而定，这里做个转换以防万一
                    depend_0 = var_obj.attrs.get('DEPEND_0', 'Epoch')
                    if hasattr(depend_0, 'decode'): depend_0 = depend_0.decode('utf-8')

                    time_res = var_obj.attrs.get('TIME_RES', 'Unknown')
                    if hasattr(time_res, 'decode'): time_res = time_res.decode('utf-8')
                except Exception as e:
                    print(f"获取变量 {var_name} 属性时出错: {e}")
                    depend_0 = 'Epoch'
                    time_res = 'Unknown'

                # --- 检查时间轴 ---
                if depend_0 not in cdf:
                    print(f"错误: 变量 {var_name} 依赖的时间轴 {depend_0} 不存在。")
                    continue

                # 获取完整的时间数据 (spacepy 会自动转换为 python datetime 对象数组)
                # 这一步将时间轴加载到内存。对于日文件，这通常很快。
                times_all = cdf[depend_0][...]

                # --- 时间切片 ---
                # 使用 bisect 算法在有序的时间数组中快速找到起止索引
                idx_start = bisect.bisect_left(times_all, start_dt)
                idx_end = bisect.bisect_right(times_all, end_dt)

                if idx_start >= idx_end:
                    print(f"提示: 变量 {var_name} 在 {start_time_str} 到 {end_time_str} 期间无数据。")
                    continue

                # 提取对应范围内的数据和时间
                # 注意：spacepy 支持切片操作
                data_sliced = var_obj[idx_start:idx_end]
                times_sliced = times_all[idx_start:idx_end]

                # --- 组织输出结果 ---
                if time_res not in results:
                    results[time_res] = {
                        'timestamps': None,  # 该分辨率下的时间戳
                        'data': {}
                    }

                # 存储时间戳（如果该分辨率尚未存储）
                if results[time_res]['timestamps'] is None:
                    results[time_res]['timestamps'] = times_sliced

                # 存储变量数据
                results[time_res]['data'][var_name] = data_sliced

    except spacepy.pycdf.CDFError as e:
        print(f"CDF读取错误 (请检查是否安装了 NASA CDF C 库): {e}")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None

    return results

#返回值的结构如下所示：
"""{
    "1 min": {
        "timestamps": [datetime对象列表...],
        "data": {
            "BF1": [数值...],
            "BGSM": [数值...]
        }
    },
    "3 sec": {
        "timestamps": [datetime对象列表...],
        "data": {
            "B3GSM": [数值...]
        }
    }
}"""
# ==========================================
# 示例调用部分
# ==========================================
if __name__ == "__main__":
    # 配置路径 (请修改为实际路径)
    # 假设路径结构为 ./Data/Wind/1996/wi_h0_mfi_19961225_v04.cdf
    my_root_path = "I:\\stdata\\wind\\mfi\\wi_h0_mfi"

    # 想要读取的变量，混合了不同分辨率
    # BF1 是 1分钟分辨率
    # B3GSM 是 3秒分辨率
    target_vars = ['BF1', 'B3GSM']

    # 输入时间精确到秒
    t_start = "1996-12-25 12:00:00"
    t_end = "1997-1-26 12:05:00"

    data = read_wind_data_spacepy(my_root_path, target_vars, t_start, t_end)

    if data:
        print("\n读取成功，数据摘要:")
        for res, content in data.items():
            print(f"\n--- 分辨率: {res} ---")
            ts = content['timestamps']
            print(f"时间范围: {ts[0]} 到 {ts[-1]}")
            print(f"数据点数: {len(ts)}")

            for var, val in content['data'].items():
                print(f"  变量: {var}")
                print(f"  数据形状: {val.shape}")
                print(f"  前3个值: {val[:3]}")