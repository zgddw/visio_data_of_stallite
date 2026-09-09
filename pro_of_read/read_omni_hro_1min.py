import os
import datetime
from spacepy import datamodel as dm
import numpy as np

def read_omni_hro_1min(root_path, var_names, start_time, end_time):
    """
    读取 OMNI HRO 1min CDF 数据，输出格式与 daycdfread.py 保持一致
    """
    # 临时容器，模仿 daycdfread 的结构
    tmp = {
        "t": [],
        "vars": {v: [] for v in var_names}
    }

    # 从起始时间的月初开始迭代
    current_date = datetime.datetime(start_time.year, start_time.month, 1,0,0,0)

    while current_date <= end_time:
        year_str = str(current_date.year)
        month_str = current_date.strftime('%Y%m')

        # 构建路径: 根目录/年份/omni_hro_1min_年月01_v01.cdf
        file_name = f"omni_hro_1min_{month_str}01_v01.cdf"
        file_path = os.path.join(root_path, year_str, file_name)

        if os.path.exists(file_path):
            print(f"正在读取: {file_path}")
            try:
                # 使用 spacepy.datamodel 读取
                data_cdf = dm.fromCDF(file_path)

                if 'Epoch' in data_cdf:
                    times = np.array(data_cdf['Epoch'])
                    
                    # 筛选时间范围的掩码
                    mask = (times >= start_time) & (times <= end_time)
                    
                    if np.any(mask):
                        # 存储时间轴切片
                        tmp["t"].append(times[mask])
                        
                        # 存储各个变量的切片
                        for v in var_names:
                            if v in data_cdf:
                                tmp["vars"][v].append(np.array(data_cdf[v])[mask])
                            else:
                                # 如果文件中缺失某个变量，填充对应长度的 NaN
                                fill_nan = np.full(np.sum(mask), np.nan)
                                tmp["vars"][v].append(fill_nan)
                else:
                    print(f"警告: 文件中未找到 Epoch: {file_path}")

            except Exception as e:
                print(f"读取文件时发生错误: {e}")
        else:
            print(f"跳过: 文件未找到 {file_path}")

        # 月份递进逻辑
        if current_date.month == 12:
            current_date = datetime.datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime.datetime(current_date.year,current_date.month + 1, 1, 1)

    # ===== 数据合并与格式化 (参考 daycdfread 逻辑) =====
    if not tmp["t"]:
        print(f"[WARN] No data found: {start_time} ~ {end_time}")
        return {}

    data = {}
    data["t"] = np.concatenate(tmp["t"])

    for v in var_names:
        if not tmp["vars"][v]:
            continue
            
        # 1. 合并并转换为 float
        arr = np.asarray(np.concatenate(tmp["vars"][v]), dtype=float)

        # 2. 识别无效值 (参考 daycdfread 的 1e25 标准)
        # OMNI 常见的填充值有 999.9, 9999.9, 或极小的负数
        invalid = (
            ~np.isfinite(arr) | 
            (np.abs(arr) > 1e25) | 
            (arr <= -1e25) |
            (arr >= 999.0)   # 针对 OMNI 特有的高数值填充项
        )

        if invalid.any():
            arr[invalid] = np.nan
            
        data[v] = arr
        
    return data


# --- 使用示例 ---
if __name__ == "__main__":
    data_root = r'I:\stdata\omni\hro_1min'
    
    start_dt = datetime.datetime(2023, 12, 30, 0, 0)
    end_dt = datetime.datetime(2024, 1, 2, 23, 59)
    
    # 现在可以像 daycdfread 一样传入变量列表
    target_vars = ['SYM_H', 'BX_GSE', 'BY_GSM']

    res = read_omni_hro_1min(data_root, target_vars, start_dt, end_dt)

    if res:
        print("\n" + "=" * 30)
        print(f"数据读取成功！")
        print(f"时间轴长度: {len(res['t'])}")
        for v in target_vars:
            if v in res:
                valid_count = np.count_nonzero(~np.isnan(res[v]))
                print(f"变量 {v}: 有效点数 {valid_count}")
        print("=" * 30)