import os
import glob
import bisect
import numpy as np
import pandas as pd
import spacepy.pycdf
from datetime import datetime, timedelta
from stname import ALIASES 

#=============================================================
# cdfdataread 用于读取 cdf 数据文件，支持多分辨率独立输出
"""
输入参数：
root_path   : 数据文件根目录
var_names   : 需要读取的参数名称列表 (例如: ['B1GSE', 'BGSE'])
start_time  : datetime 类型，数据读取的起始时间
end_time    : datetime 类型，数据读取的结束时间
file_prefix : 数据文件前缀

输出参数：
final_output: 字典类型 (dict)。键为变量名，值为该变量专属的 pandas.DataFrame，
              每个 DataFrame 都自带独立的 't' 列。
"""
def cdfdataread(  
    root_path,
    var_names,
    start_time,
    end_time,
    file_prefix
):
    start_dt = start_time
    end_dt   = end_time

    # 生成日期列表
    date_list = []
    d = start_dt.date()
    while d <= end_dt.date():
        date_list.append(d)
        d += timedelta(days=1)

    # 临时容器：直接按变量名（var_names）分别存放各自的数据与时间轴
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
                
                # 1. 获取该变量专属的时间轴依赖 (DEPEND_0)
                dep_name = cdf[v].attrs.get('DEPEND_0', 'Epoch')
                if hasattr(dep_name, 'encode'):
                    dep_name = str(dep_name)
                
                if dep_name not in cdf:
                    continue

                # 2. 提取并保留 CDF 元数据用于去异常值
                if not var_data_tmp[v]["attrs"]:
                    var_data_tmp[v]["attrs"] = {
                        'FILLVAL': cdf[v].attrs.get('FILLVAL'),
                        'VALIDMIN': cdf[v].attrs.get('VALIDMIN'),
                        'VALIDMAX': cdf[v].attrs.get('VALIDMAX')
                    }

                times = cdf[dep_name][...]
                
                # 3. 使用二分法定位时间切片
                i0 = bisect.bisect_left(times, start_dt) if date_obj == start_dt.date() else 0
                i1 = bisect.bisect_right(times, end_dt)   if date_obj == end_dt.date()   else len(times)
                
                if i0 >= i1:
                    continue

                var_data_tmp[v]["t"].append(times[i0:i1])
                var_data_tmp[v]["data"].append(cdf[v][i0:i1])

    # ===== 核心处理阶段：去异常、降维、构建多变量独立 DataFrame =====
    final_output = {}

    for v in var_names:
        if not var_data_tmp[v]["t"]:
            print(f"[WARN] No data found for variable: {v}")
            continue

        # 1. 合并时间戳，并使用 np.squeeze() 强制降维
        # 完美解决 Wind MFI 卫星中时间轴形状为 (N, 1) 导致的 TypeError
        t_arr = np.concatenate(var_data_tmp[v]["t"])
        t_arr = np.squeeze(t_arr)

        # 2. 合并数据并强制转换为 float64 类型以支持 np.nan
        arr = np.asarray(np.concatenate(var_data_tmp[v]["data"]), dtype=float)

        # 3. 动态去异常值
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

        # 兜底物理极值清洗
        
        invalid |= (np.abs(arr) > 1e25)

        if invalid.any():
            arr[invalid] = np.nan

        # 4. 获取可能存在的标准化名称
        std_name = ALIASES.get(v, v)

        # 5. 为当前变量创建独立的 DataFrame (自带 't' 列表)
        df_single = pd.DataFrame()
        df_single['t'] = pd.to_datetime(t_arr)

        # 6. 多维解包逻辑
        if arr.ndim == 2:
            v_lower = v.lower()
            std_lower = std_name.lower()
            # 识别是否为磁场/速度等三分量矢量
            is_vector = any(k in v_lower or k in std_lower for k in ['psp_fld_l2_mag_RTN_1min','b', 'mag', 'field', 'gse', 'v', 'vel'])
            
            if arr.shape[1] == 3 and is_vector:
                # 为了区分多变量(如 B1GSE 和 BGSE)，将其原变量名作为前缀
                prefix = std_name if std_name not in ["Bx", "By", "Bz", "Bt", "Vx", "Vy", "Vz"] else v
                df_single[f'{prefix}_X'] = arr[:, 0]
                df_single[f'{prefix}_Y'] = arr[:, 1]
                df_single[f'{prefix}_Z'] = arr[:, 2]
            else:
                # 能谱或其他2D矩阵按行打包，保持 DataFrame 为 1D
                df_single[std_name] = [row for row in arr]
                
        elif arr.ndim == 1:
            # ==========================================
            # 安全校验：确保数据长度与时间轴长度严格一致
            # ==========================================
            if len(arr) == len(df_single):
                df_single[std_name] = arr
            elif len(arr) == 0:
                print(f"[WARN] 变量 {std_name} 在此时段内无有效数据 (长度为0)，已自动跳过。")
                # 数据为空，直接跳过赋值，我们的下游分拣机非常聪明，会自动忽略它
            else:
                print(f"[WARN] 变量 {std_name} 数据长度 ({len(arr)}) 与时间轴 ({len(df_single)}) 不匹配，已跳过。")
            # ==========================================
        else:
            # 高维数据防错
            df_single[std_name] = [slice_ for slice_ in arr]

        # 存入最终的输出字典
        final_output[v] = df_single

    return final_output


# import matplotlib.pyplot as plt
# import pandas as pd
# from datetime import datetime
# # 确保导入了你最新修改的 cdfdataread

# start_time = datetime(2023, 1, 1, 0, 0, 0)
# end_time = datetime(2023, 1, 2, 0, 0, 0)
# root_path = r"I:\stdata\wind\mfi\wi_h0_mfi"
# var_names = ["B1GSE", "BGSE"]
# file_prefix = "wi_h0_mfi"

# # 1. 读取数据（此时 data 是一个字典）
# data = cdfdataread(root_path, var_names, start_time, end_time, file_prefix)

# # 2. 修改了这里的判断逻辑！字典不能用 .empty
# if data:
#     print("成功读取到的变量:", list(data.keys()))
    
#     # 创建 3 个子图，分别对比 X, Y, Z 三个分量
#     fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
#     components = ['X', 'Y', 'Z']
    
#     # 从字典中安全提取对应的 DataFrame
#     df_b1 = data.get("B1GSE", pd.DataFrame())
#     df_bgse = data.get("BGSE", pd.DataFrame())
    
#     for i, (ax, comp) in enumerate(zip(axes, components)):
#         # --- 绘制高分辨率 B1GSE ---
#         col_b1 = f"B1GSE_{comp}"  # 根据新版 cdfdataread，列名应该是 B1GSE_X, B1GSE_Y 等
#         if not df_b1.empty and col_b1 in df_b1.columns:
#             ax.plot(df_b1['t'], df_b1[col_b1], 
#                     color='tab:blue', linewidth=0.5, alpha=0.7, label=f"B1GSE_{comp} (High-Res)")
        
#         # --- 绘制低分辨率 BGSE ---
#         col_bgse = f"BGSE_{comp}"
#         if not df_bgse.empty and col_bgse in df_bgse.columns:
#             ax.plot(df_bgse['t'], df_bgse[col_bgse], 
#                     color='tab:orange', linewidth=1.2, label=f"BGSE_{comp} (Low-Res)")
                
#         ax.set_ylabel(f"B_{comp} (nT)", fontsize=11)
#         ax.grid(True, linestyle="--", alpha=0.5)
#         ax.legend(loc="upper right")
        
#     axes[0].set_title("Wind/MFI: B1GSE vs BGSE Resolution Comparison", fontsize=13)
#     axes[-1].set_xlabel("Time (UTC)", fontsize=11)
    
#     # 自动格式化横坐标时间轴
#     plt.gcf().autofmt_xdate()
#     plt.tight_layout()
#     plt.show()
# else:
#     print("[WARN] 读取到的数据字典为空，请检查路径或文件是否存在。")


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
