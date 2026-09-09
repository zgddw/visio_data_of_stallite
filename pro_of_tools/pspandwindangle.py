from spacepy import pycdf
import numpy as np
import datetime

def extract_conjunctions_with_distance(wind_file, psp_file, output_file="conjunction_analysis.txt"):
    # 1. 打开 CDF 文件
    with pycdf.CDF(wind_file) as cdf_wind, pycdf.CDF(psp_file) as cdf_psp:
        # 提取时间、经度 (HGI_LON) 和 径向距离 (RAD_AU)
        t_wind = cdf_wind['Epoch'][...]
        lon_wind = cdf_wind['HGI_LON'][...]
        r_wind = cdf_wind['RAD_AU'][...]
        lat_wind = cdf_wind['HGI_LAT'][...]
        lat_psp = cdf_psp['HGI_LAT'][...]
        t_psp = cdf_psp['Epoch'][...]
        lon_psp = cdf_psp['HGI_LON'][...]
        r_psp = cdf_psp['RAD_AU'][...]

    # 2. 时间对齐
    common_times, idx_w, idx_p = np.intersect1d(t_wind, t_psp, return_indices=True)
    
    if len(common_times) == 0:
        print("未找到重叠时间段。")
        return

    # 提取对齐后的数据
    lons_w, lons_p = lon_wind[idx_w], lon_psp[idx_p]
    rs_w, rs_p = r_wind[idx_w], r_psp[idx_p]
    lats_w, lats_p = lat_wind[idx_w], lat_psp[idx_p]
    # 3. 计算经度差 (考虑 0/360 度循环)
    lon_diff = np.abs(lons_w - lons_p)
    lon_diff = np.where(lon_diff > 180, 360 - lon_diff, lon_diff)
    lat_diff = np.abs(lats_w - lats_p)
    # 计算径向距离差 (Wind 通常在 1AU，PSP 较小，差值反映了空间间隔)
    r_diff_abs = np.abs(rs_w - rs_p)

    # 4. 识别经度差时段
    lon_condition = lon_diff < 30#(lon_diff >= 30) & (lon_diff <= 60)
    lat_condition = (lat_diff < 5)
    condition = lon_condition & lat_condition
    periods = []
    start_idx = None

    for i in range(len(condition)):
        if condition[i] and start_idx is None:
            start_idx = i
        elif not condition[i] and start_idx is not None:
            # 记录该时段内的径向距离差统计量
            current_r_diffs = r_diff_abs[start_idx : i]
            periods.append({
                'start': common_times[start_idx],
                'end': common_times[i-1],
                'min_dr': np.min(current_r_diffs),
                'max_dr': np.max(current_r_diffs)
            })
            start_idx = None
    
    # 5. 保存并格式化输出
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Wind & PSP 卫星对准分析 (HGI经度差 < 30°)\n")
        f.write("-" * 85 + "\n")
        header = f"{'开始时间':^22} | {'结束时间':^22} | {'最小距离差(AU)':^15} | {'最大距离差(AU)':^15}\n"
        f.write(header)
        f.write("-" * 85 + "\n")
        
        for p in periods:
            f.write(f"{str(p['start']):22} | {str(p['end']):22} | {p['min_dr']:15.4f} | {p['max_dr']:15.4f}\n")

    print(f"分析完成！共提取 {len(periods)} 个时段。")
    print(f"数据已保存至: {output_file}")

# 执行
winddata = r'I:\stdata\wind\wind_helio1hr_position_19941102_v01.cdf'
pspdata = r'I:\stdata\psp\psp_helio1hr_position\psp_helio1hr_position_20180813_v01.cdf'
output_file = r'E:\ZhaoWenjin_Reserch\CME&ICMErelation\事例筛选\psp&wind\low5and30anglr.txt'
extract_conjunctions_with_distance(winddata, pspdata,output_file=output_file)