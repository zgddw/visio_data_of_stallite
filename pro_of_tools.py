import numpy as np
from datetime import datetime, timedelta
import math
import pandas as pd
import warnings
from stname import ALIASES, PLOT_STYLE
import os
import requests
from cdasws import CdasWs
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

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

def build_datasets_multires(data_dict):

    """
    数据格式标准化函数
    完美兼容：
    """
    datasets = {}
    
    # 画图面板分发配置 (包含所有分量和纯量的标准代号)
    panel_config = {
        "Bt": ["Bt"],
        "Bxyz": ["Bx", "By", "Bz"],
        "Vp": ["Vp"],
        "Vxyz": ["Vx", "Vy", "Vz"],
        "Tp": ["Tp"], 
        "Txyz": ["Tx", "Ty", "Tz", "T_par", "T_perp"], 
        "Np": ["Np"],
        "Ne": ["Ne"],
        "Nhe2":["Nhe2"],
        "No6":["No6"],
        "Nfe10":["Nfe10"],
        "Nc":["Nc"],
        "Fluxhe":["Fluxhe"],
        "Fluxo":["Fluxo"],
        "Fluxfe":["Fluxfe"],
        "Fluxc":["Fluxc"],
        "theta": ["theta"],
        "phi": ["phi"],
        "Pdy": ["Pdy"],
        "beta": ["beta"],
        "NevsEnergy": ["NevsEnergy"],
        "SymH": ["SymH"],
        "GIC": ["GIC"]
    }

    for raw_var, obj in data_dict.items():
        if obj is None or getattr(obj, 'empty', False):
            continue
            
        if hasattr(obj, 'empty'):
            t_axis = obj['t'].values
            
            # 获取当前变量所有的有效数据列（排除时间轴 't'）
            data_cols = [c for c in obj.columns if c != 't']
            if not data_cols:
                continue
                
            # 获取这个变量在 ALIASES 里的核心含义 (父名称意义)
            parent_std = ALIASES.get(raw_var, raw_var)
            
            # 判断当前物理量在当前的 DataFrame 里究竟是不是一个“多分量矢量”
            is_vector = len(data_cols) > 1
            
            for col in data_cols:
                base_std = col
                
                # 如果你在 ALIASES 里已经精确指定了这一列的含义，优先用你的定义
                if col in ALIASES:
                    base_std = ALIASES[col]
                    
                # 否则，进入全自动「家族嗅探机制」
                else:
                    # 1. 动态嗅探它的物理学家族（磁场、速度、温度等）
                    lower_name = (parent_std + "_" + col).lower()
                    
                    if 'vel' in lower_name or 'v_' in lower_name or lower_name.startswith('v'):
                        family = "V"
                    elif 'temp' in lower_name or 't_' in lower_name or lower_name.startswith('t'):
                        family = "T"
                    elif 'mag' in lower_name or 'b_' in lower_name or 'bgse' in lower_name or lower_name.startswith('b'):
                        family = "B"
                    elif 'elec' in lower_name or 'density' in lower_name or 'dens' in lower_name or lower_name.startswith('n'):
                        family = "N"
                    elif 'elec_field' in lower_name or 'e_field' in lower_name:
                        family = "E"
                    else:
                        family = parent_std[0].upper() if parent_std else "X"
                    
                    # 2. 根据它究竟是“矢量”还是“纯量”来做最终代号转换
                    if is_vector:
                        # ==========================================
                        # 【分支一】有分量：拼装成 Bx/By/Bz、Vx/Vy/Vz、Tx/Ty/Tz
                        # ==========================================
                        if col.endswith(('_X', '_0', '_x')): 
                            base_std = f"{family}x"
                        elif col.endswith(('_Y', '_1', '_y')): 
                            base_std = f"{family}y"
                        elif col.endswith(('_Z', '_2', '_z')): 
                            base_std = f"{family}z"
                    else:
                        # ==========================================
                        # 【分支二】无分量（纯量）：直接转化为该家族的总标量名
                        # ==========================================
                        if family == "B": base_std = "Bt"    # 磁场总强度
                        elif family == "V": base_std = "Vp"  # 质子总速度
                        elif family == "T": base_std = "Tp"  # 质子标量温度
                        else:
                            base_std = parent_std

                # 3. 寻找对应的画图面板
                target_panel = None
                for panel, vars_in_panel in panel_config.items():
                    if base_std in vars_in_panel:
                        target_panel = panel
                        break
                        
                # 4. 组装数据和样式
                if target_panel:
                    y_data = obj[col].values
                    
                    style_info = PLOT_STYLE.get(base_std, {})
                    unit_raw = style_info.get("unit", "")
                    style_kwargs = style_info.get("style", {})
                    
                    clean_unit = unit_raw.split("\n")[1].strip("()") if "\n" in unit_raw else unit_raw
                    label_str = f"{col} ({clean_unit})" if clean_unit else col
                    
                    line_dict = {
                        "x": t_axis,
                        "y": y_data,
                        "label": label_str,
                        "style": style_kwargs,
                        "unit": unit_raw
                    }
                    
                    datasets.setdefault(target_panel, []).append(line_dict)

    # 按照 panel_config 的顺序进行排序，确保画图时垂直顺序固定
    ordered_datasets = {k: datasets[k] for k in panel_config.keys() if k in datasets}
    
    return ordered_datasets


def data_download(base_dir, dataset, years):
    # 初始化 NASA 客户端
    cdas = CdasWs()
    
    # --------------------------------------------------
    # 阶段一：收集文件，规划存储路径（按月细化 + 全局抗断网版）
    # --------------------------------------------------
    download_tasks = []
    print("【阶段一】正在向 NASA 服务器获取文件清单（采用按月分段检索）...")
    
    for year in years:
        # 按照年份创建独立的文件夹
        year_dir = os.path.join(base_dir, str(year))
        if not os.path.exists(year_dir):
            os.makedirs(year_dir)
            
        print(f" -> 📂 开始检索 {year} 年的数据...")
        
        # 将一年拆分为 12 个月逐个检索，避免单次请求过大导致被 NASA 挂断
        for month in range(1, 13):
            start_time = f"{year}-{month:02d}-01T00:00:00Z"
            # 计算结束时间：如果是12月，跨年；否则是下个月1号前一秒
            if month == 12:
                end_time = f"{year}-12-31T23:59:59Z"
            else:
                end_time = f"{year}-{(month+1):02d}-01T00:00:00Z"
                
            try:
                # 核心防线：用最广泛的 Exception 拦截 NASA 库内部的所有连接断开和报错
                status, result = cdas.get_original_files(dataset, start_time, end_time)
                
                if status == 200 and result is not None:
                    # 兼容 NASA 返回的 list 或单个对象
                    file_list = result if isinstance(result, list) else [result]
                        
                    month_task_count = 0
                    for file_info in file_list:
                        file_url = file_info.get('Name') if isinstance(file_info, dict) else file_info
                        if file_url:
                            file_name = file_url.split('/')[-1]
                            save_path = os.path.join(year_dir, file_name)
                            
                            download_tasks.append({
                                'url': file_url,
                                'save_path': save_path,
                                'file_name': file_name,
                                'year': year
                            })
                            month_task_count += 1
                    
                    if month_task_count > 0:
                        print(f"    [成功] {year}年{month:02d}月：找到 {month_task_count} 个文件")
                
                elif status == 404:
                    pass # 404 说明该月份确实没数据，静默跳过
                else:
                    print(f"    [提示] {year}年{month:02d}月：无数据（状态码:{status}）")
                    
            except Exception as e:
                # 当网络断开（RemoteDisconnected）或没数据触发Bug时，精准拦截，不影响接下来的月份
                print(f"    ⚠️ [跳过] {year}年{month:02d}月检索失败（服务器拒绝或无数据），已自动切入下个月。")
                continue

    total_files = len(download_tasks)
    print(f"\n清单收集完毕！所有年份总计需要下载 {total_files} 个文件。\n")
    
    if total_files == 0:
        print("❌ 未能收集到任何待下载文件，请检查是否需要开启科学上网代理。")
        return

    # --------------------------------------------------
    # 配置网络请求会话（阶段二和三共用）
    # --------------------------------------------------
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    # 💡 强力建议：如果你有科学上网软件，请取消下面几行的注释并修改端口（比如 7890 或 10809）
    # session.proxies = {
    #     "http": "http://127.0.0.1:7890",
    #     "https": "http://127.0.0.1:7890",
    # }

    # --------------------------------------------------
    # 阶段二：执行批量下载循环
    # --------------------------------------------------
    print("【阶段二】开始跨年份批量下载...")
    for idx, task in enumerate(download_tasks, 1):
        url = task['url']
        save_path = task['save_path']
        file_name = task['file_name']
        
        print(f"[{idx}/{total_files}] 目标: /{task['year']}/ -> {file_name}")
        
        try:
            head_res = session.head(url, timeout=15)
            remote_size = int(head_res.headers.get('content-length', 0))
            task['remote_size'] = remote_size
            
            if os.path.exists(save_path):
                if os.path.getsize(save_path) == remote_size and remote_size > 0:
                    print(" -> [跳过] 本地已存在该文件且大小一致。")
                    continue
                else:
                    os.remove(save_path)
            
            response = session.get(url, stream=True, timeout=(30, 90))
            response.raise_for_status()
            
            with open(save_path, 'wb') as f, tqdm(
                total=remote_size, unit='iB', unit_scale=True, desc=f"下载中"
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    size = f.write(chunk)
                    bar.update(size)
                    
        except Exception as e:
            print(f" -> [临时失败] 下载错误: {e}")
            task['remote_size'] = task.get('remote_size', -1)

    # --------------------------------------------------
    # 阶段三：全量下载结束后的【整体最终复检】
    # --------------------------------------------------
    print("\n【阶段三】所有文件下载尝试已结束，正在进行全量最终复检...")
    failed_reports = []
    success_count = 0

    for task in download_tasks:
        save_path = task['save_path']
        url = task['url']
        
        remote_size = task.get('remote_size', -1)
        if remote_size <= 0:
            try:
                head_res = session.head(url, timeout=10)
                remote_size = int(head_res.headers.get('content-length', 0))
            except:
                remote_size = -1

        if not os.path.exists(save_path):
            failed_reports.append((task, "文件未成功下载"))
        else:
            local_size = os.path.getsize(save_path)
            if remote_size > 0 and local_size != remote_size:
                failed_reports.append((task, f"大小不匹配(本地:{local_size}, 远程:{remote_size})"))
            elif local_size < 1000:
                failed_reports.append((task, "文件过小，可能下载到了错误网页"))
            else:
                success_count += 1

    print("\n================== 最终全量复检盘点报告 ==================")
    print(f" 计划下载总数: {total_files} 个")
    print(f" 成功通过校验: {success_count} 个")
    print(f" 最终失败/损坏: {len(failed_reports)} 个")
    print("==========================================================")

    if failed_reports:
        print("\n❌ 以下文件未能下载成功：")
        for task, reason in failed_reports:
            print(f" 📂 [年份 {task['year']}] {task['file_name']} -> 原因: {reason}")
        print("\n💡 提示：直接再次运行本脚本，程序会直接去补抓上面失败的文件。")
    else:
        print("\n🎉 完美！所有年份的数据文件均已成功下载，且全部通过了复检！")

# 如果在 test.py 中直接运行测试，可以保留下面这行
# data_download("./psp_raw_data", 'PSP_FLD_L2_MAG_RTN', [2022, 2023, 2024])
# 日期转换实例
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