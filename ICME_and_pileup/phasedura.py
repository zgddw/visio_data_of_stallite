import datetime
import statistics
import re

def parse_datetime(t_str):
    """解析时间字符串，支持带微秒和不带微秒的格式"""
    t_str = t_str.strip()
    try:
        return datetime.datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S.%f")

def process_file(filename, is_solo):
    """
    处理数据文件，返回符合条件的三个时长列表
    is_solo: True 表示 solo 列表，False 表示 PSP 列表
    """
    t1_list = []  # 阶段1: 激波 → 密度堆积开始
    t2_list = []  # 阶段2: 密度堆积持续时间
    t3_list = []  # 阶段3: ICME 本体持续时间

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 跳过标题行
    start_idx = 0
    if is_solo:
        # solo 第一行以 '#' 开头
        while start_idx < len(lines) and lines[start_idx].strip().startswith('#'):
            start_idx += 1
    else:
        # PSP 第一行是列名
        if lines and "Shock" in lines[0]:
            start_idx = 1

    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue

        # 提取所有时间戳（可能带微秒）和数字（整数或浮点数），顺序与列顺序一致
        tokens = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?|\d+\.\d+|\d+', line)

        if is_solo:
            # 列顺序: 序号, 激波, ICME开始, ICME结束, Mark1, Mark2, 均值, 背景, Ratio, 持续时长
            if len(tokens) < 10:
                continue
            shock_str = tokens[1]
            icme_start_str = tokens[2]
            icme_end_str = tokens[3]
            mark1_str = tokens[4]
            mark2_str = tokens[5]
            ratio_str = tokens[8]
        else:
            # PSP列顺序: 激波, ICME开始, ICME结束, Mark1, Mark2, 内部均值, 前均值, 后均值, 外部最大, Ratio, 持续时长
            if len(tokens) < 11:
                continue
            shock_str = tokens[0]
            icme_start_str = tokens[1]
            icme_end_str = tokens[2]
            mark1_str = tokens[3]
            mark2_str = tokens[4]
            ratio_str = tokens[9]

        try:
            ratio = float(ratio_str)
        except ValueError:
            continue

        # 筛选条件 1: Ratio >= 0.5
        if ratio < 0.5:
            continue

        # 解析时间
        shock_dt = parse_datetime(shock_str)
        icme_start_dt = parse_datetime(icme_start_str)
        icme_end_dt = parse_datetime(icme_end_str)
        mark1_dt = parse_datetime(mark1_str)
        mark2_dt = parse_datetime(mark2_str)

        # 筛选条件 2: 有激波（激波时间 != ICME开始时间）
        if shock_dt == icme_start_dt:
            continue

        # 计算三个阶段时长（小时）
        t1 = (mark1_dt - shock_dt).total_seconds() / 3600.0
        t2 = (mark2_dt - mark1_dt).total_seconds() / 3600.0
        icme_start_effective = max(icme_start_dt, mark2_dt)  # 若堆积结束晚于原ICME开始，则取堆积结束
        t3 = (icme_end_dt - icme_start_effective).total_seconds() / 3600.0

        # 过滤非法负值
        if t1 >= 0 and t2 >= 0 and t3 >= 0:
            t1_list.append(t1)
            t2_list.append(t2)
            t3_list.append(t3)

    return t1_list, t2_list, t3_list


def main():
    psp_file = r"E:\ZhaoWenjin_Reserch\pileup\有激波psp.txt"
    solo_file = r"E:\ZhaoWenjin_Reserch\pileup\solo列表.txt"

    t1_all, t2_all, t3_all = [], [], []

    # 处理 PSP 文件
    t1, t2, t3 = process_file(psp_file, is_solo=False)
    t1_all.extend(t1)
    t2_all.extend(t2)
    t3_all.extend(t3)

    # 处理 SOLO 文件
    t1, t2, t3 = process_file(solo_file, is_solo=True)
    t1_all.extend(t1)
    t2_all.extend(t2)
    t3_all.extend(t3)

    if not t1_all:
        print("没有符合条件的事件（有激波且 Ratio ≥ 0.5）。")
        return

    avg_t1 = statistics.mean(t1_all)
    avg_t2 = statistics.mean(t2_all)
    avg_t3 = statistics.mean(t3_all)

    print(f"符合条件的总事件数: {len(t1_all)}")
    print(f"阶段1 (激波→密度堆积开始) 平均时长: {avg_t1:.3f} 小时")
    print(f"阶段2 (密度堆积区持续时间) 平均时长: {avg_t2:.3f} 小时")
    print(f"阶段3 (ICME本体持续时间) 平均时长: {avg_t3:.3f} 小时")


if __name__ == "__main__":
    main()