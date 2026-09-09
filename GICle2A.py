import os
import glob
import numpy as np

def count_gic_less_than_2A_recursive(root_dir):
    """
    递归遍历 root_dir 及其所有子文件夹，统计所有 man*.txt 文件中
    GIC < 2.0 的点数占总有效点数的百分比。
    """
    total_points = 0
    less_than_2 = 0

    # 使用 ** 递归匹配所有子目录中的 man*.txt 文件
    file_pattern = os.path.join(root_dir, "man**", "man*.txt")
    file_list = glob.glob(file_pattern, recursive=True)

    if not file_list:
        print(f"未找到匹配的文件，请检查路径：{root_dir}")
        return

    print(f"找到 {len(file_list)} 个文件，开始处理...")

    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # 跳过以 '%' 开头的注释行
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('%')]

        for line in data_lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                gic = abs(float(parts[3]))
                if np.isnan(gic):
                    continue
                total_points += 1
                if gic < 3.0:
                    less_than_2 += 1
            except ValueError:
                # 忽略转换失败的行（如含 'NaN' 字符串）
                continue

    if total_points == 0:
        print("未读取到任何有效数据。")
        return

    percentage = (less_than_2 / total_points) * 100
    print(f"总有效数据点数：{total_points}")
    print(f"GIC < 2.0 A 的点数：{less_than_2}")
    print(f"占比：{percentage:.6f}%")

if __name__ == "__main__":
    # 请将下面的路径改为存放所有年份文件夹的根目录
    root_directory = "I:\GIC_data"   # 例如：r"C:\Users\xxx\GASUM"
    count_gic_less_than_2A_recursive(root_directory)