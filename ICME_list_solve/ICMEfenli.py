import os


def split_icme_by_satellite(input_filename):
    # 用于存储每个卫星对应的数据行
    satellite_groups = {}

    # 检查文件是否存在
    if not os.path.exists(input_filename):
        print(f"错误：找不到文件 {input_filename}")
        return

    print(f"正在处理文件: {input_filename} ...")

    with open(input_filename, 'r', encoding='utf-8') as f:
        for line in f:
            # 去掉行首尾空格
            clean_line = line.strip()

            # 跳过空行或可能的注释行（通常以 # 开头）
            if not clean_line or clean_line.startswith('#'):
                continue

            # 使用 split() 分割，默认处理空格和制表符
            parts = clean_line.split()

            # 根据数据格式，第二列 (索引为 1) 是卫星名称
            if len(parts) > 1:
                sat_name = parts[1]

                # 如果这个卫星还没在字典里，创建一个新列表
                if sat_name not in satellite_groups:
                    satellite_groups[sat_name] = []

                satellite_groups[sat_name].append(clean_line)

    # 将分类好的数据写入各自的文件
    for sat_name, lines in satellite_groups.items():
        # 清理文件名中可能存在的特殊字符（如果有的话）
        safe_name = sat_name.replace('/', '_').replace('\\', '_')
        output_filename = f"ICME_{safe_name}.txt"

        with open(output_filename, 'w', encoding='utf-8') as out_f:
            out_f.write("\n".join(lines) + "\n")

        print(f"成功导出: {output_filename} (共 {len(lines)} 条记录)")


# 运行程序
if __name__ == "__main__":
    # 请确保该文件名与你上传的文件名一致
    target_file = 'I:\\stdata\\list\\icme\\HELIO4CAST_ICMECAT_v23\\HELIO4CAST_ICMECAT_v23.txt'
    split_icme_by_satellite(target_file)