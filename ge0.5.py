# filter_ratio.py
import sys

def filter_ratio(input_file, output_file, threshold=0.5):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        print("文件为空")
        return

    # 解析标题行，找出 Ratio 列的索引
    header = lines[0].strip().split()
    try:
        ratio_idx = header.index('Ratio')
    except ValueError:
        print("未找到 'Ratio' 列，请检查文件格式")
        return

    # 保留标题行
    filtered = [lines[0]]

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) <= ratio_idx:
            continue
        try:
            ratio = float(parts[ratio_idx])
            if ratio > threshold:
                filtered.append(line + '\n')  # 恢复换行符
        except ValueError:
            # 若转换失败则跳过该行
            continue

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(filtered)

    print(f"已提取 {len(filtered)-1} 行，保存至 {output_file}")

if __name__ == '__main__':
    filter_ratio('annotations_results.txt', 'filtered_results.txt')