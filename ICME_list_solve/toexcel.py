import pandas as pd

# 读取文本文件，去除空行
with open(r'E:\ZhaoWenjin_Reserch\pileup\solo列表.txt', 'r', encoding='utf-8') as f:
    lines = [line.rstrip('\n') for line in f if line.strip()]

# 第一行为列名，按两个空格分割
header = lines[0].split('  ')  # 注意是两个空格

# 解析数据行
data_rows = []
for line in lines[1:]:
    parts = line.split('  ')
    # 如果某行分割后列数不一致，可根据需要处理（这里假设格式正确）
    if len(parts) == len(header):
        data_rows.append(parts)
    else:
        print(f"警告：第 {len(data_rows)+2} 行列数不匹配，已跳过")

# 构建 DataFrame
df = pd.DataFrame(data_rows, columns=header)

# 转换日期时间列（根据实际列名）
datetime_cols = ['Shock_Datetime', 'Start_Time', 'End_Time', 'Mark1', 'Mark2']
for col in datetime_cols:
    df[col] = pd.to_datetime(df[col])

# 转换数值列
numeric_cols = ['Mean_interval', 'Max_background', 'Ratio', 'Duration_hours']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col])

# 导出为 Excel（需要安装 openpyxl 或 xlsxwriter）
df.to_excel('solo列表.xlsx', index=False, engine='openpyxl')

print("转换完成！生成文件：solo列表.xlsx")