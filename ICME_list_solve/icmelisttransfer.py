import os
import pandas as pd

def convert_psp_catalog(input_path, output_path):
    """
    将原始空分/带有ISO时间的ICME清单转换为标准的 mainplot.py 所需的CSV清单
    """
    if not os.path.exists(input_path):
        print(f"❌ 找不到输入的原始文件: {input_path}")
        return

    parsed_records = []
    
    print("⏳ 开始解析原始清单并标准化时间格式...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # 🛡️ 核心过滤：只处理以科学事件特征开头的行，安全跳过换行引起的残缺行或备注
            if not line.startswith("ICME_"):
                continue
            
            # 按空格/制表符切分
            parts = line.split()
            if len(parts) < 5:
                print(f"⚠️ 警告: 第 {line_num} 行数据列数不足，已跳过。")
                continue
            
            # 提取时间字符串列 (包含 T 和 Z 标识)
            shock_raw = parts[2]
            start_raw = parts[3]
            end_raw = parts[4]
            
            try:
                # 🌟 利用 pandas 强大的 to_datetime 自动脱掉 'T' 和 'Z'，并标准化为 %Y-%m-%d %H:%M:%S
                shock_dt = pd.to_datetime(shock_raw).strftime('%Y-%m-%d %H:%M:%S')
                start_dt = pd.to_datetime(start_raw).strftime('%Y-%m-%d %H:%M:%S')
                end_dt = pd.to_datetime(end_raw).strftime('%Y-%m-%d %H:%M:%S')
                
                parsed_records.append({
                    "shock_Datetime": shock_dt,
                    "Start_Datetime": start_dt,
                    "End_Datetime": end_dt
                })
            except Exception as e:
                print(f"⚠️ 第 {line_num} 行时间解析失败，已跳过。错误原因: {e}")
                
    # 转换成 DataFrame 并保存为逗号分隔的标准 CSV 格式
    if parsed_records:
        df_out = pd.DataFrame(parsed_records)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存文件
        df_out.to_csv(output_path, index=False, sep=',')
        print(f"\n🎉 转换大成功！新清单已保存至: {output_path}")
        print(f"📊 共成功提取并标准化了 {len(df_out)} 条事件。")
        print("\n💡 转换后的前 5 行数据预览：")
        print(df_out.head())
    else:
        print("❌ 转换失败：未在文件中提取到任何有效的 ICME_ 事件数据。")

if __name__ == "__main__":
    # ================= 配置你的文件路径 =================
    # 1. 刚才下载或存放原始 PSP 文本的路径
    raw_txt_path = r"I:\list\icme\HELIO4CAST_ICMECAT_v23\divided\ICME_PSP.txt" 
    
    # 2. 转换后生成的、供你的 mainplot.py 读取的那个目标文件路径
    target_csv_path = r"E:\ZhaoWenjin_Reserch\ICME\结果图像\PSP\list.txt"  
    # ===================================================

    convert_psp_catalog(raw_txt_path, target_csv_path)