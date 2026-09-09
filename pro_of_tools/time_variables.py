import spacepy.pycdf as cdf

def debug_time_variables(cdf_file):
    """调试函数：检查CDF文件中的时间变量属性"""
    try:
        with cdf.CDF(cdf_file) as cdf_file_obj:
            all_vars = list(cdf_file_obj.keys())
            time_vars = [var for var in all_vars if 'Epoch' in var or 'Time' in var]

            print(f"文件: {cdf_file}")
            print("时间变量信息:")
            for var in time_vars:
                data = cdf_file_obj[var][...]
                print(f"  {var}:")
                print(f"    数据类型: {data.dtype}")
                print(f"    数据形状: {data.shape}")
                if len(data) > 0:
                    print(f"    第一个值: {data[0]}")
                    print(f"    第一个值类型: {type(data[0])}")

                # 检查变量属性
                attrs = cdf_file_obj[var].attrs
                print(f"    属性: {list(attrs.keys())}")
                for attr in attrs:
                    try:
                        print(f"      {attr}: {attrs[attr]}")
                    except:
                        print(f"      {attr}: [无法显示]")
                print()

    except Exception as e:
        print(f"调试时出错: {e}")


# 在读取文件前先调试
debug_time_variables('E:/study/reserach/ab_icme/V_1/1995/wi_h0_mfi_19950101_v05.cdf')