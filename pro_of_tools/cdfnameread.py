import spacepy.pycdf as pycdf
import sys
import os


def list_cdf_variables_spacepy(cdf_file_path):
    """
    使用spacepy.pycdf读取CDF文件并输出所有变量的名称

    参数:
        cdf_file_path (str): CDF文件路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(cdf_file_path):
            print(f"错误: 文件 '{cdf_file_path}' 不存在")
            return

        # 打开CDF文件
        with pycdf.CDF(cdf_file_path) as cdf_file:
            # 获取所有变量
            variables = list(cdf_file.keys())

            print(f"CDF文件: {os.path.basename(cdf_file_path)}")
            print(f"包含 {len(variables)} 个变量:")
            print("-" * 50)
            # 按字母顺序排序并输出变量名称
            print(f"{variables},")

    except Exception as e:
        print(f"读取CDF文件时出错: {str(e)}")


def main():
    """主函数"""
    cdf_file_path = r'I:\stdata\psp\isois-epihi\let1\2022\psp_isois-epihi_l2-let1-rates60_20220216_v25.cdf'
    list_cdf_variables_spacepy(cdf_file_path)




if __name__ == "__main__":
    main()