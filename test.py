import pro_of_tools

base_dir = r"I:\stdata\solo\mag\RTN_NORMAL"        # 数据下载的总根目录
dataset = 'SOLO_L2_MAG-RTN-NORMAL'     # 卫星数据集 ID
years = [2024,2025] 
pro_of_tools.data_download(base_dir,dataset,years)