import pro_of_tools

base_dir = r"H:\solo\swa\SOLO_L2_SWA-EAS1-NM3D-PSD"        # 数据下载的总根目录
dataset = 'SOLO_L2_SWA-EAS1-NM3D-PSD'     # 卫星数据集 ID
years = [2020,2021,2022,2023,2024,2025,2026] 
pro_of_tools.data_download(base_dir,dataset,years)