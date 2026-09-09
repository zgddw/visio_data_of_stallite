import pandas as pd
import numpy as np

def resample_to_1min(data_dict):
    """
    将 read 函数返回的数据字典平滑/重采样为 1 分钟均值。
    """
    if not data_dict or 't' not in data_dict:
        return {}

    # 1. 构建 DataFrame
    # 强制确保 't' 是一维的 (针对你之前遇到的 2D array 问题)
    df = pd.DataFrame(data_dict)
    df['t'] = pd.to_datetime(df['t'].ravel())
    df.set_index('t', inplace=True)

    # 2. 处理重复索引 (针对你遇到的 duplicate labels 问题)
    # 如果同一秒有多个数据，先取平均，确保索引唯一
    if df.index.has_duplicates:
        df = df.groupby(level=0).mean()

    # 3. 进行 1 分钟重采样 (Resample)
    # '1T' 代表 1 分钟，mean() 会计算每分钟内的平均值
    # label='left' 表示使用分钟的开头作为时间戳 (例如 10:00:00 代表 10:00-10:01 之间的数据)
    resampled_df = df.resample('1T', label='left').mean()

    # 4. 如果需要填补分钟缺失 (可选)
    # 如果某分钟完全没数据，mean() 会产生 NaN。如果需要插值可以使用:
    # resampled_df = resampled_df.interpolate(method='linear')

    # 5. 还原为字典格式返回
    result = {col: resampled_df[col].values for col in resampled_df.columns}
    result['t'] = resampled_df.index.to_pydatetime() # 转回 datetime 对象数组
    
    return result