import numpy as np

def rtnthetaphi(b):
    # 确保输入是 float 类型的 numpy 数组
    b = np.asarray(b,dtype=float)
    
    # 计算总磁场，遇到 nan 会返回 nan 而不会崩溃
    btotal = np.sqrt(np.nansum(b**2, axis=1))
    
    # 防止分母为 0
    btotal_safe = np.where(btotal == 0, np.nan, btotal)
    
    # 计算 theta (-90 到 90)
    # 使用 np.clip 防止因浮点数微小误差导致值超出 [-1, 1] 范围
    theta = np.arcsin(np.clip(b[:,2]/btotal_safe, -1, 1)) / np.pi * 180
    
    # 计算 phi (0 到 360)
    phi = np.arctan2(b[:,1], b[:,0]) / np.pi * 180 % 360
    
    return theta, phi