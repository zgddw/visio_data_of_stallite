import os
import pandas as pd
import numpy as np
import compcalculate
import matplotlib.pyplot as plt
from datetime import datetime
from pro_of_read import cdfdataread        
from pro_of_tools import build_datasets_multires 
from Events_plot import ICME_plot 


# ==================== 1. 基础路径和数据源配置 ====================

#列表读取
event_list_path = os.path.join(r"E:\ZhaoWenjin_Reserch\ICME\结果图像\PSP", "list.txt") 
if not os.path.exists(event_list_path):
    raise FileNotFoundError(f"找不到清洗后的事件清单文件: {event_list_path}，请先运行清洗脚本！")
events_df = pd.read_csv(event_list_path, sep=",")
out_path = r"E:\ZhaoWenjin_Reserch\ICME\结果图像\PSP"
#磁场数据
magroot_path = r"I:\stdata\psp\FLD\PSP_FLD_L2_MAG_RTN_1MIN"
magvar_names = ["psp_fld_l2_mag_RTN_1min"]
magfile_prefix = "psp_fld_l2_mag_rtn_1min"
#电子数据
eleroot_path = r"I:\stdata\psp\FLD\PSP_FLD_L3_SQTN_RFS_V1V2"
elevar_names = ["electron_density"]
elefile_prefix = "psp_fld_l3_sqtn_rfs_v1v2"
#质子数据
proroot_path = r"I:\stdata\psp\sweap\psp_swp_spc_l3i"
provar_names = ["np_moment","vp_moment_RTN","wp_moment"]
profile_prefix = "psp_swp_spc_l3i"
##超热离子
smsroot_path = r"I:\stdata\psp\isois-epihi\let1"
smsvar_names = ["A_He_Flux","A_O_Flux","A_Fe_Flux","A_C_Flux"]
smsfile_prefix = "psp_isois-epihi_l2-let1-rates60"
# ==================== 2. 开启事件驱动的大循环 ====================
for idx, row in events_df.iterrows():
    t_shock = datetime.strptime(row["shock_Datetime"], "%Y-%m-%d %H:%M:%S")
    t_start = datetime.strptime(row["Start_Datetime"], "%Y-%m-%d %H:%M:%S")
    t_end   = datetime.strptime(row["End_Datetime"], "%Y-%m-%d %H:%M:%S")
    if t_shock == t_start:
        actual_shock = None
    else:
        actual_shock = t_shock
    
    event_info = {
        "shock": actual_shock, # 如果为 None，Events_plot 内部就会自动不画绿线
        "start": t_start,
        "end":   t_end
    }
    search_start = t_shock - pd.Timedelta(hours=6)
    search_end   = t_end + pd.Timedelta(hours=6)
    ##读取数据
    eledata = cdfdataread(eleroot_path, elevar_names, search_start, search_end, elefile_prefix)
    magdata = cdfdataread(magroot_path, magvar_names, search_start, search_end, magfile_prefix)
    prodata = cdfdataread(proroot_path, provar_names, search_start, search_end, profile_prefix)
    smsdata = cdfdataread(smsroot_path, smsvar_names, search_start, search_end, smsfile_prefix)

    #计算总磁场
    df_bt = magdata["psp_fld_l2_mag_RTN_1min"]
    b_cols = [c for c in df_bt.columns if c != 't']
    if len(b_cols) == 3:
        b_total = np.sqrt(df_bt[b_cols[0]]**2 + df_bt[b_cols[1]]**2 + df_bt[b_cols[2]]**2)
        df_btotal = pd.DataFrame({
            't': df_bt['t'].values,
            'Bt': b_total.values
        })
        magdata["Btotal_calculated"] = df_btotal
    #计算磁场角度
    df_b = magdata["psp_fld_l2_mag_RTN_1min"]
    b_cols = [c for c in df_b.columns if c != 't']
    if len(b_cols) == 3:
        print("🧭 [主函数] 正在利用 rtnthetaphi 计算磁场倾角 theta 和方位角 phi...")
        b_array = df_b[b_cols].values
        theta, phi = compcalculate.rtnthetaphi(b_array)
        
        # 塞入 magdata 字典中，命名为标准代号 theta 和 phi
        magdata["theta"] = pd.DataFrame({'t': df_b['t'].values, 'theta': theta})
        magdata["phi"]   = pd.DataFrame({'t': df_b['t'].values, 'phi': phi})

    #计算温度
    # =====================================================================
    # 🚀 🌟 【新增】质子最概然热速度 (vth) 转热力学温度 (P_TEMP) 实时计算块 🌟
    # =====================================================================
    if "wp_moment" in prodata and prodata["wp_moment"] is not None:
        df_vth = prodata["wp_moment"]
        # 动态获取除时间列 't' 之外的数据列名
        vth_col = [c for c in df_vth.columns if c != 't'][0]
        # 提取速度值（假设原始单位是 km/s）
        vth_values = df_vth[vth_col].values
        # 物理公式转换: T = 60.57 * vth^2
        T_proton = 60.57 * (vth_values ** 2)
        # 🧠 核心无感对接：将计算出的温度包装成 "P_TEMP" 塞回字典
        prodata["P_TEMP"] = pd.DataFrame({
            't': df_vth['t'].values,
            'P_TEMP': T_proton
        })
        del prodata["wp_moment"]
        print("🔥 [INFO] 已成功将 PSP 主质子群径向热速度转换为标准质子温度 (P_TEMP)。")
    # =====================================================================

    # ... 接下来是你原有的 Beta 计算代码 ...
    # 此时下面的代码调用 df_t = prodata["P_TEMP"] 时，就会无缝拿到我们刚刚算好的温度！

    # (2) 计算等离子体参数 Pdy (热压) 和 beta
    df_n = prodata.get("np_moment")
    df_t = prodata.get("P_TEMP")
    if (df_b is not None) and (df_n is not None) and (df_t is not None):
        print("🔥 [主函数] 正在利用 pandbeta 对齐时间轴并计算等离子体 Beta 与热压...")
        b_cols = [c for c in df_b.columns if c != 't']
        res_beta = compcalculate.Pandbeta(
            b=b_total, 
            n=df_n.iloc[:, 1].values, 
            T=df_t.iloc[:, 1].values, 
            tn=df_n['t'].values, 
            tb=df_b['t'].values
        )
        
        # ✨ 彻底修正 KeyError：安全动态获取时间轴
        if 't' in res_beta:
            calc_time = res_beta['t']
        elif 't_np' in res_beta:
            calc_time = res_beta['t_np']
        else:
            calc_time = df_n['t'].values  # 安全兜底：长度与等离子体数据完全一致
        
        # ✨ 彻底修正：这里必须使用上面定义好的 calc_time 变量！
        prodata["Pdy"]  = pd.DataFrame({'t': calc_time, 'Pdy': res_beta['Pth']})
        prodata["beta"] = pd.DataFrame({'t': calc_time, 'beta': res_beta['beta']})


    df_vel = prodata["vp_moment_RTN"]
    # 计算总速度
    vel_cols = [c for c in df_vel.columns if c != 't']
    if len(vel_cols) == 3:
        v_total = np.sqrt(df_vel[vel_cols[0]]**2 + df_vel[vel_cols[1]]**2 + df_vel[vel_cols[2]]**2)
        df_vp = pd.DataFrame({
            't': df_vel['t'].values,
            'Vp': v_total.values
        })
        prodata["Vp_calculated"] = df_vp
    del prodata["vp_moment_RTN"]    
    magdatasets = build_datasets_multires(magdata)
    eledatasets = build_datasets_multires(eledata)
    prodatasets = build_datasets_multires(prodata)
    smsdatasets = build_datasets_multires(smsdata)
    datasets = {}
    #合并数据
    all_datasets_to_merge = [magdatasets, eledatasets,prodatasets,smsdatasets] 
    
    for ds in all_datasets_to_merge:
        if not ds: # 如果某个数据源没读到数据，安全跳过
            continue
        for panel_name, curves in ds.items():
            # 如果 datasets 里还没有这个 panel，就新建一个空列表；然后把曲线装进去
            datasets.setdefault(panel_name, []).extend(curves)
    if not datasets:
        print("[WARN] 所有数据源均为空，跳过此事件。")
        continue
    ICME_plot(
        datasets=datasets,
        event=event_info,
        time_window=(search_start, search_end),
        savepath=out_path
    )
    break