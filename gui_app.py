import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton, 
                             QListWidget, QCheckBox, QGroupBox, QFileDialog, QMessageBox,
                             QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ==========================================
# 核心依赖环境与安全导入
# ==========================================
try:
    import compcalculate
    from pro_of_read import cdfdataread        
    from pro_of_tools import build_datasets_multires 
    from Events_plot import ICME_plot 
except ImportError as e:
    print(f"⚠️ [警告] 导入自定义算法模块失败: {e}")

# ==========================================
# 后台数据处理与合并线程 (高阶1D/2D自适应多选版)
# ==========================================
class PlotWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self):
        try:
            satellite = self.config['satellite']
            selected_events = self.config['selected_events'] 
            out_path = self.config['out_path']
            flags = self.config['plot_flags'] 
           
            total_count = len(selected_events)
            self.progress.emit(f"🔥 [参数精细化批处理启动] 目标卫星: {satellite}，共 {total_count} 个区间...")

            success_count = 0
            
            # ==================== 核心循环：批处理各个事件 ====================
            for i, event_row in enumerate(selected_events, 1):
                try:
                    t_start_str = event_row["Start_Datetime"]
                    self.progress.emit(f"--------------------------------------------------")
                    self.progress.emit(f"🔄 [{i}/{total_count}] 正在深度挖掘分析窗口: {t_start_str} ...")

                    # 时间解析与外扩 6 小时
                    t_shock_str = event_row.get("shock_Datetime", t_start_str)
                    t_shock = datetime.strptime(t_shock_str, "%Y-%m-%d %H:%M:%S")
                    t_start = datetime.strptime(t_start_str, "%Y-%m-%d %H:%M:%S")
                    t_end   = datetime.strptime(event_row["End_Datetime"], "%Y-%m-%d %H:%M:%S")
                    
                    actual_shock = None if t_shock == t_start else t_shock
                    event_info = {"shock": actual_shock, "start": t_start, "end": t_end}
                    
                    search_start = t_shock - pd.Timedelta(hours=6)
                    search_end   = t_end + pd.Timedelta(hours=6)

                    eledata, magdata, prodata, smsdata = {}, {}, {}, {}

                    #数据读取预加载
                    if flags['mag_bt'] or flags['mag_comp'] or flags['mag_ang'] or flags['pro_beta']:
                        if satellite == "WIND" :
                            raw_mag_var = "BGSE"
                            mag_cdf_type = "wi_h0_mfi"  
                        elif satellite == "PSP" :
                            raw_mag_var = "psp_fld_l2_mag_RTN_1min"
                            mag_cdf_type = "psp_fld_l2_mag_rtn_1min"
                        elif satellite == "ACE" :
                            raw_mag_var = "BGSM"
                            mag_cdf_type = "ac_h0_mfi"
                        elif satellite == "SOLO" :
                            raw_mag_var = "B_RTN"
                            mag_cdf_type = "solo_l2_mag-rtn-normal" 
                            

                        magdata = cdfdataread(self.config['mag_path'], [raw_mag_var], search_start, search_end, mag_cdf_type)
                        df_bt = magdata.get(raw_mag_var)
                        if df_bt is not None:
                            b_cols = [c for c in df_bt.columns if c != 't']
                            if len(b_cols) == 3:
                                b_total = np.sqrt(df_bt[b_cols[0]]**2 + df_bt[b_cols[1]]**2 + df_bt[b_cols[2]]**2)
                                magdata["Btotal_calculated"] = pd.DataFrame({'t': df_bt['t'].values, 'Bt': b_total.values})
                                theta, phi = compcalculate.rtnthetaphi(df_bt[b_cols].values)
                                magdata["theta"] = pd.DataFrame({'t': df_bt['t'].values, 'theta': theta})
                                magdata["phi"]   = pd.DataFrame({'t': df_bt['t'].values, 'phi': phi})
                    if flags['ele_ne']:
                        if satellite == "WIND":
                            raw_ele_var = "elect_density"  
                            ele_cdf_type = "wi_k0_3dp"
                        if satellite == "PSP":
                            raw_ele_var = "Electron_Counts_ChanE"
                            ele_cdf_type = "psp_isois-epilo_l2-pe"
                        if satellite == "ACE":
                            raw_ele_var = "Electron_lo"
                            ele_cdf_type = "ac_k0_epm"
                        if satellite == "SOLO":
                            raw_ele_var = "Electron_Flux_S"
                            ele_cdf_type = "solo_l3_epd-ept-1min"
                        eledata = cdfdataread(self.config['ele_path'], [raw_ele_var], search_start, search_end, ele_cdf_type)
                    if flags['pro_np'] or flags['pro_vp'] or flags['pro_tp'] or flags['pro_beta']:
                        if satellite == "WIND":
                            raw_pro_vars = ["P_DENS","P_VELS","P_TEMP"]  
                            pro_cdf_type = "wi_pm_3dp"
                        if satellite == "PSP": 
                            raw_pro_vars = ["np_moment","vp_moment_RTN","wp_moment"]
                            pro_cdf_type = "psp_swp_spc_l3i"
                        if satellite == "ACE":
                            raw_pro_vars = ["Np", "Vp", "Tpr"]
                            pro_cdf_type = "ac_h0_swe"
                        if satellite == "SOLO":
                            raw_pro_vars = ["N", "V_RTN", "T"]
                            pro_cdf_type = "solo_l2_swa-pas-grnd-mom"
                        prodata = cdfdataread(self.config['pro_path'], raw_pro_vars, search_start, search_end, pro_cdf_type)
                        
                        if satellite == "PSP" and "wp_moment" in prodata and prodata["wp_moment"] is not None:
                            df_vth = prodata["wp_moment"]
                            vth_col = [c for c in df_vth.columns if c != 't'][0]
                            T_proton = 60.57 * (df_vth[vth_col].values ** 2)
                            prodata["P_TEMP"] = pd.DataFrame({'t': df_vth['t'].values, 'P_TEMP': T_proton})
                            del prodata["wp_moment"]

                        if satellite == "WIND" : tmeppv = "P_VELS"
                        if satellite == "PSP" : tmeppv = "vp_moment_RTN"
                        if satellite == "ACE" : tmeppv = "Vp"
                        if satellite == "SOLO" : tmeppv = "V_RTN"

                        df_vel = prodata.get(tmeppv)
                        if df_vel is not None:
                            vel_cols = [c for c in df_vel.columns if c != 't']
                            if len(vel_cols) == 3:
                                v_total = np.sqrt(df_vel[vel_cols[0]]**2 + df_vel[vel_cols[1]]**2 + df_vel[vel_cols[2]]**2)
                                prodata["Vp_calculated"] = pd.DataFrame({'t': df_vel['t'].values, 'Vp': v_total.values})
                            del prodata[tmeppv]

                        if {flags['pro_beta'] and ("Btotal_calculated" in magdata)} or flags['pro_tp']:
                            if satellite == "WIND" : 
                                tmepnp = "P_DENS"
                                tmeptp = "P_TEMP"
                            if satellite == "PSP" : 
                                tmepnp = "np_moment"
                                tmeptp = 'P_TEMP'
                            if satellite == "ACE" : 
                                tmepnp = "Np"
                                tmeptp = "Tpr"
                            if satellite == "AOLO" :
                                tmepnp = "V_RTN"
                                tmeptp = "T"

                            df_n = prodata.get(tmepnp)
                            df_t = prodata.get(tmeptp)
                            if (df_n is not None) and (df_t is not None) and ("Btotal_calculated" in magdata):
                                res_beta = compcalculate.Pandbeta(b=magdata["Btotal_calculated"]['Bt'].values, n=df_n.iloc[:, 1].values, T=df_t.iloc[:, 1].values, tn=df_n['t'].values, tb=magdata["Btotal_calculated"]['t'].values)
                                calc_time = res_beta.get('t', res_beta.get('t_np', df_n['t'].values))
                                prodata["Pdy"]  = pd.DataFrame({'t': calc_time, 'Pdy': res_beta['Pth']})
                                prodata["beta"] = pd.DataFrame({'t': calc_time, 'beta': res_beta['beta']})
                    particle_modes = self.config.get('particle_modes', {})
                    if any(mode != 0 for mode in particle_modes.values()): 
                        # 根据卫星设置变量和前缀
                        if satellite == "WIND":
                            sms_vars = ["n_tc_he2plus", "n_tc_o6plus", "n_tc_fe10plus", "n_tc_cplus"]  # 根据实际CDF调整
                            sms_prefix = "wi_l2_3min_sms" 
                        if satellite == "ACE":
                            sms_vars = ["flux_He", "flux_O", "flux_Fe", "flux_C"]  # 根据实际CDF调整
                            sms_prefix = "ac_h1_sis"
                        if satellite == "PSP":
                            sms_vars = ["A_He_Flux", "A_O_Flux", "A_Fe_Flux", "A_C_Flux"]  # 根据实际CDF调整
                            sms_prefix = "psp_isois-epihi_l2-let1-rates60"
                        if satellite == "SOLO":
                            sms_vars = ["n_tc_he2plus", "n_tc_o6plus", "n_tc_fe10plus", "n_tc_cplus",
                                        "A_He_Flux", "A_O_Flux", "A_Fe_Flux", "A_C_Flux"]  # 根据实际CDF调整
                            sms_prefix = "wi_l2_3min_sms"
                        smsdata = cdfdataread(self.config['sms_path'], sms_vars, search_start, search_end, sms_prefix)
                    
                    final_mag = {}
                    if flags['mag_comp'] and ("BGSE" in magdata or "psp_fld_l2_mag_RTN_1min" in magdata or "BGSM" in magdata or "B_RTN" in magdata):
                        if satellite == "WIND":k = "BGSE" 
                        if satellite == "PSP":k = "psp_fld_l2_mag_RTN_1min"
                        if satellite == "ACE":k = "BGSM" 
                        if satellite == "SOLO":k = "B_RTN"
                        final_mag[k] = magdata[k]
                    if flags['mag_bt'] and "Btotal_calculated" in magdata: final_mag["Btotal_calculated"] = magdata["Btotal_calculated"]
                    if flags['mag_ang']:
                        if "theta" in magdata: final_mag["theta"] = magdata["theta"]
                        if "phi" in magdata: final_mag["phi"] = magdata["phi"]
                    
                    final_pro = {}
                    if flags['pro_np'] and ("P_DENS" in prodata or "np_moment" in prodata):
                        if satellite == "WIND" : k = "P_DENS"
                        if satellite == "PSP" : k = "np_moment"
                        if satellite == "ACE" : k = "Np"
                        if satellite == "SOLO" : k = "V_RTN"
                        final_pro[k] = prodata[k]
                    if flags['pro_vp'] and "Vp_calculated" in prodata: final_pro["Vp_calculated"] = prodata["Vp_calculated"]
                    if flags['pro_tp'] and tmeptp in prodata: final_pro[tmeptp]  =prodata[tmeptp]
                    if flags['pro_beta']:
                        if "Pdy" in prodata: final_pro["Pdy"] = prodata["Pdy"]
                        if "beta" in prodata: final_pro["beta"] = prodata["beta"]

                    magdatasets = build_datasets_multires(final_mag) if final_mag else {}
                    eledatasets = build_datasets_multires(eledata) if eledata else {}
                    prodatasets = build_datasets_multires(final_pro) if final_pro else {}
                    
                    datasets = {}
                    for ds in [magdatasets, eledatasets, prodatasets]:
                        if not ds: continue
                        for panel_name, curves in ds.items():
                            datasets.setdefault(panel_name, []).extend(curves)
                    # 放在 run 方法开头（卫星判断之后）
                    SATELLITE_VARS = {
                        'WIND': {
                            'He': {'dens': 'n_tc_he2plus'},
                            'O':  {'dens': 'n_tc_o6plus'},
                            'Fe': {'dens': 'n_tc_fe10plus'},
                            'C':  {'dens': 'n_tc_cplus'},
                        },
                        'PSP': {
                            'He': {'dens': 'psp_he_density', 'flux': 'A_He_Flux'},
                            'O':  {'dens': 'psp_o_density',  'flux': 'A_O_Flux'},
                            'Fe': {'dens': 'psp_fe_density', 'flux': 'A_Fe_Flux'},
                            'C':  {'dens': 'psp_c_density',  'flux': 'A_C_Flux'},
                        },
                        'ACE': {
                            'He': {'dens': 'ace_he_dens',    'flux': 'flux_He'},
                            'O':  {'dens': 'ace_o_dens',     'flux': 'flux_O'},
                            'Fe': {'dens': 'ace_fe_dens',    'flux': 'flux_Fe'},
                            'C':  {'dens': 'ace_c_dens',     'flux': 'flux_C'},
                        },
                        'SOLO': {
                            'He': {'dens': 'solo_he_n',      'flux': 'solo_he_flux'},
                            'O':  {'dens': 'solo_o_n',       'flux': 'solo_o_flux'},
                            'Fe': {'dens': 'solo_fe_n',      'flux': 'solo_fe_flux'},
                            'C':  {'dens': 'solo_c_n',       'flux': 'solo_c_flux'},
                        },
                    }
                    particle_vars = SATELLITE_VARS.get(satellite, {})
                    for particle, mode_idx in particle_modes.items():
                        if mode_idx == 0:   # 关闭
                            continue
                        vars_dict = particle_vars.get(particle)
                        if not vars_dict:
                            continue
                        if mode_idx in (1, 4, 5):
                            dens_var = vars_dict['dens']
                            if dens_var in smsdata:
                                # 利用 build_datasets_multires 处理密度（它会自动生成 Nhe2 等面板）
                                dens_ds = build_datasets_multires({dens_var: smsdata[dens_var]})
                                for panel, curves in dens_ds.items():
                                    datasets.setdefault(panel, []).extend(curves)
                        
                        # 通量处理（mode_idx in [2,3,4,5]）
                        if mode_idx in (2, 3, 4, 5):
                            flux_var = vars_dict['flux']
                            if flux_var in smsdata:
                                df = smsdata[flux_var]
                                t = df['t'].values
                                data_cols = [c for c in df.columns if c != 't']
                                if not data_cols:
                                    continue
                                col = data_cols[0]
                                vals = df[col].values
                                
                                # 判断是否为多通道能谱（检查第一个元素是否为数组）
                                is_multichannel = isinstance(vals[0], (list, np.ndarray)) if len(vals) > 0 else False
                                if mode_idx in (2, 4):   # 通量(1D) 或 密度+通量(1D)
                                    if is_multichannel:
                                        # 对每行求和得到总通量
                                        y_1d = np.array([np.nansum(v) for v in vals])
                                    else:
                                        y_1d = vals
                                    # 构建1D线图
                                    line_dict = {
                                        "x": t,
                                        "y": y_1d,
                                        "label": f"{particle} Flux",
                                        "style": {"linewidth": 2, "color": "darkviolet"},
                                        "unit": f"{particle} Flux\n"
                                    }
                                    # 面板名称为 "Fluxhe", "Fluxo" 等（与 stname 中一致）
                                    panel_name = f"Flux{particle.lower()}"
                                    datasets.setdefault(panel_name, []).append(line_dict)
                                
                                if mode_idx in (3, 5):   # 通量(2D) 或 密度+通量(2D)
                                    if is_multichannel:
                                        # 构建能谱矩阵
                                        flux_matrix = np.array([np.asarray(v) for v in vals])
                                        # 如果各通道长度不一致，可以填充nan，但通常一致
                                        # 获取能量轴（这里简单用序号，实际可以从元数据获取）
                                        energy_axis = np.arange(flux_matrix.shape[1], dtype=float)
                                        spec_dict = {
                                            "t": t,
                                            "y": energy_axis,
                                            "z": flux_matrix,
                                            "x": t,
                                            "label": f"{particle} Spec",
                                            "unit": "Energy\n(eV)"
                                        }
                                        panel_name = f"Flux{particle.lower()}"
                                        datasets.setdefault(panel_name, []).append(spec_dict)
                                    else:
                                        # 如果不是多通道，无法画能谱，可降级为1D或忽略
                                        # 这里简单忽略，或者您可打印警告
                                        pass

                    if not datasets:
                        self.progress.emit(f"⚠️ [跳过] 事件 {t_start_str} 未选中或无任何有效保留物理图层。")
                        continue

                    # 提交绘图管线渲染
                    ICME_plot(datasets=datasets, event=event_info, time_window=(search_start, search_end), savepath=out_path)
                    self.progress.emit(f"🎨 [完成] 成功渲染保存高维画幅图！")
                    success_count += 1

                except Exception as event_error:
                    self.progress.emit(f"❌ [局部异常跳过] 窗口 {t_start_str} 发生故障: {str(event_error)}")
                    continue

            self.finished.emit(True, f"🎉 批量任务结束！完美渲染生成 {success_count} / {total_count} 张复合高精图。")
        except Exception as e:
            self.finished.emit(False, f"全局引擎严重阻断故障:\n{str(e)}")

# ==========================================
# PyQt5 主界面布局与高级联动过滤舱
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Space Physics ICME Data Visualizer (Spectrum Adapt v5.0)")
        self.resize(1420, 900)
        self.plot_boxes = {}
        self.events_df = pd.DataFrame()
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_widget = QWidget()
        left_panel = QVBoxLayout(left_widget)
        left_widget.setFixedWidth(640) 

        # --- 1. 卫星选择 ---
        sat_group = QGroupBox("1. 目标卫星选择")
        sat_layout = QVBoxLayout()
        self.sat_combo = QComboBox()
        self.sat_combo.addItems(["WIND", "PSP", "ACE", "SOLO"])
        self.sat_combo.currentTextChanged.connect(self.update_default_paths) 
        sat_layout.addWidget(QLabel("选择当前分析卫星 (自动同步默认物理根路径及CDF字段标准):"))
        sat_layout.addWidget(self.sat_combo)
        sat_group.setLayout(sat_layout)
        left_panel.addWidget(sat_group)

        # --- 2. 物理量精细筛选矩阵 (Grid) ---
        filter_group = QGroupBox("2. 科学画幅参数项精细过滤决策舱")
        grid_layout = QGridLayout()

        grid_layout.addWidget(QLabel("<b>[磁场参数]:</b>"), 0, 0)
        self.cb_mag_bt = QCheckBox("总磁场 (Bt)")
        self.cb_mag_comp = QCheckBox("三分量 (Bxyz/RTN)")
        self.cb_mag_ang = QCheckBox("倾角 (Theta/Phi)")
        grid_layout.addWidget(self.cb_mag_bt, 0, 1)
        grid_layout.addWidget(self.cb_mag_comp, 0, 2)
        grid_layout.addWidget(self.cb_mag_ang, 0, 3)

        grid_layout.addWidget(QLabel("<b>[粒子参数]:</b>"), 1, 0)
        self.cb_ele_ne = QCheckBox("电子密度 (Ne)")
        self.cb_ele_fe = QCheckBox("电子通量 (Fluxe)")
        self.cb_pro_np = QCheckBox("质子密度 (Np)")
        self.cb_pro_fp = QCheckBox("质子通量 (Fluxp)")
        self.cb_pro_vp = QCheckBox("质子速度 (Vp)")
        self.cb_pro_tp = QCheckBox("质子温度 (Tp)")
        self.cb_pro_beta = QCheckBox("等离子体 Beta & 热压")
        grid_layout.addWidget(self.cb_ele_ne, 1, 1)
        grid_layout.addWidget(self.cb_ele_fe, 1, 2)
        grid_layout.addWidget(self.cb_pro_np, 1, 3)
        grid_layout.addWidget(self.cb_pro_fp, 2, 1)
        grid_layout.addWidget(self.cb_pro_vp, 2, 2)
        grid_layout.addWidget(self.cb_pro_tp, 2, 3)
        grid_layout.addWidget(self.cb_pro_beta, 3, 1)

        self.particle_modes = {}  
        particles = [
            ("He", "氦离子 (He)"),
            ("O",  "氧离子 (O)"),
            ("Fe", "铁离子 (Fe)"),
            ("C",  "碳离子 (C)")
        ]
        # 每个粒子占一行，放在第3、4、5、6行（原来第4行有C，但为了整齐我们重新排）
        row_start = 4
        for idx, (key, label) in enumerate(particles):
            row = row_start + idx
            grid_layout.addWidget(QLabel(label), row, 1)   # 第1列放标签
            combo = QComboBox()
            combo.addItems(["关闭", "密度", "通量(1D)", "通量(2D)", "密度+通量(1D)", "密度+通量(2D)"])
            combo.setCurrentIndex(0)   # 默认关闭，您也可以改为1等
            setattr(self, f"combo_sms_{key.lower()}", combo)   # 动态生成属性，如 self.combo_sms_he
            self.particle_modes[key] = combo
            grid_layout.addWidget(combo, row, 2)   # 第2列放组合框

        self.all_checkboxes = [
            self.cb_mag_bt, self.cb_mag_comp, self.cb_mag_ang,self.cb_ele_fe,self.cb_pro_fp,
            self.cb_ele_ne, self.cb_pro_np, self.cb_pro_vp, self.cb_pro_tp, self.cb_pro_beta
        ]
        for cb in self.all_checkboxes:
            cb.setChecked(False)
            cb.stateChanged.connect(self.sync_path_inputs_state)

        filter_group.setLayout(grid_layout)
        left_panel.addWidget(filter_group)

        # --- 3. 原始路径装配 ---
        path_group = QGroupBox("3. 空间科学原始数据源路径装配")
        path_layout = QVBoxLayout()

        def create_path_row(label_text, is_file=False):
            row = QHBoxLayout()
            line_edit = QLineEdit()
            btn = QPushButton("浏览...")
            btn.clicked.connect(lambda: self.select_file(line_edit, is_file))
            row.addWidget(line_edit)
            row.addWidget(btn)
            path_layout.addWidget(QLabel(label_text))
            path_layout.addLayout(row)
            return line_edit
        
        self.event_list_input = create_path_row("事件清单索引文件 (list.txt):", is_file=True)
        self.mag_path_input = create_path_row("磁场观测根目录 (Magnetic Data Dir):")
        self.ele_path_input = create_path_row("电子密度根目录 (Electron Data Dir):")
        self.pro_path_input = create_path_row("质子矩/动力学根目录 (Proton Data Dir):")
        self.sms_path_input = create_path_row("少数离子成分根目录 (Composition Dir):")
        self.out_path_input = create_path_row("图像保存路径 (Output Location):")

        path_group.setLayout(path_layout)
        left_panel.addWidget(path_group)

        # --- 4. 批处理时间舱 ---
        event_group = QGroupBox("4. ICME 事件多选批处理决策舱 (支持 Ctrl/Shift/鼠标拖拽多选)")
        event_layout = QVBoxLayout()
        
        self.event_list_widget = QListWidget()
        self.event_list_widget.setSelectionMode(QListWidget.ExtendedSelection) 
        
        load_event_btn = QPushButton("📂 载入并智能解析事件清单")
        load_event_btn.clicked.connect(self.load_events)
        event_layout.addWidget(load_event_btn)
        event_layout.addWidget(self.event_list_widget)
        
        self.run_btn = QPushButton("🚀 启动自适应参数矩阵批处理绘图")
        self.run_btn.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #008CBA; color: white; padding: 11px; border-radius: 5px;")
        self.run_btn.clicked.connect(self.start_processing)
        event_layout.addWidget(self.run_btn)

        event_group.setLayout(event_layout)
        left_panel.addWidget(event_group)
        main_layout.addWidget(left_widget)

        # ==================== 右侧：实时监测舱 ====================
        right_group = QGroupBox("多事件批量自适应图层渲染实时流监控舱")
        right_layout = QVBoxLayout()
        self.log_viewer = QListWidget()
        self.log_viewer.setStyleSheet("background-color: #1a1a1a; color: #00ff66; font-family: Consolas; font-size: 13px;")
        right_layout.addWidget(self.log_viewer)
        right_group.setLayout(right_layout)
        main_layout.addWidget(right_group)
        
        self.update_default_paths(self.sat_combo.currentText())

    # ==================== UI智能灰显与联动机制 ====================
    def sync_path_inputs_state(self):
        mag_any = self.cb_mag_bt.isChecked() or self.cb_mag_comp.isChecked() or self.cb_mag_ang.isChecked() or self.cb_pro_beta.isChecked()
        self.mag_path_input.setEnabled(mag_any)
        self.ele_path_input.setEnabled(self.cb_ele_ne.isChecked())
        pro_any = self.cb_pro_np.isChecked() or self.cb_pro_vp.isChecked() or self.cb_pro_tp.isChecked() or self.cb_pro_beta.isChecked()
        self.pro_path_input.setEnabled(pro_any)
        # 重离子：如果有任何一个组合框不是“关闭”，则启用 sms_path_input
        sms_any = any(self.particle_modes[p].currentIndex() != 0 for p in self.particle_modes)
        self.sms_path_input.setEnabled(sms_any)
        # 全局粒子图像形式已被删除，无需再处理

    def update_default_paths(self, sat):
        if sat == "WIND":
            self.event_list_input.setText(r"I:\list\icme\chi\windicmes_ustc_split.txt")
            self.mag_path_input.setText(r"I:\stdata\wind\mfi\wi_h0_mfi")
            self.ele_path_input.setText(r"I:\stdata\wind\3dp\3dp_k0")
            self.pro_path_input.setText(r"I:\stdata\wind\3dp\3dp_pm")
            self.sms_path_input.setText(r"I:\stdata\wind\SMS\WI_L2_3MIN_SMS-STICS-NVT-SOLARWIND")
            self.out_path_input.setText(r"E:\ZhaoWenjin_Reserch\ICME\结果图像\WIND")
        elif sat == "PSP":
            self.event_list_input.setText(r"E:\ZhaoWenjin_Reserch\ICME\结果图像\PSP\list.txt")
            self.mag_path_input.setText(r"I:\stdata\psp\FLD\PSP_FLD_L2_MAG_RTN_1MIN")
            self.ele_path_input.setText(r"I:\stdata\psp\isois-epilo\pe")
            self.pro_path_input.setText(r"I:\stdata\psp\sweap\psp_swp_spc_l3i")
            self.sms_path_input.setText(r"I:\stdata\psp\isois-epihi\let1")
            self.out_path_input.setText(r"E:\ZhaoWenjin_Reserch\ICME\结果图像\PSP")
        elif sat == "ACE":
            self.event_list_input.setText(r"I:\list\icme\chi\windicmes_ustc_split.txt")
            self.mag_path_input.setText(r"I:\stdata\ace\AC_H0_MFI")
            self.ele_path_input.setText(r"I:\stdata\ace\EPM\AC_EPM_K0")
            self.pro_path_input.setText(r"I:\stdata\ace\AC_H0_SWE")
            self.sms_path_input.setText(r"I:\stdata\ace\AC_H1_SIS")
            self.out_path_input.setText(r"E:\ZhaoWenjin_Reserch\ICME\结果图像\ACE")
        elif sat == "SOLO":
            self.event_list_input.setText(r"E:\ZhaoWenjin_Reserch\ICME\结果图像\SOLO\list.txt")
            self.mag_path_input.setText(r"I:\stdata\solo\mag\RTN_NORMAL")
            self.ele_path_input.setText(r"I:\stdata\solo\epd\SOLO_L3_EPD-EPT-1MIN")
            self.pro_path_input.setText(r"I:\stdata\solo\swa\solo_l2_swa-pas-grnd-mom")
            self.sms_path_input.setText(r"I:\stdata\psp\isois-epihi\let1")
            self.out_path_input.setText(r"E:\ZhaoWenjin_Reserch\ICME\结果图像\SOLO")
        self.event_list_widget.clear()
        self.sync_path_inputs_state()
        self.append_log(f"🔄 已自动调配并挂载 {sat} 卫星原生默认出图路径。")

    def select_file(self, line_edit, is_file=False):
        if is_file:
            path, _ = QFileDialog.getOpenFileName(self, "选择事件清单", "", "Text Files (*.txt);;All Files (*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择数据根目录")
        if path: line_edit.setText(os.path.normpath(path))

    def append_log(self, text):
        self.log_viewer.addItem(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
        self.log_viewer.scrollToBottom()

    def load_events(self):
        list_path = self.event_list_input.text()
        if not os.path.exists(list_path):
            QMessageBox.warning(self, "文件丢失", f"无法定位该清单:\n{list_path}")
            return
        try:
            self.events_df = pd.read_csv(list_path, sep=",")
            self.event_list_widget.clear()
            for idx, row in self.events_df.iterrows():
                time_mark = row.get("Start_Datetime", f"Index_{idx}")
                self.event_list_widget.addItem(f"[{idx}] 📌 激发起始时刻: {time_mark}")
            self.append_log(f"✅ 成功读取清单中 {len(self.events_df)} 个 ICME 事件。可自由 Ctrl/Shift 多选。")
        except Exception as e:
            QMessageBox.critical(self, "解析阻断", f"CSV读入冲突: {str(e)}")

    def start_processing(self):
        selected_items = self.event_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "未选定事件", "请按住键盘 Ctrl 键或用鼠标滑动框选至少一个事件区间！")
            return

        plot_flags = {
            'mag_bt': self.cb_mag_bt.isChecked(), 'mag_comp': self.cb_mag_comp.isChecked(), 'mag_ang': self.cb_mag_ang.isChecked(),
            'ele_ne': self.cb_ele_ne.isChecked(),'ele_fe':self.cb_ele_fe.isChecked(), 'pro_np': self.cb_pro_np.isChecked(),
            'pro_fe':self.cb_pro_fp.isChecked(),'pro_vp': self.cb_pro_vp.isChecked(),'pro_tp': self.cb_pro_tp.isChecked(), 
            'pro_beta': self.cb_pro_beta.isChecked(),
        }
        
        if not any(plot_flags.values()):
            QMessageBox.warning(self, "未勾选数据", "请至少勾选一个物理量子分量，否则画布无法渲染！")
            return

        selected_events_list = []
        for item in selected_items:
            event_idx = int(item.text().split("]")[0][1:])
            selected_events_list.append(self.events_df.iloc[event_idx].to_dict())
        particle_modes = {key: combo.currentIndex() for key, combo in self.particle_modes.items()}
        config = {
            'satellite': self.sat_combo.currentText(),
            'ele_path': self.ele_path_input.text(), 'mag_path': self.mag_path_input.text(),
            'pro_path': self.pro_path_input.text(), 'sms_path': self.sms_path_input.text(),
            'out_path': self.out_path_input.text(), 'plot_flags': plot_flags,
            'selected_events': selected_events_list,
            'particle_modes': particle_modes
        }
        
        self.run_btn.setEnabled(False)
        self.append_log(f"⏳ 正在拉起自适应科学出图流处理引擎...")
        
        self.worker = PlotWorker(config)
        self.worker.progress.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self, success, message):
        self.run_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "任务圆满完成", message)
        else:
            QMessageBox.critical(self, "运行遭遇阻断", message)

if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    window = MainWindow() 
    window.show()
    sys.exit(app.exec_())