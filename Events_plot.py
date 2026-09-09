import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from matplotlib.ticker import LogLocator, LogFormatterExponent
from matplotlib.ticker import MultipleLocator,AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

def ICME_plot(
        datasets,
        event,
        time_window=None,
        savepath=None,
        show=False,
        energy=[],
        figureformat='png'):
    # ---------- 1. 自动识别并过滤有效数据 ----------
    valid_datasets = {
        name: lines for name, lines in datasets.items()
        if lines and len(lines) > 0
    }

    n_panels = len(valid_datasets)

    if n_panels == 0:
        print("No valid data found in datasets to plot.")
        return
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    # ---------- 2. 创建画布 ----------
    fig, axes = plt.subplots(
        nrows=n_panels,
        ncols=1,
        figsize=(20, 2.5 * n_panels),
        sharex=True
    )

    if n_panels == 1:
        axes = [axes]

    # ---------- 3. 循环绘图 (使用过滤后的数据) ----------
    for ax, (panel_name, lines) in zip(axes, valid_datasets.items()):
        divider = make_axes_locatable(ax)
        cax_placeholder = divider.append_axes("right", size="3%", pad=0.1)
        
        # ✨ 核心通用探测：只要数据中存在 'z' 键，就判定为 2D 能谱图，适用于任何粒子！
        is_2d_spec = "z" in lines[0]

        if is_2d_spec:
            cax_placeholder.set_visible(True) # 唤醒颜色轴槽位
            spec_data = lines[0]
            t_data = spec_data["t"]       
            y_data = spec_data["y"] # 能量通道      
            z_data = spec_data["z"] # 通量矩阵      
            t_num = mdates.date2num(t_data)
            
            # 安全对数处理，防 0 报错
            z_data_safe = np.where(z_data <= 0,np.nan, z_data)
            z_log = np.log10(z_data_safe)
            
            # 绘制 2D Color Map
            mesh = ax.pcolormesh(t_num, y_data, z_log.T, cmap='jet', shading='auto')
            ax.set_yscale('log')
            ax.set_xlim(t_num.min(), t_num.max())
            ax.set_ylim(y_data.min(), y_data.max())
            cbar = plt.colorbar(mesh, cax=cax_placeholder, orientation='vertical')
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label(spec_data.get('unit', 'Log10 Flux'), fontsize=14)
            
        else:
            # 常规 1D 曲线绘制
            cax_placeholder.set_visible(False)
            for line in lines:
                y = np.asarray(line["y"], dtype=float)
                if any(key in panel_name for key in ["Tp", "Np", r"\beta"]):
                    y[y <= 0] = np.nan
                if np.all(np.isnan(y)):
                    continue
                ax.plot(line["x"], y, 
                        **line.get('style',{}),
                        zorder=2)
                    
                    
        # 坐标轴设置
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis='both', which='major', direction='in', length=15, width=2, labelsize=18, colors='black')
        ax.tick_params(axis='both', which='minor', direction='in', length=7.5, width=2, colors='black')   

        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        
        # 坐标轴字体设置
        unit = lines[0].get("unit", "") if lines else ""
        title_font = {'fontsize': 24, 'fontweight': 'bold', 'color': 'black'}
        ax.set_ylabel(unit, fontdict=title_font)
        ax.grid(False)
        ax.tick_params(axis='y', labelsize=22)
        ax.tick_params(axis='x', labelsize=22)

        # 磁场分量
        if "Bxyz" in panel_name:
            all_y = np.concatenate([np.asarray(l["y"], dtype=float) for l in lines])
            ymax = np.nanmax(np.abs(all_y))
            ax.set_ylim(-ymax, ymax)
            ax.set_ylabel(f"{r'$B_xB_yB_z$'}\n(nT)")
            ax.axhline(0, color="k", linewidth=1.2, dashes=[10, 10])
        # 速度分量
        if "Vxyz" in panel_name:
            all_y = np.concatenate([np.asarray(l["y"], dtype=float) for l in lines])
            ymax = np.nanmax(np.abs(all_y))
            ax.set_ylim(-ymax, ymax)
            ax.set_ylabel(f"{r'$V_xV_yV_z$'}\n(km/s)")
            ax.axhline(0, color="k", linewidth=1.2, dashes=[10, 10])
        # 角度范围设置
        if "theta" in panel_name:
            ax.set_ylim(-90, 90)
            ax.yaxis.set_major_locator(MultipleLocator(45))
            ax.axhline(0, color="k", linestyle="--", linewidth=2)
        if "phi" in panel_name:
            ax.set_ylim(0, 360)
            ax.yaxis.set_major_locator(MultipleLocator(90))
            ax.axhline(0, color="k", linestyle="--", linewidth=2)
            
        # ✨ 完美修复：将原有的永真表达式纠正为健壮的 any() 家族探测
        # if any(k in panel_name for k in ["Fluxhe", "Fluxo", "Fluxfe", "Fluxc"]) and is_2d_spec:
        #     cax_placeholder.set_visible(True) # 重新唤醒颜色轴槽位
        #     spec_data = lines[0]
        #     t_data = spec_data["x"]       
        #     y_data = spec_data["y"]       
        #     z_data = spec_data["z"]       
        #     t_num = mdates.date2num(t_data)
        #     z_data_safe = np.where(z_data <= 0,np.nan,z_data)
        #     z_log = np.log10(z_data_safe)
        #     mesh = ax.pcolormesh(t_num, y_data, z_log.T, cmap='jet', shading='auto')
        #     ax.set_yscale('log')
        #     cbar = plt.colorbar(mesh, cax=cax_placeholder, orientation='vertical')
        #     cbar.ax.tick_params(labelsize=14)
        #     cbar.set_label('Log10 Flux', fontsize=14)
            
        # 对数坐标设置
        if any(key in panel_name for key in ["Tp", "Np","beta","Pdy","Ne"]):
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
            ax.yaxis.set_minor_formatter(plt.NullFormatter())
            
        if "GIC" in panel_name:
            all_y = np.concatenate([np.asarray(l["y"], dtype=float) for l in lines])
            ymax = 25 * 1.1  
            ax.set_ylim(-ymax, ymax)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.axhline(y=3, color='blue', linewidth=2, dashes=[10, 10])
            ax.axhline(y=10, color='r', linewidth=2, dashes=[10, 10])
            ax.axhline(y=-3, color='blue', linewidth=2, dashes=[10, 10])
            ax.axhline(y=-10, color='r', linewidth=2, dashes=[10, 10])
            
        # 绘制事件分界线 (Start/End)
        if not any(excluded in panel_name for excluded in ["GIC", "SymH"]):
            ax.axvline(event["start"], color="r", linestyle="-", linewidth=2)
            ax.axvline(event["end"], color="r", linestyle="-", linewidth=2)
            if event.get("shock") is not None:
                ax.axvline(event["shock"], color="green", linewidth=2)

    fig.align_ylabels()
    # ---------- 4. 时间窗口与格式化 ----------
    if time_window is not None:
        t0, t1 = time_window
        axes[-1].set_xlim(t0, t1)

    axes[-1].set_xlabel(f"{event['start'].strftime('%Y%m%d')}", fontsize=30)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    axes[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    
    plt.tight_layout()

    # ---------- 5. 保存 / 显示 ----------
    if savepath is not None:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        t_str = event["start"].strftime("%Y%m%d_%H%M") + event["end"].strftime("%Y%m%d_%H%M")
        filename = f"{t_str}.{figureformat}"
        full_path = os.path.join(savepath, filename)
        plt.savefig(full_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)