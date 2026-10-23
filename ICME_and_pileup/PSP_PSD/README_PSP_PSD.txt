PSP ICME 密度堆积区综合绘图 — SPE 电子相空间密度版
================================================

本包只包含 PSP 专用新版，不覆盖原有 PSP 绘图程序，也不修改 SolO 文件。
将本文件夹中的 3 个 .py 文件放在同一个新目录，例如：
E:\ZhaoWenjin_Reserch\anaysis_prog\PSP_SPE_PAD_综合绘图修正版\
原工程的 pro_of_read.py 仍应位于 E:\ZhaoWenjin_Reserch\anaysis_prog\。

入口：PSP_ICME_density_pileup_plot.py
依赖：ICME_density_pileup_plot_common.py、psp_spe_pad.py，以及已有的
pro_of_read.py。需要 numpy、pandas、matplotlib、spacepy。

默认读取本机现有的：
  MAG: I:\stdata\psp\FLD\PSP_FLD_L2_MAG_RTN_1MIN\YYYY\
  SPC: I:\stdata\psp\sweap\psp_swp_spc_l3i\YYYY\
  SPE: I:\stdata\psp\sweap\spe\PSP_SWP_SPE_SF0_L3_PAD\YYYY\
  事件表: E:\ZhaoWenjin_Reserch\anaysis_prog\ICME_density_pileup_search_PSP_SOLO\结果_PSP_多参数密度堆积区\PSP_ICME_密度堆积区列表.csv

默认能量范围 70–2000 eV。入口 CONFIG 的 PAD_ENERGY_RANGE_EV 可修改，
或者运行时传 --energy-range 下限 上限，单位都是 eV，例如：
  python PSP_ICME_density_pileup_plot.py --energy-range 100 1000

首次建议只画一个事件，验证路径和图像：
  python PSP_ICME_density_pileup_plot.py --event-id ICME_PSP_MOESTL_20230310_01

也可指定自己的表、输出目录和分辨率：
  python PSP_ICME_density_pileup_plot.py --table "D:\我的列表.csv" --event-id ICME_PSP_MOESTL_20230310_01 --energy-range 70 2000 --output-dir "D:\PSP图像" --dpi 120

无 --event-id 时会依次处理事件表中所有“已检出”堆积区。
只保存 PNG 与 CSV 清单，不弹出绘图窗口，不保存 PDF，不进行 NASA 网络查询。

物理量说明
--------
PSP SPE L3 PAD 的 EFLUX_VS_PA_E 是微分能量通量，非原生 PSD。
程序读取逐时刻 ENERGY_VALS，并对每个能道先进行非相对论电子换算：
  f = m_e^2 * J_E / (2 * E^2)
其中 J_E 换算为 SI，E 为焦耳；最后输出 s^3/km^6，和 SolO
NM3D PSD 使用相同的数值单位。然后对选定能窗内、各投掷角的有效
能道 PSD 求和，形成时间 × 投掷角色图。它是离散能道求和，
不是对能量连续积分；用于跨任务比较时须保持相同能窗及解释口径。

PITCHANGLE/ENERGY_VALS 支持时间变化的 (N,12)/(N,32) 结构。
质量标志默认剔除计数溢出、替代能量表、spoiler 测试相关位，
但保留快照模式和机械衰减器标志，并在日志/清单注明非零标志。
若要更严格过滤，可修改 CONFIG 的 PAD_QUALITY_BAD_BITS。
非零质量标志、仪器视场遮挡、两传感器合成伪影均需人工核验，
不应把程序输出的 BDE 候选点直接当作已确认物理事件。

验证
----
在本机 2023-03-10 的真实 MAG/SPC/SPE CDF 和已有事件表上实测：
2944 个 PAD 时间点、12 个投掷角通道，完成一张十面板 PNG。
实际输出目录可由 --output-dir 指定，默认是本目录的“综合图像_PSP”。

产品说明：https://spase-metadata.org/NASA/NumericalData/ParkerSolarProbe/SWEAP/SPAN-E/Level3/PitchAngleDistribution/PT13.981S
SPAN-E caveats：https://sweap.cfa.harvard.edu/spe_data_release_notes.pdf
