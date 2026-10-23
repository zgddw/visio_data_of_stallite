ICME 密度堆积区综合绘图（PSP / Solar Orbiter 分开版）
==========================================================

文件
----
1. PSP_ICME_density_pileup_plot.py
   PSP 专用入口程序。
2. SOLO_ICME_density_pileup_plot.py
   Solar Orbiter 专用入口程序。
3. ICME_density_pileup_plot_common.py
   两个入口共用的读取、计算与绘图函数。必须与两个入口放在同一目录。

程序按已生成的密度堆积区列表逐条绘图，每个堆积区输出一张完整的多面板图，
不是把每个物理参数保存成独立图片。PNG 用于查看和插入文档，PDF 为矢量版备份。

图像面板（从上到下）
--------------------
1. BDE 判据时间点：仅当最接近 0° 与 180° 的通量都大于等于 90° 附近通量时，
   在对应时间画红色圆点。PSP 的 PA 产品中心为 7.5°、22.5°、...、172.5°，
   因此 0° / 180° 使用最近的两个端点角通道，90° 使用最近的两个通道的均值。
2. 电子投掷角通量谱图。
3. |B_T|，RTN-T 分量绝对值。
4. RTN 磁场三分量 B_R、B_T、B_N（有符号）。
5. 磁场方位角 phi = atan2(B_T, B_R)，范围 [-180°, 180°]。
6. 磁场纬向角 theta = atan2(B_N, sqrt(B_R^2 + B_T^2))，范围 [-90°, 90°]。
7. 质子速度模长。
8. 质子温度（图中以 10^6 K，即百万 K 为单位显示）。
9. 质子密度。
10. 质子 beta（由质子热压和总磁场模长计算，与检索程序定义保持一致）。

所有面板默认显示 Pileup_Start_UTC 前 2 小时到 Pileup_End_UTC 后 2 小时的背景；堆积区本身的起止边界以红色虚线标记。可在入口脚本配置 PLOT_CONTEXT_HOURS 调整该间隔。
电子能量范围默认 200–500 eV，可在两套入口脚本中修改：
    "PAD_ENERGY_RANGE_EV": (200.0, 500.0)
例如改成 (200.0, 1000.0)。

输入事件表（路径由你自己指定）
------------------------------
先运行对应卫星的 ICME 密度堆积区检索程序，生成包含 Pileup_Start_UTC 与 Pileup_End_UTC 列的检索结果表。
绘图程序不会强制使用预设列表路径：运行时会弹出文件选择窗口，让你自行选 CSV、XLSX 或 XLS 文件。
如果文件选择窗口不可用，可以在终端中输入完整路径；也可以用命令行参数直接指定。

示例：
    python PSP_ICME_density_pileup_plot.py --table "D:\你的目录\PSP密度堆积区列表.csv"
    python SOLO_ICME_density_pileup_plot.py --table "D:\你的目录\SOLO密度堆积区列表.xlsx"

也可以不提供 --table，运行后通过文件窗口选择：
    python PSP_ICME_density_pileup_plot.py
    python SOLO_ICME_density_pileup_plot.py

如果希望直接在入口脚本中固定路径，分别修改脚本顶部 CONFIG 中的：
    "DETECTED_TABLE": Path(r"D:\你的目录\密度堆积区列表.csv")
如果留为 None，启动时就会让你选择。若明确输入了不存在的路径，程序会报错提示，不会悄悄改读其它表。

绘图输出目录也可自行指定，例如：
    python PSP_ICME_density_pileup_plot.py --table "D:\你的目录\列表.csv" --output-dir "D:\你的目录\图像输出"

仅在不想弹出窗口、希望使用程序默认位置自动查找时，才使用：
    python PSP_ICME_density_pileup_plot.py --no-table-dialog

运行依赖
--------
请将本包 3 个 .py 文件放在你的工程目录，确认 pro_of_read.py 与 CDF 数据文件夹可访问，然后运行上面的命令。

依赖：Python 3.9+、numpy、pandas、matplotlib、spacepy，以及你既有工程里的 pro_of_read.py / cdfdataread()。

默认数据路径
------------
PSP MAG: I:\stdata\psp\FLD\PSP_FLD_L2_MAG_RTN_1MIN
PSP SPC: I:\stdata\psp\sweap\psp_swp_spc_l3i
PSP SPE PAD: I:\stdata\psp\sweap\spe\PSP_SWP_SPE_SF0_L3_PAD
SOLO MAG: I:\stdata\solo\mag\RTN_NORMAL
SOLO SWA-PAS moments: I:\stdata\solo\swa\solo_l2_swa-pas-grnd-mom
SOLO SWA-EAS PAD: I:\stdata\solo\swa\solo_l2_swa-eas-pad-def
如实际目录不同，请修改对应入口脚本顶部的路径和文件前缀。

关于电子 PAD 的重要说明
-----------------------
PSP 使用 SWEAP/SPE L3 PAD 产品（每日文件通常命名为 psp_swp_spe_sf0_l3_pad_YYYYMMDD_v01.cdf）。读取时优先尝试说明文件标为 data/谱图视图的 EFLUX_VS_PA_E_atE_byP，再回退 EFLUX_VS_PA_E_byE_atP 和基础变量 EFLUX_VS_PA_E；每个候选变量均检查 time × pitch angle × energy 的维度及有效通量。
常见 CDF 变量包含 EFLUX_VS_PA_E、PITCHANGLE 和 ENERGY_VALS。该产品的采样间隔约为 13.981 秒。
数据说明：https://spase-metadata.org/NASA/NumericalData/ParkerSolarProbe/SWEAP/SPAN-E/Level3/PitchAngleDistribution/PT13.981S

Solar Orbiter 的 SWA-EAS PAD-DEF 目录可能提供的是 elevation × energy × azimuth 四维传感器数据，
而不是直接带有投掷角坐标的 time × pitch-angle × energy 三维数组。程序只在 CDF 中找到明确的
投掷角轴和相匹配通量数组时，才绘制真正的电子投掷角谱图并计算 BDE 红点。如果只有原始四维
EAS 数据，程序会在投掷角面板写明原因，但继续绘制磁场、质子速度、温度、密度和 beta。此处不会
把 EAS 传感器方位角直接冒充 0°–180° 投掷角；需要先使用 EAS 视线方向、航天器姿态和同期磁场
方向将原始角分布转换到磁场对齐投掷角后，才能进行可靠的 BDE 判定。
参考：https://doi.org/10.1051/0004-6361/202140959

输出
----
PSP 保存到“综合图像_PSP”，SOLO 保存到“综合图像_SOLO”。每个堆积区输出：
  <事件编号_起始时间_结束时间>.png
  <事件编号_起始时间_结束时间>.pdf
并生成“<卫星>_综合图像生成清单.csv”；失败项写入独立的绘图失败清单。

注意
----
已完成语法检查，并使用按 PSP skeleton table 的 12 个 PA × 32 个能量通道构造的模拟 CDF 数据，测试了
PAD 候选变量回退、200–500 eV 能量筛选、BDE 标记以及事件前后各 2 小时的 PNG/PDF 图像生成。当前附件是
CDF skeleton table（记录数为 0），不是实际观测 CDF；本运行环境也不能访问你 Windows 本地 I: 盘和
pro_of_read.py，因此真实事件 CDF 尚未完成端到端验证。运行后如果仍没有投掷角，面板中的诊断信息会列出
搜索目录、通量变量候选和具体失败原因。
