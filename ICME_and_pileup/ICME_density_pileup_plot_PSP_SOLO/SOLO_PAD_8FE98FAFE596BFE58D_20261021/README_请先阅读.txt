SolO PAD 修复（仅 Solar Orbiter，不包含、不修改 PSP）。

本次错误根因：通用 PAD 函数错误地要求 SolO 文件存在 PITCHANGLE 变量。
SolO 的 SWA-EAS-PAD-DEF 实际不含独立 PITCHANGLE，需使用 EPOCH、
SWA_EAS_PAD_DEF_Data、SWA_EAS_ELEVATION、SWA_EAS_ENERGY、
SWA_EAS_AZIMUTH 和 EAS_TO_SRF，配合同时刻 MAG-SRF 计算投掷角。

用法：将压缩包内全部三个 py 文件放在同一目录运行，勿仅替换 SOLO 入口。
尤其注意旧目录里不要残留别的同名 ICME_density_pileup_plot_common.py。
入口脚本中配置：
  PAD_ROOT: SolO PAD-DEF 数据根目录（可递归搜索）
  PAD_SRF_MAG_ROOT: SolO MAG-SRF 数据根目录（不能指向 RTN）
  MAG_ROOT: 原 RTN MAG 目录，绘制其余磁场面板
  PAD_TARGET_ENERGY_EV: 300.0（最近能道）
  PAD_MAG_MAX_DELTA_SECONDS: 90（MAG-SRF 匹配最大时间差）
  PLOT_CONTEXT_HOURS: 2.0

示例：
python SOLO_ICME_density_pileup_plot.py --table "D:\\research\\events.csv"

结果自动保存 SOLO_密度堆积区_BDE时间占比.csv。
BDE_Ratio_Percent：BDE 时间 / 有效 PAD 覆盖时间；数据缺失为 NaN。
数据跨度不足/仪器角度有盲区时，占比可能不可靠；需查看 PAD_Coverage_Percent。

说明：用户上传的 CDF skeleton 仅记录变量结构。修复已通过语法及伪造记录测试，
但当前运行环境没有 spacepy，无法以真实 CDF 文件完整运行。
仪器视线与电子速度向量的符号约定，默认 PAD_LOOK_IS_OPPOSITE_VELOCITY=True；
应用于物理定量结论前仍须根据仪器规范核实，注意 0°/180°方向会随符号变动。
