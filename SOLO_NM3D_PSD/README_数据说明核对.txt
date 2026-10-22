SolO EAS1/EAS2 NM3D PSD 数据说明核对修正版（仅 SolO，PSP 不变）

CDF 说明与程序字段对应：
 EAS1/EAS2 数据 SWA_EAS1_Data / SWA_EAS2_Data，数组形状 (Nt,16,64,32)，单位 s^3 km^-6。
 EAS1/EAS2 能量 SWA_EAS1_ENERGY / SWA_EAS2_ENERGY，64 能道，单位 eV，非时间依赖。
 SWA_EAS_ELEVATION 16 个角度、SWA_EAS_AZIMUTH 32 个角度，二者均可为非时间依赖元数据。
 EAS1_TO_SRF / EAS2_TO_SRF 为各仪器科学坐标系至本体 SRF 的 3x3 常量变换矩阵。
 QUALITY_FLAG (Nt,) 标称 0--4、填充值 255，默认剔除超出 0--4 的时间样本；具体质量含义须参考 SWA caveats。

变更：
 1. 更明确地处理 3x3 固定旋转矩阵；即使恰好 Nt=3，也不会误判成时间序列。
 2. 电子方向符号默认不翻转：NM3D 文件的 SWA_EAS_AZIMUTH 文档称 electron flow direction，配置可改。
    但仰角文字为 instrument look directions，仍需用真实文件/已知单向 strahl 做符号校验。
 3. 过滤质量标志填充值；可通过 PAD_MAX_QUALITY_FLAG 修改严格程度。
 4. 完整保留按需下载、已存在文件跳过、共享时间轴、BDE占比等原有行为。

注意：CDF skeleton 文件没有真实记录，未执行真实 NM3D 与同期 MAG-SRF 的端到端验证。
请先选有 EAS1/EAS2/MAG-SRF 同期数据的单个事件，检查投掷角 0/180 方位与 B_R 的关系。
若有局部数据读取失败，提供 [PAD 提示] 的完整日志及实际文件的变量维度输出。
