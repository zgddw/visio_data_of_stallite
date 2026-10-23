ICME 密度堆积区自动检索程序（PSP / Solar Orbiter 分开版）
============================================================

文件
----
1. PSP_ICME_density_pileup_search.py
   PSP 专用程序。
2. SOLO_ICME_density_pileup_search.py
   Solar Orbiter 专用程序。
3. ICME_PSP.txt
   本包附带的 HELIO4CAST/PSP 事件表后备输入。若你要使用自己整理的
   “有激波/无激波”列表，请优先检查脚本顶部 EVENT_LIST_FILES。

依赖
----
Python 3.9+；numpy、pandas；可选 openpyxl（写 Excel 用）；以及你原有工程的
pro_of_read.py，其中必须能导入 cdfdataread()。
程序会尝试从脚本所在目录、E:\ZhaoWenjin_Reserch\pileup 和
E:\ZhaoWenjin_Reserch\anaysis_prog 查找 pro_of_read.py。若它在其他地方，
请修改脚本顶部 CUSTOM_MODULE_DIRS。

如何运行
--------
将脚本放在你的 Windows 工程目录，确认数据目录、事件表路径正确，然后分别运行：

    python PSP_ICME_density_pileup_search.py
    python SOLO_ICME_density_pileup_search.py

两个脚本分别运行、分别输出，互不混用结果。

数据路径与事件列表
------------------
PSP 默认使用：
  磁场目录：I:\stdata\psp\FLD\PSP_FLD_L2_MAG_RTN_1MIN
  等离子体目录：I:\stdata\psp\sweap\psp_swp_spc_l3i
  有激波列表：E:\ZhaoWenjin_Reserch\pileup\有激波psp.txt
  无激波列表：E:\ZhaoWenjin_Reserch\pileup\无激波psp.txt
  如果两份自定义列表都不存在，后备读取 ICME_PSP.txt；如果只找到其中一份，
  程序会尝试从综合表补充缺失的有/无激波类别。

Solar Orbiter 默认使用：
  磁场目录：I:\stdata\solo\mag\RTN_NORMAL
  等离子体目录：I:\stdata\solo\swa\solo_l2_swa-pas-grnd-mom
  可选分列表：E:\ZhaoWenjin_Reserch\pileup\有激波solo.txt 和 无激波solo.txt
  如果上述分列表都不存在，先尝试你之前 seaana.py 中使用的综合列表：
  E:\ZhaoWenjin_Reserch\pileup\solo列表.txt；再尝试脚本同目录的 ICME_SOLO.txt。
  如果只找到其中一份分列表，程序会尝试从综合表补充缺失类别。

上述路径沿用你之前项目文件里出现过的目录及命名。若你当前文件夹或文件名前缀有变化，
只需修改相应脚本顶部的 EVENT_LIST_FILES、FALLBACK_EVENT_FILES、MAG_ROOT、MAG_PREFIX、
PLASMA_ROOT、PLASMA_PREFIX，不需要改核心算法。

事件列表的基本格式
------------------
每行至少要有 2 个可解析的时间戳。三时间列格式按：激波时刻、ICME 本体开始、ICME 本体结束。
若是独立的无激波列表且仍保留三时间列，程序忽略第一列，将第二、三时间列作为本体起止时间；
若只有两个时间戳，则按本体开始、结束解释。支持“YYYY-MM-DD HH:MM:SS”以及
“YYYY-MM-DDTHH:MMZ”形式。其它列会忽略。

检索算法与阈值
--------------
- 有激波：激波时刻到 ICME 本体结束后 2 h。
- 无激波：ICME 本体开始前 6 h 到本体结束后 2 h。
- 窗口时长从整个检索范围最大长度开始，每次缩短 1 min，直至 5 min；每个长度都按 1 min 步长滑动。
- 对候选窗口与其前、后连续 2 h 的均值比较：
    Np >= 2 × max(Np_pre, Np_post)
    beta_p >= 2 × max(beta_pre, beta_post)
    mean(|B_T|) <= 0.6 × min(mean(|B_T|)_pre, mean(|B_T|)_post)
- 磁场第三项判据改为 RTN 坐标的 T 分量强度 |B_T|，采用绝对值以避免分量正负号影响“强度”比较。
  注意：若 CDF 读取模块将三维 RTN 向量展开为 `_X/_Y/_Z`，程序按第二分量 `_Y` 对应 T 分量；若文件列名直接标为 `B_RTN_T`，则优先使用该列。
- Beta 仍使用总磁场模长 |B| 计算，不使用单独的 B_T；这是为了保持 beta 的物理定义。第三项判据单独使用 |B_T|。
- 候选区及两侧参照区中，三个参数各自的有效覆盖率均需 >= 90%。不插值补全缺测。
- 每个 ICME 默认最终最多输出一个密度堆积区，避免同一事件重复检出过多区间。
- 可配置参数：两个候选区之间的最大合并间隔为 `MAX_GAP_BETWEEN_REGIONS_MINUTES = 120`，单位分钟。
  例如可改为 60、90 或 180 分钟。
- 若相邻候选区间的间隔不超过该阈值，程序将它们归入同一个候选簇，并尝试用最早起点至最晚终点合并为较长事件。
  合并后的整段必须重新通过至少一个物理判据；如果合并后不再通过，则不会伪造一个不满足阈值的长事件，
  而是保留该候选簇内距离 ICME 本体前端最近的原始有效区间。
- 若候选区形成多个彼此间隔超过阈值的候选簇，只保留距离 ICME 本体前端最近的候选簇，其余较远候选簇舍弃。
- “距离 ICME 本体前端”按区间与 `ICME_Start_UTC` 的时间距离计算：在本体前的候选区测量其后端至本体前端的距离；
  跨越本体前端时距离为 0；位于本体开始之后的候选区测量其前端至本体前端的距离。
- 同一距离下优先持续时间较长、等级较高的候选；局部核心只在最终保留的候选区/候选簇内记录，不借用已舍弃远端事件的核心信息。

事件等级划分
------------
- 强典型密度堆积区：质子密度、Beta、|B_T| 三项条件全部满足。
- 典型密度堆积区：质子密度满足，且 Beta、|B_T| 中至少一项满足，但不是三项全满足。
- 弱典型密度堆积区：仅质子密度条件满足。
- 非典型密度堆积区：质子密度条件不满足，但 Beta、|B_T| 中至少一项满足（包括其中一项或两项满足）。仍属于检出事件，保留在“密度堆积区列表.csv”。
- 非密度堆积区：三个条件全部不满足。对应的 ICME 单独写入“未检出密度堆积区的ICME列表.csv”，不与检出事件混在一起。
- 数据不足或读取失败：单独进入无法判定列表，不归为非密度堆积区。

关于 beta 的定义（请特别注意）
-----------------------------
本版本使用 PSP 的 wp_moment 按你旧版 seaana.py 的换算得到质子温度，SOLO 的 T 按 eV 转成 K，
然后按 beta_p = 2*mu0*n_p*kB*T_p/B^2 计算**质子 beta**。如果你的研究所说的“等离子体 beta”
采用 compcalculate.Pandbeta() 中不同的压力组成/温度定义，正式统计前务必统一定义。

输出
----
各脚本在自身所在目录创建独立结果文件夹：
- 结果_PSP_多参数密度堆积区
- 结果_SOLO_多参数密度堆积区

每个文件夹会输出：
- `密度堆积区列表.csv`：只放检出的强典型、典型、弱典型和非典型密度堆积区，不混入三项全不满足的 ICME。
- `未检出密度堆积区的ICME列表.csv`：数据足够，但质子密度、Beta、|B_T| 三项判据全部不满足的 ICME（非密度堆积区）。该表与检出列表分开保存。
- `数据不足或读取失败的ICME列表.csv`：数据覆盖不足或读取过程失败、无法可靠判定的 ICME，不将其误判为非密度堆积区。
- `ICME检索摘要.csv`：仍包含全部处理过的 ICME 事件状态。
- 可选 Excel 文件，包含“密度堆积区列表”“未检出堆积区ICME”“数据不足或读取失败”“ICME检索摘要”四个工作表。

结果字段包括事件编号、激波状态、ICME/检索/堆积区起止时间、持续时间、到 ICME 本体整体及前端的间隔、
质子密度、Beta、|B_T| 三个判据状态、候选区与前后 2 h 均值、各自倍数/比例、有效覆盖率、局部核心区等级、合并候选数和筛选说明。|B_T| 的字段以 `BT_abs_` 开头。
ICME 检索摘要另记录筛选前候选数量、最终保留数量、合并/舍弃数量以及所使用的间隔阈值。

测试说明
--------
程序已经通过 Python 语法编译检查，并用合成时间序列验证了核心窗口扫描/分类逻辑。
当前环境无法访问你 Windows 电脑上的实际 CDF 文件和 pro_of_read.py，所以尚未完成真实数据的
端到端运行。首次运行请先选 1 个 PSP、1 个 SOLO 事件检查变量列与 beta 数值，再批量处理。
