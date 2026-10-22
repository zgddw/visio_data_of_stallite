SolO NM3D PSD 导入修正版

修复内容
1. SOLO_ICME_density_pileup_plot.py 不再在启动导入阶段执行：
   from pro_of_read import cdfdataread
2. cdfdataread 改为首次真正读取 MAG/等离子体数据时延迟加载。
3. 延迟加载会打印开始、完成/失败和耗时，便于判断是导入慢还是数据读取慢。
4. 绘图公共模块和 solo_nm3d_pad 也在 main() 中加载，并检查模块路径，避免误用旧文件。
5. 未修改 PSP 程序，也未修改 pro_of_read.py 的 cdfdataread 实现。
6. solo_nm3d_pad.py 的 _mag_frame/read_solo_pad 不再导入公共绘图模块的时间解析函数，改用本地实现，避免 PAD 阶段再次触发公共模块初始化。
7. 自动下载的 EAS 文件按年份保存到 dataset\\YYYY\\；读取端兼容旧的直存文件，并且同名有效文件不会重复下载。
8. MAG-SRF 每日文件名只有一个 YYYYMMDD；现在按日期筛选所需文件。TT2000 先以原始整数定位事件区间，再只转换该区间的时间和磁场记录。
9. NASA 清单请求设有连接/响应超时和两次尝试；下载过程打印进度。PSD 的零值算有效能道，负值及填充值仍剔除。
10. 绘图、排版和 PNG 保存阶段均打印进度；入口一开始打印 Python 路径。
11. 绘图使用 Agg 后台模式，不弹出图形窗口；只保存 PNG，不再生成 PDF。PNG 默认 120 dpi，可用 --dpi 180 提高分辨率。

替换方法
将本目录中的以下三个 Python 文件复制到：
E:\ZhaoWenjin_Reserch\anaysis_prog\SOLO_NM3D_PSD\

SOLO_ICME_density_pileup_plot.py
ICME_density_pileup_plot_common.py
solo_nm3d_pad.py

其中后两个文件是与原目录一致的配套文件，建议一起复制以通过版本和路径检查。

推荐运行
在 PowerShell 中执行：

cd E:\ZhaoWenjin_Reserch\anaysis_prog\SOLO_NM3D_PSD
..\.venv\Scripts\python.exe .\SOLO_ICME_density_pileup_plot.py --offline --table "你的事件列表.csv"

本机另一套 D:\app\conda\python.exe 已用同一个事件验证可运行，启动和绘图明显更快。推荐改用：
D:\app\conda\python.exe -u .\SOLO_ICME_density_pileup_plot.py --table "你的事件列表.csv"

如果希望使用默认弹窗选择事件列表，可省略 --table；如果只想测试启动参数，可执行：

..\.venv\Scripts\python.exe .\SOLO_ICME_density_pileup_plot.py --help

正常日志应先出现：
[启动] 正在加载 SolO 绘图模块 ...

真正开始读取物理量时才出现：
[延迟导入] 正在加载 pro_of_read.cdfdataread ...
[延迟导入完成] pro_of_read 导入耗时 ... s

诊断
如果启动阶段仍然长时间没有输出，检查当前运行的脚本是否确实是上述路径中的修正版；日志中的“模块路径”必须指向 SOLO_NM3D_PSD 文件夹。
如果延迟导入失败，日志会显示 [延迟导入失败] 及原始异常。重点检查：
1. E:\ZhaoWenjin_Reserch\anaysis_prog\pro_of_read.py 是否存在；
2. E:\ZhaoWenjin_Reserch\anaysis_prog\stname.py 是否存在；
3. 当前 Python 是否为项目 .venv 中的 Python；
4. spacepy 是否安装在该 Python 环境中。

PAD 阶段正常情况下会出现：
[PAD] 开始读取 SolO NM3D PSD；时间解析器使用本地实现。

自动下载阶段还会显示：
[PAD检查] EAS1 2024-...：本地找到 ... 个文件。
[PAD网络] 正在查询 ... 的 NASA 文件清单 ...

2026-10-21 实测：
2020-06-15 15:00--15:10 UTC 的本地 PAD 读取成功，EAS1/EAS2 各处理 6 条记录，共 12 个时刻。一次联网测试将缺少的 EAS2 文件下载至 2020 年子目录，约 15.3 MiB。
同一事件的完整绘图在项目 .venv 下曾成功生成 PNG、PDF 和结果 CSV。更新为 PNG 专用后，D:\app\conda\python.exe 实测完整运行约 7 秒，其中 PAD 读取约 1 秒、PNG 保存约 0.6 秒，并且未生成 PDF。示例时段缺少 PAS 质子密度/速度/温度，所以相应面板为空。
若停在“正在加载 SolO 绘图模块”，独立导入诊断曾显示项目 .venv 的 NumPy 本地扩展加载耗时超过 45 秒；此时尚未开始读取 H: 的 EAS 文件。可用项目环境运行：
..\.venv\Scripts\python.exe -u -c "import time; t=time.perf_counter(); import numpy; print(time.perf_counter()-t)"
