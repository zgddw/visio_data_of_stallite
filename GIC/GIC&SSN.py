import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 1. 准备数据 (请在此处替换为你的真实数据)
years = np.arange(1999, 2021)
# 模拟数据：假设这是一个增长趋势
NOFGIC = [3, 13, 18, 9, 10, 7, 2, 1, 0, 0, 0, 0, 2, 2, 3,0, 3, 3, 5, 0,0, 0]
SSN = [123.075,161.075,170.2833,163.6083,99.0333,65.3333,45.725,24.625,
       12.6333,4.15,4.75,24.9,80.7583,84.3916,93.7083,113.6083,69.7833,
       39.825,21.8166,7.0083,3.5833,8.7917]

# 2. 风格设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

# --- 绘制左侧轴 (柱状图) ---
color_bar = '#34495e' # 深灰色/蓝色，显稳重
bars = ax1.bar(years, NOFGIC, color=color_bar, alpha=0.7, label='NO.of events')
ax1.set_xlabel('Year', fontsize=18)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax1.set_ylim(0, 20)
ax1.set_ylabel('NO.of events', fontsize=18, color=color_bar)
ax1.tick_params(axis='y', labelcolor=color_bar,direction='in') # 坐标刻度颜色与图形一致

# --- 绘制右侧轴 (折线图) ---
ax2 = ax1.twinx() # 创建共享 x 轴的第二个 y 轴
color_line = '#e74c3c' # 红色，突出趋势
line, = ax2.plot(years, SSN, color=color_line, marker='s',
                 markersize=4, linewidth=1.5, label='SSN(annual average)')
ax2.set_ylabel('SSN(annual average)', fontsize=18, color=color_line)
ax2.tick_params(axis='y', labelcolor=color_line,direction='in') # 坐标刻度颜色与图形一致

# 3. 细节微调
ax1.set_xticks(years)
ax1.set_xticklabels(years, rotation=45)

# 移除上方边框，保留左右轴
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
# 确保顶部也有刻度线且向内
ax1.tick_params(top=True, labeltop=False, direction='in')

# 4. 合并图例
# 由于是两个不同的 Axes 对象，需要手动合并标签
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right',fontsize=18, frameon=False)

# 5. 自动布局并保存
plt.tight_layout()
plt.savefig('dual_axis_chart.tif', format='tif', dpi=300,bbox_inches='tight')
plt.show()