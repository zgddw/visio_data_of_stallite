import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'PingFang SC',
    'Hiragino Sans GB', 'Noto Sans CJK SC',
    'WenQuanYi Zen Hei', 'Arial Unicode MS'
]
plt.rcParams['axes.unicode_minus'] = False

# ================= 物理参数（可修改） =================
T = 3*1.0e6                 # 温度 (K)
R_specific = 1.38e4       # 比气体常数 (J/(kg·K))
rho = 1.0e-20             # 密度 (kg/m^3)，不影响速度曲线
GM = 1.327e20             # 太阳引力常数 GM (m^3/s^2)
R_sun = 6.957e8           # 太阳半径 (m)

vc = np.sqrt(R_specific * T)
rc = GM / (2.0 * vc**2)
rc_Rsun = rc / R_sun

print(f"vc = {vc:.3e} m/s = {vc/1e3:.2f} km/s")
print(f"rc = {rc:.3e} m = {rc_Rsun:.3f} R_sun")

# ================= 方程求解：牛顿迭代法 =================
def g(y):
    """g(y) = y^2 - 2 ln y，在 y = 1 处取极小值 1"""
    return y * y - 2.0 * np.log(y)

def g_prime(y):
    """g'(y) = 2y - 2/y"""
    return 2.0 * y - 2.0 / y

def newton_iterate(R, y0, lo, hi, tol=1e-13, max_iter=200):
    """
    牛顿迭代法求解 g(y) = R，初值 y0，解限制在 (lo, hi) 内。
    迭代式：y_{n+1} = y_n - (y_n^2 - 2ln y_n - R) / (2y_n - 2/y_n)
    带简单越界保护：若牛顿步越界，则折半向边界靠拢。
    """
    y = y0
    for _ in range(max_iter):
        f  = y * y - 2.0 * np.log(y) - R
        fp = 2.0 * y - 2.0 / y
        if abs(fp) < 1e-16:
            break                       # 导数退化，终止
        y_new = y - f / fp
        # ---- 越界保护 ----
        if y_new <= lo:
            y_new = 0.5 * (y + lo)
        if y_new >= hi:
            y_new = 0.5 * (y + hi)
        # ---- 收敛判据 ----
        if abs(y_new - y) < tol * max(abs(y_new), 1.0):
            y = y_new
            break
        y = y_new
    return y

def solve_g(R):
    """用牛顿迭代法求解 g(y) = R 的所有正根"""
    if R < 1.0:                          # g(y) 的最小值为 1，R<1 无解
        return []

    roots = []

    # ---------------- 亚声速分支：0 < y < 1 ----------------
    if R - 1.0 < 0.5:
        # y 接近 1，利用 g(1+δ) ≈ 1 + 2δ² 的展开：y ≈ 1 - sqrt((R-1)/2)
        y0_sub = 1.0 - np.sqrt(0.5 * (R - 1.0))
    else:
        # R 较大时 -2 ln y 主导：y ≈ exp(-R/2)
        y0_sub = np.exp(-0.5 * R)
    roots.append(newton_iterate(R, y0_sub, 1e-300, 1.0))

    # ---------------- 超声速分支：y > 1 ----------------
    if R > 1.0:
        if R - 1.0 < 0.5:
            y0_sup = 1.0 + np.sqrt(0.5 * (R - 1.0))
        else:
            # R 较大时 y² 主导：y ≈ sqrt(R + ln R)
            y0_sup = np.sqrt(R + np.log(R))
        roots.append(newton_iterate(R, y0_sup, 1.0, 1e300))

    return roots

# ================= 采样策略：分段变步长（保持不变） =================
n_side = 20000
ratios_left  = np.logspace(np.log10(1e-4), np.log10(1.0), n_side)
ratios_right = np.logspace(np.log10(1.0), np.log10(1e3),  n_side)

r_left  = rc_Rsun * ratios_left[::-1]
r_right = rc_Rsun * ratios_right

r_near = np.unique(np.concatenate([r_left, [rc_Rsun], r_right]))
r_far  = np.linspace(rc_Rsun * 1e3, 600, 500)
r_over_Rsun = np.unique(np.concatenate([r_near, r_far]))

r_over_Rsun = r_over_Rsun[r_over_Rsun > 0.005]
x_norm = r_over_Rsun * R_sun / rc

print(f"总采样点数: {len(r_over_Rsun)}")

# ================= 求解并缓存 =================
C_values = [-1, -2, -3, -4, -5, -6, 1, 2, 3, 4, 5, 6]
results = {}

for C in C_values:
    u_sub = np.full_like(x_norm, np.nan)
    u_sup = np.full_like(x_norm, np.nan)
    for j, x in enumerate(x_norm):
        R = 4.0 * np.log(x) + 4.0 / x + C
        roots = solve_g(R)
        if len(roots) >= 1:
            u_sub[j] = roots[0] * vc
        if len(roots) == 2:
            u_sup[j] = roots[1] * vc
    results[C] = (u_sub, u_sup)

# ================= 绘图（与原程序一致） =================
fig, ax = plt.subplots(figsize=(12, 7.5))
colors = plt.cm.jet(np.linspace(0, 1, len(C_values)))

for i, C in enumerate(C_values):
    u_sub, u_sup = results[C]
    ax.plot(r_over_Rsun, u_sub, linestyle='-', color=colors[i], linewidth=1.2)
    ax.plot(r_over_Rsun, u_sup, linestyle='-',  color=colors[i], linewidth=1.6)

ax.plot(rc_Rsun, vc, 'ko', markersize=9)
ax.text(rc_Rsun + 12, vc * 0.75, r'临界点 $(r_c,v_c)$', fontsize=11)

ax.set_xlim(0, 600)
ax.set_ylim(1e3, 2e6)
ax.set_yscale('log')
ax.set_xlabel(r'距离  $r / R_\odot$', fontsize=13)
ax.set_ylabel(r'速度  $u$ (m/s)', fontsize=13)
ax.set_title('帕克太阳风解：不同 $C$ 值下的 $u$–$r$ 曲线', fontsize=15)
ax.grid(True, which='both', linestyle='-', alpha=0.5)

legend_elements = [
    Line2D([0], [0], color='k', linestyle='-'),
    Line2D([0], [0], color='k', linestyle='-')
]
for i, C in enumerate(C_values):
    legend_elements.append(Line2D([0], [0], color=colors[i], lw=2, label=f'C={C}'))
ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

# ================= 局部放大插图 =================
ax_ins = fig.add_axes([0.38, 0.16, 0.36, 0.36])
for C in [-3, -4, -5, -6, 1, 3]:
    i = C_values.index(C)
    u_sub, u_sup = results[C]
    ax_ins.plot(r_over_Rsun, u_sub, linestyle='-', color=colors[i], linewidth=1.0)
    ax_ins.plot(r_over_Rsun, u_sup, linestyle='-',  color=colors[i], linewidth=1.3,
                label=f'C={C}')
ax_ins.plot(rc_Rsun, vc, 'ko', markersize=6)
ax_ins.set_xlim(rc_Rsun * 0.3, rc_Rsun * 3)
ax_ins.set_ylim(vc * 0.3, vc * 3)
ax_ins.set_yscale('log')
ax_ins.set_xlabel(r'$r/R_\odot$', fontsize=9)
ax_ins.set_ylabel(r'$u$ (m/s)', fontsize=9)
ax_ins.set_title('临界点附近放大', fontsize=10)
ax_ins.tick_params(labelsize=8)
ax_ins.grid(True, which='both', linestyle=':', alpha=0.4)
ax_ins.legend(fontsize=7, loc='best', ncol=2)

# ================= Figure 2：单独画 C=-3 =================
fig2, ax2 = plt.subplots(figsize=(10, 6.5))

i3 = C_values.index(-3)
u_sub_3, u_sup_3 = results[-3]

ax2.plot(r_over_Rsun, u_sub_3, linestyle='-', color='royalblue', linewidth=1.8)
ax2.plot(r_over_Rsun, u_sup_3, linestyle='-', color='royalblue', linewidth=2.0)

ax2.plot(rc_Rsun, vc, 'ko', markersize=10, zorder=5)
ax2.axhline(vc,      color='gray', linestyle=':', linewidth=1)
ax2.axvline(rc_Rsun, color='gray', linestyle=':', linewidth=1)
ax2.text(rc_Rsun + 8, vc * 0.55, r'临界点 $(r_c,\,v_c)$', fontsize=12)

ax2.set_xlim(0, 600)
ax2.set_ylim(1e3, 2e6)
ax2.set_yscale('log')
ax2.set_xlabel(r'距离  $r / R_\odot$', fontsize=13)
ax2.set_ylabel(r'速度  $u$ (m/s)',     fontsize=13)
ax2.set_title(r'$C=-3$ 时的帕克太阳风解', fontsize=14)
ax2.grid(True, which='both', linestyle='-', alpha=0.5)
ax2.legend(fontsize=11, loc='lower right')

ax2_ins = fig2.add_axes([0.60, 0.55, 0.28, 0.30])
ax2_ins.plot(r_over_Rsun, u_sub_3, '-', color='royalblue', linewidth=1.2)
ax2_ins.plot(r_over_Rsun, u_sup_3, '-',  color='royalblue',   linewidth=1.4)
ax2_ins.plot(rc_Rsun, vc, 'ko', markersize=6)
ax2_ins.axhline(vc,      color='gray', linestyle=':', linewidth=0.8)
ax2_ins.axvline(rc_Rsun, color='gray', linestyle=':', linewidth=0.8)
ax2_ins.set_xlim(rc_Rsun * 0.4, rc_Rsun * 2.5)
ax2_ins.set_ylim(vc * 0.4, vc * 2.5)
ax2_ins.set_yscale('log')
ax2_ins.set_xlabel(r'$r/R_\odot$', fontsize=9)
ax2_ins.set_ylabel(r'$u$ (m/s)',    fontsize=9)
ax2_ins.set_title(r'临界点附近（$C=-3$）', fontsize=10)
ax2_ins.tick_params(labelsize=8)
ax2_ins.grid(True, which='both', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.show()