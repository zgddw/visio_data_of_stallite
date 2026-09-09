"""SOLO + PSP 完整处理程序

功能：
1. 读取两个原始 Excel，排除 SOLO 的 Has_Shock=False 事件；
2. 将 PSP 阶段映射为统一阶段；
3. 按六个参数的合并全局范围等距分箱，计算各阶段频数和百分比；
4. 计算同一阶段内所有事件的参数平均值；
5. 输出统计 Excel 和六参数同画布折线图（SVG，无需 matplotlib）。

运行示例：
python solo_psp_full_pipeline.py
python solo_psp_full_pipeline.py --solo "E:\\...\\solo参数0.5阈值.xlsx" --psp "E:\\...\\psp有激波参数分布0.5阈值.xlsx" -n 20
"""

from __future__ import annotations
import argparse, html, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SOLO_DEFAULT = Path(r"E:\ZhaoWenjin_Reserch\pileup\solo参数0.5阈值.xlsx")
PSP_DEFAULT = Path(r"E:\ZhaoWenjin_Reserch\pileup\psp有激波参数分布0.5阈值.xlsx")
OUT_DEFAULT = Path("outputs")
PARAMS = [
    "B_total_mean_nT",
    "N_mean_cm-3",
    "V_total_mean_km/s",
    "T_mean_K",
    "Pth_mean_nPa",
    "beta_mean",
]
PHASES = ["Shock_to_Mark1", "Mark1_to_Mark2", "ICME"]
PHASE_DISPLAY = {
    "Shock_to_Mark1": "Sheath",
    "Mark1_to_Mark2": "Pileup",
    "ICME": "ICME",
}
PSP_MAP = {
    "Shock_to_DensityStart": "Shock_to_Mark1",
    "Density_Enhancement": "Mark1_to_Mark2",
    "ICME": "ICME",
}
COLORS = {PHASES[0]: "#2a6fbb", PHASES[1]: "#e68a2e", PHASES[2]: "#3c9d62"}
LABELS = {
    "B_total_mean_nT": "Magnetic field B (nT)",
    "N_mean_cm-3": "Density N (cm-3)",
    "V_total_mean_km/s": "Speed V (km/s)",
    "T_mean_K": "Temperature T (K)",
    "Pth_mean_nPa": "Thermal pressure Pth (nPa)",
    "beta_mean": "Plasma beta",
}


def truth(s):
    return (
        s.fillna(False)
        if pd.api.types.is_bool_dtype(s)
        else s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    )


def read_solo(path):
    d = pd.read_excel(path)
    req = {"Event_Index", "Has_Shock", "Phase", *PARAMS}
    miss = req - set(d.columns)
    if miss:
        raise ValueError(f"SOLO 缺少列：{', '.join(sorted(miss))}")
    d = d.loc[truth(d.Has_Shock)].copy()
    d["Event_ID"] = "SOLO-" + d.Event_Index.astype(str)
    d["Source"] = "SOLO"
    d["Original_Phase"] = d.Phase
    d["Unified_Phase"] = d.Phase
    return d[["Event_ID", "Source", "Original_Phase", "Unified_Phase", *PARAMS]]


def read_psp(path):
    d = pd.read_excel(path)
    req = {
        "Event_No",
        "Phase",
        "B_total_mean_nT",
        "np_mean_cm-3",
        "V_total_mean_km/s",
        "T_mean_K",
        "Pth_mean_nPa",
        "beta_mean",
    }
    miss = req - set(d.columns)
    if miss:
        raise ValueError(f"PSP 缺少列：{', '.join(sorted(miss))}")
    d = d.copy()
    d["Event_ID"] = "PSP-" + d.Event_No.astype(str)
    d["Source"] = "PSP"
    d["Original_Phase"] = d.Phase
    d["Unified_Phase"] = d.Phase.map(PSP_MAP)
    if d.Unified_Phase.isna().any():
        raise ValueError(
            f"PSP 存在未映射阶段：{d.loc[d.Unified_Phase.isna(), 'Phase'].unique().tolist()}"
        )
    return d.rename(columns={"np_mean_cm-3": "N_mean_cm-3"})[
        ["Event_ID", "Source", "Original_Phase", "Unified_Phase", *PARAMS]
    ]


def make_frequency(d, n):
    rows = []
    for p in PARAMS:
        vals = pd.to_numeric(d[p], errors="coerce").dropna()
        lo, hi = vals.min(), vals.max()
        edges = (
            np.array([lo - 0.5, hi + 0.5])
            if np.isclose(lo, hi)
            else np.linspace(lo, hi, n + 1)
        )
        for phase in PHASES:
            v = pd.to_numeric(
                d.loc[d.Unified_Phase == phase, p], errors="coerce"
            ).dropna()
            counts, _ = np.histogram(v, bins=edges)
            for i, c in enumerate(counts, 1):
                rows.append(
                    {
                        "Parameter": p,
                        "Phase": phase,
                        "Bin_Number": i,
                        "Bin_Lower": edges[i - 1],
                        "Bin_Upper": edges[i],
                        "Frequency": int(c),
                    }
                )
    return pd.DataFrame(rows)


def make_svg(freq, means, outpath):
    W, H, mx, top, gx, gy = 1800, 1080, 70, 120, 45, 70
    pw, ph = (W - 2 * mx - 2 * gx) / 3, 390
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1080" viewBox="0 0 1800 1080">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd}.axis{stroke:#333}</style>",
        '<text x="900" y="38" text-anchor="middle" font-size="24" font-weight="bold">Combined SOLO + PSP: parameter distributions by phase</text>',
    ]
    for i, phase in enumerate(PHASES):
        x = 650 + i * 190
        out += [
            f'<line x1="{x}" y1="75" x2="{x + 30}" y2="75" stroke="{COLORS[phase]}" stroke-width="3"/>',
            f'<text x="{x + 38}" y="81" font-size="14">{phase}</text>',
        ]
    for idx, (p, label) in enumerate(LABELS.items()):
        row, col = divmod(idx, 3)
        x0 = mx + col * (pw + gx)
        y0 = top + row * (ph + gy)
        left, right, bottom, pt = x0 + 60, x0 + pw - 15, y0 + ph - 45, y0 + 38
        s = freq[freq.Parameter == p]
        b = s.drop_duplicates("Bin_Number").sort_values("Bin_Number")
        xv = (b.Bin_Lower.to_numpy(float) + b.Bin_Upper.to_numpy(float)) / 2
        lo, hi = float(b.Bin_Lower.min()), float(b.Bin_Upper.max())
        ymax = max(
            10.0, float((s.Frequency / means.Event_Count.max() * 100).max()) * 1.12
        )
        out.append(
            f'<text x="{(left + right) / 2:.1f}" y="{y0 + 18}" text-anchor="middle" font-size="17" font-weight="bold">{html.escape(label)}</text>'
        )
        for t in range(6):
            yy = bottom - t * (bottom - pt) / 5
            out += [
                f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}"/>',
                f'<text x="{left - 8}" y="{yy + 5:.1f}" text-anchor="end" font-size="12">{t * ymax / 5:.0f}</text>',
            ]
        out += [
            f'<line class="axis" x1="{left}" y1="{pt}" x2="{left}" y2="{bottom}"/><line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            f'<text x="{x0 + 15}" y="{(pt + bottom) / 2:.1f}" transform="rotate(-90 {x0 + 15} {(pt + bottom) / 2:.1f})" text-anchor="middle" font-size="13">Frequency (%)</text>',
            f'<text x="{(left + right) / 2:.1f}" y="{y0 + ph - 5}" text-anchor="middle" font-size="12">{html.escape(label)}</text>',
        ]
        mapx = lambda x: left + (x - lo) / (hi - lo or 1) * (right - left)
        mapy = lambda y: bottom - y / ymax * (bottom - pt)
        for phase in PHASES:
            fd = s[s.Phase == phase].sort_values("Bin_Number")
            md = means[(means.Parameter == p) & (means.Phase == phase)]
            if fd.empty or md.empty:
                continue
            mean_value = float(md.iloc[0].Mean_Value)
            ys = fd.Frequency.to_numpy(float) / float(md.iloc[0].Event_Count) * 100
            pts = " ".join(f"{mapx(x):.1f},{mapy(y):.1f}" for x, y in zip(xv, ys))
            xm = mapx(mean_value)
            out += [
                f'<polyline points="{pts}" fill="none" stroke="{COLORS[phase]}" stroke-width="2.5"/>',
                *[
                    f'<circle cx="{mapx(x):.1f}" cy="{mapy(y):.1f}" r="3" fill="{COLORS[phase]}"/>'
                    for x, y in zip(xv, ys)
                ],
                f'<line x1="{xm:.1f}" y1="{pt}" x2="{xm:.1f}" y2="{bottom}" stroke="{COLORS[phase]}" stroke-width="2" stroke-dasharray="8,6"/>',
            ]
            # 在虚线顶部显示均值数值，确保靠近左边界的均值也清晰可见。
            out.append(
                f'<text x="{xm + 4:.1f}" y="{pt + 14}" font-size="10" fill="{COLORS[phase]}">mean={mean_value:.4g}</text>'
            )
    out.append("</svg>")
    outpath.write_text("\n".join(out), encoding="utf-8")


def make_png(freq, means, outpath, font_size=50):
    """用 Pillow 直接栅格化绘图，输出 PNG，不依赖 matplotlib。"""
    # 3 行 × 2 列布局：六个参数按三行两列排列。
    W, H, mx, top, gx, gy = 1500, 1500, 80, 130, 70, 75
    pw, ph = (W - 2 * mx - gx) / 2, 390
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    bold_path = Path(r"C:\Windows\Fonts\arialbd.ttf")
    font = ImageFont.truetype(font_path, font_size) if font_path.exists() else ImageFont.load_default()
    bold = ImageFont.truetype(bold_path, font_size + 2) if bold_path.exists() else font
    dr.text(
        (W / 2, 20),
        "Combined SOLO + PSP: parameter distributions by phase",
        fill="#222",
        anchor="ma",
        font=bold,
    )
    for i, phase in enumerate(PHASES):
        x = 650 + i * 190
        dr.line((x, 65, x + 30, 65), fill=COLORS[phase], width=3)
        display_name = PHASE_DISPLAY.get(phase, phase)   # 新增
        dr.text((x + 38, 58), display_name, fill="#222", font=font)
    for idx, (p, label) in enumerate(LABELS.items()):
        row, col = divmod(idx, 2)
        x0 = mx + col * (pw + gx)
        y0 = top + row * (ph + gy)
        left, right, bottom, pt = x0 + 60, x0 + pw - 15, y0 + ph - 45, y0 + 38
        s = freq[freq.Parameter == p]
        b = s.drop_duplicates("Bin_Number").sort_values("Bin_Number")
        xv = (b.Bin_Lower.to_numpy(float) + b.Bin_Upper.to_numpy(float)) / 2
        lo, hi = float(b.Bin_Lower.min()), float(b.Bin_Upper.max())
        ymax = max(
            10.0, float((s.Frequency / means.Event_Count.max() * 100).max()) * 1.12
        )
        dr.text(
            ((left + right) / 2, y0 + 5), label, fill="#222", anchor="ma", font=bold
        )
        dr.line((left, pt, left, bottom), fill="#333")
        dr.line((left, bottom, right, bottom), fill="#333")
        # 根据范围自动取 1/2/5×10^n 的横坐标步长（优先整十、整百、整千）。
        raw_step = (hi - lo) / 5 if hi > lo else 1
        power = 10 ** math.floor(math.log10(raw_step))
        normalized = raw_step / power
        step = (
            1
            if normalized <= 1
            else 2
            if normalized <= 2
            else 5
            if normalized <= 5
            else 10
        ) * power
        first = math.ceil(lo / step) * step
        xticks = []
        v = first
        while v <= hi + step * 1e-9:
            xticks.append(v)
            v += step
        for v in xticks:
            xx = left + (v - lo) / (hi - lo or 1) * (right - left)
            dr.line((xx, bottom, xx, bottom + 6), fill="#333")
            text = f"{v:.0f}" if abs(v) >= 1 else f"{v:.2g}"
            dr.text((xx, bottom + 10), text, fill="#333", anchor="ma", font=font)
        for t in range(6):
            yy = bottom - t * (bottom - pt) / 5
            dr.line((left, yy, right, yy), fill="#dddddd")
            dr.text(
                (left - 8, yy),
                f"{t * ymax / 5:.0f}",
                fill="#333",
                anchor="rm",
                font=font,
            )
        mapx = lambda x: left + (x - lo) / (hi - lo or 1) * (right - left)
        mapy = lambda y: bottom - y / ymax * (bottom - pt)
        for phase_index, phase in enumerate(PHASES):
            fd = s[s.Phase == phase].sort_values("Bin_Number")
            md = means[(means.Parameter == p) & (means.Phase == phase)]
            if fd.empty or md.empty:
                continue
            ys = fd.Frequency.to_numpy(float) / float(md.iloc[0].Event_Count) * 100
            pts = [(mapx(x), mapy(y)) for x, y in zip(xv, ys)]
            dr.line(pts, fill=COLORS[phase], width=3)
            for x, y in pts:
                dr.ellipse((x - 3, y - 3, x + 3, y + 3), fill=COLORS[phase])
            xm = mapx(float(md.iloc[0].Mean_Value))
            yy = pt
            while yy < bottom:
                dr.line((xm, yy, xm, min(yy + 8, bottom)), fill=COLORS[phase], width=2)
                yy += 14
            # 标出均值数值；当两个均值非常接近（如第一阶段热压）时，文字仍分别显示。
            dr.text(
                (xm + 5, pt + 6 + phase_index * 14),
                f"mean={float(md.iloc[0].Mean_Value):.4g}",
                fill=COLORS[phase],
                font=font,
            )
    im.save(outpath, dpi=(300, 300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", type=Path, default=SOLO_DEFAULT)
    ap.add_argument("--psp", type=Path, default=PSP_DEFAULT)
    ap.add_argument("-o", "--output-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("-n", "--bins", type=int, default=10)
    ap.add_argument("--font-size", type=int, default=25, help="图中文字字号，默认 18")
    a = ap.parse_args()
    if a.bins < 1:
        ap.error("--bins 必须大于等于 1")
    if a.font_size < 6:
        ap.error("--font-size 必须大于等于 6")
    d = pd.concat([read_solo(a.solo), read_psp(a.psp)], ignore_index=True)
    d[PARAMS] = d[PARAMS].apply(pd.to_numeric, errors="coerce")
    f = make_frequency(d, a.bins)
    e = (
        d.groupby("Unified_Phase")
        .Event_ID.nunique()
        .rename("Event_Count")
        .rename_axis("Phase")
        .reset_index()
    )
    mw = (
        d.groupby("Unified_Phase")[PARAMS]
        .mean()
        .reset_index()
        .rename(columns={"Unified_Phase": "Phase"})
        .merge(e, on="Phase")
    )
    ml = mw.melt(
        id_vars=["Phase", "Event_Count"],
        value_vars=PARAMS,
        var_name="Parameter",
        value_name="Mean_Value",
    )
    sc = (
        d.groupby(["Source", "Unified_Phase"])
        .Event_ID.nunique()
        .rename("Event_Count")
        .reset_index()
    )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    xlsx = a.output_dir / "SOLO与PSP合并_分阶段参数统计.xlsx"
    png = a.output_dir / "SOLO与PSP_六参数分阶段频数折线图.png"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        f.to_excel(w, sheet_name="频数统计", index=False)
        mw.to_excel(w, sheet_name="阶段均值", index=False)
        ml.to_excel(w, sheet_name="同阶段参数均值", index=False)
        sc.to_excel(w, sheet_name="各来源事件数", index=False)
        d.to_excel(w, sheet_name="合并后原始数据", index=False)
    make_png(f, ml, png, font_size=a.font_size)
    print(f"Excel：{xlsx.resolve()}")
    print(f"图形：{png.resolve()}")
    print(f"事件数：{d.Event_ID.nunique()}，阶段记录：{len(d)}")


if __name__ == "__main__":
    main()
