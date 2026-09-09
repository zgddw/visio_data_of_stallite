"""计算 PSP 与 Solar Orbiter 事件期间的日心距离。

事件区间使用事件表中的 Mark1--Mark2（与 Duration_h 对应），CDF 使用
一小时分辨率的 RAD_AU 变量。结果写入 Excel 的三个工作表，并另存 CSV。

运行示例（PowerShell）：
  & 'C:\\path\\to\\python.exe' .\\event_heliocentric_distance.py

依赖：pandas、openpyxl，以及 NASA CDF 工具包中的 cdfdump.exe。
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PSP_SHOCK = Path(r"E:\ZhaoWenjin_Reserch\pileup\有激波psp.txt")
DEFAULT_PSP_NO_SHOCK = Path(r"E:\ZhaoWenjin_Reserch\pileup\无激波psp.txt")
DEFAULT_SOLO = Path(r"E:\ZhaoWenjin_Reserch\pileup\solo有阈值.xlsx")
DEFAULT_PSP_CDF = Path(r"I:\stdata\psp\position\psp_helio1hr_position\psp_helio1hr_position_20180813_v01.cdf")
DEFAULT_SOLO_CDF = Path(r"I:\stdata\solo\solo_helio1hr_position_20200211_v01.cdf")


def read_rad_au(cdf_path: Path, cdfdump: str) -> pd.DataFrame:
    """用 NASA CDF 工具读取 Epoch 与 RAD_AU，避免额外安装 cdflib。"""
    if not cdf_path.is_file():
        raise FileNotFoundError(f"找不到 CDF 文件：{cdf_path}")
    command = [cdfdump, "-noheader", "-noformat", "-dump", "data", "-vars", "Epoch,RAD_AU", str(cdf_path)]
    completed = subprocess.run(command, check=True, capture_output=True)
    text = completed.stdout.decode("utf-8", errors="replace")
    epoch_block, rad_block = text.split("RAD_AU (No:", maxsplit=1)
    record_pattern = re.compile(r"Record #\s*\d+:\s*(.+)")
    epoch_text = record_pattern.findall(epoch_block)
    distance_text = record_pattern.findall(rad_block)
    if len(epoch_text) != len(distance_text) or not epoch_text:
        raise RuntimeError(f"无法从 {cdf_path.name} 配对 Epoch 与 RAD_AU（{len(epoch_text)} vs {len(distance_text)} 条）。")
    times = [datetime.strptime(value.strip(), "%d-%b-%Y %H:%M:%S.%f") for value in epoch_text]
    distances = pd.to_numeric(pd.Series(distance_text), errors="coerce")
    result = pd.DataFrame({"sample_time": times, "distance_au": distances})
    return result.loc[result["distance_au"].between(0, 1000)].reset_index(drop=True)


def load_psp_events(path: Path, event_type: str) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    table.insert(0, "event_id", [f"PSP-{event_type}-{i:02d}" for i in range(1, len(table) + 1)])
    table.insert(1, "satellite", "PSP")
    table.insert(2, "event_type", event_type)
    return table.rename(columns={"Mark1": "event_start", "Mark2": "event_end"})


def load_solo_events(path: Path) -> pd.DataFrame:
    table = pd.read_excel(path)
    table.insert(0, "event_id", [f"SOLO-threshold-{i:02d}" for i in range(1, len(table) + 1)])
    table.insert(1, "satellite", "Solar Orbiter")
    table.insert(2, "event_type", "threshold")
    return table.rename(columns={"Mark1": "event_start", "Mark2": "event_end"})


def summarize_events(events: pd.DataFrame, position: pd.DataFrame) -> pd.DataFrame:
    distance_series = position.set_index("sample_time")["distance_au"].sort_index()
    rows = []
    for _, event in events.iterrows():
        start, end = pd.Timestamp(event["event_start"]), pd.Timestamp(event["event_end"])
        augmented_index = distance_series.index.union(pd.DatetimeIndex([start, end])).sort_values()
        samples = distance_series.reindex(augmented_index).interpolate(method="time").loc[start:end].dropna()
        original_count = int(position["sample_time"].between(start, end, inclusive="both").sum())
        elapsed_seconds = (samples.index - samples.index[0]).total_seconds().to_numpy()
        mean_distance = np.trapezoid(samples.to_numpy(), x=elapsed_seconds) / elapsed_seconds[-1]
        row = event.to_dict()
        row.update({
            "distance_min_au": samples.min(), "distance_max_au": samples.max(),
            "distance_mean_au": mean_distance,
            "distance_range_au": f"{samples.min():.5f}–{samples.max():.5f}",
            "cdf_sample_count": original_count,
            "calculation": "线性插值至 Mark1/Mark2；梯形积分时间加权平均",
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    for column in ("distance_min_au", "distance_max_au", "distance_mean_au"):
        result[column] = result[column].round(5)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="提取事件期间的日心距离（AU）")
    parser.add_argument("--psp-shock", type=Path, default=DEFAULT_PSP_SHOCK)
    parser.add_argument("--psp-no-shock", type=Path, default=DEFAULT_PSP_NO_SHOCK)
    parser.add_argument("--solo-events", type=Path, default=DEFAULT_SOLO)
    parser.add_argument("--psp-cdf", type=Path, default=DEFAULT_PSP_CDF)
    parser.add_argument("--solo-cdf", type=Path, default=DEFAULT_SOLO_CDF)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cdfdump", default=shutil.which("cdfdump.exe") or "cdfdump.exe")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    psp_position = read_rad_au(args.psp_cdf, args.cdfdump)
    solo_position = read_rad_au(args.solo_cdf, args.cdfdump)
    psp_shock = summarize_events(load_psp_events(args.psp_shock, "shock"), psp_position)
    psp_no_shock = summarize_events(load_psp_events(args.psp_no_shock, "no_shock"), psp_position)
    solo = summarize_events(load_solo_events(args.solo_events), solo_position)
    excel_path = args.output_dir / "事件日心距离汇总.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        psp_shock.to_excel(writer, sheet_name="PSP_有激波", index=False)
        psp_no_shock.to_excel(writer, sheet_name="PSP_无激波", index=False)
        solo.to_excel(writer, sheet_name="Solar_Orbiter", index=False)
    psp_shock.to_csv(args.output_dir / "PSP_有激波_日心距离.csv", index=False, encoding="utf-8-sig")
    psp_no_shock.to_csv(args.output_dir / "PSP_无激波_日心距离.csv", index=False, encoding="utf-8-sig")
    solo.to_csv(args.output_dir / "Solar_Orbiter_日心距离.csv", index=False, encoding="utf-8-sig")
    print(f"已写入：{excel_path.resolve()}")


if __name__ == "__main__":
    main()
