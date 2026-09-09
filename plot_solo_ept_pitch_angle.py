"""Plot Solar Orbiter/EPT electron pitch-angle distributions for an event list.

The event list must contain the columns ``Event_Index``, ``Mark1`` and ``Mark2``.
For every event, the script uses solo-epd-loader 0.4.4 to load EPT level-3 data,
then creates a time--pitch-angle scatter plot.  The colour is the mean corrected
electron differential flux over the requested EPT electron energy channels.

Example (PowerShell)
--------------------
python .\plot_solo_ept_pitch_angle.py `
  --event-list "E:\ZhaoWenjin_Reserch\pileup\solo列表.txt" `
  --data-root "I:\stdata\solo" `
  --output-dir .\solo_pitch_angle_figures `
  --events 5 6 --channel-start 0 --channel-stop 3

``data-root`` is the *base* directory expected by solo-epd-loader, i.e. it
contains ``l3\epd\ept\...``.  If your archive has a different layout (such as
``SOLO_L3_EPD-EPT-1MIN\2025``), use a small staging folder/symlink layout, or
run with ``--autodownload`` to have the loader create its standard layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from solo_epd_loader import epd_load


# EPT level-3 names used by solo-epd-loader.  S=sun, A=anti-sun,
# N=north, D=south (the naming follows the L3 CDF product).
VIEWINGS = ("A", "S", "N", "D")


def read_event_list(filename: Path) -> pd.DataFrame:
    """Read the whitespace-delimited list and parse its two selected time limits."""
    events = pd.read_csv(filename, sep=r"\s{2,}", comment="#", header=None)
    # The first commented line is a header, so assign its known column positions.
    if events.shape[1] < 6:
        raise ValueError("事件列表至少应含 Event_Index、…、Mark1、Mark2 六列。")
    events = events.rename(columns={0: "Event_Index", 4: "Mark1", 5: "Mark2"})
    events["Mark1"] = pd.to_datetime(events["Mark1"])
    events["Mark2"] = pd.to_datetime(events["Mark2"])
    invalid = events["Mark2"] <= events["Mark1"]
    if invalid.any():
        raise ValueError(f"以下事件的 Mark2 不晚于 Mark1: {events.loc[invalid, 'Event_Index'].tolist()}")
    return events


def load_l3(start: pd.Timestamp, end: pd.Timestamp, data_root: Path, autodownload: bool):
    """Load whole UTC days covering [start, end] with solo-epd-loader 0.4.4."""
    # L3 files are daily.  Loading complete boundary days avoids losing an event
    # that starts/ends mid-day, then the dataframe is clipped below.
    return epd_load(
        sensor="ept",
        level="l3",
        startdate=start.to_pydatetime(),
        enddate=end.to_pydatetime(),
        path=str(data_root),
        autodownload=autodownload,
        pos_timestamp="start",
    )


def mean_electron_flux(df: pd.DataFrame, viewing: str, first: int, last: int) -> pd.Series:
    """Return positive mean corrected flux for one telescope and energy range."""
    prefix = f"Electron_Corrected_Flux_{viewing}_"
    requested = [f"{prefix}{channel}" for channel in range(first, last + 1)]
    missing = [name for name in requested if name not in df.columns]
    if missing:
        raise KeyError(
            f"找不到电子通量列 {missing}。可用列示例: "
            f"{[c for c in df.columns if c.startswith(prefix)][:4]}"
        )
    # Corrected flux can legitimately be negative.  It cannot be plotted with a
    # logarithmic colour scale, hence those samples are masked rather than altered.
    flux = df[requested].mean(axis=1).where(lambda x: x > 0)
    return flux


def plot_event(df: pd.DataFrame, event_id: int, start: pd.Timestamp, end: pd.Timestamp,
               channel_start: int, channel_stop: int, output: Path) -> None:
    """Make one time--pitch-angle plot from the interval-clipped L3 dataframe."""
    data = df.loc[(df.index >= start) & (df.index <= end)].copy()
    if data.empty:
        raise RuntimeError("所选 Mark1--Mark2 时段没有可用的 L3 EPT 数据。")

    times, angles, fluxes = [], [], []
    for viewing in VIEWINGS:
        angle_name = f"Pitch_Angle_{viewing}"
        if angle_name not in data.columns:
            raise KeyError(f"L3 数据中没有 {angle_name}。")
        flux = mean_electron_flux(data, viewing, channel_start, channel_stop)
        good = data[angle_name].notna() & flux.notna() & data[angle_name].between(0, 180)
        times.append(data.index[good].to_numpy())
        angles.append(data.loc[good, angle_name].to_numpy(float))
        fluxes.append(flux.loc[good].to_numpy(float))

    t = np.concatenate(times)
    pitch = np.concatenate(angles)
    intensity = np.concatenate(fluxes)
    if not len(intensity):
        raise RuntimeError("所有通量均为非正值或缺测，无法使用对数颜色标度作图。")

    # Robust limits prevent a few exceptional points from washing out the PAD.
    vmin, vmax = np.nanpercentile(intensity, [2, 98])
    vmin = max(vmin, np.finfo(float).tiny)
    vmax = max(vmax, vmin * 1.01)

    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    points = ax.scatter(
        t, pitch, c=intensity, s=20, marker="s", cmap="turbo",
        norm=LogNorm(vmin=vmin, vmax=vmax), linewidths=0, rasterized=True,
    )
    ax.set(xlim=(start, end), ylim=(0, 180), xlabel="Time (UTC)", ylabel="Pitch angle (deg)")
    ax.set_yticks(np.arange(0, 181, 30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.grid(True, color="0.85", linewidth=0.6)
    ax.set_title(f"Solar Orbiter / EPD-EPT electrons — event {event_id}"
                 f"\nchannels {channel_start}–{channel_stop}; Mark1–Mark2")
    cbar = fig.colorbar(points, ax=ax, pad=0.015)
    cbar.set_label(r"Mean corrected electron flux [particles/(s cm$^2$ sr MeV)]")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--event-list", type=Path, required=True, help="含 Mark1、Mark2 的列表 txt")
    parser.add_argument("--data-root", type=Path, required=True, help="solo-epd-loader 数据根目录")
    parser.add_argument("--output-dir", type=Path, default=Path("solo_pitch_angle_figures"))
    parser.add_argument("--events", type=int, nargs="+", help="仅绘制指定 Event_Index；默认全部")
    parser.add_argument("--channel-start", type=int, default=0, help="电子能道起始编号（含）")
    parser.add_argument("--channel-stop", type=int, default=3, help="电子能道终止编号（含），EPT L3 为 0–15")
    parser.add_argument("--autodownload", action="store_true", help="缺失 L3 文件时由 loader 自动下载")
    return parser.parse_args()


def main() -> None:
    # 临时调试用（仅限测试）
    args = argparse.Namespace(
        event_list=Path("E:/ZhaoWenjin_Reserch/pileup/solo列表.txt"),
        data_root=Path(r"I:\stdata\solo\epd\SOLO_L3_EPD-EPT-1MIN"),
        output_dir=Path("./solo_pitch_angle_figures"),
        events=[11,2,3,4,5, 6],
        channel_start=0,
        channel_stop=15,
        autodownload=True
    )
    if not (0 <= args.channel_start <= args.channel_stop <= 15):
        raise ValueError("EPT L3 电子能道编号必须在 0–15，且 start <= stop。")
    events = read_event_list(args.event_list)
    if args.events:
        events = events[events["Event_Index"].isin(args.events)]
        if events.empty:
            raise ValueError("--events 指定的 Event_Index 不在列表中。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in events.iterrows():
        event_id = int(row["Event_Index"])
        start, end = row["Mark1"], row["Mark2"]
        print(f"Event {event_id}: {start} -- {end}")
        df, _df_rtn, _df_hci, _energies, _metadata = load_l3(
            start, end, args.data_root, args.autodownload
        )
        output = args.output_dir / f"solo_ept_pitch_angle_event_{event_id:03d}.png"
        plot_event(df, event_id, start, end, args.channel_start, args.channel_stop, output)
        print(f"  saved: {output}")


if __name__ == "__main__":
    main()
