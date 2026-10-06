"""
Concrete illustration of the mixed-frequency problem:
VWC at 15-min vs daily meteorological inputs.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "figures"

# Load 15-min VWC
raw = pd.read_csv(BASE / "data/station_data/RM_SM_01_calibrated.csv",
                  parse_dates=["Measurement Time"])
raw = raw.set_index("Measurement Time").sort_index()
vwc_15min = raw["5 cm VWC [m3/m3]"]

# Load daily article data
daily = pd.read_csv(BASE / "soil_moisture_article_data/RAAM_cov/RM01.csv",
                    parse_dates=["date"]).set_index("date")

# Find a nice rainfall event — late Oct 2016 (dry period then rain)
t0, t1 = "2016-10-10", "2016-10-25"

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False,
                         gridspec_kw={"height_ratios": [2, 1, 2]})

# ── Panel A: What you actually observe ──
ax = axes[0]
ax.set_title("What the data actually looks like", fontweight="bold",
             fontsize=12, loc="left")

# 15-min VWC
s = vwc_15min.loc[t0:t1].dropna()
ax.plot(s.index, s.values, color="steelblue", linewidth=0.8,
        label="VWC 5cm — measured every 15 min (96 points/day)")

# Mark the daily values
vwc_daily = daily["VWC5"].loc[t0:t1]
ax.plot(vwc_daily.index, vwc_daily.values, "ko", markersize=6,
        label="VWC5 daily average (1 point/day)")

ax.set_ylabel("VWC [m³/m³]")
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(pd.Timestamp(t0), pd.Timestamp(t1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

# ── Panel B: Daily rainfall — the covariate ──
ax = axes[1]
ax.set_title("Meteorological input: daily rainfall from KNMI", fontweight="bold",
             fontsize=12, loc="left")

rain = daily["rd"].loc[t0:t1]
# Plot as a single bar per day — this is all we know
bars = ax.bar(rain.index, rain.values, width=0.8, color="dodgerblue",
              edgecolor="navy", linewidth=0.5)
ax.set_ylabel("Rainfall [mm/day]")
ax.set_xlim(pd.Timestamp(t0), pd.Timestamp(t1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

# Annotate
for i, (date, val) in enumerate(rain.items()):
    if val > 0:
        ax.text(date, val + 0.2, f"{val:.1f}", ha="center", fontsize=7,
                color="navy")

# ── Panel C: The mismatch problem ──
ax = axes[2]
ax.set_title("The problem: WHEN did it rain within the day?", fontweight="bold",
             fontsize=12, loc="left")

# Zoom to a 3-day window around a rain event
t2, t3 = "2016-10-13", "2016-10-17"
s2 = vwc_15min.loc[t2:t3].dropna()
ax.plot(s2.index, s2.values, color="steelblue", linewidth=1.2,
        label="VWC 5cm (15-min)")

# Show the daily rainfall as a shaded block for each day
for date in pd.date_range(t2, t3, freq="D"):
    date_str = date.strftime("%Y-%m-%d")
    if date in rain.index and rain.loc[date] > 0:
        r = rain.loc[date]
        ax.axvspan(date, date + pd.Timedelta(days=1),
                   alpha=0.15, color="dodgerblue")
        ax.text(date + pd.Timedelta(hours=12), ax.get_ylim()[1] * 0.98,
                f"rd={r:.1f} mm\n(somewhere\nthis day)",
                ha="center", va="top", fontsize=8, color="navy",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

# Draw arrows showing the sharp VWC jump
# Find the biggest jump in this window
diffs = s2.diff()
jump_time = diffs.idxmax()
jump_val = s2.loc[jump_time]
pre_val = s2.loc[:jump_time].iloc[-2]
ax.annotate(f"VWC jumps here\n({jump_time.strftime('%H:%M')})",
            xy=(jump_time, jump_val),
            xytext=(jump_time + pd.Timedelta(hours=8), jump_val + 0.01),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontsize=9, color="red", fontweight="bold")

ax.set_ylabel("VWC [m³/m³]")
ax.set_xlabel("Time")
ax.legend(loc="upper left", fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
ax.tick_params(axis="x", rotation=20)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_mixed_freq_explained.png", bbox_inches="tight", dpi=150)
print(f"Saved {OUT_DIR / 'fig_mixed_freq_explained.png'}")
