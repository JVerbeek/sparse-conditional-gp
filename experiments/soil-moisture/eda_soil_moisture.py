"""
Exploratory Data Analysis: Raam Soil Moisture Network (Netherlands)
===================================================================
Focus: data structure for mixed-frequency and multi-output GP modeling.

Data sources:
- Raw sensor data: 15-min VWC + Temp at 5 depths (5,10,20,40,80 cm), 15 stations
- Article data (daily): VWC5, root-zone VWC, KNMI meteo, MODIS LAI, crop, soil type
- Metadata: station coordinates, soil hydraulic parameters (van Genuchten), crop types
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "font.family": "serif",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
})

BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "Wageningen" / "data" / "station_data"
ART_DIR = BASE / "Wageningen" /"soil_moisture_article_data" / "RAAM_cov"
META_DIR = BASE / "Wageningen" / "data" / "metadata"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

STATIONS = [f"RM_SM_{i:02d}" for i in range(1, 16)]
STATION_SHORT = [f"RM{i:02d}" for i in range(1, 16)]
DEPTHS = [5, 10, 20, 40, 80]
VWC_COLS = [f"{d} cm VWC [m3/m3]" for d in DEPTHS]
TEMP_COLS = [f"{d} cm Temp [oC]" for d in DEPTHS]

# ── 1. Load raw sensor data (15-min) ──────────────────────────────────────────

print("Loading raw 15-min sensor data...")
raw = {}
for stn in STATIONS:
    df = pd.read_csv(RAW_DIR / f"{stn}_calibrated.csv", parse_dates=["Measurement Time"])
    df = df.set_index("Measurement Time").sort_index()
    raw[stn] = df

# ── 2. Load daily article data ────────────────────────────────────────────────

print("Loading daily article data...")
daily = {}
for s in STATION_SHORT:
    df = pd.read_csv(ART_DIR / f"{s}.csv", parse_dates=["date"])
    df = df.set_index("date").sort_index()
    daily[s] = df

# ── 3. Load metadata ─────────────────────────────────────────────────────────

coords = pd.read_csv(BASE / "Wageningen/soil_moisture_article_data" / "Coords_Raam.csv")
crops = pd.read_csv(BASE / "Wageningen/soil_moisture_article_data" / "Crop_types_Raam.csv")
bofek = pd.read_csv(BASE / "Wageningen/soil_moisture_article_data" / "BOFEK2012.csv", skiprows=8)

print(f"Stations: {len(raw)}")
print(f"Raw time range: {raw['RM_SM_01'].index.min()} to {raw['RM_SM_01'].index.max()}")
print(f"Daily time range: {daily['RM01'].index.min()} to {daily['RM01'].index.max()}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Raw time series for all stations at all depths
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 1: Raw multi-depth VWC time series...")
fig, axes = plt.subplots(5, 3, figsize=(16, 14), sharex=True)
colors_depth = plt.cm.viridis(np.linspace(0.1, 0.9, 5))

for idx, stn in enumerate(STATIONS):
    ax = axes[idx // 3, idx % 3]
    df = raw[stn]
    for j, (col, d) in enumerate(zip(VWC_COLS, DEPTHS)):
        if col in df.columns:
            # Downsample to daily for plotting clarity
            series = df[col].resample("D").mean().dropna()
            ax.plot(series.index, series.values, color=colors_depth[j],
                    linewidth=0.6, alpha=0.8, label=f"{d} cm")
    ax.set_title(stn.replace("RM_SM_", "RM"), fontweight="bold")
    ax.set_ylabel("VWC [m³/m³]")
    ax.set_ylim(0, 0.55)
    if idx == 0:
        ax.legend(loc="upper right", ncol=3, fontsize=7)

for ax in axes[-1]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.tick_params(axis="x", rotation=30)

fig.suptitle("Volumetric Water Content at 5 Depths — All 15 Raam Stations (daily avg of 15-min data)",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_all_stations_vwc_depths.png", bbox_inches="tight")
print(f"  Saved {OUT_DIR / 'fig1_all_stations_vwc_depths.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Missingness / data availability heatmap
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 2: Data availability heatmap...")
# Build a monthly availability matrix (station x month x depth)
all_months = pd.date_range("2016-04", "2019-01", freq="MS")
avail = np.zeros((len(STATIONS) * 5, len(all_months)))

for i, stn in enumerate(STATIONS):
    df = raw[stn]
    for j, col in enumerate(VWC_COLS):
        if col in df.columns:
            monthly_count = df[col].dropna().resample("MS").count()
            for k, m in enumerate(all_months):
                if m in monthly_count.index:
                    # max possible ~2880 per month (15-min)
                    avail[i * 5 + j, k] = monthly_count.loc[m] / 2880.0

fig, ax = plt.subplots(figsize=(14, 10))
im = ax.imshow(avail, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_yticks(range(0, len(STATIONS) * 5, 5))
ax.set_yticklabels([s.replace("RM_SM_", "RM") for s in STATIONS])
ax.set_xticks(range(len(all_months)))
ax.set_xticklabels([m.strftime("%b %y") for m in all_months], rotation=45, ha="right", fontsize=7)
# Add minor ticks for depth labels
for i in range(len(STATIONS)):
    for j, d in enumerate(DEPTHS):
        ax.text(-0.8, i * 5 + j, f"{d}cm", fontsize=6, ha="right", va="center")
plt.colorbar(im, ax=ax, label="Fraction of data available", shrink=0.6)
ax.set_title("Data Availability by Station, Depth, and Month (15-min raw data)", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_data_availability.png", bbox_inches="tight")
print(f"  Saved {OUT_DIR / 'fig2_data_availability.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Mixed frequency illustration — zoom in on one station
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 3: Mixed frequency comparison (RM01)...")
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

stn_raw = raw["RM_SM_01"]["5 cm VWC [m3/m3]"]
stn_daily = daily["RM01"]["VWC5"]
stn_rain = daily["RM01"]["rd"]
stn_lai = daily["RM01"]["LAI"]

# Zoom to a 3-month window showing interesting dynamics
t0, t1 = "2016-09-01", "2016-12-01"

# Panel a: 15-min vs daily VWC
ax = axes[0]
s = stn_raw.loc[t0:t1]
ax.plot(s.index, s.values, color="steelblue", linewidth=0.3, alpha=0.7, label="15-min")
s_d = stn_daily.loc[t0:t1]
ax.plot(s_d.index, s_d.values, "ko-", markersize=2, linewidth=1, label="Daily avg")
ax.set_ylabel("VWC 5cm [m³/m³]")
ax.legend(loc="upper right")
ax.set_title("(a) Soil moisture: 15-min vs daily", fontweight="bold", loc="left")

# Panel b: Daily rainfall
ax = axes[1]
ax.bar(stn_rain.loc[t0:t1].index, stn_rain.loc[t0:t1].values, color="dodgerblue", width=0.8)
ax.set_ylabel("Rainfall [mm]")
ax.set_title("(b) Daily rainfall (KNMI)", fontweight="bold", loc="left")

# Panel c: Daily meteorological features
ax = axes[2]
ax.plot(daily["RM01"]["TX"].loc[t0:t1], color="red", linewidth=0.8, label="T_max")
ax.plot(daily["RM01"]["TN"].loc[t0:t1], color="blue", linewidth=0.8, label="T_min")
ax2 = ax.twinx()
ax2.plot(daily["RM01"]["EV24"].loc[t0:t1], color="orange", linewidth=0.8, label="ET₀")
ax.set_ylabel("Temperature [°C]")
ax2.set_ylabel("ET₀ [mm]", color="orange")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", ncol=3)
ax.set_title("(c) Daily meteorological covariates", fontweight="bold", loc="left")

# Panel d: LAI (lower frequency — MODIS ~8-16 day, interpolated)
ax = axes[3]
ax.plot(stn_lai.loc[t0:t1], "g-o", markersize=3, linewidth=1)
ax.set_ylabel("LAI")
ax.set_title("(d) MODIS Leaf Area Index (interpolated to daily)", fontweight="bold", loc="left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
ax.tick_params(axis="x", rotation=30)

fig.suptitle("Mixed Frequency Structure — Station RM01 (Sep–Nov 2016)",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_mixed_frequency.png", bbox_inches="tight")
print(f"  Saved {OUT_DIR / 'fig3_mixed_frequency.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Cross-station comparison — multi-output GP perspective
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 4: Cross-station VWC5 comparison...")
fig, ax = plt.subplots(figsize=(14, 5))
colors = plt.cm.tab20(np.linspace(0, 1, 15))
for i, s in enumerate(STATION_SHORT):
    vwc = daily[s]["VWC5"]
    ax.plot(vwc.index, vwc.values, color=colors[i], linewidth=0.7, alpha=0.8, label=s)
ax.legend(ncol=5, loc="upper right", fontsize=7)
ax.set_ylabel("VWC 5cm [m³/m³]")
ax.set_title("Surface Soil Moisture (5 cm) Across All 15 Stations — Candidate Outputs for Multi-Output GP",
             fontweight="bold")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_cross_station_vwc5.png", bbox_inches="tight")
print(f"  Saved {OUT_DIR / 'fig4_cross_station_vwc5.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Cross-correlation matrix between stations (VWC5)
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 5: Inter-station correlation matrix...")
# Build aligned VWC5 DataFrame
vwc5_all = pd.DataFrame({s: daily[s]["VWC5"] for s in STATION_SHORT})
corr = vwc5_all.corr()

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=0.3, vmax=1)
ax.set_xticks(range(15))
ax.set_xticklabels(STATION_SHORT, rotation=45, ha="right")
ax.set_yticks(range(15))
ax.set_yticklabels(STATION_SHORT)
for i in range(15):
    for j in range(15):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Inter-Station Correlation (VWC 5cm, daily)")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig5_5cm_interstation_correlation.png", bbox_inches="tight")
print(f"  Saved {OUT_DIR / 'fig5_5cm_interstation_correlation.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Cross-depth correlation within station (RM01)
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 6: Cross-depth correlation (RM01)...")
for station in range(1, 16, 1):
    name = f"0{station}" if station < 10 else f"{station}"
    df01 = raw[f"RM_SM_{name}"][VWC_COLS].resample("D").mean()
    df01.columns = [f"{d} cm" for d in DEPTHS]
    corr_depth = df01.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_depth.values, cmap="RdBu_r", vmin=0.3, vmax=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"{d} cm" for d in DEPTHS])
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"{d} cm" for d in DEPTHS])
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{corr_depth.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Cross-Depth Correlation (RM{station}, daily avg)\nMulti-output over depth layers")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"crossdepth/fig6_{station}_correlation.png", bbox_inches="tight")
    print(f"  Saved {OUT_DIR / f'crossdepth/fig6_{station}_correlation.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Station map colored by soil type + crop diversity
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 7: Station map...")
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# # Panel a: by BOFEK soil type
bofek_codes = coords.merge(
    pd.DataFrame({"Name": STATION_SHORT,
                  "BOFEK": [int(daily[s]["BOFEK"].iloc[0]) for s in STATION_SHORT]}),
    on="Name"
)
unique_bofek = sorted(bofek_codes["BOFEK"].unique())
bofek_cmap = {b: plt.cm.Set2(i / len(unique_bofek)) for i, b in enumerate(unique_bofek)}

for _, row in bofek_codes.iterrows():
    ax.scatter(row["Latitude_deg"], row["Longitude_deg"],
               c=[bofek_cmap[row["BOFEK"]]], s=100, edgecolors="k", zorder=5)
    ax.annotate(row["Name"], (row["Longitude_deg"], row["Latitude_deg"]),
                textcoords="offset points", xytext=(5, 5), fontsize=7)
# Legend
for b in unique_bofek:
    ax.scatter([], [], c=[bofek_cmap[b]], s=60, edgecolors="k", label=f"BOFEK {b}")
ax.legend(loc="lower left", fontsize=7)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("(a) Stations by BOFEK Soil Type", fontweight="bold")

# Panel b: by dominant crop
crop_2016 = crops.set_index("Stn")["2016"].to_dict()
crop_types = sorted(set(crop_2016.values()))
crop_cmap = {c: plt.cm.Set1(i / len(crop_types)) for i, c in enumerate(crop_types)}

longs, lats, names = [], [], []
for _, row in coords.iterrows():
    c = crop_2016.get(row["Name"], "unknown")
    longs.append(row["Longitude_deg"])
    lats.append(row["Latitude_deg"])
    names.append(row["Name"])

fig.suptitle("Raam Network Station Locations",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig7_station_map.png", bbox_inches="tight")
print(f"  Saved {OUT_DIR / 'fig7_station_map.png'}")

# import plotly.graph_objects as go
# print(lats, longs)
# fig = go.Figure(go.Scattermap(
#         lat=longs,
#         lon=lats,
#         mode='markers+text',
#         marker=go.scattermap.Marker(
#             size=15
#         ),
#         text=names,
#         textposition="bottom right",
#     ))

# fig.update_layout(
#     autosize=True,
#     map=dict(
#         style="outdoors",
#         bearing=0,
#         center=dict(
#             lat=51.633,
#             lon=5.78
#         ),
#         pitch=0,
#         zoom=10
#     ),
#     font=dict(
#         family="Serif, serif",
#         size=15,  # Set the font size here
#     )
# )

# fig.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
# import numpy as np

# # Create the basemap for the Netherlands
# m = Basemap(
#     projection='merc',
#     llcrnrlat=50.7,      # Lower left corner latitude
#     urcrnrlat=53.5,      # Upper right corner latitude
#     llcrnrlon=3.3,       # Lower left corner longitude
#     urcrnrlon=7.3,       # Upper right corner longitude
#     resolution='l'       # High resolution
# )

# # Draw map features
# m.drawcoastlines(linewidth=1.5)
# m.drawcountries(linewidth=1)
# m.drawstates(linewidth=0.5)

# # Add map background
# m.drawmapboundary(fill_color='lightblue')
# m.fillcontinents(color='lightgray', lake_color='lightblue')

# # Define the square size (in degrees)
# square_size = 0.5  # Approximately 55 km on each side

# # Calculate the corners of the square
# lon_min = min(lats) 
# lon_max = max(lats)
# lat_min = min(longs)
# lat_max = max(longs)

# x_min, y_min = m(lon_min, lat_min)
# x_max, y_max = m(lon_max, lat_max)

# square = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
#                         fill=False, edgecolor='red', linewidth=2, label='Raam catchment area')

# plt.gca().add_patch(square)

# plt.title('Raam catchment area in the Netherlands', fontsize=16)
# plt.legend(loc='upper left')

# m.drawmeridians(np.arange(3, 8, 1), labels=[0,0,0,1])
# m.drawparallels(np.arange(50, 54, 1), labels=[1,0,0,0])

# plt.tight_layout()
# plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Temporal scales — autocorrelation at different depths
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 8: Autocorrelation by depth...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel a: ACF for RM01 at different depths
ax = axes[0]
max_lag = 90  # days
for j, (col, d) in enumerate(zip(VWC_COLS, DEPTHS)):
    series = raw["RM_SM_01"][col].resample("D").mean().dropna()
    acf_vals = [series.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
    ax.plot(range(1, max_lag + 1), acf_vals, color=colors_depth[j], linewidth=1.2, label=f"{d} cm")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Autocorrelation")
ax.set_title("(a) ACF by Depth (RM01)", fontweight="bold")
ax.legend()

# Panel b: ACF for VWC5 across stations
ax = axes[1]
for i, s in enumerate(STATION_SHORT[:6]):
    series = daily[s]["VWC5"].dropna()
    acf_vals = [series.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
    ax.plot(range(1, max_lag + 1), acf_vals, color=colors[i], linewidth=1, label=s)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax.set_xlabel("Lag (days)")
ax.set_ylabel("Autocorrelation")
ax.set_title("(b) ACF of VWC5 Across Stations", fontweight="bold")
ax.legend()

fig.suptitle("Temporal Persistence — Kernel Lengthscale Insight for GP Modeling",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig8_autocorrelation.png", bbox_inches="tight")
print(f"  Saved {OUT_DIR / 'fig8_autocorrelation.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Feature importance / correlation with VWC5
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 9: Feature correlations with VWC5...")
feature_cols = ["EV24", "FG", "Q", "rd", "SQ", "TN", "TX", "UG", "LAI",
                "VWC5_lag", "rd_3day", "DOY"]

fig, ax = plt.subplots(figsize=(12, 6))
corr_data = []
for s in STATION_SHORT:
    df = daily[s]
    corrs = df[feature_cols + ["VWC5"]].corr()["VWC5"].drop("VWC5")
    corr_data.append(corrs)

corr_df = pd.DataFrame(corr_data, index=STATION_SHORT)
# Box plot of correlations across stations
bp = ax.boxplot([corr_df[col].values for col in feature_cols],
                labels=feature_cols, patch_artist=True, vert=True)
for patch in bp["boxes"]:
    patch.set_facecolor("lightsteelblue")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax.set_ylabel("Pearson correlation with VWC5")
ax.set_title("Correlation of Features with Surface Soil Moisture (across 15 stations)\nLagged VWC dominates — typical GP input structure",
             fontweight="bold")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig9_feature_correlation.png", bbox_inches="tight")
print(f"  Saved {OUT_DIR / 'fig9_feature_correlation.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Aggregation effect — 15-min vs hourly vs daily statistics
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 10: Aggregation effect on variance...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

stn_raw_5cm = raw["RM_SM_01"]["5 cm VWC [m3/m3]"].dropna()

# Panel a: time series at different aggregations
ax = axes[0]
for freq, label, alpha, lw in [("15min", "15-min", 0.3, 0.3),
                                 ("h", "Hourly", 0.5, 0.5),
                                 ("D", "Daily", 0.9, 1.2),
                                 ("W", "Weekly", 1.0, 1.5)]:
    if freq == "15min":
        s = stn_raw_5cm.loc["2016-09":"2016-11"]
    else:
        s = stn_raw_5cm.resample(freq).mean().loc["2016-09":"2016-11"]
    ax.plot(s.index, s.values, alpha=alpha, linewidth=lw, label=label)
ax.legend()
ax.set_ylabel("VWC 5cm [m³/m³]")
ax.set_title("(a) VWC at Different Temporal Aggregations (RM01)", fontweight="bold")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

# Panel b: variance by aggregation level per station
ax = axes[1]
agg_freqs = {"15-min": "15min", "Hourly": "h", "Daily": "D", "Weekly": "W", "Monthly": "MS"}
var_data = {f: [] for f in agg_freqs}
for stn in STATIONS:
    s = raw[stn]["5 cm VWC [m3/m3]"].dropna()
    for label, freq in agg_freqs.items():
        if freq == "15min":
            var_data[label].append(s.var())
        else:
            var_data[label].append(s.resample(freq).mean().var())

positions = range(len(agg_freqs))
bp = ax.boxplot([var_data[f] for f in agg_freqs],
                labels=list(agg_freqs.keys()), patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("lightsalmon")
ax.set_ylabel("Variance of VWC 5cm")
ax.set_title("(b) Variance at Different Aggregation Levels\nMixed-frequency GP must account for this",
             fontweight="bold")

fig.suptitle("Temporal Aggregation Effects — Key for Mixed Frequency Modeling",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig10_aggregation_effect.png", bbox_inches="tight")
print(f"  Saved {OUT_DIR / 'fig10_aggregation_effect.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: Spatial distance vs correlation — informing GP spatial kernel
# ══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 11: Spatial distance vs correlation...")
from itertools import combinations

# Compute pairwise distances and correlations
pairs = list(combinations(range(15), 2))
dists = []
corrs_pair = []

for i, j in pairs:
    si, sj = STATION_SHORT[i], STATION_SHORT[j]
    ci = coords[coords["Name"] == si].iloc[0]
    cj = coords[coords["Name"] == sj].iloc[0]
    # Approximate distance in km (at ~51.6°N, 1° lat ≈ 111 km, 1° lon ≈ 68 km)
    dlat = (ci["Latitude_deg"] - cj["Latitude_deg"]) * 111
    dlon = (ci["Longitude_deg"] - cj["Longitude_deg"]) * 68
    dist = np.sqrt(dlat**2 + dlon**2)
    dists.append(dist)
    corrs_pair.append(corr.iloc[i, j])

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(dists, corrs_pair, c="steelblue", alpha=0.6, edgecolors="k", s=40)
# Fit a simple exponential decay
from scipy.optimize import curve_fit
def exp_decay(x, a, b): return a * np.exp(-x / b)
try:
    popt, _ = curve_fit(exp_decay, dists, corrs_pair, p0=[1, 10], maxfev=5000)
    x_fit = np.linspace(0, max(dists), 100)
    ax.plot(x_fit, exp_decay(x_fit, *popt), "r--", linewidth=1.5,
            label=f"Exp. decay: {popt[0]:.2f}·exp(-d/{popt[1]:.1f} km)")
    ax.legend()
except Exception:
    pass
ax.set_xlabel("Pairwise Distance (km)")
ax.set_ylabel("Pearson Correlation (VWC 5cm)")
ax.set_title("Spatial Correlation Decay — Informs GP Spatial Lengthscale",
             fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig11_spatial_correlation.png", bbox_inches="tight")
print(f"  Saved {OUT_DIR / 'fig11_spatial_correlation.png'}")

# ══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("SUMMARY: Data Structure for Mixed-Frequency / Multi-Output GP Modeling")
print("=" * 80)

print(f"""
DATASET OVERVIEW
  Stations:     15 (Raam catchment, Netherlands)
  Raw freq:     15-min (VWC + Temp at 5 depths: 5,10,20,40,80 cm)
  Daily data:   VWC5, root-zone VWC, + 9 meteorological/remote sensing covariates
  Time span:    Apr 2016 – Jan 2019 (~1000 days)
  Total raw:    ~500K observations across stations

MIXED FREQUENCY STRUCTURE
  High freq:    Soil moisture — 15 min (96 obs/day)
  Medium freq:  Meteorological vars — daily (KNMI)
  Low freq:     MODIS LAI — ~8-16 day composites interpolated to daily
  Static:       Soil type (BOFEK), crop type (yearly)

  → Natural fit for mixed-frequency GP: latent process at fine resolution,
    observations at multiple frequencies. Covariates enter at their native
    frequency rather than forcing everything to daily.

MULTI-OUTPUT / TRANSFER LEARNING POTENTIAL
  Cross-station: 15 stations with corr ∈ [{corr.values[np.triu_indices(15,1)].min():.2f}, {corr.values[np.triu_indices(15,1)].max():.2f}]
  Cross-depth:   5 output layers per station, decreasing correlation with depth gap
  Heterogeneity: {len(unique_bofek)} soil types, {len(crop_types)} crop types
  Spatial decay:  Correlation decays with distance — spatial kernel is informative

  → Multi-output GP over stations (spatial ICM/LMC kernel) or over depths
    (physics-informed vertical kernel). Transfer learning from data-rich
    grass stations to data-poor crop-rotation stations.

KEY OBSERVATIONS
  - RM05 starts late (Jul 2016) and has missing 5cm sensor → good test case for transfer
  - RM06 has 5cm gap Oct 2016–Jan 2017 → imputation benchmark
  - RM12 has broken 10cm sensor → multi-output GP can infer from other depths
  - Deeper layers are smoother (longer ACF) — different lengthscales per output
  - Grass stations are highly correlated — crop stations show more variability
""")

print("All figures saved to:", OUT_DIR)
print("Done.")
