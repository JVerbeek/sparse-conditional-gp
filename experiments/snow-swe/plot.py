"""Two plots of data/nearby.npz: one winter, and all years.
    python plot.py         # one winter = water year 2019
    python plot.py 2016    # choose the winter
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
d = np.load(HERE / "data" / "nearby.npz")
WY = int(sys.argv[1]) if len(sys.argv) > 1 else 2019

labels = [str(l).split(" pillow")[0] for l in d["source_labels"]]
sources = [(d[f"source{i}_t"], d[f"source{i}_y"]) for i in range(len(labels))]
tt, ty = d["target_t"], d["target_y"]
tlabel = str(d["target_label"]).replace(" course", "")
COLORS = ["C0", "C1", "C2"]


def draw(ax, lo, hi, lw, ms):
    for (t, y), lab, c in zip(sources, labels, COLORS):
        m = (t >= lo) & (t <= hi)
        ax.plot(t[m], y[m], lw=lw, color=c, label=f"{lab} pillow (daily)")
    m = (tt >= lo) & (tt <= hi)
    ax.plot(tt[m], ty[m], "o", ms=ms, color="C3", mec="white", mew=0.8, zorder=5,
            label=f"{tlabel} snow course (monthly)")
    ax.set_ylabel("SWE [in]")
    leg = ax.legend(frameon=False, fontsize=9)
    for h in leg.get_lines():
        h.set_linewidth(1.5); h.set_markersize(7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


# one winter
fig, ax = plt.subplots(figsize=(11, 4))
draw(ax, WY - 1 + 10 / 12, WY + 6 / 12, lw=1.3, ms=9)
plt.tight_layout(); plt.savefig(FIG / f"simple_winter_{WY}.png", dpi=130); plt.close()

# all years with pillow data
lo = min(t.min() for t, _ in sources)
fig, ax = plt.subplots(figsize=(15, 4))
draw(ax, lo, 2027, lw=0.5, ms=3)
ax.set_title(f"{int(lo)}-2026")
plt.tight_layout(); plt.savefig(FIG / "simple_all_years.png", dpi=130); plt.close()
print("wrote", FIG / f"simple_winter_{WY}.png", "and", FIG / "simple_all_years.png")
