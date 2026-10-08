"""bbox sharpness figures (same style as make_plots.py)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
d = pd.read_csv(sys.argv[1]); OUT = Path(sys.argv[2])
sns.set_theme(context="paper", style="white")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 18, "axes.titlesize": 18,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 13, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
S2 = sns.color_palette("Set2"); ACC, NOT, GREY = S2[2], S2[1], "#4d4d4d"
DAYS = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30]; DL = ["3", "6", "8", "10", "13", "15", "17", "20.5", "24", "28", "30"]
d["label"] = np.where(d.true_label == 1, "Acceptable", "Not Acceptable"); d["area_pct"] = d.area_frac * 100


def save(fig, name):
    sns.despine(fig=fig); fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{e}")
    plt.close(fig); print("saved", name)


def logx(a):
    a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([2, 3, 4, 6, 10, 20]))
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def dayax(a):
    a.set_xticks(DAYS); a.set_xticklabels(DL); a.tick_params(axis="x", labelsize=13); a.set_xlabel("Day")


# 10: bbox model prediction still tracks original size
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for a, day in zip(ax, (24, 30)):
    g = d[d.day == day]
    for lab, c in (("Acceptable", ACC), ("Not Acceptable", NOT)):
        h = g[g.label == lab]
        a.scatter(h.area_pct, h.p_orig, s=40, color=c, edgecolor="k", linewidth=.5, alpha=.85, label=lab)
    a.set_xscale("log"); logx(a)
    a.set_title(f"Day {day}   (Spearman ρ = {spearmanr(g.area_pct, g.p_orig)[0]:.2f})", fontsize=15)
    a.set_xlabel("Original organoid area (% of frame)"); a.axhline(.5, color="0.6", ls="--", lw=1, zorder=0)
ax[0].set_ylabel("bbox model P(Acceptable)"); ax[0].legend(loc="upper left")
save(fig, "10_bbox_prediction_vs_original_size")

# 11: cropping creates a size -> sharpness link
rows = []
for day, g in d.groupby("day"):
    rows.append({"day": day, "bbox": spearmanr(g.sharp_bbox, g.area_frac, nan_policy="omit")[0],
                 "orig": spearmanr(g.sharp_orig, g.area_frac, nan_policy="omit")[0]})
r = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
ax[0].plot(r.day, r.orig, "o-", color=GREY, lw=2, ms=7, mec="k", mew=.6, label="Normal input (no crop)")
ax[0].plot(r.day, r.bbox, "o-", color=S2[3], lw=2.2, ms=7, mec="k", mew=.6, label="bbox-cropped input")
ax[0].axhline(0, color="0.6", ls="--", lw=1); dayax(ax[0])
ax[0].set_ylabel("Spearman ρ\n(input sharpness vs size)"); ax[0].legend(loc="lower left")
g = d[d.day == 24]
for lab, c in (("Acceptable", ACC), ("Not Acceptable", NOT)):
    h = g[g.label == lab]
    ax[1].scatter(h.area_pct, h.sharp_bbox * 1e3, s=40, color=c, edgecolor="k", linewidth=.5, alpha=.85, label=lab)
ax[1].set_xscale("log"); logx(ax[1])
ax[1].set_xlabel("Original organoid area (% of frame)"); ax[1].set_ylabel("bbox input sharpness\n(Laplacian var. ×10³)")
ax[1].set_title("Day 24, bbox inputs", fontsize=15); ax[1].legend(loc="upper left")
save(fig, "11_bbox_sharpness_vs_size")

# 12: degrade big organoids to a small organoid's resolution -> bbox model P drops
big = d[d.degrade_factor < 1].assign(dp=lambda x: x.p_degraded - x.p_orig)
s = big.groupby("day").dp
fig, ax = plt.subplots(figsize=(9.5, 5))
ax.fill_between(s.median().index, s.quantile(.25), s.quantile(.75), color=NOT, alpha=.25, lw=0)
ax.plot(s.median().index, s.median(), "o-", color=NOT, lw=2.2, ms=7, mec="k", mew=.6,
        label="Median Δ (IQR shaded)")
for day, pct in (big.assign(down=big.dp < 0).groupby("day").down.mean() * 100).items():
    if day >= 17:
        ax.text(day, s.quantile(.75)[day] + .012, f"{pct:.0f}%↓", ha="center", fontsize=11, color="0.3")
ax.axhline(0, color="0.6", ls="--", lw=1); dayax(ax)
ax.set_ylabel("Δ bbox P(Acceptable)\n(small-resolution − real)"); ax.legend(loc="lower left")
ax.set_title("Larger organoids re-rendered at a small organoid's resolution (same shape)", fontsize=14)
save(fig, "12_bbox_resolution_test")
