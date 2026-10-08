"""Fig 15: organoid size by number of Acceptable votes (incl. borderline 2/5 and 3/5)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

df = pd.read_csv(Path(sys.argv[1])); OUT = Path(sys.argv[2])
five = df[(df.n_acc + df.n_not) == 5].copy()
sns.set_theme(context="paper", style="white")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 18, "axes.titlesize": 17,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "savefig.dpi": 300, "savefig.bbox": "tight"})
S2 = sns.color_palette("Set2")
pal = {0: S2[1], 1: S2[1], 2: "#f6c4a8", 3: "#c3cbe3", 4: S2[2], 5: S2[2]}
fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
for a, d in zip(ax, (17, 24, 30)):
    five["area"] = five[f"area_{d}"] * 100
    sns.boxplot(data=five, x="n_acc", y="area", color="lightgray", showfliers=False, linewidth=1, ax=a, width=.6)
    sns.stripplot(data=five, x="n_acc", y="area", hue="n_acc", palette=pal, size=6, edgecolor="k", linewidth=.6,
                  jitter=.2, legend=False, ax=a)
    a.axvspan(1.5, 3.5, color="0.92", zorder=0)
    a.set_yscale("log"); a.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([2, 3, 4, 6, 8, 12, 20]))
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    rho = spearmanr(five.area, five.n_acc)[0]
    a.set_title(f"Day {d}   (Spearman ρ = {rho:.2f})", fontsize=15)
    a.set_xlabel("Acceptable votes (of 5)"); a.set_ylabel("Organoid area (% of frame)" if d == 17 else "")
    a.set_xticklabels(["0", "1", "2\nborder", "3\nborder", "4", "5"])
fig.text(.5, -0.02, "shaded = borderline 2/5 and 3/5 organoids (no official label; not used in training)",
         ha="center", fontsize=12, color="0.4")
sns.despine(fig=fig); fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(OUT / f"15_size_by_votes_incl_borderline.{e}")
print("saved 15")
