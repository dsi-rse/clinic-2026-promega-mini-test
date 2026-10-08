"""Fig 14: size-matched AUC (Acceptable vs Not Acceptable organoids of the same size)."""
import sys
from pathlib import Path
import pandas as pd, seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pd.read_csv(Path(sys.argv[1]) / "size_matched_auc.csv"); OUT = Path(sys.argv[1])
R = R[R.caliper == 0.10]
sns.set_theme(context="paper", style="white")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 18, "xtick.labelsize": 15,
                     "ytick.labelsize": 15, "legend.fontsize": 13, "legend.frameon": False, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})
S2 = sns.color_palette("Set2")
DAYS = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30]; DL = ["3", "6", "8", "10", "13", "15", "17", "20.5", "24", "28", "30"]
M = [("traj", "Size trajectory (days 3–X)", "#4d4d4d", "D", "-"), ("size", "Size (that day)", S2[0], "o", "-"),
     ("std", "CNN, standard aug", S2[2], "s", "--"), ("strong", "CNN, strong aug", S2[1], "^", "--"),
     ("bbox", "CNN, bbox crop", S2[3], "v", "--")]
fig, ax = plt.subplots(figsize=(10.5, 5.6))
for key, lab, c, mk, ls in M:
    g = R[R.model == key].groupby("day").matched_auc
    ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), color=c, marker=mk, ms=7, mec="k", mew=.6, lw=2.2 if ls == "-" else 1.6,
                ls=ls, capsize=3, label=lab)
n = R[R.model == "size"].groupby("day").n_not_matched.first()
for d, v in n.items():
    ax.text(d, 0.315, f"{v}", ha="center", fontsize=11, color="0.4")
ax.text(0.99, 0.085, "grey numbers = Not Acceptable organoids with a same-size (±10%) Acceptable match",
        transform=ax.transAxes, ha="right", fontsize=10.5, color="0.4")
ax.axhline(.5, color="0.6", ls="--", lw=1, zorder=0)
ax.set_ylim(.28, .92); ax.set_xticks(DAYS); ax.set_xticklabels(DL); ax.tick_params(axis="x", labelsize=13)
ax.set_xlabel("Day"); ax.set_ylabel("Size-matched AUC\n(same-size pairs; 0.5 = nothing beyond size)")
ax.legend(loc="upper left", ncol=2, fontsize=12)
sns.despine(fig=fig); fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(OUT / f"14_size_matched_auc.{e}")
print("saved 14")
