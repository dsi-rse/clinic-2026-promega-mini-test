"""Fig 13: Grad-CAM (predicted class) heat on organoid for std / strong / bbox models."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sns.set_theme(context="paper", style="white")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 18, "axes.titlesize": 17,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 13, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
S2 = sns.color_palette("Set2")
M = [("std", "Standard aug", S2[2], "o"), ("strong", "Strong aug", S2[1], "^"), ("bboxstd", "bbox crop", S2[3], "s")]
DAYS = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30]; DL = ["3", "6", "8", "10", "13", "15", "17", "20.5", "24", "28", "30"]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for key, lab, c, mk in M:
    r = pd.read_csv(D / f"focus_pred_idor_main_kfold4x10_{key}_rep1.csv")
    r["contrast"] = (r.focus / r.mask_frac) / ((1 - r.focus) / (1 - r.mask_frac))
    g = r.groupby("day")
    ax[0].plot(g.focus.median().index, g.focus.median() * 100, marker=mk, color=c, lw=2.2, ms=7, mec="k", mew=.6, label=lab)
    ax[0].plot(g.mask_frac.median().index, g.mask_frac.median() * 100, color=c, ls="--", lw=1.4)
    med, lo, hi = g.contrast.median(), g.contrast.quantile(.25), g.contrast.quantile(.75)
    ax[1].fill_between(med.index, lo, hi, color=c, alpha=.15, lw=0)
    ax[1].plot(med.index, med, marker=mk, color=c, lw=2.2, ms=7, mec="k", mew=.6, label=lab)
ax[0].plot([], [], color="0.4", ls="--", lw=1.4, label="Organoid area (chance)")
ax[0].set_ylabel("Share of total heat\non organoid (%)"); ax[0].legend(loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=12)
ax[0].set_title("Raw share (bbox high partly because\nthe organoid fills ~35–40% of its frame)", fontsize=14)
ax[1].set_yscale("log"); ax[1].set_ylim(0.15, 12); ax[1].axhline(1, color="0.6", ls="--", lw=1)
ax[1].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.25, 0.5, 1, 2, 4, 8]))
ax[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[1].set_ylabel("Heat per organoid pixel ÷\nheat per background pixel")
ax[1].set_title("Fair comparison: organoid vs background\nheat density (1× = no focus)", fontsize=14)
for a in ax:
    a.set_xticks(DAYS); a.set_xticklabels(DL); a.tick_params(axis="x", labelsize=13); a.set_xlabel("Day")
sns.despine(fig=fig); fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(OUT / f"13_gradcam_focus_std_strong_bbox_predclass.{e}")
print("saved 13")
