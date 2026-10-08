"""Fig 18: growth (size percentile vs Acceptable organoids) for cyst vs non-cyst Not Acceptable organoids."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv(Path(sys.argv[1]) / "translucent_groups.csv"); OUT = Path(sys.argv[2])
sns.set_theme(context="paper", style="white")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 15,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 11.5, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
S2 = sns.color_palette("Set2")
DAYS = sorted(d.day.unique())
acc = d[d.group == "Acceptable"]
d["pct"] = [(acc[acc.day == r.day].area_frac < r.area_frac).mean() * 100 for r in d.itertuples()]
overgrow = {"BA2_96_2_E12_nosplit", "BA2_96_2_H2_nosplit", "BA1_96_1_C7_nosplit"}
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
for a, (title, sel, col) in zip(ax, (("Cyst + stalled growth (5 organoids)", lambda o: o not in overgrow, S2[1]),
                                     ("Cyst + kept growing / overgrowth (3 organoids)", lambda o: o in overgrow, S2[3]))):
    nc = d[d.group == "Not Acc: no cyst"].groupby("day").pct
    a.fill_between(nc.median().index, nc.quantile(.25), nc.quantile(.75), color="0.6", alpha=.2, lw=0)
    a.plot(nc.median().index, nc.median(), color="0.35", lw=2.5, ls="--", label="Not Acc., no cyst (median, IQR)")
    a.axhline(50, color=S2[2], lw=2.5, label="Acceptable (median = 50th pct)")
    for o in sorted(d[(d.group == "Not Acc: cyst")].organoid_id.unique()):
        if not sel(o):
            continue
        s = d[d.organoid_id == o].sort_values("day")
        a.plot(s.day, s.pct, "o-", color=col, lw=1.8, ms=5, mec="k", mew=.5, alpha=.9)
        if o in overgrow:
            off = {"BA2_96_2_E12_nosplit": 4, "BA2_96_2_H2_nosplit": -4, "BA1_96_1_C7_nosplit": 0}[o]
            a.text(s.day.iloc[-1] + .4, s.pct.iloc[-1] + off, o.replace("_nosplit", ""), fontsize=10.5, va="center")
    a.plot([], [], "o-", color=col, label="individual cyst organoids")
    a.set_title(title); a.set_xticks(DAYS); a.set_xticklabels([f"{x:g}" for x in DAYS], fontsize=12)
    a.set_xlabel("Day"); a.set_xlim(2, 33.5); a.set_ylim(-3, 103)
    a.axvspan(2, 10.5, color="0.95", zorder=0)
ax[0].set_ylabel("Size percentile among\nAcceptable organoids that day")
ax[0].legend(loc="upper right")
ax[0].text(6.2, 34, "shaded: days 3-10,\nwhen Not Acc. organoids\nare bigger than typical", ha="center", va="top",
           fontsize=10.5, color="0.4")
sns.despine(fig=fig); fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(OUT / f"18_cyst_organoids_growth.{e}")
print("saved 18")
