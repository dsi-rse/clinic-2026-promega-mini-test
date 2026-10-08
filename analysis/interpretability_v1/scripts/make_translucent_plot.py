"""Fig 17: translucent (cyst-like) fraction vs size, and what it adds to the size trajectory."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
df = pd.read_csv(sys.argv[1]); OUT = Path(sys.argv[2])
sns.set_theme(context="paper", style="white")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 15,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
S2 = sns.color_palette("Set2"); ACC, NOT, BRD = S2[2], S2[1], "0.65"
DAYS = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30]
fig, ax = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1.15, 1, 1]})

# (a) day 30: size vs translucent fraction
g = df[df.day == 30]
for lab, c, z in (("borderline", BRD, 1), ("Acceptable", ACC, 2), ("Not Acceptable", NOT, 3)):
    h = g[g.label == lab]
    ax[0].scatter(h.area_frac * 100, h.translucent_frac * 100 + 0.05, s=45, color=c, edgecolor="k", linewidth=.5,
                  alpha=.85, zorder=z, label=lab.replace("borderline", "Borderline (2/5, 3/5)"))
ax[0].set_xscale("log"); ax[0].set_yscale("log")
for a_, ticks in ((ax[0].xaxis, [2, 3, 4, 6, 10, 20]), (ax[0].yaxis, [0.1, 1, 10, 30])):
    a_.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    a_.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a_.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].axhline(5, color="0.6", ls="--", lw=1)
ax[0].set_xlabel("Organoid area (% of frame)"); ax[0].set_ylabel("Translucent area (% of organoid)")
ax[0].set_title("Day 30: cyst-like translucent regions", fontsize=15); ax[0].legend(loc="upper left", fontsize=11)

# (b) % of organoids with >5% translucent, by day
c = df[df.in_main]
p = c.assign(t=c.translucent_frac > .05).groupby(["day", "label"]).t.mean().unstack() * 100
ax[1].plot(p.index, p["Acceptable"], "o-", color=ACC, lw=2.2, ms=7, mec="k", mew=.6, label="Acceptable")
ax[1].plot(p.index, p["Not Acceptable"], "o-", color=NOT, lw=2.2, ms=7, mec="k", mew=.6, label="Not Acceptable")
ax[1].set_xticks(DAYS); ax[1].set_xticklabels([f"{d:g}" for d in DAYS], fontsize=12)
ax[1].set_xlabel("Day"); ax[1].set_ylabel("Organoids with >5%\ntranslucent area (%)"); ax[1].legend(loc="upper left")

# (c) AUC: size trajectory vs + translucent fraction on that day (same 4x10 folds)
ids = sorted(c.organoid_id.unique())
y = c.drop_duplicates("organoid_id").set_index("organoid_id").loc[ids, "label"].eq("Acceptable").astype(int).values
A = np.log(c.pivot(index="organoid_id", columns="day", values="area_frac").loc[ids])
L = c.pivot(index="organoid_id", columns="day", values="translucent_frac").loc[ids].fillna(0)
res = []
for k, d in enumerate(DAYS):
    if d < 17:
        continue
    for name, X in (("Size trajectory", A[DAYS[:k + 1]].values),
                    ("+ translucent area (that day)", np.c_[A[DAYS[:k + 1]].values, L[d].values])):
        for rep in range(10):
            pr = np.empty(len(ids))
            for tr, te in StratifiedKFold(4, shuffle=True, random_state=1 + rep * 1000).split(ids, y):
                pr[te] = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            res.append({"day": d, "model": name, "auc": roc_auc_score(y, pr)})
res = pd.DataFrame(res)
for name, col, mk in (("Size trajectory", "#4d4d4d", "D"), ("+ translucent area (that day)", S2[4], "s")):
    s = res[res.model == name].groupby("day").auc
    ax[2].errorbar(s.mean().index, s.mean(), yerr=s.std(), color=col, marker=mk, ms=7, mec="k", mew=.6, lw=2.2,
                   capsize=3, label=name)
ax[2].set_xticks([17, 20.5, 24, 28, 30]); ax[2].set_xticklabels(["3–17", "3–20.5", "3–24", "3–28", "3–30"], fontsize=12)
ax[2].set_xlabel("Days used"); ax[2].set_ylabel("AUC (4×10 CV)"); ax[2].legend(loc="upper left"); ax[2].set_ylim(.8, .96)
sns.despine(fig=fig); fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(OUT / f"17_translucent_regions.{e}")
print("saved 17")
