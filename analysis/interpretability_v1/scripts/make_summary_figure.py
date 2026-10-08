"""Fig 00: one-page summary (growth curves, day 10->17 growth, simple vs AI models, failure types)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")
D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sns.set_theme(context="paper", style="white")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
S2 = sns.color_palette("Set2"); ACC, NOT = S2[2], S2[1]
t = pd.read_csv(D / "translucent_idor.csv"); t = t[t.in_main].copy()
t["lab"] = np.where(t.label == "Acceptable", "Acceptable", "Not Acceptable")

fig = plt.figure(figsize=(17, 10.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1.35, .75, 1.1], hspace=.5, wspace=.45)

# A: growth curves
ax = fig.add_subplot(gs[0, :2])
for lab, c in (("Acceptable", ACC), ("Not Acceptable", NOT)):
    s = t[t.lab == lab].groupby("day").area_frac
    med, lo, hi = s.median() * 100, s.quantile(.25) * 100, s.quantile(.75) * 100
    ax.fill_between(med.index, lo, hi, color=c, alpha=.22, lw=0)
    ax.plot(med.index, med, "o-", color=c, lw=2.6, ms=7, mec="k", mew=.6,
            label=f"{lab} (n={t[t.lab == lab].organoid_id.nunique()})")
ax.set_yscale("log"); ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 2, 4, 8]))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.axvspan(10, 17, color="0.93", zorder=0)
ax.text(13.5, 0.85, "groups split\n(day 10–17)", ha="center", fontsize=12, color="0.35")
days = sorted(t.day.unique()); ax.set_xticks(days); ax.set_xticklabels([f"{d:g}" for d in days])
ax.set_xlabel("Day"); ax.set_ylabel("Organoid area (% of frame)"); ax.legend(loc="upper left")
ax.set_title("A   Good organoids keep growing; bad ones stall", loc="left", fontweight="bold")

# B: growth day 10 -> 17
ax = fig.add_subplot(gs[0, 2])
w = t.pivot(index="organoid_id", columns="day", values="area_frac")
lab = t.drop_duplicates("organoid_id").set_index("organoid_id").lab.loc[w.index]
g = pd.DataFrame({"fold": w[17] / w[10], "lab": lab})
sns.boxplot(data=g, x="lab", y="fold", order=["Acceptable", "Not Acceptable"], color="lightgray", showfliers=False, width=.55, ax=ax)
sns.stripplot(data=g, x="lab", y="fold", order=["Acceptable", "Not Acceptable"], hue="lab", palette={"Acceptable": ACC, "Not Acceptable": NOT},
              size=5, jitter=.22, edgecolor="k", linewidth=.5, legend=False, ax=ax)
p = mannwhitneyu(g[g.lab == "Acceptable"].fold, g[g.lab != "Acceptable"].fold).pvalue
ax.set_xticklabels(["Acceptable", "Not\nAcceptable"]); ax.set_xlabel(""); ax.set_ylabel("Growth, day 10 → 17 (× size)")
ax.set_title("B   Growth after day 10", loc="left", fontweight="bold")
ax.text(.5, .97, f"p = {p:.0e}", transform=ax.transAxes, ha="center", va="top", fontsize=12)

# C: simple measurements vs AI models (AUC, 4x10 CV; diamonds = unseen plate)
ax = fig.add_subplot(gs[1, :2])
sb = pd.read_csv(D / "amanda_test/model_plots/size_baseline_idor_main_kfold4x10.csv")
lt = pd.read_csv(D / "lstm_vs_traj.csv")
traj = sb[sb.day == 30]  # size/CNN day-30 rows
rows = [("CNN, day-30 image", sb[sb.day == 30].std_auc.mean(), None, "AI"),
        ("CNN, strong aug.", sb[sb.day == 30].strong_auc.mean(), None, "AI"),
        ("LSTM, image series 3–30", lt[lt.window == 30].lstm_auc.mean(), None, "AI"),
        ("Size, day 30", sb[sb.day == 30].size_auc.mean(), 0.844, "simple"),
        ("Growth, day 10→17", 0.84, 0.844, "simple"),
        ("Size history, days 3–30", lt[lt.window == 30].traj_auc.mean(), 0.855, "simple"),
        ("Size history + cyst check", 0.923, None, "simple")]
y_ = np.arange(len(rows))[::-1]
for yy, (name, auc, lopo, kind) in zip(y_, rows):
    ax.barh(yy, auc - .5, left=.5, color=S2[3] if kind == "AI" else S2[0], edgecolor="k", lw=.6, height=.65)
    ax.text(max(auc, lopo or 0) + .012, yy, f"{auc:.2f}", va="center", fontsize=12)
    if lopo:
        ax.plot(lopo, yy, "D", color="#4d4d4d", ms=8, mec="k", mew=.6)
ax.set_yticks(y_); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(.5, 1.0)
ax.set_xlabel("AUC (0.5 = chance)")
ax.plot([], [], "s", color=S2[3], ms=11, label="AI image models"); ax.plot([], [], "s", color=S2[0], ms=11, label="simple measurements")
ax.plot([], [], "D", color="#4d4d4d", ms=8, label="tested on an unseen plate")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=12); ax.set_title("C   Simple measurements match or beat AI models", loc="left", fontweight="bold")

# D: why the 25 rejected organoids failed
ax = fig.add_subplot(gs[1, 2])
cats = [("Stalled growth\n(caught by size)", 19, S2[1]), ("Cysts /\novergrowth", 3, S2[4]),
        ("Misshapen", 2, S2[5]), ("Unexplained", 1, "0.7")]
ax.barh(range(len(cats))[::-1], [c[1] for c in cats], color=[c[2] for c in cats], edgecolor="k", lw=.6, height=.65)
for i, (n_, v, _) in zip(range(len(cats))[::-1], cats):
    ax.text(v + .3, i, str(v), va="center", fontsize=13)
ax.set_yticks(range(len(cats))[::-1]); ax.set_yticklabels([c[0] for c in cats]); ax.set_xlim(0, 22)
ax.set_xlabel("Rejected organoids (of 25)")
ax.set_title("D   Why organoids were rejected", loc="left", fontweight="bold")

sns.despine(fig=fig)
for e in ("png", "pdf"):
    fig.savefig(OUT / f"00_summary.{e}")
print("saved 00_summary")
