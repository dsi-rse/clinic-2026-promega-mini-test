"""Interpretability figures for the idor_main 4x10 CV results (style: seaborn paper,
no grid, Set2 muted palette, large fonts, despined)."""
import glob, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
D = Path(sys.argv[1])            # plotdata dir
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
MP = D / "amanda_test/model_plots"
RUNS = D / "model_tests/lstm_runs/idor_main"

sns.set_theme(context="paper", style="white", font_scale=1.0)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 18, "axes.titlesize": 18,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 13, "legend.frameon": False,
    "axes.linewidth": 1.1, "savefig.dpi": 300, "savefig.bbox": "tight",
})
S2 = sns.color_palette("Set2")
C = {"size": S2[0], "std": S2[2], "strong": S2[1], "bbox": S2[3], "shape": S2[4],
     "sizeshape": S2[6], "traj": "#4d4d4d", "acc": S2[2], "not": S2[1]}
DAYS = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30]
DL = ["3", "6", "8", "10", "13", "15", "17", "20.5", "24", "28", "30"]


def finish(fig, ax, name, chance=None):
    if chance is not None:
        for a in np.atleast_1d(ax):
            a.axhline(chance, color="0.6", ls="--", lw=1, zorder=0)
    sns.despine(fig=fig, trim=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("saved", name)


def day_axis(ax, label=True):
    ax.set_xticks(DAYS); ax.set_xticklabels(DL, rotation=0)
    ax.tick_params(axis="x", labelsize=13)
    if label:
        ax.set_xlabel("Day")


# ---------------- data ----------------
sb = pd.read_csv(MP / "size_baseline_idor_main_kfold4x10.csv")
feat = pd.read_csv(MP / "mask_features_idor_main.csv")
feat["label"] = np.where(feat.true_label == 1, "Acceptable", "Not Acceptable")

# size trajectory under the same folds (StratifiedKFold seed 1+r*1000 on sorted ids;
# checked below against the CNN oof fold column)
ids = sorted(feat.organoid_id.unique())
y = feat.drop_duplicates("organoid_id").set_index("organoid_id").loc[ids, "true_label"].values
la = feat.pivot(index="organoid_id", columns="day", values="log_area").loc[ids]
oof_std = pd.read_csv(RUNS / "base_effnet_kfold4x10_std/day_30/oof_predictions.csv")
f1 = oof_std[oof_std.repeat == 1].set_index("organoid_id").loc[ids, "fold"].values
sk = list(StratifiedKFold(4, shuffle=True, random_state=1).split(ids, y))
mine = np.empty(len(ids), int)
for k, (_, te) in enumerate(sk):
    mine[te] = k + 1
assert (mine == f1).all(), "fold reconstruction does not match CNN folds"
traj = []
for k, d in enumerate(DAYS):
    if k == 0:
        continue
    X = la[[x for x in DAYS[:k + 1]]].values
    for rep in range(10):
        p = np.empty(len(ids))
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=1 + rep * 1000).split(ids, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
            p[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        traj.append({"day": d, "repeat": rep + 1, "traj_bal_acc": balanced_accuracy_score(y, p > .5),
                     "traj_auc": roc_auc_score(y, p)})
traj = pd.DataFrame(traj)

# ---------------- 1. growth curves ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
g = feat.assign(area_pct=feat.area_frac * 100)
for lab, col in (("Acceptable", C["acc"]), ("Not Acceptable", C["not"])):
    s = g[g.label == lab].groupby("day").area_pct
    med, lo, hi = s.median(), s.quantile(.25), s.quantile(.75)
    ax[0].fill_between(med.index, lo, hi, color=col, alpha=.25, lw=0)
    ax[0].plot(med.index, med, "o-", color=col, lw=2.2, ms=6, mec="k", mew=.6,
               label=f"{lab} (n={g[g.label == lab].organoid_id.nunique()})")
ax[0].set_yscale("log"); ax[0].set_ylabel("Organoid area (% of frame)")
ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
day_axis(ax[0]); ax[0].legend(loc="upper left")
ax[0].set_title("Median ± IQR", fontsize=15)
aucs = [roc_auc_score(g[g.day == d].true_label, g[g.day == d].area_frac) for d in DAYS]
ax[1].bar(range(len(DAYS)), np.array(aucs) - .5, bottom=.5,
          color=[C["acc"] if a > .5 else C["not"] for a in aucs], edgecolor="k", lw=.6)
ax[1].set_xticks(range(len(DAYS))); ax[1].set_xticklabels(DL, fontsize=12, rotation=45)
ax[1].set_ylabel("AUC (bigger = Acceptable)"); ax[1].set_xlabel("Day"); ax[1].set_ylim(.2, .9)
ax[1].text(.02, .04, "Not Acc. bigger", transform=ax[1].transAxes, fontsize=12, color="0.35")
ax[1].text(.02, .95, "Acc. bigger", transform=ax[1].transAxes, fontsize=12, color="0.35", ha="left", va="top")
finish(fig, ax[1], "01_growth_curves_size_by_outcome", chance=.5)

# ---------------- 2. size vs CNNs ----------------
fig, ax = plt.subplots(figsize=(9.5, 5.2))
series = [("size_bal_acc", "Size (one day)", C["size"], "o", sb),
          ("traj_bal_acc", "Size trajectory (days 3–X)", C["traj"], "D", traj),
          ("std_bal_acc", "CNN, standard aug", C["std"], "s", sb),
          ("strong_bal_acc", "CNN, strong aug", C["strong"], "^", sb),
          ("bboxstd_bal_acc", "CNN, bbox crop (size removed)", C["bbox"], "v", sb)]
for col, lab, c, mk, df in series:
    s = df.groupby("day")[col]
    ax.errorbar(s.mean().index, s.mean(), yerr=s.std(), color=c, marker=mk, ms=7, mec="k", mew=.6,
                lw=2.2 if col.startswith(("size", "traj")) else 1.6, capsize=3, label=lab,
                ls="-" if col.startswith(("size", "traj")) else "--")
ax.set_ylabel("Balanced accuracy"); ax.set_ylim(.4, .9); day_axis(ax)
ax.legend(loc="upper left", ncol=1)
finish(fig, ax, "02_size_vs_cnn_balanced_accuracy", chance=.5)

# ---------------- 3. occlusion ----------------
occ = []
for s in ("std", "strong"):
    o = pd.read_csv(MP / f"occlusion_spatial_idor_main_kfold4x10_{s}_rep1.csv")
    for (d, cond), g2 in o[o.day.isin([24, 28, 30])].groupby(["day", "condition"]):
        g2 = g2.dropna(subset=["prob_acceptable"])
        occ.append({"setup": s, "day": d, "condition": cond, "auc": roc_auc_score(g2.true_label, g2.prob_acceptable)})
occ = pd.DataFrame(occ)
order = ["original", "silhouette", "rim_texture_removed", "core_texture_removed", "shifted",
         "scaled_0.7", "scaled_1.4", "no_organoid"]
names = ["Original", "Silhouette\n(size+outline)", "Rim texture\nremoved", "Core texture\nremoved",
         "Shifted", "Shrunk\n0.7×", "Enlarged\n1.4×", "Organoid\nremoved"]
fig, ax = plt.subplots(figsize=(12, 4.8))
sns.barplot(data=occ, x="condition", y="auc", hue="setup", order=order, hue_order=["std", "strong"],
            palette=[C["std"], C["strong"]], errorbar=("sd"), capsize=.08, err_kws={"linewidth": 1.2},
            edgecolor="k", linewidth=.6, ax=ax)
sns.stripplot(data=occ, x="condition", y="auc", hue="setup", order=order, hue_order=["std", "strong"],
              dodge=True, palette=["k", "k"], size=4, alpha=.6, legend=False, ax=ax)
ax.set_xticklabels(names, fontsize=12); ax.set_xlabel(""); ax.set_ylabel("AUC (days 24/28/30)")
ax.set_ylim(.3, .95)
h, l = ax.get_legend_handles_labels(); ax.legend(h[:2], ["Standard aug", "Strong aug"], loc="upper right", ncol=2)
finish(fig, ax, "03_occlusion_auc_by_condition", chance=.5)

# ---------------- 4. rescale intervention ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for i, s in enumerate(("std", "strong")):
    o = pd.read_csv(MP / f"occlusion_spatial_idor_main_kfold4x10_{s}_rep1.csv")
    w = o.pivot_table(index=["day", "organoid_id"], columns="condition", values="prob_acceptable").reset_index()
    for cond, col, lab in (("scaled_1.4", C["acc"], "Enlarged 1.4× (2× area)"), ("scaled_0.7", C["not"], "Shrunk 0.7× (½ area)")):
        dlt = (w[cond] - w["original"]).groupby(w.day)
        med, lo, hi = dlt.median(), dlt.quantile(.25), dlt.quantile(.75)
        ax[i].fill_between(med.index, lo, hi, color=col, alpha=.25, lw=0)
        ax[i].plot(med.index, med, "o-", color=col, lw=2.2, ms=6, mec="k", mew=.6, label=lab)
    ax[i].axhline(0, color="0.6", ls="--", lw=1); day_axis(ax[i])
    ax[i].set_title("Standard aug" if s == "std" else "Strong aug")
ax[0].set_ylabel("Δ P(Acceptable)\n(scaled − original)"); ax[0].legend(loc="lower left")
finish(fig, ax, "04_rescale_intervention_size_effect")

# ---------------- 5. size vs shape ----------------
fig, ax = plt.subplots(figsize=(9.5, 5))
for col, lab, c, mk in (("size_auc", "Size", C["size"], "o"), ("shape_auc", "Shape only (scale-free)", C["shape"], "s"),
                        ("sizeshape_auc", "Size + shape", C["sizeshape"], "D")):
    s = sb.groupby("day")[col]
    ax.errorbar(s.mean().index, s.mean(), yerr=s.std(), color=c, marker=mk, ms=7, mec="k", mew=.6, lw=2, capsize=3, label=lab)
ax.set_ylabel("AUC"); ax.set_ylim(.35, .9); day_axis(ax); ax.legend(loc="upper left")
finish(fig, ax, "05_size_vs_shape_auc", chance=.5)

# ---------------- 6. Grad-CAM focus ----------------
fig, ax = plt.subplots(figsize=(9.5, 5))
for s, lab in (("std", "Standard aug"), ("strong", "Strong aug")):
    f = pd.read_csv(MP / f"focus_idor_main_kfold4x10_{s}_rep1_alldays.csv")
    m = f.groupby("day").focus.mean() * 100
    se = f.groupby("day").focus.sem() * 100
    ax.errorbar(m.index, m, yerr=se, color=C[s], marker="o", ms=7, mec="k", mew=.6, lw=2.2, capsize=3, label=lab)
area = f.groupby("day").mask_frac.mean() * 100
ax.plot(area.index, area, color="0.5", ls="--", lw=1.4, label="Organoid area, % of image (chance)")
ax.set_ylabel("Share of total Grad-CAM\nheat on organoid (%)"); day_axis(ax); ax.legend(loc="upper left")
finish(fig, ax, "06_gradcam_heat_on_organoid")

# ---------------- 7. rotation consistency ----------------
rot = []
for fpath in (D / "rot").glob("*.csv"):        # flattened copies: <setup>_f<fold>_d<day>.csv
    s, _, d = fpath.stem.split("_"); d = d[1:]
    r = pd.read_csv(fpath)
    for a in ("90", "180", "270"):
        rot += [{"setup": s, "day": f"Day {d}", "angle": f"{a}°", "consistency": v} for v in r[f"consistency_{a}deg"]]
rot = pd.DataFrame(rot)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for i, d in enumerate(["Day 24", "Day 30"]):
    sns.boxplot(data=rot[rot.day == d], x="angle", y="consistency", hue="setup", hue_order=["std", "strong"],
                palette=[C["std"], C["strong"]], showfliers=False, linewidth=1, ax=ax[i])
    ax[i].set_title(d); ax[i].set_xlabel("Rotation"); ax[i].set_ylabel("")
    ax[i].get_legend().remove() if i == 0 else ax[i].legend(ax[i].get_legend_handles_labels()[0], ["Standard aug", "Strong aug"], loc="lower right")
ax[0].set_ylabel("Grad-CAM rotation\nconsistency (cosine)")
finish(fig, ax, "07_gradcam_rotation_consistency", chance=.7)

# ---------------- 8. per-class recall ----------------
rec = []
for s in ("std", "strong"):
    for d, dl in zip(DAYS, DL):
        o = pd.read_csv(RUNS / f"base_effnet_kfold4x10_{s}/day_{dl}/oof_predictions.csv")
        for r, g2 in o.groupby("repeat"):
            pred = g2.prob_acceptable > .5
            rec.append({"setup": s, "day": d, "Not Acceptable caught": (~pred[g2.true_label == 0]).mean(),
                        "Acceptable correct": pred[g2.true_label == 1].mean()})
rec = pd.DataFrame(rec)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for i, s in enumerate(("std", "strong")):
    for col, c in (("Acceptable correct", C["acc"]), ("Not Acceptable caught", C["not"])):
        m = rec[rec.setup == s].groupby("day")[col]
        ax[i].errorbar(m.mean().index, m.mean() * 100, yerr=m.std() * 100, color=c, marker="o", ms=6,
                       mec="k", mew=.6, lw=2, capsize=3, label=col)
    ax[i].set_title("Standard aug" if s == "std" else "Strong aug"); day_axis(ax[i]); ax[i].set_ylim(0, 100)
ax[0].set_ylabel("Recall (%)"); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
finish(fig, ax, "08_per_class_recall_cnn")

pd.concat([traj.groupby("day").mean()[["traj_bal_acc", "traj_auc"]]], axis=1).round(3).to_csv(OUT / "size_trajectory_summary.csv")
print("done")
