"""V2 evaluation: per-day CNNs and gap-tolerant LSTMs on the edge-filtered cohort
(data/cohorts/idor_main/full), against size baselines on the SAME organoids and folds.

Run from the repo root on the cluster (core_env):
    python analysis/interpretability_v2/scripts/perday_eval.py <out_dir>
Writes perday_cnn.csv, perday_lstm.csv, perday_ranks.csv and v2_summary.png/pdf to <out_dir>.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
RUNS = Path("/net/projects2/promega/project_data/model_tests/lstm_runs")
NEW, OLD = RUNS / "idor_main_perday", RUNS / "idor_main"
DAYS = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30]
CNN = {"std": "std_native", "strong": "strong_native", "bbox": "bboxstd", "brightbbox": "brightbbox"}  # new -> old run
NAMES = {"std": "CNN standard aug", "strong": "CNN strong aug", "bbox": "CNN bbox", "brightbbox": "CNN brightened + bbox"}
CYST = ["BA2_96_2_E12_nosplit", "BA2_96_2_H2_nosplit", "BA1_96_1_C7_nosplit"]

meta = {}
for p in ("train", "val", "test"):
    meta.update(json.load(open(f"data/cohorts/idor_main/full/{p}.json")))
old140 = set()
for p in ("train", "val", "test"):
    old140 |= set(json.load(open(f"data/cohorts/idor_main/series/{p}.json")))
lab = {o: int(v["label"] == "Acceptable") for o, v in meta.items()}
plate = lambda o: "_".join(o.split("_")[:3])

# ---- log mask area per organoid/day (clipped masks, same as the models) ----
area_csv = OUT / "perday_mask_area.csv"
if area_csv.exists():
    A = pd.read_csv(area_csv)
else:
    A = pd.DataFrame([{"organoid_id": o, "day": float(tp["mdl_day"]),
                       "log_area": float(np.log((np.asarray(Image.open(tp["mask_paths"]["clipped"]).convert("L")) > 127).sum()))}
                      for o, v in meta.items() for tp in v["timepoints"]])
    A.to_csv(area_csv, index=False)
LA = A.pivot(index="organoid_id", columns="day", values="log_area")

lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))


def oof_fit(X, y, folds):
    p = np.empty(len(y))
    for k in np.unique(folds):
        te = folds == k; tr = ~te
        p[te] = lr().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def same_size(p, a, y):
    pr = [(n, q) for n in np.where(y == 0)[0] for q in np.where(y == 1)[0] if abs(a[n] - a[q]) <= np.log(1.1)]
    return (np.mean([(p[q] > p[n]) + .5 * (p[q] == p[n]) for n, q in pr]), len(pr)) if pr else (np.nan, 0)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def score(o, size_X, a):
    """o: OOF rows (repeat, fold, organoid_id, prob_acceptable). size_X(ids) -> size features."""
    out = {k: [] for k in ("auc", "bal", "size_auc", "size_bal", "flex_auc", "same", "gain", "gain_flex",
                           "ba2_auc", "ba2_size", "ba2_same", "old_auc")}
    for r, g in o.groupby("repeat"):
        g = g.set_index("organoid_id"); ids = sorted(g.index)
        y = np.array([lab[i] for i in ids]); p = g.loc[ids, "prob_acceptable"].values; f = g.loc[ids, "fold"].values
        S = size_X(ids); av = a(ids)
        P = pd.get_dummies([plate(i) for i in ids], drop_first=True).values.astype(float)
        ps = oof_fit(S, y, f); p0 = oof_fit(np.c_[S, P], y, f); p1 = oof_fit(np.c_[S, P, logit(p)], y, f)
        out["auc"].append(roc_auc_score(y, p)); out["bal"].append(balanced_accuracy_score(y, p > .5))
        out["size_auc"].append(roc_auc_score(y, ps)); out["size_bal"].append(balanced_accuracy_score(y, ps > .5))
        out["same"].append(same_size(p, av, y)[0]); out["gain"].append(roc_auc_score(y, p1) - roc_auc_score(y, p0))
        # two-way size: a spline on size lets BOTH very small and very large be bad
        Fx = np.c_[SplineTransformer(n_knots=4, degree=3).fit_transform(S[:, :1]), S[:, 1:], P]
        q0 = oof_fit(Fx, y, f); q1 = oof_fit(np.c_[Fx, logit(p)], y, f)
        out["flex_auc"].append(roc_auc_score(y, q0)); out["gain_flex"].append(roc_auc_score(y, q1) - roc_auc_score(y, q0))
        b = np.array([i.startswith("BA2") for i in ids])
        out["ba2_auc"].append(roc_auc_score(y[b], p[b])); out["ba2_size"].append(roc_auc_score(y[b], av[b]))
        out["ba2_same"].append(same_size(p[b], av[b], y[b])[0])
        k = np.array([i in old140 for i in ids]); out["old_auc"].append(roc_auc_score(y[k], p[k]))
    res = {k: float(np.nanmean(v)) for k, v in out.items()}
    res["gain_wins"] = int(sum(x > 0 for x in out["gain"])); res["gain_flex_wins"] = int(sum(x > 0 for x in out["gain_flex"]))
    res["n"] = int(o.organoid_id.nunique()); res["n_bad"] = int(sum(1 - lab[i] for i in o.organoid_id.unique()))
    return res


rows, ranks = [], []
for d in DAYS:
    size_X = lambda ids, d=d: LA.loc[ids, [d]].values
    a = lambda ids, d=d: LA.loc[ids, d].values
    for v in CNN:
        o = pd.read_csv(NEW / f"base_effnet_kfold4x10_{v}" / f"day_{d:g}" / "oof_predictions.csv")
        r = score(o, size_X, a)
        old = pd.read_csv(OLD / f"base_effnet_kfold4x10_{CNN[v]}" / f"day_{d:g}" / "oof_predictions.csv")
        r["old_run_auc"] = float(np.mean([roc_auc_score(g.true_label, g.prob_acceptable) for _, g in old.groupby("repeat")]))
        rows.append(dict(model=v, day=d, **r))
        if d in (24, 28, 30):
            m = o.groupby("organoid_id").prob_acceptable.mean()
            for c in CYST + [i for i in m.index if i not in old140 and lab[i] == 0]:
                if c in m.index:
                    ranks.append(dict(model=v, day=d, organoid=c.replace("_nosplit", ""), kind="cyst" if c in CYST else "new bad",
                                      p_acceptable=round(float(m[c]), 3), rank=int((m <= m[c]).sum()), n=len(m)))
C = pd.DataFrame(rows); C.to_csv(OUT / "perday_cnn.csv", index=False)

# ---- LSTM: size-history baseline = size on last day + growth slope over the available days ----
def hist_X(ids, W):
    X = []
    for i in ids:
        s = LA.loc[i, [d for d in DAYS if d <= W]].dropna()
        X.append([s[W], np.polyfit(s.index.values.astype(float), s.values, 1)[0]])
    return np.array(X)


lrows = []
for W in (13, 17, 24, 30):
    for ro in ("last", "mean"):
        o = pd.concat(map(pd.read_csv, glob.glob(str(NEW / f"temporal_lstm_kfold4x10_{ro}" / f"days_3-{W}" / "oof_rep*.csv"))))
        r = score(o, lambda ids, W=W: hist_X(ids, W), lambda ids, W=W: LA.loc[ids, W].values)
        oldf = glob.glob(str(OLD / "temporal_lstm_kfold4x10" / f"days_3-{W}" / "oof_rep*.csv"))
        r["old_run_auc"] = float(np.mean([roc_auc_score(g.true_label, g.prob_acceptable)
                                          for _, g in pd.concat(map(pd.read_csv, oldf)).groupby("repeat")])) if ro == "last" else np.nan
        lrows.append(dict(model=f"LSTM {ro}", window=f"3-{W}", max_day=W, **r))
        if W == 30:
            m = o.groupby("organoid_id").prob_acceptable.mean()
            for c in CYST + [i for i in m.index if i not in old140 and lab[i] == 0]:
                ranks.append(dict(model=f"LSTM {ro}", day=30, organoid=c.replace("_nosplit", ""), kind="cyst" if c in CYST else "new bad",
                                  p_acceptable=round(float(m[c]), 3), rank=int((m <= m[c]).sum()), n=len(m)))
L = pd.DataFrame(lrows); L.to_csv(OUT / "perday_lstm.csv", index=False)
R = pd.DataFrame(ranks); R.to_csv(OUT / "perday_ranks.csv", index=False)

pd.set_option("display.width", 250)
cols = ["n", "n_bad", "size_auc", "flex_auc", "auc", "old_run_auc", "old_auc", "same", "gain", "gain_wins",
        "gain_flex", "gain_flex_wins", "ba2_size", "ba2_auc", "ba2_same"]
for v in CNN:
    print(f"\n=== {NAMES[v]} ===")
    print(C[C.model == v].set_index("day")[cols].round(3).to_string())
print("\n=== LSTM (size baseline = size on last day + growth slope) ===")
print(L.set_index(["window", "model"])[cols].round(3).to_string())
print("\n=== ranks (1 = most confidently bad) ===")
print(R.pivot_table(index=["kind", "organoid"], columns=["model", "day"], values="rank").to_string())

# ---- summary figure ----
sns.set_theme(context="paper", style="white", font_scale=1.5)
pal = dict(zip(["size"] + list(CNN), ["#4d4d4d"] + sns.color_palette("Set2", 4).as_hex()))
fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))
sz = C[C.model == "std"].set_index("day").size_auc
ax[0].plot(range(len(DAYS)), sz.values, "o--", color=pal["size"], lw=2, label="size (bigger = better)")
ax[0].plot(range(len(DAYS)), C[C.model == "std"].set_index("day").flex_auc.values, "o-", color="#000000", lw=2.5,
           label="size (too small or too big = bad)")
for v in CNN:
    s = C[C.model == v].set_index("day").auc
    ax[0].plot(range(len(DAYS)), s.values, "o-", color=pal[v], lw=1.8, label=NAMES[v])
    ax[1].plot(range(len(DAYS)), C[C.model == v].set_index("day").same.values, "o-", color=pal[v], lw=1.8, label=NAMES[v])
for a_, t, yl in ((ax[0], "Per-day models vs size", "AUC"), (ax[1], "Same-size pairs (size can't help)", "AUC on same-size pairs")):
    a_.set_xticks(range(len(DAYS))); a_.set_xticklabels([f"{d:g}" for d in DAYS]); a_.set_xlabel("Day")
    a_.set_ylabel(yl); a_.set_title(t); a_.axhline(0.5, color="#999999", ls="--", lw=1)
ax[0].legend(frameon=False, fontsize=10, loc="upper left")
ws = ["3-13", "3-17", "3-24", "3-30"]
x = np.arange(len(ws)); w = 0.2
Ll = L[L.model == "LSTM last"].set_index("window")
ax[2].bar(x - 1.5 * w, Ll.size_auc.loc[ws].values, w, color="#bdbdbd", label="size + growth")
ax[2].bar(x - 0.5 * w, Ll.flex_auc.loc[ws].values, w, color="#4d4d4d", label="size (two-way) + growth")
for k, (ro, c) in enumerate((("last", "#8da0cb"), ("mean", "#66c2a5"))):
    ax[2].bar(x + (0.5 + k) * w, L[L.model == f"LSTM {ro}"].set_index("window").auc.loc[ws].values, w, color=c, label=f"LSTM {ro}")
ax[2].set_xticks(x); ax[2].set_xticklabels([f"days {s}" for s in ws]); ax[2].set_ylim(0.4, 1)
ax[2].axhline(0.5, color="#999999", ls="--", lw=1); ax[2].set_ylabel("AUC"); ax[2].set_title("Time-series models vs size + growth")
ax[2].legend(frameon=False, fontsize=11, loc="upper left")
sns.despine(fig=fig); fig.tight_layout()
fig.savefig(OUT / "v2_summary.png", dpi=150, bbox_inches="tight"); fig.savefig(OUT / "v2_summary.pdf", bbox_inches="tight")
