"""Size-matched concordance + likelihood-ratio test: do models know anything beyond size?"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MP, RUNS = D / "amanda_test/model_plots", D / "model_tests/lstm_runs/idor_main"
DAYS = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30]; DL = ["3", "6", "8", "10", "13", "15", "17", "20.5", "24", "28", "30"]

feat = pd.read_csv(MP / "mask_features_idor_main.csv")
ids = sorted(feat.organoid_id.unique())
y = feat.drop_duplicates("organoid_id").set_index("organoid_id").loc[ids, "true_label"].values
la = feat.pivot(index="organoid_id", columns="day", values="log_area").loc[ids]


def cv_probs(X, rep):
    p = np.empty(len(ids))
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=1 + rep * 1000).split(ids, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
        p[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def loglik(X):
    m = LogisticRegression(penalty=None, max_iter=1000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


oof = {k: {} for k in ("std", "strong", "bboxstd")}
for k in oof:
    for d, dl in zip(DAYS, DL):
        o = pd.read_csv(RUNS / f"base_effnet_kfold4x10_{k}/day_{dl}/oof_predictions.csv")
        oof[k][d] = {r: g.set_index("organoid_id").loc[ids, "prob_acceptable"].values for r, g in o.groupby("repeat")}

rows, lrt = [], []
for k, d in enumerate(DAYS):
    a = la[d].values
    for cal in (0.10, 0.05):
        diff = np.abs(a[:, None] - a[None, :]) <= np.log(1 + cal)
        pairs = [(n, q) for n in np.where(y == 0)[0] for q in np.where(y == 1)[0] if diff[n, q]]
        nn = len({n for n, _ in pairs})
        for rep in range(10):
            probs = {"size": cv_probs(a[:, None], rep),
                     "traj": cv_probs(la[DAYS[:k + 1]].values, rep) if k > 0 else None,
                     "std": oof["std"][d][rep + 1], "strong": oof["strong"][d][rep + 1], "bbox": oof["bboxstd"][d][rep + 1]}
            for m, p in probs.items():
                if p is None or not pairs:
                    continue
                conc = np.mean([(p[q] > p[n]) + .5 * (p[q] == p[n]) for n, q in pairs])
                rows.append({"day": d, "caliper": cal, "repeat": rep + 1, "model": m, "matched_auc": conc,
                             "n_pairs": len(pairs), "n_not_matched": nn,
                             "mean_logratio": float(np.mean([a[q] - a[n] for n, q in pairs]))})
    # likelihood-ratio test: size vs size + model (model prob as an out-of-fold covariate)
    for m in ("std", "strong", "bbox"):
        key = "bboxstd" if m == "bbox" else m
        ps = []
        for rep in range(10):
            p = np.clip(oof[key][d][rep + 1], 1e-6, 1 - 1e-6)
            X1 = a[:, None]; X2 = np.c_[a, np.log(p / (1 - p))]
            ps.append(chi2.sf(2 * (loglik(X2) - loglik(X1)), 1))
        lrt.append({"day": d, "model": m, "median_p": np.median(ps), "frac_p_lt_05": np.mean(np.array(ps) < .05)})

R = pd.DataFrame(rows); L = pd.DataFrame(lrt)
R.to_csv(OUT / "size_matched_auc.csv", index=False); L.to_csv(OUT / "size_plus_model_lrt.csv", index=False)
for cal in (0.10, 0.05):
    print(f"\n=== size-matched AUC (caliper ±{cal:.0%} area), mean over 10 repeats ===")
    t = R[R.caliper == cal].groupby(["day", "model"]).matched_auc.mean().unstack()[["size", "traj", "std", "strong", "bbox"]]
    info = R[(R.caliper == cal) & (R.model == "size")].groupby("day")[["n_pairs", "n_not_matched", "mean_logratio"]].first()
    print(pd.concat([t.round(3), info.assign(mean_logratio=info.mean_logratio.round(3))], axis=1).to_string())
print("\n=== LRT: does adding the model's prediction to size improve fit? (median p over repeats; frac p<.05) ===")
print(L.pivot(index="day", columns="model", values="median_p").map(lambda v: f"{v:.3g}").join(
      L.pivot(index="day", columns="model", values="frac_p_lt_05").add_suffix("_frac<.05")).to_string())
