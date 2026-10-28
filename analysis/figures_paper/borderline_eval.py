"""Borderline (majority-label) brightbbox CNN vs the clear-only run, both against two-cutoff size
on the same organoids and folds (Fig S b). Cohort: scripts/splits/make_splits.py scripts/splits/configs/idor.json
--min-majority-votes 3 --output-suffix _main_minvotes3 (regular-image votes only). Run from the repo root
on the cluster (core_env): python analysis/figures_paper/borderline_eval.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

RUNS = Path("/net/projects2/promega/project_data/model_tests/lstm_runs")
MV3 = RUNS / "idor_main_minvotes3_perday/base_effnet_kfold4x10_brightbbox"
MAIN = RUNS / "idor_main_perday/base_effnet_kfold4x10_brightbbox"
OUT = Path("/net/projects2/promega/project_data/amanda_test/model_plots/borderline_mv3"); OUT.mkdir(parents=True, exist_ok=True)
DAYS = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30]


def load(c):
    m = {}
    for p in ("train", "val", "test"):
        m.update(json.load(open(f"data/cohorts/{c}/full/{p}.json")))
    return m


meta, clear = load("idor_main_minvotes3"), load("idor_main")
assert set(clear) <= set(meta) and all(clear[o]["label"] == meta[o]["label"] for o in clear)
lab = {o: int(v["label"] == "Acceptable") for o, v in meta.items()}
border = {o for o in meta if o not in clear}
print(f"{len(meta)} organoids: {len(clear)} clear, {len(border)} borderline "
      f"({sum(lab[o] for o in border)} Acc 3/5, {sum(1 - lab[o] for o in border)} Not 2/5)")

LA = pd.DataFrame([{"organoid_id": o, "day": float(tp["mdl_day"]),
                    "log_area": float(np.log((np.asarray(Image.open(tp["mask_paths"]["clipped"]).convert("L")) > 127).sum()))}
                   for o, v in meta.items() for tp in v["timepoints"]]).pivot(index="organoid_id", columns="day", values="log_area")
plate = lambda o: "_".join(o.split("_")[:3])
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))


def size2(ids, f, d):
    """two-cutoff size (spline on log area + plate), OOF on folds f"""
    S = LA.loc[ids, [d]].values
    P = pd.get_dummies([plate(i) for i in ids], drop_first=True).values.astype(float)
    X = np.c_[SplineTransformer(n_knots=4, degree=3).fit_transform(S), P]
    p = np.empty(len(ids))
    for k in np.unique(f):
        te = f == k
        p[te] = lr().fit(X[~te], y_of(ids)[~te]).predict_proba(X[te])[:, 1]
    return p


y_of = lambda ids: np.array([lab[i] for i in ids])


def auc(ids, p, keep):
    k = np.array([i in keep for i in ids]); y = y_of(ids)[k]
    return roc_auc_score(y, p[k]) if 0 < y.sum() < len(y) else np.nan


rows = []
for d in DAYS:
    for name, run, pool in (("trained with borderline", MV3, set(meta)), ("trained clear only", MAIN, set(clear))):
        o = pd.read_csv(run / f"day_{d}" / "oof_predictions.csv")
        r = {k: [] for k in ("cnn_clear", "size_clear", "cnn_all", "size_all", "cnn_border", "size_border")}
        for _, g in o.groupby("repeat"):
            g = g.set_index("organoid_id"); ids = sorted(g.index)
            p, f = g.loc[ids, "prob_acceptable"].values, g.loc[ids, "fold"].values
            s = size2(ids, f, float(d))
            for sub, keep in (("clear", set(clear)), ("all", set(meta)), ("border", border)):
                if sub != "clear" and pool == set(clear):
                    continue
                r[f"cnn_{sub}"].append(auc(ids, p, keep)); r[f"size_{sub}"].append(auc(ids, s, keep))
        n = o.organoid_id.nunique()
        rows.append({"day": d, "model": name, "n": n, **{k: float(np.nanmean(v)) if v else np.nan for k, v in r.items()}})
R = pd.DataFrame(rows)
R.to_csv(OUT / "borderline_vs_clear.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(3).to_string(index=False))
