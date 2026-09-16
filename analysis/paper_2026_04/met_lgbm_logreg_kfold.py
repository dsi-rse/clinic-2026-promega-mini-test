#!/usr/bin/env python3
"""Met + Morphology 10×4-fold repeated CV: LGBM vs LogReg, mirroring combined_kfold fold structure.

Key design: SEED=1, all_org_ids computed from full labeled organoids upfront,
folds split on the full list then subsetted per day — exactly matching combined_kfold.py.

Keys produced per day:
  met_nan_lgbm / met_nan_logreg — metabolite features with NaN floor
  met_raw_lgbm / met_raw_logreg — metabolite features raw (no floor)
  morph_lgbm   / morph_logreg   — morphology features

Outputs:
  analysis_output/images/met_lgbm_logreg_kfold_nan_raw.json  (early days: met nan/raw only)
  analysis_output/images/met_morph_lgbm_logreg_kfold.json    (all days: met_nan + morph)

Usage:
    python3 -m analysis.paper_2026_04.met_lgbm_logreg_kfold
    python3 -m analysis.paper_2026_04.met_lgbm_logreg_kfold --days Dy03 Dy06 Dy08 Dy10
    sbatch analysis/paper_2026_04/submit_met_lgbm_logreg_kfold.slurm
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    DAY_ORDER,
    LABEL_TO_INT,
    OrganoidDataset,
    idor_ba1_ba2_filters,
    require_complete_series,
)

from .metabolites_train import _features_for_day_all as _met_features_all
from .combined_kfold import _filter_fold
from analysis.multimodel.morphology_train import (
    _features_for_day as _morph_features_all,
    _load_morph_df,
)

warnings.filterwarnings("ignore", category=UserWarning)

SEED      = 1
N_FOLDS   = 4
N_REPEATS = 10
ALL_DATA_PATH = "data/all_data.json"
OUTPUT_PATH          = ANALYSIS_OUTPUT_DIR / "images" / "met_lgbm_logreg_kfold_139.json"
OUTPUT_PATH_NAN_RAW  = ANALYSIS_OUTPUT_DIR / "images" / "met_lgbm_logreg_kfold_nan_raw.json"
OUTPUT_PATH_ALL      = ANALYSIS_OUTPUT_DIR / "images" / "met_morph_lgbm_logreg_kfold.json"

LGBM_PARAM_GRID = {
    "max_depth":         [3, 6],
    "num_leaves":        [15, 31],
    "min_child_samples": [5, 10],
    "learning_rate":     [0.05, 0.1],
    "n_estimators":      [100, 300],
}

LOGREG_GRID = {
    "C":        [0.01, 0.1, 1.0, 10.0],
    "penalty":  ["l1", "l2"],
    "max_iter": [1000],
}


def _train_lgbm_fold(X_tr, y_tr, X_te, fold_seed: int) -> Optional[np.ndarray]:
    if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
        return None
    spw = float(np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1))
    model = lgb.LGBMClassifier(
        objective="binary", scale_pos_weight=spw,
        random_state=fold_seed, verbosity=-1, n_jobs=1,
    )
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=fold_seed)
    grid = GridSearchCV(model, LGBM_PARAM_GRID, cv=inner_cv,
                        scoring="f1", n_jobs=-1, refit=True)
    grid.fit(X_tr, y_tr)
    return grid.predict_proba(X_te)[:, 1]


def _train_logreg_fold(X_tr, y_tr, X_te, fold_seed: int) -> Optional[np.ndarray]:
    if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
        return None
    imputer = SimpleImputer(strategy="median")
    X_tr_i = imputer.fit_transform(X_tr)
    X_te_i = imputer.transform(X_te)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_i)
    X_te_s = scaler.transform(X_te_i)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=fold_seed)
    model = LogisticRegression(class_weight="balanced", solver="saga",
                               random_state=fold_seed)
    grid = GridSearchCV(model, LOGREG_GRID, cv=inner_cv,
                        scoring="f1_weighted", n_jobs=-1, refit=True)
    grid.fit(X_tr_s, y_tr)
    return grid.predict_proba(X_te_s)[:, 1]


def run_day(
    day: str,
    ds: OrganoidDataset,
    morph_df,
    all_org_ids: List[str],
    all_labels: np.ndarray,
    n_folds: int,
    n_repeats: int,
    verbose: bool,
) -> Optional[dict]:
    X_nan, y_nan, _, ids_nan = _met_features_all(ds, day, malate_mode="nan")
    X_raw, y_raw, _, ids_raw = _met_features_all(ds, day, malate_mode="raw")
    X_morph, y_morph, _, ids_morph = _morph_features_all(ds, morph_df, day)

    if len(X_nan) == 0 and len(X_morph) == 0:
        if verbose:
            print(f"  [{day}] no met or morph data, skipping")
        return None

    feature_sets = []
    if len(X_nan) > 0:
        feature_sets += [("met_nan", X_nan, y_nan, ids_nan),
                         ("met_raw", X_raw, y_raw, ids_raw)]
    if len(X_morph) > 0:
        feature_sets += [("morph", X_morph, y_morph, ids_morph)]

    mod_keys = [f"{p}_{m}" for p, *_ in feature_sets for m in ("lgbm", "logreg")]
    repeat_bas: Dict[str, List[float]] = {k: [] for k in mod_keys}
    repeat_cms: Dict[str, List]        = {k: [] for k in mod_keys}
    repeat_details: List[dict]         = []

    for rep in range(n_repeats):
        rep_seed = SEED + rep * 1000
        outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep_seed)
        oof: Dict[str, np.ndarray] = {k: np.full(len(all_org_ids), np.nan) for k in mod_keys}

        for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(all_org_ids, all_labels)):
            fold_seed = rep_seed + fold_i * 97
            tr_oids = [all_org_ids[i] for i in tr_idx]
            te_oids = [all_org_ids[i] for i in te_idx]

            # Mirror combined_kfold: carve out 15% val from train (for img early-stopping parity)
            sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=fold_seed)
            inner_tr_idx, _ = next(sss.split(tr_oids, all_labels[tr_idx]))
            inner_tr_oids = [tr_oids[i] for i in inner_tr_idx]

            for prefix, X_feat, y_feat, feat_ids in feature_sets:
                X_tr_m, y_tr_m, _ = _filter_fold(X_feat, y_feat, feat_ids, inner_tr_oids)
                X_te_m, y_te_m, valid_te_m = _filter_fold(X_feat, y_feat, feat_ids, te_oids)

                if len(X_tr_m) == 0 or len(X_te_m) == 0:
                    continue

                p_lgbm = _train_lgbm_fold(X_tr_m, y_tr_m, X_te_m, fold_seed)
                if p_lgbm is not None:
                    for oid, prob in zip(valid_te_m, p_lgbm):
                        oof[f"{prefix}_lgbm"][all_org_ids.index(oid)] = prob

                p_logreg = _train_logreg_fold(X_tr_m, y_tr_m, X_te_m, fold_seed)
                if p_logreg is not None:
                    for oid, prob in zip(valid_te_m, p_logreg):
                        oof[f"{prefix}_logreg"][all_org_ids.index(oid)] = prob

            if verbose:
                print(f"  [{day}] rep={rep+1}/{n_repeats}  fold={fold_i+1}/{n_folds}")

        for k in mod_keys:
            valid = ~np.isnan(oof[k])
            if valid.sum() < 2:
                continue
            yt = all_labels[valid]
            yp = (oof[k][valid] >= 0.5).astype(int)
            if len(np.unique(yt)) < 2:
                continue
            repeat_bas[k].append(float(balanced_accuracy_score(yt, yp)))
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            repeat_cms[k].append(cm.tolist())

        repeat_details.append({
            "seed":       rep_seed,
            "org_ids":    all_org_ids,
            "true_labels": all_labels.tolist(),
            "oof_probs":  {k: [None if np.isnan(v) else float(v) for v in oof[k]]
                           for k in mod_keys},
        })

    results = {}
    for k in mod_keys:
        bas = repeat_bas[k]
        if bas:
            results[k] = {
                "balanced_accuracy_mean":     float(np.mean(bas)),
                "balanced_accuracy_std":      float(np.std(bas)),
                "n_repeats":                  len(bas),
                "n_folds":                    n_folds,
                "repeat_balanced_accuracies": bas,
                "repeat_confusion_matrices":  repeat_cms[k],
            }
    if results:
        results["repeat_details"] = repeat_details
    return results or None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",      nargs="+", default=None)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-folds",   type=int, default=N_FOLDS)
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    ds = OrganoidDataset(
        ALL_DATA_PATH, splits=None,
        filters=[*idor_ba1_ba2_filters(), require_complete_series(drop_stitched=False)],
    )
    all_org_ids = [o for o in ds.organoid_ids if ds.organoid_label(o) in LABEL_TO_INT]
    all_labels  = np.array([LABEL_TO_INT[ds.organoid_label(o)] for o in all_org_ids])
    print(f"Organoids: {len(all_org_ids)}  "
          f"({all_labels.sum()} NAcc, {(all_labels==0).sum()} Acc)")
    print(f"Protocol: {args.n_repeats}×{args.n_folds}-fold  SEED={SEED}")

    days = args.days or list(DAY_ORDER)
    out_path = OUTPUT_PATH_ALL

    morph_df = _load_morph_df()

    all_results = {}
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path) as f:
            all_results = json.load(f)
        print(f"Resuming from {out_path}  ({len(all_results)} days already done)")

    for day in days:
        if day in all_results:
            print(f"[{day}] already done, skipping")
            continue
        if day not in ds.days:
            print(f"[{day}] no data in dataset, skipping")
            continue
        print(f"\n[{day}] running {args.n_repeats}×{args.n_folds}-fold ...")
        day_res = run_day(day, ds, morph_df, all_org_ids, all_labels,
                          args.n_folds, args.n_repeats, args.verbose)
        if day_res:
            all_results[day] = day_res
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved → {out_path}")

    print("\n\n=== LGBM vs LogReg comparison (mean BA) ===")
    header = f"{'Day':<10}{'met_lgbm':>12}{'met_logreg':>12}{'morph_lgbm':>12}{'morph_logreg':>12}"
    print(header)
    for day in DAY_ORDER:
        if day not in all_results:
            continue
        r = all_results[day]
        row = f"{day:<10}"
        for k in ["met_nan_lgbm", "met_nan_logreg", "morph_lgbm", "morph_logreg"]:
            v = r.get(k)
            cell = f"{v['balanced_accuracy_mean']:.3f}" if v else "—"
            row += f"{cell:>12}"
        print(row)


if __name__ == "__main__":
    main()
