#!/usr/bin/env python3
"""
size_baseline_kfold.py — mask-based size and shape baselines, under the SAME repeated
k-fold CV as train_base_model_kfold.py, for comparison with the image CNNs.

Features (from the clipped mask, the mask the CNN input and occlusion tests use):
    size       log(area as a fraction of the frame)
    shape      scale-free descriptors of the largest component: circularity
               (4*pi*A/P^2), solidity (A/convex hull), eccentricity, aspect ratio,
               log(1 + number of concavities >= CONCAVITY_MIN of the area)
    sizeshape  both
Model: standardised, class-balanced logistic regression fit per fold, so directions
are learned from the training folds only. Shape-only vs size+shape asks whether
shape carries information beyond size (size and shape can co-vary biologically).

Folds come from train_base_model_kfold._fold_splits with the same ids/labels/seeds,
and are checked against each CNN run's oof_predictions.csv, so per-repeat scores are
paired with the CNNs organoid-for-organoid.

    python analysis/images/cnn_lstm/size_baseline_kfold.py --label idor_main \\
        --cnn-subdirs base_effnet_kfold4x10_std base_effnet_kfold4x10_strong \\
        --out .../size_baseline_idor_main_kfold4x10.csv \
        --features-out .../mask_features_idor_main.csv
"""
from __future__ import annotations
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label as cc_label, regionprops
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.train_base_model import DAY_RANGES
from analysis.images.cnn_lstm.train_base_model_kfold import _fold_splits, _label, _well
from analysis.images.cnn_lstm.organoid_dataset import load_split_from_json


def day_str(day: float) -> str:
    return str(int(day)) if float(day) == int(day) else str(day)


CONCAVITY_MIN = 0.02  # concavity (convex hull minus organoid) must be >= 2% of area
SHAPE_COLS = ["circularity", "solidity", "eccentricity", "aspect_ratio", "log1p_concavities"]
FEATURE_SETS = {"size": ["log_area"], "shape": SHAPE_COLS, "sizeshape": ["log_area"] + SHAPE_COLS}


def mask_features(path):
    a = np.asarray(Image.open(path).convert("L")).astype(np.float32)
    m = a > (0.5 * a.max() if a.max() > 0 else 0.5)
    rp = max(regionprops(cc_label(m)), key=lambda r: r.area)
    gaps = regionprops(cc_label(rp.image_convex & ~rp.image))
    return {
        "area_frac": float(m.mean()),
        "log_area": float(np.log(m.mean())),
        "circularity": float(4 * np.pi * rp.area / max(rp.perimeter, 1) ** 2),
        "solidity": float(rp.solidity),
        "eccentricity": float(rp.eccentricity),
        "aspect_ratio": float(rp.major_axis_length / max(rp.minor_axis_length, 1)),
        "log1p_concavities": float(np.log1p(sum(g.area >= CONCAVITY_MIN * rp.area for g in gaps))),
    }


def cnn_repeats(path):
    """{repeat: (fold_of {oid: fold}, oof {oid: (label, prob)})} from a CNN oof CSV."""
    out = defaultdict(lambda: ({}, {}))
    for r in csv.DictReader(open(path)):
        f, o = out[int(r["repeat"])]
        f[r["organoid_id"]] = int(r["fold"])
        o[r["organoid_id"]] = (int(r["true_label"]), float(r["prob_acceptable"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"))
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/net/projects2/promega/project_data/model_tests/lstm_runs"))
    ap.add_argument("--cnn-subdirs", nargs="+", required=True,
                    help="k-fold CNN run dirs under runs_root/label/ to pair with (and check folds against)")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--image-type", default="clipped")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--features-out", type=Path, default=None,
                    help="Optional CSV of per-organoid, per-day mask features (for plots).")
    args = ap.parse_args()

    meta, ids = {}, []
    for phase in ("train", "val", "test"):
        i, m = load_split_from_json(args.cohorts_dir / args.label / "series" / f"{phase}.json")
        meta.update(m); ids += i
    ids = sorted(set(ids))                      # same ordering as train_base_model_kfold
    y = np.array([_label(meta, o) for o in ids])
    groups = np.array([_well(o) for o in ids])

    rows, feat_rows = [], []
    for day in DAY_RANGES:
        feats = []
        for o, lab in zip(ids, y):
            tp = min(meta[o]["timepoints"], key=lambda t: abs(t["mdl_day"] - day))
            feats.append(mask_features(tp["mask_paths"][args.image_type]))
            feat_rows.append({"day": day, "organoid_id": o, "true_label": int(lab), **feats[-1]})
        Xs = {k: np.array([[f[c] for c in cols] for f in feats]) for k, cols in FEATURE_SETS.items()}
        cnn = {s: cnn_repeats(args.runs_root / args.label / s / f"day_{day_str(day)}" / "oof_predictions.csv")
               for s in args.cnn_subdirs}
        for rep in range(args.n_repeats):
            splits = _fold_splits(ids, y, groups, args.n_folds, rep, None)
            fold_of = {ids[i]: k + 1 for k, (_, te) in enumerate(splits) for i in te}
            for s, reps in cnn.items():
                if reps[rep + 1][0] != fold_of:
                    raise SystemExit(f"fold mismatch vs {s} day {day} repeat {rep + 1}")
            row = {"day": day, "repeat": rep + 1}
            for name, X in Xs.items():
                prob = np.empty(len(ids))
                for tr, te in splits:
                    lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
                    prob[te] = lr.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                row[f"{name}_bal_acc"] = balanced_accuracy_score(y, prob > 0.5)
                row[f"{name}_auc"] = roc_auc_score(y, prob)
            for s, reps in cnn.items():
                o = reps[rep + 1][1]
                yy = [o[i][0] for i in ids]; pp = [o[i][1] for i in ids]
                tag = s.rsplit("_", 1)[-1]
                row[f"{tag}_bal_acc"] = balanced_accuracy_score(yy, np.array(pp) > 0.5)
                row[f"{tag}_auc"] = roc_auc_score(yy, pp)
            rows.append(row)
        r = [x for x in rows if x["day"] == day]
        print(f"day {day:>5}: " + "  ".join(f"{k} {np.mean([x[k + '_bal_acc'] for x in r]):.3f}" for k in Xs) + "  | "
              + "  ".join(f"{s.rsplit('_', 1)[-1]} {np.mean([x[s.rsplit('_', 1)[-1] + '_bal_acc'] for x in r]):.3f}"
                          for s in args.cnn_subdirs))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {args.out}")
    if args.features_out:
        with open(args.features_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(feat_rows[0].keys()))
            w.writeheader(); w.writerows(feat_rows)
        print(f"Wrote {len(feat_rows)} rows -> {args.features_out}")


if __name__ == "__main__":
    main()
