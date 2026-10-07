#!/usr/bin/env python3
"""
size_baseline_kfold.py — organoid size alone, under the SAME repeated k-fold CV as
train_base_model_kfold.py, as a baseline for the image CNNs.

Feature: organoid area as a fraction of the frame (from the clipped mask, the mask
the CNN input and occlusion tests use). Model: class-balanced logistic regression on
log(area), fit per fold, so the direction (bigger or smaller = Acceptable) is learned
from the training folds only.

Folds come from train_base_model_kfold._fold_splits with the same ids/labels/seeds,
and are checked against each CNN run's oof_predictions.csv, so per-repeat scores are
paired with the CNNs organoid-for-organoid.

    python analysis/images/cnn_lstm/size_baseline_kfold.py --label idor_main \\
        --cnn-subdirs base_effnet_kfold4x10_std base_effnet_kfold4x10_strong \\
        --out .../size_baseline_idor_main_kfold4x10.csv
"""
from __future__ import annotations
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.train_base_model import DAY_RANGES
from analysis.images.cnn_lstm.train_base_model_kfold import _fold_splits, _label, _well
from analysis.images.cnn_lstm.organoid_dataset import load_split_from_json


def day_str(day: float) -> str:
    return str(int(day)) if float(day) == int(day) else str(day)


def mask_area(path):
    a = np.asarray(Image.open(path).convert("L")).astype(np.float32)
    return float((a > (0.5 * a.max() if a.max() > 0 else 0.5)).mean())


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
    args = ap.parse_args()

    meta, ids = {}, []
    for phase in ("train", "val", "test"):
        i, m = load_split_from_json(args.cohorts_dir / args.label / "series" / f"{phase}.json")
        meta.update(m); ids += i
    ids = sorted(set(ids))                      # same ordering as train_base_model_kfold
    y = np.array([_label(meta, o) for o in ids])
    groups = np.array([_well(o) for o in ids])

    rows = []
    for day in DAY_RANGES:
        area = []
        for o in ids:
            tp = min(meta[o]["timepoints"], key=lambda t: abs(t["mdl_day"] - day))
            area.append(mask_area(tp["mask_paths"][args.image_type]))
        X = np.log(np.array(area)).reshape(-1, 1)
        cnn = {s: cnn_repeats(args.runs_root / args.label / s / f"day_{day_str(day)}" / "oof_predictions.csv")
               for s in args.cnn_subdirs}
        for rep in range(args.n_repeats):
            splits = _fold_splits(ids, y, groups, args.n_folds, rep, None)
            fold_of = {ids[i]: k + 1 for k, (_, te) in enumerate(splits) for i in te}
            for s, reps in cnn.items():
                if reps[rep + 1][0] != fold_of:
                    raise SystemExit(f"fold mismatch vs {s} day {day} repeat {rep + 1}")
            prob = np.empty(len(ids))
            for tr, te in splits:
                lr = LogisticRegression(class_weight="balanced").fit(X[tr], y[tr])
                prob[te] = lr.predict_proba(X[te])[:, 1]
            row = {"day": day, "repeat": rep + 1,
                   "size_bal_acc": balanced_accuracy_score(y, prob > 0.5),
                   "size_auc": roc_auc_score(y, prob)}
            for s, reps in cnn.items():
                o = reps[rep + 1][1]
                yy = [o[i][0] for i in ids]; pp = [o[i][1] for i in ids]
                tag = s.rsplit("_", 1)[-1]
                row[f"{tag}_bal_acc"] = balanced_accuracy_score(yy, np.array(pp) > 0.5)
                row[f"{tag}_auc"] = roc_auc_score(yy, pp)
            rows.append(row)
        r = [x for x in rows if x["day"] == day]
        print(f"day {day:>5}: size bal_acc {np.mean([x['size_bal_acc'] for x in r]):.3f}"
              f"±{np.std([x['size_bal_acc'] for x in r]):.3f}  AUC {np.mean([x['size_auc'] for x in r]):.3f}  | "
              + "  ".join(f"{s.rsplit('_', 1)[-1]} {np.mean([x[s.rsplit('_', 1)[-1] + '_bal_acc'] for x in r]):.3f}"
                          for s in args.cnn_subdirs))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
