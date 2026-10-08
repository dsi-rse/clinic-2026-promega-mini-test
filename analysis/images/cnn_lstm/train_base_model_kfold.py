#!/usr/bin/env python3
"""
train_base_model_kfold.py — (repeated) K-fold CV version of the single-image baseline.

Faithful to the coworker's evaluation approach: instead of one train/val/test
split, pool all organoids and run K-fold cross-validation per day. Each fold
trains on the other folds (with a small inner val split for early stopping) and
predicts the held-out fold; per-day metrics are the out-of-fold aggregate plus
the across-fold mean/std. This averages out the single-split checkpoint-selection
noise that collapsed the strong-aug run.

Reuses the model, dataset, training loop, and augmentation from train_base_model
unchanged — only the CV scaffold is new. Folds are well-grouped (StratifiedGroupKFold
on the base well) so daughter organoids never straddle folds (leakage-safe); when
every well has a single organoid, plain StratifiedKFold is used for tighter class balance.

With --n-repeats R the whole K-fold CV is re-run R times with reshuffled folds
(seed SEED + r*1000, as in combined_kfold.py); per-day metrics are mean±std over
repeats, and oof_predictions.csv keeps every repeat's per-organoid probabilities.

    python analysis/images/cnn_lstm/train_base_model_kfold.py \\
        --image-type clipped --strong-aug --n-folds 4 --n-repeats 10 \\
        --splits-dir data/cohorts/idor_main/series \\
        --output-dir .../base_effnet_kfold4x10_strongaug
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.train_base_model import (
    BaselineEfficientNet, SingleDayOrganoidDataset, evaluate, ForegroundColorJitter,
    TARGET_SIZE, DAY_RANGES, BATCH_SIZE, NUM_WORKERS, MAX_EPOCHS, PATIENCE, GRAD_CLIP, LR,
    set_seed, _BOUNDARY_DAYS,
)
from analysis.images.cnn_lstm.organoid_dataset import load_split_from_json

SEED = 1


def _label(meta, oid):
    s = str((meta.get(oid) or {}).get("label", "")).strip().lower()
    return 1 if s in ("good", "acceptable", "accepted") else 0


def _well(oid):
    # base well id: BA1_96_1_B9_nosplit -> BA1_96_1_B9 (groups daughters together)
    return "_".join(oid.split("_")[:4])


def _make_train_tf(target_day, strong_aug):
    if strong_aug:
        degrees = 0 if float(target_day) in _BOUNDARY_DAYS else 180
        return T.Compose([
            T.Resize(TARGET_SIZE),
            T.RandomHorizontalFlip(0.5),
            T.RandomAffine(degrees=degrees, translate=(0.1, 0.1), fill=[178, 178, 178]),
            ForegroundColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        ])
    return T.Compose([
        T.Resize(TARGET_SIZE), T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5), T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    ])


def _train_one(train_ds, val_ds, device, pos_weight, select):
    tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                    pin_memory=(device.type == "cuda"))
    vl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model = BaselineEfficientNet().to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.Adam(model.classifier.parameters(), lr=LR)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    best_score, best_state, bad = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        if epoch == 4:  # biphasic: unfreeze last 2 blocks, drop LR (same as single-split)
            model.unfreeze_backbone()
            opt = optim.Adam(model.parameters(), lr=LR * 0.1)
            sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
        model.train()
        for imgs, labels, _ in tl:
            imgs = imgs.to(device); labels = labels.to(device)
            opt.zero_grad(); logits = model(imgs); loss = crit(logits, labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
        vloss, vacc, vp, vr, vf, vauc, vap, _, _, vbal = evaluate(model, vl, crit, device)
        sch.step(vloss)
        score = vbal if select == "bal" else vacc
        if score > best_score + 1e-4:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state, strict=True)
    return model


@torch.no_grad()
def _predict(model, ds, device):
    model.eval()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    out = {}
    for imgs, labels, ids in loader:
        p = torch.sigmoid(model(imgs.to(device))).cpu().numpy()
        for oid, pr, lab in zip(ids, np.atleast_1d(p), labels.numpy()):
            out[oid] = (float(pr), int(lab))
    return out


def _load_folds_file(path, ids):
    """Per-repeat fold assignments: JSON list with one {organoid_id: fold} dict per repeat.

    Ids may be in any spelling of the base well ("BA1 96_1 A2", "BA1_96_1_A2_nosplit");
    they are matched to the cohort by base well. Every cohort organoid must be assigned.
    """
    with open(path) as f:
        reps = json.load(f)
    by_well = {_well(o): o for o in ids}
    out = []
    for r, assign in enumerate(reps):
        m = {by_well[_well(k.replace(" ", "_"))]: int(v) for k, v in assign.items()
             if _well(k.replace(" ", "_")) in by_well}
        missing = [o for o in ids if o not in m]
        if missing:
            raise SystemExit(f"--folds-file repeat {r + 1}: {len(missing)} cohort organoids "
                             f"unassigned, e.g. {missing[:5]}")
        out.append(m)
    return out


def _fold_splits(ids, y, groups, n_folds, rep, folds_from_file):
    """(train_idx, test_idx) per fold for one repeat. Repeat r reshuffles with seed
    SEED + r*1000 (same scheme as combined_kfold.py)."""
    if folds_from_file is not None:
        assign = folds_from_file[rep]
        fold_of = np.array([assign[o] for o in ids])
        return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0])
                for k in sorted(set(fold_of))]
    seed = SEED + rep * 1000
    if len(set(groups)) == len(ids):
        # One organoid per well: nothing to keep together, and StratifiedGroupKFold
        # stratifies poorly with singleton groups (Not-Acceptable per fold 3..8), so
        # use plain StratifiedKFold (also what combined_kfold.py uses).
        return list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(ids, y))
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(ids, y, groups))


def run_day(day, ids, meta, device, args):
    y = np.array([_label(meta, o) for o in ids])
    groups = np.array([_well(o) for o in ids])
    if len(np.unique(y)) < 2 or np.bincount(y).min() < args.n_folds:
        print(f"  day {day}: too few of a class for {args.n_folds}-fold, skipping")
        return None
    train_tf = _make_train_tf(day, args.strong_aug)
    eval_tf = T.Compose([T.Resize(TARGET_SIZE)])
    day_dir = args.output_dir / f"day_{day}"
    oof_rows, rep_bal, rep_auc, fold_bal = [], [], [], []
    prob_sum, prob_n, true_of = {}, {}, {}
    for rep in range(args.n_repeats):
        rep_seed = SEED + rep * 1000
        oof, oof_fold = {}, {}
        for fi, (tr_idx, te_idx) in enumerate(
                _fold_splits(ids, y, groups, args.n_folds, rep, args.folds_from_file)):
            _run_fold(day, rep, fi, rep_seed, ids, tr_idx, te_idx, meta, train_tf, eval_tf,
                      device, args, day_dir, oof, oof_fold, fold_bal)
        if not oof:
            continue
        yy = [v[1] for v in oof.values()]; probs = [v[0] for v in oof.values()]
        rep_bal.append(balanced_accuracy_score(yy, [1 if p > 0.5 else 0 for p in probs]))
        if len(set(yy)) > 1:
            rep_auc.append(roc_auc_score(yy, probs))
        for oid, (p, lab) in oof.items():
            oof_rows.append([day, rep + 1, oof_fold[oid], oid, lab, f"{p:.6f}"])
            prob_sum[oid] = prob_sum.get(oid, 0.0) + p
            prob_n[oid] = prob_n.get(oid, 0) + 1
            true_of[oid] = lab
        print(f"  day {day} repeat {rep + 1}/{args.n_repeats}: OOF bal_acc {rep_bal[-1]:.3f}")
    if not oof_rows:
        return None
    # Per-organoid OOF probabilities (every repeat), so thresholds can be re-tuned
    # later without retraining.
    day_dir.mkdir(parents=True, exist_ok=True)
    with open(day_dir / "oof_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "repeat", "fold", "organoid_id", "true_label", "prob_acceptable"])
        w.writerows(sorted(oof_rows, key=lambda r: (r[1], r[3])))
    # Mean probability across repeats per organoid (the "smoothed" prediction).
    mo = sorted(prob_sum)
    mp = [prob_sum[o] / prob_n[o] for o in mo]; my = [true_of[o] for o in mo]
    res = {
        "balanced_accuracy": float(np.mean(rep_bal)),          # mean over repeats
        "balanced_accuracy_repeat_std": float(np.std(rep_bal)),
        "balanced_accuracy_mean": float(np.mean(fold_bal)) if fold_bal else float("nan"),
        "balanced_accuracy_std": float(np.std(fold_bal)) if fold_bal else float("nan"),
        "roc_auc": float(np.mean(rep_auc)) if rep_auc else float("nan"),
        "roc_auc_repeat_std": float(np.std(rep_auc)) if rep_auc else float("nan"),
        "balanced_accuracy_meanprob": float(balanced_accuracy_score(my, [1 if p > 0.5 else 0 for p in mp])),
        "n": len(mo), "n_folds": args.n_folds, "n_repeats": len(rep_bal),
    }
    print(f"  day {day}: bal_acc {res['balanced_accuracy']:.3f}±{res['balanced_accuracy_repeat_std']:.3f} "
          f"over {res['n_repeats']} repeats  AUC {res['roc_auc']:.3f}  "
          f"(fold mean {res['balanced_accuracy_mean']:.3f}±{res['balanced_accuracy_std']:.3f}, n={res['n']})")
    return res


def _run_fold(day, rep, fi, rep_seed, ids, tr_idx, te_idx, meta, train_tf, eval_tf,
              device, args, day_dir, oof, oof_fold, fold_bal):
    set_seed(rep_seed + fi)
    tr = [ids[i] for i in tr_idx]; te = [ids[i] for i in te_idx]
    tr_y = [_label(meta, o) for o in tr]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=rep_seed + fi)
    it_idx, iv_idx = next(sss.split(tr, tr_y))
    itr = [tr[i] for i in it_idx]; iv = [tr[i] for i in iv_idx]
    train_ds = SingleDayOrganoidDataset(itr, meta, day, transform=train_tf,
                                        image_type=args.image_type, bbox_crop=args.bbox_crop,
                                       brighten=args.brighten)
    val_ds = SingleDayOrganoidDataset(iv, meta, day, transform=eval_tf,
                                      image_type=args.image_type, bbox_crop=args.bbox_crop,
                                       brighten=args.brighten)
    test_ds = SingleDayOrganoidDataset(te, meta, day, transform=eval_tf,
                                       image_type=args.image_type, bbox_crop=args.bbox_crop,
                                       brighten=args.brighten)
    if len(train_ds) == 0 or len(test_ds) == 0:
        return
    tl_labels = [s["label"] for s in train_ds.samples]
    ng = max(sum(tl_labels), 1); nb = max(len(tl_labels) - sum(tl_labels), 1)
    pw = torch.tensor([(nb / ng) * args.pos_weight_scale], device=device)
    model = _train_one(train_ds, val_ds, device, pw, args.select)
    preds = _predict(model, test_ds, device)
    oof.update(preds)
    oof_fold.update({oid: fi + 1 for oid in preds})
    if args.save_models and (rep + 1) in args.save_model_repeats:
        # rep_<r>/fold_<k>/ holds the fold's model + the held-out ids it never trained on,
        # so Grad-CAM (make_gradcam_rotation_check.py --repeat r --fold k) stays leakage-free.
        fold_dir = day_dir / f"rep_{rep + 1}" / f"fold_{fi + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "target_day": day, "repeat": rep + 1, "fold": fi + 1,
                    "strong_aug": args.strong_aug},
                   fold_dir / f"model_day_{day}.pth")
        with open(fold_dir / "test_ids.json", "w") as f:
            json.dump(sorted(preds), f, indent=2)
    yy = [v[1] for v in preds.values()]; pp = [1 if v[0] > 0.5 else 0 for v in preds.values()]
    if len(set(yy)) > 1:
        fold_bal.append(balanced_accuracy_score(yy, pp))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  day {day} rep {rep + 1} fold {fi + 1}/{args.n_folds}: test n={len(preds)}"
          + (f" bal_acc={fold_bal[-1]:.3f}" if len(set(yy)) > 1 else " (single-class fold)"))

def main():
    ap = argparse.ArgumentParser(description="Repeated K-fold CV single-image baseline")
    ap.add_argument("--image-type", default="clipped", choices=["clipped", "std"])
    ap.add_argument("--splits-dir", default="data/cohorts/idor_balsel/series")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--strong-aug", action="store_true")
    ap.add_argument("--bbox-crop", action="store_true")
    ap.add_argument("--brighten", action="store_true",
                    help="Contrast-stretch each organoid's interior to 0..1 (see SingleDayOrganoidDataset).")
    ap.add_argument("--pos-weight-scale", type=float, default=1.0)
    ap.add_argument("--select", default="bal", choices=["bal", "acc"],
                    help="Inner-val checkpoint metric: bal (balanced acc, default) or acc.")
    ap.add_argument("--n-repeats", type=int, default=1,
                    help="Repeat the K-fold CV this many times, reshuffling folds each time "
                         "(seed SEED + r*1000). Metrics are mean±std over repeats.")
    ap.add_argument("--folds-file", type=Path, default=None,
                    help="JSON list (one {organoid_id: fold} dict per repeat) to use fixed "
                         "fold assignments, e.g. a collaborator's. Overrides --n-folds/--n-repeats.")
    ap.add_argument("--save-models", action="store_true",
                    help="Save each fold's model + held-out ids to day_<d>/rep_<r>/fold_<k>/ "
                         "(needed for Grad-CAM on CV models).")
    ap.add_argument("--save-model-repeats", default=None,
                    help="Comma-separated repeats to save models for (e.g. 1,2). Default: all.")
    ap.add_argument("--days", default=None,
                    help="Comma-separated subset of days to run (e.g. 24,30). Default: all.")
    args = ap.parse_args()
    days = DAY_RANGES
    if args.days:
        wanted = {float(d) for d in args.days.split(",") if d.strip()}
        days = [d for d in DAY_RANGES if float(d) in wanted]
        if len(days) != len(wanted):
            raise SystemExit(f"--days {args.days}: valid days are {DAY_RANGES}")
    if args.strong_aug and args.image_type != "clipped":
        print("[warn] --strong-aug fill=178 assumes --image-type clipped")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   folds: {args.n_folds}   strong_aug: {args.strong_aug}")

    # Pool train+val+test into one organoid set + one meta dict.
    from analysis.images.cnn_lstm.organoid_dataset import resolve_split_path
    meta, ids = {}, []
    for phase in ["train", "val", "test"]:
        p = resolve_split_path(args.splits_dir, phase)
        pids, pmeta = load_split_from_json(p)
        meta.update(pmeta); ids.extend(pids)
    ids = sorted(set(ids))
    y = [_label(meta, o) for o in ids]
    print(f"Pooled organoids: {len(ids)}  ({sum(y)} Acceptable, {len(y)-sum(y)} Not)")

    args.folds_from_file = None
    if args.folds_file:
        args.folds_from_file = _load_folds_file(args.folds_file, ids)
        args.n_repeats = len(args.folds_from_file)
        args.n_folds = len(set(args.folds_from_file[0].values()))
        print(f"Folds from {args.folds_file}: {args.n_repeats} repeats x {args.n_folds} folds")
    args.save_model_repeats = (set(range(1, args.n_repeats + 1)) if not args.save_model_repeats
                               else {int(r) for r in args.save_model_repeats.split(",")})
    print(f"Repeats: {args.n_repeats}   folds: {args.n_folds}"
          + (f"   saving models for repeats {sorted(args.save_model_repeats)}" if args.save_models else ""))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # --days runs (e.g. one Slurm array task per day) get their own results file so
    # parallel tasks sharing an output dir don't overwrite each other.
    out = args.output_dir / ("baseline_kfold_results.json" if not args.days else
                             f"baseline_kfold_results_days_{args.days.replace(',', '_')}.json")
    results = {}
    for day in days:
        print(f"\n{'='*60}\nDAY {day} — {args.n_folds}-fold CV\n{'='*60}")
        r = run_day(day, ids, meta, device, args)
        if r:
            results[str(day)] = r
        # incremental save after every day so a wall-time kill can't wipe the run
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
    print(f"\nSaved {out}")
    print(f"\n{'day':>6} {'bal (repeats)':>16} {'fold mean±std':>16} {'AUC':>6}")
    for day in days:
        r = results.get(str(day))
        if r:
            print(f"{day:>6} {r['balanced_accuracy']:>7.3f}±{r['balanced_accuracy_repeat_std']:<7.3f} "
                  f"{r['balanced_accuracy_mean']:>7.3f}±{r['balanced_accuracy_std']:<7.3f} {r['roc_auc']:>6.3f}")


if __name__ == "__main__":
    main()
