#!/usr/bin/env python3
"""
train_temporal_lstm_kfold.py — repeated k-fold CV for the EfficientNet+LSTM model.

Wraps train_temporal_ablation_lstm.train_for_day_range UNCHANGED (same model,
augmentation, loss, unfreezing and early stopping) in the SAME folds as
train_base_model_kfold.py (via its _fold_splits: same ids, labels and seeds), so the
LSTM is paired organoid-for-organoid with the single-image CNNs and the size
baselines. Each fold trains on the other folds (with a stratified 15% inner val
split for early stopping) and its best checkpoint scores the held-out fold.

One window (days 3..--max-day) per call. --repeats picks which CV repeats to run so
a Slurm array can split windows x repeats into short tasks; each repeat writes its
own oof_rep<r>.csv under days_3-<max_day>/.

    python analysis/images/cnn_lstm/train_temporal_lstm_kfold.py --max-day 17 \\
        --repeats 1 --splits-dir data/cohorts/idor_main/series \\
        --output-dir .../temporal_lstm_kfold4x10
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.train_temporal_ablation_lstm import (
    OrganoidCNN_LSTM, train_for_day_range, set_seed, BATCH_SIZE, NUM_WORKERS,
)
from analysis.images.cnn_lstm.train_base_model_kfold import SEED, _fold_splits, _label, _well
from analysis.images.cnn_lstm.organoid_dataset import (
    OrganoidTimeSeriesDataset, load_split_from_json, resolve_split_path,
)


@torch.no_grad()
def predict(model_path, ids, meta, max_day, image_type, readout, device):
    ckpt = torch.load(model_path, map_location=device)
    model = OrganoidCNN_LSTM(readout=readout).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True); model.eval()
    eval_tf = transforms.Compose([transforms.Resize((384, 384), interpolation=InterpolationMode.BILINEAR)])
    ds = OrganoidTimeSeriesDataset(ids, meta, max_day=max_day, transform=eval_tf, image_type=image_type)
    out = {}
    for seqs, days, labels, _, oids in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS):
        p = torch.sigmoid(model(seqs.to(device), days.to(device).float())).cpu().numpy()
        for oid, pr, lab in zip(oids, np.atleast_1d(p), labels.numpy()):
            out[oid] = (float(pr), int(lab))
    del model
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-day", type=float, required=True, help="Window is days 3..max-day.")
    ap.add_argument("--splits-dir", default="data/cohorts/idor_main/series")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--image-type", default="clipped", choices=["clipped", "std"])
    ap.add_argument("--readout", default="last", choices=["last", "mean"])
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--repeats", default="1",
                    help="Comma-separated 1-based CV repeats to run (e.g. 1 or 1,2,3).")
    ap.add_argument("--keep-models", action="store_true",
                    help="Keep each fold's checkpoint (default: delete after scoring).")
    args = ap.parse_args()

    max_day = int(args.max_day) if float(args.max_day).is_integer() else args.max_day
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, ids = {}, []
    for phase in ("train", "val", "test"):
        i, m = load_split_from_json(resolve_split_path(args.splits_dir, phase))
        meta.update(m); ids += i
    ids = sorted(set(ids))                              # same ordering as train_base_model_kfold
    y = np.array([_label(meta, o) for o in ids])
    groups = np.array([_well(o) for o in ids])
    win_dir = args.output_dir / f"days_3-{max_day}"
    print(f"Device {device} | window days 3-{max_day} | {len(ids)} organoids "
          f"({y.sum()} Acc, {len(y) - y.sum()} Not) | readout {args.readout}")

    for rep in [int(r) - 1 for r in args.repeats.split(",")]:
        rep_seed = SEED + rep * 1000
        oof, fold_of = {}, {}
        for fi, (tr_idx, te_idx) in enumerate(_fold_splits(ids, y, groups, args.n_folds, rep, None)):
            set_seed(rep_seed + fi)
            tr = [ids[i] for i in tr_idx]; te = [ids[i] for i in te_idx]
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=rep_seed + fi)
            it_idx, iv_idx = next(sss.split(tr, [_label(meta, o) for o in tr]))
            fold_dir = win_dir / f"rep_{rep + 1}" / f"fold_{fi + 1}"
            res = train_for_day_range(max_day, [tr[i] for i in it_idx], [tr[i] for i in iv_idx], te,
                                      meta, meta, meta, device, fold_dir,
                                      image_type=args.image_type, readout=args.readout)
            preds = predict(Path(res["model_path"]), te, meta, max_day, args.image_type, args.readout, device)
            oof.update(preds); fold_of.update({o: fi + 1 for o in preds})
            with open(fold_dir / "test_ids.json", "w") as f:
                json.dump(sorted(preds), f, indent=2)
            if not args.keep_models:
                Path(res["model_path"]).unlink(missing_ok=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        yy = [oof[o][1] for o in sorted(oof)]; pp = [oof[o][0] for o in sorted(oof)]
        with open(win_dir / f"oof_rep{rep + 1}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["max_day", "repeat", "fold", "organoid_id", "true_label", "prob_acceptable"])
            for o in sorted(oof):
                w.writerow([max_day, rep + 1, fold_of[o], o, oof[o][1], f"{oof[o][0]:.6f}"])
        print(f"\n== days 3-{max_day} repeat {rep + 1}: OOF bal_acc "
              f"{balanced_accuracy_score(yy, np.array(pp) > 0.5):.3f}  AUC {roc_auc_score(yy, pp):.3f} "
              f"(n={len(oof)})")


if __name__ == "__main__":
    main()
