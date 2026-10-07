#!/usr/bin/env python3
"""
compute_bbox_sharpness.py — does the bbox-crop model infer size from sharpness?

bbox crop stretches every organoid to fill the frame, so a small organoid (few
source pixels) is magnified more and comes out blurrier than a large one. For
each organoid, scored by the k-fold bbox model that HELD IT OUT, this records:

    area_frac       organoid area as a fraction of the native frame
    crop_h          native height of the letterboxed bbox crop (stretch = 384/crop_h)
    sharp_bbox      Laplacian variance inside the organoid in the bbox model input
    sharp_orig      the same in the normal (uncropped) model input, for contrast
    p_orig          bbox-model P(Acceptable) on the real bbox input
    p_degraded      ... after re-rendering at the median Not-Acceptable crop size
                    for that day (downsample by target/crop_h, then back up):
                    same shape, a small organoid's resolution. Only organoids
                    larger than the target are degraded; others keep p_orig.
    sharp_degraded  sharpness of that degraded input

If p_degraded < p_orig for large organoids, the bbox model reads sharpness as a
size cue; if it barely moves, it is using shape/texture instead.

    python analysis/images/cnn_lstm/compute_bbox_sharpness.py --label idor_main \\
        --model-subdir base_effnet_kfold4x10_bboxstd --repeat 1 --all-days \\
        --out .../bbox_sharpness_idor_main_rep1.csv
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion, laplace
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.train_base_model import (
    BaselineEfficientNet, SingleDayOrganoidDataset, TARGET_SIZE, DAY_RANGES,
)
from analysis.images.cnn_lstm.organoid_dataset import load_split_from_json

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
BBOX_PAD = 10  # SingleDayOrganoidDataset default


def day_str(day: float) -> str:
    return str(int(day)) if float(day) == int(day) else str(day)


def native_mask(path):
    a = np.asarray(Image.open(path).convert("L")).astype(np.float32)
    return a > (0.5 * a.max() if a.max() > 0 else 0.5)


def crop_height(mask):
    """Native height of the letterboxed bbox crop, mirroring SingleDayOrganoidDataset."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    ch = min(h, ys.max() + BBOX_PAD) - max(0, ys.min() - BBOX_PAD)
    cw = min(w, xs.max() + BBOX_PAD) - max(0, xs.min() - BBOX_PAD)
    return max(ch, int(round(cw * TARGET_SIZE[0] / TARGET_SIZE[1])))


def sharpness(x):
    """Laplacian variance inside the organoid (dark pixels, eroded so the
    organoid/background edge does not dominate) of a normalised model input."""
    g = (x * STD + MEAN).mean(dim=0).numpy()
    org = binary_erosion(g < 0.45, iterations=4)
    return float(laplace(g)[org].var()) if org.sum() > 50 else float("nan")


def degrade(x, factor):
    """Re-render at `factor` of the source resolution: down then back up."""
    h, w = x.shape[-2:]
    small = F.interpolate(x.unsqueeze(0), size=(max(8, round(h * factor)), max(8, round(w * factor))),
                          mode="bilinear", align_corners=False, antialias=True)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)


@torch.no_grad()
def process_day(day, meta, args, device):
    rep_dir = args.runs_root / args.label / args.model_subdir / f"day_{day_str(day)}" / f"rep_{args.repeat}"
    folds = sorted(rep_dir.glob("fold_*"))
    if not folds:
        print(f"  [skip] no fold models: {rep_dir}"); return []
    tf = T.Compose([T.Resize(TARGET_SIZE)])
    tp_of = lambda o: min(meta[o]["timepoints"], key=lambda t: abs(t["mdl_day"] - day))
    all_ids = [o for fd in folds for o in json.load(open(fd / "test_ids.json"))]
    crop_h = {o: crop_height(native_mask(tp_of(o)["mask_paths"][args.image_type])) for o in all_ids}
    lab = {o: 1 if str(meta[o].get("label", "")).strip().lower() in ("good", "acceptable", "accepted") else 0
           for o in all_ids}
    target = float(np.median([crop_h[o] for o in all_ids if lab[o] == 0]))
    rows = []
    for fd in folds:
        state = torch.load(fd / f"model_day_{day_str(day)}.pth", map_location=device)
        model = BaselineEfficientNet().to(device)
        model.load_state_dict(state.get("state_dict", state), strict=True); model.eval()
        ids = json.load(open(fd / "test_ids.json"))
        bb = SingleDayOrganoidDataset(ids, meta, day, transform=tf, image_type=args.image_type, bbox_crop=True)
        og = SingleDayOrganoidDataset(ids, meta, day, transform=tf, image_type=args.image_type, bbox_crop=False)
        orig_x = {og[i][2]: og[i][0] for i in range(len(og))}
        for i in range(len(bb)):
            x, _, oid = bb[i]
            p0 = torch.sigmoid(model(x.unsqueeze(0).to(device))).item()
            factor = min(1.0, target / crop_h[oid])
            xd = degrade(x, factor) if factor < 1.0 else x
            pd = torch.sigmoid(model(xd.unsqueeze(0).to(device))).item() if factor < 1.0 else p0
            rows.append({"day": day, "fold": int(fd.name.split("_")[1]), "organoid_id": oid, "true_label": lab[oid],
                         "area_frac": round(float(native_mask(tp_of(oid)["mask_paths"][args.image_type]).mean()), 5),
                         "crop_h": crop_h[oid], "target_crop_h": target, "degrade_factor": round(factor, 4),
                         "sharp_bbox": sharpness(x), "sharp_orig": sharpness(orig_x[oid]),
                         "sharp_degraded": sharpness(xd), "p_orig": round(p0, 6), "p_degraded": round(pd, 6)})
        del model
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model-subdir", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--day", type=float, default=30)
    ap.add_argument("--all-days", action="store_true")
    ap.add_argument("--image-type", default="clipped", choices=["clipped", "std"])
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/net/projects2/promega/project_data/model_tests/lstm_runs"))
    ap.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = {}
    for phase in ("train", "val", "test"):
        meta.update(load_split_from_json(args.cohorts_dir / args.label / "series" / f"{phase}.json")[1])
    rows = []
    for day in (list(DAY_RANGES) if args.all_days else [args.day]):
        r = process_day(day, meta, args, device)
        rows += r
        if r:
            big = [x for x in r if x["degrade_factor"] < 1]
            print(f"day {day}: n={len(r)}  degraded {len(big)}  mean dP {np.mean([x['p_degraded'] - x['p_orig'] for x in big]):+.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
