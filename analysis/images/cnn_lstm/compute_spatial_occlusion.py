#!/usr/bin/env python3
"""
compute_spatial_occlusion.py — what in the image does the single-image model rely on?

For each organoid, re-scores the k-fold base model (train_base_model_kfold.py
--save-models) that HELD IT OUT on spatially edited versions of the same image:

    original              unchanged
    no_organoid           organoid (dilated mask) filled with background gray:
                          a blank frame, i.e. the model's prior
    silhouette            organoid filled with its own mean colour: keeps size +
                          outline, removes all internal texture
    rim_texture_removed   outer rim (distance-to-edge < RIM_FRAC of the organoid's
                          max inner distance) filled with the organoid mean colour
    core_texture_removed  the rest (core) filled with the organoid mean colour
    shifted               organoid moved to a random spot in the frame

The clipped (mean_fill_clip) images already have every non-organoid pixel set to
the background gray (178), so the model can only use the organoid itself plus its
size and position; these conditions ask WHICH organoid properties it uses.
Grad-CAM shows where attribution lands; occlusion tests whether the model NEEDS
it. Clipped images only. Preprocessing mirrors SingleDayOrganoidDataset
(no bbox crop): [0,1] float -> PIL -> Resize(TARGET_SIZE) -> ImageNet normalise.

Writes one row per organoid x condition (prob_acceptable); summarise downstream.

    python analysis/images/cnn_lstm/compute_spatial_occlusion.py --label idor_main \\
        --model-subdir base_effnet_kfold4x10_std --repeat 1 --all-days \\
        --out .../occlusion_spatial_idor_main_kfold4x10_std_rep1.csv
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.io import imread
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.train_base_model import (
    BaselineEfficientNet, TARGET_SIZE, DAY_RANGES, BG_FILL_U8,
)
from analysis.images.cnn_lstm.organoid_dataset import load_split_from_json

CONDITIONS = ["original", "no_organoid", "silhouette", "rim_texture_removed",
              "core_texture_removed", "shifted"]
RIM_FRAC = 0.25   # rim = outer quarter of the organoid by distance to its edge
MIN_SHIFT = 0.15  # shifted: move at least this fraction of the frame size
BG = BG_FILL_U8.astype(np.float32) / 255.0
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def day_str(day: float) -> str:
    return str(int(day)) if float(day) == int(day) else str(day)


def load_rgb01(path):
    img = imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img[..., :3].astype(np.float32) / 255.0


def load_mask(path, hw):
    m = Image.open(path).convert("L").resize((hw[1], hw[0]), Image.NEAREST)
    a = np.asarray(m).astype(np.float32)
    return a > (0.5 * a.max() if a.max() > 0 else 0.5)


def shift_organoid(img, mask, rng, tries=200):
    """Paste the organoid onto a blank frame at a random offset (>= MIN_SHIFT of the
    frame), keeping it fully inside the image."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    for _ in range(tries):
        dy = int(rng.integers(-y0, h - y1)); dx = int(rng.integers(-x0, w - x1))
        if np.hypot(dy / h, dx / w) < MIN_SHIFT:
            continue
        out = np.empty_like(img); out[:] = BG
        out[ys + dy, xs + dx] = img[ys, xs]
        return out
    return None


def edit(img, mask, dil, cond, rng):
    """Return (edited image, ok). Removal/moving uses the dilated mask `dil` so no
    halo is left behind; texture fills use the exact `mask` so size, outline and
    mean colour stay true."""
    out = img.copy()
    mean = img[mask].mean(axis=0)
    if cond == "original":
        return out, True
    if cond == "no_organoid":
        out[dil] = BG; return out, True
    if cond == "silhouette":
        out[mask] = mean; return out, True
    if cond in ("rim_texture_removed", "core_texture_removed"):
        dist = distance_transform_edt(mask)
        rim = mask & (dist <= RIM_FRAC * dist.max())
        out[rim if cond == "rim_texture_removed" else (mask & ~rim)] = mean
        return out, True
    if cond == "shifted":
        moved = shift_organoid(img, dil, rng)
        return (moved, True) if moved is not None else (out, False)
    raise ValueError(cond)


def to_input(img01):
    pil = Image.fromarray((np.clip(img01, 0, 1) * 255).astype(np.uint8))
    a = np.asarray(T.Resize(TARGET_SIZE)(pil)).astype(np.float32) / 255.0
    x = torch.from_numpy(np.transpose(a, (2, 0, 1))).float()
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)


def nearest_tp(meta, oid, day):
    tps = (meta.get(oid) or {}).get("timepoints") or []
    return min(tps, key=lambda tp: abs(tp["mdl_day"] - day)) if tps else None


@torch.no_grad()
def process_day(day, meta, args, device, rng):
    rep_dir = args.runs_root / args.label / args.model_subdir / f"day_{day_str(day)}" / f"rep_{args.repeat}"
    fold_dirs = sorted(rep_dir.glob("fold_*"))
    if not fold_dirs:
        print(f"  [skip] no fold models for day {day}: {rep_dir}")
        return []
    rows = []
    for fd in fold_dirs:
        state = torch.load(fd / f"model_day_{day_str(day)}.pth", map_location=device)
        model = BaselineEfficientNet().to(device)
        model.load_state_dict(state.get("state_dict", state), strict=True); model.eval()
        fold = int(fd.name.split("_")[1])
        for oid in json.load(open(fd / "test_ids.json")):
            tp = nearest_tp(meta, oid, day)
            img_p = tp and tp.get("img_paths", {}).get(args.image_type)
            msk_p = tp and tp.get("mask_paths", {}).get(args.image_type)
            if not img_p or not msk_p:
                print(f"  [warn] missing image/mask: {oid} day {day}"); continue
            img = load_rgb01(img_p)
            mask = load_mask(msk_p, img.shape[:2])
            if not mask.any():
                print(f"  [warn] empty mask: {oid} day {day}"); continue
            dil = binary_dilation(mask, iterations=args.dilate) if args.dilate else mask
            lab = 1 if str(meta[oid].get("label", "")).strip().lower() in ("good", "acceptable", "accepted") else 0
            for cond in CONDITIONS:
                e, ok = edit(img, mask, dil, cond, rng)
                p = torch.sigmoid(model(to_input(e).to(device))).item() if ok else float("nan")
                rows.append({"day": day, "fold": fold, "organoid_id": oid, "true_label": lab,
                             "condition": cond, "prob_acceptable": round(p, 6),
                             "mask_frac": round(float(mask.mean()), 5)})
        del model
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="cohort label, e.g. idor_main")
    ap.add_argument("--model-subdir", required=True, help="e.g. base_effnet_kfold4x10_std")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--day", type=float, default=30)
    ap.add_argument("--all-days", action="store_true")
    ap.add_argument("--image-type", default="clipped", choices=["clipped"],
                    help="Fill gray (178) is the clipped background, so clipped only.")
    ap.add_argument("--dilate", type=int, default=5,
                    help="Mask dilation in native pixels, so the organoid's rim/halo is "
                         "removed with it (default 5).")
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/net/projects2/promega/project_data/model_tests/lstm_runs"))
    ap.add_argument("--cohorts-dir", type=Path, default=Path("data/cohorts"))
    ap.add_argument("--seed", type=int, default=0, help="Seed for the shifted condition's placement.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    meta = {}
    for phase in ("train", "val", "test"):
        _, m = load_split_from_json(args.cohorts_dir / args.label / "series" / f"{phase}.json")
        meta.update(m)

    rows = []
    for day in (list(DAY_RANGES) if args.all_days else [args.day]):
        r = process_day(day, meta, args, device, rng)
        rows += r
        if r:
            n = len({x["organoid_id"] for x in r})
            mean = {c: np.nanmean([x["prob_acceptable"] for x in r if x["condition"] == c]) for c in CONDITIONS}
            print(f"day {day}: {n} organoids  mean P(Acc) " +
                  "  ".join(f"{c}={v:.3f}" for c, v in mean.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
