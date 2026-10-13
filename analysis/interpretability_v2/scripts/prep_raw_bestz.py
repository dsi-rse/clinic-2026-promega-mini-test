"""Raw best-focus images + masks for the raw-bbox CNN test (V2 cohort).

Skips the processed-image step (575x575 @ 9 um/px, padding, mean fill) but otherwise
follows main's pipeline/images/resize/resize_aspect_ratio.py exactly:
  * pixels: best-Z raw TIFF (all_data aspect_ratio.ar_raw_tif); uint16 -> uint8 by
    /65535*255, uint8 kept as is (same as read_raw_image_as_bgr); grayscale.
  * mask:   the mmseg mask predicted on the std 512x384 image, resized NEAREST back
    to the raw frame (the pipeline's "unstretch" step).
  * scale:  one um/px per day = the coarsest native scale imaged that day (1.6871 for
    days 3-15, 2.4628 for days 17-30), so within a per-day model every image has the same
    physical resolution. Only images at a finer scale are resampled (INTER_AREA for the
    image, NEAREST for the mask).
Writes PNGs to <out>/images, <out>/masks and cohort JSONs (data/cohorts/idor_main/full
plus img_paths/mask_paths["raw"] and raw_um_per_px) to <out>/cohort.

Self-check: for every timepoint the pipeline's 575x575 @ 9 um/px mask and (unfilled)
image are rebuilt from the raw versions and compared with resized_575_square(_masks).

    python analysis/interpretability_v2/scripts/prep_raw_bestz.py <out_dir>
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import tifffile

OUT = Path(sys.argv[1])
(OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "masks").mkdir(exist_ok=True); (OUT / "cohort").mkdir(exist_ok=True)
INTER = "/net/projects2/promega/2026_04_15_data/intermediate"

ad = json.load(open("data/all_data.json")); ad = ad.get("records", ad)
splits = {p: json.load(open(f"data/cohorts/idor_main/full/{p}.json")) for p in ("train", "val", "test")}
tps = [tp for s in splits.values() for v in s.values() for tp in v["timepoints"]]

# per-day target scale = coarsest native um/px that day
um_of = {tp["key"]: float(ad[tp["key"]]["images"]["aspect_ratio"]["ar_orig_um_per_px"]) for tp in tps}
day_um = defaultdict(float)
for tp in tps:
    day_um[float(tp["mdl_day"])] = max(day_um[float(tp["mdl_day"])], um_of[tp["key"]])
print("target um/px by day:", {d: round(u, 4) for d, u in sorted(day_um.items())})


def read_raw_gray(path):
    img = tifffile.imread(path)
    if img.ndim >= 3 and img.shape[0] not in (3, 4) and img.shape[-1] not in (3, 4):
        img = img[0]
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    chan_diff = 0
    if img.ndim == 3:
        img = img[:, :, :3]
        chan_diff = int(np.abs(img.astype(int) - img[:, :, :1].astype(int)).max())
        img = img.mean(axis=2).round().astype(np.uint8)
    return img, chan_diff


def pad_sq(a, t, mode):
    a = a[:t, :t]; h, w = a.shape[:2]; top, left = (t - h) // 2, (t - w) // 2
    pw = ((top, t - h - top), (left, t - w - left)) + (((0, 0),) if a.ndim == 3 else ())
    return np.pad(a, pw, mode="edge") if mode == "edge" else np.pad(a, pw, mode="constant")


stats = defaultdict(list)
new_paths = {}
for i, tp in enumerate(tps):
    ar = ad[tp["key"]]["images"]["aspect_ratio"]
    raw, cd = read_raw_gray(ar["ar_raw_tif"]); H, W = raw.shape
    std_mask = cv2.imread(tp["mask_paths"]["std"], cv2.IMREAD_GRAYSCALE)
    m_raw = cv2.resize(std_mask, (W, H), interpolation=cv2.INTER_NEAREST)       # pipeline "unstretch"

    # --- self-check against the pipeline's 575 @ 9 um/px outputs ---
    um = um_of[tp["key"]]; s9 = um / 9.0
    h9, w9 = int(round(H * s9)), int(round(W * s9))
    m575 = pad_sq(cv2.resize(m_raw, (w9, h9), interpolation=cv2.INTER_NEAREST), 575, "constant")
    ref_m = cv2.imread(tp["mask_paths"]["clipped"], cv2.IMREAD_GRAYSCALE)
    stats["mask575_mismatch_px"].append(int(((m575 > 127) != (ref_m > 127)).sum()))
    ref_i = cv2.imread(f"{INTER}/resized_575_square/{tp['main_id']}.png", cv2.IMREAD_GRAYSCALE)
    if ref_i is not None:
        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        i575 = pad_sq(cv2.resize(bgr, (w9, h9), interpolation=cv2.INTER_LINEAR), 575, "edge")[:, :, 0]
        stats["img575_mean_abs_diff"].append(float(np.abs(i575.astype(float) - ref_i.astype(float)).mean()))
    stats["rgb_channel_maxdiff"].append(cd)

    # --- raw output at the day's common scale ---
    tgt = day_um[float(tp["mdl_day"])]; s = um / tgt
    if abs(s - 1) > 1e-6:
        size = (int(round(W * s)), int(round(H * s)))
        img_o = cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
        m_o = cv2.resize(m_raw, size, interpolation=cv2.INTER_NEAREST)
        stats["resampled"].append(tp["key"])
    else:
        img_o, m_o = raw, m_raw
    ip, mp = OUT / "images" / f"{tp['main_id']}.png", OUT / "masks" / f"{tp['main_id']}.png"
    cv2.imwrite(str(ip), img_o); cv2.imwrite(str(mp), np.where(m_o > 127, 255, 0).astype(np.uint8))
    stats["organoid_px_diam"].append((float(tp["mdl_day"]), 2 * np.sqrt((m_o > 127).sum() / np.pi)))
    new_paths[tp["key"]] = (str(ip), str(mp), tgt)
    if i % 200 == 0:
        print(f"{i}/{len(tps)}", flush=True)

for p, s in splits.items():
    for v in s.values():
        for tp in v["timepoints"]:
            ip, mp, tgt = new_paths[tp["key"]]
            tp["img_paths"]["raw"] = ip; tp["mask_paths"]["raw"] = mp
            tp.setdefault("um_per_px", {})["raw"] = tgt
    json.dump(s, open(OUT / "cohort" / f"{p}.json", "w"), indent=1)

mm = np.array(stats["mask575_mismatch_px"]); dd = np.array(stats["img575_mean_abs_diff"])
print(f"\nSELF-CHECK mask -> 575: exact match {int((mm == 0).sum())}/{len(mm)}, max mismatching px {mm.max()}")
print(f"SELF-CHECK image -> 575 (vs resized_575_square): n={len(dd)}, mean abs diff median {np.median(dd):.3f}, max {dd.max():.3f} (0-255)")
print(f"RGB stitched images: max channel difference {max(stats['rgb_channel_maxdiff'])}")
print(f"resampled to the day's scale: {len(stats['resampled'])} images")
dm = defaultdict(list)
for d, x in stats["organoid_px_diam"]:
    dm[d].append(x)
print("organoid width in raw output (median px):", {d: int(np.median(v)) for d, v in sorted(dm.items())})
