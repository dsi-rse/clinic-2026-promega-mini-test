import json, sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.io import imread

meta = {}
for p in ("train", "val", "test"):
    meta.update(json.load(open(f"data/cohorts/idor_main/series/{p}.json")))
ids = sys.argv[1].split()
for oid in ids:
    tp = min(meta[oid]["timepoints"], key=lambda t: abs(t["mdl_day"] - 30))
    img = imread(tp["img_paths"]["clipped"]); g = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
    m = np.asarray(Image.open(tp["mask_paths"]["clipped"]).convert("L")) > 127
    core = binary_erosion(m, iterations=4)
    v = g[core]
    q = np.percentile(v, [10, 25, 50, 75, 90, 95])
    print(f"{oid[:20]:20} label={meta[oid]['label'][:3]} bg={g[0,0]:.0f} pctl10-95 {np.round(q).astype(int)}  "
          + "  ".join(f">{t}:{(v > t).mean():.2f}" for t in (80, 100, 120, 140)))
