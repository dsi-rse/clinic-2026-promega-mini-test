"""Texture on contrast-stretched organoid interiors + brightened example montage."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_erosion, uniform_filter, laplace
from skimage.exposure import equalize_adapthist
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.io import imread

meta = {}
for p in ("train", "val", "test"):
    meta.update(json.load(open(f"data/cohorts/idor_main/series/{p}.json")))


def stretched(oid, day):
    tp = min(meta[oid]["timepoints"], key=lambda t: abs(t["mdl_day"] - day))
    img = imread(tp["img_paths"]["clipped"]); g = (img.mean(axis=2) if img.ndim == 3 else img).astype(np.float32)
    m = np.asarray(Image.open(tp["mask_paths"]["clipped"]).convert("L")) > 127
    inner = binary_erosion(m, iterations=6)
    lo, hi = np.percentile(g[inner], [1, 99]) if inner.sum() > 100 else (0, 255)
    s = np.clip((g - lo) / max(hi - lo, 1), 0, 1)       # each organoid's own dark range -> 0..1
    return s, m, inner


def tex(oid, day):
    s, m, inner = stretched(oid, day)
    if inner.sum() < 200:
        return None
    ys, xs = np.nonzero(inner); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c, cm = s[y0:y1, x0:x1], inner[y0:y1, x0:x1]
    q = (c * 63).astype(np.uint8) + 1; q[~cm] = 0                      # 64 levels, 0 = outside
    G = graycomatrix(q, [1, 3, 6], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=65, symmetric=True, normed=False).astype(float)
    G[0, :] = 0; G[:, 0] = 0; G /= G.sum(axis=(0, 1), keepdims=True) + 1e-12
    lbp = local_binary_pattern((c * 255).astype(np.uint8), 8, 2, "uniform")[cm]
    h = np.bincount(lbp.astype(int), minlength=10) / lbp.size
    locsd = np.sqrt(np.maximum(uniform_filter(c ** 2, 7) - uniform_filter(c, 7) ** 2, 0))[cm]
    out = {"organoid_id": oid, "day": day, "label": meta[oid]["label"], "log_area": float(np.log(m.mean()))}
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        v = graycoprops(G, prop)
        out[f"glcm_{prop}_fine"], out[f"glcm_{prop}_coarse"] = float(v[0].mean()), float(v[2].mean())
    out.update({"lbp_entropy": float(-(h[h > 0] * np.log(h[h > 0])).sum()), "lbp_flat_frac": float(h[8] + h[0]),
                "local_sd_mean": float(locsd.mean()), "local_sd_var": float(locsd.var()),
                "laplacian_var": float(laplace(c)[cm].var()), "interior_skew": float(((c[cm] - c[cm].mean()) ** 3).mean() / (c[cm].std() ** 3 + 1e-9))})
    return out


rows = [r for oid in meta for d in (10, 17, 24, 30) if (r := tex(oid, d))]
pd.DataFrame(rows).to_csv("texture_bright.csv", index=False)

# montage: day 24, same-size good vs bad, brightened interiors
f = pd.DataFrame(rows); f24 = f[f.day == 24].set_index("organoid_id")
acc5 = [o for o in f24.index if meta[o]["label"] == "Acceptable" and meta[o]["n_votes_good"] == 5]
nots = [o for o in f24.index if meta[o]["label"] != "Acceptable"]
pairs = []
for n in sorted(nots, key=lambda o: -f24.loc[o, "log_area"]):
    a = min([x for x in acc5 if x not in [p[0] for p in pairs]], key=lambda x: abs(f24.loc[x, "log_area"] - f24.loc[n, "log_area"]))
    if abs(f24.loc[a, "log_area"] - f24.loc[n, "log_area"]) < np.log(1.15):
        pairs.append((a, n))
    if len(pairs) == 4:
        break
fig, ax = plt.subplots(2, 8, figsize=(20, 5.6))
for k, (a, n) in enumerate(pairs):
    for r, oid in enumerate((a, n)):
        s, m, _ = stretched(oid, 24)
        ys, xs = np.nonzero(m); pad = 10
        crop = s[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]
        crop_m = m[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]
        clahe = equalize_adapthist(crop, clip_limit=.02)
        show = np.where(crop_m, clahe, .85)
        col = "#5b72b0" if r == 0 else "#e5603b"
        for j, (im, t) in enumerate(((crop, "stretched"), (show, "stretched + local contrast"))):
            axx = ax[r, 2 * k + j]; axx.imshow(im, cmap="gray", vmin=0, vmax=1); axx.set_xticks([]); axx.set_yticks([])
            axx.set_title((f"{'GOOD' if r == 0 else 'BAD'} {oid.replace('_nosplit','')} ({meta[oid]['n_votes_good']}/5)\n" if j == 0 else "\n") + t,
                          fontsize=9.5, color=col)
            for sp in axx.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(1.5)
fig.suptitle("Day 24, same-size pairs (good top, bad bottom): organoid interiors brightened to show texture", fontsize=13, y=1.02)
fig.tight_layout(); fig.savefig("tex.png", dpi=120, bbox_inches="tight"); fig.savefig("tex.pdf", bbox_inches="tight")
print("pairs", pairs, "rows", len(rows))
