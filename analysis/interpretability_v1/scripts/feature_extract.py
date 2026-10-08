"""Interpretable per-organoid, per-day image features (clipped 575x575 images + masks)."""
import json, sys
import numpy as np, pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion, binary_opening, laplace, sobel
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
from skimage.measure import label, regionprops, find_contours

meta = {}
for p in ("train", "val", "test"):
    meta.update(json.load(open(f"data/cohorts/idor_main/series/{p}.json")))


def feats(img_path, mask_path):
    img = imread(img_path); g = (img.mean(axis=2) if img.ndim == 3 else img).astype(np.float32)
    m = np.asarray(Image.open(mask_path).convert("L")) > 127
    lab = label(m); rp = max(regionprops(lab), key=lambda r: r.area); m = lab == rp.label
    core = binary_erosion(m, iterations=10); rim = m & ~binary_erosion(m, iterations=6)
    inner = binary_erosion(m, iterations=4)
    v = g[inner] if inner.sum() > 50 else g[m]
    # boundary roughness: perimeter vs perimeter of a smoothed (opened) mask
    smooth = binary_opening(m, iterations=6); ps = max(regionprops(label(smooth))[0].perimeter, 1) if smooth.any() else rp.perimeter
    # texture (GLCM) on organoid interior, 32 grey levels
    ys, xs = np.nonzero(inner if inner.any() else m)
    crop = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; cm = (inner if inner.any() else m)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    q = np.clip(crop / 256 * 32, 0, 31).astype(np.uint8); q[~cm] = 0
    glcm = graycomatrix(q, [2], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
    glcm[0, :, :, :] = 0; glcm[:, 0, :, :] = 0; glcm = glcm / max(glcm.sum(axis=(0, 1), keepdims=True).min(), 1e-9)
    grad = np.hypot(sobel(g, 0), sobel(g, 1))
    edge = m & ~binary_erosion(m, iterations=2)
    return {
        "log_area": float(np.log(m.mean())),
        # shape (scale-free)
        "circularity": 4 * np.pi * rp.area / rp.perimeter ** 2, "solidity": rp.solidity,
        "elongation": rp.major_axis_length / max(rp.minor_axis_length, 1),
        "n_concavities": sum(r.area >= .02 * rp.area for r in regionprops(label(rp.image_convex & ~rp.image))),
        "boundary_roughness": rp.perimeter / ps,
        # interior darkness / texture
        "mean_intensity": float(v.mean()), "intensity_sd": float(v.std()),
        "translucent_frac": float((v > 100).mean()), "dense_frac": float((v < 20).mean()),
        "texture_sharpness": float(laplace(g)[inner].var()) if inner.sum() > 50 else np.nan,
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()), "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        # edge / rim
        "rim_minus_core": float(g[rim].mean() - (g[core].mean() if core.sum() > 50 else g[m].mean())),
        "rim_sd": float(g[rim].std()),
        "edge_sharpness": float(grad[edge].mean()),
    }


rows = []
for oid, r in meta.items():
    for tp in r["timepoints"]:
        try:
            f = feats(tp["img_paths"]["clipped"], tp["mask_paths"]["clipped"])
        except Exception as e:
            print("fail", oid, tp["mdl_day"], e, file=sys.stderr); continue
        rows.append({"organoid_id": oid, "day": tp["mdl_day"], "label": r["label"], "n_votes_good": r["n_votes_good"], **f})
out = "/net/projects2/promega/project_data/amanda_test/model_plots/feature_screen_idor_main.csv"
pd.DataFrame(rows).to_csv(out, index=False); print("wrote", out, len(rows))
