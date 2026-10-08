"""Day-30 raw-image edge crops: size-matched Acceptable (5/5) vs Not Acceptable pairs."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label
from skimage.filters import threshold_otsu
from skimage.io import imread

feat = pd.read_csv("mask_features_idor_main.csv"); f30 = feat[feat.day == 30].set_index("organoid_id")
meta = {}
for p in ("train", "val", "test"):
    meta.update(json.load(open(f"data/cohorts/idor_main/series/{p}.json")))
ad = json.load(open("data/all_data.json")); ad = ad.get("records", ad)
cyst = {"BA1_96_1_C7_nosplit", "BA2_96_1_B3_nosplit", "BA2_96_1_E3_nosplit", "BA2_96_1_F7_nosplit",
        "BA2_96_2_E12_nosplit", "BA2_96_2_G4_nosplit", "BA2_96_2_G6_nosplit", "BA2_96_2_H2_nosplit"}

acc5 = [o for o in f30.index if meta[o]["label"] == "Acceptable" and meta[o]["n_votes_good"] == 5]
nots = [o for o in f30.index if meta[o]["label"] != "Acceptable" and o not in cyst]
pairs, used = [], set()
for n in sorted(nots, key=lambda o: -f30.loc[o, "area_frac"]):          # biggest Not first: hardest for size
    cand = [a for a in acc5 if a not in used]
    a = min(cand, key=lambda x: abs(np.log(f30.loc[x, "area_frac"] / f30.loc[n, "area_frac"])))
    if abs(np.log(f30.loc[a, "area_frac"] / f30.loc[n, "area_frac"])) <= np.log(1.15):
        pairs.append((a, n)); used.add(a)
    if len(pairs) == 4:
        break


def raw(oid):
    """Raw Z1 image + the organoid's clipped mask mapped back onto it (the clipped image
    is the raw frame centre-padded to a square, then resized to 575)."""
    bw = "_".join(oid.split("_")[:4]); p = bw.split("_")
    key = f"{p[0]} {p[1]}_{p[2]} Dy30 {p[3]}"
    path = ad[key]["images"]["aspect_ratio"]["ar_raw_tif"]
    img = imread(path).astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    lo, hi = np.percentile(img, [0.5, 99.5]); img = np.clip((img - lo) / (hi - lo), 0, 1)
    H, W = img.shape; S = max(H, W)
    tp = min(meta[oid]["timepoints"], key=lambda t: abs(t["mdl_day"] - 30))
    from PIL import Image
    m = np.asarray(Image.open(tp["mask_paths"]["clipped"]).convert("L").resize((S, S), Image.NEAREST)) > 127
    oy, ox = (S - H) // 2, (S - W) // 2
    return img, m[oy:oy + H, ox:ox + W], path.split("/")[-1]


from scipy.ndimage import binary_dilation, uniform_filter
fig, ax = plt.subplots(len(pairs), 4, figsize=(17, 4.3 * len(pairs)), gridspec_kw={"width_ratios": [1.36, 1, 1.36, 1]})
for i, (a, n) in enumerate(pairs):
    for j, oid in enumerate((a, n)):
        img, m, fname = raw(oid)
        dark = img < threshold_otsu(img)
        lab_, nlab = label(dark)
        cy, cx = [int(v.mean()) for v in np.nonzero(m)]          # mapped-mask centre ~ organoid location
        comp = lab_ == lab_[cy, cx] if lab_[cy, cx] else dark
        bright = img > 0.55                                         # real background, not vignette shading
        edge = comp & binary_dilation(bright, iterations=4)        # organoid pixels touching bright background
        ey, ex = np.nonzero(edge)
        # the true-edge point on the organoid's top side (favouring points near its centre column)
        Z = 75
        order = np.argsort(ey + 0.25 * np.abs(ex - cx)) if len(ey) else []
        zy0 = zx0 = None
        for k in order[::25]:                                         # first candidate whose window is ~half organoid, half background
            y0_, x0_ = int(np.clip(ey[k] - Z, 0, img.shape[0] - 2 * Z)), int(np.clip(ex[k] - Z, 0, img.shape[1] - 2 * Z))
            win_org = comp[y0_:y0_ + 2 * Z, x0_:x0_ + 2 * Z].mean(); win_bg = bright[y0_:y0_ + 2 * Z, x0_:x0_ + 2 * Z].mean()
            if win_org > .25 and win_bg > .25:
                zy0, zx0 = y0_, x0_; break
        if zy0 is None:
            zy0, zx0 = int(np.clip(cy - Z, 0, img.shape[0] - 2 * Z)), int(np.clip(cx - Z, 0, img.shape[1] - 2 * Z))
        full = ax[i, 2 * j]; full.imshow(img, cmap="gray", vmin=0, vmax=1)
        full.add_patch(plt.Rectangle((zx0, zy0), 2 * Z, 2 * Z, fill=False, ec="#fc8d62", lw=2))
        z = ax[i, 2 * j + 1]; z.imshow(img[zy0:zy0 + 2 * Z, zx0:zx0 + 2 * Z], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        labl = "ACCEPTABLE" if oid == a else "NOT ACCEPTABLE"
        col = "#5b72b0" if oid == a else "#e5603b"
        full.set_title(f"{labl}  {oid.replace('_nosplit', '')}\n{meta[oid]['n_votes_good']}/{meta[oid]['n_votes_total']} Acc votes,"
                       f" area {f30.loc[oid, 'area_frac']*100:.1f}%", fontsize=11, color=col, fontweight="bold")
        z.set_title("edge zoom (raw, full res.)", fontsize=10, color=col)
        for a_ in (full, z):
            a_.set_xticks([]); a_.set_yticks([])
            for sp in a_.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2)
fig.suptitle("Day 30, raw microscope images (Z1, 16-bit, contrast-stretched): same-size Acceptable vs Not Acceptable",
             fontsize=14, y=1.0)
fig.tight_layout(); fig.savefig("edge.png", dpi=110, bbox_inches="tight"); fig.savefig("edge.pdf", bbox_inches="tight")
print("pairs:", pairs)
