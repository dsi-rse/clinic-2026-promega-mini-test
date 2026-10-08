"""Day-10 translucency: good vs bad organoids with high vs low translucent fraction (zoomed, highlighted)."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.io import imread

df = pd.read_csv("feature_screen.csv"); df["plate"] = df.organoid_id.str.split("_").str[:3].str.join("_")
d10 = df[df.day == 10].copy()
d10["tz"] = d10.groupby("plate").translucent_frac.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
meta = {}
for p in ("train", "val", "test"):
    meta.update(json.load(open(f"data/cohorts/idor_main/series/{p}.json")))
acc, nac = d10[d10.label == "Acceptable"], d10[d10.label != "Acceptable"]
groups = [("GOOD, high translucency", acc.nlargest(4, "tz")), ("GOOD, low translucency", acc.nsmallest(4, "tz")),
          ("BAD, high translucency", nac.nlargest(4, "tz")), ("BAD, low translucency", nac.nsmallest(4, "tz"))]
print("day-10 translucent fraction: good median %.3f, bad median %.3f; >2%%: good %.0f%%, bad %.0f%%" % (
    acc.translucent_frac.median(), nac.translucent_frac.median(), (acc.translucent_frac > .02).mean() * 100, (nac.translucent_frac > .02).mean() * 100))

fig, ax = plt.subplots(4, 8, figsize=(19, 10.5))
for gi, (title, sel) in enumerate(groups):
    col = "#5b72b0" if title.startswith("GOOD") else "#e5603b"
    for k, r in enumerate(sel.itertuples()):
        tp = min(meta[r.organoid_id]["timepoints"], key=lambda t: abs(t["mdl_day"] - 10))
        img = imread(tp["img_paths"]["clipped"]); g = (img.mean(axis=2) if img.ndim == 3 else img).astype(np.float32)
        m = np.asarray(Image.open(tp["mask_paths"]["clipped"]).convert("L")) > 127
        inner = binary_erosion(m, iterations=4); tl = inner & (g > 100)
        ys, xs = np.nonzero(m); pad = 12
        y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
        crop = g[y0:y1, x0:x1] / 255
        a1, a2 = ax[gi, 2 * k], ax[gi, 2 * k + 1]
        a1.imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        ov = np.dstack([crop] * 3); ov[tl[y0:y1, x0:x1]] = [0.2, 0.85, 0.95]
        a2.imshow(ov, interpolation="nearest")
        a1.set_title(f"{r.organoid_id.replace('_nosplit','')}\n{r.n_votes_good}/5 Acc", fontsize=9.5, color=col)
        a2.set_title(f"translucent {r.translucent_frac*100:.1f}%", fontsize=9.5, color=col)
        for a in (a1, a2):
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_edgecolor(col); s.set_linewidth(1.5)
    ax[gi, 0].set_ylabel(title, fontsize=12, fontweight="bold", color=col)
fig.suptitle("Day 10 (zoomed to each organoid): translucent pixels highlighted in cyan; translucency ranked within each plate",
             fontsize=13, y=1.0)
fig.tight_layout(); fig.savefig("d10.png", dpi=120, bbox_inches="tight"); fig.savefig("d10.pdf", bbox_inches="tight")
