"""What the model actually sees under each condition (exact model-input preprocessing)."""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import binary_dilation
from torchvision import transforms as T

sys.argv = ["x"]
import analysis.images.cnn_lstm.compute_spatial_occlusion as O
from analysis.images.cnn_lstm.train_base_model import SingleDayOrganoidDataset, TARGET_SIZE
from analysis.images.cnn_lstm.organoid_dataset import load_split_from_json

sns.set_theme(context="paper", style="white")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "axes.titlesize": 15})
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
unnorm = lambda x: np.clip((x.squeeze(0) * STD + MEAN).permute(1, 2, 0).numpy(), 0, 1)

meta = {}
for p in ("train", "val", "test"):
    meta.update(load_split_from_json(f"data/cohorts/idor_main/series/{p}.json")[1])
CASES = [("BA2_96_1_F7_nosplit", "Not Acceptable, small (3.4% of frame)"),
         ("BA2_96_2_B10_nosplit", "Acceptable, large (9.0% of frame)")]
COLS = ["Original\n(normal model input)", "bbox crop\n(size 'removed')", "Shrunk 0.7×\n(½ area)",
        "Enlarged 1.4×\n(2× area)", "Silhouette\n(size + outline only)"]
ZOOM = 48  # half-width (model-input pixels) of the centred zoom window

fig, ax = plt.subplots(4, 5, figsize=(19, 15.5), gridspec_kw={"height_ratios": [1, 1, 1, 1]})
rng = np.random.default_rng(0)
for ci, (oid, desc) in enumerate(CASES):
    tp = O.nearest_tp(meta, oid, 30)
    img = O.load_rgb01(tp["img_paths"]["clipped"])
    m = O.load_mask(tp["mask_paths"]["clipped"], img.shape[:2]); d = binary_dilation(m, iterations=5)
    bbox = SingleDayOrganoidDataset([oid], meta, 30, transform=T.Compose([T.Resize(TARGET_SIZE)]),
                                    image_type="clipped", bbox_crop=True)[0][0]
    views = [unnorm(O.to_input(img)), unnorm(bbox.unsqueeze(0)),
             unnorm(O.to_input(O.edit(img, m, d, "scaled_0.7", rng)[0])),
             unnorm(O.to_input(O.edit(img, m, d, "scaled_1.4", rng)[0])),
             unnorm(O.to_input(O.edit(img, m, d, "silhouette", rng)[0]))]
    for j, v in enumerate(views):
        a = ax[2 * ci, j]; a.imshow(v); a.set_xticks([]); a.set_yticks([])
        # zoom window centred on the organoid's TOP EDGE in this view (boundary
        # detail shows blur / texture scale; the dark interior is featureless)
        dark = np.argwhere(v.mean(axis=2) < 0.45)
        if len(dark):
            cx = int(np.median(dark[:, 1])); col = dark[np.abs(dark[:, 1] - cx) < 3]
            cy = int(col[:, 0].min()) + ZOOM // 3
        else:
            cy, cx = np.array(v.shape[:2]) // 2
        y0, x0 = np.clip(cy - ZOOM, 0, v.shape[0] - 2 * ZOOM), np.clip(cx - ZOOM, 0, v.shape[1] - 2 * ZOOM)
        a.add_patch(plt.Rectangle((x0, y0), 2 * ZOOM, 2 * ZOOM, fill=False, ec="#fc8d62", lw=2))
        z = ax[2 * ci + 1, j]; z.imshow(v[y0:y0 + 2 * ZOOM, x0:x0 + 2 * ZOOM], interpolation="nearest")
        z.set_xticks([]); z.set_yticks([])
        for s in z.spines.values():
            s.set_edgecolor("#fc8d62"); s.set_linewidth(2)
        if ci == 0:
            a.set_title(COLS[j], fontsize=15, fontweight="bold")
    ax[2 * ci, 0].set_ylabel(desc.replace(", ", ",\n"), fontsize=14, fontweight="bold")
    ax[2 * ci + 1, 0].set_ylabel("zoom (same\npixel scale)", fontsize=13)
fig.suptitle("What the model sees (day 30): every panel is the exact 384×512 model input; "
             "zoom boxes are the same size in every panel", fontsize=15, y=0.995)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"explain.{ext}", dpi=200, bbox_inches="tight")
