"""Montage: bad organoids that growth and every image model miss, next to a size-matched
good organoid (same plate, same day, closest mask area), as the brightened bbox CNN saw
them. Days 24 / 28 / 30.

    python analysis/interpretability_v2/scripts/unexplained_montage.py <area_csv> <out_dir>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.train_base_model import SingleDayOrganoidDataset, input_resize

AREA, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
meta = {}
for p in ("train", "val", "test"):
    meta.update(json.load(open(ROOT / f"data/cohorts/idor_main/full/{p}.json")))
LA = pd.read_csv(AREA).pivot(index="organoid_id", columns="day", values="log_area")
good = {o for o, v in meta.items() if v["label"] == "Acceptable"}
plate = lambda o: "_".join(o.split("_")[:3])
BAD = ["BA1_96_1_A8", "BA2_96_1_F3", "BA2_96_1_D8", "BA2_96_1_G5", "BA2_96_2_G7", "BA2_96_2_H2", "BA2_96_1_A5"]
DAYS = (24, 28, 30)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
tf = T.Compose(input_resize("clipped", True))


def show(oid, d):
    ds = SingleDayOrganoidDataset([oid], meta, d, transform=tf, image_type="clipped", bbox_crop=True,
                                  brighten=True, exact_day=True)
    return None if len(ds) == 0 else np.clip((ds[0][0] * STD + MEAN).permute(1, 2, 0).numpy(), 0, 1)


fig, ax = plt.subplots(len(BAD), 2 * len(DAYS), figsize=(4.2 * 2 * len(DAYS), 3.6 * len(BAD)))
for i, b in enumerate(BAD):
    bo = b + "_nosplit"; v = meta[bo]
    for j, d in enumerate(DAYS):
        a_bad, a_good = ax[i, 2 * j], ax[i, 2 * j + 1]
        img = show(bo, d) if pd.notna(LA.loc[bo].get(d)) else None
        cand = [o for o in good if plate(o) == plate(bo) and pd.notna(LA.loc[o].get(d))] if img is not None else []
        m = min(cand, key=lambda o: abs(LA.loc[o, d] - LA.loc[bo, d])) if cand else None
        for a, o, col, tag in ((a_bad, bo, "#e5603b", "BAD"), (a_good, m, "#5b72b0", "matched good")):
            a.set_xticks([]); a.set_yticks([])
            im = show(o, d) if o else None
            if im is None:
                a.text(0.5, 0.5, "no image\n(cut off at edge)" if o == bo else "", ha="center", va="center", fontsize=12, color="#777777")
                for s in a.spines.values():
                    s.set_visible(False)
                continue
            a.imshow(im)
            for s in a.spines.values():
                s.set_edgecolor(col); s.set_linewidth(2.5)
            vv = meta[o]
            a.set_title(f"{tag}  {o.replace('_nosplit', '').split('_')[-1]}  day {d}\narea {np.exp(LA.loc[o, d]):,.0f} px · votes good {vv['n_votes_good']}/{vv['n_votes_total']}",
                        fontsize=11, color=col)
    ax[i, 0].set_ylabel(f"{b}\nvotes good {v['n_votes_good']}/{v['n_votes_total']}", fontsize=12, color="#e5603b", fontweight="bold")
fig.suptitle("Rejected organoids that growth rate and every image model miss, next to a same-plate, same-day, same-size accepted organoid\n"
             "(brightened bbox input, exactly as the CNN saw it; bbox makes every organoid fill the frame, so compare shape/texture, not size)",
             fontsize=15, y=1.0)
fig.tight_layout()
fig.savefig(OUT / "unexplained_rejects_brightbbox.png", dpi=80, bbox_inches="tight")
fig.savefig(OUT / "unexplained_rejects_brightbbox.pdf", bbox_inches="tight")
print("saved")
