"""Big-but-rejected organoids vs size-matched accepted ones, days 13-30, same scale."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.io import imread

ids = sys.argv[1].split()
odd, refs = ids[:4], ids[4:]
meta = {}
for p in ("train", "val", "test"):
    meta.update(json.load(open(f"data/cohorts/idor_main/series/{p}.json")))
DAYS = [13, 17, 20.5, 24, 28, 30]
rows = [x for pair in zip(odd, refs) for x in pair]
fig, ax = plt.subplots(len(rows), len(DAYS), figsize=(2.6 * len(DAYS), 2.75 * len(rows)))
for i, oid in enumerate(rows):
    r = meta[oid]; votes = f"{r['n_votes_good']}/{r['n_votes_total']} Acc"
    for j, d in enumerate(DAYS):
        tp = min(r["timepoints"], key=lambda t: abs(t["mdl_day"] - d))
        img = imread(tp["img_paths"]["clipped"])
        a = ax[i, j]; a.imshow(img, cmap="gray" if img.ndim == 2 else None); a.set_xticks([]); a.set_yticks([])
        if i == 0:
            a.set_title(f"Day {d:g}", fontsize=15, fontweight="bold")
        col = "#e5603b" if oid in odd else "#5b72b0"
        for s in a.spines.values():
            s.set_edgecolor(col); s.set_linewidth(3 if oid in odd else 1.5)
    tag = "REJECTED" if oid in odd else "accepted, same size"
    ax[i, 0].set_ylabel(f"{tag}\n{oid.replace('_nosplit', '')}\n({votes})", fontsize=11,
                        color="#c0392b" if oid in odd else "#34495e", fontweight="bold" if oid in odd else "normal")
fig.suptitle("Big organoids raters rejected (red) vs accepted organoids of the same day-30 size (blue); same scale",
             fontsize=14, y=1.0)
fig.tight_layout()
fig.savefig("odd.png", dpi=110, bbox_inches="tight"); fig.savefig("odd.pdf", bbox_inches="tight")
