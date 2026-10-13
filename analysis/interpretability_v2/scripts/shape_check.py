"""Do simple mask-shape parameters set apart the rejected organoids that growth misses?

Shape per organoid and day (24, 28, 30) from the clipped (9 um/px) masks: solidity,
circularity, elongation, necks (narrow pinch points), skeleton branches, edge roughness.
Plots where the growth-missed rejects sit vs accepted and other rejected organoids (values
z-scored within plate & day against accepted organoids), then tests, with the same 4x10 CV
as the growth-band rule, whether growth band + shape catches more rejects.

    python analysis/interpretability_v2/scripts/shape_check.py <area_csv> <out_dir>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy.ndimage import binary_fill_holes, distance_transform_edt, binary_opening
from skimage.measure import label, regionprops, perimeter
from skimage.morphology import skeletonize, disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
AREA, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
meta = {}
for p in ("train", "val", "test"):
    meta.update(json.load(open(ROOT / f"data/cohorts/idor_main/full/{p}.json")))
y = pd.Series({o: int(v["label"] == "Acceptable") for o, v in meta.items()})
plate = pd.Series({o: "_".join(o.split("_")[:3]) for o in meta})
MISSED = [m + "_nosplit" for m in ("BA1_96_1_A8", "BA2_96_1_F3", "BA2_96_1_D8", "BA2_96_1_G5", "BA2_96_2_G7", "BA2_96_2_H2", "BA2_96_1_A5")]
DAYS = (24, 28, 30)


def shape(mask):
    m = binary_fill_holes(mask)
    lab = label(m); m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    rp = regionprops(m.astype(int))[0]
    per = perimeter(m)
    # necks: thin parts removed by an opening at ~1/4 of the organoid's max half-width
    r = max(2, int(distance_transform_edt(m).max() * 0.5))
    opened = binary_opening(m, structure=disk(r))
    n_lobes = int(label(opened).max())
    sk = skeletonize(m); nb = sum(1 for yx in zip(*np.where(sk)) if sk[max(0, yx[0]-1):yx[0]+2, max(0, yx[1]-1):yx[1]+2].sum() - 1 == 1)
    smooth = binary_opening(m, structure=disk(3))
    return dict(solidity=rp.solidity, circularity=4 * np.pi * rp.area / per ** 2,
                elongation=rp.major_axis_length / max(rp.minor_axis_length, 1),
                lobes_after_neck_cut=n_lobes, skeleton_ends=nb,
                edge_roughness=per / max(perimeter(smooth), 1))


cache = OUT / "shape_features.csv"
if cache.exists():
    S = pd.read_csv(cache)
else:
    rows = []
    for o, v in meta.items():
        for tp in v["timepoints"]:
            if tp["mdl_day"] in DAYS:
                mk = np.asarray(Image.open(tp["mask_paths"]["clipped"]).convert("L")) > 127
                rows.append(dict(organoid_id=o, day=tp["mdl_day"], **shape(mk)))
    S = pd.DataFrame(rows); S.to_csv(cache, index=False)
# lobes_after_neck_cut and edge_roughness are ~constant across organoids (MAD ~ 0), so their z-scores
# explode; they are computed and cached but left out of the plot and the CV test.
FEATS = ["solidity", "circularity", "elongation", "skeleton_ends"]
S["plate"] = S.organoid_id.map(plate); S["good"] = S.organoid_id.map(y)
for f in FEATS:   # z vs accepted organoids of the same plate and day
    ref = S[S.good == 1].groupby(["plate", "day"])[f].agg(["median", lambda s: 1.4826 * (s - s.median()).abs().median() + 1e-9])
    ref.columns = ["med", "mad"]; j = S.join(ref, on=["plate", "day"]); S[f + "_z"] = (S[f] - j.med) / j.mad
S["group"] = np.where(S.organoid_id.isin(MISSED), "rejected, missed by growth", np.where(S.good == 1, "accepted", "rejected, other"))

# per-organoid: mean z over the available late days
P = S.groupby("organoid_id")[[f + "_z" for f in FEATS]].mean()
P["group"] = S.groupby("organoid_id").group.first()
print("median z (vs accepted, same plate & day), days 24-30:")
print(P.groupby("group").median().round(2).T.to_string())
print("\nmissed rejects, per organoid:")
print(P[P.group.str.contains("missed")].round(1).rename(index=lambda s: s.replace("_nosplit", "")).to_string())

sns.set_theme(context="paper", style="white", font_scale=1.4)
pal = {"accepted": "#5b72b0", "rejected, other": "#bdbdbd", "rejected, missed by growth": "#e5603b"}
fig, ax = plt.subplots(1, len(FEATS), figsize=(3.3 * len(FEATS), 4.8), sharey=False)
L = P.reset_index().melt(id_vars=["organoid_id", "group"], var_name="feature", value_name="z")
for a, f in zip(ax, FEATS):
    d = L[L.feature == f + "_z"]
    sns.stripplot(data=d[d.group != "rejected, missed by growth"], x="group", y="z", order=list(pal), hue="group", hue_order=list(pal), palette=pal, legend=False, size=3.5, alpha=.6, jitter=.25, ax=a)
    sns.stripplot(data=d[d.group == "rejected, missed by growth"], x="group", y="z", order=list(pal), hue="group", hue_order=list(pal), palette=pal, legend=False, size=8, jitter=.1, ax=a, edgecolor="k", linewidth=.6)
    a.axhline(0, color="#999999", ls="--", lw=1); a.set_title(f.replace("_", " "), fontsize=13); a.set_xlabel("")
    a.set_xticks(range(3)); a.set_xticklabels(["accepted", "other\nrejected", "missed\nrejected"], fontsize=11); a.set_ylabel("z vs accepted (same plate, day)" if a is ax[0] else "")
sns.despine(fig=fig); fig.suptitle("Shape at days 24-30: where do the rejects that growth misses sit?", fontsize=15, y=1.02)
fig.tight_layout(); fig.savefig(OUT / "shape_missed_rejects.png", dpi=130, bbox_inches="tight"); fig.savefig(OUT / "shape_missed_rejects.pdf", bbox_inches="tight")

# ---- fair test: does shape help, over ALL organoids, cross-validated? ----
A = pd.read_csv(AREA).pivot(index="organoid_id", columns="day", values="log_area")
iv = [(15, 17), (17, 20.5), (20.5, 24), (24, 28), (28, 30)]
G = pd.DataFrame({f"{a}-{b}": A[b] - A[a] for a, b in iv})
ids = np.array(sorted(set(A.index) & set(P.index))); yy = y[ids].values
X_shape = P.loc[ids, [f + "_z" for f in FEATS]].fillna(0).values
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
res = {"growth band": [], "growth band + shape": [], "shape only": []}; caught = {k: pd.Series(0.0, index=ids) for k in res}
for r in range(10):
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=1 + r * 1000).split(ids, yy):
        dev = pd.Series(np.nan, index=ids)
        for pl in plate[ids].unique():
            ref = G.loc[[i for i in ids[tr] if plate[i] == pl and y[i] == 1]]; med = ref.median(); mad = 1.4826 * (ref - med).abs().median()
            k = [i for i in ids if plate[i] == pl]; dev[k] = ((G.loc[k] - med) / mad).abs().mean(axis=1).values
        dv = dev.fillna(dev.median()).values[:, None]
        for name, X in (("growth band", dv), ("growth band + shape", np.c_[dv, X_shape]), ("shape only", X_shape)):
            m = lr().fit(X[tr], yy[tr]); p = m.predict_proba(X[te])[:, 1]
            caught[name][ids[te]] += (p < 0.5).astype(float) / 10
            res[name].append(roc_auc_score(yy[te], p))
print("\nCV over all organoids (4x10):")
for k, v in res.items():
    c = caught[k]
    print(f"  {k:22} AUC {np.mean(v):.2f} | bad caught {c[yy == 0].mean():.0%} | good wrongly flagged {c[yy == 1].mean():.0%} | "
          f"missed-7 caught {c[[m for m in MISSED if m in c.index]].mean():.0%}")
