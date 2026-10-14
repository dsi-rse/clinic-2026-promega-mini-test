"""Paper figures for the image-model part:
  Fig A  what we ran and how well it works (inputs, day-30 ladder, over time, LSTM vs size + growth)
  Fig B  what the models look at (Grad-CAM, mask only, rescaling, LSTM day blanking)
  Fig C  what no model catches (prediction vs size, % caught, shape, the missed rejects)

All vote counts are regular-image votes only (cohort meta, efd3ff7); labels are the official
>= 4/5 labels. Sources:
  V2 (data/cohorts/idor_main/full, 155-198 organoids/day): per-day CNN + LSTM OOF predictions,
     perday_eval.py tables, LSTM day dropout, shape features, mask areas.
  V1 (data/cohorts/idor_main/series, 140 organoids): occlusion / rescale / Grad-CAM (Fig B a-d).

Run from the repo root on the cluster (core_env):
    python analysis/figures_paper/make_paper_figs.py <out_dir>
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("/net/projects2/promega/project_data")
MP, RUNS = DATA / "amanda_test/model_plots", DATA / "model_tests/lstm_runs/idor_main_perday"
V2R = MP / "v2/votefix_eval"   # perday_eval.py on the regular-vote LSTMs (= interpretability_v2/results)
V1F = ROOT / "analysis/interpretability_v1/figures"

sns.set_theme(context="paper", style="white")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
S2 = sns.color_palette("Set2")
GOOD, BAD = "#5b72b0", "#e5603b"
CNN = [("std", "CNN standard", S2[2], "o"), ("strong", "CNN strong aug", S2[1], "^"),
       ("bbox", "CNN bbox crop", S2[3], "s"), ("brightbbox", "CNN bright. bbox", S2[5], "D")]
DAYS = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30]


def load_meta(c):
    m = {}
    for p in ("train", "val", "test"):
        m.update(json.load(open(ROOT / f"data/cohorts/idor_main/{c}/{p}.json")))
    return m


meta, meta1 = load_meta("full"), load_meta("series")
y = pd.Series({o: int(v["label"] == "Acceptable") for o, v in meta.items()})
plate = pd.Series({o: "_".join(o.split("_")[:3]) for o in meta})
LA = pd.read_csv(MP / "v2/perday_mask_area.csv").pivot(index="organoid_id", columns="day", values="log_area")
short = lambda o: o.replace("_nosplit", "")
well = lambda o: short(o).split("_")[-1]


def tag(a, s):
    a.text(-0.12, 1.06, s, transform=a.transAxes, fontsize=16, fontweight="bold", va="bottom")


def day_axis(a):
    a.set_xticks(DAYS); a.set_xticklabels(["3", "", "8", "", "13", "", "17", "20.5", "24", "28", "30"]); a.set_xlabel("Day")


def oof_mean(path):
    return pd.read_csv(path).groupby("organoid_id").prob_acceptable.mean()


# ---------- the rejects no method catches (fixed rule, 4x10 CV means) ----------
iv = [(15, 17), (17, 20.5), (20.5, 24), (24, 28), (28, 30)]
G = pd.DataFrame({f"{a}-{b}": LA[b] - LA[a] for a, b in iv})
ids = np.array(sorted(LA.index)); yy = y[ids].values
P = pd.Series(0.0, index=ids)
for r in range(10):
    for tr, te in StratifiedKFold(4, shuffle=True, random_state=1 + r * 1000).split(ids, yy):
        dev = pd.Series(np.nan, index=ids)
        for pl in plate[ids].unique():
            ref = G.loc[[i for i in ids[tr] if plate[i] == pl and y[i] == 1]]; med = ref.median(); mad = 1.4826 * (ref - med).abs().median()
            k = [i for i in ids if plate[i] == pl]; dev[k] = ((G.loc[k] - med) / mad).abs().mean(axis=1).values
        dv = dev.fillna(dev.median()).values[:, None]
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(dv[tr], yy[tr])
        P[ids[te]] += m.predict_proba(dv[te])[:, 1] / 10
T = pd.DataFrame({"growth band": P})
for k, *_ in CNN:   # latest late day with an image
    s = {}
    for d in ("24", "28", "30"):
        s.update(oof_mean(RUNS / f"base_effnet_kfold4x10_{k}/day_{d}/oof_predictions.csv").to_dict())
    T[k] = pd.Series(s)
T["LSTM"] = pd.concat([pd.read_csv(f) for f in sorted((RUNS / "temporal_lstm_kfold4x10_mean/days_3-30").glob("oof_rep*.csv"))]
                      ).groupby("organoid_id").prob_acceptable.mean()
Tb = T.loc[[o for o in T.index if y[o] == 0]]
MISSED = sorted(Tb.index[(Tb.fillna(1) >= 0.5).all(axis=1)])
T.loc[Tb.index].assign(missed_by_all=lambda d: d.index.isin(MISSED)).round(3).to_csv(OUT / "rejects_by_method.csv")
print(f"missed by every method: {len(MISSED)}/{len(Tb)}", [short(o) for o in MISSED])

# ---------- exact model inputs (dataset code the models were trained with) ----------
import torch
from torchvision import transforms as TT
from analysis.images.cnn_lstm.train_base_model import SingleDayOrganoidDataset, input_resize
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def show(oid, d, bbox=True, bright=True):
    ds = SingleDayOrganoidDataset([oid], meta, d, transform=TT.Compose(input_resize("clipped", bbox)), image_type="clipped",
                                  bbox_crop=bbox, brighten=bright, exact_day=True)
    return None if len(ds) == 0 else np.clip((ds[0][0] * STD + MEAN).permute(1, 2, 0).numpy(), 0, 1)


def frame(a, col, lw=1.5):
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(lw)


# ---------- shared names / colours ----------
cnn = pd.read_csv(V2R / "perday_cnn.csv"); lstm = pd.read_csv(V2R / "perday_lstm.csv")
SIZE1, SIZE2, SIZEG = "0.72", "0.25", "0.45"          # grays: size one cutoff / two cutoffs / + growth
NAMES = {"std": "CNN, whole image", "strong": "CNN, whole image\n+ strong augment.", "bbox": "CNN, cropped to organoid",
         "brightbbox": "CNN, cropped +\nbrightened", "LSTM mean": "LSTM, image\nsequence days 3–30"}
COL = {"std": S2[2], "strong": S2[1], "bbox": S2[3], "brightbbox": S2[5], "LSTM mean": "#6a51a3"}
MM2 = lambda la: np.exp(la) * 81e-6   # mask pixels at 9 um/px (all_data ar_target_um_per_px) -> mm^2
EXAMPLES = ["BA1_96_1_H6_nosplit", "BA2_96_2_C3_nosplit", "BA2_96_2_E12_nosplit"]   # same organoids as the Grad-CAM panel
EXNOTE = {"BA1_96_1_H6_nosplit": "accepted", "BA2_96_2_C3_nosplit": "rejected, stalled", "BA2_96_2_E12_nosplit": "rejected, overgrown"}


def exlabel(o):
    v = meta[o]
    return f"{short(o).replace('_96_', ' ')}\n{EXNOTE[o]}\n({v['n_votes_good']}/{v['n_votes_total']} Acc. votes)"


def save(fig, name):
    [sns.despine(ax=ax_) for ax_ in fig.axes if not ax_.images]
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{e}")
    plt.close(fig); print("saved", name)


def mask_img(o, d):
    tp = next((t for t in meta[o]["timepoints"] if t["mdl_day"] == d), None)
    return None if tp is None else np.asarray(Image.open(tp["mask_paths"]["clipped"]).convert("L")) > 127


# kind of reject: stalled vs overgrown (latest late-day size vs same-plate accepted median)
acc_med = {(pl, d): LA.loc[[o for o in LA.index if y[o] == 1 and plate[o] == pl], d].median() for pl in plate.unique() for d in DAYS}
kind = {}
for o in LA.index:
    if y[o] == 1:
        kind[o] = "accepted"
    else:
        d = next((dd_ for dd_ in (30, 28, 24) if pd.notna(LA.loc[o].get(dd_))), None)
        kind[o] = "rejected, overgrown" if d and LA.loc[o, d] > acc_med[(plate[o], d)] else "rejected, stalled"
kind = pd.Series(kind)
KC = {"accepted": GOOD, "rejected, stalled": BAD, "rejected, overgrown": "#b2182b"}

# ======================= FIG A: what we ran, and how well it works =======================
fig = plt.figure(figsize=(16, 21.5))
gs = GridSpec(3, 3, figure=fig, height_ratios=[1.15, 1.35, 1], hspace=.42, wspace=.32)

# a: model inputs, one organoid per row
sub = gs[0, :].subgridspec(3, 10, wspace=.06, hspace=.1, width_ratios=[1, 1, 1, 1, .2, 1, 1, 1, 1, 1])
COLS = [("SIZE ONLY\n(mask area)", None), ("CNN, whole\nimage", (False, False)),
        ("CNN, cropped\nto organoid", (True, False)), ("CNN, cropped\n+ brightened", (True, True))]
for i, o in enumerate(EXAMPLES):
    col = GOOD if y[o] == 1 else BAD
    for j, (nm, mode) in enumerate(COLS):
        a = fig.add_subplot(sub[i, j])
        if mode is None:
            m = mask_img(o, 30); a.imshow(m, cmap="gray_r", vmin=0, vmax=1.4)
            a.text(.5, .06, f"area = {MM2(LA.loc[o, 30]):.2f} mm²", transform=a.transAxes, ha="center", fontsize=8.5,
                   bbox=dict(fc="w", ec="none", alpha=.85, pad=1))
        else:
            im = show(o, 30, *mode)
            if im is not None:
                a.imshow(im)
        frame(a, "0.4" if mode is None else col)
        if i == 0:
            a.set_title(nm + "\nday 30", fontsize=10, fontweight="bold" if mode is None else "normal")
        if j == 0:
            a.set_ylabel(exlabel(o), fontsize=9, color=col)
            if i == 0:
                tag(a, "a")
    for j, d in enumerate((3, 10, 17, 24, 30)):
        a = fig.add_subplot(sub[i, 5 + j]); im = show(o, d, False, False)
        if im is not None:
            a.imshow(im)
        frame(a, col)
        if i == 0:
            a.set_title(("LSTM: the whole image sequence\n" if j == 2 else "\n") + f"day {d}", fontsize=10)

def mask_auc(v, d):
    """mean over repeats of the per-repeat OOF AUC (as perday_eval.py does for the other CNNs)"""
    o = pd.read_csv(RUNS / f"base_effnet_kfold4x10_{v}/day_{d}/oof_predictions.csv")
    return float(np.mean([roc_auc_score(g.true_label, g.prob_acceptable) for _, g in o.groupby("repeat")]))


COL.update({"mask": "#d9d9d9", "maskbbox": "#a6a6a6"})
MASK_DAYS = ["17", "20.5", "24", "28", "30"]

# b: the ladder at day 30
a = fig.add_subplot(gs[1, :2]); tag(a, "b")
c30 = cnn[cnn.day == 30].set_index("model"); l30 = lstm[(lstm.window == "3-30")].set_index("model")
LAD = [("Size, one cutoff\n(bigger = better)", c30.loc["std", "size_auc"], SIZE1, "size"),
       ("Size, two cutoffs\n(too small or too big = bad)", c30.loc["std", "flex_auc"], SIZE2, "size"),
       ("Size + growth rate\n(days 3–30, two cutoffs)", l30.loc["LSTM mean", "flex_auc"], SIZEG, "size")] + \
      [("CNN, mask only, whole frame\n(size + outline)", mask_auc("mask", "30"), COL["mask"], "mask"),
       ("CNN, mask only, cropped\n(outline only)", mask_auc("maskbbox", "30"), COL["maskbbox"], "mask")] + \
      [(NAMES[k].replace("\n", " "), c30.loc[k, "auc"], COL[k], "img") for k in ("std", "strong", "bbox", "brightbbox")] + \
      [("LSTM, image sequence days 3–30\n(averaged over all frames)", l30.loc["LSTM mean", "auc"], COL["LSTM mean"], "img")]
yy_ = np.array([12, 11, 10, 8, 7, 5, 4, 3, 2, 1], float)   # gaps between the three groups
for (nm, v, c, grp), yv in zip(LAD, yy_):
    a.barh(yv, v - .5, left=.5, color=c, edgecolor="k", lw=.6, height=.7, hatch="//" if grp == "mask" else None)
    a.text(v + .005, yv, f"{v:.2f}", va="center", fontsize=11, fontweight="bold" if v == max(x[1] for x in LAD) else "normal")
for yl in (9, 6):
    a.axhline(yl, color="0.5", lw=1, ls="--")
a.text(.505, 12.5, "SIZE ONLY (one number per organoid, no image)", fontsize=10, color="0.3", va="bottom")
a.text(.505, 8.5, "MASK ONLY (CNN trained on the black-and-white mask: no texture)", fontsize=10, color="0.3", va="bottom")
a.text(.505, 5.5, "IMAGE MODELS (see the whole picture)", fontsize=10, color="0.3", va="bottom")
a.set_yticks(yy_); a.set_yticklabels([x[0] for x in LAD], fontsize=10); a.set_xlim(.5, .95)
a.set_xlabel("AUC at day 30 (0.5 = chance; mean of 4-fold × 10-repeat CV, same 155 organoids)")
a.set_title("Day 30: the best image model (0.88) only ties size with two cutoffs (0.87)", fontsize=13)

# c: why two cutoffs - day-30 size density, accepted vs not accepted
a = fig.add_subplot(gs[1, 2]); tag(a, "c")
from scipy.stats import gaussian_kde
d30 = pd.DataFrame({"mm2": MM2(LA[30]), "y": y}).dropna()
xs_ = np.linspace(np.log10(d30.mm2.min()) - .15, np.log10(d30.mm2.max()) + .15, 400)
for lab, c, nm in ((1, GOOD, "Acceptable"), (0, BAD, "Not Acceptable")):
    v = np.log10(d30[d30.y == lab].mm2.values)
    dens = gaussian_kde(v)(xs_)
    a.fill_between(10 ** xs_, dens, color=c, alpha=.3, lw=0); a.plot(10 ** xs_, dens, color=c, lw=2.2, label=f"{nm} (n={len(v)})")
    a.plot(10 ** v, np.full(len(v), -.06 if lab else -.14), "|", color=c, ms=9, mew=1.2)
a.axhline(0, color="0.6", lw=.8)
a.set_xscale("log"); a.set_xticks([0.5, 1, 2, 4, 8]); a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
a.set_xlabel("Day-30 organoid area (mm²)"); a.set_ylabel("Density"); a.set_ylim(bottom=-.2)
a.legend(loc="upper right", fontsize=8.5)
a.set_title("Why two cutoffs: Not Acceptable organoids\nare both smaller and larger than Acceptable")
n_off = int(((y == 0) & LA[30].reindex(y.index).isna()).sum())
a.text(.99, .62, f"+{n_off} Not Acc. with no day-30\nimage (cut off at frame edge)", transform=a.transAxes, ha="right", fontsize=7.5, color="0.4")

# d: growth curves
a = fig.add_subplot(gs[2, 0]); tag(a, "d")
for k, c in KC.items():
    A_ = MM2(LA.loc[kind.index[kind == k]])
    med, lo, hi = A_.median(), A_.quantile(.25), A_.quantile(.75)
    a.fill_between(med.index, lo, hi, color=c, alpha=.18, lw=0)
    a.plot(med.index, med, "o-", color=c, lw=2.2, ms=5, mec="k", mew=.4, ls="--" if "over" in k else "-", label=f"{k} (n={int((kind == k).sum())})")
a.axvspan(14.5, 17.5, color="#fde7c8", zorder=0)
a.set_yscale("log"); a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
a.set_yticks([0.1, 0.2, 0.5, 1, 2, 5]); a.set_ylabel("Organoid area (mm²)"); day_axis(a); a.legend(loc="upper left", fontsize=8.5)
a.set_title("Rejects stall or overgrow;\ngroups split at days 15–17 (shaded)")

# e: over time - image models vs size
a = fig.add_subplot(gs[2, 1]); tag(a, "e")
b0 = cnn[cnn.model == "std"].sort_values("day")
a.plot(b0.day, b0.flex_auc, "-", color=SIZE2, lw=3, label="size, two cutoffs")
a.plot(b0.day, b0.size_auc, "--", color=SIZE1, lw=2, label="size, one cutoff")
for k in ("std", "strong", "bbox", "brightbbox"):
    d = cnn[cnn.model == k].sort_values("day")
    a.plot(d.day, d.auc, "o-", color=COL[k], lw=1.4 if k != "brightbbox" else 2.2, ms=4 if k != "brightbbox" else 6, mec="k", mew=.4,
           label=NAMES[k].replace("\n", " "))
for v, nm in (("mask", "CNN, mask only (whole frame)"), ("maskbbox", "CNN, mask only (cropped)")):
    a.plot([float(d) for d in MASK_DAYS], [mask_auc(v, d) for d in MASK_DAYS], "s--", color=COL[v], lw=1.6, ms=5, mec="k", mew=.4, label=nm)
a.axvspan(2, 18, color="0.94", zorder=0)
a.axhline(.5, color="0.7", ls=":", lw=1); a.set_ylim(.4, .95); a.set_ylabel("AUC"); day_axis(a); a.set_xlim(2, 31)
a.legend(loc="upper left", fontsize=7.5)
a.set_title("Every day: image models never beat size;\nbefore day 20 they are worse")

# f: does the LSTM add anything?
a = fig.add_subplot(gs[2, 2]); tag(a, "f")
W = ["3-13", "3-17", "3-24", "3-30"]; x = np.arange(4)
l0 = lstm[lstm.model == "LSTM mean"].set_index("window").loc[W]
a.plot(x, l0.flex_auc, "-", color=SIZEG, lw=3, marker="o", ms=6, label="size + growth rate (two cutoffs)")
a.plot(x, l0.auc, "-", color=COL["LSTM mean"], lw=2.2, marker="s", ms=7, mec="k", mew=.5, label="LSTM, image sequence\n(averaged over all frames)")
a.set_xticks(x); a.set_xticklabels([f"days\n{w}" for w in W]); a.axhline(.5, color="0.7", ls=":", lw=1)
a.set_ylim(.4, .95); a.set_ylabel("AUC"); a.legend(loc="upper left", fontsize=8.5)
a.set_title("LSTM adds nothing: below size + growth\nfor every window")
save(fig, "figA_models_and_performance")

# ======================= FIG B: what the models look at =======================
fig = plt.figure(figsize=(16, 15))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1.25, 1], hspace=.38, wspace=.32)

# a: Grad-CAM examples (V1 rotation figures, 0° input + standard / strong overlays), same organoids as Fig A-a
sub = gs[0, :2].subgridspec(3, 3, wspace=.04, hspace=.1)
EXF = {"BA1_96_1_H6_nosplit": "Dy30_acc_bothcorrect_BA1_96_1_H6.png", "BA2_96_2_C3_nosplit": "Dy30_notacc_bothcaught_BA2_96_2_C3.png",
       "BA2_96_2_E12_nosplit": "Dy30_notacc_bothmissed_BA2_96_2_E12.png"}
BOX = {"input": (.0915, .2835), "std": (.3015, .4925), "strong": (.8015, .9925)}
for i, o in enumerate(EXAMPLES):
    im = Image.open(V1F / "gradcam_examples" / EXF[o]).convert("RGB"); W_, H_ = im.size
    col = GOOD if y[o] == 1 else BAD
    for j, (k, nm) in enumerate((("input", "input, day 30"), ("std", "Grad-CAM: CNN, whole image"), ("strong", "Grad-CAM: CNN + strong augment."))):
        y0, y1 = BOX[k]; a = fig.add_subplot(sub[i, j])
        a.imshow(im.crop((int(.008 * W_), int(y0 * H_), int(.218 * W_), int(y1 * H_)))); a.set_xticks([]); a.set_yticks([])
        if i == 0:
            a.set_title(nm, fontsize=11)
        if j == 0:
            a.set_ylabel(exlabel(o), fontsize=9, color=col)
            if i == 0:
                a.text(-0.6, 1.1, "a", transform=a.transAxes, fontsize=16, fontweight="bold", va="bottom")

# b: Grad-CAM focus by model
a = fig.add_subplot(gs[0, 2]); tag(a, "b")
for key, lab, c in (("std", "CNN, whole image", COL["std"]), ("strong", "CNN + strong augment.", COL["strong"]), ("bboxstd", "CNN, cropped", COL["bbox"])):
    r = pd.read_csv(MP / f"focus_pred_idor_main_kfold4x10_{key}_rep1.csv")
    r["contrast"] = (r.focus / r.mask_frac) / ((1 - r.focus) / (1 - r.mask_frac))
    med = r.groupby("day").contrast.median()
    a.plot(med.index, med, "o-", color=c, lw=1.8, ms=5, mec="k", mew=.4, label=lab)
a.set_yscale("log"); a.axhline(1, color="0.5", ls="--", lw=1); day_axis(a)
a.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.25, 0.5, 1, 2, 4, 8]))
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×")); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.text(30.5, 1.05, "no focus", fontsize=8, color="0.4", ha="right", va="bottom")
a.set_ylabel("Grad-CAM heat on organoid ÷ background\n(per pixel)"); a.legend(loc="lower right", fontsize=8.5)
a.set_title("Heat sits on the organoid from day ~17;\nstrong augment. looks at background early")

# c: mask only
a = fig.add_subplot(gs[1, 0]); tag(a, "c")
occ = []
for s in ("std", "strong"):
    o = pd.read_csv(MP / f"occlusion_spatial_idor_main_kfold4x10_{s}_rep1.csv")
    for (dd_, cond), g2 in o[o.day.isin([24, 28, 30])].groupby(["day", "condition"]):
        g2 = g2.dropna(subset=["prob_acceptable"]); occ.append(dict(setup=s, condition=cond, auc=roc_auc_score(g2.true_label, g2.prob_acceptable)))
occ = pd.DataFrame(occ)
order = ["original", "silhouette", "core_texture_removed", "no_organoid"]
sns.barplot(data=occ, x="condition", y="auc", hue="setup", order=order, hue_order=["std", "strong"], palette=[COL["std"], COL["strong"]],
            errorbar="sd", capsize=.08, edgecolor="k", linewidth=.6, ax=a)
a.set_xticklabels(["original\nimage", "mask only\n(no texture)", "inside texture\nremoved", "organoid\nremoved"], fontsize=9.5)
a.axhline(.5, color="0.6", ls=":", lw=1); a.set_ylim(.3, .95); a.set_xlabel(""); a.set_ylabel("AUC, days 24–30")
h, _ = a.get_legend_handles_labels(); a.legend(h[:2], ["CNN, whole image", "CNN + strong augment."], loc="upper right", fontsize=8.5)
a.set_title("Show the trained CNN only the mask:\nit still works → it reads size/outline")

# d: rescale
a = fig.add_subplot(gs[1, 1]); tag(a, "d")
o = pd.read_csv(MP / "occlusion_spatial_idor_main_kfold4x10_std_rep1.csv")
w = o.pivot_table(index=["day", "organoid_id"], columns="condition", values="prob_acceptable").reset_index()
for cond, c, nm in (("scaled_1.4", GOOD, "same organoid enlarged 1.4×"), ("scaled_0.7", BAD, "same organoid shrunk 0.7×")):
    dl = (w[cond] - w["original"]).groupby(w.day)
    med, lo, hi = dl.median(), dl.quantile(.25), dl.quantile(.75)
    a.fill_between(med.index, lo, hi, color=c, alpha=.2, lw=0); a.plot(med.index, med, "o-", color=c, lw=2, ms=5, mec="k", mew=.4, label=nm)
a.axhline(0, color="0.6", ls=":", lw=1); day_axis(a); a.set_ylabel("change in P(Acceptable)\n(CNN, whole image)"); a.legend(loc="lower left", fontsize=8.5)
a.set_title("Only resize the organoid:\nbigger → 'more acceptable' (from day 17)")

# e: LSTM day dropout
a = fig.add_subplot(gs[1, 2]); tag(a, "e")
dd = pd.read_csv(MP / "v2/lstm_day_dropout.csv")
g = dd[(dd.readout == "mean") & (dd.window == "3-30")]
base = g.drop_duplicates("organoid_id").set_index("organoid_id")
out = []
for day, h in g.groupby("day"):
    pb_ = base.p_full.copy(); pb_[h.organoid_id] = h.set_index("organoid_id").p_blank
    out.append((day, roc_auc_score(base.label, pb_) - roc_auc_score(base.label, base.p_full)))
out = np.array(out)
a.bar(range(len(out)), out[:, 1], color=COL["LSTM mean"], edgecolor="k", lw=.5)
a.set_xticks(range(len(out))); a.set_xticklabels([f"{d:g}" for d in out[:, 0]], fontsize=9); a.set_xlabel("Day whose image is blanked")
a.axhline(0, color="0.5", lw=.8); a.set_ylabel("change in LSTM AUC"); a.set_ylim(-.06, .06)
a.set_title(f"LSTM (averaged over frames): blanking any\none day changes AUC by ≤ {np.abs(out[:, 1]).max():.2f}")
save(fig, "figB_what_models_look_at")

# ======================= FIG C: what no model catches =======================
fig = plt.figure(figsize=(16, 11))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.15], hspace=.4, wspace=.32)

a = fig.add_subplot(gs[0, 0]); tag(a, "a")
p = oof_mean(RUNS / "base_effnet_kfold4x10_brightbbox/day_30/oof_predictions.csv")
d = pd.DataFrame({"p": p, "a": LA[30], "k": kind}).dropna()
for k, c in KC.items():
    s = d[d.k == k]; a.scatter(MM2(s.a), s.p, s=22, color=c, edgecolor="k", lw=.4, alpha=.85, label=k)
a.set_xscale("log"); a.set_xticks([0.5, 1, 2, 4, 8]); a.axhline(.5, color="0.6", ls="--", lw=1); a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
a.set_xlabel("Day-30 organoid area (mm²)"); a.set_ylabel("CNN P(Acceptable)\n(cropped + brightened)"); a.legend(loc="lower right", fontsize=8.5)
a.set_title("Best CNN's predictions follow size:\nsmall → bad, very big → borderline (~0.5)")

a = fig.add_subplot(gs[0, 1]); tag(a, "b")
caught = (Tb < .5).where(Tb.notna()).mean() * 100
caught["any"] = 100 * (1 - len(MISSED) / len(Tb))
xs = np.arange(len(caught))
a.bar(xs, caught, color=[SIZEG] + [COL[k] for k in ("std", "strong", "bbox", "brightbbox", "LSTM mean")] + ["0.15"], edgecolor="k", lw=.5)
for i, v in enumerate(caught):
    a.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
a.set_xticks(xs); a.set_xticklabels(["growth\nrate", "CNN\nwhole", "CNN\nstrong", "CNN\ncrop", "CNN crop\n+ bright", "LSTM\n", "any\nmethod"], fontsize=8.5)
a.set_ylabel(f"% of the {len(Tb)} rejects caught"); a.set_ylim(0, 100)
a.set_title(f"{len(MISSED)} of {len(Tb)} rejects ({len(MISSED) / len(Tb):.0%}) are caught\nby no method")

a = fig.add_subplot(gs[0, 2]); tag(a, "c")
S = pd.read_csv(MP / "v2/shape_features.csv")
FEATS = ["solidity", "circularity", "elongation", "skeleton_ends"]
S["plate"] = S.organoid_id.map(plate); S["good"] = S.organoid_id.map(y)
for f in FEATS:
    ref = S[S.good == 1].groupby(["plate", "day"])[f].agg(["median", lambda s: 1.4826 * (s - s.median()).abs().median() + 1e-9])
    ref.columns = ["med", "mad"]; j = S.join(ref, on=["plate", "day"]); S[f] = (S[f] - j.med) / j.mad
Sp = S.groupby("organoid_id")[FEATS].mean()
Sp["group"] = np.where(Sp.index.isin(MISSED), "missed by all", np.where(y[Sp.index] == 1, "accepted", "other rejects"))
L = Sp.reset_index().melt(id_vars=["organoid_id", "group"], var_name="feature", value_name="z")
L["z"] = L.z.clip(-6, 6)
pal = {"accepted": GOOD, "other rejects": "0.7", "missed by all": BAD}
sns.boxplot(data=L, x="feature", y="z", hue="group", hue_order=list(pal), palette=pal, showfliers=False, linewidth=.8, ax=a)
a.axhline(0, color="0.6", ls=":", lw=1); a.set_xlabel(""); a.set_ylabel("shape vs accepted (z), days 24–30")
a.set_xticklabels(["solidity", "roundness", "elongation", "branch\nends"]); a.legend(loc="upper left", fontsize=8)
a.set_title("They are less round, but shape also flags\nmany good ones (no gain in CV)")

sub = gs[1, :].subgridspec(2, len(MISSED), wspace=.05, hspace=.25)
good = [o for o in meta if y[o] == 1]
for i, b in enumerate(MISSED):
    d = next(dd_ for dd_ in (30, 28, 24) if pd.notna(LA.loc[b].get(dd_)))
    m = min([o for o in good if plate[o] == plate[b] and pd.notna(LA.loc[o].get(d))], key=lambda o: abs(LA.loc[o, d] - LA.loc[b, d]))
    for r_, (o, col, nm) in enumerate(((b, BAD, "rejected,\nmissed by all"), (m, GOOD, "accepted,\nsame plate & size"))):
        a = fig.add_subplot(sub[r_, i]); im = show(o, d)
        if im is not None:
            a.imshow(im)
        frame(a, col, 2.2)
        v = meta[o]
        a.set_title(f"{short(o).replace('_96_', ' ')} day {d}\n{v['n_votes_good']}/{v['n_votes_total']} Acc. votes", fontsize=9, color=col)
        if i == 0:
            a.set_ylabel(nm, fontsize=10, color=col)
            if r_ == 0:
                tag(a, "d")
save(fig, "figC_what_no_model_catches")
