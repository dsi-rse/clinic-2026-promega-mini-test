"""Paper figures for the image-model part:
  Fig 1  the models work, and they work because of size (inputs, balanced accuracy + AUC, confusion
         matrix, over time, too small or too big, best CNN vs size, LSTM vs size + growth)
  Fig 2  what the models look at (Grad-CAM, test-time ablation, mask-only retraining, rescaling,
         LSTM day blanking, the rejects no model catches)
  Fig S  supplement (growth curves, every model every day, borderline-vote robustness, % caught, shape, day-30 ladder)
Also writes day30_metrics.csv and rejects_by_method.csv.

All vote counts are regular-image votes only (cohort meta, efd3ff7); labels are the official
>= 4/5 labels. Sources:
  V2 (data/cohorts/idor_main/full, 155-198 organoids/day): per-day CNN + LSTM OOF predictions,
     perday_eval.py tables, LSTM day dropout, shape features, mask areas.
  V1 (data/cohorts/idor_main/series, 140 organoids): occlusion / rescale / Grad-CAM (Fig 2 a-c, e).
  Borderline run (data/cohorts/idor_main_minvotes3): model_plots/borderline_mv3/borderline_vs_clear.csv (Fig S c).

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
GOOD, BAD = "#2A7F8E", "#C8553D"   # labels only: Acceptable teal, Not Acceptable burnt orange
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
SIZE1, SIZE2, SIZEG = "#A39E93", "#3D3D3D", "#6F6A60"   # warm grays: size one cutoff / two cutoffs / + growth
NAMES = {"std": "CNN, whole image", "strong": "CNN, whole image\n+ strong augment.", "bbox": "CNN, cropped to organoid",
         "brightbbox": "CNN, cropped +\nbrightened", "LSTM mean": "LSTM, image\nsequence days 3–30"}
COL = {"std": "#7D8B4E", "strong": "#5B7C99", "bbox": "#8C5E3C", "brightbbox": "#E0A526", "LSTM mean": "#6B4E71"}   # olive, dusty blue, walnut, mustard (best), plum
MM2 = lambda la: np.exp(la) * 81e-6   # mask pixels at 9 um/px (all_data ar_target_um_per_px) -> mm^2
GC = MP / "gradcam_v2"   # gradcam_v2.py: Grad-CAM of the current CNNs + the 3 example organoids (typical accepted / stalled / overgrown)
GCX = np.load(GC / "gradcam_examples.npz") if (GC / "gradcam_examples.npz").exists() else None   # None: Grad-CAM panels left as placeholders
EXAMPLES = [str(o) for o in GCX["examples"]] if GCX is not None else \
    ["BA1_96_1_A7_nosplit", "BA1_96_1_G1_nosplit", "BA2_96_2_C6_nosplit"]   # gradcam_v2.py's rule picks these; same organoids in Fig 1 a and Fig 2 a
EXNOTE = dict(zip(EXAMPLES, ["accepted", "rejected, stalled", "rejected, overgrown"]))
VARIANTS_ = ["std", "strong", "bbox", "brightbbox"]
GB_TITLE = "Whole-image CNNs only look at the organoid\nonce they start to work (~day 17)"


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
KC = {"accepted": GOOD, "rejected, stalled": BAD, "rejected, overgrown": "#7A2E1F"}


# ---------- per-repeat metrics straight from the OOF predictions ----------
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import SplineTransformer
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))


def oof(k, d="30"):
    if k == "LSTM mean":
        return pd.concat([pd.read_csv(f) for f in sorted((RUNS / "temporal_lstm_kfold4x10_mean/days_3-30").glob("oof_rep*.csv"))])
    return pd.read_csv(RUNS / f"base_effnet_kfold4x10_{k}/day_{d}/oof_predictions.csv")


def size_oof(o, d, two=True):
    """size model OOF on the SAME organoids and folds as OOF table o (spline = two cutoffs), + plate"""
    out = []
    for r, g in o.groupby("repeat"):
        g = g.set_index("organoid_id"); i_ = sorted(g.index); f = g.loc[i_, "fold"].values; t = y[i_].values
        S = LA.loc[i_, [d]].values
        X = np.c_[SplineTransformer(n_knots=4, degree=3).fit_transform(S) if two else S,
                  pd.get_dummies(plate[i_].values, drop_first=True).values.astype(float)]
        p = np.empty(len(i_))
        for k in np.unique(f):
            te = f == k; p[te] = lr().fit(X[~te], t[~te]).predict_proba(X[te])[:, 1]
        out.append(pd.DataFrame({"repeat": r, "organoid_id": i_, "true_label": t, "prob_acceptable": p}))
    return pd.concat(out)


def metrics(o):
    A_, B_, cm = [], [], np.zeros((2, 2))
    for _, g in o.groupby("repeat"):
        t, p = g.true_label.values, g.prob_acceptable.values
        A_.append(roc_auc_score(t, p)); B_.append(balanced_accuracy_score(t, p > .5)); cm += confusion_matrix(t, p > .5, labels=[1, 0])
    return dict(auc=np.mean(A_), auc_sd=np.std(A_), bal=np.mean(B_), bal_sd=np.std(B_), cm=cm / len(A_), n=o.organoid_id.nunique())


IMG = ["std", "strong", "bbox", "brightbbox", "LSTM mean"]
M = {k: metrics(oof(k)) for k in IMG}
ob = oof("brightbbox")
M["size1"], M["size2"] = metrics(size_oof(ob, 30, two=False)), metrics(size_oof(ob, 30))
pd.DataFrame({k: {m: v[m] for m in ("auc", "auc_sd", "bal", "bal_sd", "n")} for k, v in M.items()}).T.round(3).to_csv(OUT / "day30_metrics.csv")
print(pd.read_csv(OUT / "day30_metrics.csv").to_string())


def mask_auc(v, d):
    o = pd.read_csv(RUNS / f"base_effnet_kfold4x10_{v}/day_{d}/oof_predictions.csv")
    return float(np.mean([roc_auc_score(g.true_label, g.prob_acceptable) for _, g in o.groupby("repeat")]))


COL.update({"mask": "#DDD6C1", "maskbbox": "#B5AA8C"})
MASK_DAYS = ["17", "20.5", "24", "28", "30"]
SHORTN = {"std": "CNN\nwhole image", "strong": "CNN whole\n+ strong aug.", "bbox": "CNN\ncropped", "brightbbox": "CNN cropped\n+ brightened",
          "LSTM mean": "LSTM\nimage series"}

# ======================= FIG 1: the models work, and they work because of size =======================
fig = plt.figure(figsize=(20, 17))
gs = GridSpec(3, 4, figure=fig, height_ratios=[1.15, 1, 1], hspace=.45, wspace=.36)

# a: model inputs, same 3 organoids
sub = gs[0, :].subgridspec(3, 10, wspace=.06, hspace=.1, width_ratios=[1, 1, 1, 1, .2, 1, 1, 1, 1, 1])
COLS = [("SIZE ONLY\n(mask area)", None), ("CNN, whole\nimage", (False, False)),
        ("CNN, cropped\nto organoid", (True, False)), ("CNN, cropped\n+ brightened", (True, True))]
MCOL = {None: SIZE2, (False, False): COL["std"], (True, False): COL["bbox"], (True, True): COL["brightbbox"]}
for i, o in enumerate(EXAMPLES):
    col = GOOD if y[o] == 1 else BAD
    for j, (nm, mode) in enumerate(COLS):
        a = fig.add_subplot(sub[i, j])
        if mode is None:
            a.imshow(mask_img(o, 30), cmap="gray_r", vmin=0, vmax=1.4)
            a.text(.5, .06, f"area = {MM2(LA.loc[o, 30]):.2f} mm²", transform=a.transAxes, ha="center", fontsize=8.5,
                   bbox=dict(fc="w", ec="none", alpha=.85, pad=1))
        else:
            im = show(o, 30, *mode)
            if im is not None:
                a.imshow(im)
        frame(a, MCOL[mode], 2)
        if i == 0:
            a.set_title(nm + "\nday 30", fontsize=10, color=MCOL[mode], fontweight="bold")
        if j == 0:
            a.set_ylabel(exlabel(o), fontsize=9, color=col)
            if i == 0:
                tag(a, "a")
    for j, d in enumerate((3, 10, 17, 24, 30)):
        a = fig.add_subplot(sub[i, 5 + j]); im = show(o, d, False, False)
        if im is not None:
            a.imshow(im)
        frame(a, COL["LSTM mean"], 2)
        if i == 0:
            a.set_title(("LSTM: the whole image series\n" if j == 2 else "\n") + f"day {d}", fontsize=10, color=COL["LSTM mean"],
                        fontweight="bold" if j == 2 else "normal")

# b: day-30 performance of the image models
a = fig.add_subplot(gs[1, :2]); tag(a, "b")
x = np.arange(len(IMG)); w = .38
for off, met, hatch, nm in ((-w / 2, "bal", None, "balanced accuracy"), (w / 2, "auc", "..", "AUC")):
    v = [M[k][met] for k in IMG]; e = [M[k][met + "_sd"] for k in IMG]
    a.bar(x + off, np.array(v) - .5, w, bottom=.5, yerr=e, capsize=3, color=[COL[k] for k in IMG], edgecolor="k", lw=.6, hatch=hatch,
          error_kw=dict(lw=.8), label=nm)
    for xi, vi, ei in zip(x + off, v, e):
        a.text(xi, vi + ei + .008, f"{vi:.2f}", ha="center", fontsize=9.5, fontweight="bold" if vi == max(v) else "normal")
a.set_xticks(x); a.set_xticklabels([SHORTN[k] for k in IMG], fontsize=10); a.set_ylim(.5, 1); a.set_ylabel("Day-30 score (0.5 = chance)")
h, l = a.get_legend_handles_labels()
from matplotlib.patches import Patch
a.legend([Patch(fc="w", ec="k"), Patch(fc="w", ec="k", hatch="..")], ["balanced accuracy", "AUC"], loc="upper left", fontsize=9.5)
a.set_title(f"Day 30: the best model (CNN cropped + brightened) reaches AUC {M['brightbbox']['auc']:.2f},\n"
            f"balanced accuracy {M['brightbbox']['bal']:.2f} (mean ± SD over 10 CV repeats, {M['brightbbox']['n']} organoids)", fontsize=12)

# c: confusion matrix of the best model
a = fig.add_subplot(gs[1, 2]); a.text(-0.2, 1.06, "c", transform=a.transAxes, fontsize=16, fontweight="bold", va="bottom")
cm = M["brightbbox"]["cm"]; pct = cm / cm.sum(axis=1, keepdims=True) * 100
a.imshow(pct, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("teal", ["#FBF7EE", GOOD]), vmin=0, vmax=100)
for r_ in range(2):
    for c_ in range(2):
        a.text(c_, r_, f"{pct[r_, c_]:.0f}%\n(n ≈ {cm[r_, c_]:.0f})", ha="center", va="center", fontsize=12,
               color="w" if pct[r_, c_] > 55 else "k")
a.set_xticks([0, 1]); a.set_xticklabels(["Acceptable", "Not Acc."]); a.set_yticks([0, 1]); a.set_yticklabels(["Acceptable", "Not Acc."])
a.set_xlabel("Predicted"); a.set_ylabel("Expert label")
a.set_title("Best model, day 30: % of each expert class\n(mean organoid count per CV repeat)")

# d: over time
a = fig.add_subplot(gs[1, 3]); tag(a, "d")
b0 = cnn[cnn.model == "brightbbox"].sort_values("day")
a.plot(b0.day, b0.flex_auc, "-", color=SIZE2, lw=3, label="size alone (two cutoffs)")
for k in ("std", "strong", "bbox", "brightbbox"):
    d = cnn[cnn.model == k].sort_values("day")
    a.plot(d.day, d.auc, "o-", color=COL[k], lw=1.2 if k != "brightbbox" else 2.4, ms=3.5 if k != "brightbbox" else 6, mec="k", mew=.4,
           alpha=.6 if k != "brightbbox" else 1, label=SHORTN[k].replace("\n", " "))
l0 = lstm[lstm.model == "LSTM mean"].set_index("window").loc[["3-13", "3-17", "3-24", "3-30"]]
a.plot(l0.max_day, l0.auc, "s--", color=COL["LSTM mean"], lw=1.8, ms=6, mec="k", mew=.4, label="LSTM (days 3 → x)")
a.axhline(.5, color="0.7", ls=":", lw=1); a.set_ylim(.4, .95); a.set_ylabel("AUC"); day_axis(a); a.set_xlim(2, 31)
a.legend(loc="upper left", fontsize=7.5)
a.set_title("Image models only work from ~day 20;\nsize alone is as good or better every day")

# e: growth curves - rejects stall or overgrow
a = fig.add_subplot(gs[2, 0]); tag(a, "e")
for k, c in KC.items():
    A_ = MM2(LA.loc[kind.index[kind == k]]); med, lo, hi = A_.median(), A_.quantile(.25), A_.quantile(.75)
    a.fill_between(med.index, lo, hi, color=c, alpha=.18, lw=0)
    a.plot(med.index, med, "o-", color=c, lw=2.2, ms=5, mec="k", mew=.4, ls="--" if "over" in k else "-", label=f"{k} (n={int((kind == k).sum())})")
a.axvspan(14.5, 17.5, color="#fde7c8", zorder=0)
a.set_yscale("log"); a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
a.set_yticks([0.1, 0.2, 0.5, 1, 2, 5]); a.set_ylabel("Organoid area (mm²)"); day_axis(a); a.legend(loc="upper left", fontsize=8.5)
a.set_title("Rejects stall or overgrow;\ngroups split at days 15–17 (shaded)")

# f: too small or too big
a = fig.add_subplot(gs[2, 1]); tag(a, "f")
from scipy.stats import gaussian_kde
d30 = pd.DataFrame({"mm2": MM2(LA[30]), "y": y}).dropna()
xs_ = np.linspace(np.log10(d30.mm2.min()) - .15, np.log10(d30.mm2.max()) + .15, 400)
for lab_, c, nm in ((1, GOOD, "Acceptable"), (0, BAD, "Not Acceptable")):
    v = np.log10(d30[d30.y == lab_].mm2.values); dens = gaussian_kde(v)(xs_)
    a.fill_between(10 ** xs_, dens, color=c, alpha=.3, lw=0); a.plot(10 ** xs_, dens, color=c, lw=2.2, label=f"{nm} (n={len(v)})")
a.set_xscale("log"); a.set_xticks([0.5, 1, 2, 4, 8]); a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
a.set_xlabel("Day-30 organoid area (mm²)"); a.set_ylabel("Density"); a.set_ylim(bottom=0); a.legend(loc="upper right", fontsize=8.5)
a.set_title("Not Acceptable organoids are\ntoo small or too big")

# g: best image model vs size alone
a = fig.add_subplot(gs[2, 2]); tag(a, "g")
G_ = [("size,\none cutoff", "size1", SIZE1), ("size,\ntwo cutoffs", "size2", SIZE2), ("best image\nmodel", "brightbbox", COL["brightbbox"])]
x = np.arange(len(G_))
for off, met, hatch in ((-w / 2, "bal", None), (w / 2, "auc", "..")):
    v = [M[k][met] for _, k, _ in G_]; e = [M[k][met + "_sd"] for _, k, _ in G_]
    a.bar(x + off, np.array(v) - .5, w, bottom=.5, yerr=e, capsize=3, color=[c for *_, c in G_], edgecolor="k", lw=.6, hatch=hatch, error_kw=dict(lw=.8))
    for xi, vi, ei in zip(x + off, v, e):
        a.text(xi, vi + ei + .008, f"{vi:.2f}", ha="center", fontsize=8.5)
a.set_xticks(x); a.set_xticklabels([n for n, *_ in G_], fontsize=9.5); a.set_ylim(.5, 1); a.set_ylabel("Day-30 score")
a.legend([Patch(fc="w", ec="k"), Patch(fc="w", ec="k", hatch="..")], ["bal. accuracy", "AUC"], loc="upper left", fontsize=8.5)
a.set_title("Size alone (two cutoffs) matches the\nbest image model on AUC and beats\nit on balanced accuracy (same folds)")

# h: does the LSTM add anything?
a = fig.add_subplot(gs[2, 3]); tag(a, "h")
W = ["3-13", "3-17", "3-24", "3-30"]; x = np.arange(4)
a.plot(x, l0.flex_auc, "-", color=SIZEG, lw=3, marker="o", ms=6, label="size + growth rate")
a.plot(x, l0.auc, "-", color=COL["LSTM mean"], lw=2.2, marker="s", ms=7, mec="k", mew=.5, label="LSTM, image series")
a.set_xticks(x); a.set_xticklabels([f"days\n{w_}" for w_ in W]); a.axhline(.5, color="0.7", ls=":", lw=1)
a.set_ylim(.4, .95); a.set_ylabel("AUC"); a.legend(loc="upper left", fontsize=8.5)
a.set_title("The image series adds nothing:\nLSTM ≤ size + growth rate in every window")
save(fig, "fig1_models_and_size")

# ======================= FIG 2: what the models look at =======================
fig = plt.figure(figsize=(19, 21))
gs = GridSpec(3, 1, figure=fig, height_ratios=[1.45, 1, .85], hspace=.4)
top = gs[0].subgridspec(1, 3, width_ratios=[2.1, .08, 1], wspace=.25)

# a: Grad-CAM table - rows = the 4 CNNs (current models, own input each), columns = organoid x day
GMOD = [("std", "CNN,\nwhole image"), ("strong", "CNN, whole\n+ strong aug."), ("bbox", "CNN,\ncropped"), ("brightbbox", "CNN, cropped\n+ brightened")]
sub = top[0].subgridspec(len(GMOD), 2 * len(EXAMPLES), wspace=.04, hspace=.06)
jet = plt.get_cmap("jet")
for i, (v, vn) in enumerate(GMOD):
    for j, (o, d) in enumerate([(o, d) for o in EXAMPLES for d in ("10", "30")]):
        a = fig.add_subplot(sub[i, j]); col = GOOD if y[o] == 1 else BAD
        k = f"{o}|{d}|{v}"
        if GCX is not None and f"{k}|img" in GCX.files:
            img, cam = GCX[f"{k}|img"].astype(float), GCX[f"{k}|cam"].astype(float)
            a.imshow(np.clip(.55 * img + .45 * jet(cam)[..., :3], 0, 1))
            a.text(.03, .04, f"P={float(GCX[f'{k}|p']):.2f}", transform=a.transAxes, fontsize=7.5, color="w",
                   bbox=dict(fc="k", ec="none", alpha=.55, pad=1))
        elif GCX is None:   # placeholder: the model's own input, no heat map yet
            im = show(o, float(d), v in ("bbox", "brightbbox"), v == "brightbbox")
            if im is not None:
                a.imshow(im)
            a.text(.5, .5, "Grad-CAM\npending", transform=a.transAxes, ha="center", va="center", fontsize=8, color="0.3",
                   bbox=dict(fc="w", ec="none", alpha=.7, pad=1))
        frame(a, COL[v], 2)
        if i == 0:
            a.set_title(f"day {d}", fontsize=10)
            if d == "10":
                a.text(1.02, 1.32, exlabel(o).replace("\n", " · ", 1), transform=a.transAxes, ha="center", fontsize=9, color=col)
        if j == 0:
            a.set_ylabel(vn, fontsize=10, color=COL[v], fontweight="bold")
            if i == 0:
                a.text(-0.55, 1.3, "a", transform=a.transAxes, fontsize=16, fontweight="bold", va="bottom")

# b: heat on the organoid vs background, by day (all organoids, current models)
a = fig.add_subplot(top[2]); tag(a, "b")
if (GC / "gradcam_focus.csv").exists():
    GF = pd.read_csv(GC / "gradcam_focus.csv")
else:   # interim: the V1 Grad-CAM runs (no cropped + brightened model)
    GF = pd.concat([pd.read_csv(MP / f"focus_pred_idor_main_kfold4x10_{k}_rep1.csv").assign(model=v)
                    for k, v in (("std", "std"), ("strong", "strong"), ("bboxstd", "bbox"))])
    a.text(.98, .02, "interim: earlier models; cropped + brightened pending", transform=a.transAxes, ha="right", fontsize=7.5, color="0.4")
# strong aug. was only trained on the whole image; both cropped models use standard aug.
GLINE = {"std": (COL["std"], "-", "CNN, whole image"), "strong": (COL["strong"], "-", "CNN, whole + strong aug."),
         "bbox": (COL["bbox"], "-", "CNN, cropped"), "brightbbox": (COL["brightbbox"], "-", "CNN, cropped + brightened")}
for v, (c, ls, nm) in GLINE.items():
    med = GF[GF.model == v].groupby("day").focus.median() * 100
    a.plot(med.index, med, marker="o", ls=ls, color=c, lw=2, ms=5, mec="k", mew=.4, label=nm)
# no-focus reference: the organoid's share of the input (heat spread evenly would give exactly this)
for v, c, nm in (("std", "0.35", "whole"), ("bbox", "0.35", "cropped")):
    ref = GF[GF.model == v].groupby("day").mask_frac.median() * 100
    a.plot(ref.index, ref, ":", color=c, lw=1.8)
    if pd.notna(ref.iloc[-1]): a.text(30.6, ref.iloc[-1], f"organoid's share\nof the {nm} image", fontsize=7.5, color=c, va="center", ha="left")
a.axvspan(2, 17, color="0.93", zorder=0)
a.text(9.5, .97, "models near chance\n(days 3–17)", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=8.5, color="0.35")
day_axis(a); a.set_xlim(2, 31); a.set_ylim(0, 100)
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
a.set_ylabel("% of the Grad-CAM heat on the organoid\n(median; dotted = no focus)"); a.legend(loc="center right", fontsize=7.5)
a.set_title(GB_TITLE)

mid = gs[1].subgridspec(1, 2, width_ratios=[1.2, 1], wspace=.25)
# c: remove the texture (retrain on the mask) or the organoid (test time)
a = fig.add_subplot(mid[0]); tag(a, "c")
PAIRS = [("whole frame", "std", "mask"), ("cropped", "bbox", "maskbbox")]
x = np.arange(2)
for off, idx, nm, hatch in ((-w / 2, 1, "trained on the image", None), (w / 2, 2, "trained on the mask only", "//")):
    v = [mask_auc(p_[idx], "30") if idx == 2 else M[p_[idx]]["auc"] for p_ in PAIRS]
    a.bar(x + off, np.array(v) - .5, w, bottom=.5, color=[COL[p_[idx]] for p_ in PAIRS], edgecolor="k", lw=.6, hatch=hatch, label=nm)
    for xi, vi in zip(x + off, v):
        a.text(xi, vi + .008, f"{vi:.2f}", ha="center", fontsize=9.5)
_o = pd.read_csv(MP / "occlusion_spatial_idor_main_kfold4x10_std_rep1.csv"); _o = _o[_o.day.isin([24, 28, 30])].dropna(subset=["prob_acceptable"])
gone = np.mean([roc_auc_score(g2.true_label, g2.prob_acceptable) for _, g2 in _o[_o.condition == "no_organoid"].groupby("day")])
a.bar(2.2, gone - .5, w, bottom=.5, color="w", edgecolor="k", lw=.6, hatch="xx"); a.text(2.2, max(gone, .5) + .008, f"{gone:.2f}", ha="center", fontsize=9.5)
a.axhline(M["size2"]["auc"], color=SIZE2, lw=2, ls="--"); a.text(2.45, M["size2"]["auc"] + .008, "size alone", ha="right", fontsize=8.5, color=SIZE2)
a.axhline(.5, color="0.6", ls=":", lw=1)
a.set_xticks([0, 1, 2.2]); a.set_xticklabels(["CNN,\nwhole frame", "CNN,\ncropped", "organoid erased from the\ntest image (whole-image\nCNN, days 24–30)"]); a.set_ylim(.45, .95)
a.set_ylabel("AUC, day 30"); a.legend(loc="upper left", fontsize=8.5)
a.set_title("No texture needed: a CNN trained on the mask alone\ngets most of the way; no organoid → chance")

# d: rescale
a = fig.add_subplot(mid[1]); tag(a, "d")
o = pd.read_csv(MP / "occlusion_spatial_idor_main_kfold4x10_std_rep1.csv")
w2 = o.pivot_table(index=["day", "organoid_id"], columns="condition", values="prob_acceptable").reset_index()
for cond, ls, nm in (("scaled_1.4", "-", "enlarged 1.4×"), ("scaled_0.7", "--", "shrunk 0.7×")):
    dl = (w2[cond] - w2["original"]).groupby(w2.day)
    med, lo, hi = dl.median(), dl.quantile(.25), dl.quantile(.75)
    a.fill_between(med.index, lo, hi, color=COL["std"], alpha=.15, lw=0)
    a.plot(med.index, med, marker="o", ls=ls, color=COL["std"], lw=2, ms=5, mec="k", mew=.4, label=nm)
a.axhline(0, color="0.6", ls=":", lw=1); day_axis(a); a.set_ylabel("change in P(Acceptable)\n(CNN, whole image)"); a.legend(loc="lower left", fontsize=8.5)
a.set_title("Only resize the organoid:\nbigger → 'more acceptable'")

# g: what no model catches
sub = gs[2].subgridspec(2, len(MISSED), wspace=.05, hspace=.3)
good = [o for o in meta if y[o] == 1]
for i, b in enumerate(MISSED):
    d = next(dd_ for dd_ in (30, 28, 24) if pd.notna(LA.loc[b].get(dd_)))
    m = min([o for o in good if plate[o] == plate[b] and pd.notna(LA.loc[o].get(d))], key=lambda o: abs(LA.loc[o, d] - LA.loc[b, d]))
    for r_, (o, col, nm) in enumerate(((b, BAD, "rejected,\nmissed by all"), (m, GOOD, "accepted,\nsame plate\n& size"))):
        a = fig.add_subplot(sub[r_, i]); im = show(o, d)
        if im is not None:
            a.imshow(im)
        frame(a, col, 2.2)
        v = meta[o]
        a.set_title(f"{short(o).replace('_96_', ' ')} day {d}\n{v['n_votes_good']}/{v['n_votes_total']} Acc. votes", fontsize=9, color=col)
        if i == 0:
            a.set_ylabel(nm, fontsize=10, color=col)
            if r_ == 0:
                tag(a, "e")
                a.text(0, 1.45, f"What no model catches: {len(MISSED)} of {len(Tb)} rejects ({len(MISSED) / len(Tb):.0%}) are called Acceptable by every "
                       "method (growth rule, all CNNs, LSTM); next to each, the same-plate accepted organoid closest in size",
                       transform=a.transAxes, fontsize=12, va="bottom")
save(fig, "fig2_what_models_look_at")

# ======================= SUPPLEMENT =======================
fig = plt.figure(figsize=(23, 11))
gs = GridSpec(2, 4, figure=fig, hspace=.45, wspace=.55)

a = fig.add_subplot(gs[0, 0]); tag(a, "a")
for k in ("std", "strong", "bbox", "brightbbox"):
    d = cnn[cnn.model == k].sort_values("day")
    a.plot(d.day, d.auc, "o-", color=COL[k], lw=1.6, ms=4, mec="k", mew=.4, label=SHORTN[k].replace("\n", " "))
for v, nm in (("mask", "CNN, mask only, whole frame"), ("maskbbox", "CNN, mask only, cropped")):
    a.plot([float(d) for d in MASK_DAYS], [mask_auc(v, d) for d in MASK_DAYS], "s--", color=COL[v], lw=1.6, ms=5, mec="k", mew=.4, label=nm)
b0 = cnn[cnn.model == "std"].sort_values("day")
a.plot(b0.day, b0.flex_auc, "-", color=SIZE2, lw=3, label="size, two cutoffs"); a.plot(b0.day, b0.size_auc, "--", color=SIZE1, lw=2, label="size, one cutoff")
a.axhline(.5, color="0.7", ls=":", lw=1); a.set_ylim(.4, .95); a.set_ylabel("AUC"); day_axis(a); a.set_xlim(2, 31); a.legend(loc="upper left", fontsize=7)
a.set_title("All CNNs and mask-only CNNs, every day")

a = fig.add_subplot(gs[0, 1]); tag(a, "b")
BL = pd.read_csv(MP / "borderline_mv3/borderline_vs_clear.csv")
b1, b2 = BL[BL.model == "trained clear only"], BL[BL.model == "trained with borderline"]
a.plot(b1.day, b1.size_clear, "-", color=SIZE2, lw=3, label="size, two cutoffs")
a.plot(b1.day, b1.cnn_clear, "o-", color=COL["brightbbox"], lw=2, ms=5, mec="k", mew=.4, label="CNN trained on clear (≥4/5) only")
a.plot(b2.day, b2.cnn_clear, "D--", color="k", lw=1.5, ms=5, label=f"CNN trained + 50 borderline (3/5 majority)")
a.axhline(.5, color="0.7", ls=":", lw=1); a.set_ylim(.4, .95); a.set_ylabel("AUC on the clear organoids"); day_axis(a); a.set_xlim(2, 31)
a.legend(loc="upper left", fontsize=7.5)
a.set_title("Adding borderline organoids (majority\nlabel) changes nothing (CNN cropped + bright.)")

a = fig.add_subplot(gs[0, 2]); tag(a, "c")
caught = (Tb < .5).where(Tb.notna()).mean() * 100
caught["any"] = 100 * (1 - len(MISSED) / len(Tb))
xs = np.arange(len(caught))
a.bar(xs, caught, color=[SIZEG] + [COL[k] for k in ("std", "strong", "bbox", "brightbbox", "LSTM mean")] + ["0.15"], edgecolor="k", lw=.5)
for i, v in enumerate(caught):
    a.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
a.set_xticks(xs); a.set_xticklabels(["growth\nrate", "CNN\nwhole", "CNN\nstrong", "CNN\ncrop", "CNN\ncrop +\nbright", "LSTM", "any\nmethod"], fontsize=8)
a.set_ylabel(f"% of the {len(Tb)} rejects caught"); a.set_ylim(0, 100)
a.set_title(f"Rejects caught by each method;\n{len(MISSED)} of {len(Tb)} by none")

a = fig.add_subplot(gs[0, 3]); tag(a, "d")
S = pd.read_csv(MP / "v2/shape_features.csv")
FEATS = ["solidity", "circularity", "elongation", "skeleton_ends"]
S["plate"] = S.organoid_id.map(plate); S["good"] = S.organoid_id.map(y)
for f in FEATS:
    ref = S[S.good == 1].groupby(["plate", "day"])[f].agg(["median", lambda s: 1.4826 * (s - s.median()).abs().median() + 1e-9])
    ref.columns = ["med", "mad"]; j = S.join(ref, on=["plate", "day"]); S[f] = (S[f] - j.med) / j.mad
Sp = S.groupby("organoid_id")[FEATS].mean()
Sp["group"] = np.where(Sp.index.isin(MISSED), "missed by all", np.where(y[Sp.index] == 1, "accepted", "other rejects"))
L = Sp.reset_index().melt(id_vars=["organoid_id", "group"], var_name="feature", value_name="z"); L["z"] = L.z.clip(-6, 6)
pal = {"accepted": GOOD, "other rejects": "0.7", "missed by all": BAD}
sns.boxplot(data=L, x="feature", y="z", hue="group", hue_order=list(pal), palette=pal, showfliers=False, linewidth=.8, ax=a)
a.axhline(0, color="0.6", ls=":", lw=1); a.set_xlabel(""); a.set_ylabel("shape vs accepted (z), days 24–30")
a.set_xticklabels(["solidity", "roundness", "elongation", "branch\nends"]); a.legend(loc="upper left", fontsize=8)
a.set_title("The missed rejects are less round, but\nshape adds no CV gain")

a = fig.add_subplot(gs[1, 0]); tag(a, "e")
c30 = cnn[cnn.day == 30].set_index("model"); l30 = lstm[lstm.window == "3-30"].set_index("model")
LAD = [("size, one cutoff", c30.loc["std", "size_auc"], SIZE1, None), ("size, two cutoffs", c30.loc["std", "flex_auc"], SIZE2, None),
       ("size + growth rate", l30.loc["LSTM mean", "flex_auc"], SIZEG, None),
       ("CNN, mask only, whole", mask_auc("mask", "30"), COL["mask"], "//"), ("CNN, mask only, cropped", mask_auc("maskbbox", "30"), COL["maskbbox"], "//")] + \
      [(SHORTN[k].replace("\n", " "), c30.loc[k, "auc"] if k in c30.index else l30.loc[k, "auc"], COL[k], None) for k in IMG]
for i, (nm, v, c, h_) in enumerate(LAD[::-1]):
    a.barh(i, v - .5, left=.5, color=c, edgecolor="k", lw=.6, height=.7, hatch=h_); a.text(v + .005, i, f"{v:.2f}", va="center", fontsize=9)
a.set_yticks(range(len(LAD))); a.set_yticklabels([x_[0] for x_ in LAD[::-1]], fontsize=9); a.set_xlim(.5, .95); a.set_xlabel("AUC, day 30")
a.set_title("Every model at day 30")
# f: test-time ablation on the trained CNNs
a = fig.add_subplot(gs[1, 1]); tag(a, "f")
occ = []
for s in ("std", "strong"):
    o = pd.read_csv(MP / f"occlusion_spatial_idor_main_kfold4x10_{s}_rep1.csv")
    for (dd_, cond), g2 in o[o.day.isin([24, 28, 30])].groupby(["day", "condition"]):
        g2 = g2.dropna(subset=["prob_acceptable"]); occ.append(dict(setup=s, condition=cond, auc=roc_auc_score(g2.true_label, g2.prob_acceptable)))
occ = pd.DataFrame(occ)
order = ["original", "silhouette", "core_texture_removed", "no_organoid"]
sns.barplot(data=occ, x="condition", y="auc", hue="setup", order=order, hue_order=["std", "strong"], palette=[COL["std"], COL["strong"]],
            errorbar="sd", capsize=.08, edgecolor="k", linewidth=.6, ax=a)
a.set_xticklabels(["original", "mask\nonly", "inside\ntexture\nremoved", "organoid\nremoved"], fontsize=9)
a.axhline(.5, color="0.6", ls=":", lw=1); a.set_ylim(.3, .95); a.set_xlabel(""); a.set_ylabel("AUC, days 24–30")
h, _ = a.get_legend_handles_labels(); a.legend(h[:2], ["CNN, whole image", "CNN + strong aug."], loc="upper right", fontsize=8)
a.set_title("Edit the test image: mask only still\nworks; no organoid → chance")

# g: LSTM day blanking
a = fig.add_subplot(gs[1, 2]); tag(a, "g")
dd = pd.read_csv(MP / "v2/lstm_day_dropout.csv")
g = dd[(dd.readout == "mean") & (dd.window == "3-30")]
base = g.drop_duplicates("organoid_id").set_index("organoid_id")
out = []
for day, h_ in g.groupby("day"):
    pb_ = base.p_full.copy(); pb_[h_.organoid_id] = h_.set_index("organoid_id").p_blank
    out.append((day, roc_auc_score(base.label, pb_) - roc_auc_score(base.label, base.p_full)))
out = np.array(out)
a.bar(range(len(out)), out[:, 1], color=COL["LSTM mean"], edgecolor="k", lw=.5)
a.set_xticks(range(len(out))); a.set_xticklabels([f"{d:g}" for d in out[:, 0]], fontsize=8, rotation=90); a.set_xlabel("Day blanked")
a.axhline(0, color="0.5", lw=.8); a.set_ylabel("change in LSTM AUC"); a.set_ylim(-.06, .06)
a.set_title(f"LSTM: blanking any one day\nchanges AUC by ≤ {np.abs(out[:, 1]).max():.2f}")

save(fig, "figS_supplement")
