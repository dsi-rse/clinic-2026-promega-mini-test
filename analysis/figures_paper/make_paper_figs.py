"""Paper figures for the image-model part: Fig A (single-day CNNs + trajectory LSTM vs size)
and Fig B (what the models use, and the rejects no method catches).

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
T["LSTM"] = pd.concat([pd.read_csv(f) for f in sorted((RUNS / "temporal_lstm_kfold4x10_last/days_3-30").glob("oof_rep*.csv"))]
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


# ======================= FIG A =======================
cnn = pd.read_csv(V2R / "perday_cnn.csv"); lstm = pd.read_csv(V2R / "perday_lstm.csv")
fig = plt.figure(figsize=(16, 21))
gs = GridSpec(4, 3, figure=fig, height_ratios=[1.05, 1, 1, 1], hspace=.45, wspace=.3)

# same three organoids as the Grad-CAM examples in Fig B
EXAMPLES = ["BA1_96_1_H6_nosplit", "BA2_96_2_C3_nosplit", "BA2_96_2_E12_nosplit"]
sub = gs[0, :].subgridspec(3, 10, wspace=.06, hspace=.1, width_ratios=[1, 1, 1, 1, .25, 1, 1, 1, 1, 1])
INP = [("CNN standard / strong", False, False), ("CNN brightened", False, True), ("CNN bbox", True, False), ("CNN brightened bbox", True, True)]
for i, o in enumerate(EXAMPLES):
    col = GOOD if y[o] == 1 else BAD
    for j, (nm, bb, br) in enumerate(INP):
        a = fig.add_subplot(sub[i, j]); im = show(o, 30, bb, br)
        if im is not None:
            a.imshow(im)
        frame(a, col)
        if i == 0:
            a.set_title(nm + "\n(day 30)", fontsize=10)
        if j == 0:
            a.set_ylabel(f"{short(o).replace('_96_', ' ')}\n{'Acc.' if col == GOOD else 'Not Acc.'} "
                         f"{meta[o]['n_votes_good']}/{meta[o]['n_votes_total']}", fontsize=9, color=col)
            if i == 0:
                tag(a, "a")
    for j, d in enumerate((3, 10, 17, 24, 30)):
        a = fig.add_subplot(sub[i, 5 + j]); im = show(o, d, False, False)
        if im is not None:
            a.imshow(im)
        else:
            a.text(.5, .5, "no image", ha="center", va="center", transform=a.transAxes, color="0.5")
        frame(a, col)
        if i == 0:
            a.set_title(("LSTM sequence\n" if j == 0 else "\n") + f"day {d}", fontsize=10)

# b: growth curves, rejects split into stalled vs overgrown (latest late-day size vs same-plate accepted median)
a = fig.add_subplot(gs[1, :2]); tag(a, "b")
FRAME = 575 ** 2
acc_med = {(pl, d): LA.loc[[o for o in LA.index if y[o] == 1 and plate[o] == pl], d].median() for pl in plate.unique() for d in DAYS}
kind = {}
for o in LA.index:
    if y[o] == 1:
        kind[o] = "Acceptable"
    else:
        d = next((dd_ for dd_ in (30, 28, 24) if pd.notna(LA.loc[o].get(dd_))), None)
        kind[o] = "Not Acc., overgrown" if d and LA.loc[o, d] > acc_med[(plate[o], d)] else "Not Acc., stalled"
kind = pd.Series(kind)
GC = {"Acceptable": GOOD, "Not Acc., stalled": BAD, "Not Acc., overgrown": "#b2182b"}
for k, c in GC.items():
    A_ = np.exp(LA.loc[kind.index[kind == k]]) / FRAME * 100
    med, lo, hi = A_.median(), A_.quantile(.25), A_.quantile(.75)
    a.fill_between(med.index, lo, hi, color=c, alpha=.18, lw=0)
    a.plot(med.index, med, "o-", color=c, lw=2.4, ms=6, mec="k", mew=.5, ls="--" if "overgrown" in k else "-",
           label=f"{k} (n={int((kind == k).sum())})")
a.axvspan(14.5, 17.5, color="#fde7c8", zorder=0); a.text(16, .02, "groups split\n(days 15→17)", transform=a.get_xaxis_transform(), ha="center", fontsize=10, color="0.35")
a.set_yscale("log"); a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
a.set_ylabel("Organoid area (% of frame), median ± IQR"); day_axis(a); a.legend(loc="upper left")
a.set_title("Accepted organoids grow in a band; rejects stall or overgrow after day 15")

a = fig.add_subplot(gs[1, 2]); tag(a, "c")
d30 = pd.DataFrame({"a": LA[30], "y": y}).dropna()
d30["fifth"] = pd.qcut(d30.a, 5, labels=["1\nsmallest", "2", "3", "4", "5\nlargest"])
pb = (1 - d30.groupby("fifth").y.mean()) * 100
a.bar(range(5), pb, color=[BAD, "0.75", "0.75", "0.75", "#b2182b"], edgecolor="k", lw=.6)
for i, v in enumerate(pb):
    a.text(i, v + 1.2, f"{v:.0f}%", ha="center", fontsize=10)
a.set_xticks(range(5)); a.set_xticklabels(pb.index); a.set_xlabel("Day-30 size fifth (20% of organoids each)")
a.set_ylabel("% Not Acceptable"); a.set_title(f"% rejected by day-30 size (n={len(d30)})\n→ size must be modelled two-way")

a = fig.add_subplot(gs[2, 0]); tag(a, "d")
b0 = cnn[cnn.model == "std"].sort_values("day")
a.plot(b0.day, b0.flex_auc, "-", color="k", lw=2.6, label="two-way size (spline)")
a.plot(b0.day, b0.size_auc, "--", color="0.5", lw=2, label="bigger = better size")
for k, lab, c, mk in CNN:
    d = cnn[cnn.model == k].sort_values("day")
    a.plot(d.day, d.auc, marker=mk, color=c, lw=1.6, ms=6, mec="k", mew=.5, label=lab)
a.axhline(.5, color="0.7", ls=":", lw=1); a.set_ylim(.4, .95); a.set_ylabel("AUC (4×10 CV)"); day_axis(a)
a.set_title("Single-day CNNs vs size, same organoids"); a.legend(loc="upper left", ncol=1)

a = fig.add_subplot(gs[2, 1]); tag(a, "e")
W =["3-13", "3-17", "3-24", "3-30"]; x = np.arange(4)
l0 = lstm[lstm.model == "LSTM last"].set_index("window").loc[W]
a.plot(x, l0.flex_auc, "-", color="k", lw=2.6, label="two-way size + growth")
a.plot(x, l0.size_auc, "--", color="0.5", lw=2, label="size + growth (linear)")
for k, c, mk in (("LSTM last", "#4d4d4d", "o"), ("LSTM mean", "#9e9ac8", "s")):
    d = lstm[lstm.model == k].set_index("window").loc[W]
    a.plot(x, d.auc, marker=mk, color=c, lw=1.8, ms=7, mec="k", mew=.5, label=k.replace("LSTM", "LSTM,") + " readout")
a.set_xticks(x); a.set_xticklabels([f"days {w}" for w in W]); a.axhline(.5, color="0.7", ls=":", lw=1)
a.set_ylim(.4, .95); a.set_ylabel("AUC (4×10 CV)"); a.set_title("Trajectory LSTM vs size history"); a.legend(loc="upper left")

a = fig.add_subplot(gs[3, :2]); tag(a, "g")
H = pd.concat([cnn, lstm.assign(day=lstm.max_day)])
rows = [k for k, *_ in CNN] + ["LSTM last", "LSTM mean"]
H["row"] = H.model
hm = H.pivot_table(index="row", columns="day", values="gain_flex").loc[rows]
lab = hm.apply(lambda c: c.map(lambda v: "" if pd.isna(v) else ("0" if abs(v) < .005 else f"{v:+.2f}".replace("0.", "."))))
sns.heatmap(hm, cmap="RdBu_r", center=0, vmin=-.08, vmax=.08, annot=lab, fmt="", annot_kws={"fontsize": 7},
            cbar_kws={"label": "Δ AUC", "shrink": .8}, ax=a, linewidths=.4, linecolor="w")
a.set_yticklabels(["CNN std", "CNN strong", "CNN bbox", "CNN br. bbox", "LSTM last", "LSTM mean"], rotation=0)
a.set_xticklabels([f"{c:g}" for c in hm.columns], rotation=0, fontsize=8); a.set_xlabel("Day (LSTM: last day of window)"); a.set_ylabel("")
a.set_title("Gain over two-way size + plate")

a = fig.add_subplot(gs[2, 2]); tag(a, "f")
for k, lab, c, mk in CNN:
    d = cnn[cnn.model == k].sort_values("day")
    a.plot(d.day, d.same, marker=mk, color=c, lw=1.6, ms=6, mec="k", mew=.5, label=lab)
a.axhline(.5, color="0.5", ls="--", lw=1.2); a.set_ylim(.3, .85); day_axis(a)
a.set_ylabel("AUC within same-size pairs (±10%)"); a.set_title("With size held fixed, CNNs are near chance"); a.legend(loc="upper left", ncol=2)

a = fig.add_subplot(gs[3, 2]); tag(a, "h")
p = oof_mean(RUNS / "base_effnet_kfold4x10_brightbbox/day_30/oof_predictions.csv")
d = pd.DataFrame({"p": p, "a": LA[30], "y": y}).dropna()
for lab, c, nm in ((1, GOOD, "Acceptable"), (0, BAD, "Not Acceptable")):
    s = d[d.y == lab]; a.scatter(np.exp(s.a) / 1e3, s.p, s=22, color=c, edgecolor="k", lw=.4, alpha=.85, label=nm)
a.set_xscale("log"); a.axhline(.5, color="0.6", ls="--", lw=1)
a.set_xlabel("Day-30 organoid area (×1000 px)"); a.set_ylabel("CNN P(Acceptable), OOF mean")
a.set_title("Bright. bbox CNN, day 30: small or big = bad"); a.legend(loc="upper left")
a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
[sns.despine(ax=ax_) for ax_ in fig.axes if not ax_.images]
for e in ("png", "pdf"):
    fig.savefig(OUT / f"figA_image_models_vs_size.{e}")
plt.close(fig); print("saved A")

# ======================= FIG B =======================
fig = plt.figure(figsize=(16, 21))
gs = GridSpec(4, 3, figure=fig, height_ratios=[1, 1.15, 1, 1.15], hspace=.45, wspace=.3)

a = fig.add_subplot(gs[0, 0]); tag(a, "a")
for s, lab, c in (("std", "standard aug", S2[2]), ("strong", "strong aug", S2[1])):
    o = pd.read_csv(MP / f"occlusion_spatial_idor_main_kfold4x10_{s}_rep1.csv")
    w = o.pivot_table(index=["day", "organoid_id"], columns="condition", values="prob_acceptable").reset_index()
    for cond, ls, nm in (("scaled_1.4", "-", "enlarged 1.4×"), ("scaled_0.7", "--", "shrunk 0.7×")):
        med = (w[cond] - w["original"]).groupby(w.day).median()
        a.plot(med.index, med, ls=ls, marker="o", color=c, lw=1.8, ms=5, mec="k", mew=.4, label=f"{nm}, {lab}")
a.axhline(0, color="0.6", ls=":", lw=1); day_axis(a); a.set_ylabel("Δ P(Acceptable), median")
a.set_title("Rescaling the same organoid moves the CNN"); a.legend(loc="lower left", fontsize=8)

a = fig.add_subplot(gs[0, 1]); tag(a, "b")
occ = []
for s in ("std", "strong"):
    o = pd.read_csv(MP / f"occlusion_spatial_idor_main_kfold4x10_{s}_rep1.csv")
    for (dd, cond), g2 in o[o.day.isin([24, 28, 30])].groupby(["day", "condition"]):
        g2 = g2.dropna(subset=["prob_acceptable"]); occ.append(dict(setup=s, condition=cond, auc=roc_auc_score(g2.true_label, g2.prob_acceptable)))
occ = pd.DataFrame(occ)
order = ["original", "silhouette", "rim_texture_removed", "core_texture_removed", "no_organoid"]
sns.barplot(data=occ, x="condition", y="auc", hue="setup", order=order, hue_order=["std", "strong"], palette=[S2[2], S2[1]],
            errorbar="sd", capsize=.08, edgecolor="k", linewidth=.6, ax=a)
a.set_xticklabels(["original", "mask only\n(no texture)","rim texture\nremoved", "core texture\nremoved", "organoid\nremoved"], fontsize=9)
a.axhline(.5, color="0.6", ls=":", lw=1); a.set_ylim(.3, .95); a.set_xlabel(""); a.set_ylabel("AUC (days 24/28/30)")
h, _ = a.get_legend_handles_labels(); a.legend(h[:2], ["standard aug", "strong aug"], loc="upper right")
a.set_title("The mask alone keeps the signal")

a = fig.add_subplot(gs[0, 2]); tag(a, "c")
for key, lab, c, mk in (("std", "standard aug", S2[2], "o"), ("strong", "strong aug", S2[1], "^"), ("bboxstd", "bbox crop", S2[3], "s")):
    r = pd.read_csv(MP / f"focus_pred_idor_main_kfold4x10_{key}_rep1.csv")
    r["contrast"] = (r.focus / r.mask_frac) / ((1 - r.focus) / (1 - r.mask_frac))
    med = r.groupby("day").contrast.median()
    a.plot(med.index, med, marker=mk, color=c, lw=1.8, ms=6, mec="k", mew=.5, label=lab)
a.set_yscale("log"); a.axhline(1, color="0.6", ls=":", lw=1); day_axis(a)
a.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.25, 0.5, 1, 2, 4, 8]))
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×")); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.set_ylabel("Grad-CAM heat density,\norganoid ÷ background"); a.set_title("Where Grad-CAM heat falls (1× = no focus)"); a.legend(loc="upper left")

# d: Grad-CAM examples, cropped from the V1 rotation figures (0° input, std and strong overlays)
sub = gs[1, :2].subgridspec(3, 3, wspace=.04, hspace=.12)
EX = [("Dy30_acc_bothcorrect_BA1_96_1_H6.png", "BA1_96_1_H6_nosplit"), ("Dy30_notacc_bothcaught_BA2_96_2_C3.png", "BA2_96_2_C3_nosplit"),
      ("Dy30_notacc_bothmissed_BA2_96_2_E12.png", "BA2_96_2_E12_nosplit")]
BOX = {"input": (.0915, .2835), "std": (.3015, .4925), "strong": (.8015, .9925)}
for i, (fn, oid) in enumerate(EX):
    im = Image.open(V1F / "gradcam_examples" / fn).convert("RGB"); W_, H_ = im.size
    v = meta1[oid]
    for j, (k, nm) in enumerate((("input", "input (day 30)"), ("std", "Grad-CAM, standard aug"), ("strong", "Grad-CAM, strong aug"))):
        y0, y1 = BOX[k]; a = fig.add_subplot(sub[i, j])
        a.imshow(im.crop((int(.008 * W_), int(y0 * H_), int(.218 * W_), int(y1 * H_)))); a.set_xticks([]); a.set_yticks([])
        if i == 0:
            a.set_title(nm, fontsize=11)
        if j == 0:
            col = GOOD if v["label"] == "Acceptable" else BAD
            a.set_ylabel(f"{short(oid).replace('_96_', ' ')}\n{'Acc.' if col == GOOD else 'Not Acc.'} {v['n_votes_good']}/{v['n_votes_total']}", fontsize=9, color=col)
        if i == 0 and j == 0:
            a.text(-0.55, 1.12, "d", transform=a.transAxes, fontsize=16, fontweight="bold", va="bottom")

a = fig.add_subplot(gs[1, 2]); tag(a, "e")
dd = pd.read_csv(MP / "v2/lstm_day_dropout.csv")
SER = [("last", "3-30", "#4d4d4d", "o", "-"), ("last", "3-24", "#4d4d4d", "s", "--"), ("mean", "3-30", "#9e9ac8", "o", "-")]
for ro, w, c, mk, ls in SER:
    g = dd[(dd.readout == ro) & (dd.window == w)]
    base = g.drop_duplicates("organoid_id").set_index("organoid_id")
    out = []
    for day, h in g.groupby("day"):
        pdrop = base.p_full.copy(); pdrop[h.organoid_id] = h.set_index("organoid_id").p_drop
        out.append((day, roc_auc_score(base.label, pdrop) - roc_auc_score(base.label, base.p_full)))
    out = np.array(out); a.plot(out[:, 0], out[:, 1], marker=mk, ls=ls, color=c, lw=1.8, ms=6, mec="k", mew=.5, label=f"{ro} readout, days {w}")
a.axhline(0, color="0.6", ls=":", lw=1); a.axvspan(14.2, 17.8, color="#fde7c8", zorder=0); day_axis(a)
a.text(16, .015, "growth\npivot", ha="center", fontsize=8, color="0.4")
a.set_ylabel("Δ AUC when that day is removed"); a.set_title("LSTM: only the last frame matters"); a.legend(loc="lower left", fontsize=8)

a = fig.add_subplot(gs[2, 0]); tag(a, "f")
for ro, w, c, mk, ls in SER:
    g = dd[(dd.readout == ro) & (dd.window == w)]
    s = (g.p_drop - g.p_full).abs().groupby(g.day).mean()
    a.plot(s.index, s, marker=mk, ls=ls, color=c, lw=1.8, ms=6, mec="k", mew=.5, label=f"{ro}, days {w}")
a.axvspan(14.2, 17.8, color="#fde7c8", zorder=0); day_axis(a)
a.set_ylabel("mean |Δ P(Acceptable)|"); a.set_title("LSTM prediction change per removed day"); a.legend(loc="upper left", fontsize=8)

a = fig.add_subplot(gs[2, 1]); tag(a, "g")
caught = (Tb < .5).where(Tb.notna()).mean() * 100          # among rejects each method can score
caught["any"] = 100 * (1 - len(MISSED) / len(Tb))
xs = np.arange(len(caught))
a.bar(xs, caught, color=[BAD] * (len(xs) - 1) + ["#8c2d04"], edgecolor="k", lw=.5)
for i, v in enumerate(caught):
    a.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
a.set_xticks(xs); a.set_xticklabels(["growth\nband", "CNN\nstd", "CNN\nstrong", "CNN\nbbox", "CNN br.\nbbox", "LSTM", "any\nmethod"], fontsize=9)
a.set_ylabel(f"% of rejects caught (OOF P < 0.5, n={len(Tb)})"); a.set_ylim(0, 100)
a.set_title(f"{len(MISSED)} of {len(Tb)} rejects ({len(MISSED) / len(Tb):.0%}) caught by no method")

a = fig.add_subplot(gs[2, 2]); tag(a, "h")
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
a.axhline(0, color="0.6", ls=":", lw=1); a.set_xlabel(""); a.set_ylabel("z vs accepted, days 24–30")
a.set_xticklabels(["solidity", "circularity", "elongation", "branch\nends"]); a.legend(loc="upper left", fontsize=8)
a.set_title("Missed rejects are less round, but shape\nadds no CV gain (flags many good too)")
[sns.despine(ax=ax_) for ax_ in fig.axes if not ax_.images]

# i: the missed rejects (latest late day) vs a same-plate, same-day, same-size accepted organoid
sub = gs[3, :].subgridspec(2, len(MISSED), wspace=.05, hspace=.22)
good = [o for o in meta if y[o] == 1]
for i, b in enumerate(MISSED):
    d = next(dd_ for dd_ in (30, 28, 24) if pd.notna(LA.loc[b].get(dd_)))
    m = min([o for o in good if plate[o] == plate[b] and pd.notna(LA.loc[o].get(d))], key=lambda o: abs(LA.loc[o, d] - LA.loc[b, d]))
    for r_, (o, col, nm) in enumerate(((b, BAD, "rejected"), (m, GOOD, "matched accepted"))):
        a = fig.add_subplot(sub[r_, i]); im = show(o, d)
        if im is not None:
            a.imshow(im)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(2.2)
        v = meta[o]
        a.set_title(f"{short(o).replace('_96_', ' ')} d{d}\n{v['n_votes_good']}/{v['n_votes_total']} Acc. votes", fontsize=9, color=col)
        if i == 0:
            a.set_ylabel(nm, fontsize=11, color=col)
        if i == 0 and r_ == 0:
            tag(a, "i")
for e in ("png", "pdf"):
    fig.savefig(OUT / f"figB_what_models_use.{e}")
print("saved B")
