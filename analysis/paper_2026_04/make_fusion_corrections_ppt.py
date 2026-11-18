#!/usr/bin/env python3
"""PPT: which organoids are corrected / broken when fusing models.

Reads fusion_corrections_{per_organoid,summary}.csv (from fusion_corrections.py).

Usage:  python3 -m analysis.paper_2026_04.make_fusion_corrections_ppt
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

IMG = "analysis_output/images/"
OUT_PPT = "figures/fusion_corrections.pptx"
DAYS = ["Dy03", "Dy06", "Dy08", "Dy10", "Dy13", "Dy15", "Dy17", "Dy20_5", "Dy24", "Dy28", "Dy30"]
FUSIONS = {
    "all_lgbm": "met LGBM + morph LGBM + img",
    "all_logreg": "met LR + morph LR + img",
    "mixed": "met LR + morph LGBM + img",
    "mixed2": "met LGBM + morph LR + img",
}
HEAD = "mixed2"
GREEN, RED, GREY = "#2ca02c", "#d62728", "#999999"

SLIDE_W, SLIDE_H, M = Inches(13.33), Inches(7.5), Inches(0.3)


def _hex(h):
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _tb(slide, text, left, top, width, height, fontsize=18, bold=False,
        color="#000000", align=PP_ALIGN.LEFT):
    tf = slide.shapes.add_textbox(left, top, width, height).text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.color.rgb = Pt(fontsize), bold, _hex(color)


def _title_slide(prs, title, subtitle):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = _hex("#1F3864")
    _tb(s, title, M, Inches(2.2), Inches(12.5), Inches(1.6), 38, True, "#FFFFFF", PP_ALIGN.CENTER)
    _tb(s, subtitle, M, Inches(4.2), Inches(12.5), Inches(1.5), 16, False, "#BDD7EE", PP_ALIGN.CENTER)


def _fig_slide(prs, heading, fig, note=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _tb(s, heading, M, M, Inches(12.5), Inches(0.55), 22, True, "#1F3864")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    h = Inches(5.6) if note else Inches(6.4)
    pic = s.shapes.add_picture(buf, M, Inches(0.85), height=h)
    if pic.width > Inches(12.7):
        r = Inches(12.7) / pic.width
        pic.width, pic.height = int(pic.width * r), int(pic.height * r)
    if note:
        _tb(s, note, M, Inches(6.55), Inches(12.5), Inches(0.9), 14, False, "#333333")
    plt.close(fig)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def plot_diverging(summ):
    d = summ[summ.fusion == HEAD].set_index("day").reindex(DAYS)
    x = np.arange(len(DAYS))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x, d.corrected, color=GREEN, label="corrected (≥1 source wrong)")
    ax.bar(x, d.rescued, bottom=d.corrected, color="#98df8a", label="rescued (majority of sources wrong)")
    ax.bar(x, -d.broken, color=RED, label="broken (fusion wrong, ≥1 source right)")
    for i, (c, b) in enumerate(zip(d.corrected + d.rescued, d.broken)):
        ax.text(i, c + 0.6, f"+{c}", ha="center", fontsize=9, color=GREEN)
        ax.text(i, -b - 2.2, f"−{b}", ha="center", fontsize=9, color=RED)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(DAYS, rotation=45)
    ax.set_ylabel("# organoids"); ax.set_ylim(-38, 55)
    ax.set_title(f"{HEAD}: {FUSIONS[HEAD]} (n≈140)", fontsize=11)
    ax.legend(fontsize=9, ncol=3, loc="upper center", frameon=False); _style(ax)
    return fig


def plot_net(summ):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for fk, c in zip(FUSIONS, ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"]):
        d = summ[summ.fusion == fk].set_index("day").reindex(DAYS)
        ax.plot(DAYS, d.net_gain, "-o", color=c, label=f"{fk}: {FUSIONS[fk]}  (Σ={int(d.net_gain.sum()):+d})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Net organoids gained\n(corrected + rescued − broken)")
    ax.tick_params(axis="x", rotation=45); ax.legend(fontsize=9, frameon=False); _style(ax)
    return fig


def plot_label_split(df):
    df = df.copy()
    df["chg"] = df.category.map(lambda c: "corrected" if c in ("corrected", "rescued")
                                else "broken" if c.startswith("broken") else "none")
    df = df[df.chg != "none"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, chg, col in zip(axes, ["corrected", "broken"], [GREEN, RED]):
        t = (df[df.chg == chg].groupby(["fusion", "label"]).size().unstack(fill_value=0)
             .reindex(list(FUSIONS), fill_value=0))
        x = np.arange(len(t))
        ax.bar(x - 0.2, t[0], 0.4, color=col, label="Not Acceptable")
        ax.bar(x + 0.2, t[1], 0.4, color=col, alpha=0.45, hatch="//", label="Acceptable")
        ax.set_xticks(x); ax.set_xticklabels(t.index, rotation=20)
        ax.set_title(f"{chg} (organoid-days, all 11 days)"); ax.legend(frameon=False); _style(ax)
    axes[0].set_ylabel("# organoid-days")
    return fig


def plot_heatmap(df, top=30):
    d = df[df.fusion == HEAD].copy()
    d["v"] = d.category.map(lambda c: 1 if c in ("corrected", "rescued")
                            else -1 if c.startswith("broken") else 0)
    M_ = d.pivot(index="organoid", columns="day", values="v").reindex(columns=DAYS)
    lab = d.groupby("organoid").label.first()
    score = M_.abs().sum(axis=1)
    keep = score.sort_values(ascending=False).head(top).index
    M_ = M_.loc[keep]
    order = (M_.sum(axis=1)).sort_values(ascending=False).index
    M_ = M_.loc[order]
    fig, ax = plt.subplots(figsize=(8, 9))
    cm = matplotlib.colors.ListedColormap([RED, "#f2f2f2", GREEN])
    ax.imshow(M_.values, cmap=cm, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(DAYS))); ax.set_xticklabels(DAYS, rotation=45)
    ax.set_yticks(range(len(M_)))
    ax.set_yticklabels([f"{o}  [{'Acc' if lab[o] == 1 else 'NotAcc'}]" for o in M_.index], fontsize=8)
    ax.set_title(f"{HEAD}: {top} organoids with most corrections/breakages\n"
                 "green = corrected, red = broken, grey = unchanged", fontsize=10)
    return fig


def main():
    summ = pd.read_csv(IMG + "fusion_corrections_summary.csv")
    df = pd.read_csv(IMG + "fusion_corrections_per_organoid.csv")
    tot = summ.groupby("fusion").net_gain.sum()
    h = summ[summ.fusion == HEAD]

    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    _title_slide(prs, "Which organoids are corrected by model fusion?",
                 "Per-organoid comparison of fused (mean-probability) vs. single-source predictions\n"
                 "met + morph + img, OOF probabilities averaged over 10 repeats of 4-fold CV")
    _fig_slide(prs, f"Corrected vs. broken organoids per day ({HEAD})", plot_diverging(summ),
               "Corrected = fused call right while ≥1 single-source model wrong. Broken = fused call wrong "
               "while ≥1 source right. Fusion never breaks an organoid on which all sources agree and are correct.")
    _fig_slide(prs, "Net gain per day across fusion sets", plot_net(summ),
               "Largest gains are mid-to-late (Dy15–Dy24); by Dy28–Dy30 few errors remain to fix. "
               "all_lgbm is net-negative at Dy13/Dy15.")
    _fig_slide(prs, "Corrections are mostly Not Acceptable organoids", plot_label_split(df),
               "Single-source models over-call Acceptable; averaging pulls those organoids back. "
               "Breakages are split more evenly between the two classes.")
    _fig_slide(prs, f"Repeat corrected / broken organoids ({HEAD})", plot_heatmap(df))
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _tb(s, "Takeaways", M, M, Inches(12.5), Inches(0.55), 22, True, "#1F3864")
    _tb(s, "\n".join([
        f"• Net organoid-days gained over 11 days: " + ", ".join(f"{k} {int(v):+d}" for k, v in tot.sort_values(ascending=False).items()),
        "• ~79% of corrections are Not Acceptable organoids that single-source models call Acceptable.",
        "• Fusion breaks Acceptable organoids where the minority-right source is outvoted.",
        "• A small set of organoids is repeatedly corrected (e.g. BA2 96_1 F5, 9/11 days) or repeatedly broken (e.g. BA2 96_2 G7, 8/11 days).",
        "• Caveat: label coding 1 = Acceptable assumed; counts are organoid-days, not independent organoids.",
    ]), M, Inches(1.1), Inches(12.5), Inches(5), 18)
    prs.save(OUT_PPT)
    print("Saved", OUT_PPT)


if __name__ == "__main__":
    main()
