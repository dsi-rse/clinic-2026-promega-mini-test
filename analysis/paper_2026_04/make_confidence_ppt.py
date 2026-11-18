#!/usr/bin/env python3
"""Build figures/confidence_vs_survey.pptx — data, experiments and results for the
model-confidence vs human-survey analysis.

Reads (no recomputation, rule 16):
  analysis_output/images/confidence_vs_survey.json            (Exp A, 139 cohort, 4 modalities)
  analysis_output/images/confidence_zones_{Dy24,Dy28,Dy30}.json (Exp B, 248 IDOR + 96 BA4)
  analysis_output/figures/{data_overview,confidence_*}.png

Usage (sbatch; see submit_confidence_ppt.slurm):
    python3 -m analysis.paper_2026_04.make_confidence_ppt
"""

import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

from pipeline.data_loader import ANALYSIS_OUTPUT_DIR, FIGURE_DIR

IMG_DIR = ANALYSIS_OUTPUT_DIR / "images"
OUT = Path("figures/confidence_vs_survey.pptx")

# ---- palette: deep teal-ink dominant, coral accent, colorblind-safe modality hues ----
INK = RGBColor(0x12, 0x34, 0x3B)
CORAL = RGBColor(0xE4, 0x57, 0x2E)
TINT = RGBColor(0xEA, 0xF2, 0xF3)
CARD = RGBColor(0xF4, 0xF8, 0xF8)
MUTED = RGBColor(0x5B, 0x7C, 0x84)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TEXT = RGBColor(0x1F, 0x2D, 0x31)
MOD_COLOR = {
    "met_nan": RGBColor(0xE6, 0x9F, 0x00),
    "morph": RGBColor(0x00, 0x9E, 0x73),
    "img": RGBColor(0x56, 0xB4, 0xE9),
    "fusion": INK,
}
MOD_LABEL = {"met_nan": "Metabolite", "morph": "Morphology", "img": "Image", "fusion": "Fusion (mean)"}
HEAD, BODY = "Cambria", "Calibri"
W, H = 13.333, 7.5

DAYS_A = ["Dy03", "Dy06", "Dy08", "Dy10", "Dy13", "Dy15", "Dy17", "Dy20_5", "Dy24", "Dy28", "Dy30"]
DAY_LBL = {d: d.replace("Dy", "D").replace("_5", ".5") for d in DAYS_A}


# ------------------------------------------------------------------ helpers
def _run(p, text, size, bold=False, color=TEXT, font=BODY, italic=False):
    r = p.add_run()
    r.text = text
    r.font.size, r.font.bold, r.font.italic = Pt(size), bold, italic
    r.font.color.rgb, r.font.name = color, font
    return r


def txt(slide, x, y, w, h, text, size=16, bold=False, color=TEXT, font=BODY, align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP, name=None, italic=False):
    """Text box; ``text`` may be a str or a list of (text, {overrides}) paragraphs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = text if isinstance(text, list) else [(text, {})]
    for i, (t, o) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = o.get("align", align)
        if "space_after" in o:
            p.space_after = Pt(o["space_after"])
        _run(p, t, o.get("size", size), o.get("bold", bold), o.get("color", color), o.get("font", font),
             o.get("italic", italic))
    return tb


def bullets(slide, x, y, w, h, items, size=16, color=TEXT, gap=8, name=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Inches(0.28))))
        pPr.set("indent", str(-int(Inches(0.28))))
        bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
        pPr.append(bu)
        if isinstance(it, tuple):  # (bold lead, rest)
            _run(p, it[0], size, True, color)
            _run(p, it[1], size, False, color)
        else:
            _run(p, it, size, False, color)
    return tb


def box(slide, x, y, w, h, fill=CARD, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.06
    if name:
        s.name = name
    return s


def stat(slide, x, y, w, h, big, label, big_color=CORAL, fill=CARD, big_size=40):
    box(slide, x, y, w, h, fill, name="stat-card")
    vh = big_size / 72 * 1.35  # value box height scales with its font size
    txt(slide, x + 0.25, y + 0.2, w - 0.5, vh, big, big_size, True, big_color, HEAD, name="stat-value")
    ly = y + 0.2 + vh + 0.05
    txt(slide, x + 0.25, ly, w - 0.5, y + h - ly - 0.1, label, 14, False, TEXT, name="stat-label")


def new_slide(prs, title, n, source=None, dark=False, notes=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = INK if dark else WHITE
    if title:
        txt(s, 0.6, 0.4, W - 1.2, 1.0, title, 30, True, WHITE if dark else INK, HEAD,
            anchor=MSO_ANCHOR.MIDDLE, name="title")
    if source:
        txt(s, 0.6, 7.02, 10.5, 0.3, source, 10, False, MUTED, name="source")
    txt(s, W - 1.2, 7.02, 0.6, 0.3, str(n), 10, False, MUTED, align=PP_ALIGN.RIGHT, name="slide-number")
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def img_fit(slide, path, x, y, w, h, name=None):
    iw, ih = Image.open(path).size
    sc = min(w / iw, h / ih)
    pw, ph = iw * sc, ih * sc
    pic = slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                                   Inches(pw), Inches(ph))
    if name:
        pic.name = name
    return pic


def style_chart(chart, legend=True, vmin=None, vmax=None, vfmt="0.0", vtitle=None, size=12):
    chart.font.size = Pt(size)
    chart.font.name = BODY
    chart.font.color.rgb = MUTED
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
    va, ca = chart.value_axis, chart.category_axis
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0xD9, 0xE3, 0xE5)
    va.major_gridlines.format.line.width = Pt(0.75)
    va.format.line.fill.background()
    va.tick_labels.number_format = vfmt
    va.tick_labels.number_format_is_linked = False
    ca.has_major_gridlines = False
    ca.format.line.color.rgb = RGBColor(0xB8, 0xC8, 0xCB)
    if vtitle:
        va.has_title = True
        va.axis_title.text_frame.text = vtitle
        r = va.axis_title.text_frame.paragraphs[0].runs[0]
        r.font.size, r.font.bold, r.font.color.rgb = Pt(size), False, MUTED


def line_chart(slide, x, y, w, h, cats, series, colors, **kw):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series:
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    style_chart(ch, **kw)
    for s, c in zip(ch.plots[0].series, colors):
        s.smooth = False
        s.format.line.color.rgb = c
        s.format.line.width = Pt(2.75 if c == INK else 2.0)
        s.marker.style = XL_MARKER_STYLE.CIRCLE
        s.marker.size = 7
        s.marker.format.fill.solid()
        s.marker.format.fill.fore_color.rgb = c
        s.marker.format.line.color.rgb = c
    return gf


def bar_chart(slide, x, y, w, h, cats, series, colors, labels=True, fmt="0.00", **kw):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals in series:
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    style_chart(ch, **kw)
    pl = ch.plots[0]
    pl.gap_width = 60
    pl.overlap = -5
    for s, c in zip(pl.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = c
    if labels:
        pl.has_data_labels = True
        dl = pl.data_labels
        dl.number_format = fmt
        dl.number_format_is_linked = False
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        dl.font.size = Pt(12)
        dl.font.color.rgb = TEXT
    return gf


def table(slide, x, y, w, col_w, rows, header=True, size=13, row_h=0.42, bold_first_col=True, name="table"):
    nr, nc = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    gf.name = name
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for a in ("bandRow", "firstRow"):
        tblPr.set(a, "0")
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.1)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            is_h = header and i == 0
            c.fill.fore_color.rgb = INK if is_h else (CARD if i % 2 else WHITE)
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            _run(p, str(val), size, is_h or (bold_first_col and j == 0), WHITE if is_h else TEXT)
    return gf


# ------------------------------------------------------------------ data
def load() -> dict:
    a = json.load(open(IMG_DIR / "confidence_vs_survey.json"))
    b = {d: json.load(open(IMG_DIR / f"confidence_zones_{d}.json"))[d] for d in ("Dy24", "Dy28", "Dy30")}
    return {"a": a, "b": b}


def zfmt(z: dict) -> str:
    return "–" if z["accuracy"] is None else f"{z['accuracy']:.2f} (n={z['n']})"


# ------------------------------------------------------------------ deck
def build() -> None:
    d = load()
    A, B = d["a"], d["b"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    n = 0

    def ns(*a, **k):
        nonlocal n
        n += 1
        return new_slide(prs, *a, n=n, **k)

    # 1 Title
    n += 1
    s = new_slide(prs, None, n, dark=True)
    txt(s, 0.9, 2.0, 11.5, 1.8, "Does model confidence agree with human raters?", 46, True, WHITE, HEAD,
        anchor=MSO_ANCHOR.BOTTOM, name="title")
    txt(s, 0.9, 4.05, 11.0, 1.0,
        "Organoid quality classification: confidence scores vs 5-rater survey votes, and what that means for defining model accuracy",
        20, False, RGBColor(0xCA, 0xDC, 0xDF), name="subtitle")
    txt(s, 0.9, 6.2, 11, 0.4, "Promega brain-organoid project  |  branch lding/combined_model  |  October 2026", 14, False,
        RGBColor(0x9F, 0xBC, 0xC1), name="byline")

    # 2 Executive summary
    f30 = A["Dy30"]
    s = ns("Summary: confidence tracks rater agreement", notes=(
        "Headline: errors concentrate where humans disagree, so accuracy vs the hard consensus label understates the "
        "model on ambiguous organoids. Numbers: Exp A uses the 139-organoid series cohort (5-0 and 4-1 only). "
        "Exp B uses 248 IDOR organoids (includes 3-2) and 96 unseen BA4 organoids. The 0.83 vs 0.71-0.74 comparison "
        "is across different cohorts/fold structures, so it is suggestive, not controlled."))
    st = f30["fusion"]["strata"]
    stat(s, 0.6, 1.6, 2.95, 2.2, f"{st['5-0']['accuracy']*100:.0f}% vs {st['4-1']['accuracy']*100:.0f}%",
         "fusion accuracy on unanimous (5-0) vs 4-1 organoids, Dy30", big_size=30)
    stat(s, 3.75, 1.6, 2.95, 2.2, f"ρ = {f30['fusion']['spearman_p_vs_vote_share']:.2f}",
         "Spearman, fusion P(Not Acceptable) vs rater vote share, Dy30 (0.64 at Dy28)", big_size=30)
    l30 = B["Dy30"]
    stat(s, 6.9, 1.6, 2.95, 2.2,
         f"{l30['lgbm']['idor_oof']['stats']['balanced_accuracy']:.2f}",
         "LGBM balanced accuracy on all 248 IDOR organoids incl. 3-2 splits (out-of-fold)", big_size=30)
    stat(s, 10.05, 1.6, 2.7, 2.2, f"ρ = {l30['lgbm']['unseen']['stats']['survey_acceptable_vs_p_acceptable']['spearman']:.2f}",
         "survey acceptable share vs model P(Acceptable) on 96 unseen BA4 organoids", big_size=30)
    bullets(s, 0.6, 4.2, 12.1, 2.6, [
        ("Errors cluster where humans disagree. ", "Near-unanimous organoids are almost always right; 4-1 splits carry most of the misses."),
        ("Confidence is informative, but model-dependent. ", "LGBM high-confidence zones are 0.88–0.91 accurate vs 0.54–0.71 in low zones; LogReg probabilities are too compressed to gate on."),
        ("Accuracy vs one hard label is a moving target. ", "It depends on how many ambiguous organoids are in the cohort, so report it by rater-agreement stratum."),
    ], size=17, gap=10, name="summary-bullets")

    # 3 Question
    s = ns("Five raters, one label: what is 'accuracy'?", source="Survey: Dy30, 5 regular-image votes per organoid; consensus = ≥4 of 5 agree")
    bullets(s, 0.6, 1.7, 6.3, 4.8, [
        "Ground truth is a Dy30 survey, not an assay. Five raters vote Acceptable or Not Acceptable per organoid.",
        "The consensus label needs ≥4 of 5 votes. 3-2 splits get no label in the paper cohort.",
        "A model scored against that single label is penalized equally for missing a 5-0 organoid and a coin-flip 3-2.",
        ("Question: ", "does the model's confidence score move with human agreement, and does that give a better definition of accuracy?"),
    ], size=18, gap=14, name="question-bullets")
    for i, (lab, sub, col) in enumerate([("5-0", "unanimous, most certain", INK),
                                          ("4-1", "one dissenter", MUTED),
                                          ("3-2", "near-even, most ambiguous", CORAL)]):
        yy = 1.7 + i * 1.6
        box(s, 7.5, yy, 5.2, 1.4, CARD, name="stratum-card")
        txt(s, 7.8, yy + 0.2, 1.6, 1.0, lab, 40, True, col, HEAD, anchor=MSO_ANCHOR.MIDDLE, name="stratum-name")
        txt(s, 9.4, yy + 0.2, 3.1, 1.0, sub, 16, False, TEXT, anchor=MSO_ANCHOR.MIDDLE, name="stratum-desc")

    # 4 Data pipeline
    s = ns("Data: three modalities across 11 days", source="Source: data/all_data.json, 5,168 organoid-day records (Dy03–Dy30)")
    img_fit(s, FIGURE_DIR / "data_overview.png", 0.5, 1.5, 8.2, 5.4, name="data-overview")
    stat(s, 9.0, 1.6, 3.7, 1.55, "5,168", "records: ~470 organoids × 11 days", big_size=34)
    stat(s, 9.0, 3.3, 3.7, 1.55, "3", "modalities: images, 6 metabolite assays, mask morphology", big_size=34)
    stat(s, 9.0, 5.0, 3.7, 1.55, "Dy03–Dy30", "label from the Dy30 survey; features from every day", big_size=30)

    # 5 Cohorts
    s = ns("Four cohorts, defined by how much survey agreement they need",
           source="Counts from cohorts.py expectations and OOF true_labels; positive class = Not Acceptable (rule 9)",
           notes="139 = complete 11-day series without stitched organoids (Exp A). 248 = IDOR col2 (all classified BA1+BA2), simple-majority labels (Exp B train). 198 = strong consensus. 96 = BA4 labelled, never in IDOR.")
    table(s, 0.6, 1.7, 12.1, [3.4, 0.9, 1.4, 1.5, 2.2, 2.7], [
        ["Cohort", "n", "Acceptable", "Not Acc.", "Vote strata (5-0/4-1/3-2)", "Used for"],
        ["Series cohort (BA1+BA2, 11-day complete)", "139", "114", "25", "88 / 51 / 0", "Exp A: four-modality OOF"],
        ["Strong consensus (≥4/5)", "198", "165", "33", "119 / 79 / 0", "Paper label definition"],
        ["Full IDOR col2 (all classified)", "248", "191", "57", "119 / 79 / 50", "Exp B: train + OOF"],
        ["Unseen BA4 (labelled)", "96", "39", "57", "5 votes each", "Exp B: held-out test"],
    ], size=14, row_h=0.62, name="cohort-table")
    bullets(s, 0.6, 5.1, 12.1, 1.7, [
        "Full cohort resolves 3-2 / 2-3 splits by simple majority; no organoid is dropped for disagreement.",
        "BA4 is outside the IDOR list: a genuine unseen batch, but with a very different class balance (59% Not Acceptable vs 23%).",
    ], size=16, gap=8, name="cohort-bullets")

    # 6 Strata chart
    s = ns("Rater agreement: one in five IDOR organoids is a 3-2 split",
           source="Vote strata from Dy30 regular-image votes (vote_strength.py)")
    bar_chart(s, 0.6, 1.6, 7.3, 5.2, ["5-0", "4-1", "3-2"],
              [("Series cohort (n=139)", [88, 51, 0]), ("Full IDOR (n=248)", [119, 79, 50])],
              [MUTED, CORAL], fmt="0", vmin=0, vmax=140, vfmt="0", vtitle="organoids")
    bullets(s, 8.3, 1.9, 4.4, 4.5, [
        "The 139-organoid cohort used in the earlier multimodal work contains no 3-2 organoids.",
        "Any accuracy measured there is on the easy 85% of the label space.",
        "The 248 cohort adds the 50 organoids humans themselves split on.",
    ], size=17, gap=14, name="strata-bullets")

    # 7 Design
    s = ns("Two experiments", notes="Exp A reuses OOF probabilities saved by combined_kfold.py. Exp B trains fresh metabolite LGBM/LogReg on 248 and scores BA4. IDOR scores in B are honest out-of-fold; BA4 scores come from a refit on all 248.")
    for i, (ttl, items) in enumerate([
        ("A. Confidence vs votes, four modalities", [
            "Cohort: 139 series organoids, all 11 days",
            "Models: metabolite LGBM, morphology LGBM, image EfficientNet-B0, plus mean-probability fusion",
            "Protocol: 10 × 4-fold CV, out-of-fold P(Not Acceptable) averaged over repeats",
            "Compare to Dy30 vote share: Spearman, accuracy by 5-0 / 4-1 stratum",
        ]),
        ("B. Confidence zones on a wider cohort", [
            "Cohort: 248 IDOR (incl. 3-2) + 96 unseen BA4",
            "Models: metabolite-only LGBM and LogReg; Dy24, Dy28, Dy30",
            "Protocol: IDOR 5 × 4-fold OOF; BA4 from a refit on all 248",
            "Outputs: high vs low confidence accuracy, score boxplots, survey-vs-probability correlation",
        ]),
    ]):
        x0 = 0.6 + i * 6.2
        box(s, x0, 1.6, 5.9, 5.1, CARD, name="design-card")
        txt(s, x0 + 0.3, 1.85, 5.3, 0.9, ttl, 22, True, INK, HEAD, name="design-title")
        bullets(s, x0 + 0.3, 2.9, 5.3, 3.7, items, size=16, gap=10, name="design-bullets")

    # 8 Exp A: BA by day
    s = ns("Exp A: signal builds after Dy20; fusion reaches 0.86",
           source="Balanced accuracy vs consensus label, 139-organoid cohort, out-of-fold, mean over repeats at 0.5 threshold")
    line_chart(s, 0.6, 1.5, 8.4, 5.4, [DAY_LBL[x] for x in DAYS_A],
               [(MOD_LABEL[m], [A[x][m]["balanced_accuracy"] for x in DAYS_A]) for m in MOD_COLOR],
               list(MOD_COLOR.values()), vmin=0.4, vmax=1.0, vfmt="0.0", vtitle="balanced accuracy")
    bullets(s, 9.3, 1.9, 3.5, 4.8, [
        "Early days sit near chance (0.50–0.63) for every modality.",
        "From Dy20.5 all four reach 0.74–0.78.",
        f"At Dy30: fusion {f30['fusion']['balanced_accuracy']:.2f}, morphology {f30['morph']['balanced_accuracy']:.2f}, metabolite {f30['met_nan']['balanced_accuracy']:.2f}, image {f30['img']['balanced_accuracy']:.2f}.",
    ], size=16, gap=12, name="ba-bullets")

    # 9 Exp A: rho by day
    s = ns("Exp A: confidence follows vote share more as organoids mature",
           source="Spearman: out-of-fold P(Not Acceptable) vs Dy30 Not-Acceptable vote share (0–5 of 5), n=139")
    line_chart(s, 0.6, 1.5, 8.4, 5.4, [DAY_LBL[x] for x in DAYS_A],
               [(MOD_LABEL[m], [A[x][m]["spearman_p_vs_vote_share"] for x in DAYS_A]) for m in MOD_COLOR],
               list(MOD_COLOR.values()), vmin=0.0, vmax=0.8, vfmt="0.0", vtitle="Spearman ρ")
    bullets(s, 9.3, 1.9, 3.5, 4.8, [
        "Correlation is near zero before Dy10 and rises steadily.",
        f"Fusion peaks at Dy28 (ρ = {A['Dy28']['fusion']['spearman_p_vs_vote_share']:.2f}), above any single modality.",
        "The score carries graded information, not just a binary call.",
    ], size=16, gap=12, name="rho-bullets")

    # 10 Exp A: accuracy by stratum
    s = ns("Exp A: nearly all errors sit in the 4-1 stratum",
           source="Dy30 accuracy vs consensus label by rater-agreement stratum (5-0: n=88, 4-1: n=51); no 3-2 in this cohort")
    mods = list(MOD_COLOR)
    bar_chart(s, 0.6, 1.5, 7.8, 5.4, [MOD_LABEL[m] for m in mods],
              [("5-0 unanimous", [f30[m]["strata"]["5-0"]["accuracy"] for m in mods]),
               ("4-1 one dissenter", [f30[m]["strata"]["4-1"]["accuracy"] for m in mods])],
              [INK, CORAL], vmin=0.5, vmax=1.05, vfmt="0.0", vtitle="accuracy")
    bullets(s, 8.8, 1.9, 3.9, 4.8, [
        "5-0 organoids: 97–99% correct for every modality.",
        "4-1 organoids: 84–88% at Dy30, with fusion and morphology at the top (0.88).",
        "The 4-1 stratum holds 16 of the 25 Not Acceptable organoids, so recall of the minority class lives there.",
    ], size=16, gap=12, name="strata-acc-bullets")

    # 11 Exp A figure
    s = ns("Exp A: Dy30 fusion, confidence by human vote count",
           source="figures/confidence_vs_survey.png")
    img_fit(s, FIGURE_DIR / "confidence_vs_survey.png", 0.6, 1.5, 12.1, 5.4, name="fig-conf-vs-survey")

    # 12 Exp B cohort
    s = ns("Exp B: IDOR is mostly Acceptable, unseen BA4 is mostly not",
           source="Dy30 labels (simple majority of 5 regular votes)",
           notes="The prevalence shift matters when reading BA4 results. Spearman and AUC are less prevalence-sensitive than accuracy; balanced accuracy corrects for it partly.")
    bar_chart(s, 0.6, 1.6, 6.2, 5.2, ["IDOR (n=248)", "Unseen BA4 (n=96)"],
              [("Acceptable", [191 / 248, 39 / 96]), ("Not Acceptable", [57 / 248, 57 / 96])],
              [MUTED, CORAL], fmt="0%", vmin=0, vmax=1.0, vfmt="0%", vtitle="share of cohort")
    bullets(s, 7.2, 1.9, 5.5, 4.8, [
        "Training/OOF on IDOR: 23% Not Acceptable.",
        "Held-out BA4: 59% Not Acceptable, a different batch with a different quality mix.",
        "Features are identical (metabolite concentration, initial concentration, growth) and checked column-by-column between cohorts.",
        "Read BA4 accuracy with the prevalence shift in mind; rank-based metrics (AUC, Spearman) travel better.",
    ], size=16, gap=12, name="prevalence-bullets")

    # 13 Zones chart
    s = ns("Exp B: LGBM's high-confidence zone is clearly more accurate",
           source="Dy30, LGBM, IDOR out-of-fold (n=248). Confidence = max(p, 1−p); zone cutoff on the x-axis",
           notes="High zone = confidence >= cutoff, low zone = below. At cutoff 0.7 the low zone is only 37 organoids but 54% accurate.")
    z = B["Dy30"]["lgbm"]["idor_oof"]["zones"]
    ts = ["0.7", "0.8", "0.9"]
    bar_chart(s, 0.6, 1.5, 7.4, 5.4, [f"≥ {t}" for t in ts],
              [("High-confidence zone", [z[t]["high"]["accuracy"] for t in ts]),
               ("Low-confidence zone", [z[t]["low"]["accuracy"] for t in ts])],
              [INK, CORAL], vmin=0.4, vmax=1.0, vfmt="0.0", vtitle="accuracy in zone")
    bullets(s, 8.4, 1.9, 4.3, 4.8, [
        f"At 0.9: {z['0.9']['high']['n']} organoids at {z['0.9']['high']['accuracy']:.2f} vs {z['0.9']['low']['n']} at {z['0.9']['low']['accuracy']:.2f}.",
        f"At 0.7: {z['0.7']['high']['n']} at {z['0.7']['high']['accuracy']:.2f} vs {z['0.7']['low']['n']} at {z['0.7']['low']['accuracy']:.2f}.",
        "Raising the cutoff trades coverage for accuracy; the low zone is where a human re-check pays off.",
    ], size=16, gap=12, name="zone-bullets")

    # 14 Zones table
    s = ns("Exp B: zone accuracy for both models, both cohorts",
           source="accuracy (n organoids) per zone, Dy30. LogReg probabilities are compressed: almost nothing passes 0.9")
    rows = [["Model / cohort", "Cutoff", "High zone", "Low zone"]]
    for m, mn in (("lgbm", "LGBM"), ("logreg", "LogReg")):
        for c, cn in (("idor_oof", "IDOR OOF"), ("unseen", "BA4 unseen")):
            for t in ("0.7", "0.9"):
                zz = B["Dy30"][m][c]["zones"][t]
                rows.append([f"{mn} / {cn}", f"≥ {t}", zfmt(zz["high"]), zfmt(zz["low"])])
    table(s, 0.6, 1.6, 12.1, [4.0, 1.6, 3.25, 3.25], rows, size=14, row_h=0.5, name="zone-table")

    # 15 Boxplot
    s = ns("Exp B: score distributions, IDOR vs unseen",
           source="Dy30. Left: IDOR positive, IDOR negative, unseen. Right: unseen split by its own survey label")
    img_fit(s, FIGURE_DIR / "confidence_boxplot_Dy30.png", 0.6, 1.45, 7.0, 5.5, name="fig-boxplot")
    lg, lr = B["Dy30"]["lgbm"], B["Dy30"]["logreg"]
    bullets(s, 7.9, 1.8, 4.8, 5.0, [
        f"IDOR separates only moderately (AUC {lg['idor_oof']['stats']['auc']:.2f} LGBM, {lr['idor_oof']['stats']['auc']:.2f} LogReg); the IDOR-positive box is wide.",
        "Unseen BA4 scores sit high overall because most BA4 organoids are Not Acceptable.",
        f"Split by survey label, BA4 separates well (AUC {lg['unseen']['stats']['auc']:.2f} / {lr['unseen']['stats']['auc']:.2f}).",
        "LogReg scores span a narrow band (roughly 0.15–0.95 IDOR, 0.3–0.8 BA4).",
    ], size=15, gap=10, name="boxplot-bullets")

    # 16 Correlation
    s = ns("Exp B: survey acceptable share vs model P(Acceptable)",
           source="Dy30. x: n_Acceptable / 5 from regular votes (jittered). Spearman in each panel title")
    img_fit(s, FIGURE_DIR / "confidence_survey_corr_Dy30.png", 0.6, 1.45, 7.0, 5.5, name="fig-corr")
    bullets(s, 7.9, 1.8, 4.8, 5.0, [
        f"IDOR out-of-fold: Spearman {lg['idor_oof']['stats']['survey_acceptable_vs_p_acceptable']['spearman']:.2f} (LGBM), {lr['idor_oof']['stats']['survey_acceptable_vs_p_acceptable']['spearman']:.2f} (LogReg).",
        f"Unseen BA4: {lg['unseen']['stats']['survey_acceptable_vs_p_acceptable']['spearman']:.2f} for both models.",
        "Higher human acceptable-ness means higher model P(Acceptable), with wide scatter inside each vote level.",
        "Organoids humans split on (0.4–0.6) get middling model scores, but so do many clear-cut ones.",
    ], size=15, gap=10, name="corr-bullets")

    # 17 Robustness
    s = ns("Exp B across days: Dy30 and Dy28 hold, Dy24 LogReg does not",
           source="BA = balanced accuracy at 0.5; AUC vs label; ρ = Spearman(survey acceptable share, P(Acceptable))")
    rows = [["Day", "Model", "IDOR BA", "IDOR AUC", "IDOR ρ", "BA4 BA", "BA4 AUC", "BA4 ρ"]]
    for day in ("Dy24", "Dy28", "Dy30"):
        for m, mn in (("lgbm", "LGBM"), ("logreg", "LogReg")):
            i, u = B[day][m]["idor_oof"]["stats"], B[day][m]["unseen"]["stats"]
            rows.append([day, mn, f"{i['balanced_accuracy']:.2f}", f"{i['auc']:.2f}",
                         f"{i['survey_acceptable_vs_p_acceptable']['spearman']:.2f}",
                         f"{u['balanced_accuracy']:.2f}", f"{u['auc']:.2f}",
                         f"{u['survey_acceptable_vs_p_acceptable']['spearman']:.2f}"])
    table(s, 0.6, 1.6, 12.1, [1.5, 1.8, 1.4, 1.5, 1.3, 1.5, 1.6, 1.5], rows, size=14, row_h=0.52, name="robust-table")
    txt(s, 0.6, 5.6, 12.1, 1.2,
        "IDOR balanced accuracy is 0.71–0.74 on every day and model, about ten points below the 0.80–0.87 seen on the 139-organoid cohort at Dy28–Dy30. "
        "The cohorts and fold structures differ, so this is suggestive rather than a controlled comparison.",
        15, False, TEXT, name="robust-note")

    # 18 Interpretation
    s = ns("Defining accuracy with rater disagreement in mind", notes="These are proposals for the paper's evaluation section, not yet implemented beyond the analyses shown.")
    items = [
        ("Stratify", "Report accuracy by rater-agreement stratum (5-0 / 4-1 / 3-2) next to the headline number."),
        ("Soft-label agreement", "Report rank correlation or Brier vs the vote share, not only agreement with the hard label."),
        ("Confidence-gated", "Report accuracy at a stated coverage: e.g. 0.91 on the 59% of organoids LGBM is ≥0.9 confident on."),
        ("Rater ceiling", "State the achievable ceiling from rater agreement; 100% against a 4-1 label is not the target."),
    ]
    for i, (h, t) in enumerate(items):
        x0 = 0.6 + (i % 2) * 6.2
        y0 = 1.7 + (i // 2) * 2.55
        box(s, x0, y0, 5.9, 2.3, CARD, name="proposal-card")
        txt(s, x0 + 0.3, y0 + 0.25, 5.3, 0.5, h, 22, True, INK, HEAD, name="proposal-title")
        txt(s, x0 + 0.3, y0 + 0.9, 5.3, 1.3, t, 16, False, TEXT, name="proposal-text")

    # 19 Caveats and next
    s = ns("Caveats and next steps")
    txt(s, 0.6, 1.6, 5.8, 0.5, "Caveats", 22, True, INK, HEAD, name="caveats-title")
    bullets(s, 0.6, 2.2, 5.8, 4.6, [
        "Exp B is metabolite-only; morphology and image models are not yet on the 248 cohort or BA4.",
        "BA4 is one batch with a different class balance; Dy24 LogReg on BA4 is weak (BA 0.62).",
        "Exp A and B use different cohorts and CV structures, so cross-comparison is suggestive.",
        "Five raters is a coarse estimate of agreement; vote share is not a calibrated probability.",
    ], size=15, gap=10, name="caveats-bullets")
    txt(s, 6.9, 1.6, 5.8, 0.5, "Next steps", 22, True, INK, HEAD, name="next-title")
    bullets(s, 6.9, 2.2, 5.8, 4.6, [
        "Run the four-modality OOF pipeline on the 248 cohort so 3-2 organoids appear in Exp A.",
        "Calibrate probabilities (isotonic / Platt) and re-draw confidence zones; LogReg is under-confident.",
        "Score BA4 with fusion models and add BA4 vote strata.",
        "Add a stratified-accuracy table to the paper's evaluation section.",
    ], size=15, gap=10, name="next-bullets")

    # 20 Appendix reproducibility
    s = ns("Appendix: reproducibility", source="All scripts read all_data.json via pipeline.data_loader; outputs under $ANALYSIS_OUTPUT_DIR")
    table(s, 0.6, 1.6, 12.1, [4.6, 7.5], [
        ["Item", "Location"],
        ["Exp A script", "analysis/paper_2026_04/confidence_vs_survey.py"],
        ["Exp B script", "analysis/paper_2026_04/confidence_zones.py (--days Dy24 Dy28 Dy30)"],
        ["Slurm", "submit_confidence_vs_survey.slurm, submit_confidence_zones.slurm (array, one day per task)"],
        ["Results JSON", "analysis_output/images/confidence_vs_survey.json, confidence_zones_<day>.json"],
        ["Per-organoid CSV", "analysis_output/images/confidence_vs_survey.csv, confidence_zones_organoids_<day>.csv"],
        ["Figures", "analysis_output/figures/confidence_*.png"],
        ["Protocols", "Exp A: 10×4-fold, SEED=1. Exp B: 5×4-fold, SEED=1, LGBM/LogReg grid search as in met_lgbm_logreg_kfold.py"],
    ], size=13, row_h=0.62, name="repro-table")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print("saved", OUT, "slides:", len(prs.slides))


if __name__ == "__main__":
    build()
