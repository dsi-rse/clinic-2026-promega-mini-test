#!/usr/bin/env python3
"""Which organoids does fusion correct (or break) relative to its single-source models?

Per day and per fusion set, each organoid's OOF probability is averaged over the
repeats for every source (met / morph / img) and for the fused mean-probability.
Labels: 1 = Acceptable, 0 = Not Acceptable (as stored in true_labels).

Categories (fused prediction vs. the single-source predictions):
  corrected   — fusion right, >=1 source wrong
    rescued   —   fusion right, a majority of sources wrong
  broken      — fusion wrong, >=1 source right
    broken_all—   fusion wrong, every source right
  all_right / all_wrong — no change

Outputs (analysis_output/images/):
  fusion_corrections_per_organoid.csv   one row per (day, fusion, organoid)
  fusion_corrections_summary.csv        counts per (day, fusion)
Also prints a summary and the organoids most consistently corrected/broken.

Usage:  python3 -m analysis.paper_2026_04.fusion_corrections
"""

import json

import numpy as np
import pandas as pd

from pipeline.data_loader import ANALYSIS_OUTPUT_DIR, DAY_ORDER
from analysis.paper_2026_04.fusion_lgbm_vs_logreg import (
    COMBINED_PATH, FUSION_SETS, MET_MORPH_PATH,
)

OUT_DIR = ANALYSIS_OUTPUT_DIR / "images"


def _mean_probs(mm_day, comb_day):
    """Return org_ids, labels, {source: mean OOF prob over repeats}, {source: [per-repeat arrays]}."""
    per_src = {}
    labels, ids = {}, None
    for mm_rep, comb_rep in zip(mm_day["repeat_details"], comb_day["repeat_details"]):
        cidx = {o: i for i, o in enumerate(comb_rep["org_ids"])}
        for k, v in mm_rep["oof_probs"].items():
            per_src.setdefault(k, {})
            for o, p in zip(mm_rep["org_ids"], v):
                per_src[k].setdefault(o, []).append(np.nan if p is None else p)
        per_src.setdefault("img", {})
        for o in mm_rep["org_ids"]:
            ci = cidx.get(o)
            p = comb_rep["oof_probs"]["img"][ci] if ci is not None else None
            per_src["img"].setdefault(o, []).append(np.nan if p is None else p)
        labels.update(dict(zip(mm_rep["org_ids"], mm_rep["true_labels"])))
    return labels, per_src


def categorize(fused_ok, src_ok):
    n_wrong = int((~src_ok).sum())
    if fused_ok:
        if n_wrong == 0:
            return "all_right"
        return "rescued" if n_wrong > len(src_ok) / 2 else "corrected"
    if n_wrong == len(src_ok):
        return "all_wrong"
    return "broken_all" if n_wrong == 0 else "broken"


def main():
    mm = json.loads(MET_MORPH_PATH.read_text())
    comb = json.loads(COMBINED_PATH.read_text())
    rows = []
    for day in DAY_ORDER:
        if day not in mm or day not in comb or "repeat_details" not in mm[day]:
            continue
        labels, per_src = _mean_probs(mm[day], comb[day])
        for fk, sources in FUSION_SETS.items():
            for o, y in labels.items():
                # per-repeat fused prob, then average
                mat = np.array([per_src[s][o] for s in sources])  # (src, repeats)
                if np.isnan(mat).all():
                    continue
                src_p = np.nanmean(mat, axis=1)
                if np.isnan(src_p).any():
                    continue
                fused_rep = np.nanmean(mat, axis=0)
                fused_p = float(np.nanmean(fused_rep))
                src_ok = (src_p >= 0.5).astype(int) == y
                fused_ok = (fused_p >= 0.5) == bool(y)
                row = dict(day=day, fusion=fk, organoid=o, label=y, fused_prob=fused_p,
                           fused_ok=bool(fused_ok), frac_repeats_fused_ok=float(
                               np.mean((fused_rep >= 0.5) == bool(y))),
                           category=categorize(fused_ok, src_ok))
                for s, p, ok in zip(sources, src_p, src_ok):
                    row[f"p_{s}"] = float(p)
                    row[f"ok_{s}"] = bool(ok)
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "fusion_corrections_per_organoid.csv", index=False)

    cats = ["all_right", "corrected", "rescued", "broken", "broken_all", "all_wrong"]
    summ = (df.groupby(["fusion", "day", "category"]).size().unstack(fill_value=0)
              .reindex(columns=cats, fill_value=0))
    summ["net_gain"] = summ["corrected"] + summ["rescued"] - summ["broken"] - summ["broken_all"]
    summ = summ.reset_index()
    summ["day"] = pd.Categorical(summ["day"], DAY_ORDER, ordered=True)
    summ = summ.sort_values(["fusion", "day"])
    summ.to_csv(OUT_DIR / "fusion_corrections_summary.csv", index=False)

    pd.set_option("display.width", 200)
    for fk in FUSION_SETS:
        print(f"\n=== {fk} ===")
        print(summ[summ.fusion == fk].drop(columns="fusion").to_string(index=False))

    # Which organoids are corrected most often across days (all_lgbm as headline)
    print("\n=== label breakdown of corrected/rescued vs broken (all fusions, all days) ===")
    df["changed"] = df.category.map(lambda c: "corrected" if c in ("corrected", "rescued")
                                    else "broken" if c.startswith("broken") else "none")
    print(pd.crosstab([df.fusion, df.changed], df.label.map({1: "Acceptable", 0: "NotAcceptable"})))

    for fk in ("all_lgbm", "mixed2"):
        d = df[df.fusion == fk]
        g = d.groupby("organoid").agg(
            n_days=("day", "size"),
            corrected=("changed", lambda s: (s == "corrected").sum()),
            broken=("changed", lambda s: (s == "broken").sum()),
            label=("label", "first"))
        print(f"\n--- {fk}: top organoids by # days corrected ---")
        print(g.sort_values(["corrected", "broken"], ascending=[False, True]).head(15).to_string())
        print(f"--- {fk}: top organoids by # days broken ---")
        print(g.sort_values(["broken", "corrected"], ascending=[False, True]).head(10).to_string())


if __name__ == "__main__":
    main()
