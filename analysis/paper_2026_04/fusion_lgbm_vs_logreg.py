#!/usr/bin/env python3
"""Compute LGBM vs LogReg fusion from met+morph OOF probs + img OOF probs.

Two fusion sets (mean probability of three modalities):
  all_lgbm   — met_nan_lgbm  + morph_lgbm  + img
  all_logreg — met_nan_logreg + morph_logreg + img

Inputs:
  met_morph_lgbm_logreg_kfold.json  (from met_lgbm_logreg_kfold.py, with repeat_details)
  combined_results_kfold_series_idor_139.json  (img OOF probs in repeat_details)

Output:
  analysis_output/images/fusion_lgbm_vs_logreg.json

Usage:
    python3 -m analysis.paper_2026_04.fusion_lgbm_vs_logreg
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from pipeline.data_loader import ANALYSIS_OUTPUT_DIR, DAY_ORDER

MET_MORPH_PATH = ANALYSIS_OUTPUT_DIR / "images" / "met_morph_lgbm_logreg_kfold.json"
COMBINED_PATH  = ANALYSIS_OUTPUT_DIR / "images" / "combined_results_kfold_series_idor_139.json"
OUTPUT_PATH    = ANALYSIS_OUTPUT_DIR / "images" / "fusion_lgbm_vs_logreg.json"

FUSION_SETS = {
    "all_lgbm":   ["met_nan_lgbm",   "morph_lgbm",   "img"],
    "all_logreg": ["met_nan_logreg", "morph_logreg",  "img"],
}


def _org_id_index(org_ids):
    return {oid: i for i, oid in enumerate(org_ids)}


def compute_fusion(day: str, mm_day: dict, combined_day: dict) -> dict:
    """Compute fusion BAs for one day using OOF probs from both sources."""
    mm_details  = mm_day.get("repeat_details", [])
    comb_details = combined_day.get("repeat_details", [])

    if not mm_details or not comb_details:
        return {}

    n_repeats = min(len(mm_details), len(comb_details))
    results = {fk: {"repeat_balanced_accuracies": [], "repeat_confusion_matrices": []}
               for fk in FUSION_SETS}

    for rep in range(n_repeats):
        mm_rep   = mm_details[rep]
        comb_rep = comb_details[rep]

        # Both use same org_ids / seeds (SEED=1 structure)
        org_ids    = mm_rep["org_ids"]
        true_labels = np.array(mm_rep["true_labels"])
        n = len(org_ids)

        # Build prob arrays — None → NaN
        def _arr(probs_list):
            return np.array([np.nan if v is None else v for v in probs_list])

        mm_probs = {k: _arr(mm_rep["oof_probs"][k])
                    for k in mm_rep["oof_probs"]}

        # img probs come from combined's repeat_details; align by org_ids
        comb_org_ids = comb_rep["org_ids"]
        comb_idx = _org_id_index(comb_org_ids)
        img_probs = np.full(n, np.nan)
        for i, oid in enumerate(org_ids):
            ci = comb_idx.get(oid)
            if ci is not None:
                img_probs[i] = comb_rep["oof_probs"]["img"][ci]

        all_source_probs = {**mm_probs, "img": img_probs}

        for fk, sources in FUSION_SETS.items():
            stacked = np.stack([all_source_probs[s] for s in sources
                                if s in all_source_probs], axis=0)
            mean_prob = np.nanmean(stacked, axis=0)
            valid = ~np.isnan(mean_prob)
            if valid.sum() < 2:
                continue
            yt = true_labels[valid]
            yp = (mean_prob[valid] >= 0.5).astype(int)
            if len(np.unique(yt)) < 2:
                continue
            results[fk]["repeat_balanced_accuracies"].append(
                float(balanced_accuracy_score(yt, yp))
            )
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            results[fk]["repeat_confusion_matrices"].append(cm.tolist())

    out = {}
    for fk in FUSION_SETS:
        bas = results[fk]["repeat_balanced_accuracies"]
        if bas:
            out[fk] = {
                "balanced_accuracy_mean":     float(np.mean(bas)),
                "balanced_accuracy_std":      float(np.std(bas)),
                "n_repeats":                  len(bas),
                "repeat_balanced_accuracies": bas,
                "repeat_confusion_matrices":  results[fk]["repeat_confusion_matrices"],
            }
    return out


def main():
    mm_data   = json.loads(MET_MORPH_PATH.read_text())
    comb_data = json.loads(COMBINED_PATH.read_text())

    # Also pull reference single-modality BAs from combined for comparison
    ref_keys = ["met_nan", "morph", "img", "met_nan+morph+img_mean_prob"]

    all_results = {}
    for day in DAY_ORDER:
        mm_day   = mm_data.get(day)
        comb_day = comb_data.get(day)
        if mm_day is None or comb_day is None:
            continue
        if "repeat_details" not in mm_day:
            print(f"[{day}] missing repeat_details in met_morph JSON — skipping")
            continue

        day_res = compute_fusion(day, mm_day, comb_day)
        if day_res:
            all_results[day] = day_res
            print(f"[{day}] computed")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {OUTPUT_PATH}")

    print("\n\n=== LGBM vs LogReg Fusion (mean BA) ===")
    cols = list(FUSION_SETS.keys()) + ref_keys
    header = f"{'Day':<10}" + "".join(f"{c:>16}" for c in cols)
    print(header)
    for day in DAY_ORDER:
        if day not in all_results and day not in comb_data:
            continue
        row = f"{day:<10}"
        for c in list(FUSION_SETS.keys()):
            v = all_results.get(day, {}).get(c, {})
            m = v.get("balanced_accuracy_mean")
            row += f"{f'{m:.3f}' if m else '—':>16}"
        for c in ref_keys:
            v = comb_data.get(day, {}).get(c, {})
            m = v.get("balanced_accuracy_mean")
            row += f"{f'{m:.3f}' if m else '—':>16}"
        print(row)


if __name__ == "__main__":
    main()
