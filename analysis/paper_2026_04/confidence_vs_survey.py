#!/usr/bin/env python3
"""Model confidence vs human survey agreement (Dy30 regular 5-rater votes).

Question: when the models are "confident", do the human raters agree with each
other (and with the model)? And does the model's error rate concentrate on the
organoids humans themselves disagree on? If so, balanced accuracy against the
hard consensus label understates the model on ambiguous organoids, and the
"ceiling" for accuracy is set by rater agreement, not by 1.0.

Inputs (OOF probabilities, 10x repeated 4-fold CV, 139-organoid cohort):
  analysis_output/images/combined_results_kfold_series_idor_139.json
Human votes: Dy30 ``label.regular_votes`` via ``get_survey_vote_counts``.

Per organoid, per day, per modality (met_nan, morph, img, mean-fusion of the
three) the OOF P(Not Acceptable) is averaged across repeats, then compared with
the human Not-Acceptable vote share (0/5 .. 5/5).

Outputs (``$ANALYSIS_OUTPUT_DIR/images/``):
  confidence_vs_survey.json   summary stats per (day, modality)
  confidence_vs_survey.csv    per-organoid table for Dy30 (for ad-hoc digging)
  figures/confidence_vs_survey.png

Usage (submit via sbatch; see submit_confidence_vs_survey.slurm):
    python3 -m analysis.paper_2026_04.confidence_vs_survey
"""

import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    FIGURE_DIR,
    LABEL_DAY,
    get_survey_vote_counts,
    iter_organoid_records,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COMBINED_PATH = ANALYSIS_OUTPUT_DIR / "images" / "combined_results_kfold_series_idor_139.json"
ALL_DATA_PATH = Path("data/all_data.json")
OUT_JSON = ANALYSIS_OUTPUT_DIR / "images" / "confidence_vs_survey.json"
OUT_CSV = ANALYSIS_OUTPUT_DIR / "images" / "confidence_vs_survey.csv"
OUT_FIG = FIGURE_DIR / "confidence_vs_survey.png"

MODALITIES = ("met_nan", "morph", "img", "fusion")
STRATA = ("5-0", "4-1", "3-2")
EXPECTED_N = 139
EXPECTED_VOTES = 5
HEADLINE_DAYS = ("Dy20_5", "Dy24", "Dy28", "Dy30")


def load_na_votes(org_ids: list[str]) -> dict[str, int]:
    """Not-Acceptable regular-vote count (0..5) per organoid, from Dy30."""
    wanted = set(org_ids)
    out: dict[str, int] = {}
    for oid, by_day, _batch in iter_organoid_records(ALL_DATA_PATH):
        if oid not in wanted:
            continue
        rec = by_day.get(LABEL_DAY)
        assert rec is not None, f"{oid}: no {LABEL_DAY} record"
        n_acc, total = get_survey_vote_counts(rec)
        assert total == EXPECTED_VOTES, f"{oid}: {total} regular votes, expected 5"
        out[oid] = total - n_acc
    missing = wanted - set(out)
    assert not missing, f"{len(missing)} organoids missing from all_data: {sorted(missing)[:3]}"
    return out


def mean_oof(day_block: dict) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Organoid ids, true labels, and repeat-averaged OOF probs per modality."""
    reps = day_block["repeat_details"]
    org_ids = reps[0]["org_ids"]
    labels = np.asarray(reps[0]["true_labels"], dtype=int)
    probs: dict[str, np.ndarray] = {}
    for m in ("met_nan", "morph", "img"):
        stack = np.array(
            [[np.nan if v is None else v for v in r["oof_probs"][m]] for r in reps],
            dtype=float,
        )
        probs[m] = np.nanmean(stack, axis=0)
    # Fusion = mean over the modalities available for that organoid (morph can be missing).
    probs["fusion"] = np.nanmean([probs[m] for m in ("met_nan", "morph", "img")], axis=0)
    for m, p in probs.items():
        assert p.shape == (len(org_ids),), f"{m}: bad shape"
        n_nan = int(np.isnan(p).sum())
        if n_nan:
            logger.warning("%s: %d of %d organoids have no OOF prob (excluded from that modality)", m, n_nan, len(p))
    assert not np.isnan(probs["fusion"]).any(), "fusion NaN: organoid with no modality"
    return org_ids, labels, probs


def stratum_of(n_na: int) -> str:
    return {0: "5-0", 5: "5-0", 1: "4-1", 4: "4-1", 2: "3-2", 3: "3-2"}[n_na]


def summarize(p: np.ndarray, y: np.ndarray, n_na: np.ndarray) -> dict:
    """Confidence-vs-human statistics for one (day, modality)."""
    ok = ~np.isnan(p)
    p, y, n_na = p[ok], y[ok], n_na[ok]
    share = n_na / EXPECTED_VOTES  # human soft label for Not Acceptable
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    strata = np.array([stratum_of(int(v)) for v in n_na])
    rho, rho_p = spearmanr(p, share)
    res: dict = {
        "n": int(len(y)),
        "spearman_p_vs_vote_share": float(rho),
        "spearman_pvalue": float(rho_p),
        "brier_vs_hard_label": float(np.mean((p - y) ** 2)),
        "brier_vs_vote_share": float(np.mean((p - share) ** 2)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)),
        "strata": {},
    }
    for s in STRATA:
        mk = strata == s
        yt, pp = y[mk], pred[mk]
        res["strata"][s] = {
            "n": int(mk.sum()),
            "n_not_acceptable": int(yt.sum()),
            "accuracy": float((pp == yt).mean()) if mk.any() else None,
            "mean_confidence": float(conf[mk].mean()) if mk.any() else None,
            "mean_p_na_on_na": float(p[mk & (y == 1)].mean()) if (mk & (y == 1)).any() else None,
            "mean_p_na_on_acc": float(p[mk & (y == 0)].mean()) if (mk & (y == 0)).any() else None,
        }
    return res


def plot(results: dict, day_data: dict) -> None:
    """Left: p(NA) vs human NA vote count at Dy30 (fusion). Right: accuracy by stratum."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    n_na, p = day_data["n_na"], day_data["probs"]["fusion"]
    rng = np.random.default_rng(42)
    ax.scatter(n_na + rng.uniform(-0.15, 0.15, len(n_na)), p, s=18, alpha=0.6, color="#0072B2")
    for k in range(EXPECTED_VOTES + 1):
        mk = n_na == k
        if mk.any():
            ax.hlines(p[mk].mean(), k - 0.3, k + 0.3, color="#D55E00", lw=2.5)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("Human 'Not Acceptable' votes (of 5)")
    ax.set_ylabel("Model P(Not Acceptable), fusion, OOF mean")
    rho = results[LABEL_DAY]["fusion"]["spearman_p_vs_vote_share"]
    ax.set_title(f"{LABEL_DAY}: model confidence vs rater votes (Spearman {rho:.2f})")

    ax = axes[1]
    width, styles = 0.2, {"met_nan": "#E69F00", "morph": "#009E73", "img": "#56B4E9", "fusion": "#000000"}
    for i, m in enumerate(MODALITIES):
        accs = [results[LABEL_DAY][m]["strata"][s]["accuracy"] or 0.0 for s in STRATA]
        ax.bar(np.arange(len(STRATA)) + (i - 1.5) * width, accs, width, label=m, color=styles[m])
    ax.set_xticks(range(len(STRATA)))
    ax.set_xticklabels([f"{s}\n(n={results[LABEL_DAY]['fusion']['strata'][s]['n']})" for s in STRATA])
    ax.set_ylabel("Accuracy vs consensus label")
    ax.set_xlabel("Rater agreement stratum (majority-minority)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{LABEL_DAY}: accuracy by rater agreement")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)


def main() -> None:
    combined = json.load(open(COMBINED_PATH))
    results: dict[str, dict] = {}
    day_data: dict = {}
    for day in sorted(combined):
        org_ids, y, probs = mean_oof(combined[day])
        assert len(org_ids) == EXPECTED_N, f"{day}: {len(org_ids)} organoids, expected {EXPECTED_N}"
        votes = load_na_votes(org_ids)
        n_na = np.array([votes[o] for o in org_ids])
        results[day] = {m: summarize(probs[m], y, n_na) for m in MODALITIES}
        if day == LABEL_DAY:
            day_data = {"org_ids": org_ids, "y": y, "n_na": n_na, "probs": probs}
        logger.info(
            "%s fusion: spearman=%.2f BA=%.3f",
            day,
            results[day]["fusion"]["spearman_p_vs_vote_share"],
            results[day]["fusion"]["balanced_accuracy"],
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["organoid_id", "label_na", "human_na_votes", "stratum", *MODALITIES])
        for i, oid in enumerate(day_data["org_ids"]):
            w.writerow(
                [oid, int(day_data["y"][i]), int(day_data["n_na"][i]), stratum_of(int(day_data["n_na"][i]))]
                + [f"{day_data['probs'][m][i]:.4f}" for m in MODALITIES]
            )
    plot(results, day_data)

    for day in HEADLINE_DAYS:
        for m in MODALITIES:
            r = results[day][m]
            logger.info(
                "%-7s %-8s rho=%.2f  acc by stratum: %s",
                day,
                m,
                r["spearman_p_vs_vote_share"],
                "  ".join(f"{s}={r['strata'][s]['accuracy'] or float('nan'):.2f}(n={r['strata'][s]['n']})" for s in STRATA),
            )
    logger.info("wrote %s, %s, %s", OUT_JSON, OUT_CSV, OUT_FIG)


if __name__ == "__main__":
    main()
