#!/usr/bin/env python3
"""Confidence zones, IDOR-vs-unseen score distributions, and score-vs-survey correlation.

Cohorts (survey consistency NOT required — 3-2 / 2-3 splits are kept and resolved by
simple majority of the 5 regular Dy30 votes, same as ``cohorts.py`` "full"):
  - IDOR  : the 248 BA1+BA2 col2 organoids  (trained on, scored out-of-fold)
  - unseen: BA4 organoids with 5 Dy30 regular votes (never in the training set)

Metabolite features only, LightGBM and LogReg (same trainers as met_lgbm_logreg_kfold).
IDOR scores are OOF probabilities from repeated stratified k-fold (averaged over
repeats); unseen scores come from a model refit on all 248 IDOR organoids.

Three analyses per (day, model); score = P(Not Acceptable), confidence = max(p, 1-p):
  1. Confidence zones: accuracy / balanced accuracy in the high- vs low-confidence zone
     for a sweep of thresholds, on IDOR-OOF and on unseen.
  2. Boxplot of P(NA) for IDOR-positive, IDOR-negative and unseen (plus unseen split
     by its own survey label).
  3. Survey acceptable-ness (n_Acceptable/5) vs model P(Acceptable): Spearman/Pearson.

Outputs: ``$ANALYSIS_OUTPUT_DIR/images/confidence_zones.json``, per-organoid CSV, and
``$FIGURE_DIR/confidence_{zones,boxplot,survey_corr}_<day>.png``.

Usage (submit via sbatch, see submit_confidence_zones.slurm):
    python3 -m analysis.paper_2026_04.confidence_zones --days Dy24 Dy28 Dy30
"""

import argparse
import csv
import json
import logging
import os
import sys
import warnings
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from analysis.paper_2026_04.met_lgbm_logreg_kfold import (
    _train_lgbm_fold,
    _train_logreg_fold,
)
from analysis.paper_2026_04.metabolites_train import _features_for_day_all
from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    FIGURE_DIR,
    LABEL_DAY,
    LABEL_TO_INT,
    OrganoidDataset,
    get_survey_vote_counts,
    require_batches,
)

# cohorts.py lives in a digit-named package (not importable via -m); mirror vote_strength.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2026_06_metabolite_pred"))
from cohorts import build_cohort, simple_majority_label_fn  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ALL_DATA_PATH = "data/all_data.json"
SEED = 1
N_FOLDS = 4
N_REPEATS = 5
EXPECTED_IDOR = 248
EXPECTED_VOTES = 5
THRESHOLDS = (0.6, 0.7, 0.8, 0.9)
MODELS = {"lgbm": _train_lgbm_fold, "logreg": _train_logreg_fold}
OUT_DIR = ANALYSIS_OUTPUT_DIR / "images"


def build_datasets() -> tuple[OrganoidDataset, OrganoidDataset]:
    """(IDOR 248 'full' cohort, BA4 unseen cohort), both simple-majority labelled."""
    idor, counts = build_cohort("full", ALL_DATA_PATH)
    assert len(idor.organoid_ids) == EXPECTED_IDOR
    logger.info("IDOR cohort: %d  %s", len(idor.organoid_ids), dict(counts))
    unseen = OrganoidDataset(
        ALL_DATA_PATH, filters=[require_batches("BA4")], label_fn=simple_majority_label_fn
    )
    n_u = len(unseen.organoid_ids)
    assert n_u > 0
    assert not set(unseen.organoid_ids) & set(idor.organoid_ids), "unseen overlaps IDOR"
    logger.info("Unseen BA4 (labelled): %d  %s", n_u, dict(Counter(unseen.organoid_labels().values())))
    return idor, unseen


def human_acceptable_share(ds: OrganoidDataset, ids: list[str]) -> np.ndarray:
    """n_Acceptable / 5 from the Dy30 regular votes, per organoid id."""
    out = []
    for oid in ids:
        rec = ds.get_record(oid, LABEL_DAY)
        assert rec is not None, f"{oid}: no {LABEL_DAY} record"
        acc, total = get_survey_vote_counts(rec)
        assert total == EXPECTED_VOTES, f"{oid}: {total} votes"
        out.append(acc / EXPECTED_VOTES)
    return np.array(out)


def features(ds: OrganoidDataset, day: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    X, y, names, ids = _features_for_day_all(ds, day, malate_mode="nan")
    n0 = len(ds.organoid_ids)
    if len(ids) != n0:
        logger.warning("[%s] %d of %d organoids dropped at feature step (missing day data)", day, n0 - len(ids), n0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), list(names), list(ids)


def idor_oof_and_unseen(
    model_fn, X: np.ndarray, y: np.ndarray, Xu: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat-averaged OOF P(NA) on IDOR, and refit-on-all P(NA) for unseen."""
    oof = np.zeros((N_REPEATS, len(y)))
    for rep in range(N_REPEATS):
        rep_seed = SEED + rep * 1000
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=rep_seed)
        for fold_i, (tr, te) in enumerate(cv.split(X, y)):
            p = model_fn(X[tr], y[tr], X[te], rep_seed + fold_i * 97)
            logger.info("  rep %d/%d fold %d/%d done", rep + 1, N_REPEATS, fold_i + 1, N_FOLDS)
            assert p is not None
            oof[rep, te] = p
    p_unseen = model_fn(X, y, Xu, SEED)
    assert p_unseen is not None
    return oof.mean(axis=0), np.asarray(p_unseen)


def zone_stats(p: np.ndarray, y: np.ndarray) -> dict:
    """Accuracy / BA in high vs low confidence zones across thresholds."""
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    rows = {}
    for t in THRESHOLDS:
        hi = conf >= t
        row = {}
        for name, mk in (("high", hi), ("low", ~hi)):
            n = int(mk.sum())
            row[name] = {
                "n": n,
                "n_not_acceptable": int(y[mk].sum()),
                "accuracy": float((pred[mk] == y[mk]).mean()) if n else None,
                "balanced_accuracy": float(balanced_accuracy_score(y[mk], pred[mk]))
                if n and len(np.unique(y[mk])) == 2
                else None,
            }
        rows[f"{t:.1f}"] = row
    return rows


def group_stats(p: np.ndarray, y: np.ndarray, share_acc: np.ndarray) -> dict:
    out = {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, (p >= 0.5).astype(int))),
        "mean_p_na_positive": float(p[y == 1].mean()),
        "mean_p_na_negative": float(p[y == 0].mean()),
    }
    rho, rho_p = spearmanr(share_acc, 1 - p)
    r, r_p = pearsonr(share_acc, 1 - p)
    out["survey_acceptable_vs_p_acceptable"] = {
        "spearman": float(rho), "spearman_p": float(rho_p), "pearson": float(r), "pearson_p": float(r_p),
    }
    return out


def plot_zones(day: str, res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = {"lgbm": "#0072B2", "logreg": "#D55E00"}
    for ax, cohort in zip(axes, ("idor_oof", "unseen")):
        for mname in MODELS:
            z = res[mname][cohort]["zones"]
            ts = sorted(z)
            for zone, ls, mk in (("high", "-", "o"), ("low", "--", "s")):
                ax.plot([float(t) for t in ts],
                        [z[t][zone]["accuracy"] if z[t][zone]["accuracy"] is not None else np.nan for t in ts],
                        ls=ls, marker=mk, color=colors[mname], label=f"{mname} {zone}-conf")
        ax.set_title(f"{day}: {cohort}")
        ax.set_xlabel("confidence threshold  (max(p,1-p))")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("accuracy in zone")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"confidence_zones_{day}.png", dpi=200)
    plt.close(fig)


def plot_box(day: str, data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for r, mname in enumerate(MODELS):
        d = data[mname]
        groups = [
            ("IDOR\npositive", d["p_idor"][d["y_idor"] == 1]),
            ("IDOR\nnegative", d["p_idor"][d["y_idor"] == 0]),
            ("Unseen\n(BA4)", d["p_unseen"]),
        ]
        ax = axes[r, 0]
        ax.boxplot([g[1] for g in groups], tick_labels=[f"{g[0]}\n(n={len(g[1])})" for g in groups], showfliers=True)
        ax.axhline(0.5, color="grey", ls=":")
        ax.set_ylabel(f"{mname} P(Not Acceptable)")
        ax.set_title(f"{day} {mname}: IDOR (OOF) vs unseen")
        groups_b = [
            ("Unseen\npositive", d["p_unseen"][d["y_unseen"] == 1]),
            ("Unseen\nnegative", d["p_unseen"][d["y_unseen"] == 0]),
        ]
        ax = axes[r, 1]
        ax.boxplot([g[1] for g in groups_b], tick_labels=[f"{g[0]}\n(n={len(g[1])})" for g in groups_b])
        ax.axhline(0.5, color="grey", ls=":")
        ax.set_title(f"{day} {mname}: unseen split by survey label")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"confidence_boxplot_{day}.png", dpi=200)
    plt.close(fig)


def plot_corr(day: str, data: dict, res: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    rng = np.random.default_rng(42)
    for r, mname in enumerate(MODELS):
        d = data[mname]
        for c, (cohort, p, s) in enumerate((("idor_oof", d["p_idor"], d["s_idor"]),
                                            ("unseen", d["p_unseen"], d["s_unseen"]))):
            ax = axes[r, c]
            ax.scatter(s + rng.uniform(-0.02, 0.02, len(s)), 1 - p, s=14, alpha=0.55, color="#0072B2")
            rho = res[mname][cohort]["stats"]["survey_acceptable_vs_p_acceptable"]["spearman"]
            ax.set_title(f"{day} {mname} {cohort}  (Spearman {rho:.2f})")
            if r == 1:
                ax.set_xlabel("survey acceptable share (n_Acceptable / 5)")
            if c == 0:
                ax.set_ylabel("model P(Acceptable)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"confidence_survey_corr_{day}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", default=["Dy24", "Dy28", "Dy30"])
    args = ap.parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(sorted(args.days))  # per-day-set outputs so parallel jobs don't collide
    out_json = OUT_DIR / f"confidence_zones_{tag}.json"
    out_csv = OUT_DIR / f"confidence_zones_organoids_{tag}.csv"

    idor, unseen = build_datasets()
    results: dict = {}
    csv_rows: list[list] = []
    for day in sorted(args.days):
        X, y, names, ids = features(idor, day)
        Xu, yu, names_u, ids_u = features(unseen, day)
        assert names == names_u, "feature columns differ between IDOR and unseen"
        assert set(y) <= set(LABEL_TO_INT.values())
        s_idor = human_acceptable_share(idor, ids)
        s_unseen = human_acceptable_share(unseen, ids_u)
        results[day] = {"n_idor": len(ids), "n_unseen": len(ids_u)}
        data: dict = {}
        for mname, fn in MODELS.items():
            logger.info("[%s] %s: %dx%d-fold CV on %d IDOR + refit -> %d unseen", day, mname, N_REPEATS, N_FOLDS, len(ids), len(ids_u))
            p_idor, p_unseen = idor_oof_and_unseen(fn, X, y, Xu)
            data[mname] = {"p_idor": p_idor, "y_idor": y, "s_idor": s_idor,
                           "p_unseen": p_unseen, "y_unseen": yu, "s_unseen": s_unseen}
            results[day][mname] = {
                "idor_oof": {"stats": group_stats(p_idor, y, s_idor), "zones": zone_stats(p_idor, y)},
                "unseen": {"stats": group_stats(p_unseen, yu, s_unseen), "zones": zone_stats(p_unseen, yu)},
                "mannwhitney_p_unseen_pos_vs_neg": float(mannwhitneyu(p_unseen[yu == 1], p_unseen[yu == 0]).pvalue),
            }
            for cohort, oids, pp, yy, ss in (("idor_oof", ids, p_idor, y, s_idor), ("unseen", ids_u, p_unseen, yu, s_unseen)):
                for o, a, b, c in zip(oids, pp, yy, ss):
                    csv_rows.append([day, mname, cohort, o, f"{a:.4f}", int(b), f"{c:.1f}"])
            st = results[day][mname]
            logger.info("[%s] %s  IDOR BA=%.3f rho=%.2f | unseen BA=%.3f rho=%.2f", day, mname,
                        st["idor_oof"]["stats"]["balanced_accuracy"],
                        st["idor_oof"]["stats"]["survey_acceptable_vs_p_acceptable"]["spearman"],
                        st["unseen"]["stats"]["balanced_accuracy"],
                        st["unseen"]["stats"]["survey_acceptable_vs_p_acceptable"]["spearman"])
        plot_zones(day, results[day])
        plot_box(day, data)
        plot_corr(day, data, results[day])
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "model", "cohort", "organoid_id", "p_not_acceptable", "label_na", "survey_acceptable_share"])
        w.writerows(csv_rows)
    logger.info("wrote %s, %s", out_json, out_csv)


if __name__ == "__main__":
    main()
