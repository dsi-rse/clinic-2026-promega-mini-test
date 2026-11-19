#!/usr/bin/env python3
"""Metabolite classifier, 5-fold CV on the series_idor cohort: flux vs old concentration.

Feature sets (per day, all 6 metabolites):
  met_conc — old features: ``concentration_uM`` + ``initial_concentration``
             (values < CONCENTRATION_FLOOR -> NaN, as in the paper pipeline)
  met_flux — Pedro's corrected fields: ``flux_nmol_per_day`` + ``end_concentration_mM``

Both use the SAME organoids and the SAME fold assignments, so the comparison is
paired. Models: LightGBM and LogReg (grid search in the inner 3-fold, as in
``met_lgbm_logreg_kfold``). Fold structure mirrors ``combined_kfold`` (15% of the
training fold held out, then k-fold on the full labeled organoid list).

Input is a flux-bearing JSON (default ``data/all_data_flux.json``) because
``data/all_data.json`` does not yet carry the flux fields.

Output: $ANALYSIS_OUTPUT_DIR/metabolites/met_flux_kfold_series_idor.json

Usage:
    make run ARGS="-m analysis.paper_2026_04.met_flux_kfold"
    make run ARGS="-m analysis.paper_2026_04.met_flux_kfold --days Dy30 --n-repeats 1"
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from pipeline.data_loader import (
    ANALYSIS_OUTPUT_DIR,
    CONCENTRATION_FLOOR,
    DAY_ORDER,
    LABEL_TO_INT,
    REQUIRED_METABOLITES,
    OrganoidDataset,
    idor_ba1_ba2_filters,
    require_complete_series,
)

from .met_lgbm_logreg_kfold import _train_lgbm_fold, _train_logreg_fold

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

SEED = 42
N_FOLDS = 5
N_REPEATS = 1
EXPECTED_N = 134  # series_idor, stitched excluded (see notes/cohorts_cheatsheet.md)
DEFAULT_DATA = "data/all_data_flux.json"
OUTPUT_PATH = ANALYSIS_OUTPUT_DIR / "metabolites" / "met_flux_kfold_series_idor.json"

# feature-set name -> (primary field, secondary field)
FEATURE_FIELDS: dict[str, tuple[str, str]] = {
    "met_conc": ("concentration_uM", "initial_concentration"),
    "met_flux": ("flux_nmol_per_day", "end_concentration_mM"),
}


def _growth_columns(
    ds: OrganoidDataset,
    day: str,
    prev_day: str,
    org_ids: list[str],
    field: str,
    floor_concentration: bool,
) -> np.ndarray:
    """(n_organoids, 6) array of field(day) - field(prev_day); NaN if either is missing."""

    def value(oid: str, d: str, met: str) -> float:
        rec = ds.get_record(oid, d)
        v = (((rec or {}).get("metabolite") or {}).get(met) or {}).get(field)
        if v is None:
            return np.nan
        if floor_concentration and field == "concentration_uM" and v < CONCENTRATION_FLOOR:
            return np.nan
        return float(v)

    G = np.array(
        [[value(o, day, m) - value(o, prev_day, m) for m in REQUIRED_METABOLITES] for o in org_ids]
    )
    n_nan = int(np.isnan(G).any(axis=1).sum())
    if n_nan:
        logger.warning(
            "[%s] %s growth vs %s: %d/%d organoids have a NaN delta (kept)",
            day,
            field,
            prev_day,
            n_nan,
            len(org_ids),
        )
    return G


def build_features(
    ds: OrganoidDataset,
    day: str,
    org_ids: list[str],
    fields: tuple[str, str],
    *,
    floor_concentration: bool,
    growth: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix (rows follow ``org_ids``; NaN where a value is missing) + names.

    Reads persisted fields only (rule 16). Organoids are never dropped here
    (rule 11): a missing value becomes NaN and is counted/logged.

    With ``growth`` a per-metabolite delta of the primary field vs the previous
    day (value_t - value_{t-1}) is appended; none at Dy03. Unlike
    ``OrganoidDataset._add_growth_features`` (which drops organoids lacking a
    previous-day value), a missing delta is NaN so the cohort stays at 134.
    """
    names = [f"{m}_{f}" for m in REQUIRED_METABOLITES for f in fields]
    X = np.full((len(org_ids), len(names)), np.nan)
    n_missing_rows = 0
    for i, oid in enumerate(org_ids):
        rec = ds.get_record(oid, day)
        mets = (rec or {}).get("metabolite") or {}
        row_missing = False
        for j, met in enumerate(REQUIRED_METABOLITES):
            block = mets.get(met) or {}
            for k, field in enumerate(fields):
                v = block.get(field)
                if v is None:
                    row_missing = True
                    continue
                if floor_concentration and field == "concentration_uM" and v < CONCENTRATION_FLOOR:
                    v = np.nan
                X[i, 2 * j + k] = v
        n_missing_rows += row_missing
    if n_missing_rows:
        logger.warning(
            "[%s] %s: %d/%d organoids have >=1 missing value (kept as NaN)",
            day,
            fields[0],
            n_missing_rows,
            len(org_ids),
        )
    assert X.shape[0] == len(org_ids), "organoid count changed in build_features"
    if growth and day in DAY_ORDER and DAY_ORDER.index(day) > 0:
        prev_day = DAY_ORDER[DAY_ORDER.index(day) - 1]
        G = _growth_columns(ds, day, prev_day, org_ids, fields[0], floor_concentration)
        X = np.hstack([X, G])
        names = names + [f"{m}_{fields[0]}_growth" for m in REQUIRED_METABOLITES]
        assert X.shape[0] == len(org_ids), "organoid count changed adding growth"
    return X, names


def run_day(
    day: str,
    ds: OrganoidDataset,
    all_org_ids: list[str],
    all_labels: np.ndarray,
    n_folds: int,
    n_repeats: int,
    growth: bool = False,
    fold_spec: list[dict] | None = None,
) -> dict | None:
    """Per-day CV. With ``fold_spec`` (``repeat_details`` of a combined_kfold JSON) the
    organoid order, labels, per-repeat seeds and fold assignments are taken from it, so
    the folds are identical to the other modalities'."""
    if fold_spec is not None:
        all_org_ids = list(fold_spec[0]["org_ids"])
        all_labels = np.array(fold_spec[0]["true_labels"], dtype=int)
        n_repeats = len(fold_spec)
        n_folds = int(max(fold_spec[0]["fold_assignments"])) + 1
    feature_sets = {}
    for name, fields in FEATURE_FIELDS.items():
        X, _ = build_features(
            ds,
            day,
            all_org_ids,
            fields,
            floor_concentration=(name == "met_conc"),
            growth=growth,
        )
        feature_sets[name] = X

    mod_keys = [f"{p}_{m}" for p in feature_sets for m in ("lgbm", "logreg")]
    repeat_bas: dict[str, list[float]] = {k: [] for k in mod_keys}
    repeat_cms: dict[str, list] = {k: [] for k in mod_keys}
    repeat_details: list[dict] = []

    for rep in range(n_repeats):
        rep_seed = SEED + rep * 1000
        oof = {k: np.full(len(all_org_ids), np.nan) for k in mod_keys}
        if fold_spec is not None:
            spec = fold_spec[rep]
            assert spec["org_ids"] == all_org_ids, "organoid order differs between repeats"
            rep_seed = int(spec["seed"])
            fa = np.array(spec["fold_assignments"])
            splits = [(np.where(fa != f)[0], np.where(fa == f)[0]) for f in range(n_folds)]
        else:
            outer_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep_seed)
            splits = list(outer_cv.split(all_org_ids, all_labels))

        for fold_i, (tr_idx, te_idx) in enumerate(splits):
            fold_seed = rep_seed + fold_i * 97
            sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=fold_seed)
            inner_idx, _ = next(sss.split(tr_idx, all_labels[tr_idx]))
            tr_final = tr_idx[inner_idx]

            for name, X in feature_sets.items():
                y_tr = all_labels[tr_final]
                for model, trainer in (("lgbm", _train_lgbm_fold), ("logreg", _train_logreg_fold)):
                    p = trainer(X[tr_final], y_tr, X[te_idx], fold_seed)
                    if p is not None:
                        oof[f"{name}_{model}"][te_idx] = p
            print(f"  [{day}] rep={rep + 1}/{n_repeats} fold={fold_i + 1}/{n_folds}", flush=True)

        for k in mod_keys:
            valid = ~np.isnan(oof[k])
            assert valid.sum() == len(all_org_ids), (
                f"{k}: OOF covers {valid.sum()}/{len(all_org_ids)} organoids"
            )
            yp = (oof[k] >= 0.5).astype(int)
            repeat_bas[k].append(float(balanced_accuracy_score(all_labels, yp)))
            repeat_cms[k].append(confusion_matrix(all_labels, yp, labels=[0, 1]).tolist())

        repeat_details.append(
            {
                "seed": rep_seed,
                "fold_assignments": fa.tolist() if fold_spec is not None else None,
                "org_ids": all_org_ids,
                "true_labels": all_labels.tolist(),
                "oof_probs": {k: [float(v) for v in oof[k]] for k in mod_keys},
            }
        )

    results: dict = {}
    for k in mod_keys:
        bas = repeat_bas[k]
        results[k] = {
            "balanced_accuracy_mean": float(np.mean(bas)),
            "balanced_accuracy_std": float(np.std(bas)),
            "n_repeats": len(bas),
            "n_folds": n_folds,
            "repeat_balanced_accuracies": bas,
            "repeat_confusion_matrices": repeat_cms[k],
        }
    results["repeat_details"] = repeat_details
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-data", default=DEFAULT_DATA)
    parser.add_argument("--days", nargs="+", default=None)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument(
        "--growth",
        action="store_true",
        help="append day-over-day delta of the primary field per metabolite",
    )
    parser.add_argument(
        "--folds-from",
        type=str,
        default=None,
        help="Per-day combined_kfold JSON template with {day}, e.g. "
        "analysis_output/images/combined_results_kfold_series_idor_133_{day}.json. "
        "Uses its organoids, seeds and fold assignments (identical folds); "
        "--n-folds/--n-repeats are then read from it.",
    )
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    ds = OrganoidDataset(
        args.all_data,
        splits=None,
        filters=[*idor_ba1_ba2_filters(), require_complete_series(drop_stitched=True)],
    )
    all_org_ids = sorted(o for o in ds.organoid_ids if ds.organoid_label(o) in LABEL_TO_INT)
    all_labels = np.array([LABEL_TO_INT[ds.organoid_label(o)] for o in all_org_ids])
    assert len(all_org_ids) == EXPECTED_N, (
        f"series_idor cohort expected {EXPECTED_N}, got {len(all_org_ids)}"
    )
    n0 = len(all_org_ids)
    print(
        f"Organoids: {len(all_org_ids)}  ({all_labels.sum()} NAcc, {(all_labels == 0).sum()} Acc)"
    )
    print(
        f"Protocol: {args.n_repeats}x{args.n_folds}-fold  SEED={SEED}  growth={args.growth}  data={args.all_data}"
    )

    all_results: dict[str, dict] = {}
    for day in args.days or list(DAY_ORDER):
        if day not in ds.days:
            logger.warning("[%s] not in dataset, skipping", day)
            continue
        print(f"\n[{day}]")
        fold_spec = None
        if args.folds_from:
            fpath = Path(args.folds_from.format(day=day))
            if not fpath.exists():
                logger.warning("[%s] fold file %s not found, skipping", day, fpath)
                continue
            fold_spec = json.loads(fpath.read_text())[day]["repeat_details"]
            ids = fold_spec[0]["org_ids"]
            assert set(ids) <= set(all_org_ids), "fold file has organoids outside the cohort"
            logger.info(
                "[%s] folds from %s: %d organoids (cohort %d)", day, fpath.name, len(ids), n0
            )
        res = run_day(
            day, ds, all_org_ids, all_labels, args.n_folds, args.n_repeats, args.growth, fold_spec
        )
        if res:
            all_results[day] = res
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nSaved -> {args.out}")
    print(
        f"\n{'Day':<8}"
        + "".join(f"{k:>16}" for k in ("conc_lgbm", "flux_lgbm", "conc_logreg", "flux_logreg"))
    )
    for day in DAY_ORDER:
        r = all_results.get(day)
        if not r:
            continue
        cells = [
            r[f"met_{a}_{b}"]["balanced_accuracy_mean"]
            for b in ("lgbm", "logreg")
            for a in ("conc", "flux")
        ]
        # order: conc_lgbm, flux_lgbm, conc_logreg, flux_logreg
        print(f"{day:<8}" + "".join(f"{c:>16.3f}" for c in cells))


if __name__ == "__main__":
    main()
