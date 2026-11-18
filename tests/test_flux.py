"""Pedro's corrected metabolite flux (2026-10) folded into the flux-enriched data.

Covers the ingest rules (label parsing, Dy20/21 -> Dy20.5, exclusions), the data
semantics Pedro described (flux = (end - blank) * 50 uL / day; negatives zeroed), that
legacy fields are untouched, and the flux feature matrix / winsorization.
"""

import numpy as np
import pytest
from invariants import assert_subset

from pipeline.data_loader import REQUIRED_METABOLITES
from pipeline.metabolites.flux import ALL_ASSAYS, FLUX_FIELDS, label_to_record_id
from pipeline.metabolites.winsorize import LOW_Q, winsorize_per_day

FLUX = "flux_nmol_per_day"
DAY = "Dy30"
ASSAY_VOLUME_UL = 50.0


@pytest.mark.parametrize(
    "label, expected",
    [
        ("BA1_96_1_Dy03_A1", "BA1 96_1 Dy03 A1"),
        ("BA2_96_2_Dy28_h12", "BA2 96_2 Dy28 H12"),
        ("BA1_96_1_Dy20_H11", "BA1 96_1 Dy20.5 H11"),
        ("BA2_96_2_Dy21_C5", "BA2 96_2 Dy20.5 C5"),
    ],
)
def test_label_to_record_id(label, expected):
    assert label_to_record_id(label) == expected


def test_label_to_record_id_rejects_garbage():
    with pytest.raises(ValueError):
        label_to_record_id("not_a_label")


def _blocks(data):
    for rid, rec in data.items():
        for assay, blk in (rec.get("metabolite") or {}).items():
            yield rid, rec, assay, blk


def test_legacy_fields_untouched(all_data, flux_data):
    assert set(all_data) == set(flux_data), "flux ingest changed the record set"
    for rid, _rec, assay, blk in _blocks(all_data):
        new = flux_data[rid]["metabolite"][assay]
        for k, v in blk.items():
            assert new.get(k) == v, f"{rid}/{assay}/{k} changed by flux ingest"


def test_flux_fields_present_and_json_safe(flux_data):
    n = 0
    for rid, _rec, assay, blk in _blocks(flux_data):
        if FLUX not in blk:
            continue
        n += 1
        for k in FLUX_FIELDS:
            v = blk[k]
            assert v is None or np.isfinite(v), f"{rid}/{assay}/{k}={v!r} (NaN must be null)"
    assert n > 20000, f"only {n} flux blocks written"


def test_pedro_zero_correction(flux_data):
    # Negative blank / end concentration are physically impossible; Pedro zeroed them.
    for rid, _rec, assay, blk in _blocks(flux_data):
        for k in ("end_concentration_mM", "blank_mM"):
            v = blk.get(k)
            assert v is None or v >= 0, f"{rid}/{assay}/{k}={v} is negative"


def test_flux_matches_end_minus_blank(flux_data):
    # On Dy03 the interval is one day, so flux == (end - blank) * 50 uL (nmol = mM * uL).
    ok = total = 0
    for rid, _rec, _assay, blk in _blocks(flux_data):
        if rid.split()[2] != "Dy03":  # NB: record ``day.id`` is "Dy3", so key on the record id
            continue
        f, e, b = blk.get(FLUX), blk.get("end_concentration_mM"), blk.get("blank_mM")
        if None in (f, e, b):
            continue
        total += 1
        ok += abs(f - (e - b) * ASSAY_VOLUME_UL) <= 1e-3 * max(1.0, abs(f))
    assert total > 500
    assert ok / total > 0.99, f"flux == (end-blank)*50 only for {ok}/{total} Dy03 blocks"


def test_exclusions_are_nulled(flux_data):
    from pipeline.metabolites.flux import load_flux

    xlsx = "/net/spaces/clinic/promega/2026_04_15_data/raw/corrected_metabolite_from_pedro_202610.xlsx"
    _values, exclusions = load_flux(xlsx)
    assert exclusions, "no exclusions parsed"
    checked = 0
    for rid, assays in exclusions.items():
        if rid not in flux_data:
            continue
        mets = (flux_data[rid].get("metabolite") or {})
        for assay, blk in mets.items():
            if ALL_ASSAYS in assays or assay in assays:
                if FLUX in blk:
                    checked += 1
                    assert blk[FLUX] is None, f"{rid}/{assay} excluded but flux={blk[FLUX]}"
    assert checked > 50


def test_cohort_unchanged_by_flux_ingest(flux_cohort, full_cohort):
    # Flux ingest must not change which organoids / labels define the cohort.
    assert set(flux_cohort.organoid_ids) == set(full_cohort.organoid_ids)


def test_flux_feature_matrix_join_integrity(flux_cohort):
    ds = flux_cohort
    X, y, names, ids = ds.get_metabolite_features(
        "all", DAY, include_growth=False, include_initial=True, field=FLUX
    )
    assert X.shape[0] == len(y) == len(ids) > 0
    assert X.shape[1] == len(names) == 2 * len(REQUIRED_METABOLITES)
    assert len(set(ids)) == len(ids), "duplicate organoid ids"
    assert_subset(ids, ds.organoid_ids, context=f"flux features@{DAY}")
    assert set(np.unique(y)).issubset({0, 1})
    flux_cols = [i for i, n in enumerate(names) if n.endswith(f"_{FLUX}")]
    assert len(flux_cols) == len(REQUIRED_METABOLITES)
    assert not np.isnan(X[:, flux_cols]).any(), "NaN in flux columns"
    # Secondary column is the END concentration (not legacy initial_concentration).
    assert all(
        f"{m}_end_concentration_mM" in names for m in REQUIRED_METABOLITES
    ), names


def test_flux_not_clipped_by_um_floor(flux_cohort):
    # Glucose consumption is large and negative (< -500); it must survive as real signal.
    X, _y, names, _ids = flux_cohort.get_metabolite_features(
        "all", DAY, include_initial=False, field=FLUX
    )
    assert not np.isnan(X).any()
    assert (X < -500).any(), "no flux below -500; floor was applied to flux?"


def test_flux_growth_only_reduces(flux_cohort):
    ds = flux_cohort
    _, _, _, ids0 = ds.get_metabolite_features("all", DAY, include_growth=False, field=FLUX)
    _, _, _, ids1 = ds.get_metabolite_features("all", DAY, include_growth=True, field=FLUX)
    assert_subset(ids1, ids0, context="flux growth vs no-growth")


def test_flux_winsorized_preserves_organoids(flux_cohort):
    ds = flux_cohort
    _, _, _, ids_raw = ds.get_metabolite_features("all", DAY, field=FLUX, winsorize=False)
    _, _, _, ids_win = ds.get_metabolite_features("all", DAY, field=FLUX, winsorize=True)
    assert set(ids_raw) == set(ids_win)


@pytest.mark.parametrize("met", sorted(REQUIRED_METABOLITES))
def test_persisted_flux_win_matches_per_day_winsorization(flux_data, met):
    # Rule 16: the persisted ``flux_nmol_per_day_win`` must equal per-day 1/99 clipping
    # of the persisted flux.
    vals, days, stored = [], [], []
    for rec in flux_data.values():
        blk = (rec.get("metabolite") or {}).get(met)
        if not blk or FLUX not in blk:
            continue
        v = blk[FLUX]
        vals.append(np.nan if v is None else v)
        days.append((rec.get("day") or {}).get("id"))
        w = blk.get(f"{FLUX}_win")
        stored.append(np.nan if w is None else w)
    vals, stored = np.array(vals), np.array(stored)
    expected = winsorize_per_day(vals, days, LOW_Q, 1 - LOW_Q)
    finite = np.isfinite(vals)
    assert finite.sum() > 3000
    assert np.allclose(stored[finite], expected[finite]), f"{met}: stale/incorrect flux_win"
    assert np.isnan(stored[~finite]).all(), f"{met}: flux_win present where flux is null"
