#!/usr/bin/env python3
"""Fold Pedro's corrected metabolite flux (2026-10) into ``all_data.json``.

Source: ``corrected_metabolite_from_pedro_202610.xlsx`` (sheet ``Metabolites``, long
format, one row per organoid-day-assay; sheet ``MEtabolite Exclusions`` lists
organoid-day/assay pairs to drop, ``ALL`` = every assay).

Semantics (per Pedro):
  * ``Flux_nmol_per_day`` is the value we model. Negative = consumed, positive = secreted.
  * ``Blank (mM)`` is the START concentration; the column misleadingly named
    ``Initial  Concentration (mM)`` is the END concentration. Pedro zeroed negatives
    in both (physically impossible).

Written into each assay block of ``record["metabolite"]``:
  ``flux_nmol_per_day``, ``end_concentration_mM``, ``blank_mM``  (null if missing/excluded).
Legacy fields (``concentration_uM`` ...) are left untouched.

Usage (writes a NEW file; never overwrites the input):
    PYTHONPATH=. python -m pipeline.metabolites.flux \
        --xlsx <raw>/corrected_metabolite_from_pedro_202610.xlsx \
        --all-data data/all_data.json --out data/all_data_flux.json
"""

import argparse
import json
import logging
import math
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DAY_20_21 = "Dy20.5"  # same collapse as metabolite_mapper (Dy20/Dy21 -> Dy20.5)
FLUX_FIELDS = ("flux_nmol_per_day", "end_concentration_mM", "blank_mM")
SOURCE_COLUMNS = {
    "flux_nmol_per_day": "Flux_nmol_per_day",
    "end_concentration_mM": "Initial  Concentration (mM)",
    "blank_mM": "Blank (mM)",
}
ALL_ASSAYS = "ALL"
_LABEL_RE = re.compile(r"^(BA\d+)_(96_\d+)_(Dy\d+)_([A-Za-z]\d+)$")


def label_to_record_id(label: str) -> str:
    """``BA1_96_1_Dy03_A1`` -> ``BA1 96_1 Dy03 A1`` (Dy20/Dy21 -> Dy20.5)."""
    m = _LABEL_RE.match(label.strip())
    if m is None:
        raise ValueError(f"unparseable label {label!r}")
    ba, plate, day, well = m.groups()
    if day in ("Dy20", "Dy21"):
        day = DAY_20_21
    return f"{ba} {plate} {day} {well.upper()}"


def _null(v: object) -> float | None:
    """NaN/None -> None so JSON never contains NaN."""
    if v is None:
        return None
    f = float(v)
    return None if math.isnan(f) else f


def load_flux(xlsx: Path) -> tuple[dict, dict]:
    """Return ``(values, exclusions)``.

    values:     ``{record_id: {assay: {flux_nmol_per_day, end_concentration_mM, blank_mM}}}``
    exclusions: ``{record_id: set(assay or "ALL")}``
    """
    df = pd.read_excel(xlsx, sheet_name="Metabolites")
    ex = pd.read_excel(xlsx, sheet_name=1)

    values: dict = {}
    for row in df.to_dict("records"):
        rec = label_to_record_id(row["Label"])
        block = {k: _null(row[c]) for k, c in SOURCE_COLUMNS.items()}
        values.setdefault(rec, {})[row["Metabolite"].strip()] = block

    exclusions: dict = {}
    for fname, met in zip(ex["Filename"], ex["Metabolite"]):
        exclusions.setdefault(label_to_record_id(fname), set()).add(str(met).strip())
    return values, exclusions


def _split_variants(all_data: dict) -> dict[str, list[str]]:
    """``{base_id: [base split_1, base split_2]}`` for split records in all_data."""
    out: dict[str, list[str]] = {}
    for rid in all_data:
        if "split" in rid:
            base = rid.replace("split_1", "").replace("split_2", "").strip()
            out.setdefault(base, []).append(rid)
    return out


def add_flux_fields(all_data: dict, values: dict, exclusions: dict) -> dict:
    """Write flux fields into ``all_data`` in place; return a count summary."""
    splits = _split_variants(all_data)
    n_written = n_excluded = n_unmatched = 0
    for rid, assays in values.items():
        targets = [rid] if rid in all_data else splits.get(rid, [])
        if not targets:
            n_unmatched += 1
            continue
        excluded = exclusions.get(rid, set())
        for t in targets:
            mets = all_data[t].setdefault("metabolite", {})
            for assay, block in assays.items():
                dest = mets.setdefault(assay, {})
                if ALL_ASSAYS in excluded or assay in excluded:
                    dest.update({k: None for k in FLUX_FIELDS})
                    n_excluded += 1
                else:
                    dest.update(block)
                    n_written += 1
    return {"written": n_written, "excluded": n_excluded, "unmatched_ids": n_unmatched}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, required=True)
    ap.add_argument("--all-data", type=Path, default=Path("data/all_data.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.resolve() == args.all_data.resolve():
        raise SystemExit("--out must differ from --all-data (refusing to overwrite input)")

    with open(args.all_data) as f:
        all_data = json.load(f)
    values, exclusions = load_flux(args.xlsx)
    summary = add_flux_fields(all_data, values, exclusions)
    logger.info("flux summary: %s", summary)
    with open(args.out, "w") as f:
        json.dump(all_data, f, indent=2)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
