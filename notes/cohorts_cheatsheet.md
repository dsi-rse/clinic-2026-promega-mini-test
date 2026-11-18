# Cohort cheat sheet — which organoid set is which

Counts verified 2026-11-18 against `data/all_data.json` and `notes/Organoids_list_IDOR.csv`.
All cohorts below are **BA1 + BA2 only** (BA3 has no metabolite data; BA4 is the unseen batch, in no cohort here).

## The cohorts

```
266  IDOR list, column 1
 ├─ 248  reached Dy30 + Dy30 survey          → paper "full cohort" (majority vote)
 │   └─ 198  strong consensus (>=4 of 5)     → paper "strong-consensus cohort"
 └─ 134  clean images on all 11 days         → fusion k-fold cohort (`series_idor`)
          (all 134 are inside the 248, and all are >=4/5 votes)
```

| Name | n | BA1 | BA2 | Definition | Where it comes from |
|---|---|---|---|---|---|
| IDOR list (col 1) | 266 | 96 | 170 | Partner-curated "analyzed" organoids, split ones excluded, day/Stitched stripped to unique IDs. Many never reached Dy30. | `notes/Organoids_list_IDOR.csv` col 1 |
| Full cohort | 248 | 92 | 156 | The 266 that reached Dy30 and have a Dy30 survey (5 regular votes each). Majority vote always exists with 5 raters (5-0, 4-1, 3-2), so nobody is dropped for disagreement. | CSV col 2 |
| Strong consensus | 198 | 78 | 120 | Subset of 248 with top vote >= 4 of 5. The other 50 are 3-2 splits. | `label.regular_votes` at Dy30 |
| `series_idor` | 134 | 56 | 78 | The 266 filtered to a complete clean image series (rules below). 140 if stitched are kept. | `filters_for_mode("series_idor")` in `pipeline/data_loader.py` |
| Canonical split (old) | 220 | 78 | 142 | BA1+BA2, >=4 consensus at Dy30, complete metabolites, valid images. **Not a paper cohort.** | `data/splits/canonical_2026_winter.csv` |

Vote breakdown of the 248: 5/5 = 119, 4/5 = 79, 3/5 = 50.

## `series_idor` rules (every one of the 11 days, Dy03..Dy30)

- BA1/BA2 and on the IDOR list (col 1, the 266)
- Day record exists (no missing days)
- Not a blank image (`verification.blank`)
- `edge_fraction` present and <= 0.05
- Not split, not stitched (`drop_stitched`; turning it off gives 140)
- Clipped mean-fill image path exists

It does not check votes or metabolites.

## The 18 in the 266 but not in the 248

13 have no Dy30 record at all; 5 reach Dy30 but have 0 survey votes.

## Open questions

1. **132 vs 134.** Figure 7 caption and `combined_kfold.py` comments say n = 132 (139 with stitched).
   Current data gives 134 (140). The saved `combined_results_kfold_series_idor.json` (under
   `$DATA_ROOT/analysis_output/images/`) would show which organoids the figure really used.
2. **No 3-2 organoids in the 134 — explained.** The 50 organoids with 3-2 regular votes have
   `label.value = None` in `all_data.json` (stored consensus needs >=4 of 5). `OrganoidDataset._build`
   skips any organoid whose label is `None`, so every loader-built dataset is automatically >=4/5.
   Consequence: the loader can only build the 198 cohort. The paper's "full cohort of 248 by majority
   vote" cannot come from `label.value`; its 50 3-2 labels must be derived from raw `regular_votes`
   or another source. Check how any "full cohort" figure labelled them.
3. **What "analyzed" means** in the IDOR list header is not defined beyond the CSV's own header text.
4. **220 vs 198/248.** `STATUS.md` still calls 220 the >=4-consensus count; the abstract says 198.
   The 220 predates the IDOR-list cohorts.

## Flux fields (separate topic)

`data/all_data_flux.json` has `flux_nmol_per_day`, `end_concentration_mM`, `blank_mM` in every
assay block, but `data/all_data.json` does not. See `notes/HANDOFF_2026_11.md`.

Source: `data/Corrected metabolite values to UChicago.xlsx` (sheets `Metabolites`, `MEtabolite Exclusions`).
Flux = (end - blank) * volume / day; negative = consumed. 25,344 rows (4,224 organoid-days x 6 assays),
22,717 with a flux number, 22,577 valid after exclusions.

Why 2,789 of 24,924 assay blocks have null flux (checked against the xlsx, 2026-11-18):

| Null blocks | Batch | Cause |
|---|---|---|
| 135 | BA1 (36), BA2 (99) | On Pedro's exclusion sheet (115 entries / 140 rows; 96 are MalateGlo). The xlsx *has* a flux number for them; `flux.py` nulls it on purpose. Largest gap: BA2 96_2 Dy10 (89). |
| 2,654 | BA4 | No `Blank (mM)` in the xlsx (BA4 blanks exist Dy03-Dy15, partial at Dy17, none from Dy21), so flux is empty. BA4 is in no cohort here. |

Blank looks like a fixed medium constant: 6 distinct values per day (one per metabolite), identical
across BA1/BA2 plates. Filling BA4 blanks from BA1/BA2 is possible but unconfirmed (ask Pedro). Parked.

### Flux vs concentration classifier (5-fold CV, `series_idor` n=134)

Script `analysis/paper_2026_04/met_flux_kfold.py` (`make run ARGS="-m analysis.paper_2026_04.met_flux_kfold [--growth] --n-repeats 10"`).
Same organoids and folds for both feature sets; LGBM + LogReg; missing values kept as NaN (n stays 134).
- `met_conc`: `concentration_uM` + `initial_concentration` (floor -500 uM -> NaN). `met_flux`: `flux_nmol_per_day` + `end_concentration_mM` (no floor).
- Result (10x5, no growth): flux ~ concentration. Paired differences mostly within +-0.03, below the
  per-repeat SD (0.02-0.04); only clear gap is Dy03 LogReg (flux 0.068 worse, 0/10 repeats better).
  Both weak before ~Dy20 (0.5-0.7), strong from Dy24 (0.82-0.93). LogReg >= LGBM throughout.
- `--growth` appends per-metabolite delta of the primary field vs the previous day (none at Dy03).
- Outputs (gitignored): `analysis_output/metabolites/met_flux_kfold_series_idor_10x5[_growth].json`.

## Inverted-image surveys

26 organoids in `all_data.json` have `label.inverted_votes` (a second survey pass with the image
inverted) — not 2. 19 are in the 248 and 15 are in the 134, so they are not removed as low quality;
the inverted bucket is only a disagreement veto on the label (`surveys_mapper.compute_survey_majority`).
In the 248 but not the 134: `BA1 96_1 A8`, `BA2 96_1 F3`, `BA2 96_1 H3`, `BA2 96_2 A2` (H3 and A2 also
have null labels). Pedro's "two organoids with inverted image" is unidentified — may be a different
thing (e.g. flipped images).
