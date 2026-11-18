# Handoff — 2026-11 (before cluster downtime / relocation)

Snapshot of where `lding/combined_model` stood when the cluster home folders were wiped
(security incident) and development moved to a new folder/location. Written from the repo
contents only; Claude's per-project memory did not survive, so this file replaces it.

Committed state: `3c46853` ("cluster version 2026.11 before major cluster downtime"),
plus `b91a66f` (2026-06-04 worklog). Branch `lding/combined_model`, pushed to origin.

## 1. What to verify first in the new location

1. `$DATA_ROOT` (default `/net/projects2/promega/2026_04_15_data`) is reachable. The
   scripts read saved results from `$DATA_ROOT/analysis_output/` and cannot be rerun
   without it. Specifically check `analysis_output/images/combined_results_kfold_series_idor_139.json`.
2. The conda env. It lived at `/net/projects/CLS/lding/conda_env/core_env` (outside
   home, because the install exhausted the home quota — see `worklog_2026_06_04.md`).
   Rebuild from `core_env.yaml` with `--prefix <project-storage-path>` if it is gone.
3. The Pedro flux source file `corrected_metabolite_from_pedro_202610.xlsx` (not found
   under `raw/` when checked). Needed to regenerate `data/all_data_flux.json`.

## 2. Hard-coded paths to update when moving

| Where | What |
|---|---|
| `analysis/paper_2026_04/submit_*.slurm` (7 files, listed below) | `cd /net/spaces/clinic/CLS/lding/gitcode/2025-promega-mini-test` and `/net/projects/CLS/lding/conda_env/core_env` |
| `tests/test_flux.py` | references the `/net/spaces/clinic/...` checkout path |
| `analysis/paper_2026_04/combined_lgbm.py:65` | hard-coded `/net/projects2/promega/.../image_embeddings.csv` (violates "no hard-coded data paths"; should use `$DATA_ROOT`) |
| `run_all_lstm.sh`, `run_cohorts.sh` | `/net/projects2/promega/project_data/...` run/plot dirs |
| `analysis/multimodal/README.md` | `conda activate /net/projects2/promega` |

Slurm files with the old checkout path: `submit_confidence_ppt`, `submit_confidence_vs_survey`,
`submit_confidence_zones`, `submit_fusion_corrections`, `submit_fusion_corrections_ppt`,
`submit_met_lgbm_logreg_kfold_134`, `submit_morph134_ppt` (all `.slurm` in `analysis/paper_2026_04/`).

Everything else should go through `make run` / `$DATA_ROOT` per `AGENTS.md`.

## 3. Work in flight (committed but not tracked in beads)

**October 2026 confidence / fusion analysis** — `analysis/paper_2026_04/`:
- `confidence_vs_survey.py` — does model error concentrate on organoids the 5 raters
  disagreed on? (139 cohort, Dy30 regular votes; OOF probs from 10x repeated 4-fold CV.)
- `confidence_zones.py` — confidence zones; 248 IDOR (OOF) vs 96 unseen BA4 (refit on all 248);
  metabolite-only, LightGBM + LogReg; array job over Dy24/Dy28/Dy30.
- `fusion_corrections.py` — which organoids fusion corrects or breaks vs single sources.
- `make_confidence_ppt.py`, `make_fusion_corrections_ppt.py` — build `figures/*.pptx`
  from saved JSON (no recomputation).
- `data_overview_figure.py` — 5-panel overview figure.
- `combined_lgbm.py` — three-thread (met + morph + optional image PCA) LightGBM;
  corresponds to bead `eyq.16`.

**Metabolite flux (Pedro, 2026-10)**
- `pipeline/metabolites/flux.py` folds `flux_nmol_per_day`, `end_concentration_mM`,
  `blank_mM` into assay blocks and writes `data/all_data_flux.json` (new file; legacy
  fields untouched). `data_loader.py` and `winsorize.py` already know the new fields;
  `tests/test_flux.py` covers it.
- Gaps: no model has been trained on flux yet; `data/all_data.json` does not contain the
  flux fields (so rule 16 is not satisfied until it is regenerated through the pipeline);
  relationship between Pedro's `flux_nmol_per_day` and the paper's size-normalised
  exchange rate (nmol/day/um^3, see `analysis/2026_06_metabolite_pred/README.md`) is undocumented.

## 4. Housekeeping caveats

- The big commit `3c46853` swept in everything untracked: root `all_data.json` (13 MB),
  `data/all_data_flux.json` (31 MB), `figures/*.pptx`, and several legacy top-level
  folders (`combined_classifier/`, `image_classifier/`, `metabolite_classifier/`,
  `file_utils/`, `split_data*.py`, `config.py`, `Dockerfile`). Per `AGENTS.md`, generated
  figures and data copies should not live in the repo; consider removing them
  (`git rm --cached`) and gitignoring `figures/` and the root `all_data.json`.
- `STATUS.md` is mostly a 2026-05-12 snapshot; treat its "Not started" rows with care
  (e.g. vote-confidence stratification now exists as `confidence_vs_survey.py`).
- `bd` (beads) was not on `PATH` in the session that wrote this, so no beads were filed
  or synced. Beads to file: October confidence/fusion analysis; flux modelling +
  `all_data.json` regeneration; path cleanup above; repo cleanup of swept-in files.
