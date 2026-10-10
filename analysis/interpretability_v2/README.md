# Interpretability analysis — V2 (2026-10): larger, edge-filtered cohort

V1 (`../interpretability_v1/`) required a usable image on every day 3–30, which dropped
58 organoids, mostly ones that grew large enough to touch the frame edge on day 24/28.
V2 keeps them: each day uses every organoid whose image on that day passes the edge
filter (≤ 5%), and the LSTM skips cut-off days instead of dropping the organoid.

Cohort: `data/cohorts/idor_main/full` (IDOR Classified, not split, ≥ 4/5 raters agree).

| | organoids | bad |
|---|---|---|
| V1 (complete series) | 140 | 25 |
| V2 per-day CNN, days 3–20.5 | 197–198 | 33 |
| V2 per-day CNN, day 24 / 28 / 30 | 186 / 163 / 155 | 31 / 27 / 30 |
| V2 LSTM, days 3–13 / 3–17 / 3–24 / 3–30 | 198 / 198 / 186 / 155 | 33 / 33 / 31 / 30 |

Same 4-fold × 10-repeat CV as V1 (folds rebuilt per day). Code: `cb365d7`
(`--exact-day` for the CNN; padded sequences + masked `last`/`mean` readout and
`--require-last-day` for the LSTM). CNN inputs are undistorted (`ee24cd0`).

## Findings

1. **Size works both ways.** At day 30, 61% of the smallest fifth are bad, 0–10% of the
   middle, and 23% of the largest fifth. Too small = stalled; too big = overgrown
   (cysts, and the large organoids V1 had dropped). A size model that allows both
   ("two-way", spline on size) reaches AUC 0.87 at day 30; the bigger-is-better size model
   only 0.70.
2. **No image model beats two-way size.** Against size + plate the bbox CNNs looked strong
   at day 30 (+0.10 to +0.14 AUC, 9–10/10 repeats) and ranked the big-but-bad organoids
   (cysts, new bad ones) as bad. Against two-way size + plate the gain is ≤ +0.03 on every
   day for every CNN and both LSTMs (about half the repeats positive), i.e. the CNNs learned
   "too big is bad" — still size.
3. **The V1 full-series rule hid the overgrowth failure mode.** With only 3 big-but-bad
   organoids (the cysts) V1 models could not learn it; adding the 5 edge-touching bad ones
   is what made it learnable.
4. **LSTM:** `last` readout beats `mean` pooling late (days 3–30: 0.82 vs 0.75); `mean` is
   better only for days 3–13 (0.60 vs 0.50). Neither beats two-way size + growth.
5. **Same-size pairs:** the bbox CNNs are above chance at days 15–20.5 and 30 (0.60–0.71),
   as in V1; within BA2 alone this mostly shrinks, and it adds nothing over two-way size
   + plate (finding 2).

Caveat: the two-way size baseline places spline knots using all organoids of the day
(not refit per fold); the effect on AUC is negligible but it is not strictly out-of-fold.

## Files
- `figures/v2_summary.png|pdf` — per-day AUC vs both size models; same-size AUC; LSTM vs
  size + growth.
- `results/perday_cnn.csv`, `results/perday_lstm.csv` — per day/window: n, n_bad,
  `size_auc` (bigger = better), `flex_auc` (two-way), `auc`, `old_run_auc` (V1 run),
  `old_auc` (V2 model scored on the V1 140), same-size AUC, gains over size + plate
  (`gain`) and two-way size + plate (`gain_flex`) with repeats won, BA2-only columns.
- `results/perday_ranks.csv` — where each model ranks the 3 cysts and the new bad organoids.
- `scripts/perday_eval.py` — produces all of the above (run on the cluster, see docstring).

Runs: `/net/projects2/promega/project_data/model_tests/lstm_runs/idor_main_perday/`
(`base_effnet_kfold4x10_{std,strong,bbox,brightbbox}`, `temporal_lstm_kfold4x10_{last,mean}`).
