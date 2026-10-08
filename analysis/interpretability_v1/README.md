# Interpretability analysis — V1 (2026-10)

What do image models use to predict organoid quality (Acceptable vs Not Acceptable at
day 30), and is there anything in the images beyond size and growth?

Cohort: `data/cohorts/idor_main` (140 organoids, 115 Acceptable / 25 Not Acceptable,
complete day 3–30 series, `main`'s `all_data.json`, official labels). All models are
evaluated with the same 4-fold × 10-repeat CV folds
(`train_base_model_kfold._fold_splits`), so every comparison is paired.

## Findings

1. **Growth after day 10 is the main predictor.** Acceptable and Not Acceptable
   organoids grow alike to day 10; afterwards Acceptable ones keep growing and
   Not Acceptable ones stall. Growth day 10→17 alone: AUC 0.84 (also on an unseen plate).
   Size history (days 3–30): AUC 0.86.
2. **The image models (CNN, strong aug, bbox crop, LSTM) mostly learn size.** They do
   not beat size/growth measurements; on same-size organoids they are near chance
   except a modest signal in days 15–20.5 (bbox models), likely shape/organisation.
   Rescaling an organoid (same shape/texture) moves the CNN's prediction → it uses size
   directly. bbox cropping does not fully remove size (resampling blur + shape).
3. **Beyond size:** large pale, cyst-like (translucent) regions at days 28–30 mark a
   distinct failure mode — cystic overgrowth — that growth curves rate as *good*
   (E12, H2, C7). Adding translucency raises day-30 AUC 0.86 → 0.92 (CV; threshold
   chosen by eye, not yet validated on an unseen plate).
4. **Why the 25 were rejected:** 19 stalled growth (caught by size), 3 cystic
   overgrowth, 2 misshapen (tail / chain of lobes), 1 unexplained (G7).
5. **Did not help:** 15 hand-made shape/texture/edge features (only the day-30
   cyst/brightness group survives FDR), brightened-interior texture (14 measures, none
   survive), CNNs trained on brightened images (no gain beyond size), adding the 38
   borderline (2/5, 3/5) organoids to training, day-10 "translucent buds" (both groups).

### Caveats
- **Batch effect.** BA2 organoids are larger and more often rejected (5% vs 17–35%).
  Early-day size signals (days 3–10) are largely batch; late size and growth rates hold
  within batch and on an unseen plate (leave-one-plate-out). BA1 has only 3 rejects.
- **Labels are noisy.** 40% of 5-rater organoids had a non-unanimous vote; raters only saw
  day 30 (day-30 results are partly "raters see size"; earlier days are true forecasts).
- Day-30 size correlates with the share of Acceptable votes (ρ = 0.48).

## Figures (`figures/`, PNG + PDF)

| # | Figure | Shows |
|---|---|---|
| 00 | summary | one-page summary: growth curves, day 10→17 growth, simple vs AI models, why rejected |
| 01 | growth_curves_size_by_outcome | size over days by outcome; per-day size AUC (pooled — early reversal is mostly batch) |
| 02 | size_vs_cnn_balanced_accuracy | size / size trajectory vs CNNs (std, strong, bbox) by day |
| 03 | occlusion_auc_by_condition | silhouette, rim/core texture removed, shifted, rescaled, organoid removed |
| 04 | rescale_intervention_size_effect | enlarging/shrinking an organoid moves CNN predictions (from day 17) |
| 05 | size_vs_shape_auc | scale-free shape vs size vs both |
| 06 | gradcam_heat_on_organoid | share of Grad-CAM heat on organoid (toward Acceptable; superseded by 13) |
| 07 | gradcam_rotation_consistency | heatmaps follow the organoid under rotation (both augs) |
| 08 | per_class_recall_cnn | CNN catches ~55–65% of Not Acceptable late |
| 09 | explain_conditions_what_model_sees | exact model inputs: original, bbox, shrunk, enlarged, silhouette |
| 10–12 | bbox_* | bbox predictions still track size; cropping creates a sharpness cue; resolution test |
| 13 | gradcam_focus_std_strong_bbox_predclass | predicted-class Grad-CAM, raw share vs organoid/background contrast |
| 14 | size_matched_auc | same-size pairs: CNNs ≈ chance; size history still predictive |
| 15 | size_by_votes_incl_borderline | size rises with # Acceptable votes; borderline in between |
| 16 | big_but_rejected_vs_matched_accepted | the big-but-rejected organoids vs same-size accepted, days 13–30 |
| 17 | translucent_regions | cyst-like translucent area: specific to rejection, adds at day 30 |
| 18 | cyst_organoids_growth | cyst organoids: 5 stall (growth flip), 3 overgrow |
| 19 | edges_day30_raw_size_matched | raw full-res edge crops, same-size pairs (no clear layer difference) |
| 20 | feature_screen_beyond_size | 15 features × 11 days, gain over plate-adjusted size, FDR |
| 21 | day10_translucency_good_vs_bad | day-10 "translucent" pixels = surface buds, in both groups |
| 22 | texture_brightened_day24_size_matched | brightened interiors, same-size pairs |
| — | gradcam_examples/ | rotation Grad-CAM, std vs strong, day 24/30 cases |

## Results tables (`results/`)
`size_trajectory_summary.csv`, `size_matched_auc.csv`, `size_plus_model_lrt.csv`,
`feature_screen_results.csv`.

## Reproducing
Training / inference scripts live in `analysis/images/cnn_lstm/` (committed):
`train_base_model_kfold.py` (`--n-repeats`, `--bbox-crop`, `--brighten`, `--save-models`),
`train_temporal_lstm_kfold.py`, `compute_spatial_occlusion.py`, `compute_focus.py`
(`--kfold`, `--bbox-crop`, `--cam-target`), `make_gradcam_rotation_check.py`,
`size_baseline_kfold.py`, `compute_bbox_sharpness.py`.

Run outputs are on the cluster:
- models / out-of-fold predictions: `/net/projects2/promega/project_data/model_tests/lstm_runs/idor_main/`
  (`base_effnet_kfold4x10_{std,strong,bboxstd,brightstd,brightbbox}`, `temporal_lstm_kfold4x10`)
- per-organoid CSVs (focus, occlusion, size baseline, mask/translucent/feature tables):
  `/net/projects2/promega/project_data/amanda_test/model_plots/`

`scripts/` holds the figure and ad-hoc analysis scripts. Plotting scripts take a local
data folder that mirrors those cluster paths (`<data>/amanda_test/model_plots/...`,
`<data>/model_tests/lstm_runs/idor_main/...`) and an output folder, e.g.
`python scripts/make_plots.py <data> figures`. Scripts that read images
(`feature_extract.py`, `translucent_*.py`, `*_montage.py`, `texture_bright.py`,
`explain_conditions.py`, `borderline.py`) run from the repo root on the cluster with
`core_env`; `borderline.py`/`translucent_all.py` expect the relaxed cohort from
`scripts/splits/make_splits.py configs/idor.json --min-majority-votes 3 --output-suffix _minvotes3_tmp`.

## Open ideas (V2)
- **Consistent brightness/contrast across images.** Raw images differ in exposure and
  have strong vignetting; brightening was per organoid. Try flat-field / illumination
  correction and a shared intensity normalisation (same scale for every image) before
  texture features or CNN training.
- **Raw 16-bit Z-stacks** (832×1128, 5–6 planes) instead of 575×575 8-bit clipped images,
  e.g. best-focus plane or a focus stack — edge "layers" raters judge may only show there.
- Ask raters what "layers" / "wrinkles" look like; measure "organised outlined
  compartments" (figure 22 impression) if confirmed.
- Validate the cyst threshold on an unseen plate; leave-one-plate-out for the CNNs.
- Growth + metabolites (Liya's models) to break the ~0.85 ceiling.
