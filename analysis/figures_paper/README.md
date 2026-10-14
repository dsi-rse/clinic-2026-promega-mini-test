# Paper figures: image models (single-day CNNs, trajectory LSTM) and interpretability

`make_paper_figs.py` builds all three on the cluster (see docstring). Vote counts are
regular-image votes only (inverted re-show votes never used); labels are the official ≥ 4/5 labels.
Every panel title states its finding.

**Fig A — `figA_models_and_performance`: what we ran and how well it works**
- a: what each model gets, for the same 3 organoids used in Fig B (accepted / stalled reject / overgrown reject): size only (mask area), CNN whole image (also used with strong augmentation), CNN cropped, CNN cropped + brightened, LSTM image sequence
- b: day-30 ladder — size models (one number per organoid) vs image models; best image model 0.88 ties size with two cutoffs 0.87
- c: why two cutoffs — day-30 size density, Acceptable vs Not Acceptable (Not Acceptable is both smaller and larger)
- d: growth curves — accepted vs stalled vs overgrown rejects; groups split at days 15–17
- e: every day — image models never beat size; before day 20 they are worse
- f: LSTM (mean-pooled over all frames) vs size + growth rate for each window — the LSTM adds nothing

**Fig B — `figB_what_models_look_at`: what the models see**
- a: Grad-CAM on the same 3 organoids (whole-image CNN, strong-augmentation CNN)
- b: Grad-CAM heat density, organoid vs background, by day and model
- c: mask only (texture removed) keeps the AUC → the CNN reads size/outline
- d: resizing the same organoid moves the prediction (bigger → more acceptable from day 17)
- e: LSTM (mean-pooled) day blanking — no single day changes AUC by more than 0.02

**Fig C — `figC_what_no_model_catches`**
- a: best CNN's day-30 prediction vs size (small → bad; very big → borderline)
- b: % of rejects caught by each method; 7 of 33 (21%) caught by none
- c: shape of those 7 vs others (less round, but shape adds no CV gain)
- d: the 7 vs same-plate, same-size accepted organoids

Definitions for captions
- **Area**: mask pixel count × 81 µm² (every processed image is 9 µm/px, `ar_target_um_per_px` in all_data.json), in mm².
- **Size, one cutoff**: logistic regression on log mask area + plate (bigger = better).
- **Size, two cutoffs**: the same on a spline of log area, so too small and too big can both be called bad.
- **Size + growth rate**: last-day size + growth slope over the window (two cutoffs) — the LSTM's baseline.
- **Mask only**: organoid mask filled flat gray at test time (V1 models): size + outline, no texture.
- **Missed by every method** (`rejects_by_method.csv`): mean OOF P(Acceptable) ≥ 0.5 for the growth-rate
  rule, all four CNNs (latest of day 24/28/30 with an image) and the mean-pooled LSTM (where scored).
  7 of 33: BA1_96_1_A8, BA2_96_1_D8, F3, G4, G5, BA2_96_2_G7, H2.

Sources and caveats
- V2 cohort (`data/cohorts/idor_main/full`, 155–198 organoids per day): all performance numbers, LSTM blanking, shape, montage. Same organoids and folds for size and image models.
- Fig B a–d use the V1 models (140-organoid complete-series cohort; whole-image V1 inputs were stretched 1.33×, reruns matched within ±0.04 AUC).
- All LSTM results use the mean-pooled readout (averages every frame). The last-frame readout scores higher at day 30 (0.82) but is effectively a last-day model (blanking day 30 costs 0.15 AUC), so it is not shown.
- LSTM blanking uses one CV repeat; that repeat's mean-pooled model is weaker than the 10-repeat average (AUC 0.61 vs 0.74 at days 3–30). AUCs are means of 4-fold × 10-repeat CV (no error bars in A-b).
