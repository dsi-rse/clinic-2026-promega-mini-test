# Paper figures: image models (single-day CNNs, trajectory LSTM) and interpretability

`make_paper_figs.py` builds all figures on the cluster (see docstring). Vote counts are
regular-image votes only (inverted re-show votes never used); labels are the official ≥ 4/5 labels.
Every panel title states its finding.

**Fig 1 — `fig1_models_and_size`: the models work, and they work because of size**
- a: what each model gets, for the same 3 organoids used in Fig 2 (accepted / stalled reject / overgrown reject): size only (mask area), CNN whole image, CNN cropped, CNN cropped + brightened, LSTM image series
- b: day-30 balanced accuracy and AUC of the image models (mean ± SD over 10 CV repeats); best = CNN cropped + brightened, AUC 0.88, balanced accuracy 0.78
- c: confusion matrix of the best model at day 30 (94% of Acceptable and 63% of Not Acceptable correct)
- d: every day — image models only work from ~day 20; size alone is as good or better
- e: day-30 size density — Not Acceptable organoids are too small or too big
- f: best image model vs size alone, same organoids and folds — size with two cutoffs: AUC 0.87, balanced accuracy 0.83
- g: LSTM vs size + growth rate for each window — the image series adds nothing

**Fig 2 — `fig2_what_models_look_at`: what the models look at**
- a: Grad-CAM on the same 3 organoids (whole-image CNN, strong-augmentation CNN)
- b: Grad-CAM heat on organoid vs background by day — the CNNs only look at the organoid from day ~17, when they start to work
- c: no texture needed — CNNs retrained on the mask alone vs on the image (day 30), and the organoid erased from the test image (→ chance)
- d: resizing the same organoid moves the prediction (bigger → more acceptable from day 17)
- e: the 7 of 33 rejects no method catches, next to same-plate accepted organoids of the same size

**Fig S — `figS_supplement`**
- a: growth curves — rejects stall or overgrow; groups split at days 15–17
- b: every CNN and mask-only CNN, every day, against size
- c: borderline-vote robustness — training with the 50 borderline organoids (majority 3/5 label) gives the same AUC on the clear organoids
- d: % of rejects caught by each method; 7 of 33 by none
- e: shape of the missed rejects (less round, but shape adds no CV gain)
- f: every model at day 30 (AUC)
- g: test-time edits of the trained CNNs (mask only, inside texture removed, organoid removed)
- h: LSTM day blanking — no single day changes AUC by more than 0.02

Colours (one meaning each): Acceptable teal, Not Acceptable burnt orange (labels only); size models warm grays;
CNNs olive → avocado, best CNN mustard; LSTM plum; mask-only CNNs hatched sand.

Tables: `day30_metrics.csv` (Fig 1 b, f), `rejects_by_method.csv` (Fig 2 e, Fig S d), `borderline_vs_clear.csv` (Fig S c, from `borderline_eval.py`).

Definitions for captions
- **Area**: mask pixel count × 81 µm² (every processed image is 9 µm/px, `ar_target_um_per_px` in all_data.json), in mm².
- **Balanced accuracy**: mean of the per-class recalls at P(Acceptable) ≥ 0.5; the size models use class-balanced logistic regression.
- **Size, one cutoff**: logistic regression on log mask area + plate (bigger = better).
- **Size, two cutoffs**: the same on a spline of log area, so too small and too big can both be called bad.
- **Size + growth rate**: last-day size + growth slope over the window (two cutoffs) — the LSTM's baseline.
- **Mask only, test time (Fig S g)**: organoid filled flat gray on the trained V1 models: size + outline, no texture.
- **Mask-only CNN (Fig 2 c, Fig S b)**: a CNN *trained* on the mask (organoid black on the 178-gray background, `--mask-only`), same 4-fold × 10-repeat CV and folds as the image CNNs, days 17–30. Whole frame = size + outline; cropped = outline only (the crop removes absolute size).
- **Borderline run (Fig S c)**: cohort `idor_main_minvotes3` = the 198 clear organoids (same labels) + 50 borderline (26 at 3/5 → Acceptable, 24 at 2/5 → Not Acceptable), regular-image votes only; brightened-bbox CNN, all days, 4 × 10 CV, scored on the clear organoids.
- **Missed by every method** (`rejects_by_method.csv`): mean OOF P(Acceptable) ≥ 0.5 for the growth-rate
  rule, all four CNNs (latest of day 24/28/30 with an image) and the mean-pooled LSTM (where scored).
  7 of 33: BA1_96_1_A8, BA2_96_1_D8, F3, G4, G5, BA2_96_2_G7, H2.

Sources and caveats
- V2 cohort (`data/cohorts/idor_main/full`, 155–198 organoids per day): all performance numbers, LSTM blanking, shape, montage. Same organoids and folds for size and image models.
- Fig 2 a, b, d, the 'organoid erased' bar in Fig 2 c and Fig S g use the V1 models (140-organoid complete-series cohort; whole-image V1 inputs were stretched 1.33×, reruns matched within ±0.04 AUC).
- All LSTM results use the mean-pooled readout (averages every frame). The last-frame readout scores higher at day 30 (0.82) but is effectively a last-day model (blanking day 30 costs 0.15 AUC), so it is not shown.
- LSTM blanking uses one CV repeat; that repeat's mean-pooled model is weaker than the 10-repeat average (AUC 0.61 vs 0.74 at days 3–30).
- Labels: the pre-30 April 2026 all_data.json pooled INV votes with the originals (BA2 96_1 F3, BA2 96_2 D4, BA4 96_1 E3 came out Uncertain); the current file and these figures use original-image votes only (all three Not Acceptable). P. Perdigão's recount found the same issue (2 organoids Uncertain → Bad; which two is still to be confirmed).
