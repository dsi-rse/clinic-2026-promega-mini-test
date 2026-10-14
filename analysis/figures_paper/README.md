# Paper figures: image models (single-day CNNs, trajectory LSTM) and interpretability

`make_paper_figs.py` builds both figures on the cluster (see docstring). All vote counts are
regular-image votes only; inverted re-show votes are never used. Labels are the official ≥ 4/5 labels.

**Fig A — `figA_image_models_vs_size`**: image models reach AUC ~0.82–0.88 at day 30 but never beat two-way size.
- a: exact model inputs (dataset code used in training) for the same 3 organoids as the Grad-CAM panel (B-d), day 30 + LSTM sequence
- b: growth curves: accepted vs stalled vs overgrown rejects (reject split by latest late-day size vs same-plate accepted median); groups split days 15→17
- c: % Not Acceptable by day-30 size fifth (61 / 0 / 10 / 3 / 23%), why size must be modelled two-way
- d: single-day CNN AUC by day vs bigger-is-better and two-way size (V2, same organoids/folds)
- e: LSTM (last / mean readout) vs size + growth baselines
- f: same-size-pair AUC (CNNs near chance)
- g: gain over two-way size + plate, every model and day (≤ +0.03)
- h: brightened-bbox CNN prediction vs day-30 size (learned "too small or too big = bad")

"Two-way size" = logistic regression on a spline of log area (+ plate), so both too-small and
too-big can be bad; "bigger = better" is linear in log area. "Mask only" (B-b) = the organoid
mask filled flat gray: size + outline, no texture.

**Fig B — `figB_what_models_use`**: CNNs read size/silhouette, the LSTM reads its last frame, 21% of rejects are caught by nothing.
- a–d: V1 models (140-organoid complete-series cohort): rescale intervention, occlusion, Grad-CAM heat density, Grad-CAM examples
- e–f: LSTM day dropout (V2, repeat-1 checkpoints)
- g: % of rejects caught per method; h: shape z-scores; i: the missed rejects vs same-size accepted

"Missed by every method" (fixed rule, `rejects_by_method.csv`): a reject whose mean out-of-fold
P(Acceptable) over 10 repeats is ≥ 0.5 for the growth-band rule, all four CNNs (latest of day
24/28/30 with an image) and the last-readout LSTM (where scored). 7 of 33: BA1_96_1_A8,
BA2_96_1_D8, F3, G4, G5, BA2_96_2_G7, H2.

Caveats: Fig B a–d come from the V1 models (non-bbox V1 inputs were stretched 1.33×; reruns
matched within ±0.04 AUC). B-a strong-aug "shrunk" at day 28 has a −0.48 median (real data,
one day only). Day-dropout uses one repeat.
