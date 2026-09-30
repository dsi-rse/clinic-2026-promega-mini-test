#!/bin/bash
#SBATCH --job-name=met-lgbm-lr-134
#SBATCH --partition=general
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --output=logs/met_lgbm_logreg_kfold_134_%j.out
#SBATCH --error=logs/met_lgbm_logreg_kfold_134_%j.err

# Rerun with drop_stitched=True (n=134, excluding 6 stitched organoids).
# Output: analysis_output/images/met_morph_lgbm_logreg_kfold_134.json

cd /net/spaces/clinic/CLS/lding/gitcode/2025-promega-mini-test
mkdir -p logs

conda run --no-capture-output -p /net/projects/CLS/lding/conda_env/core_env python3 \
    -m analysis.paper_2026_04.met_lgbm_logreg_kfold \
    --n-repeats 10 \
    --n-folds 4
