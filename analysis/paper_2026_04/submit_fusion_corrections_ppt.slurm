#!/bin/bash
#SBATCH --job-name=fusion-ppt
#SBATCH --partition=general
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:20:00
#SBATCH --output=logs/fusion_ppt_%j.out
#SBATCH --error=logs/fusion_ppt_%j.err

cd /net/spaces/clinic/CLS/lding/gitcode/2025-promega-mini-test
mkdir -p logs

conda run --no-capture-output -p /net/projects/CLS/lding/conda_env/core_env python3 -u -X faulthandler \
    -m analysis.paper_2026_04.make_fusion_corrections_ppt
