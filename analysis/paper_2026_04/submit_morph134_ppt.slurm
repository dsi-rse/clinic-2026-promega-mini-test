#!/bin/bash
#SBATCH --job-name=morph134-ppt
#SBATCH --partition=general
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:10:00
#SBATCH --output=logs/morph134_ppt_%j.out
#SBATCH --error=logs/morph134_ppt_%j.err

cd /net/spaces/clinic/CLS/lding/gitcode/2025-promega-mini-test
mkdir -p logs figures

conda run --no-capture-output -p /net/projects/CLS/lding/conda_env/core_env python3 \
    -m analysis.paper_2026_04.make_morph134_ppt
