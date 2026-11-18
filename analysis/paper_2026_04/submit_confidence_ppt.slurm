#!/bin/bash
#SBATCH --job-name=conf-ppt
#SBATCH --partition=general
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --output=logs/confidence_ppt_%j.out
#SBATCH --error=logs/confidence_ppt_%j.err

cd /net/spaces/clinic/CLS/lding/gitcode/2025-promega-mini-test
mkdir -p logs figures
export PYTHONPATH=$(pwd)
/net/projects/CLS/lding/conda_env/core_env/bin/python -u -m analysis.paper_2026_04.make_confidence_ppt
