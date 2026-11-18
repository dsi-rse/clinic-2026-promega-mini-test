#!/bin/bash
#SBATCH --job-name=conf-vs-survey
#SBATCH --partition=general
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=0:15:00
#SBATCH --output=logs/confidence_vs_survey_%j.out
#SBATCH --error=logs/confidence_vs_survey_%j.err

cd /net/spaces/clinic/CLS/lding/gitcode/2025-promega-mini-test
mkdir -p logs

export PYTHONPATH=$(pwd)
/net/projects/CLS/lding/conda_env/core_env/bin/python -u -m analysis.paper_2026_04.confidence_vs_survey
