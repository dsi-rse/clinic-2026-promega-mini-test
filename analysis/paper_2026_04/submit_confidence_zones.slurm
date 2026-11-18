#!/bin/bash
#SBATCH --job-name=conf-zones
#SBATCH --partition=general
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/confidence_zones_%A_%a.out
#SBATCH --error=logs/confidence_zones_%A_%a.err

# One day per array task:  sbatch analysis/paper_2026_04/submit_confidence_zones.slurm
DAYS=(Dy30 Dy28 Dy24)
cd /net/spaces/clinic/CLS/lding/gitcode/2025-promega-mini-test
mkdir -p logs

export PYTHONPATH=$(pwd)
/net/projects/CLS/lding/conda_env/core_env/bin/python -u -m analysis.paper_2026_04.confidence_zones --days ${DAYS[$SLURM_ARRAY_TASK_ID]}
