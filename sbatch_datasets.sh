#!/bin/bash
#SBATCH --job-name=sgs_datasets
#SBATCH --partition=pi_abodner
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd dataset_creation
echo "=== box-mean normalization (control) ==="
uv run python sgs_boxmean_norm.py
echo "=== filter ratios fs=2, fs=8 (cross-filter-width test) ==="
uv run python sgs_filter_ratios.py
echo "done $(date)"
