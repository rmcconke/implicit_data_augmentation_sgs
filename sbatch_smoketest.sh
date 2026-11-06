#!/bin/bash
#SBATCH --job-name=sgs_smoke
#SBATCH --partition=pi_abodner
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=0:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
uv run pytest tests/ -q
