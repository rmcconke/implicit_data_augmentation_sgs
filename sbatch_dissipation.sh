#!/bin/bash
#SBATCH --job-name=sgs_dissip
#SBATCH --partition=pi_abodner
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
while read -r D; do
    echo "===== $D"
    uv run src/analyze_dissipation.py "$D" --device cuda:0 || echo "FAILED: $D"
done < dissipation_targets.txt
