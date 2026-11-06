#!/bin/bash
#SBATCH --job-name=ryley_train
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Defaults to mit_normal_gpu. To use another partition:
#   sbatch --partition=pi_abodner --time=72:00:00 sbatch_sweep_gpu.sh <config> <subdir>

if [ $# -lt 2 ]; then
    echo "Usage: sbatch sbatch_sweep_gpu.sh <config_file> <subdirectory>"
    exit 1
fi

CONFIG_FILE=$1
SUBDIR=$2

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Config file: $CONFIG_FILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

uv run src/train.py "$CONFIG_FILE" --device cuda:0 --subdirectory "$SUBDIR"

echo "Job finished on $(date)"
