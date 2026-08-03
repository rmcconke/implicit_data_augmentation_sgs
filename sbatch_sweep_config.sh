#!/bin/bash
#SBATCH --job-name=ryley_sweepy_sweepy
#SBATCH --partition=pi_abodner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j_%Y_%m_%d_%H:%M:%S.out
#SBATCH --error=logs/%x_%j_%Y_%m_%d_%H:%M:%S.err

# Check if config file argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No config file provided"
    echo "Usage: sbatch sbatch_sweep_config.sh <config_file>"
    exit 1
fi

CONFIG_FILE=$1
SUBDIR=$2

# Print some job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Config file: $CONFIG_FILE"

# Run your Python script with uv
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
uv run src/sweep.py "$CONFIG_FILE" --device cuda:0 --subdirectory "$SUBDIR"

echo "Job finished on $(date)"