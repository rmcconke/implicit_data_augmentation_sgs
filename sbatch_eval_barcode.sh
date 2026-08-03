#!/bin/bash
#SBATCH --job-name=ryley_eval
#SBATCH --partition=pi_abodner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No directory provided"
    echo "Usage: sbatch run_spectra.sh <directory>"
    exit 1
fi

DIRECTORY=$1

# Print some job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Directory: $DIRECTORY"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo ""

# Run evaluation with spectra computation
echo "Processing: $DIRECTORY"
uv run src/eval_model.py "$DIRECTORY"

echo "Completed: $DIRECTORY"
echo "Job finished on $(date)"