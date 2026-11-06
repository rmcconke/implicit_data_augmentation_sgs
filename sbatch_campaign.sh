#!/bin/bash
#SBATCH --job-name=sgs_campaign
#SBATCH --partition=pi_abodner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# One array task per line of campaign_manifest.txt: "<sweep|train> <config> <subdir>".
# Throttle with --array=1-N%K at submit time so the group keeps GPUs free.

MANIFEST=${MANIFEST:-campaign_manifest.txt}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
KIND=$(echo "$LINE" | awk '{print $1}')
CONFIG=$(echo "$LINE" | awk '{print $2}')
SUBDIR=$(echo "$LINE" | awk '{print $3}')

echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Array job: $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID"
echo "Kind: $KIND  Config: $CONFIG  Subdir: $SUBDIR"
nvidia-smi

# Figs 5(b)/6(b) report inference time on an H100. Refuse to produce timings on
# anything else rather than silently recording the wrong hardware.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"
if [[ "$GPU_NAME" != *"H100"* ]]; then
    echo "ERROR: expected an H100, got '$GPU_NAME'. Timings would be invalid." >&2
    exit 1
fi

if [ "$KIND" == "train" ]; then
    uv run src/train.py "$CONFIG" --device cuda:0 --subdirectory "$SUBDIR"
else
    uv run src/sweep.py "$CONFIG" --device cuda:0 --subdirectory "$SUBDIR"
fi

echo "Job finished on $(date)"
