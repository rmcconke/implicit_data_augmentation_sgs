#!/bin/bash
#SBATCH --job-name=ryley_testing
#SBATCH --partition=pi_abodner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Check if experiment folder argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No experiment folder provided"
    echo "Usage: sbatch run_tests.sh <experiment_folder>"
    exit 1
fi

EXPERIMENT_FOLDER=$1

# Define excluded animal descriptors (adjective + animal)
EXCLUDED_ANIMALS=("pet_pika" "solid_hermit" "super_spider" "polite_hog" "hot_piglet" "bold_macaw" "normal_sloth" "social_vervet")

# Print some job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment folder: $EXPERIMENT_FOLDER"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo ""

# Get list of subdirectories
SUBDIRS=("$EXPERIMENT_FOLDER"/*/)
NUM_SUBDIRS=${#SUBDIRS[@]}

if [ $NUM_SUBDIRS -eq 0 ]; then
    echo "No subdirectories found in $EXPERIMENT_FOLDER"
    exit 1
fi

echo "Found $NUM_SUBDIRS subdirectories"
echo ""

# Loop over all subdirectories and run evaluation
PROCESSED_COUNT=0
SKIPPED_COUNT=0

for subdir in "${SUBDIRS[@]}"; do
    # Check if any excluded animal descriptor is in the path
    SHOULD_SKIP=false
    for animal in "${EXCLUDED_ANIMALS[@]}"; do
        if [[ "$subdir" == *"$animal"* ]]; then
            SHOULD_SKIP=true
            echo "Skipping (excluded): $subdir"
            ((SKIPPED_COUNT++))
            break
        fi
    done
    
    # Process if not excluded
    if [ "$SHOULD_SKIP" = false ]; then
        echo "Processing: $subdir"
        uv run src/test_model.py "$subdir" --device cuda:0
        echo "Completed: $subdir"
        echo ""
        ((PROCESSED_COUNT++))
    fi
done

echo "All evaluations complete."
echo "Processed: $PROCESSED_COUNT subdirectories"
echo "Skipped: $SKIPPED_COUNT subdirectories"
echo "Job finished on $(date)"