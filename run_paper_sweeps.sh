#!/bin/bash
# Retraining campaign with the rotation-invariant loss. Every sweep is independent, so these
# all queue at once.

# Prints instead of submitting unless --submit is given.
if [ "$1" == "--submit" ]; then
    SB="sbatch sbatch_sweep_gpu.sh"
    TR="sbatch --job-name=ryley_sweep sbatch_training_gpu.sh"
else
    echo "DRY RUN - pass --submit to actually queue these jobs"
    SB="echo   sbatch sbatch_sweep_gpu.sh"
    TR="echo   sbatch sbatch_training_gpu.sh"
fi

# Parameter and data-size sweeps
$SB configs/sweeps/ESCNN_params.yaml               ESCNN_params
$SB configs/sweeps/CNNaug_params.yaml              CNNaug_params
$SB configs/sweeps/pointwiseAug_params.yaml        pointwiseAug_params
$SB configs/sweeps/pointwiseEigenFrame_params.yaml pointwiseEigenFrame_params

$SB configs/sweeps/ESCNN_data.yaml                 ESCNN_data
$SB configs/sweeps/CNNaug_data.yaml                CNNaug_data
$SB configs/sweeps/CNNnoaug_data.yaml              CNNnoaug_data
$SB configs/sweeps/pointwiseAug_data.yaml          pointwiseAug_data
$SB configs/sweeps/pointwiseNoAug_data.yaml        pointwiseNoAug_data
$SB configs/sweeps/pointwiseEigenFrame_data.yaml   pointwiseEigenFrame_data

$SB configs/sweeps/CNNaug_data_middle.yaml         CNNaug_data_middle
$SB configs/sweeps/CNNnoaug_data_middle.yaml       CNNnoaug_data_middle
$SB configs/sweeps/pointwiseAug_data_middle.yaml   pointwiseAug_data_middle
$SB configs/sweeps/pointwiseNoAug_data_middle.yaml pointwiseNoAug_data_middle

# Table 3 channel-centre rows. The near-wall rows fall out of the param sweeps
# above; these do not. Data-size sweeps stay near-wall only, as in Figs 5(c)/6(c).
for f in configs/middle/*.yaml; do
    $TR "$f" "table3_middle"
done

# Batch norm ablation (CNN without batch norm)
$SB configs/sweeps/CNNaug_params_nobn.yaml         CNNaug_params_nobn

# Additional seeds for the Table 3 models
for f in configs/sweeps/seeds/*.yaml; do
    $SB "$f" "seeds_$(basename "$f" .yaml)"
done

# Filter ratios (cross-filter-width test). Needs the fs2/fs8 datasets first:
#   uv run dataset_creation/sgs_filter_ratios.py
# for f in configs/sweeps/ratios/*.yaml; do
#     $SB "$f" "ratio_$(basename "$f" .yaml)"
# done

# Inference times in Figs. 5(b) and 6(b) were measured on a single NVIDIA H100.
