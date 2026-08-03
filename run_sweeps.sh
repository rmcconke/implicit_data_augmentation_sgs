#!/bin/bash

# Data sweeps
# sbatch sbatch_sweep_config.sh configs/sweeps/CNNaug_data.yaml CNNaug_data
# sbatch sbatch_sweep_config.sh configs/sweeps/CNNnoaug_data.yaml CNNnoaug_data
# sbatch sbatch_sweep_config.sh configs/sweeps/pointwiseNoAug_data.yaml pointwiseNoAug_data
# sbatch sbatch_sweep_config.sh configs/sweeps/pointwiseAug_data.yaml pointwiseAug_data
# sbatch sbatch_sweep_config.sh configs/sweeps/ESCNN_data.yaml ESCNN_data
# sbatch sbatch_sweep_config.sh configs/sweeps/pointwiseEigenFrame_data.yaml pointwiseEigenFrame_data

sbatch sbatch_sweep_config.sh configs/sweeps/CNNaug_data_middle.yaml CNNaug_data_middle
sbatch sbatch_sweep_config.sh configs/sweeps/CNNnoaug_data_middle.yaml CNNnoaug_data_middle
sbatch sbatch_sweep_config.sh configs/sweeps/pointwiseNoAug_data_middle.yaml pointwiseNoAug_data_middle
sbatch sbatch_sweep_config.sh configs/sweeps/pointwiseAug_data_middle.yaml pointwiseAug_data_middle
# sbatch sbatch_sweep_config.sh configs/sweeps/ESCNN_data.yaml ESCNN_data
# sbatch sbatch_sweep_config.sh configs/sweeps/pointwiseEigenFrame_data.yaml pointwiseEigenFrame_data

# Param sweeps
#sbatch sbatch_sweep_config.sh configs/sweeps/CNNaug_params.yaml CNNaug_params
#sbatch sbatch_sweep_config.sh configs/sweeps/pointwiseAug_params.yaml pointwiseAug_params
#sbatch sbatch_sweep_config.sh configs/sweeps/ESCNN_params.yaml ESCNN_params
#sbatch sbatch_sweep_config.sh configs/sweeps/pointwiseEigenFrame_params.yaml pointwiseEigenFrame_params
