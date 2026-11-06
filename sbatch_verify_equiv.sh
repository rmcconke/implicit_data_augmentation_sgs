#!/bin/bash
#SBATCH --job-name=sgs_verify_equiv
#SBATCH --partition=pi_abodner
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
uv run src/eval_model.py "outputs/pointwiseEigenFrame_params/26_09_25_19:43:20_rested_trout_nearwall_boxfilter_4x_sgs/" --device cuda:0
