import os
import sys
import numpy as np
import torch
import yaml
from box import Box

sys.path.append('./src/')
from data import get_test_file_lists, NumpyTimeSeriesDataset

torch.set_default_dtype(torch.float32)

DEVICE = 'cuda:0'

# Use one reference trained config to get the evaluation dataset list and the
# scaling factors for the corresponding training set.
# Run this for each region (nearwall, channel-center) you want Clark numbers in.
REFERENCE_RUNS = {
    'nearwall': 'outputs/CNNaug_params/26_05_21_12:26:23_well_pony_nearwall_boxfilter_4x_sgs',
    'middle':   'outputs/26_06_05_13:59:35_grown_jay_middle_boxfilter_4x_sgs',
}


def load_config(run_dir):
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        return Box(yaml.safe_load(f))


def clark_prediction_pointwise(inputs_9):
    """Clark on standardized pointwise inputs (B, 9).
    Returns (B, 5) deviatoric prediction in the same standardized scale.
    Caller must rescale by input_mean_magnitude**2 / target_mean_magnitude
    to compare against standardized targets, but since Park-Choi is uncentered
    and scale-invariant, the comparison is unaffected by overall scale on
    each side."""
    B = inputs_9.shape[0]
    G = inputs_9.view(B, 3, 3)
    tau = torch.einsum('bik,bjk->bij', G, G) / 12.0
    trace = (tau[:, 0, 0] + tau[:, 1, 1] + tau[:, 2, 2]) / 3.0
    tau[:, 0, 0] -= trace
    tau[:, 1, 1] -= trace
    tau[:, 2, 2] -= trace
    out = torch.stack([tau[:, 0, 0], tau[:, 1, 1],
                       tau[:, 0, 1], tau[:, 0, 2], tau[:, 1, 2]], dim=-1)
    return out


def compute_clark_correlations(reference_run_dir):
    cfg = load_config(reference_run_dir)
    print(f'\n=== Reference: {cfg.barcode} ===')

    # Load the training dataset to get the scaling factors used during training
    from data import get_tv_file_lists
    train_input_files, train_target_files, _, _ = get_tv_file_lists(cfg.dataset)
    train_dataset = NumpyTimeSeriesDataset(
        input_file_list=train_input_files,
        target_file_list=train_target_files,
        scaling_factors=None,
        augmentation_group=cfg.train_aug_group,
        pointwise=True,  # easier to compute Clark pointwise
        precompute_eigenframe_inputs=False,
        device=DEVICE,
    )
    scaling_factors = train_dataset.get_scaling_factors()

    for ds_name, params in cfg.dataset.eval_datasets.items():
        input_files, target_files = get_test_file_lists(params)
        dataset = NumpyTimeSeriesDataset(
            input_file_list=input_files,
            target_file_list=target_files,
            scaling_factors=scaling_factors,
            augmentation_group=None,
            pointwise=True,
            precompute_eigenframe_inputs=False,
            device=DEVICE,
        )
        inputs, targets = dataset.get_all_data()

        # Inputs from the dataset are standardized: hat_G / input_mean_magnitude.
        # Targets are standardized: hat_tau / target_mean_magnitude.
        # Clark in dimensionless form takes hat_G and produces hat_tau.
        # Park-Choi correlation is scale-invariant, so we can compute Clark on
        # the standardized inputs and correlate against the standardized targets.
        # The overall scale factors cancel in the ratio.
        with torch.inference_mode():
            clark_pred = clark_prediction_pointwise(inputs)

        dot_sum = (clark_pred * targets).sum().item()
        pred_sq_sum = (clark_pred ** 2).sum().item()
        true_sq_sum = (targets ** 2).sum().item()
        correlation = dot_sum / (np.sqrt(pred_sq_sum) * np.sqrt(true_sq_sum))

        print(f'{ds_name}: correlation = {correlation:.3f}')


if __name__ == '__main__':
    for region, run_dir in REFERENCE_RUNS.items():
        compute_clark_correlations(run_dir)