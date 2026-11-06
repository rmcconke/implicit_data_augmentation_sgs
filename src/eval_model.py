from re import T
import torch
import numpy as np
import os
import torch.optim as optim
torch.set_default_dtype(torch.float32)

from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import new_experiment, juicy_word, plot_losses, plot_equivariance_errors
from data import get_tv_file_lists, get_test_file_lists, NumpyTimeSeriesDataset
from tqdm import tqdm
from models import get_model
from eval import (
    eval_test_set,
    plot_spectra,
    plot_spectrum,
    visualize_predictions_3D,
    mean_octahedral_equivariance_error,
    relative_octahedral_equivariance_error,
    plot_equivariance,
    eval_equivariance
)
import argparse
import global_config as global_config
from transformations import apply_octahedral_augmentation, apply_so3_augmentation
from deviatoric import to_orthonormal, invariant_criterion
import yaml
from box import Box
import time

def _load_config_label(run_dir):
    setup_path = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(setup_path):
        return None
    with open(setup_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Box(cfg)


def eval_model_after_training(experiment_dir, device, compute_spectra=False):
    experiment_dir = experiment_dir
    config = _load_config_label(experiment_dir)
    print(f"WELCOME TO THE TEST OF EXPERIMENT: {config.barcode}")

    torch.set_default_dtype(torch.float32)

    # Load the checkpoint
    checkpoint = torch.load(os.path.join(experiment_dir, f'{config.barcode}_checkpoint.pt'), map_location=device)
    best_state = checkpoint["model"]

    # Initialize the model
    model = get_model(config.model).to(device)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    if config.loss_type == 'L1':
        criterion = F.l1_loss
    elif config.loss_type == 'L2':
        criterion = F.mse_loss
    if config.task_type == 'sgs':
        criterion = invariant_criterion(criterion)

    print(f"default dtype: {torch.get_default_dtype()}")
    train_input_files, train_target_files, _, _ = get_tv_file_lists(config.dataset)

    train_dataset = NumpyTimeSeriesDataset(
        input_file_list=train_input_files,
        target_file_list=train_target_files,
        scaling_factors=None,
        augmentation_group=config.train_aug_group,
        pointwise=config.pointwise,
        precompute_eigenframe_inputs=False,
        device=device,
    )

    eval_datasets = {}
    for name, params in config.dataset.eval_datasets.items():
        input_files, target_files = get_test_file_lists(params)
        eval_datasets[name] = NumpyTimeSeriesDataset(
            input_file_list=input_files,
            target_file_list=target_files,
            scaling_factors=train_dataset.get_scaling_factors(),
            augmentation_group=config.test_aug_group,
            pointwise=config.pointwise,
            precompute_eigenframe_inputs=False,
            device=device,
        )

    with torch.inference_mode():
        eval_results = {name: {} for name in eval_datasets.keys()}
        for name, dataset in eval_datasets.items():
            inputs, targets = dataset.get_all_data()
            n_samples = inputs.shape[0]

            total_loss = 0.0
            dot_sum = 0.0
            true_sq_sum = 0.0
            pred_sq_sum = 0.0
            for i in range(0, n_samples, config.eval_batch_size):
                batch_inputs = inputs[i:i + config.eval_batch_size]
                batch_targets = targets[i:i + config.eval_batch_size]
                pred = model(batch_inputs)
                batch_targets_unstd = batch_targets * train_dataset.get_scaling_factors()['target_mean_magnitude']
                pred_unstd = pred * train_dataset.get_scaling_factors()['target_mean_magnitude']
                total_loss += criterion(pred_unstd, batch_targets_unstd).item() * batch_inputs.shape[0]

                if config.task_type == 'sgs':
                    pred_inv, targets_inv = to_orthonormal(pred), to_orthonormal(batch_targets)
                else:
                    pred_inv, targets_inv = pred, batch_targets
                dot_sum     += (pred_inv * targets_inv).sum().item()
                true_sq_sum += (targets_inv ** 2).sum().item()
                pred_sq_sum += (pred_inv ** 2).sum().item()

            target_mean_mag = train_dataset.get_scaling_factors()['target_mean_magnitude']
            targets_unstd = targets * target_mean_mag
            if config.task_type == 'sgs':
                targets_unstd = to_orthonormal(targets_unstd)
            eval_results[name]['target_variance'] = targets_unstd.var().item()
            eval_results[name]['target_second_moment'] = (targets_unstd ** 2).mean().item()
            eval_results[name]['loss'] = total_loss / n_samples
            eval_results[name]['correlation'] = dot_sum / (np.sqrt(true_sq_sum) * np.sqrt(pred_sq_sum))
            print(f"Test {name} Loss (unstandardized, non-dimensional): {eval_results[name]['loss']}")
            print(f"Test {name} Correlation: {eval_results[name]['correlation']}")

            # Equivariance error: averaged over octahedral group, batch, components, space
            equiv_err_sum = 0.0
            equiv_rel_err_sum = 0.0
            equiv_err_count = 0
            deviatoric = config.task_type == 'sgs'
            # The group expands the batch 24x. For the pointwise models that is a
            # large batched eigendecomposition, which cusolver fails to allocate
            # for while the evaluation datasets are resident on the GPU, so every
            # model call (f(x), f(g.x) and the normalizing prediction) is chunked.
            # A failure raises: skipping a batch would give a partial average.
            # Large nonlocal models can set equiv_batch_size (boxes) in the config.
            equiv_batch = config.get("equiv_batch_size", 500 if config.pointwise else None)
            for i in range(0, n_samples, config.eval_batch_size):
                batch_inputs = inputs[i:i + config.eval_batch_size]
                try:
                    err = mean_octahedral_equivariance_error(model, batch_inputs, batch_size=equiv_batch, deviatoric=deviatoric)
                    rel_err = (relative_octahedral_equivariance_error(model, batch_inputs, batch_size=equiv_batch, error=err)
                               if deviatoric else err)
                except Exception as e:
                    raise RuntimeError(f"Equivariance error failed on {name}, samples {i}:{i + batch_inputs.shape[0]}") from e
                equiv_err_sum += err.item() * batch_inputs.shape[0]
                equiv_rel_err_sum += rel_err.item() * batch_inputs.shape[0]
                equiv_err_count += batch_inputs.shape[0]
                print(f"Equivariance error: {err.item()}")
                print(f"Batch inputs: {batch_inputs.shape}")
                print(f"equivariance error count: {equiv_err_count}")
                print(f"equivariance error sum: {equiv_err_sum}")
            assert equiv_err_count == n_samples, (equiv_err_count, n_samples)
            eval_results[name]['equiv_error'] = equiv_err_sum / equiv_err_count
            eval_results[name]['equiv_error_rel'] = equiv_rel_err_sum / equiv_err_count
            print(f"Test {name} Equivariance Error: {eval_results[name]['equiv_error']} (relative: {eval_results[name]['equiv_error_rel']})")

            G = config.get('interior_size', 14)
            n_voxels = G ** 3  # 2744 at filter ratio 4
            if config.pointwise:
                input0 = inputs[:n_voxels, :]
                output0 = model(input0).reshape(G, G, G, 5).permute(3, 0, 1, 2)
                input0 = input0.reshape(G, G, G, 9).permute(3, 0, 1, 2)
                target0 = targets[:n_voxels, :].reshape(G, G, G, 5).permute(3, 0, 1, 2)
            else:
                input0 = inputs[0:1]
                output0 = model(input0)[0]
                input0 = input0[0]
                target0 = targets[0]

            visualize_predictions_3D(
                inputs=input0, preds=output0, targets=target0,
                output_path=os.path.join(experiment_dir, f'{config.barcode}_{name}_predictions_bestmodel_after.png'),
            )

        eval_results['param_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Timing analysis (done on the last test dataset)
        inputs, targets = eval_datasets['Re1000_nearwall_test'].get_all_data()
        n_warmup = 50
        n_runs = 200
        chunk_eig = config.model.model_type == 'sgsPointwiseEigenframeMLP'
        chunk_size = 25000

        if chunk_eig:
            x = inputs.repeat(8, *([1] * (inputs.ndim - 1))).to(device)
            n_points = x.shape[0]

            with torch.no_grad():
                for _ in range(n_warmup):
                    for i in range(0, x.shape[0], chunk_size):
                        model(x[i:i+chunk_size])
                torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(n_runs):
                    for i in range(0, x.shape[0], chunk_size):
                        model(x[i:i+chunk_size])
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

            per_point = elapsed / n_runs / n_points
            print(f"Eigenframe: points={n_points:,}, per_point={per_point*1e6:.4f} µs, per_10M={per_point*10e6:.4f} s")

        else:
            multiplier = 1
            best_per_point = float("inf")
            best_multiplier = 1

            while True:
                try:
                    torch.cuda.empty_cache()
                    x = inputs.repeat(multiplier, *([1] * (inputs.ndim - 1))).to(device)
                    n_points = x.shape[0] if x.ndim == 2 else x.shape[0] * x.shape[2] * x.shape[3] * x.shape[4]

                    with torch.no_grad():
                        for _ in range(n_warmup):
                            model(x)
                        torch.cuda.synchronize()

                        start = time.perf_counter()
                        for _ in range(10):
                            model(x)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start

                    per_point = elapsed / 10 / n_points
                    print(f"mult={multiplier:>6d}  points={n_points:>10,}  per_point={per_point*1e6:.4f} µs")

                    if per_point < best_per_point:
                        best_per_point = per_point
                        best_multiplier = multiplier

                    del x
                    torch.cuda.empty_cache()
                    multiplier *= 2

                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    torch.cuda.empty_cache()
                    print("OOM, stopping search.")
                    break

            print(f"\nBest multiplier from search: {best_multiplier}")

            torch.cuda.empty_cache()
            x = inputs.repeat(best_multiplier, *([1] * (inputs.ndim - 1))).to(device)
            n_points = x.shape[0] if x.ndim == 2 else x.shape[0] * x.shape[2] * x.shape[3] * x.shape[4]

            with torch.no_grad():
                for _ in range(n_warmup):
                    model(x)
                torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(n_runs):
                    model(x)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

            per_point = elapsed / n_runs / n_points
            print(f"Final: multiplier={best_multiplier}, points={n_points:,}, per_point={per_point*1e6:.4f} µs, per_10M={per_point*10e6:.4f} s")

        eval_results['time_per_point'] = per_point
        eval_results['time_10M_points'] = per_point * 10e6

        np.save(os.path.join(experiment_dir, f'{config.barcode}_eval_results.npy'), eval_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--compute_spectra', action='store_true', help='compute equivariance error spectra')

    args = parser.parse_args()
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # required on macOS/Python 3.13

    eval_model_after_training(args.experiment_dir, args.device, args.compute_spectra)