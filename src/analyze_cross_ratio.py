import argparse
import copy
import os

import numpy as np
import torch
import yaml
from box import Box

import global_config as global_config
from analyze_dissipation import load_boxes, predict, scaling_factors
from data import get_test_file_lists, get_tv_file_lists
from deviatoric import to_orthonormal
from models import get_model

# Cross-filter-width test: the filter width is not matched between training and
# evaluation. The stored targets are already dimensionless (normalized by
# delta^2 G^2), so a model trained at one ratio can be scored directly at another.
RATIOS = ['fs2', 'fs4', 'fs8']


def cross_ratio(experiment_dir, device, batch_size=5):
    config = Box(yaml.safe_load(open(os.path.join(experiment_dir, 'config.yaml'))))
    model = get_model(config.model).to(device)
    checkpoint = torch.load(os.path.join(experiment_dir, f'{config.barcode}_checkpoint.pt'), map_location=device)
    model.load_state_dict({k: v.to(device) for k, v in checkpoint['model'].items()})
    model.eval()

    train_inputs, train_targets, _, _ = get_tv_file_lists(config.dataset)
    input_scale, target_scale = scaling_factors(train_inputs, train_targets)

    results = {}
    with torch.inference_mode():
        for ratio in RATIOS:
            for name in ['Re1000_nearwall_test', 'Re1000_middle_test']:
                spec = copy.deepcopy(config.dataset.eval_datasets[name])
                spec.input_prefix = spec.input_prefix.replace('_fs4', f'_{ratio}')
                spec.target_prefix = spec.target_prefix.replace('_fs4', f'_{ratio}')
                inputs, targets = get_test_file_lists(spec)
                grad, tau_true, _ = load_boxes(inputs, targets)
                grad, tau_true = grad.to(device), tau_true.to(device)
                pred = predict(model, grad / input_scale, config.pointwise, batch_size) * target_scale

                t, p = to_orthonormal(tau_true).flatten(), to_orthonormal(pred).flatten()
                rho = (torch.dot(p, t) / (p.norm() * t.norm())).item()
                results[f'{name}@{ratio}'] = {'correlation': rho, 'rms_ratio': (p.std() / t.std()).item()}
                print(f"  trained fs4 -> evaluated {ratio:3s}  {name:22s} rho={rho:.4f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    stats = cross_ratio(args.experiment_dir, args.device)
    np.save(os.path.join(args.experiment_dir, 'cross_ratio_stats.npy'), stats)
