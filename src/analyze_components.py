import argparse
import os

import numpy as np
import torch
import yaml
from box import Box

import global_config as global_config
from analyze_dissipation import load_boxes, predict, scaling_factors
from data import get_test_file_lists, get_tv_file_lists
from models import get_model

# The five stored components plus the trace-free zz, which is what a reader wants
# to see split into normal and shear parts. These are lab-frame quantities and so
# are deliberately not rotation invariant.
COMPONENTS = ['tau_11', 'tau_22', 'tau_12', 'tau_13', 'tau_23', 'tau_33']


def with_zz(tau):
    """Append the reconstructed zz component."""
    return torch.cat([tau, (-tau[:, 0:1] - tau[:, 1:2])], dim=1)


def component_stats(experiment_dir, device, batch_size=25):
    config = Box(yaml.safe_load(open(os.path.join(experiment_dir, 'config.yaml'))))
    model = get_model(config.model).to(device)
    checkpoint = torch.load(os.path.join(experiment_dir, f'{config.barcode}_checkpoint.pt'), map_location=device)
    model.load_state_dict({k: v.to(device) for k, v in checkpoint['model'].items()})
    model.eval()

    train_inputs, train_targets, _, _ = get_tv_file_lists(config.dataset)
    input_scale, target_scale = scaling_factors(train_inputs, train_targets)

    results = {}
    with torch.inference_mode():
        for name, spec in config.dataset.eval_datasets.items():
            inputs, targets = get_test_file_lists(spec)
            grad, tau_true, _ = load_boxes(inputs, targets)
            grad, tau_true = grad.to(device), tau_true.to(device)
            pred = predict(model, grad / input_scale, config.pointwise, batch_size) * target_scale

            t, p = with_zz(tau_true), with_zz(pred)
            stats = {}
            for i, comp in enumerate(COMPONENTS):
                ti, pi = t[:, i].flatten(), p[:, i].flatten()
                stats[comp] = {
                    'correlation': (torch.dot(pi, ti) / (pi.norm() * ti.norm())).item(),
                    'nmse': ((pi - ti).pow(2).mean() / ti.pow(2).mean()).item(),
                    'rms_ratio': (pi.std() / ti.std()).item(),
                }
            results[name] = stats
            line = '  '.join(f"{c}: r={stats[c]['correlation']:.3f} nmse={stats[c]['nmse']:.3f}" for c in COMPONENTS)
            print(f"  {name:22s} {line}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    stats = component_stats(args.experiment_dir, args.device)
    np.save(os.path.join(args.experiment_dir, 'component_stats.npy'), stats)
