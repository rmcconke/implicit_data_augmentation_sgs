import argparse
import os

import numpy as np
import torch
import yaml
from box import Box

import global_config as global_config
from data import get_test_file_lists, get_tv_file_lists
from dissipation import sgs_dissipation, wall_scaled
from models import get_model

# Filter width per dataset prefix, matching dataset_creation/sgs_filter_jhtb_data.py
DELTA = {
    'channel_nearwall': 4.0026e-02,
    'channel_middle': 4.0026e-02,
    'channel5200_nearwall': 7.7132e-03,
    'channel5200_middle': 7.7132e-03,
}


def load_boxes(input_files, target_files):
    """Load a set of boxes as fields, along with the local gradient magnitude."""
    grad = np.stack([np.load(f) for f in input_files])
    tau = np.stack([np.load(f) for f in target_files])
    G = np.stack([np.load(f.replace('_grad_', '_G_')) for f in input_files])
    return (torch.from_numpy(grad).float(), torch.from_numpy(tau).float(),
            torch.from_numpy(G).float())


def scaling_factors(input_files, target_files):
    """Reproduce the global standardization applied by NumpyTimeSeriesDataset."""
    grad, tau, _ = load_boxes(input_files, target_files)
    return (torch.linalg.norm(grad, dim=1).mean().item(),
            torch.linalg.norm(tau, dim=1).mean().item())


# cusolver cannot size a workspace for a batched 3x3 eigendecomposition much
# beyond this, which the eigenframe model hits on a flattened set of boxes.
POINT_CHUNK = 8192


def predict(model, grad_std, pointwise, batch_size):
    preds = []
    for i in range(0, grad_std.shape[0], batch_size):
        x = grad_std[i:i + batch_size]
        if pointwise:
            b, c, d, h, w = x.shape
            flat = x.permute(0, 2, 3, 4, 1).reshape(-1, c)
            chunks = [model(flat[j:j + POINT_CHUNK]) for j in range(0, flat.shape[0], POINT_CHUNK)]
            out = torch.cat(chunks).reshape(b, d, h, w, -1).permute(0, 4, 1, 2, 3)
        else:
            out = model(x)
        preds.append(out)
    return torch.cat(preds)


def dissipation_stats(experiment_dir, device, batch_size=25):
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
            grad, tau_true, G = load_boxes(inputs, targets)
            grad, tau_true, G = grad.to(device), tau_true.to(device), G.to(device)

            pred = predict(model, grad / input_scale, config.pointwise, batch_size) * target_scale
            grad_hat = grad
            delta = DELTA[spec.input_prefix.split('_grad')[0]]

            eps_true = sgs_dissipation(tau_true, grad_hat, G, delta)
            eps_pred = sgs_dissipation(pred, grad_hat, G, delta)
            t, p = wall_scaled(eps_true, spec.input_prefix), wall_scaled(eps_pred, spec.input_prefix)

            tf, pf = t.flatten(), p.flatten()
            results[name] = {
                'eps_true_mean': tf.mean().item(), 'eps_pred_mean': pf.mean().item(),
                'eps_true_std': tf.std().item(), 'eps_pred_std': pf.std().item(),
                'eps_correlation': (torch.dot(pf, tf) / (pf.norm() * tf.norm())).item(),
                'backscatter_true': (tf < 0).float().mean().item(),
                'backscatter_pred': (pf < 0).float().mean().item(),
            }
            r = results[name]
            print(f"  {name:22s} <eps>+ true={r['eps_true_mean']:+.5f} pred={r['eps_pred_mean']:+.5f} "
                  f"rho={r['eps_correlation']:.4f} bs true={100*r['backscatter_true']:.1f}% pred={100*r['backscatter_pred']:.1f}%")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    stats = dissipation_stats(args.experiment_dir, args.device)
    np.save(os.path.join(args.experiment_dir, 'dissipation_stats.npy'), stats)
