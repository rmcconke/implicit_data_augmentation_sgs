"""Clark model and Table 3 models under the rotation-invariant metrics.

Applies the Clark (gradient) model, tau_hat = dev(g g^T)/12 on the stored normalized
velocity gradients, to every evaluation set and reports its uncentered correlation
(raw five-component and orthonormal-basis), NMSE and RMS amplitude ratio, its SGS
dissipation statistics and its per-component correlations. It also recomputes the
orthonormal-basis correlation, NMSE and uncentered RMS ratio of the eight Table 3
models from their checkpoints and checks them against the stored eval_results.npy.
This produces Table 4 (amplitude: RMS ratio and NMSE for every row) and the Clark
rows of Table 3 (rho), Table 5 (dissipation) and Table 7 (components). Writes
clark_results.{json,npy} and model_results.{json,npy}; analysis/clark_report.py
turns them into Markdown tables. Needs the filtered datasets and a GPU (about 6 min
on one L40S).

    .venv/bin/python analysis/clark_metrics.py all [--device cuda:0] [--out DIR]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from box import Box

import _common


def clark(grad):
    """tau_hat = dev(g g^T)/12 on raw stored normalized gradients, field layout [B,9,D,H,W] -> [B,5,D,H,W]."""
    B = grad.shape[0]
    g = grad.reshape(B, 3, 3, *grad.shape[2:])
    t = torch.einsum('bik...,bjk...->bij...', g, g) / 12.0
    tr = (t[:, 0, 0] + t[:, 1, 1] + t[:, 2, 2]) / 3.0
    return torch.stack([t[:, 0, 0] - tr, t[:, 1, 1] - tr, t[:, 0, 1], t[:, 0, 2], t[:, 1, 2]], dim=1)


def cos(p, t):
    p, t = p.double(), t.double()
    return ((p * t).sum() / (p.pow(2).sum().sqrt() * t.pow(2).sum().sqrt())).item()


def aggregate(p, t):
    pi, ti = to_orthonormal(p).double(), to_orthonormal(t).double()
    tsq = ti.pow(2).sum(1).mean()
    return {
        'rho_raw': cos(p, t),
        'rho_inv': cos(pi, ti),
        'nmse': ((pi - ti).pow(2).sum(1).mean() / tsq).item(),
        'rms_ratio': (pi.pow(2).sum(1).mean() / tsq).sqrt().item(),
        'target_second_moment_inv': (ti.pow(2).mean()).item(),  # per-component mean, eval_model convention
        'mse_inv': (pi - ti).pow(2).mean().item(),
    }


def dissipation_block(pred, tau_true, grad, G, spec):
    # verbatim logic of analyze_dissipation.dissipation_stats
    delta = DELTA[spec.input_prefix.split('_grad')[0]]
    eps_true = sgs_dissipation(tau_true, grad, G, delta)
    eps_pred = sgs_dissipation(pred, grad, G, delta)
    t, p = wall_scaled(eps_true, spec.input_prefix), wall_scaled(eps_pred, spec.input_prefix)
    tf, pf = t.flatten(), p.flatten()
    r = {
        'eps_true_mean': tf.mean().item(), 'eps_pred_mean': pf.mean().item(),
        'eps_true_std': tf.std().item(), 'eps_pred_std': pf.std().item(),
        'eps_correlation': (torch.dot(pf, tf) / (pf.norm() * tf.norm())).item(),
        'backscatter_true': (tf < 0).float().mean().item(),
        'backscatter_pred': (pf < 0).float().mean().item(),
    }
    r['mean_ratio'] = r['eps_pred_mean'] / r['eps_true_mean']
    r['std_ratio'] = r['eps_pred_std'] / r['eps_true_std']
    return r


def component_block(pred, tau_true):
    # verbatim logic of analyze_components.component_stats
    t, p = with_zz(tau_true), with_zz(pred)
    stats = {}
    for i, comp in enumerate(COMPONENTS):
        ti, pi = t[:, i].flatten(), p[:, i].flatten()
        stats[comp] = {
            'correlation': (torch.dot(pi, ti) / (pi.norm() * ti.norm())).item(),
            'nmse': ((pi - ti).pow(2).mean() / ti.pow(2).mean()).item(),
            'rms_ratio': (pi.std() / ti.std()).item(),  # centered std ratio, unlike aggregate()
        }
    return stats


def run_clark():
    cfg = Box(yaml.safe_load(open(REF_CONFIG)))
    res = {}
    with torch.inference_mode():
        for name, spec in cfg.dataset.eval_datasets.items():
            inputs, targets = get_test_file_lists(spec)
            grad, tau, G = (x.to(DEVICE) for x in load_boxes(inputs, targets))
            pred = clark(grad)
            r = {'n_boxes': len(inputs), 'shape': list(tau.shape)}
            r.update(aggregate(pred, tau))
            r['dissipation'] = dissipation_block(pred, tau, grad, G, spec)
            r['components'] = component_block(pred, tau)
            res[name] = r
            d = r['dissipation']
            print(f"CLARK {name:22s} rho_raw={r['rho_raw']:.4f} rho_inv={r['rho_inv']:.4f} nmse={r['nmse']:.4f} "
                  f"rms={r['rms_ratio']:.4f} | eps mean_ratio={d['mean_ratio']:.3f} std_ratio={d['std_ratio']:.3f} "
                  f"rho_eps={d['eps_correlation']:.4f} bs t/p={100*d['backscatter_true']:.1f}/{100*d['backscatter_pred']:.1f}%", flush=True)
            del grad, tau, G, pred
            torch.cuda.empty_cache()
    json.dump(res, open(os.path.join(OUT, 'clark_results.json'), 'w'), indent=1)
    np.save(os.path.join(OUT, 'clark_results.npy'), res)


def run_models(batch_size=25):
    allres = {}
    for run in RUNS:
        config = Box(yaml.safe_load(open(os.path.join(run, 'config.yaml'))))
        model = get_model(config.model).to(DEVICE)
        ck = torch.load(os.path.join(run, f'{config.barcode}_checkpoint.pt'), map_location=DEVICE)
        model.load_state_dict({k: v.to(DEVICE) for k, v in ck['model'].items()})
        model.eval()
        ev = np.load(os.path.join(run, f'{config.barcode}_eval_results.npy'), allow_pickle=True).item()
        tr_in, tr_tg, _, _ = get_tv_file_lists(config.dataset)
        in_scale, tg_scale = scaling_factors(tr_in, tr_tg)
        res = {'model_type': config.model.model_type, 'pointwise': config.pointwise,
               'train_aug_group': config.train_aug_group, 'input_scale': in_scale, 'target_scale': tg_scale}
        print(f"=== {config.barcode} ({config.model.model_type}, pointwise={config.pointwise}, aug={config.train_aug_group})", flush=True)
        with torch.inference_mode():
            for name, spec in config.dataset.eval_datasets.items():
                inputs, targets = get_test_file_lists(spec)
                grad, tau, _ = load_boxes(inputs, targets)
                grad, tau = grad.to(DEVICE), tau.to(DEVICE)
                pred = predict(model, grad / in_scale, config.pointwise, batch_size) * tg_scale
                r = aggregate(pred, tau)
                e = ev[name]
                r['eval_correlation'] = e['correlation']
                r['eval_loss'] = e['loss']
                r['eval_target_second_moment'] = e['target_second_moment']
                r['eval_nmse'] = e['loss'] / e['target_second_moment']
                r['corr_absdiff'] = abs(r['rho_inv'] - e['correlation'])
                r['nmse_absdiff'] = abs(r['nmse'] - r['eval_nmse'])
                res[name] = r
                print(f"  {name:22s} rho_inv={r['rho_inv']:.5f} (eval {e['correlation']:.5f}) nmse={r['nmse']:.5f} "
                      f"(eval loss/m2 {r['eval_nmse']:.5f}; m2 ours {r['target_second_moment_inv']:.6g} eval {e['target_second_moment']:.6g}) "
                      f"rms={r['rms_ratio']:.4f}", flush=True)
                del grad, tau, pred
                torch.cuda.empty_cache()
        allres[config.barcode] = res
        del model
        torch.cuda.empty_cache()
        json.dump(allres, open(os.path.join(OUT, 'model_results.json'), 'w'), indent=1)
    np.save(os.path.join(OUT, 'model_results.npy'), allres)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('mode', choices=['clark', 'models', 'all'])
    parser.add_argument('--device', default=None, help='default: cuda:0 if available, else cpu')
    parser.add_argument('--out', default=None, help='output directory (default: analysis/results/clark)')
    args = parser.parse_args()

    OUT = os.path.abspath(args.out) if args.out else _common.results_dir('clark')
    os.makedirs(OUT, exist_ok=True)
    _common.setup()
    from analyze_dissipation import load_boxes, scaling_factors, predict, DELTA
    from analyze_components import with_zz, COMPONENTS
    from data import get_test_file_lists, get_tv_file_lists
    from deviatoric import to_orthonormal
    from dissipation import sgs_dissipation, wall_scaled
    from models import get_model

    DEVICE = args.device or _common.default_device()
    REF_CONFIG = os.path.join(_common.one(_common.TABLE3_NEARWALL['ESCNN']), 'config.yaml')
    RUNS = ([_common.one(_common.TABLE3_NEARWALL[k]) for k in ('ESCNN', 'CNN (aug)', 'MLP (aug)', 'Eigenframe')]
            + sorted(_common.one(p) for p in _common.TABLE3_MIDDLE.values()))
    assert len(RUNS) == 8, RUNS

    if args.mode in ('clark', 'all'):
        run_clark()
    if args.mode in ('models', 'all'):
        run_models()
