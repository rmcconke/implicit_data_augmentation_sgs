"""Equivariance error in the paper's normalization (Frobenius norm over tau_ref).

src/eval_model.py stores equiv_error in standardized units: the mean Frobenius norm of
f(g.x) - g.f(x) divided by the mean raw five-component L2 norm of the un-augmented
training targets (target_mean_magnitude, src/data.py). The paper normalizes by tau_ref,
the mean Frobenius norm of the same training targets. This script multiplies every
stored equiv_error of every 26_09 run by tau_euclid / tau_frob, both computed from that
run's own training files, and writes one row per run. The CSV feeds Fig. 3
(analysis/figures/plot_sweeps.py), the Fig. 3 slopes (analysis/fig3_slopes.py) and every
E_equiv value quoted in the text. Needs the filtered training datasets; CPU, minutes.

    .venv/bin/python analysis/equiv_error_frob.py [--out analysis/results/equiv_error_frob.csv]
"""
import argparse
import csv
import glob
import os

import numpy as np
import torch
import yaml
from box import Box

import _common

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--out', default=None, help='output CSV (default: analysis/results/equiv_error_frob.csv)')
parser.add_argument('--prefix', default='26_09', help='barcode prefix of the runs to include')
args = parser.parse_args()
out_path = os.path.abspath(args.out) if args.out else os.path.join(_common.results_dir(), 'equiv_error_frob.csv')

_common.setup()
from data import get_tv_file_lists
from deviatoric import frobenius_norm

cache = {}


def taus(cfg):
    _, ttg, _, _ = get_tv_file_lists(cfg.dataset)
    key = tuple(ttg)
    if key not in cache:
        t = torch.from_numpy(np.stack([np.load(f) for f in ttg])).double()
        cache[key] = (torch.linalg.norm(t, dim=1).mean().item(), frobenius_norm(t, dim=1).mean().item(), len(ttg))
    return cache[key]


rows = []
for d in sorted(glob.glob(f'outputs/*/{args.prefix}*/')):
    f = glob.glob(d + '*eval_results.npy')
    if not f:
        continue
    cfg = Box(yaml.safe_load(open(d + 'config.yaml')))
    te, tf, n = taus(cfg)
    r = np.load(f[0], allow_pickle=True).item()
    row = {'sweep': d.split('/')[1], 'run': d.split('/')[2], 'train_fraction': cfg.dataset.get('train_fraction', 1.0),
           'n_train_boxes': n, 'tau_euclid': te, 'tau_frob': tf, 'ratio': te / tf, 'param_count': r.get('param_count')}
    for k, v in r.items():
        if isinstance(v, dict) and 'equiv_error' in v:
            row[f'{k}_equiv_error_frob'] = v['equiv_error'] * te / tf
    rows.append(row)

first = ('sweep', 'run', 'train_fraction', 'n_train_boxes', 'tau_euclid', 'tau_frob', 'ratio', 'param_count')
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in first, k))
with open(out_path, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(f'{len(rows)} runs, {len(cache)} distinct training sets -> {out_path}')
ratios = [r['ratio'] for r in rows]
print(f'ratio tau_euclid/tau_frob: min {min(ratios):.4f} max {max(ratios):.4f}')
for r in rows:
    if r['sweep'].endswith('_data') or r['sweep'].endswith('_data_middle'):
        if r['train_fraction'] == 1.0:
            reg = 'middle' if r['sweep'].endswith('middle') else 'nearwall'
            print(f"{r['sweep']:28s} f=1 val E_frob = {r[f'Re1000_{reg}_val_equiv_error_frob']:.3f}")
