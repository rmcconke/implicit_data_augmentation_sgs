"""Rotation dependence of the raw five-component metrics versus the invariant ones.

Takes the near-wall ESCNN of Table 3, predicts every box of the Re_tau = 1000 near-wall
test set, rotates prediction and target jointly by each of the 24 elements of the
octahedral group, and recomputes the correlation and MSE on the raw five stored
components and on the orthonormal-basis coefficients (src/deviatoric.py). The raw
metrics change with the rotation (relative spread around 1e-3 for rho and 6e-2 for
MSE); the invariant ones change only at round-off (around 1e-10 and 6e-9). This is a
supporting check for the rotation-invariant loss and metrics of Section 2; its numbers
are not quoted in the paper. Needs the test datasets; runs on CPU or GPU in a few minutes.

    .venv/bin/python analysis/invariance_demo.py [--run RUN_DIR] [--device cpu]
"""
import argparse

import numpy as np

import _common

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--run', default=None, help='run directory (default: Table 3 near-wall ESCNN)')
parser.add_argument('--set', default='spatiotemporal', choices=['spatiotemporal', 'anisotropy', 'reynolds'])
parser.add_argument('--device', default=None)
args = parser.parse_args()

_common.setup()
import torch
import torch.nn.functional as F
from deviatoric import to_orthonormal
from transformations import rotate_octahedral_exact

prs = _common.load_plots_module('plot_results_sgs')
dev = args.device or _common.default_device()
run = args.run or _common.one(_common.TABLE3_NEARWALL['ESCNN'])
model, cfg, sc = prs.load_model(run, dev)
ep = prs.EVAL_SETS[args.set]
n = len(prs.get_test_file_lists(prs.Box(ep))[0])
P, T = [], []
with torch.inference_mode():
    for idx in range(n):
        x, y, _ = prs.get_one_box(ep, sc, cfg.pointwise, dev, idx=idx)
        P.append(model(x[0:1]).float().cpu())
        T.append(y[0:1].float().cpu())
P = torch.cat(P).double()   # (n, 5, 14, 14, 14)
T = torch.cat(T).double()


def rho(p, t):
    return float((p * t).sum() / (p.pow(2).sum().sqrt() * t.pow(2).sum().sqrt()))


Pr = rotate_octahedral_exact(P, mode='whole_group')
Tr = rotate_octahedral_exact(T, mode='whole_group')
rows = []
for g in range(24):   # g = 0 is the identity
    p, t = Pr[:, g], Tr[:, g]
    pi, ti = to_orthonormal(p), to_orthonormal(t)
    rows.append((rho(p, t), rho(pi, ti), float(F.mse_loss(p, t)), float(F.mse_loss(pi, ti))))
r = np.array(rows)
names = ['rho raw', 'rho invariant', 'MSE raw', 'MSE invariant']
print(f'{run}, {args.set} test set, {n} boxes, 24 rotations')
for k, nm in enumerate(names):
    v = r[:, k]
    print(f'  {nm:14s} identity {v[0]:.6f}  min {v.min():.6f}  max {v.max():.6f}  max rel. spread {(v.max()-v.min())/abs(v[0]):.2e}')
