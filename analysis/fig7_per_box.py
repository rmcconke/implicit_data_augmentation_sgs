"""Fig. 7 and how representative its displayed box is (per-box correlation).

Optionally regenerates Fig. 7 (predictions_magnitude.pdf, |tau_hat^d| on one test box
for the filtered DNS, Clark and the four near-wall Table 3 models) with
plots/plot_results_sgs.py. Then, for each of the three generalization test sets of
Fig. 7, it computes the uncentered Frobenius correlation of every test box for every
model and for Clark, and reports where the displayed box (file index 11 of the sorted
file list, as in plot_results_sgs.get_one_box) falls in that distribution: median,
interquartile range, extremes and the percentile rank of the displayed box. The
per-box statistics are a supporting check and are not quoted in the paper. Writes
per_box_rho.json (and the figure with --figure). Needs the test datasets; a GPU is
recommended (minutes per set).

    .venv/bin/python analysis/fig7_per_box.py [--figure] [--sets spatiotemporal anisotropy reynolds] [--out DIR]
"""
import argparse
import json
import os

import numpy as np

import _common

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--figure', action='store_true', help='also regenerate predictions_magnitude.pdf')
parser.add_argument('--sets', nargs='+', default=['spatiotemporal', 'anisotropy', 'reynolds'])
parser.add_argument('--out', default=None, help='output directory (default: analysis/results/fig7)')
parser.add_argument('--device', default=None)
args = parser.parse_args()
out = os.path.abspath(args.out) if args.out else _common.results_dir('fig7')
os.makedirs(out, exist_ok=True)

_common.setup()
import torch
from data import NumpyTimeSeriesDataset

prs = _common.load_plots_module('plot_results_sgs')
prs.OUT_DIR = out
if args.device:
    prs.DEVICE = args.device
prs.MODEL_DIRS = {name: _common.one(pat) for name, pat in _common.TABLE3_NEARWALL.items()}
SHOWN = 11
G = 14


def full(f):
    """(5, ...) stored components -> (3, 3, ...) deviatoric tensor."""
    t11, t22, t12, t13, t23 = f
    t33 = -t11 - t22
    return np.stack([np.stack([t11, t12, t13]), np.stack([t12, t22, t23]), np.stack([t13, t23, t33])])


def rho(p, t):
    P, T = full(p), full(t)
    return float((P * T).sum() / np.sqrt((P * P).sum() * (T * T).sum()))


if args.figure:
    prs.make_combined_figure()

models = {k: prs.load_model(d, prs.DEVICE) for k, d in prs.MODEL_DIRS.items()}
_, ref_cfg, ref_sc = models['Eigenframe']   # pointwise reference for the target and the Clark input
res = {}
for set_name in args.sets:
    ep = prs.EVAL_SETS[set_name]
    inp, tgt = prs.get_test_file_lists(prs.Box(ep))
    n = len(inp)

    def load(sc, pointwise):
        ds = NumpyTimeSeriesDataset(input_file_list=inp, target_file_list=tgt, scaling_factors=sc, augmentation_group=None,
                                    pointwise=pointwise, precompute_eigenframe_inputs=False, device=prs.DEVICE)
        return ds.get_all_data()

    x, y = load(ref_sc, ref_cfg.pointwise)   # pointwise layout: (n*G^3, 9), (n*G^3, 5)
    T = y.reshape(n, G, G, G, 5).permute(0, 4, 1, 2, 3).cpu().numpy() * float(ref_sc['target_mean_magnitude'])
    xc = x.reshape(n, G, G, G, 9).permute(0, 4, 1, 2, 3)
    rows = {'Clark': []}
    for i in range(n):
        # Clark acts on unstandardized normalized gradients: rescale by s_in^2.
        c = prs.clark_prediction(xc[i:i + 1]).cpu().numpy() * float(ref_sc['input_mean_magnitude']) ** 2
        rows['Clark'].append(rho(c, T[i]))
    del x, y, xc
    for name, (m, cfg, sc) in models.items():
        xi, _ = load(sc, cfg.pointwise)
        preds = []
        with torch.inference_mode():
            for i in range(n):
                if cfg.pointwise:
                    p = m(xi[i * G**3:(i + 1) * G**3]).reshape(G, G, G, 5).permute(3, 0, 1, 2)
                else:
                    p = m(xi[i:i + 1])[0]
                preds.append(rho(p.cpu().numpy() * float(sc['target_mean_magnitude']), T[i]))
        rows[name] = preds
        del xi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    res[set_name] = rows
    print(f'== {set_name}: {n} boxes, shown box index {SHOWN} ({os.path.basename(inp[SHOWN])})')
    for k, v in rows.items():
        v = np.array(v)
        s = v[SHOWN]
        print(f'  {k:11s} shown {s:.3f}  median {np.median(v):.3f}  IQR {np.percentile(v,25):.3f}-{np.percentile(v,75):.3f}  '
              f'min {v.min():.3f} max {v.max():.3f}  percentile of shown {100*(v < s).mean():.0f}', flush=True)
    json.dump(res, open(os.path.join(out, 'per_box_rho.json'), 'w'))
