"""Inventory of every run in the sweep folders behind Figs. 3, 5 and 6.

Reads config.yaml and *_eval_results.npy of every run directory (barcodes 26_*) in the
14 parameter and data-size sweep folders and writes one CSV row per run: sweep,
barcode, whether it is an invariant-loss run (26_09/26_10), training fraction, model
settings, parameter count, inference time and every stored loss, correlation and
equivariance error. analysis/figures/plot_sweeps.py and check_numbers.py read this
file. CPU, seconds.

    .venv/bin/python analysis/figures/inventory.py [--out analysis/results/figures/inventory.csv]
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

import numpy as np
import yaml

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--out', default=None)
args = parser.parse_args()
out_path = os.path.abspath(args.out) if args.out else os.path.join(_common.results_dir('figures'), 'inventory.csv')
_common.setup()

SWEEPS = ['ESCNN_params', 'CNNaug_params', 'pointwiseAug_params', 'pointwiseEigenFrame_params',
          'CNNaug_data', 'CNNnoaug_data', 'ESCNN_data', 'pointwiseAug_data', 'pointwiseNoAug_data',
          'pointwiseEigenFrame_data', 'CNNaug_data_middle', 'CNNnoaug_data_middle',
          'pointwiseAug_data_middle', 'pointwiseNoAug_data_middle']
rows = []
for s in SWEEPS:
    for d in sorted(glob.glob(f'outputs/{s}/26_*/')):
        run = d.rstrip('/').split('/')[-1]
        f = glob.glob(d + '*eval_results.npy')
        cfg = yaml.safe_load(open(d + 'config.yaml'))
        m = cfg['model']
        row = dict(sweep=s, run=run, date=run[:8], new=run[:5] in ('26_09', '26_10'),
                   has_eval=bool(f), train_fraction=cfg['dataset'].get('train_fraction', 1.0),
                   input_prefix=cfg['dataset'].get('input_prefix'), aug=cfg.get('train_aug_group'),
                   seed=cfg.get('seed'), loss_type=cfg.get('loss_type'), model_type=m.get('model_type'),
                   hidden=str(m.get('hidden_channels', m.get('hidden_dims'))), n_layers=m.get('n_layers'),
                   n_reg_repr=m.get('n_reg_repr'), batch_norm=m.get('batch_norm'))
        if f:
            r = np.load(f[0], allow_pickle=True).item()
            row['param_count'] = r.get('param_count'); row['time_10M_points'] = r.get('time_10M_points')
            for k, v in r.items():
                if isinstance(v, dict):
                    for kk in ('loss', 'correlation', 'equiv_error', 'equiv_error_rel', 'target_second_moment'):
                        row[f'{k}_{kk}'] = v.get(kk)
        rows.append(row)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(out_path, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(len(rows), 'runs;', sum(r['new'] for r in rows), 'with the invariant loss (26_09/26_10) ->', out_path)
