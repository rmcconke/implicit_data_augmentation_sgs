"""Numbers quoted in Section 3 alongside Figs. 3, 5 and 6.

From the run inventory and the Frobenius-normalized equivariance errors: the
ESCNN/CNN validation-loss crossover in parameters (invariant-loss runs, and the
earlier 26_05 runs trained with the non-invariant loss, for comparison), ESCNN/CNN loss ratios at matched parameter
count and at matched wall-clock time, the loss reduction across the ESCNN sweep, the
excess of the selected pointwise models over their sweep minimum, the change in
validation loss and log-log slope of every data-size sweep, and the extrapolated
training-set size at which the Fig. 3 aug and no-aug fits would meet. CPU, seconds.

    .venv/bin/python analysis/figures/check_numbers.py [--inventory CSV] [--frob CSV]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--inventory', default=os.path.join(_common.RESULTS, 'figures', 'inventory.csv'))
parser.add_argument('--frob', default=os.path.join(_common.RESULTS, 'equiv_error_frob.csv'))
args = parser.parse_args()
inv = pd.read_csv(args.inventory)
frob = pd.read_csv(args.frob)
new = inv[inv.new & inv.has_eval]; old = inv[~inv.new & inv.has_eval]
L = 'Re1000_nearwall_val_loss'
def li(x, y, q): return np.exp(np.interp(np.log(q), np.log(x), np.log(y)))
def cross(df_e, df_c, col='param_count'):
    e, c = df_e.sort_values(col), df_c.sort_values(col)
    xs = np.logspace(np.log10(max(e[col].min(), c[col].min())), np.log10(min(e[col].max(), c[col].max())), 40001)
    d = np.log(li(e[col], e[L], xs)) - np.log(li(c[col], c[L], xs))
    return xs[np.where(np.diff(np.sign(d)) != 0)[0]], xs, d
for tag, d in [('invariant loss (26_09)', new), ('non-invariant loss (26_05)', old)]:
    x, _, _ = cross(d[d.sweep == 'ESCNN_params'], d[d.sweep == 'CNNaug_params'])
    print(f'{tag} ESCNN/CNN val-loss crossover (params):', x)
e = new[new.sweep == 'ESCNN_params'].sort_values('param_count'); c = new[new.sweep == 'CNNaug_params'].sort_values('param_count')
print('\nESCNN/CNN val-loss ratio at ESCNN sizes (CNN log-log interpolated):')
for _, r in e.iterrows():
    if c.param_count.min() <= r.param_count <= c.param_count.max():
        print(f'  {r.param_count:>9,d}  ESCNN {r[L]:.6f}  CNN {li(c.param_count, c[L], r.param_count):.6f}  ratio {r[L] / li(c.param_count, c[L], r.param_count):.3f}')
print('\nAt matched wall-clock time: ESCNN loss / CNN loss (CNN interpolated):')
ct = c.sort_values('time_10M_points')
for _, r in e.sort_values('time_10M_points').iterrows():
    if ct.time_10M_points.min() <= r.time_10M_points <= ct.time_10M_points.max():
        print(f'  t={r.time_10M_points:.4f}s params {r.param_count:>8,d} ratio {r[L] / li(ct.time_10M_points, ct[L], r.time_10M_points):.3f}')
print('At matched params ratio (ESCNN/CNN) at 46,872:', e[e.param_count == 46872][L].iloc[0] / li(c.param_count, c[L], 46872))
print('\nESCNN 46,872 -> 2,955,456: val loss reduction %.2f%%, factor %.2f' % (100 * (1 - e[L].iloc[-1] / e[e.param_count == 46872][L].iloc[0]), 2955456 / 46872))
for s, n in [('pointwiseAug_params', 21765), ('pointwiseEigenFrame_params', 21445)]:
    d = new[new.sweep == s]; v = d[d.param_count == n][L].iloc[0]
    print(f'{s}: val loss at {n} = {v:.6f}, min = {d[L].min():.6f}, excess {100 * (v / d[L].min() - 1):.3f}%')
print('\nData-fraction sweeps (near-wall val loss), change 0.1 -> 1.0 and log-log slope vs N:')
for s in ['ESCNN_data', 'CNNaug_data', 'CNNnoaug_data', 'pointwiseAug_data', 'pointwiseNoAug_data', 'pointwiseEigenFrame_data',
          'pointwiseAug_data_middle', 'pointwiseNoAug_data_middle', 'CNNaug_data_middle', 'CNNnoaug_data_middle']:
    d = new[new.sweep == s].sort_values('train_fraction')
    reg = 'middle' if 'middle' in s else 'nearwall'
    col = f'Re1000_{reg}_val_loss'
    sl = np.polyfit(np.log(d.train_fraction), np.log(d[col]), 1)[0]
    print(f'  {s:28s} {col:24s} {d[col].iloc[0]:.6f} -> {d[col].iloc[-1]:.6f} ({100 * (d[col].iloc[-1] / d[col].iloc[0] - 1):+.1f}%)  slope {sl:+.4f}')
print('\nFig 3 extrapolation: N at which log-log fits of no-aug and aug equivariance error meet, relative to full set')
for reg, suf in [('nearwall', ''), ('middle', '_middle')]:
    for m in ['pointwise', 'CNN']:
        a = frob[frob.sweep == (f'{m}Aug_data{suf}' if m == 'pointwise' else f'CNNaug_data{suf}')].sort_values('train_fraction')
        n = frob[frob.sweep == (f'{m}NoAug_data{suf}' if m == 'pointwise' else f'CNNnoaug_data{suf}')].sort_values('train_fraction')
        col = f'Re1000_{reg}_val_equiv_error_frob'
        pa = np.polyfit(np.log(a.train_fraction), np.log(a[col]), 1); pn = np.polyfit(np.log(n.train_fraction), np.log(n[col]), 1)
        lf = (pa[1] - pn[1]) / (pn[0] - pa[0])
        print(f'  {reg:8s} {m:9s} slopes noaug {pn[0]:+.3f} aug {pa[0]:+.3f}  gap at f=1: {n[col].iloc[-1] / a[col].iloc[-1]:.2f}x  meet at {np.exp(lf):.3g} x full set'
              + ('' if pn[0] < pa[0] else ' (diverging: no closure)'))
