"""Figs. 3, 5 and 6 from the invariant-loss runs, plus the numbers quoted with them.

Builds Fig. 5 (pointwise_sweep.pdf) and Fig. 6 (nonlocal_sweep.pdf) with the plotting
functions of plots/plot_sweep_results.py, fed from the run inventory (26_09/26_10
runs only, read from each run's eval_results.npy) instead of the per-sweep CSVs, and
Fig. 3 (implicit_augmentation.pdf) from the Frobenius/tau_ref-normalized equivariance
error of analysis/equiv_error_frob.py (plots/plot_sweep_results.py plots the stored,
five-component-normalized value instead). It also prints the data behind each figure,
the ESCNN/CNN crossover in parameters and in wall-clock time, local log-log slopes,
and the Fig. 3 slopes. Inputs: analysis/figures/inventory.py and
analysis/equiv_error_frob.py outputs. CPU, seconds.

    .venv/bin/python analysis/figures/plot_sweeps.py [--inventory CSV] [--frob CSV] [--out DIR]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, LogLocator  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--inventory', default=os.path.join(_common.RESULTS, 'figures', 'inventory.csv'))
parser.add_argument('--frob', default=os.path.join(_common.RESULTS, 'equiv_error_frob.csv'))
parser.add_argument('--out', default=None, help='output directory (default: analysis/results/figures)')
args = parser.parse_args()
OUT = os.path.abspath(args.out) if args.out else _common.results_dir('figures')
os.makedirs(OUT, exist_ok=True)
inventory_path, frob_path = os.path.abspath(args.inventory), os.path.abspath(args.frob)

_common.setup()
psr = _common.load_plots_module('plot_sweep_results')   # prints the Table 3 values on import
psr.PLOT_OUTPUT_DIR = OUT

inv = pd.read_csv(inventory_path)
inv = inv[inv['new'] & inv['has_eval']].copy()
assert inv['run'].str[:5].isin(['26_09', '26_10']).all()
inv = inv.rename(columns={'train_fraction': 'dataset_train_fraction',
                          'hidden': 'model_hidden_channels', 'n_layers': 'model_n_layers'})


def sweep(name):
    d = inv[inv['sweep'] == name].copy()
    assert len(d) > 0, name
    return d


print('\n==== invariant-loss sweep data (26_09/26_10 only) ====')
escnn, cnn = sweep('ESCNN_params'), sweep('CNNaug_params')
eig, pnt = sweep('pointwiseEigenFrame_params'), sweep('pointwiseAug_params')
data_dfs = {k: sweep(os.path.basename(v)) for k, v in psr.DATA_SWEEP_DIRS.items()}
for k, df in data_dfs.items():
    assert sorted(df['dataset_train_fraction']) == [0.1, 0.3, 0.5, 0.75, 1.0], k
    assert df['param_count'].nunique() == 1, k

cols = ['run', 'param_count', 'time_10M_points', 'Re1000_nearwall_val_loss', 'Re1000_nearwall_test_correlation']
for n, d in [('ESCNN_params', escnn), ('CNNaug_params', cnn), ('pointwiseEigenFrame_params', eig), ('pointwiseAug_params', pnt)]:
    print(f'\n-- {n}'); print(d.sort_values('param_count')[cols].to_string(index=False))
for k, d in data_dfs.items():
    print(f'\n-- {k}'); print(d.sort_values('dataset_train_fraction')[['run', 'dataset_train_fraction', 'param_count', 'Re1000_nearwall_val_loss']].to_string(index=False))

psr.plot_pointwise(eig, pnt, data_dfs)
psr.plot_nonlocal(cnn, escnn, data_dfs)
plt.close('all')


# ---------- Fig 3: implicit augmentation, Frobenius-normalized E_equiv ----------
frob = pd.read_csv(frob_path)
frob = frob[frob['run'].str[:5].isin(['26_09', '26_10'])]


def plot_implicit_augmentation_frob():
    lw, size = psr.linewidth, 3
    c_nw, c_md = psr.c_nw, psr.c_md
    series = [
        ('pointwiseNoAug_data',        'nearwall', c_nw, 'o', '--', 'MLP, no aug (near-wall)'),
        ('pointwiseAug_data',          'nearwall', c_nw, 'o', '-',  'MLP, aug (near-wall)'),
        ('CNNnoaug_data',              'nearwall', c_nw, 's', '--', 'CNN, no aug (near-wall)'),
        ('CNNaug_data',                'nearwall', c_nw, 's', '-',  'CNN, aug (near-wall)'),
        ('pointwiseNoAug_data_middle', 'middle',   c_md, 'o', '--', 'MLP, no aug (channel-center)'),
        ('pointwiseAug_data_middle',   'middle',   c_md, 'o', '-',  'MLP, aug (channel-center)'),
        ('CNNnoaug_data_middle',       'middle',   c_md, 's', '--', 'CNN, no aug (channel-center)'),
        ('CNNaug_data_middle',         'middle',   c_md, 's', '-',  'CNN, aug (channel-center)'),
    ]
    fig, ax = plt.subplots(figsize=(6, 3))
    print('\n-- Fig 3 data: E_equiv (Frobenius, val set of training region)')
    for sweep_name, region, color, marker, ls, label in series:
        df = frob[frob['sweep'] == sweep_name].sort_values('train_fraction')
        assert sorted(df['train_fraction']) == [0.1, 0.3, 0.5, 0.75, 1.0], sweep_name
        assert df['run'].str.contains(region).all()
        col = f'Re1000_{region}_val_equiv_error_frob'
        n_points = df['n_train_boxes'] * 14 ** 3
        assert np.allclose(n_points, df['train_fraction'] * psr.N_TOTAL_POINTS)
        print(f'{label:24s}', ' '.join(f'{v:.4f}' for v in df[col]),
              f'  ratio f0.1/f1.0 = {df[col].iloc[0] / df[col].iloc[-1]:.2f}',
              f'  log-log slope = {np.polyfit(np.log(n_points), np.log(df[col]), 1)[0]:.3f}')
        ax.plot(n_points, df[col], marker=marker, color=color, linestyle=ls,
                markerfacecolor=color if ls == '-' else 'none',
                markersize=size, linewidth=lw, markeredgewidth=lw, label=label)
    ax.set_xscale('log'); ax.set_yscale('log')
    # Labelled major ticks only, no minor tick labels.
    xt = [2.5e5, 5e5, 1e6, 2e6]
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_major_formatter(FixedFormatter([r'$2.5\times10^{5}$', r'$5\times10^{5}$', r'$10^{6}$', r'$2\times10^{6}$']))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ymin, ymax = ax.get_ylim()
    yt = [v for v in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0] if ymin <= v <= ymax]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in yt]))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r'Number of training points, $N_\mathrm{train}$')
    ax.set_ylabel(r'Equivariance error, $\mathcal{E}_\mathrm{equiv}$')
    psr._style_legend(ax, fontsize=6, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    p = os.path.join(OUT, 'implicit_augmentation.pdf')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print('Saved', p)


plot_implicit_augmentation_frob()


# ---------- numbers the text may reference ----------
def loglog_interp(x, y, xq):
    return np.exp(np.interp(np.log(xq), np.log(x), np.log(y)))

e = escnn.sort_values('param_count'); c = cnn.sort_values('param_count')
for metric in ['Re1000_nearwall_val_loss', 'Re1000_nearwall_test_loss']:
    xs = np.logspace(np.log10(max(e.param_count.min(), c.param_count.min())),
                     np.log10(min(e.param_count.max(), c.param_count.max())), 20001)
    diff = np.log(loglog_interp(e.param_count, e[metric], xs)) - np.log(loglog_interp(c.param_count, c[metric], xs))
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    print(f'\nESCNN-vs-CNN crossover on {metric} (log-log interpolation): ',
          ', '.join(f'{xs[i]:.3g}' for i in idx) or 'none')
for metric in ['Re1000_nearwall_val_loss']:
    for n, d in [('ESCNN', e), ('CNN', c)]:
        sl = np.diff(np.log(d[metric])) / np.diff(np.log(d.param_count))
        print(f'{n} local log-log slopes of {metric} vs params:', ' '.join(f'{s:.3f}' for s in sl))
    # time crossover
    et, ct = e.sort_values('time_10M_points'), c.sort_values('time_10M_points')
    ts = np.logspace(np.log10(max(et.time_10M_points.min(), ct.time_10M_points.min())),
                     np.log10(min(et.time_10M_points.max(), ct.time_10M_points.max())), 20001)
    diff = np.log(loglog_interp(et.time_10M_points, et[metric], ts)) - np.log(loglog_interp(ct.time_10M_points, ct[metric], ts))
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    print('crossover on wall-clock time (s/10M):', ', '.join(f'{ts[i]:.3g}' for i in idx) or 'none')
print('\nmin val loss: ESCNN %.5f at %d ; CNN %.5f at %d' % (e[metric].min(), e.loc[e[metric].idxmin(), 'param_count'],
      c[metric].min(), c.loc[c[metric].idxmin(), 'param_count']))
