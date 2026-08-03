import os
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIXGeneral']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9  # Base font size
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
linewidth_borders = 0.5
plt.rcParams['axes.linewidth'] = linewidth_borders
plt.rcParams['xtick.major.width'] = linewidth_borders
plt.rcParams['ytick.major.width'] = linewidth_borders
linewidth = 1.25


PLOT_OUTPUT_DIR = 'outputs/plots_final'
ESCNN_SWEEP_DIR = 'outputs/ESCNN_params'
CNN_SWEEP_DIR = 'outputs/CNNaug_params'
EIGENFRAME_SWEEP_DIR = 'outputs/pointwiseEigenFrame_params'
POINTWISE_SWEEP_DIR = 'outputs/pointwiseAug_params'

# Data efficiency sweep directories
DATA_SWEEP_DIRS = {
    # Nearwall
    'CNNaug':            'outputs/CNNaug_data',
    'CNNnoaug':          'outputs/CNNnoaug_data',
    'PointwiseAug':      'outputs/pointwiseAug_data',
    'PointwiseNoAug':    'outputs/pointwiseNoAug_data',
    'ESCNN':             'outputs/ESCNN_data',
    'Eigenframe':        'outputs/pointwiseEigenFrame_data',
    # Middle
    'CNNaug_md':         'outputs/CNNaug_data_middle',
    'CNNnoaug_md':       'outputs/CNNnoaug_data_middle',
    'PointwiseAug_md':   'outputs/pointwiseAug_data_middle',
    'PointwiseNoAug_md': 'outputs/pointwiseNoAug_data_middle',
}

SELECTED_MODELS = {
    'Eigenframe (nearwall)': 'outputs/pointwiseEigenFrame_params/26_05_23_16:39:46_live_kitten_nearwall_boxfilter_4x_sgs',
    'Pointwise (nearwall)':  'outputs/pointwiseAug_params/26_05_21_11:13:54_ace_burro_nearwall_boxfilter_4x_sgs',
    'ESCNN (nearwall)':      'outputs/ESCNN_params/26_05_21_11:07:09_nice_flea_nearwall_boxfilter_4x_sgs',
    'CNN (nearwall)':        'outputs/CNNaug_params/26_05_21_12:26:23_well_pony_nearwall_boxfilter_4x_sgs',
    'Eigenframe (middle)':   'outputs/26_06_05_13:53:03_main_tiger_middle_boxfilter_4x_sgs',
    'Pointwise (middle)':    'outputs/26_06_05_13:52:57_vocal_skunk_middle_boxfilter_4x_sgs',
    'ESCNN (middle)':        'outputs/26_06_05_14:16:00_whole_rat_middle_boxfilter_4x_sgs',
    'CNN (middle)':          'outputs/26_06_05_13:59:35_grown_jay_middle_boxfilter_4x_sgs',
}

# Total training set size in points: 900 boxes (15 streamwise * 3 spanwise * 20 times) * 14^3 voxels
N_TOTAL_POINTS = 900 * (14 ** 3)


def print_generalization_table_values():
    for name, run_dir in SELECTED_MODELS.items():
        matches = [f for f in os.listdir(run_dir) if f.endswith('_eval_results.npy')]
        results = np.load(os.path.join(run_dir, matches[0]), allow_pickle=True).item()
        print(name, 'params:', results.get('param_count'))
        for k, v in results.items():
            if isinstance(v, dict) and 'loss' in v:
                loss = v['loss']
                sm = v.get('target_second_moment')
                corr = v.get('correlation')
                nmse = loss / sm if sm else None
                print(f'  {k}: loss={loss:.3e}  NMSE={nmse:.3f}  corr={corr:.3f}'
                      if nmse is not None and corr is not None
                      else f'  {k}: loss={loss:.3e}')


print_generalization_table_values()

# Colors roughly matching the main plotting script
c_eig   = 'tab:orange'
c_pnt   = 'tab:red'
c_cnn   = 'tab:blue'
c_escnn = 'tab:green'

# Implicit-augmentation plot color/marker convention
c_nw = 'blue'
c_md = 'red'


def load_escnn_sweep():
    csv_path = os.path.join(ESCNN_SWEEP_DIR, 'sweep_results.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find ESCNN sweep results at {csv_path}")
    return pd.read_csv(csv_path)


def load_cnn_sweep():
    csv_path = os.path.join(CNN_SWEEP_DIR, 'sweep_results.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CNN sweep results at {csv_path}")
    return pd.read_csv(csv_path)


def load_pointwise_sweeps():
    eig_csv = os.path.join(EIGENFRAME_SWEEP_DIR, 'sweep_results.csv')
    pnt_csv = os.path.join(POINTWISE_SWEEP_DIR, 'sweep_results.csv')
    if not os.path.exists(eig_csv):
        raise FileNotFoundError(f"Could not find eigenframe sweep results at {eig_csv}")
    if not os.path.exists(pnt_csv):
        raise FileNotFoundError(f"Could not find pointwise sweep results at {pnt_csv}")
    eigenframe = pd.read_csv(eig_csv)
    pointwise = pd.read_csv(pnt_csv)
    return eigenframe, pointwise


def load_data_sweeps():
    """Load all data efficiency sweep CSVs. Returns dict of name -> DataFrame."""
    dfs = {}
    for name, sweep_dir in DATA_SWEEP_DIRS.items():
        csv_path = os.path.join(sweep_dir, 'sweep_results.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find data sweep results at {csv_path}")
        dfs[name] = pd.read_csv(csv_path)
    return dfs


def _parse_hidden_channels(s):
    """Convert string like "[16, 16, 16, 16]" to list[int]."""
    try:
        lst = ast.literal_eval(str(s))
        if isinstance(lst, (list, tuple)) and all(isinstance(x, (int, float)) for x in lst):
            return [int(x) for x in lst]
    except Exception:
        pass
    return None


def _style_legend(ax, **kwargs):
    leg = ax.legend(fancybox=False, edgecolor='black', **kwargs)
    leg.get_frame().set_linewidth(linewidth_borders)
    return leg


def _add_panel_label(ax, text, y=-0.2):
    ax.text(0.5, y, text, transform=ax.transAxes, ha='center', va='top')


def _get_param_count(df):
    """Return the param_count for a data-sweep df (fixed across all rows; take median)."""
    if 'param_count' not in df.columns:
        return None
    vals = df['param_count'].dropna()
    if vals.empty:
        return None
    return int(round(vals.median()))


def _fmt_params(n):
    """Format a parameter count compactly, e.g. 12345 -> '12.3k', 1234567 -> '1.23M'."""
    if n is None:
        return '?'
    if n >= 1_000_000:
        return f'{n/1_000_000:.2g}M'
    if n >= 1_000:
        return f'{n/1_000:.3g}k'
    return str(n)


DATA_FRAC_COL = 'dataset_train_fraction'


def _plot_data_efficiency(ax, data_dfs, models=None):
    """
    Plot val loss vs dataset_train_fraction on ax.
    models: optional list of keys to include (default: all).
    Legend entries include the fixed parameter count for each model.
    """
    lw = linewidth
    size = 3

    # (name, color, linestyle) for each data sweep model
    styles = {
        'CNNaug':         (c_cnn,   '-',  'CNN (aug)'),
        'CNNnoaug':       (c_cnn,   '--', 'CNN (no aug)'),
        'PointwiseAug':   (c_pnt,   '-',  'Pointwise (aug)'),
        'PointwiseNoAug': (c_pnt,   '--', 'Pointwise (no aug)'),
        'ESCNN':          (c_escnn, '-',  'ESCNN'),
        'Eigenframe':     (c_eig,   '-',  'Eigenframe'),
    }

    for key, (color, ls, display_name) in styles.items():
        if models is not None and key not in models:
            continue
        if key not in data_dfs:
            continue
        df = data_dfs[key].dropna(
            subset=[DATA_FRAC_COL, 'Re1000_nearwall_val_loss']
        ).sort_values(DATA_FRAC_COL)
        if df.empty:
            continue
        n_params = _get_param_count(df)
        label = f'{display_name} ({_fmt_params(n_params)} params)'
        ax.plot(
            df[DATA_FRAC_COL], df['Re1000_nearwall_val_loss'],
            'o' + ls, color=color, linewidth=lw, markersize=size, label=label,
        )

    ax.set_yscale('log')
    ax.set_xlabel('Training Data Fraction')


def _prepare_cnn_df(cnn_results: pd.DataFrame) -> pd.DataFrame:
    """Filter CNN results to depth-4 models."""
    required_cols = ['model_hidden_channels', 'Re1000_nearwall_val_loss', 'param_count', 'time_10M_points']
    missing = [c for c in required_cols if c not in cnn_results.columns]
    if missing:
        raise ValueError(f"Missing columns in CNN sweep_results.csv: {missing}")
    df = cnn_results.dropna(subset=required_cols).copy()
    parsed = df['model_hidden_channels'].apply(_parse_hidden_channels)
    df = df[parsed.notnull()].copy()
    df['hidden_list'] = parsed[parsed.notnull()]
    df['depth'] = df['hidden_list'].apply(len)
    df = df[df['depth'] == 4].copy()
    if df.empty:
        raise ValueError("No CNN results found with 4 hidden layers (depth == 4)")
    return df.sort_values('param_count')


def _prepare_escnn_df(escnn_results: pd.DataFrame) -> pd.DataFrame:
    """Filter ESCNN results to 4-layer models."""
    df = escnn_results.dropna(
        subset=['model_n_layers', 'Re1000_nearwall_val_loss', 'param_count', 'time_10M_points']
    )
    df = df[df['model_n_layers'] == 4].copy()
    if df.empty:
        raise ValueError("No ESCNN results found with model_n_layers == 4")
    return df.sort_values('param_count')


def plot_pointwise(eigenframe_results: pd.DataFrame, pointwise_results: pd.DataFrame,
                   data_dfs: dict):
    """Pointwise figure: Eigenframe + Pointwise across params, time, and data efficiency."""
    eig_df = eigenframe_results.dropna(
        subset=['Re1000_nearwall_val_loss', 'param_count', 'time_10M_points']
    ).sort_values('param_count')
    pnt_df = pointwise_results.dropna(
        subset=['Re1000_nearwall_val_loss', 'param_count', 'time_10M_points']
    ).sort_values('param_count')

    fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    lw = linewidth
    size = 3

    for df, color, label in [(eig_df, c_eig, 'Eigenframe'), (pnt_df, c_pnt, 'Pointwise')]:
        ax[0].plot(df['param_count'], df['Re1000_nearwall_val_loss'],
                   'o-', color=color, linewidth=lw, markersize=size, label=label)
        ax[1].plot(df['time_10M_points'], df['Re1000_nearwall_val_loss'],
                   'o-', color=color, linewidth=lw, markersize=size, label=label)

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Number of Parameters')
    ax[0].set_ylabel('Validation Loss')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Wall Clock Time for 10M points (s)')

    _plot_data_efficiency(ax[2], data_dfs, models=['Eigenframe', 'PointwiseAug', 'PointwiseNoAug'])

    for a in ax[:2]:
        _style_legend(a, fontsize=7, loc='lower left')
    _style_legend(ax[2], fontsize=7, loc='upper right', bbox_to_anchor=(1.0, 0.86))

    _add_panel_label(ax[0], '(a)')
    _add_panel_label(ax[1], '(b)')
    _add_panel_label(ax[2], '(c)')

    fig.tight_layout()
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(PLOT_OUTPUT_DIR, 'pointwise_sweep.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved pointwise sweep plot to {out_path}")


def plot_nonlocal(cnn_results: pd.DataFrame, escnn_results: pd.DataFrame,
                  data_dfs: dict):
    """Nonlocal figure: CNN + ESCNN (4-layer) across params, time, and data efficiency."""
    cnn_df   = _prepare_cnn_df(cnn_results)
    escnn_df = _prepare_escnn_df(escnn_results)

    fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    lw = linewidth
    size = 3

    for df, color, label in [(cnn_df, c_cnn, 'CNN'), (escnn_df, c_escnn, 'ESCNN')]:
        ax[0].plot(df['param_count'], df['Re1000_nearwall_val_loss'],
                   'o-', color=color, linewidth=lw, markersize=size, label=label)
        ax[1].plot(df['time_10M_points'], df['Re1000_nearwall_val_loss'],
                   'o-', color=color, linewidth=lw, markersize=size, label=label)

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Number of Parameters')
    ax[0].set_ylabel('Validation Loss')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Wall Clock Time for 10M points (s)')

    _plot_data_efficiency(ax[2], data_dfs, models=['CNNaug', 'CNNnoaug', 'ESCNN'])

    for a in ax:
        _style_legend(a, fontsize=7)

    _add_panel_label(ax[0], '(a)')
    _add_panel_label(ax[1], '(b)')
    _add_panel_label(ax[2], '(c)')

    fig.tight_layout()
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(PLOT_OUTPUT_DIR, 'nonlocal_sweep.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    print(f"Saved nonlocal sweep plot to {out_path}")


def plot_implicit_augmentation(data_dfs):
    """Equivariance error vs number of training points, 8 lines:
    {nearwall, middle} x {pointwise, CNN} x {aug, no-aug}.
    """
    lw = linewidth
    size = 3

    # (sweep_key, region, color, marker, linestyle, label)
    series = [
        ('PointwiseNoAug',    'nearwall', c_nw, 'o', '--', 'Pointwise, no aug (NW)'),
        ('PointwiseAug',      'nearwall', c_nw, 'o', '-',  'Pointwise, aug (NW)'),
        ('CNNnoaug',          'nearwall', c_nw, 's', '--', 'CNN, no aug (NW)'),
        ('CNNaug',            'nearwall', c_nw, 's', '-',  'CNN, aug (NW)'),
        ('PointwiseNoAug_md', 'middle',   c_md, 'o', '--', 'Pointwise, no aug (M)'),
        ('PointwiseAug_md',   'middle',   c_md, 'o', '-',  'Pointwise, aug (M)'),
        ('CNNnoaug_md',       'middle',   c_md, 's', '--', 'CNN, no aug (M)'),
        ('CNNaug_md',         'middle',   c_md, 's', '-',  'CNN, aug (M)'),
    ]

    fig, ax = plt.subplots(figsize=(6, 3))

    for sweep_key, region, color, marker, ls, label in series:
        if sweep_key not in data_dfs:
            continue
        df = data_dfs[sweep_key]
        equiv_col = f'Re1000_{region}_val_equiv_error'
        if equiv_col not in df.columns:
            continue
        # Restrict to rows where the training prefix matches the region
        prefix_col = 'dataset_input_prefix'
        if prefix_col in df.columns:
            df = df[df[prefix_col].astype(str).str.contains(region, na=False)].copy()
        df = df.dropna(subset=[DATA_FRAC_COL, equiv_col]).sort_values(DATA_FRAC_COL)
        if df.empty:
            continue
        n_points = df[DATA_FRAC_COL] * N_TOTAL_POINTS
        ax.plot(n_points, df[equiv_col],
                marker=marker, color=color, linestyle=ls,
                markerfacecolor=color if ls == '-' else 'none',
                markersize=size, linewidth=lw, markeredgewidth=lw, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Number of training points, $N_\mathrm{train}$')
    ax.set_ylabel(r'Equivariance error, $\mathcal{E}_\mathrm{equiv}$')
    _style_legend(ax, fontsize=6, loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(PLOT_OUTPUT_DIR, 'implicit_augmentation.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved implicit augmentation plot to {out_path}")


def main():
    escnn_results = load_escnn_sweep()
    cnn_results = load_cnn_sweep()
    eigenframe_results, pointwise_results = load_pointwise_sweeps()
    data_dfs = load_data_sweeps()
    plot_pointwise(eigenframe_results, pointwise_results, data_dfs)
    plot_nonlocal(cnn_results, escnn_results, data_dfs)
    plot_implicit_augmentation(data_dfs)


if __name__ == '__main__':
    main()