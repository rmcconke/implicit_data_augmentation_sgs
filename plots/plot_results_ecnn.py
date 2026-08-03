import os
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIXGeneral']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

linewidth = 0.5
plt.rcParams['axes.linewidth'] = linewidth
plt.rcParams['xtick.major.width'] = linewidth
plt.rcParams['ytick.major.width'] = linewidth


PLOT_OUTPUT_DIR = 'outputs/plots_final'
ECNN_SWEEP_DIR = 'outputs/ECNN_params'


def load_ecnn_sweep():
    csv_path = os.path.join(ECNN_SWEEP_DIR, 'sweep_results.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find sweep results at {csv_path}")
    return pd.read_csv(csv_path)


def _parse_irrep_width(s):
    """
    Convert strings like:
        '16x0e + 16x1e + 16x2e'
    into:
        16
    """
    try:
        first = str(s).split('+')[0].strip()
        return int(first.split('x')[0])
    except Exception:
        return None


def plot_ecnn_by_depth_and_width(results: pd.DataFrame):
    """
    Plot ECNN sweep results grouping by network depth (model_n_layers)
    and sweeping over irrep width.

    - Left panel:
        Re1000_nearwall_val_loss vs irrep width,
        one line per depth.

    - Right panel:
        Re1000_nearwall_val_loss vs param_count (log-log),
        one line per depth.
    """
    required_cols = [
        'model_n_layers',
        'model_hidden_irreps',
        'Re1000_nearwall_val_loss',
        'param_count',
    ]

    missing = [c for c in required_cols if c not in results.columns]
    if missing:
        raise ValueError(f"Missing columns in ECNN sweep_results.csv: {missing}")

    df = results.dropna(subset=required_cols).copy()

    parsed = df['model_hidden_irreps'].apply(_parse_irrep_width)
    df = df[parsed.notnull()].copy()

    df['width'] = parsed[parsed.notnull()].astype(int)
    df['depth'] = df['model_n_layers'].astype(int)

    df = df.dropna(subset=['width'])

    unique_depths = sorted(df['depth'].unique())

    if len(unique_depths) == 0:
        raise ValueError(
            "No valid ECNN results with parsable hidden irreps found in sweep_results.csv"
        )

    colors = plt.cm.plasma(
        np.linspace(0.1, 0.9, len(unique_depths))
    )

    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

    for idx, depth in enumerate(unique_depths):

        sub = df[df['depth'] == depth].copy()
        sub = sub.sort_values('width')

        if sub.empty:
            continue

        color = colors[idx]
        label = f'{depth}'

        # Left: val loss vs irrep width
        ax[0].plot(
            sub['width'],
            sub['Re1000_nearwall_val_loss'],
            'o-',
            linewidth=0.5,
            markersize=3,
            label=label,
            color=color,
        )

        # Right: val loss vs param count
        ax[1].plot(
            sub['param_count'],
            sub['Re1000_nearwall_val_loss'],
            'o-',
            linewidth=0.5,
            markersize=3,
            label=label,
            color=color,
        )

    ax[0].set_xlabel('Irrep Width')
    ax[0].set_ylabel('Validation Loss')
    ax[0].set_yscale('log')

    ax[1].set_xscale('log')
    ax[1].set_xlabel('Number of Parameters')

    for a in ax:
        leg = a.legend(title='ECNN depth', fancybox=False, edgecolor='black')
        leg.get_frame().set_linewidth(linewidth)

    # Panel labels
    ax[0].text(
        0.5,
        -0.18,
        '(a)',
        transform=ax[0].transAxes,
        ha='center',
        va='top'
    )

    ax[1].text(
        0.5,
        -0.18,
        '(b)',
        transform=ax[1].transAxes,
        ha='center',
        va='top'
    )

    fig.tight_layout()

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    out_path = os.path.join(
        PLOT_OUTPUT_DIR,
        'ecnn_sweep_by_depth_width.pdf'
    )

    fig.savefig(out_path, bbox_inches='tight')

    print(
        f"Saved ECNN sweep plot (one line per depth over width) to {out_path}"
    )


def main():
    results = load_ecnn_sweep()
    plot_ecnn_by_depth_and_width(results)


if __name__ == '__main__':
    main()