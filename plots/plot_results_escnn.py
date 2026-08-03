import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
ESCNN_SWEEP_DIR = 'outputs/ESCNN_params'


def load_escnn_sweep():
    csv_path = os.path.join(ESCNN_SWEEP_DIR, 'sweep_results.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find sweep results at {csv_path}"
        )

    return pd.read_csv(csv_path)


def plot_escnn_by_depth_and_width(results: pd.DataFrame):
    """
    Plot ESCNN sweep results grouping by network depth
    (model_n_layers) and sweeping over number of regular
    representations.

    - Left panel:
        Re1000_nearwall_val_loss vs n_reg_repr,
        one line per depth.

    - Right panel:
        Re1000_nearwall_val_loss vs param_count (log-log),
        one line per depth.
    """

    required_cols = [
        'model_n_layers',
        'model_n_reg_repr',
        'Re1000_nearwall_val_loss',
        'param_count',
    ]

    missing = [c for c in required_cols if c not in results.columns]

    if missing:
        raise ValueError(
            f"Missing columns in ESCNN sweep_results.csv: {missing}"
        )

    df = results.dropna(subset=required_cols).copy()

    df['depth'] = df['model_n_layers'].astype(int)
    df['width'] = df['model_n_reg_repr'].astype(int)

    unique_depths = sorted(df['depth'].unique())

    if len(unique_depths) == 0:
        raise ValueError(
            "No ESCNN results with model_n_layers found "
            "in sweep_results.csv"
        )

    colors = plt.cm.viridis_r(
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

        # Left: val loss vs regular representation width
        ax[0].plot(
            sub['width'],
            sub['Re1000_nearwall_val_loss'],
            'o-',
            linewidth=0.5,
            markersize=3,
            label=label,
            color=color,
        )

        # Right: val loss vs parameter count
        ax[1].plot(
            sub['param_count'],
            sub['Re1000_nearwall_val_loss'],
            'o-',
            linewidth=0.5,
            markersize=3,
            label=label,
            color=color,
        )

    ax[0].set_xlabel('Number of Regular Representations')
    ax[0].set_ylabel('Validation Loss')
    ax[0].set_yscale('log')

    ax[1].set_xscale('log')
    ax[1].set_xlabel('Number of Parameters')

    for a in ax:
        leg = a.legend(
            title='ESCNN depth',
            fancybox=False,
            edgecolor='black',
        )
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
        'escnn_sweep_by_nlayers.pdf'
    )

    fig.savefig(out_path, bbox_inches='tight')

    print(
        f"Saved ESCNN sweep plot "
        f"(one line per depth over width) to {out_path}"
    )


def main():
    results = load_escnn_sweep()
    plot_escnn_by_depth_and_width(results)


if __name__ == '__main__':
    main()