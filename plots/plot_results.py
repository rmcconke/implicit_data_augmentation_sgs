import os
import pandas as pd
import matplotlib.pyplot as plt
#from plot_example_SR import visualize_example_predictions, visualize_equivariance, visualize_example_fields
import torch
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import matplotlib.image as mpimg
import ast

import sys
sys.path.append('src')
from analyze_sweep import analyze_sweep
import json

torch.set_default_dtype(torch.float32)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['STIXGeneral']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 9  # Base font size
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
linewidth = 0.5
major_gridlines = 0.5
minor_gridlines = 0.5  
plt.rcParams['axes.linewidth'] = linewidth
plt.rcParams['xtick.major.width'] = linewidth
plt.rcParams['ytick.major.width'] = linewidth

c_eig = 'tab:orange'
c_pnt = 'tab:red'
c_cnn = 'tab:blue'
c_ecn = 'tab:purple'
c_escnn = 'tab:green'

PLOT_OUTPUT_DIR = 'outputs/plots_final'

def get_best_model_dir(sweep_dir):
    with open(os.path.join(sweep_dir, 'best_results.json'), 'r') as f:
        best_results = json.load(f)

    barcode = best_results['model_barcode']
    return os.path.join(sweep_dir, barcode)

eigenframe_sweep_dir = 'outputs/pointwiseEigenFrame_params'
pointwise_sweep_dir = 'outputs/pointwiseAug_params'
cnn_sweep_dir = 'outputs/CNN4layer'
# ecnn_sweep_dir = 'outputs/ECNN_params'
escnn_sweep_dir = 'outputs/ESCNN4layer'

#if not os.path.exists(os.path.join(eigenframe_sweep_dir, 'sweep_results.csv')):
#analyze_sweep(eigenframe_sweep_dir)
#if not os.path.exists(os.path.join(pointwise_sweep_dir, 'sweep_results.csv')):
#analyze_sweep(pointwise_sweep_dir)
#if not os.path.exists(os.path.join(cnn_sweep_dir, 'sweep_results.csv')):
analyze_sweep(cnn_sweep_dir)
#if not os.path.exists(os.path.join(ecnn_sweep_dir, 'sweep_results.csv')):
#analyze_sweep(ecnn_sweep_dir)
#if not os.path.exists(os.path.join(escnn_sweep_dir, 'sweep_results.csv')):
analyze_sweep(escnn_sweep_dir)

#eigenframe_model_dir = get_best_model_dir(eigenframe_sweep_dir)
#pointwise_model_dir = get_best_model_dir(pointwise_sweep_dir)
cnn_model_dir = get_best_model_dir(cnn_sweep_dir)
# ecnn_model_dir = get_best_model_dir(ecnn_sweep_dir)
escnn_model_dir = get_best_model_dir(escnn_sweep_dir)




#eigenframe_sweep_results = pd.read_csv(os.path.join(eigenframe_sweep_dir, 'sweep_results.csv'))
#pointwise_sweep_results = pd.read_csv(os.path.join(pointwise_sweep_dir, 'sweep_results.csv'))
cnn_sweep_results = pd.read_csv(os.path.join(cnn_sweep_dir, 'sweep_results.csv'))
# ecnn_sweep_results = pd.read_csv(os.path.join(ecnn_sweep_dir, 'sweep_results.csv'))
escnn_sweep_results = pd.read_csv(os.path.join(escnn_sweep_dir, 'sweep_results.csv'))


def _parse_hidden_channels(s):
    """Convert string like \"[16, 16, 16, 16]\" to list[int]."""
    try:
        lst = ast.literal_eval(str(s))
        if isinstance(lst, (list, tuple)) and all(isinstance(x, (int, float)) for x in lst):
            return [int(x) for x in lst]
    except Exception:
        pass
    return None


# Filter CNN and ESCNN sweeps to 4-layer models only
cnn_parsed = cnn_sweep_results['model_hidden_channels'].apply(_parse_hidden_channels)
cnn_sweep_results = cnn_sweep_results[cnn_parsed.notnull()].copy()
cnn_sweep_results['hidden_list'] = cnn_parsed[cnn_parsed.notnull()]
cnn_sweep_results['depth'] = cnn_sweep_results['hidden_list'].apply(len)
cnn_sweep_results_4 = cnn_sweep_results[cnn_sweep_results['depth'] == 4].copy()

escnn_sweep_results_4 = escnn_sweep_results[escnn_sweep_results['model_n_layers'] == 4].copy()
# Parameter efficiency plots
lw = 0.5
size = 3


fig, ax = plt.subplots(2,3, figsize=(6, 4))

#ax[0,0].plot(eigenframe_sweep_results['param_count'], eigenframe_sweep_results['Re1000_nearwall_val_loss'], 'o-', color=c_eig, linewidth=lw, markersize=size)
# ax[0,0].plot(pointwise_sweep_results['param_count'], pointwise_sweep_results['Re1000_nearwall_val_loss'], 'o-', color=c_pnt, linewidth=lw, markersize=size)

#ax[0,1].plot(eigenframe_sweep_results['time_10M_points'], eigenframe_sweep_results['Re1000_nearwall_val_loss'], 'o-', color=c_eig, linewidth=lw, markersize=size)
# ax[0,1].plot(pointwise_sweep_results['time_10M_points'], pointwise_sweep_results['Re1000_nearwall_val_loss'], 'o-', color=c_pnt, linewidth=lw, markersize=size)
ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')
ax[0,1].set_xlabel('Wall Clock Time (s)')
ax[0,0].set_ylabel('Validation Loss')
#ax[0,0].set_xlabel('Number of parameters')
ax[0,0].set_xscale('log')
ax[0,0].set_yscale('log')



ax[1,0].plot(cnn_sweep_results_4['param_count'], cnn_sweep_results_4['Re1000_nearwall_val_loss'], 'o-', color=c_cnn, linewidth=lw, markersize=size)
# ax[1,0].plot(ecnn_sweep_results['param_count'], ecnn_sweep_results['Re1000_nearwall_val_loss'], 'o-', color=c_ecn, linewidth=lw, markersize=size)
ax[1,0].plot(escnn_sweep_results_4['param_count'], escnn_sweep_results_4['Re1000_nearwall_val_loss'], 'o-', color=c_escnn, linewidth=lw, markersize=size)
ax[1,0].set_xscale('log')
ax[1,0].set_yscale('log')
ax[1,0].set_xlabel('Number of parameters')
ax[1,0].set_ylabel('Validation Loss')

ax[1,1].plot(cnn_sweep_results_4['time_10M_points'], cnn_sweep_results_4['Re1000_nearwall_val_loss'], 'o-', color=c_cnn, linewidth=lw, markersize=size)
# ax[1,1].plot(ecnn_sweep_results['time_10M_points'], ecnn_sweep_results['Re1000_nearwall_val_loss'], 'o-', color=c_ecn, linewidth=lw, markersize=size)
ax[1,1].plot(escnn_sweep_results_4['time_10M_points'], escnn_sweep_results_4['Re1000_nearwall_val_loss'], 'o-', color=c_escnn, linewidth=lw, markersize=size)
ax[1,1].set_xscale('log')
ax[1,1].set_yscale('log')
ax[1,1].set_xlabel('Wall clock time for 10M points(s)')



for i in range(2):
    for j in range(3):
        if j == 0:
            ax[i, j].tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False)
        elif j == 1:
            ax[i, j].tick_params(axis='y', left=True, right=False, labelleft=False, labelright=False)
            ax[i, j].set_yticklabels([])
            ax[i, j].yaxis.set_minor_formatter(plt.NullFormatter())
        else:
            ax[i, j].tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
            ax[i, j].set_yticklabels([])
            ax[i, j].yaxis.set_minor_formatter(plt.NullFormatter())

#ax[1,1].set_ylabel('Validation Loss')

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color=c_eig, marker='o', label='Eigenframe'),
    # Line2D([0], [0], color=c_pnt, marker='o', label='Pointwise'),
    Line2D([0], [0], color=c_cnn, marker='o', label='CNN'),
    # Line2D([0], [0], color=c_ecn, marker='o', label='ECNN'),
]

leg = fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
           bbox_to_anchor=(0.5, -0.1), frameon=True,
           edgecolor='black', fancybox=False)

leg.get_frame().set_linewidth(0.5)



fig.savefig(os.path.join(PLOT_OUTPUT_DIR, 'parameter_efficiency.pdf'), bbox_inches='tight')


def load_train_val_losses(model_dir):
    model_barcode = os.path.basename(model_dir)
    checkpoint = torch.load(os.path.join(model_dir, f'{model_barcode}_checkpoint.pt'))
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    return train_losses, val_losses

