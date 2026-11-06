"""ESCNN/CNN validation-loss crossover and matched-time loss ratios (Figs. 6a, 6b).

Reads the stored eval_results.npy of every 26_09 near-wall run of the ESCNN and
augmented-CNN parameter sweeps, interpolates log validation loss against log
parameter count, and reports the parameter count where the two curves cross. It then
interpolates the CNN loss at the inference time of each ESCNN model and prints the
ESCNN/CNN loss ratio at matched wall-clock time. These are the crossover parameter
count and matched-time ratios quoted in Section 3.2. CPU, instant.

    .venv/bin/python analysis/crossover.py
"""
import argparse
import glob
import math
import re

import numpy as np

import _common

argparse.ArgumentParser(description=__doc__.split('\n')[0]).parse_args()
_common.setup()


def runs(sweep):
    out = []
    for d in sorted(glob.glob(f'outputs/{sweep}/26_09*nearwall*')):
        e = np.load(glob.glob(d + '/*eval_results.npy')[0], allow_pickle=True).item()
        log = open(glob.glob(d + '/*_training_log.txt')[0]).read()
        v = [float(x) for x in re.findall(r'Saving model from epoch \d+ \(val loss: ([0-9.eE-]+)\)', log)][-1]
        out.append((e['param_count'], e['time_10M_points'], e['Re1000_nearwall_val']['loss'], v))
    return sorted(out)


E, C = runs('ESCNN_params'), runs('CNNaug_params')
print('ESCNN', [(p, round(l, 6)) for p, t, l, v in E])
print('CNN  ', [(p, round(l, 6)) for p, t, l, v in C])


def lp(r):
    return np.log10([x[0] for x in r]), np.log10([x[2] for x in r])


ex, ey = lp(E)
cx, cy = lp(C)
grid = np.linspace(max(ex.min(), cx.min()), min(ex.max(), cx.max()), 2000)
d = np.interp(grid, ex, ey) - np.interp(grid, cx, cy)
idx = np.where(np.diff(np.sign(d)) != 0)[0]
print('crossover (val loss, log-log interp) at params:', [round(10 ** grid[i]) for i in idx])


def lt(r):
    return np.log10([x[1] for x in r]), np.log10([x[2] for x in r])


ct, cy2 = lt(C)
for (p, t, l, v) in E:
    if ct.min() <= math.log10(t) <= ct.max():
        lc = 10 ** np.interp(math.log10(t), ct, cy2)
        print(f'ESCNN {p:>9d} time {t:.4f}s loss {l:.6f}  CNN at same time {lc:.6f}  ESCNN/CNN {l/lc:.3f}')
    else:
        print(f'ESCNN {p:>9d} time {t:.4f}s outside CNN time range [{10**ct.min():.4f},{10**ct.max():.4f}]')
