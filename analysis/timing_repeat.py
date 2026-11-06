"""Run-to-run variability of the inference timings of Figs. 5(b) and 6(b).

Repeats the timing procedure of src/eval_model.py (50 warmup iterations, the doubling
search for the batch size multiplier, 200 timed iterations, and the eigenframe model
timed on 8 copies of the test set in chunks of 25,000 points) N_REP times for every
26_09 near-wall model of the four parameter sweeps, on the Re_tau = 1000 near-wall
test set, and reports the mean, standard deviation and coefficient of variation in
seconds per 10M points next to the stored time_10M_points. Two differences from
eval_model.py: the forward passes run under torch.no_grad() rather than inside
torch.inference_mode(), and the multiplier search stops at 4096. The repeats came out
0 to 8% slower than the stored values, with a within-job CV below 1%. This is a
supporting check and is not quoted in the paper. Needs a GPU (the paper timings are
on an H100) and the test datasets; about an hour for all 28 models.

    .venv/bin/python analysis/timing_repeat.py [--out analysis/results/timing_repeat.json] [--n-rep 5]
"""
import argparse
import glob
import json
import os
import time

import numpy as np

import _common

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--out', default=None, help='output JSON (default: analysis/results/timing_repeat.json)')
parser.add_argument('--n-rep', type=int, default=5)
parser.add_argument('--sweeps', nargs='+',
                    default=['pointwiseEigenFrame_params', 'pointwiseAug_params', 'ESCNN_params', 'CNNaug_params'])
args = parser.parse_args()
OUT = os.path.abspath(args.out) if args.out else os.path.join(_common.results_dir(), 'timing_repeat.json')

_common.setup()
import torch
from data import NumpyTimeSeriesDataset

prs = _common.load_plots_module('plot_results_sgs')
dev = 'cuda:0'
print(torch.cuda.get_device_name(0), flush=True)

N_WARM, N_RUNS, N_REP, CHUNK = 50, 200, args.n_rep, 25000
ep = prs.EVAL_SETS['spatiotemporal']
inp_files, tgt_files = prs.get_test_file_lists(prs.Box(ep))

def load_inputs(sc, pointwise):
    ds = NumpyTimeSeriesDataset(input_file_list=inp_files, target_file_list=tgt_files, scaling_factors=sc,
                                augmentation_group=None, pointwise=pointwise, precompute_eigenframe_inputs=False, device=dev)
    return ds.get_all_data()[0]

def npts(x): return x.shape[0] if x.ndim == 2 else x.shape[0] * x.shape[2] * x.shape[3] * x.shape[4]

def time_block(model, x, chunk=None):
    with torch.no_grad():
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(N_RUNS):
            if chunk:
                for i in range(0, x.shape[0], chunk): model(x[i:i + chunk])
            else:
                model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_RUNS / npts(x)

out = {}
for sweep in args.sweeps:
    for d in sorted(glob.glob(f'outputs/{sweep}/26_09*nearwall*')):
        model, cfg, sc = prs.load_model(d, dev)
        inputs = load_inputs(sc, cfg.pointwise)
        stored = np.load(glob.glob(d + '/*eval_results.npy')[0], allow_pickle=True).item()
        eig = cfg.model.model_type == 'sgsPointwiseEigenframeMLP'
        if eig:
            x = inputs.repeat(8, *([1] * (inputs.ndim - 1))).to(dev)
            with torch.no_grad():
                for _ in range(N_WARM):
                    for i in range(0, x.shape[0], CHUNK): model(x[i:i + CHUNK])
            reps = [time_block(model, x, CHUNK) for _ in range(N_REP)]; mult = 8
        else:
            mult, best, m = 1, float('inf'), 1
            while True:
                try:
                    torch.cuda.empty_cache(); x = inputs.repeat(m, *([1] * (inputs.ndim - 1))).to(dev)
                    with torch.no_grad():
                        for _ in range(N_WARM): model(x)
                        torch.cuda.synchronize(); t0 = time.perf_counter()
                        for _ in range(10): model(x)
                        torch.cuda.synchronize()
                    pp = (time.perf_counter() - t0) / 10 / npts(x)
                    if pp < best: best, mult = pp, m
                    del x; m *= 2
                    if m > 4096: break
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    torch.cuda.empty_cache(); break
            x = inputs.repeat(mult, *([1] * (inputs.ndim - 1))).to(dev)
            with torch.no_grad():
                for _ in range(N_WARM): model(x)
            reps = [time_block(model, x) for _ in range(N_REP)]
        r = np.array(reps) * 1e7  # seconds per 10M points
        out[os.path.basename(d)] = dict(sweep=sweep, params=int(stored['param_count']), multiplier=mult,
            stored_10M=float(stored['time_10M_points']), reps_10M=r.tolist(), mean_10M=float(r.mean()),
            std_10M=float(r.std(ddof=1)), cv=float(r.std(ddof=1) / r.mean()))
        json.dump(out, open(OUT, "w"), indent=1)
        print("   reps (s per 10M):", [round(x, 5) for x in r.tolist()], flush=True)
        print(f"{sweep:28s} {stored['param_count']:>9d} mult {mult:>5d} stored {stored['time_10M_points']:.4f}s  repeat {r.mean():.4f} +- {r.std(ddof=1):.4f}s (cv {100*r.std(ddof=1)/r.mean():.2f}%)", flush=True)
        del model, x; torch.cuda.empty_cache()
json.dump(out, open(OUT, 'w'), indent=1)
