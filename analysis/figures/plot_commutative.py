"""Panels of Fig. 4 (commutative diagram) from the invariant-loss data-sweep CNNs.

Runs plots/plot_commutative.py, whose default run directories are the full-data
(training fraction 1.0) augmented and non-augmented CNNs of the 26_09 data-size
sweeps (near-wall: humane_cod, daring_burro; channel centre: witty_krill,
proud_maggot), and writes the individual raster panels (x, g.x, f(x), f(g.x), g.f(x)
and |f(g.x) - g.f(x)| for tau^d_33 on the z = 7 slice) and the colour bars as PDFs.
Fig. 4 uses the non-augmented panels; analysis/figures/build_commutative.py places them
into the vector layout of the figure. Needs one test box per region; CPU or GPU, seconds.

    .venv/bin/python analysis/figures/plot_commutative.py [--out analysis/results/figures/commutative]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--out', default=None)
parser.add_argument('--device', default=None)
args = parser.parse_args()
out = os.path.abspath(args.out) if args.out else _common.results_dir('figures', 'commutative')

_common.setup()
pc = _common.load_plots_module('plot_commutative')
pc.OUT_DIR = out
if args.device:
    pc.DEVICE = args.device
pc.make_commutative_panels()
