"""Kolmogorov length of each extraction region from the Lee & Moser channel budgets.

Reads the turbulent kinetic energy budget profiles of Lee & Moser (2015) at
Re_tau = 1000 and 5200, takes the viscous dissipation column (epsilon+, stored
positive), and reports eta+ = (epsilon+)^(-1/4) and Delta/eta = Delta+ / eta+ with
Delta+ = 40 at the centre of each extraction box (near-wall: y+ = 321; channel
centre: y+ = Re_tau), together with the range of eta+ across the box. This produces
the eta+ and Delta/eta columns of Table 1. The profile files are not part of this
repository; download LM_Channel_1000_RSTE_k_prof.dat and LM_Channel_5200_RSTE_k_prof.dat
from https://turbulence.oden.utexas.edu/channel2015/data/ into --data-dir. CPU, instant.

    .venv/bin/python analysis/eta.py [--data-dir analysis/data]
"""
import argparse
import os

import numpy as np

import _common

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--data-dir', default=os.path.join(_common.REPO, 'analysis', 'data'),
                    help='directory holding LM_Channel_{1000,5200}_RSTE_k_prof.dat')
args = parser.parse_args()

DELTA_PLUS = 40
trapezoid = getattr(np, 'trapezoid', None) or np.trapz
# Columns: y/delta, y+, Production, Turb_Transport, Visc_Transport, Pressure_Strain,
# Pressure_Transport, Viscous_Dissipation, Balance. Regions: (name, y+ min, y+ max, y+ centre).
REGIONS = {
    1000: [('near-wall', 1, 641, 321), ('center', 679, 1000, 1000)],
    5200: [('near-wall', 1, 641, 321), ('center', 4866, 5200, 5200)],
}
for re_tau, regions in REGIONS.items():
    d = np.loadtxt(os.path.join(args.data_dir, f'LM_Channel_{re_tau}_RSTE_k_prof.dat'), comments='%')
    y, eps = d[:, 1], np.abs(d[:, 7])
    print(re_tau, 'eps > 0 everywhere:', eps.min() > 0, '| y+ max in file', y.max())
    for name, lo, hi, c in regions:
        m = (y >= lo) & (y <= hi)
        em = trapezoid(eps[m], y[m]) / (y[m][-1] - y[m][0])
        ec = np.interp(c, y, eps)
        print(f'  {name} y+ {lo}-{hi}: eta+ at centre y+={c}: {ec**-0.25:.2f} (Delta/eta {DELTA_PLUS*ec**0.25:.1f}); '
              f'eta+ of box-mean eps {em**-0.25:.2f} (Delta/eta {DELTA_PLUS*em**0.25:.1f}); '
              f'eta+ over box {eps[m].max()**-0.25:.2f} to {eps[m].min()**-0.25:.2f}')
