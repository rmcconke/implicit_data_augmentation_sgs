"""Slopes and extrapolated crossing of the Fig. 3 equivariance-error lines.

For the CNN with and without augmentation in each training region, fits a line in
log-log space through the end points (training fractions 0.1 and 1.0) of the
equivariance error against training-set size, and extrapolates where the no-aug and
aug lines would meet. Reports this both for the paper's Frobenius/tau_ref
normalization and for the stored five-component normalization. These are the Fig. 3
slopes and the extrapolated crossing quoted in Section 3.1. Reads the CSV written by
analysis/equiv_error_frob.py; CPU, instant.

    .venv/bin/python analysis/fig3_slopes.py [--csv analysis/results/equiv_error_frob.csv]
"""
import argparse
import csv
import math
import os

import _common

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--csv', default=os.path.join(_common.RESULTS, 'equiv_error_frob.csv'))
args = parser.parse_args()

rows = list(csv.DictReader(open(args.csv)))
print(sorted(set(r['sweep'] for r in rows)))


def series(sweep, region, col='frob'):
    rr = [r for r in rows if r['sweep'] == sweep]
    key = f'Re1000_{region}_val_equiv_error_frob'
    if col == 'euclid':
        return sorted((float(r['train_fraction']), float(r[key]) * float(r['tau_frob']) / float(r['tau_euclid'])) for r in rr)
    return sorted((float(r['train_fraction']), float(r[key])) for r in rr)


def endpoint(p):
    (f0, e0), (f1, e1) = p[0], p[-1]
    return math.log10(e1 / e0) / math.log10(f1 / f0), e1


for region, na, au in [('middle', 'CNNnoaug_data_middle', 'CNNaug_data_middle'), ('nearwall', 'CNNnoaug_data', 'CNNaug_data')]:
    for col in ('frob', 'euclid'):
        a, b = series(na, region, col), series(au, region, col)
        if not a or not b:
            print(region, col, 'missing', na, au)
            continue
        sa, ea = endpoint(a)
        sb, eb = endpoint(b)
        # Lines anchored at f = 1: log e = log e1 + s log x; they meet where log ea + sa L = log eb + sb L.
        L = math.log10(eb / ea) / (sa - sb) if sa != sb else float('nan')
        print(f'{region:8s} {col:6s} no-aug slope {sa:.4f} aug slope {sb:.4f}  E(f=1) {ea:.4f} vs {eb:.4f}  '
              f'meet at {10**L:.1f}x data (log10 {L:.2f})  [{len(a)},{len(b)} pts]')
