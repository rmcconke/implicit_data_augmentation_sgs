"""How flat the validation loss is near the best epoch, from the training logs.

For five Table 3 models, parses the "Saving model from epoch N (val loss: L)" lines of
the training log and reports the best validation loss together with the best loss
reached 500, 2,000 and 5,000 epochs before the best epoch, as a percentage above the
best. This is a supporting check of the early-stopping protocol of Appendix B; its
numbers are not quoted in the paper. CPU, instant.

    .venv/bin/python analysis/plateau.py
"""
import argparse
import glob
import re

import _common

argparse.ArgumentParser(description=__doc__.split('\n')[0]).parse_args()
_common.setup()

RUNS = {
    'NW CNN-aug secure_fish':  _common.TABLE3_NEARWALL['CNN (aug)'],
    'C CNN-aug moral_kiwi':    _common.TABLE3_MIDDLE['CNN (aug)'],
    'C MLP-aug comic_possum':  _common.TABLE3_MIDDLE['MLP (aug)'],
    'NW ESCNN holy_osprey':    _common.TABLE3_NEARWALL['ESCNN'],
    'NW MLP-aug intent_sole':  _common.TABLE3_NEARWALL['MLP (aug)'],
}
for name, pat in RUNS.items():
    d = _common.one(pat)
    log = glob.glob(d + '/*_training_log.txt')[0]
    best = [(int(e), float(v)) for e, v in re.findall(r'Saving model from epoch (\d+) \(val loss: ([0-9.eE-]+)\)', open(log).read())]
    eb, vb = best[-1]

    def best_at(ep):
        c = [v for e, v in best if e <= ep]
        return c[-1] if c else None

    span = {k: best_at(eb - k) for k in (500, 2000, 5000) if eb - k > 0}
    print(f'{name:24s} best epoch {eb:6d} val {vb:.4f} | '
          + ' | '.join(f'{k} epochs earlier {v:.4f} ({100*(v-vb)/vb:.2f}% higher)' for k, v in span.items()))
