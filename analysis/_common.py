"""Shared paths and run selections for the scripts in analysis/.

Every script changes into the repository root (so the run directories under
outputs/ and the dataset paths in the run configs resolve as they did during
training), puts src/ on the import path, and writes under analysis/results/
unless told otherwise.
"""
import glob
import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, 'analysis', 'results')

# Table 3 models (rotation-invariant loss, barcodes 26_09*).
TABLE3_NEARWALL = {
    'Eigenframe': 'outputs/pointwiseEigenFrame_params/26_09*meet_collie*',
    'MLP (aug)':  'outputs/pointwiseAug_params/26_09*intent_sole*',
    'ESCNN':      'outputs/ESCNN_params/26_09*holy_osprey*',
    'CNN (aug)':  'outputs/CNNaug_params/26_09*secure_fish*',
}
TABLE3_MIDDLE = {
    'Eigenframe': 'outputs/table3_middle/26_09*merry_condor*',
    'MLP (aug)':  'outputs/table3_middle/26_09*comic_possum*',
    'ESCNN':      'outputs/table3_middle/26_09*choice_lynx*',
    'CNN (aug)':  'outputs/table3_middle/26_09*moral_kiwi*',
}


def setup():
    """chdir to the repository root and make src/ importable."""
    os.chdir(REPO)
    src = os.path.join(REPO, 'src')
    if src not in sys.path:
        sys.path.insert(0, src)


def results_dir(*parts):
    path = os.path.join(RESULTS, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def one(pattern):
    """The single run directory matching a glob pattern (relative to the repo root)."""
    matches = sorted(glob.glob(os.path.join(REPO, pattern)))
    assert len(matches) == 1, (pattern, matches)
    return os.path.relpath(matches[0], REPO)


def load_plots_module(name):
    """Import plots/<name>.py (plots/ is not a package)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, 'plots', f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_device():
    import torch
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'
