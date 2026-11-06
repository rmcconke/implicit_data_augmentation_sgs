# Rotational equivariance and locality in data-driven subgrid-scale closures

Code, configurations and trained models for the paper of the same title. The models are trained with a rotation-invariant loss (the mean squared error of the orthonormal-basis coefficients of the deviatoric SGS stress, `src/deviatoric.py`); these runs have barcodes starting with `26_09`. Runs with earlier barcodes (`26_02` to `26_08`) in `outputs/`, and the folders `outputs/test`, `outputs/march18_escnn` and the `outputs/26_06_05_*` runs, are from earlier development with the non-invariant loss and are kept for reference; no result in the paper uses them.

We used `uv`. To set up the environment and install requirements:
```
uv sync
```

# To get Dataset:
We have included scripts we used to download data from the Johns Hopkins Turbulence Database. See the `dataset_creation/` folder. The data paths in `src/global_config.py`, `dataset_creation/` and the run configs point to our cluster and must be edited for another machine.

# Configurations
`configs/sweeps/` is the authoritative set of configurations for the models in the paper (parameter sweeps, data-size sweeps, seeds, filter ratios and the box-mean normalization control), with `configs/middle/` for the channel-centre models of Table 3. Sweeps are run with `src/sweep.py` and single models with `src/train.py`; `run_paper_sweeps.sh` lists the commands used.

The depth sweeps of Appendix A (Figs. 8 and 9) were not stored. They can be reproduced by changing the depth (`n_layers` for the ESCNN, the length of `hidden_channels` for the CNN) in the parameter sweep configurations.

# Tests
`python -m pytest tests/`. The memorization and training-loop tests need a GPU and are skipped without one.

# Model weights:
See the `outputs/` folder for the trained models, one directory per run with its config, training log, checkpoint and `eval_results.npy`. There are also intermediate plots in these folders. Each sweep folder has a `sweep_results.csv` aggregated from the `26_09` runs with `src/analyze_sweep.py <sweep folder> --prefix 26_09`.

The two largest ESCNN checkpoints (2,955,456 and 740,448 parameters; `outputs/ESCNN_params/*moved_bison*` and `outputs/ESCNN_params/*tops_bison*`, 848 MB and 247 MB) exceed GitHub's file size limit and are not included. Their configs, logs and evaluation results are included, and the checkpoints are available on request.

# Reproducing the tables and figures
`analysis/README.md` maps each table, figure and quoted number in the paper to the script and command that produces it.

# License
MIT, see `LICENSE`.
