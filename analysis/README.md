# Reproducing the paper's tables, figures and numbers

This folder holds the scripts that turn the trained models in `outputs/` into the tables, figures and quoted numbers of the paper. All of them use the `26_09` runs (rotation-invariant loss). Run every command from the repository root with the project environment (`.venv/bin/python`, or `uv run python`). Each script has a docstring that says what it computes and which table or figure it produces, and `--help` lists its options.

Outputs go to `analysis/results/` by default (ignored by git, since everything there can be regenerated). Scripts that read model predictions need the filtered JHTDB datasets built with `dataset_creation/` at the paths in the run configs. Scripts marked GPU were run on one NVIDIA L40S or H100 through Slurm; the others run on a CPU in seconds to minutes.

## Order of the steps

Three scripts write files that others read:

```
.venv/bin/python analysis/equiv_error_frob.py        # -> analysis/results/equiv_error_frob.csv
.venv/bin/python analysis/figures/inventory.py       # -> analysis/results/figures/inventory.csv
.venv/bin/python analysis/clark_metrics.py all       # GPU -> analysis/results/clark/*.json
```

`fig3_slopes.py`, `figures/plot_sweeps.py` and `figures/check_numbers.py` read the first two, and `clark_report.py` reads the third.

## Map from the paper to the scripts

| Paper item | Script | Command |
|---|---|---|
| Table 1, $\eta^+$ and $\Delta/\eta$ columns | `analysis/eta.py` | `python analysis/eta.py --data-dir analysis/data` (see the Lee & Moser data below) |
| Table 1, barycentric coordinates | `plots/compute_barycentric.py` | `python plots/compute_barycentric.py` |
| Table 3, trained models ($\rho_\tau$) | stored in each run's `*_eval_results.npy` by `src/eval_model.py` at the end of training | print with `python plots/plot_sweep_results.py` (prints the Table 3 values first); re-evaluate one run with `python src/eval_model.py <run dir>` |
| Table 3, Clark row | `analysis/clark_metrics.py` | `python analysis/clark_metrics.py clark` (GPU), then `python analysis/clark_report.py` |
| Table 4 (RMS ratio and NMSE, all rows) | `analysis/clark_metrics.py`, `analysis/clark_report.py` | `python analysis/clark_metrics.py all` (GPU), then `python analysis/clark_report.py` |
| Table 5 (SGS dissipation), trained models | `src/analyze_dissipation.py` | `python src/analyze_dissipation.py <run dir>` for each run in `dissipation_targets.txt` (GPU; `sbatch sbatch_dissipation.sh`) |
| Table 5, Clark row | `analysis/clark_metrics.py` | as for Table 4 |
| Table 6 (filter width transfer) | `src/analyze_cross_ratio.py` | `python src/analyze_cross_ratio.py <run dir>` for each run in `xratio_targets.txt` (GPU; `sbatch sbatch_cross_ratio.sh`); needs the fs2 and fs8 datasets from `dataset_creation/sgs_filter_ratios.py` |
| Table 7 (per-component correlation), trained models | `src/analyze_components.py` | `python src/analyze_components.py <run dir>` for each run in `component_targets.txt` (GPU; `sbatch sbatch_components.sh`) |
| Table 7, Clark row | `analysis/clark_metrics.py` | as for Table 4 |
| Fig. 3 (equivariance error against training-set size) | `analysis/equiv_error_frob.py`, `analysis/figures/plot_sweeps.py` | `python analysis/equiv_error_frob.py`, then `python analysis/figures/plot_sweeps.py` (writes `implicit_augmentation.pdf`) |
| Fig. 3 slopes and extrapolated crossing (Section 3.1) | `analysis/fig3_slopes.py`, `analysis/figures/check_numbers.py` | `python analysis/fig3_slopes.py`; `python analysis/figures/check_numbers.py` |
| $\mathcal{E}_\mathrm{equiv}$ values quoted in the text, including the equivariant models at round-off | `analysis/equiv_error_frob.py` | `python analysis/equiv_error_frob.py` (one row per run in the CSV) |
| Fig. 4 (commutative diagram) | `analysis/figures/plot_commutative.py`, `analysis/figures/build_commutative.py` | `python analysis/figures/plot_commutative.py` writes the panels; `python analysis/figures/build_commutative.py --template <existing commutative.pdf>` swaps them into the vector layout (needs `pikepdf` and `pillow`) |
| Figs. 5 and 6 (pointwise and nonlocal sweeps) | `analysis/figures/plot_sweeps.py` | `python analysis/figures/plot_sweeps.py` (writes `pointwise_sweep.pdf` and `nonlocal_sweep.pdf`). `python plots/plot_sweep_results.py` gives the same two figures from the per-sweep `sweep_results.csv` files |
| ESCNN/CNN crossover (about $7 \times 10^5$ parameters) and loss ratios at matched wall-clock time (Section 3.2) | `analysis/crossover.py`, `analysis/figures/check_numbers.py` | `python analysis/crossover.py`; `python analysis/figures/check_numbers.py` |
| Other Section 3.2 numbers (loss reduction across the ESCNN sweep, data-size sweep slopes) | `analysis/figures/check_numbers.py` | `python analysis/figures/check_numbers.py` |
| Inference times in Figs. 5(b) and 6(b) | stored as `time_10M_points` in each run's `*_eval_results.npy` by `src/eval_model.py` (H100) | `python src/eval_model.py <run dir>` |
| Fig. 7 (predicted stress magnitude on one test box) | `analysis/fig7_per_box.py` (wraps `plots/plot_results_sgs.py`) | `python analysis/fig7_per_box.py --figure` (GPU recommended) |

The depth sweeps of Appendix A (Figs. 8 and 9) were not stored; see the top-level `README.md`.

## Supporting checks

These scripts check claims made in the paper but produce no number that appears in it.

| Check | Script | Command |
|---|---|---|
| How typical the Fig. 7 box is: per-box correlation of every test box, and the percentile rank of the box shown | `analysis/fig7_per_box.py` | `python analysis/fig7_per_box.py` (GPU recommended) |
| Raw five-component metrics change under rotation, the invariant metrics do not | `analysis/invariance_demo.py` | `python analysis/invariance_demo.py` |
| Run-to-run variability of the inference timings | `analysis/timing_repeat.py` | `python analysis/timing_repeat.py` (GPU, about an hour) |
| Flatness of the validation loss before the best epoch (early stopping) | `analysis/plateau.py` | `python analysis/plateau.py` |

## Lee & Moser profiles

`analysis/eta.py` reads the turbulent kinetic energy budgets of the Lee & Moser (2015) channel flow DNS, which are not included in this repository. Download `LM_Channel_1000_RSTE_k_prof.dat` and `LM_Channel_5200_RSTE_k_prof.dat` from https://turbulence.oden.utexas.edu/channel2015/data/ into `analysis/data/` (ignored by git), or pass their folder with `--data-dir`.
