# GWO-OP-SW: Order-Preserving Grey Wolf Optimizer for SOPWM

<!-- TODO: paste the concept-DOI badge Zenodo gives you after the first release -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

Code, per-run results and analysis scripts accompanying:

> T. Sinlapakorn and P. Singsathid, "Order-Preserving Grey Wolf Optimizer for
> Selective Optimal Pulse-Width Modulation in Multilevel Inverters,"
> *Bulletin of Electrical Engineering and Informatics*. (under review)

Table and figure numbers below refer to the revised manuscript.

## What the method does

Selective optimal pulse-width modulation (SOPWM) requires switching angles in
strict ascending order. A standard grey wolf optimizer searching the angles
directly breaks that order routinely, and its feasibility rate falls to zero as
the number of angles grows. GWO-OP-SW searches a non-negative increment space
and decodes it by a cumulative sum, so the ordering constraints hold by
construction. A square-wave convergence-factor schedule counters the premature
convergence that a monotonic schedule leaves unresolved.

Benchmark: instances RC45-RC50 of the CEC2020 real-world constrained
optimization suite (3- to 13-level inverters, 25-30 switching angles).

## Layout

```
sopwm_objective.py           objective, constraints and instance constants (RC45-RC50)
gwo_op_sw.py                 the proposed algorithm (GWO-OP-SW)
gwo_op_schedules.py          reparameterized GWO with selectable a-decay schedule (first ablation)
gwo_op_sw_tuning.py          square-wave variant with tunable P, phi, a_low, a_high (second ablation)
gwo_baseline.py, pso_baseline.py, woa_baseline.py,
de_baseline.py, ga_baseline.py, hho_baseline.py
                             same-protocol baselines (direct encoding)
run_all_experiments.py       main comparison: 30 runs x 7 algorithms x 6 instances
run_ablation_schedule.py     first ablation: five a-decay schedules, 50 runs per arm
run_ablation_tuning.py       second ablation: five square-wave configurations, 50 runs per arm
plot_ablation_convergence.py convergence and a(t) traces of the ablations

revision1/                   analyses added in the first revision round
  colshade_baseline.py         COLSHADE reimplemented from Gurrola-Ramos et al. (2020),
                               including its dynamic equality tolerance
  run_colshade_experiments.py  COLSHADE under the common protocol (appends to results/raw_results.csv)
  rerun_gwo_opsw_for_revision.py  GWO-OP-SW runs with per-run angles and convergence traces
  stats_reanalysis.py          descriptive statistics, Mann-Whitney + Holm, Friedman,
                               ablation tests, penalty-multiplier sensitivity
  computational_overhead.py    runtime overhead of the reparameterization
  dimension_analysis.py        feasibility and quality against structural size
  schedule_param_justification.py  leave-one-instance-out check of the schedule selection
  verify_rc50.py               independent recomputation of the RC50 best solution
  engineering_analysis.py      waveforms, spectra and engineering quantities
  landscape_bias_analysis.py   boundary angles and the off-face initialization experiment
  theory_verification.py       numerical checks of Propositions 1-4
  plot_convergence_all_instances.py  convergence of GWO-OP-SW on all six instances
  gwo_final.py, sopwm_objective.py   copies of the root files, imported by the scripts here

revision2/                   analyses added in the second revision round
  regen_de_woa_hho.py          regeneration of the DE, WOA and HHO runs (see Reproducibility)
  adopt_archive_de_woa_hho.py  cross-check of those runs against the per-run archive in raw/
  c7_crossmethod_engineering_v2.py  engineering indicators of every method from verified angles
  c7_finalize_table.py         switching-instant columns and the final cross-method table
  table10_engineering_all_instances.py  engineering quantities of GWO-OP-SW on all six instances

raw/                         per-run archive of the direct-encoding baselines, with angle vectors
results/                     main per-run results, summaries, statistical tests and figures
```

## Requirements

Python 3.11. Install with:

```
pip install -r requirements.txt
```

## Protocol

| | main comparison | ablations |
|---|---|---|
| runs per configuration | 30 | 50 |
| population | 50 | 50 |
| iterations | 4,000 | 4,000 |
| maximum number of function evaluations | 200,000 | 200,000 |
| instances | RC45-RC50 | RC45-RC50 |

Runs are seeded as `42 + run_index`, so every algorithm and every ablation arm
meets the same initial populations and the comparisons are paired. COLSHADE is
run under the same protocol, with the same seeds and the same maximum number of
function evaluations.

- Switching angles are handled in **degrees** on [0, 90] and converted to
  radians inside the objective, following the benchmark's reference
  implementation.
- The ordering margin is `1e-6` and the equality tolerance `1e-4`; a run counts
  as feasible when the mean constraint violation is at most `1e-6`.
- The decode floor of GWO-OP-SW is `delta_min = 1e-5` (`EPS_MARGIN` in
  `gwo_op_sw.py`). It is deliberately larger than the ordering margin; see
  Section 2.3 of the paper before changing it.
- Baselines use Deb's feasibility rule. In the rank-based tests an algorithm
  with no feasible run on an instance is assigned ten times the worst feasible
  objective recorded there; `stats_reanalysis.py` repeats the ranking at 5x,
  10x and 20x.

## Reproducing the paper

Each experiment script skips work already recorded in its output, so an
interrupted run can be restarted with the same command. Run from the
repository root.

```bash
# experiments
python run_all_experiments.py                       # 7 same-protocol algorithms
python revision1/run_colshade_experiments.py        # COLSHADE under the same protocol
python revision1/rerun_gwo_opsw_for_revision.py     # GWO-OP-SW angles and convergence traces
python run_ablation_schedule.py
python run_ablation_tuning.py
python plot_ablation_convergence.py

# analyses (read the stored results; minutes, not hours)
python revision1/stats_reanalysis.py
python revision1/computational_overhead.py
python revision1/dimension_analysis.py
python revision1/schedule_param_justification.py
python revision1/verify_rc50.py
python revision1/engineering_analysis.py
python revision1/landscape_bias_analysis.py
python revision1/theory_verification.py
python revision1/plot_convergence_all_instances.py
python revision2/c7_crossmethod_engineering_v2.py
python revision2/c7_finalize_table.py
python revision2/table10_engineering_all_instances.py
```

### Where each result comes from

| Paper | File | Produced by |
|---|---|---|
| Table 4 (feasibility rates) | `results/raw_results.csv`, `results/summary_table.csv` | `run_all_experiments.py`, `run_colshade_experiments.py` |
| Table 5 (a-decay schedules) | `results/ablation/schedule_summary.csv` | `run_ablation_schedule.py` |
| Table 6 (square-wave configurations) | `results/ablation/tuning_summary.csv` | `run_ablation_tuning.py` |
| Section 4.2 tests and selection check | `revision1/results_revision/engineering/ablation_combo_vs_shorthigh.csv`, `t9_*.csv` | `stats_reanalysis.py`, `schedule_param_justification.py` |
| Table 7 (main comparison) | `results/summary_table.csv` | `run_all_experiments.py` |
| Section 4.3 statistics | `.../engineering/descriptive_stats_per_instance.csv`, `pairwise_mannwhitney_per_instance.csv`, `aggregate_*.csv`, `penalty_sensitivity.csv` | `stats_reanalysis.py` |
| Section 4.3 overhead and structural size | `.../engineering/computational_overhead_per_instance.csv`, `dimension_*.csv` | `computational_overhead.py`, `dimension_analysis.py` |
| RC50 best solution and its angles | `revision1/results_revision_1e-5/rc50_verified_angles.json`, `.../engineering/rc50_independent_verification.csv` | `verify_rc50.py` |
| Figure 2 (distributions) | `results/boxplot_RC45.png`, `results/boxplot_RC50.png` | `run_all_experiments.py` |
| Figure 3 (convergence) | `.../engineering/convergence_all_instances.png` | `plot_convergence_all_instances.py` |
| Table 8 (published reference results) | `results/summary_with_literature.csv` | `run_all_experiments.py` |
| Table 9 (COLSHADE) | `results/raw_results.csv` (COLSHADE rows), `.../engineering/pairwise_mannwhitney_per_instance.csv`, `elapsed_sec_all_algorithms.csv` | `run_colshade_experiments.py`, `stats_reanalysis.py`, `computational_overhead.py` |
| Table 10 (engineering quantities) | `revision2/results_revision2/table10_engineering_all_instances.csv`, `.../engineering/engineering_summary.csv` | `table10_engineering_all_instances.py`, `engineering_analysis.py` |
| Figure 4 (waveforms and spectra) | `.../engineering/waveform_RC*.png`, `spectrum_RC*.png` | `engineering_analysis.py` |
| Table 11 (cross-method indicators) | `revision2/results_revision2/table_c7_crossmethod_final.tex`, `c7_crossmethod_final.csv` | `c7_crossmethod_engineering_v2.py`, `c7_finalize_table.py` |
| Section 4.6 (boundary angles) | `.../engineering/t11_alpha_D_bias.csv`, `t11_initialization_experiment_*.csv` | `landscape_bias_analysis.py` |
| Propositions 1-4 | printed output | `theory_verification.py` |

`.../engineering/` stands for `revision1/results_revision/engineering/`.
Kendall's W is computed from the Friedman statistic as W = chi2 / (N (k - 1)).

The statistics quoted for IUDE, eMAgES and eiLSHADE (Kumar et al., 2020) and the
published COLSHADE means (Gurrola-Ramos et al., 2020) are taken from those
publications and are not rerun here.

## Reproducibility

An independent rerun of all 180 GWO-OP-SW runs reproduces the recorded
objective values to within a relative difference of 1.1e-16, and rerunning the
best feasible run of GWO, PSO, GA and COLSHADE reproduces each to within a
relative difference of 1e-14 (largest: GA, 9.7e-15).
The original DE, WOA and HHO records had been produced by an earlier state of
their implementation files, so all 540 of their runs were regenerated with
`revision2/regen_de_woa_hho.py` under the protocol above; the rows in
`results/raw_results.csv` are the regenerated ones. They agree with the
independent per-run archive in `raw/` to within its 10-digit storage
precision.

Switching angles are stored for every run of GWO-OP-SW, DE, WOA and HHO
(`best_alpha_json` in `results/raw_results.csv`), so any reported objective
value can be recomputed from the solution with `sopwm_objective.py`.
`revision2/results_revision2/c7_reproducibility_audit.csv` records the check
applied to every angle vector used in Table 11.

Runtimes (`elapsed_sec`) of the regenerated DE, WOA and HHO runs were measured
in a different session from the other algorithms; only the GWO-OP-SW versus
GWO overhead is compared in the paper.

The plotting and table scripts `plot_convergence_all_instances.py`,
`c7_finalize_table.py` and `table10_engineering_all_instances.py` were written
for this release to regenerate Figure 3, Table 11 and Table 10 from the stored
data; their outputs match the published figure and tables.

## Notes on the code

- `revision1/gwo_final.py` is the same algorithm as `gwo_op_sw.py`; the
  revision scripts import it under its working name.
- The COLSHADE entry point is named `colshade_op_sw` for historical reasons.
  It searches the switching angles directly, without the order-preserving
  decode.
- Some comments in the scripts are in Thai.

## License

Released under the MIT License (see `LICENSE`).

## Citation

See `CITATION.cff`, or cite the paper above.
