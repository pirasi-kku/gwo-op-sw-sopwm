# GWO-OP-SW: Order-Preserving Grey Wolf Optimizer for SOPWM

<!-- TODO: paste the concept-DOI badge Zenodo gives you after the first release -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

Code and results accompanying:

> T. Sinlapakorn and P. Singsathid, "Order-Preserving Grey Wolf Optimizer for
> Selective Optimal Pulse-Width Modulation in Multilevel Inverters,"
> *Bulletin of Electrical Engineering and Informatics*. (under review)

The repository reproduces every table and figure in the paper.

## What the method does

Selective optimal pulse-width modulation (SOPWM) requires switching angles in
strict ascending order. A standard grey wolf optimizer searching the angles
directly breaks that order routinely, and its feasibility rate falls to zero as
the number of angles grows. GWO-OP-SW searches a non-negative increment space
instead and decodes it by a cumulative sum, so the ordering constraints hold by
construction rather than being repaired or penalized. A square-wave
convergence-factor schedule counters the premature convergence that a monotonic
schedule leaves unresolved.

Benchmark: instances RC45–RC50 of the CEC2020 real-world constrained
optimization suite (3- to 13-level inverters, 25–30 switching angles).

## Layout

```
sopwm_objective.py          objective, constraints and instance constants (RC45-RC50)
gwo_op_sw.py                the proposed algorithm
gwo_op_schedules.py         reparameterized GWO with selectable a-decay schedule (ablation 1)
gwo_op_sw_tuning.py         square-wave variant with tunable P, phi, a_low (ablation 2)
gwo_baseline.py             standard GWO
pso_baseline.py             PSO
woa_baseline.py             WOA
de_baseline.py              DE
ga_baseline.py              GA
hho_baseline.py             HHO

run_all_experiments.py      main comparison, 30 runs x 7 algorithms x 6 instances
run_ablation_schedule.py    ablation 1: five a-decay schedules, 50 runs per arm
run_ablation_tuning.py      ablation 2: five square-wave configurations, 50 runs per arm
plot_ablation_convergence.py convergence and a(t) trace figures

results/                    outputs corresponding to the tables and figures in the paper
```

## Requirements

Python 3.11. Install with:

```
pip install -r requirements.txt
```

## Reproducing the paper

Each script writes to `results/` and skips work already recorded there, so a
run interrupted partway can be restarted with the same command.

```bash
python run_all_experiments.py        # Tables 3, 6, 7; Figures 3, 4, 5
python run_ablation_schedule.py      # Table 4
python run_ablation_tuning.py        # Table 5
python plot_ablation_convergence.py  # Figures 1, 2
```

Runtime is roughly 3–4 s per run on a desktop CPU; the main comparison is about
1,260 runs and the two ablations about 1,800 runs in total.

### Protocol

| | main comparison | ablations |
|---|---|---|
| runs per configuration | 30 | 50 |
| population | 50 | 50 |
| iterations | 4,000 | 4,000 |
| instances | RC45–RC50 | RC45–RC50 |

Runs are seeded as `42 + run_index`, so every algorithm and every ablation arm
meets the same initial populations and the comparisons are paired.

## Results files

| file | corresponds to |
|---|---|
| `results/summary_table.csv` | Tables 3, 6, 8 (best/mean/std/worst, FR%) |
| `results/raw_results.csv` | per-run outcomes behind the summary |
| `results/statistical_tests.csv` | Friedman and Wilcoxon results |
| `results/sr_profile.csv` | success rate at varying tolerances |
| `results/ablation/schedule_summary.csv` | Table 4 |
| `results/ablation/tuning_summary.csv` | Table 5 |
| `results/figures/`, `results/ablation/*.png` | Figures 1–5 |

Literature baselines (IUDE, eMAgES, eiLSHADE, COLSHADE) are not rerun here.
Their statistics are quoted from the original publications and are stored as
constants in `run_all_experiments.py`.

## Notes on the implementation

- Switching angles are handled in **degrees** on [0°, 90°] and converted to
  radians inside the objective, following the benchmark's reference
  implementation rather than its text description.
- The ordering margin is `1e-6` and the equality tolerance `1e-4`; a run counts
  as feasible when the mean constraint violation is at most `1e-6`.
- Constraint handling uses Deb's feasibility rule. In the statistical tests, an
  algorithm producing no feasible solution on an instance is assigned ten times
  the worst feasible objective recorded on that instance, so that the paired
  tests remain defined.

## License

Released under the MIT License (see `LICENSE`).

## Citation

See `CITATION.cff`, or cite the paper above.
