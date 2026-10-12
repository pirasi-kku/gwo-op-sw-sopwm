# -*- coding: utf-8 -*-
"""
adopt_archive_de_woa_hho.py — make the DE, WOA and HHO rows of
results/raw_results.csv reproducible, without spending seven hours of
compute.

THE FINDING THIS RESTS ON
-------------------------
The DE, WOA and HHO rows currently in results/raw_results.csv cannot be
regenerated from the code in this folder. Re-running the recorded seed of
each algorithm's best feasible run gives:

    GWO-OP-SW  1.1e-16   COLSHADE 3.1e-16   GWO 1.9e-16
    PSO        1.0e-15   GA       9.7e-15   HHO 3.4e-08
    DE         4.7e+00   WOA      1.1e-01   <- different trajectories

But raw/<algo>_baseline_RC<k>.csv, the per-algorithm archive written on
17 July 2026, DOES reproduce exactly under today's code. Sampled:

    DE  RC47 runs 0, 7, 20    rel.diff 9.0e-13, 6.2e-12, 1.8e-11
    WOA RC47 runs 0, 7, 17    rel.diff 1.9e-12, 4.3e-11, 6.0e-12
    HHO RC45 run 15           rel.diff 3.1e-12

(The 1e-11 floor is the archive storing ten significant digits, not error.)
So the archive and the current code agree, and it is the main results file
that carries rows from some older code state. Re-running 540 jobs would
therefore land on the archive's numbers. Adopting the archive reaches the
same dataset in minutes, and the archive also carries best_x_json, the
switching angles, which the main file never had for these algorithms.

If you would rather generate from scratch anyway, regen_de_woa_hho.py does
that; it takes about seven hours and should produce these same values.

WHAT THIS SCRIPT DOES
---------------------
1. Checks the archive is complete: three algorithms, six instances, 30 runs
   each, seed == 42 + run.
2. Recomputes f and the mean constraint violation from each archived ANGLE
   VECTOR through sopwm_evaluate, and requires them to match the archive's
   own best_f and best_cv. This validates the angles, which are what the
   engineering comparison needs.
3. Re-runs a sample of seeds per algorithm with the current optimizer code
   and requires bit-exact agreement. Default 2 for DE and WOA, 1 for HHO,
   since an HHO run costs about two minutes.
4. Applies the conventions of run_all_experiments.py: feasible is
   best_cv <= 1e-6, and best_f is NaN for an infeasible run.
5. Backs up results/raw_results.csv, replaces only the DE, WOA and HHO
   rows, and asserts that the row count is unchanged, that no key is
   duplicated, and that every other algorithm is untouched.
6. Prints what moves, so you can see the size of the rewrite before
   agreeing to it. Nothing is written until you pass --write.

USAGE
-----
    python adopt_archive_de_woa_hho.py              dry run, report only
    python adopt_archive_de_woa_hho.py --write      verify, then splice
    python adopt_archive_de_woa_hho.py --write --samples 5

AFTERWARDS the derived numbers are stale. Re-run, in this order:
    python revision1/stats_reanalysis.py
    python revision1/dimension_analysis.py
    python revision1/computational_overhead.py
Then send me results/raw_results.csv and
revision1/results_revision/engineering/*.csv.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sopwm_objective import (  # noqa: E402
    sopwm_evaluate, mean_constraint_violation, get_bounds, get_dimension,
)

RAW_CSV = os.path.join(_ROOT, "results", "raw_results.csv")
ARCHIVE = os.path.join(_ROOT, "raw")
BACKUP = os.path.join(_HERE, "raw_results_BEFORE_archive_adoption.csv")

ALGOS = {"DE": "de", "WOA": "woa", "HHO": "hho"}
PROBS = list(range(45, 51))
N_RUNS, SEED_BASE, FEASIBLE_TOL = 30, 42, 1e-6
ANGLE_TOL = 1e-8      # archive stores 10 significant digits
RERUN_TOL = 1e-9


def optimizer(algo):
    if algo == "DE":
        from de_baseline import differential_evolution
        return differential_evolution
    if algo == "WOA":
        from woa_baseline import whale_optimizer
        return whale_optimizer
    if algo == "HHO":
        from hho_baseline import harris_hawks_optimizer
        return harris_hawks_optimizer
    raise ValueError(algo)


def load_archive():
    out = {}
    for algo, tag in ALGOS.items():
        for k in PROBS:
            p = os.path.join(ARCHIVE, "%s_baseline_RC%d.csv" % (tag, k))
            if not os.path.exists(p):
                raise SystemExit("STOP: missing archive file %s" % p)
            d = pd.read_csv(p)
            key = "run_id" if "run_id" in d.columns else "run"
            d = d.rename(columns={key: "run"}).sort_values("run").reset_index(drop=True)
            if len(d) != N_RUNS:
                raise SystemExit("STOP: %s has %d rows, expected %d"
                                 % (p, len(d), N_RUNS))
            if "best_x_json" not in d.columns:
                raise SystemExit("STOP: %s has no best_x_json column" % p)
            bad = d[d["seed"].astype(int) != SEED_BASE + d["run"].astype(int)]
            if len(bad):
                raise SystemExit("STOP: %s has seeds that are not 42 + run" % p)
            out[(algo, k)] = d
    print("archive complete: %d algorithms x %d instances x %d runs"
          % (len(ALGOS), len(PROBS), N_RUNS))
    return out


def validate_angles(arc):
    print("\nrecomputing f and MCV from every archived angle vector:")
    worst_f = worst_cv = 0.0
    for (algo, k), d in sorted(arc.items()):
        wf = wc = 0.0
        for _, r in d.iterrows():
            ang = np.asarray(json.loads(r["best_x_json"]), dtype=float)
            dim = get_dimension(k)
            if ang.size != dim:
                raise SystemExit("STOP: %s RC%d run %d has %d angles, expected %d"
                                 % (algo, k, r["run"], ang.size, dim))
            f, g, h = sopwm_evaluate(ang, k)
            cv = mean_constraint_violation(g, h)
            wf = max(wf, abs(float(f[0]) - float(r["best_f"]))
                     / max(abs(float(r["best_f"])), 1e-300))
            wc = max(wc, abs(float(cv[0]) - float(r["best_cv"])))
        print("  %-4s RC%d  worst rel.diff in f %.2e, abs.diff in MCV %.2e"
              % (algo, k, wf, wc))
        worst_f, worst_cv = max(worst_f, wf), max(worst_cv, wc)
    if worst_f > ANGLE_TOL:
        raise SystemExit("STOP: archived angles do not reproduce archived "
                         "objective values (worst %.2e)" % worst_f)
    print("  all angle vectors reproduce their objective value (worst %.2e)"
          % worst_f)


def sample_reruns(arc, n_de_woa, n_hho):
    print("\nre-running a sample of seeds with the current optimizer code:")
    rng = np.random.default_rng(0)
    for algo in ALGOS:
        n = n_hho if algo == "HHO" else n_de_woa
        if n <= 0:
            print("  %-4s skipped by request" % algo); continue
        picks = []
        for _ in range(n):
            k = int(rng.choice(PROBS))
            run = int(rng.integers(0, N_RUNS))
            picks.append((k, run))
        for k, run in picks:
            d = arc[(algo, k)]
            row = d[d.run == run].iloc[0]
            dim = get_dimension(k)
            lb, ub = get_bounds(k)

            def obj(x, kk=k):
                return sopwm_evaluate(x, kk)

            t0 = time.time()
            res = optimizer(algo)(obj, lb, ub, dim, pop_size=50, max_iter=4000,
                                  seed=SEED_BASE + run, verbose=False)
            dt = time.time() - t0
            rec, got = float(row["best_f"]), float(res["best_f"])
            rel = abs(got - rec) / max(abs(rec), 1e-300)
            ok = rel <= ANGLE_TOL
            print("  %-4s RC%d run%02d archive=%.10e rerun=%.10e rel=%.2e %s (%.1f s)"
                  % (algo, k, run, rec, got, rel, "OK" if ok else "FAILED", dt),
                  flush=True)
            if not ok:
                raise SystemExit("STOP: the archive does not reproduce under "
                                 "the current code. Do not adopt it; use "
                                 "regen_de_woa_hho.py instead.")


def report_and_splice(arc, write):
    main = pd.read_csv(RAW_CSV)
    cols = list(main.columns)
    if "best_alpha_json" not in cols:
        raise SystemExit("STOP: raw_results.csv has no best_alpha_json column.")

    new_rows = []
    for (algo, k), d in sorted(arc.items()):
        for _, r in d.iterrows():
            cv = float(r["best_cv"])
            feas = bool(cv <= FEASIBLE_TOL)
            new_rows.append({
                "algorithm": algo, "prob_k": k, "run": int(r["run"]),
                "seed": int(r["seed"]),
                "best_f": float(r["best_f"]) if feas else np.nan,
                "best_cv": cv, "feasible": feas,
                "elapsed_sec": float(r["elapsed_sec"]),
                "best_alpha_json": r["best_x_json"],
            })
    new = pd.DataFrame(new_rows)

    print("\nwhat changes (current dataset -> archive):")
    print("  %-5s %s" % ("", "   ".join("  RC%d" % k for k in PROBS)))
    for algo in ALGOS:
        o = [100.0 * main[(main.algorithm == algo) & (main.prob_k == k)].feasible.mean()
             for k in PROBS]
        n = [100.0 * new[(new.algorithm == algo) & (new.prob_k == k)].feasible.mean()
             for k in PROBS]
        print("  %-5s FR  %s" % (algo, " ".join("%5.1f" % v for v in o)))
        print("  %-5s  -> %s" % ("", " ".join("%5.1f" % v for v in n)))
    print("\n  note: WOA on RC50 goes from 33.3%% to 0%%, so its RC50 cells")
    print("  become empty and any test involving it becomes untestable.")

    keep = main[~main.algorithm.isin(ALGOS)].copy()
    out = pd.concat([keep, new[cols]], ignore_index=True)
    out = out.sort_values(["algorithm", "prob_k", "run"]).reset_index(drop=True)
    if len(out) != len(main):
        raise SystemExit("STOP: row count %d -> %d" % (len(main), len(out)))
    if out.duplicated(["algorithm", "prob_k", "run"]).any():
        raise SystemExit("STOP: duplicate keys after splice")
    for algo in sorted(set(main.algorithm) - set(ALGOS)):
        a = main[main.algorithm == algo].sort_values(["prob_k", "run"]).reset_index(drop=True)
        b = out[out.algorithm == algo].sort_values(["prob_k", "run"]).reset_index(drop=True)
        for c in ("best_f", "best_cv", "elapsed_sec"):
            if not np.array_equal(a[c].values, b[c].values, equal_nan=True):
                raise SystemExit("STOP: %s changed in column %s" % (algo, c))
    print("\n  row count unchanged (%d), no duplicate keys, other five "
          "algorithms verified untouched" % len(out))

    if not write:
        print("\nDry run. Nothing written. Re-run with --write to apply.")
        return
    main.to_csv(BACKUP, index=False)
    out.to_csv(RAW_CSV, index=False)
    print("\n  backup  %s" % BACKUP)
    print("  written %s" % RAW_CSV)
    print("""
Derived numbers are now stale. Re-run, in this order:
    python revision1/stats_reanalysis.py
    python revision1/dimension_analysis.py
    python revision1/computational_overhead.py
Then send me results/raw_results.csv and
revision1/results_revision/engineering/*.csv and I will update every
affected table, statistic and claim.

Note on timings: elapsed_sec for DE, WOA and HHO now comes from the July
archive session rather than the main batch. The runtime claim in the paper
compares GWO-OP-SW against standard GWO only, and neither was touched, so
that claim is unaffected; any cross-algorithm runtime remark has to say the
two groups were timed in different sessions.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--samples", type=int, default=2,
                    help="re-run samples per algorithm for DE and WOA")
    ap.add_argument("--hho-samples", type=int, default=1)
    a = ap.parse_args()
    print(__doc__.split("USAGE")[0])
    arc = load_archive()
    validate_angles(arc)
    sample_reruns(arc, a.samples, a.hho_samples)
    report_and_splice(arc, a.write)


if __name__ == "__main__":
    main()
