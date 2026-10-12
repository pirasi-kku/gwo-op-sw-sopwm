# -*- coding: utf-8 -*-
"""
regen_de_woa_hho.py — regenerate the DE, WOA and HHO rows of
results/raw_results.csv with the current baseline code, logging switching
angles this time.

WHY
---
An audit found that the DE, WOA and HHO rows in results/raw_results.csv
cannot be regenerated from the code now in this folder. Re-running the
recorded seed of each algorithm's best feasible run gives:

    GWO-OP-SW  1.1e-16   bit-exact
    COLSHADE   3.1e-16   bit-exact
    GWO        1.9e-16   bit-exact
    PSO        1.0e-15   bit-exact
    GA         9.7e-15   bit-exact
    HHO        3.4e-08   same trajectory, float noise only
    DE         4.7e+00   different trajectory
    WOA        1.1e-01   different trajectory

The DE and WOA baseline files were evidently edited after those runs were
made, and no earlier copy survives (this is not a git repository). The
numbers are not wrong, they were real runs, but they are not reproducible,
and Reviewer B4 asked for reproducibility specifically. HHO is included
because 3.4e-08 is not bit-exact either and it costs nothing extra to make
the whole dataset one code state.

WHAT IT DOES
------------
Runs DE, WOA and HHO over RC45-RC50, 30 runs each, seed = 42 + run,
pop_size = 50, max_iter = 4000, exactly the protocol of Section 3, and
records best_f, best_cv, feasible, elapsed_sec AND best_x_json. The angle
vectors were never saved before, which is the reason a separate re-run was
needed for the C7 engineering comparison; after this they are in the
dataset.

    feasible is best_cv <= 1e-6 and best_f is written as NaN when a run is
    infeasible, matching run_all_experiments.py exactly.

COST
----
540 runs. DE about 15 s, WOA about 9 s, HHO about 113 s per run, so roughly
45 min + 27 min + 5 h 40 min, about 7 hours in total. Run it overnight.

RESUMABLE
---------
Every completed run is appended to results_revision2/regen/<algo>.csv at
once. Re-running the script skips whatever is already there, so it can be
interrupted freely.

STAGES
------
    python regen_de_woa_hho.py            stage 1 only: run and record
    python regen_de_woa_hho.py --splice   stage 2: verify, then splice the
                                          new rows into raw_results.csv

Stage 2 refuses to run until all 540 runs are present. It backs up
raw_results.csv first, replaces only the DE, WOA and HHO rows, leaves every
other algorithm untouched, and re-checks one seed per algorithm to confirm
the new dataset reproduces bit-exactly.

AFTER STAGE 2 the derived numbers are stale. Re-run, in this order:
    python revision1/stats_reanalysis.py
    python revision1/dimension_analysis.py
    python revision1/computational_overhead.py
Then send me results/raw_results.csv and revision1/results_revision/
engineering/*.csv and I will update every affected table and claim.
"""

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

from sopwm_objective import sopwm_evaluate, get_bounds, get_dimension  # noqa: E402

RAW_CSV = os.path.join(_ROOT, "results", "raw_results.csv")
OUT_DIR = os.path.join(_HERE, "results_revision2", "regen")
os.makedirs(OUT_DIR, exist_ok=True)

POP_SIZE, MAX_ITER, N_RUNS, SEED_BASE = 50, 4000, 30, 42
FEASIBLE_TOL = 1e-6
ALGOS = ["DE", "WOA", "HHO"]          # cheapest first, so problems surface early
PROBS = [45, 46, 47, 48, 49, 50]
REL_TOL = 1e-9


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


def run_one(algo, prob_k, seed):
    dim = get_dimension(prob_k)
    lb, ub = get_bounds(prob_k)

    def obj(x, k=prob_k):
        return sopwm_evaluate(x, k)

    res = optimizer(algo)(obj, lb, ub, dim, pop_size=POP_SIZE,
                          max_iter=MAX_ITER, seed=seed, verbose=False)
    return (np.asarray(res["best_x"], dtype=float),
            float(res["best_f"]), float(res["best_cv"]))


def path_for(algo):
    return os.path.join(OUT_DIR, "%s.csv" % algo.lower())


def done_runs(algo):
    p = path_for(algo)
    if not os.path.exists(p):
        return set()
    d = pd.read_csv(p)
    return set(zip(d["prob_k"].astype(int), d["run"].astype(int)))


def append(algo, row):
    p = path_for(algo)
    pd.DataFrame([row]).to_csv(p, mode="a", header=not os.path.exists(p),
                               index=False)


def stage1():
    total = len(ALGOS) * len(PROBS) * N_RUNS
    count = 0
    t_start = time.time()
    for algo in ALGOS:
        have = done_runs(algo)
        for prob_k in PROBS:
            for run in range(N_RUNS):
                count += 1
                if (prob_k, run) in have:
                    continue
                seed = SEED_BASE + run
                t0 = time.time()
                ang, best_f, best_cv = run_one(algo, prob_k, seed)
                dt = time.time() - t0
                feasible = bool(best_cv <= FEASIBLE_TOL)
                append(algo, dict(
                    algorithm=algo, prob_k=prob_k, run=run, seed=seed,
                    best_f=best_f if feasible else np.nan,
                    best_cv=best_cv, feasible=feasible, elapsed_sec=dt,
                    best_x_json=json.dumps([float(v) for v in ang]),
                    best_f_raw=best_f))
                el = time.time() - t_start
                print("[%4d/%4d] %-4s RC%d run%02d: f=%.6e cv=%.3e feas=%s "
                      "(%.1f s, elapsed %.1f min)"
                      % (count, total, algo, prob_k, run, best_f, best_cv,
                         feasible, dt, el / 60.0), flush=True)
    print("\nStage 1 complete. Now run:  python regen_de_woa_hho.py --splice")


def stage2():
    # ---- completeness -----------------------------------------------------
    frames = {}
    for algo in ALGOS:
        p = path_for(algo)
        if not os.path.exists(p):
            raise SystemExit("STOP: %s missing. Stage 1 is not finished." % p)
        d = pd.read_csv(p)
        n = len(d)
        if n != len(PROBS) * N_RUNS:
            raise SystemExit("STOP: %s has %d rows, expected %d. Stage 1 is "
                             "not finished." % (p, n, len(PROBS) * N_RUNS))
        if d.duplicated(["prob_k", "run"]).any():
            raise SystemExit("STOP: %s has duplicate (prob_k, run) keys." % p)
        frames[algo] = d
    print("all %d new runs present" % sum(len(d) for d in frames.values()))

    # ---- the new rows must reproduce themselves ---------------------------
    print("\nverifying the new dataset reproduces bit-exactly:")
    for algo, d in frames.items():
        feas = d[d.feasible == True]  # noqa: E712
        if len(feas) == 0:
            print("  %-4s no feasible run to verify" % algo); continue
        row = feas.loc[feas.best_f.idxmin()]
        k, seed, rec = int(row.prob_k), int(row.seed), float(row.best_f)
        _, got, _ = run_one(algo, k, seed)
        rel = abs(got - rec) / max(abs(rec), 1e-300)
        print("  %-4s RC%d seed=%d rel.diff=%.3e %s"
              % (algo, k, seed, rel, "OK" if rel <= REL_TOL else "FAILED"))
        if rel > REL_TOL:
            raise SystemExit("STOP: the regenerated data does not reproduce "
                             "itself. Something is nondeterministic; do not "
                             "splice.")

    # ---- splice -----------------------------------------------------------
    main = pd.read_csv(RAW_CSV)
    backup = os.path.join(_ROOT, "revision2",
                          "raw_results_BEFORE_regen_backup.csv")
    main.to_csv(backup, index=False)
    print("\nbacked up to %s" % backup)

    keep = main[~main.algorithm.isin(ALGOS)].copy()
    n_replaced = len(main) - len(keep)
    cols = list(main.columns)
    if "best_alpha_json" not in cols:
        raise SystemExit("STOP: raw_results.csv has no best_alpha_json column.")

    new_rows = []
    for algo, d in frames.items():
        for _, r in d.iterrows():
            new_rows.append({
                "algorithm": algo, "prob_k": int(r.prob_k), "run": int(r.run),
                "seed": int(r.seed), "best_f": r.best_f, "best_cv": r.best_cv,
                "feasible": bool(r.feasible), "elapsed_sec": r.elapsed_sec,
                "best_alpha_json": r.best_x_json,
            })
    out = pd.concat([keep, pd.DataFrame(new_rows)[cols]], ignore_index=True)
    out = out.sort_values(["algorithm", "prob_k", "run"]).reset_index(drop=True)

    if len(out) != len(main):
        raise SystemExit("STOP: row count changed, %d -> %d. Not writing."
                         % (len(main), len(out)))
    if out.duplicated(["algorithm", "prob_k", "run"]).any():
        raise SystemExit("STOP: duplicate keys after splice. Not writing.")
    # untouched algorithms must be byte-identical
    for algo in sorted(set(main.algorithm) - set(ALGOS)):
        a = main[main.algorithm == algo].sort_values(["prob_k", "run"]).reset_index(drop=True)
        b = out[out.algorithm == algo].sort_values(["prob_k", "run"]).reset_index(drop=True)
        for c in ("best_f", "best_cv", "elapsed_sec"):
            assert np.array_equal(a[c].values, b[c].values, equal_nan=True), \
                "%s changed in column %s" % (algo, c)
    print("rows replaced: %d; other algorithms verified unchanged" % n_replaced)

    out.to_csv(RAW_CSV, index=False)
    print("written %s" % RAW_CSV)

    print("\nFeasibility rate, before and after, per cent:")
    print("%-5s %s" % ("", "  ".join("RC%d" % k for k in PROBS)))
    for algo in ALGOS:
        old = [100.0 * main[(main.algorithm == algo) & (main.prob_k == k)].feasible.mean()
               for k in PROBS]
        new = [100.0 * out[(out.algorithm == algo) & (out.prob_k == k)].feasible.mean()
               for k in PROBS]
        print("  %-4s old %s" % (algo, " ".join("%5.1f" % v for v in old)))
        print("  %-4s new %s" % ("", " ".join("%5.1f" % v for v in new)))

    print("""
Derived numbers are now stale. Re-run, in this order:
    python revision1/stats_reanalysis.py
    python revision1/dimension_analysis.py
    python revision1/computational_overhead.py
Then send me results/raw_results.csv and
revision1/results_revision/engineering/*.csv.

Note on timings: elapsed_sec for DE, WOA and HHO now comes from this
session, while the other algorithms keep their original measurements. The
runtime claim in the paper compares GWO-OP-SW against standard GWO only,
and neither was touched, so that claim is unaffected. Any cross-algorithm
runtime remark has to say the two groups were timed in different sessions.
""")


if __name__ == "__main__":
    if "--splice" in sys.argv:
        stage2()
    else:
        print(__doc__.split("STAGES")[0])
        stage1()
