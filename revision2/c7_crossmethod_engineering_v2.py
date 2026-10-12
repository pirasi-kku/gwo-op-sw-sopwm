# -*- coding: utf-8 -*-
"""
c7_crossmethod_engineering_v2.py — Reviewer C7, cross-method engineering
indicators. Replaces v1, which aborted on the first algorithm whose recorded
objective value could not be reproduced.

WHAT CHANGED AND WHY
--------------------
v1 re-ran one seed per (algorithm, instance) and asserted that the re-run
reproduced the recorded objective value. DE on RC47 failed that check, by a
factor of 4.7, and v1 stopped the whole job. The check was right to fire.
What it found is that the DE and WOA rows in results/raw_results.csv were
produced by a state of de_baseline.py / woa_baseline.py that no longer
exists in this folder, so those runs cannot be regenerated from today's
code. Measured on the best feasible run of each algorithm:

    GWO-OP-SW  1.1e-16   bit-exact
    COLSHADE   3.1e-16   bit-exact
    GWO        1.9e-16   bit-exact
    PSO        1.0e-15   bit-exact
    GA         9.7e-15   bit-exact
    HHO        3.4e-08   same trajectory, float noise
    DE         4.7e+00   DIFFERENT TRAJECTORY
    WOA        1.1e-01   DIFFERENT TRAJECTORY

v2 therefore does three things differently.

1. It prefers the archive over re-running. raw/<algo>_baseline_RC<k>.csv
   stores best_x_json for the baselines. Where the archived angles recompute
   to the objective value recorded in results/raw_results.csv, they ARE that
   run and no optimizer needs to be invoked. This holds for PSO and GA.
2. It re-runs only where the archive is absent or does not match, and it
   verifies the re-run the same way.
3. A failure to verify no longer aborts the job. That (algorithm, instance)
   is dropped from the comparison, the reason is recorded, and the run
   continues. Nothing unverified reaches a table.

COST
----
Three COLSHADE runs at roughly 110-160 s, two GWO runs at about 6 s.
Expect about six minutes. PSO and GA cost nothing.

HOW TO RUN
----------
    cd revision2
    python c7_crossmethod_engineering_v2.py

Resumable: verified angle vectors are cached in
results_revision2/crossmethod_angles_v2.json.

OUTPUTS (in revision2/results_revision2/)
-----------------------------------------
    crossmethod_angles_v2.json          verified angle vectors and provenance
    c7_reproducibility_audit.csv        per algorithm and instance: source,
                                        relative difference, verdict
    c7_crossmethod_indicators.csv       V1, V1 error, THD, worst targeted
    c7_targeted_harmonics.csv           V_n/V1 for eight targeted orders
    table_c7_crossmethod.tex            LaTeX, ready to \\input
    table_c7_targeted_harmonics.tex     LaTeX, ready to \\input
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "revision1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sopwm_objective import (  # noqa: E402
    sopwm_evaluate, mean_constraint_violation, get_bounds, get_dimension,
    K_HARMONICS, _SOPWM_PARAMS,
)

RAW_CSV   = os.path.join(_ROOT, "results", "raw_results.csv")
ARCHIVE   = os.path.join(_ROOT, "raw")
OUT_DIR   = os.path.join(_HERE, "results_revision2")
os.makedirs(OUT_DIR, exist_ok=True)
CACHE     = os.path.join(OUT_DIR, "crossmethod_angles_v2.json")

POP_SIZE, MAX_ITER, MAX_FES, SEED_BASE = 50, 4000, 200_000, 42
INSTANCES = [47, 49, 50]
BASELINES = ["COLSHADE", "DE", "WOA", "GWO", "PSO", "GA"]
ARCHIVE_TAG = {"DE": "de", "WOA": "woa", "PSO": "pso", "GA": "ga", "HHO": "hho"}
SHOW_ORDERS = [5, 7, 11, 13, 17, 19, 23, 25]
N_HARM_MAX = 301
REL_TOL = 1e-9


# ---------------------------------------------------------------------------
# verification: do these angles reproduce the recorded objective value?
# ---------------------------------------------------------------------------
def objective_from_angles(angles, prob_k):
    f, g, h = sopwm_evaluate(np.asarray(angles, dtype=float), prob_k)
    return float(f[0]), float(mean_constraint_violation(g, h)[0])


def from_archive(algo, prob_k, run, recorded_f):
    tag = ARCHIVE_TAG.get(algo)
    if tag is None:
        return None, "no archive file for this algorithm"
    path = os.path.join(ARCHIVE, "%s_baseline_RC%d.csv" % (tag, prob_k))
    if not os.path.exists(path):
        return None, "archive file missing"
    r = pd.read_csv(path)
    key = "run_id" if "run_id" in r.columns else "run"
    rr = r[r[key] == run]
    if len(rr) == 0 or "best_x_json" not in r.columns:
        return None, "run absent from archive"
    ang = np.asarray(json.loads(rr.iloc[0]["best_x_json"]), dtype=float)
    got, cv = objective_from_angles(ang, prob_k)
    rel = abs(got - recorded_f) / max(abs(recorded_f), 1e-300)
    if rel <= REL_TOL:
        return (ang, cv, rel), "archive"
    return None, "archive holds a different run (rel.diff %.2e)" % rel


def rerun(algo, prob_k, seed, recorded_f):
    dim = get_dimension(prob_k)
    lb, ub = get_bounds(prob_k)

    def obj(x, k=prob_k):
        return sopwm_evaluate(x, k)

    kw = dict(pop_size=POP_SIZE, max_iter=MAX_ITER, seed=seed, verbose=False)
    if algo == "COLSHADE":
        from colshade_baseline import colshade_op_sw
        res = colshade_op_sw(obj, lb, ub, dim, max_fes=MAX_FES, seed=seed,
                             verbose=False)
    elif algo == "DE":
        from de_baseline import differential_evolution
        res = differential_evolution(obj, lb, ub, dim, **kw)
    elif algo == "WOA":
        from woa_baseline import whale_optimizer
        res = whale_optimizer(obj, lb, ub, dim, **kw)
    elif algo == "GWO":
        from gwo_baseline import grey_wolf_optimizer
        res = grey_wolf_optimizer(obj, lb, ub, dim, **kw)
    elif algo == "PSO":
        from pso_baseline import particle_swarm_optimizer
        res = particle_swarm_optimizer(obj, lb, ub, dim, **kw)
    elif algo == "GA":
        from ga_baseline import genetic_algorithm
        res = genetic_algorithm(obj, lb, ub, dim, **kw)
    else:
        raise ValueError(algo)

    ang = np.asarray(res["best_x"], dtype=float)
    got = float(res["best_f"])
    rel = abs(got - recorded_f) / max(abs(recorded_f), 1e-300)
    if rel <= REL_TOL:
        return (ang, float(res["best_cv"]), rel), "re-run"
    return None, "re-run does not reproduce the recorded value (rel.diff %.2e)" % rel


def recover():
    df = pd.read_csv(RAW_CSV)
    store = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    audit = []

    for prob_k in INSTANCES:
        for algo in BASELINES:
            sub = df[(df.algorithm == algo) & (df.prob_k == prob_k)
                     & (df.feasible == True)]  # noqa: E712
            if len(sub) == 0:
                print("  --   %-9s RC%d  no feasible run in the dataset" % (algo, prob_k))
                audit.append(dict(algorithm=algo, prob_k=prob_k, source="n/a",
                                  rel_diff=np.nan, verdict="no feasible run"))
                continue
            row = sub.loc[sub.best_f.idxmin()]
            run, seed, rec = int(row.run), int(row.seed), float(row.best_f)
            key = "%s|RC%d" % (algo, prob_k)
            if key in store:
                print("  ok   %-9s RC%d  cached (%s)" % (algo, prob_k, store[key]["source"]))
                audit.append(dict(algorithm=algo, prob_k=prob_k,
                                  source=store[key]["source"],
                                  rel_diff=store[key]["rel_diff"], verdict="verified"))
                continue

            got, why = from_archive(algo, prob_k, run, rec)
            if got is None:
                print("  ..   %-9s RC%d  %s; re-running seed %d"
                      % (algo, prob_k, why, seed), flush=True)
                t0 = time.time()
                got, why = rerun(algo, prob_k, seed, rec)
                dt = time.time() - t0
            else:
                dt = 0.0

            if got is None:
                print("  FAIL %-9s RC%d  %s -- dropped from the comparison"
                      % (algo, prob_k, why), flush=True)
                audit.append(dict(algorithm=algo, prob_k=prob_k, source="none",
                                  rel_diff=np.nan, verdict=why))
                continue

            ang, cv, rel = got
            gaps = np.diff(ang)
            if gaps.size and gaps.min() <= 0:
                print("       note: %s RC%d has a non-increasing pair (min gap "
                      "%.3e deg); stored as returned, unsorted"
                      % (algo, prob_k, gaps.min()))
            print("  ok   %-9s RC%d  %s, rel.diff %.2e (%.1f s)"
                  % (algo, prob_k, why, rel, dt), flush=True)
            store[key] = dict(algorithm=algo, prob_k=prob_k, run=run, seed=seed,
                              recorded_best_f=rec, source=why, rel_diff=rel,
                              best_cv=cv,
                              min_gap_deg=float(gaps.min()) if gaps.size else None,
                              angles_deg=[float(v) for v in ang])
            json.dump(store, open(CACHE, "w", encoding="utf-8"), indent=1)
            audit.append(dict(algorithm=algo, prob_k=prob_k, source=why,
                              rel_diff=rel, verdict="verified"))

    pd.DataFrame(audit).to_csv(
        os.path.join(OUT_DIR, "c7_reproducibility_audit.csv"), index=False)
    return store


# ---------------------------------------------------------------------------
# engineering indicators
# ---------------------------------------------------------------------------
def odd_non_triplen(n_max):
    return np.array([n for n in range(1, n_max + 1, 2) if n % 3 != 0])


def fourier_amplitudes(angles_deg, s, orders):
    x = np.asarray(angles_deg, dtype=float) * np.pi / 180.0
    return {int(n): float((4.0 / (n * np.pi)) * np.sum(s * np.cos(n * x)))
            for n in orders}


def indicators(angles_deg, prob_k):
    p = _SOPWM_PARAMS[prob_k]
    s, j, m, D = p["s"], p["j"], p["m"], p["D"]
    thd_orders = odd_non_triplen(N_HARM_MAX)
    V = fourier_amplitudes(angles_deg, s, thd_orders)
    V1 = V[1]
    harm_rms = np.sqrt(sum(V[int(n)] ** 2 for n in thd_orders if n != 1))
    Vt = fourier_amplitudes(angles_deg, s, K_HARMONICS)
    targeted_pct = {n: 100.0 * abs(Vt[n]) / abs(V1) for n in Vt}
    worst = max(targeted_pct, key=targeted_pct.get)
    f, cv = objective_from_angles(angles_deg, prob_k)
    return dict(prob_k=prob_k, level=2 * j + 1, D=D, j=j, m=m,
                objective_f=f, mcv=cv, V1=V1,
                V1_target=(4.0 / np.pi) * j * m,
                V1_abs_error=abs(V1 - (4.0 / np.pi) * j * m),
                THD_pct=100.0 * harm_rms / abs(V1),
                worst_targeted_order=int(worst),
                worst_targeted_pct=targeted_pct[worst],
                transitions_per_cycle=4 * D,
                transitions_per_cell_per_cycle=4 * D / j,
                targeted_pct={int(n): targeted_pct[n] for n in SHOW_ORDERS})


def stored_proposed_angles(prob_k):
    df = pd.read_csv(RAW_CSV)
    sub = df[(df.algorithm == "GWO-OP-SW") & (df.prob_k == prob_k)
             & (df.feasible == True)]  # noqa: E712
    row = sub.loc[sub.best_f.idxmin()]
    return np.asarray(json.loads(row["best_alpha_json"]), dtype=float)


def label(name):
    return "GWO-OP-SW (proposed)" if name == "GWO-OP-SW" else name


def main():
    print(__doc__.split("HOW TO RUN")[0])
    print("=" * 70)
    print("Stage 1: recovering and verifying switching angles")
    print("=" * 70)
    store = recover()

    print("\n" + "=" * 70)
    print("Stage 2: engineering indicators")
    print("=" * 70)
    rows, harm = [], []
    for prob_k in INSTANCES:
        series = [("GWO-OP-SW", stored_proposed_angles(prob_k))]
        for algo in BASELINES:
            key = "%s|RC%d" % (algo, prob_k)
            if key in store:
                series.append((algo, np.asarray(store[key]["angles_deg"])))
        for algo, ang in series:
            d = indicators(ang, prob_k)
            tp = d.pop("targeted_pct")
            d["algorithm"] = algo
            rows.append(d)
            h = dict(algorithm=algo, prob_k=prob_k, level=d["level"])
            h.update({("V%d_pct" % n): tp[n] for n in SHOW_ORDERS})
            h["worst_targeted_order"] = d["worst_targeted_order"]
            h["worst_targeted_pct"] = d["worst_targeted_pct"]
            harm.append(h)
            print("  RC%d %-9s THD %7.3f%%  |V1 err| %.2e  worst h%-3d %6.3f%%"
                  % (prob_k, algo, d["THD_pct"], d["V1_abs_error"],
                     d["worst_targeted_order"], d["worst_targeted_pct"]))

    ind, har = pd.DataFrame(rows), pd.DataFrame(harm)
    ind.to_csv(os.path.join(OUT_DIR, "c7_crossmethod_indicators.csv"), index=False)
    har.to_csv(os.path.join(OUT_DIR, "c7_targeted_harmonics.csv"), index=False)

    lines = [r"\begin{tabular}{llccc}", r"\hline",
             r"Instance & Method & $|V_1$ error$|$ (p.u.) & THD (\%) & "
             r"Worst targeted harmonic (\% of $V_1$) \\", r"\hline"]
    for prob_k in INSTANCES:
        sub = ind[ind.prob_k == prob_k]
        lvl = int(sub.level.iloc[0])
        for i, (_, r) in enumerate(sub.iterrows()):
            head = (r"\multirow{%d}{*}{RC%d (%d-level)}" % (len(sub), prob_k, lvl)
                    if i == 0 else "")
            lines.append(r"%s & %s & %.1e & %.2f & $h_{%d}$: %.3f \\"
                         % (head, label(r.algorithm), r.V1_abs_error, r.THD_pct,
                            int(r.worst_targeted_order), r.worst_targeted_pct))
    lines += [r"\hline", r"\end{tabular}"]
    open(os.path.join(OUT_DIR, "table_c7_crossmethod.tex"), "w",
         encoding="utf-8").write("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{ll%s}" % ("c" * len(SHOW_ORDERS)), r"\hline",
             r"Instance & Method & " +
             " & ".join(r"$h_{%d}$" % n for n in SHOW_ORDERS) + r" \\", r"\hline"]
    for prob_k in INSTANCES:
        sub = har[har.prob_k == prob_k]
        lvl = int(sub.level.iloc[0])
        for i, (_, r) in enumerate(sub.iterrows()):
            head = (r"\multirow{%d}{*}{RC%d (%d-level)}" % (len(sub), prob_k, lvl)
                    if i == 0 else "")
            lines.append(r"%s & %s & %s \\"
                         % (head, label(r.algorithm),
                            " & ".join("%.3f" % r["V%d_pct" % n] for n in SHOW_ORDERS)))
    lines += [r"\hline", r"\end{tabular}"]
    open(os.path.join(OUT_DIR, "table_c7_targeted_harmonics.tex"), "w",
         encoding="utf-8").write("\n".join(lines) + "\n")

    print("\nWritten to %s" % OUT_DIR)
    for f in ("crossmethod_angles_v2.json", "c7_reproducibility_audit.csv",
              "c7_crossmethod_indicators.csv", "c7_targeted_harmonics.csv",
              "table_c7_crossmethod.tex", "table_c7_targeted_harmonics.tex"):
        print("  " + f)
    print("\nRead c7_reproducibility_audit.csv before using the tables: it says,")
    print("per algorithm and instance, whether the angles came from the archive")
    print("or a re-run, and which pairs were dropped and why.")
    print("\nSend me these files and I will write them into Section 4.5.")


if __name__ == "__main__":
    main()
