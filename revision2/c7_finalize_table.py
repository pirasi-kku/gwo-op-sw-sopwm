"""
c7_finalize_table.py -- builds Table 11 of the manuscript (engineering indicators
across methods) from the outputs of c7_crossmethod_engineering_v2.py.

It adds the two switching-instant columns to the indicators: the number of
distinct switching instants among the D angles at 1e-6 degree resolution, and
the smallest separation between consecutive angles (negative when two angles
are inverted). A row is set in bold when the solution has coincident or
inverted angles.

Inputs (all produced by c7_crossmethod_engineering_v2.py or the main run):
  results_revision2/crossmethod_angles_v2.json   verified baseline angles
  ../results/raw_results.csv                     GWO-OP-SW angles (best feasible run)
  results_revision2/c7_crossmethod_indicators.csv

Outputs (written to results_revision2/):
  c7_min_separation.csv
  c7_crossmethod_final.csv
  table_c7_crossmethod_final.tex   (the tabular used as Table 11)

HOW TO RUN
  python revision2/c7_finalize_table.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

RAW_CSV = os.path.join(_ROOT, "results", "raw_results.csv")
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "results_revision2")
IN_DIR = os.path.join(_HERE, "results_revision2")
ANGLES = os.path.join(IN_DIR, "crossmethod_angles_v2.json")
INDICATORS = os.path.join(IN_DIR, "c7_crossmethod_indicators.csv")

EPS_DEG = 1e-6          # resolution at which two switching instants coincide
INSTANCES = [47, 49, 50]
ORDER = ["GWO-OP-SW", "COLSHADE", "DE", "WOA", "GWO", "PSO", "GA"]
LEVEL = {45: 3, 46: 5, 47: 7, 48: 9, 49: 11, 50: 13}


def proposed_angles(prob_k):
    df = pd.read_csv(RAW_CSV)
    sub = df[(df.algorithm == "GWO-OP-SW") & (df.prob_k == prob_k)
             & (df.feasible == True)]  # noqa: E712
    row = sub.loc[sub.best_f.idxmin()]
    return np.asarray(json.loads(row["best_alpha_json"]), dtype=float)


def separation(angles):
    gaps = np.diff(angles)
    distinct = int(len(np.unique(np.round(angles / EPS_DEG))))
    return dict(min_gap_deg=float(gaps.min()),
                n_gaps_below_eps=int(np.sum(gaps < EPS_DEG)),
                n_gaps_nonpositive=int(np.sum(gaps <= 0)),
                alpha_min=float(angles.min()), alpha_max=float(angles.max()),
                distinct=distinct)


def main():
    store = json.load(open(ANGLES, encoding="utf-8"))
    ind = pd.read_csv(INDICATORS)
    sep_rows, final_rows = [], []
    for prob_k in INSTANCES:
        for algo in ORDER:
            if algo == "GWO-OP-SW":
                ang = proposed_angles(prob_k)
            else:
                key = "%s|RC%d" % (algo, prob_k)
                if key not in store:
                    continue
                ang = np.asarray(store[key]["angles_deg"], dtype=float)
            s = separation(ang)
            sep_rows.append(dict(algorithm=algo, prob_k=prob_k,
                                 min_gap_deg=s["min_gap_deg"],
                                 n_gaps_below_eps=s["n_gaps_below_eps"],
                                 n_gaps_nonpositive=s["n_gaps_nonpositive"],
                                 alpha_min=s["alpha_min"], alpha_max=s["alpha_max"]))
            r = ind[(ind.algorithm == algo) & (ind.prob_k == prob_k)].iloc[0]
            final_rows.append(dict(prob_k=prob_k, algorithm=algo,
                                   V1_err=r.V1_abs_error, THD=r.THD_pct,
                                   worst_order=int(r.worst_targeted_order),
                                   worst_pct=r.worst_targeted_pct,
                                   D=len(ang), distinct=s["distinct"],
                                   min_gap=s["min_gap_deg"],
                                   gaps_below_eps=s["n_gaps_below_eps"]))

    sep, fin = pd.DataFrame(sep_rows), pd.DataFrame(final_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    sep.to_csv(os.path.join(OUT_DIR, "c7_min_separation.csv"), index=False)
    fin.to_csv(os.path.join(OUT_DIR, "c7_crossmethod_final.csv"), index=False)

    lines = [r"\begin{tabular}{llccccc}", r"\hline",
             r"Instance & Method & $|V_1$ err$|$ & THD & Worst targeted & "
             r"Distinct & Min separation \\",
             r" &  & (p.u.) & (\%) & harmonic (\% $V_1$) & instants & (deg) \\",
             r"\hline"]
    for prob_k in INSTANCES:
        sub = fin[fin.prob_k == prob_k]
        for i, (_, r) in enumerate(sub.iterrows()):
            head = (r"\multirow{%d}{*}{RC%d (%d-lvl)}" % (len(sub), prob_k, LEVEL[prob_k])
                    if i == 0 else "")
            name = r"\textbf{GWO-OP-SW}" if r.algorithm == "GWO-OP-SW" else r.algorithm
            gap = "0" if r.min_gap == 0 else "%.2e" % r.min_gap
            dist = "%d" % r.distinct
            if r.gaps_below_eps > 0:
                gap, dist = r"\textbf{%s}" % gap, r"\textbf{%s}" % dist
            lines.append(r"%s & %s & %.1e & %.2f & $h_{%d}$: %.2f & %s & %s \\"
                         % (head, name, r.V1_err, r.THD, r.worst_order,
                            r.worst_pct, dist, gap))
    lines += [r"\hline", r"\end{tabular}"]
    open(os.path.join(OUT_DIR, "table_c7_crossmethod_final.tex"), "w",
         encoding="utf-8").write("\n".join(lines) + "\n")
    print("Written to %s" % OUT_DIR)


if __name__ == "__main__":
    main()
