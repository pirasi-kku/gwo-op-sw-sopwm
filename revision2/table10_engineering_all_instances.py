"""
table10_engineering_all_instances.py -- the values of Table 10 of the
manuscript: engineering quantities and targeted-harmonic magnitudes of the best
feasible GWO-OP-SW solution on each of the six instances.

It applies the same indicator calculation as c7_crossmethod_engineering_v2.py
(fundamental, THD, the targeted orders 5-25 as a percentage of V1, and the
largest k^-4-weighted targeted order) to the stored GWO-OP-SW angles in
../results/raw_results.csv.

Output: results_revision2/table10_engineering_all_instances.csv

HOW TO RUN
  python revision2/table10_engineering_all_instances.py
"""

import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from c7_crossmethod_engineering_v2 import (  # noqa: E402
    indicators, stored_proposed_angles, SHOW_ORDERS, OUT_DIR,
)


def main():
    rows = []
    for prob_k in range(45, 51):
        d = indicators(stored_proposed_angles(prob_k), prob_k)
        tp = d.pop("targeted_pct")
        row = dict(prob_k=prob_k, level=d["level"], V1=d["V1"],
                   V1_target=d["V1_target"], V1_abs_error=d["V1_abs_error"],
                   THD_pct=d["THD_pct"],
                   transitions_per_cell_per_cycle=d["transitions_per_cell_per_cycle"])
        row.update({("h%d_pct" % n): tp[n] for n in SHOW_ORDERS})
        row["worst_targeted_order"] = d["worst_targeted_order"]
        row["worst_targeted_pct"] = d["worst_targeted_pct"]
        rows.append(row)
    out = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "table10_engineering_all_instances.csv")
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(out.round(4).T)
    print("Written to %s" % path)


if __name__ == "__main__":
    main()
