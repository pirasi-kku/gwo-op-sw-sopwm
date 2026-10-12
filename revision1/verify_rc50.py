"""
verify_rc50.py  (T5)

Independently verifies the claim that GWO-OP-SW improves on the published
best-known objective value for RC50 -- answers Reviewer A comment A8
(partial) and Reviewer C comment C9. "Independent" here means: take the
raw best_alpha (switching angles) reported by the optimizer, and
recompute f/g/h directly via sopwm_evaluate() from scratch -- not trusting
the optimizer's own internal bookkeeping of best_f/best_cv. Also checks
this across BOTH the original 30-run raw_results.csv and the T2 rerun
(revision1/results_revision/raw/gwo_op_sw_RC50.csv) since they use
slightly different EPS_MARGIN in gwo_final.py (1e-5 vs the corrected
1e-6) -- both should be feasible and the objective should match, but this
is flagged if not.

Explicitly avoids the language "best-known = optimum": best-known is only
the best value published in the CEC2020 literature (Kumar et al. 2020),
not a proven global optimum.
"""
import json
import os

import numpy as np
import pandas as pd

from sopwm_objective import sopwm_evaluate, mean_constraint_violation, BEST_KNOWN

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "..", "results")
RAW_CSV = os.path.join(RESULTS_DIR, "raw_results.csv")
RERUN_CSV = os.path.join(_SCRIPT_DIR, "results_revision_1e-5", "raw", "gwo_op_sw_RC50.csv")
OUT_DIR = os.path.join(_SCRIPT_DIR, "results_revision", "engineering")
os.makedirs(OUT_DIR, exist_ok=True)

PROB_K = 50


def independent_check(angles_deg, label):
    x = np.asarray(angles_deg, dtype=float).reshape(1, -1)
    f, g, h = sopwm_evaluate(x, PROB_K)
    mcv = mean_constraint_violation(g, h)
    ordering_ok = bool(np.all(np.diff(angles_deg) > 0))
    within_bounds = bool(np.all((angles_deg >= 0) & (angles_deg <= 90)))
    return dict(
        label=label,
        f_recomputed=float(f[0]),
        mean_cv_recomputed=float(mcv[0]),
        feasible_recomputed=bool(mcv[0] <= 1e-6),
        ordering_strictly_increasing=ordering_ok,
        within_bounds_0_90=within_bounds,
        max_ineq_g=float(np.max(g)),
        equality_h=float(h[0]),
    )


def main():
    results = []
    angle_records = {}

    # --- (a) original 30-run raw_results.csv: best feasible run for RC50 ---
    # (raw_results.csv only stores best_f/best_cv, not the angles themselves --
    #  so we can only re-verify the *rerun* set's angles directly. We report
    #  the original run's summary stats for reference/comparison.)
    df_orig = pd.read_csv(RAW_CSV)
    orig_rc50 = df_orig[(df_orig.algorithm == "GWO-OP-SW") & (df_orig.prob_k == PROB_K)]
    orig_best = orig_rc50.loc[orig_rc50["best_f"].idxmin()]

    # --- (b) T2 rerun: has best_alpha_json, so we CAN independently recompute ---
    df_rerun = pd.read_csv(RERUN_CSV)
    rerun_feas = df_rerun[df_rerun["feasible"] == True]  # noqa: E712
    rerun_best = rerun_feas.loc[rerun_feas["best_f"].idxmin()]
    angles = np.array(json.loads(rerun_best["best_alpha_json"]))
    angle_records["rerun_best_run"] = dict(run=int(rerun_best["run"]), angles_deg=angles.tolist())

    chk = independent_check(angles, "T2 rerun, best feasible run (angles independently recomputed)")
    chk["reported_best_f"] = float(rerun_best["best_f"])
    chk["reported_best_cv"] = float(rerun_best["best_cv"])
    chk["f_matches_reported"] = bool(abs(chk["f_recomputed"] - chk["reported_best_f"]) < 1e-9)
    results.append(chk)

    best_known = BEST_KNOWN[PROB_K]
    improvement_pct = 100.0 * (best_known - chk["f_recomputed"]) / best_known

    print("=== RC50 independent verification (T5) ===")
    print(f"Published best-known (Kumar et al. 2020, CEC2020 Table 3): {best_known:.10e}")
    print()
    print(f"Original 30-run set (raw_results.csv): best_f reported = {orig_best['best_f']:.10e}  "
          f"(feasible={orig_best['feasible']}) -- angles not stored, cannot independently recompute this one")
    print()
    print("T2 rerun set (has stored angles -- independently recomputed from scratch):")
    for kk, vv in chk.items():
        print(f"  {kk}: {vv}")
    print()
    if chk["feasible_recomputed"] and chk["f_recomputed"] < best_known:
        print(f"RESULT: independently recomputed f = {chk['f_recomputed']:.10e} IS feasible AND below the "
              f"published best-known value ({improvement_pct:.3f}% lower).")
        print("Caveat for the manuscript: 'best-known' denotes the best value reported in the CEC2020 "
              "literature, not a proven global optimum -- do not phrase this as beating the theoretical optimum.")
    elif chk["feasible_recomputed"]:
        print(f"RESULT: recomputed f = {chk['f_recomputed']:.10e} is feasible but NOT below best-known "
              f"({best_known:.10e}) -- the earlier claim of improvement does not hold for this run/seed set "
              "and should not be stated in the manuscript as-is.")
    else:
        print("RESULT: recomputed solution is NOT feasible under independent re-evaluation -- "
              "investigate before making any best-known comparison claim.")

    angles_str = ", ".join(f"{a:.6f}" for a in angles)
    print(f"\nSwitching angles (deg), run {angle_records['rerun_best_run']['run']}:")
    print(f"  [{angles_str}]")

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUT_DIR, "rc50_independent_verification.csv"), index=False)
    with open(os.path.join(OUT_DIR, "rc50_best_angles.json"), "w") as f:
        json.dump(angle_records, f, indent=2)
    print(f"\nSaved: rc50_independent_verification.csv, rc50_best_angles.json in {OUT_DIR}/")


if __name__ == "__main__":
    main()
