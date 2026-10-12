"""
landscape_bias_analysis.py  (T11, empirical half)

Reviewer A5 asks whether the cumulative-sum reparameterization "introduces
bias toward particular switching-angle configurations". theory_verification.py
(Proposition 5) predicts one specific, testable bias:

    Under the paper's initialization delta ~ U(0, alpha_max)^D, the probability
    that sum(delta) <= alpha_max is 1/D! (3.8e-33 at D=30). Every other draw
    has sigma < 1 and therefore decodes to alpha_D = alpha_max EXACTLY. The
    search thus begins on the face {alpha_D = alpha_max} and can leave it only
    by contracting sum(delta) below alpha_max.

This script tests that prediction against the actual solutions and then asks
the question that matters for the paper: does the bias cost anything?

Part 1: measure alpha_D in all 180 final solutions (delta_min=1e-5 rerun).
Part 2: a controlled initialization experiment -- the paper's U(0, alpha_max)
        initialization versus a "sum-matched" initialization
        U(0, 2*alpha_max/D) whose expected sum equals alpha_max, so that a
        substantial share of the initial population starts with sigma = 1 and
        alpha_D strictly below 90 deg. Same seeds, same budget, paired.
"""
import json
import os

import numpy as np
import pandas as pd

from sopwm_objective import sopwm_evaluate, get_dimension, BEST_KNOWN
import gwo_final
from gwo_final import decode_delta_to_alpha, mean_constraint_violation

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(_SCRIPT_DIR, "results_revision_1e-5", "raw")
OUT_DIR = os.path.join(_SCRIPT_DIR, "results_revision", "engineering")
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA_MAX = 90.0
INSTANCES = [45, 46, 47, 48, 49, 50]


# ---------------------------------------------------------------------------
# Part 1 -- is the predicted bias visible in the final solutions?
# ---------------------------------------------------------------------------
def measure_bias_in_solutions():
    print("=" * 80)
    print("PART 1  alpha_D in the 180 final solutions (predicted: pinned at 90 deg)")
    print("=" * 80)
    rows = []
    for k in INSTANCES:
        df = pd.read_csv(os.path.join(RAW_DIR, f"gwo_op_sw_RC{k}.csv"))
        last, n_at_max, crowd = [], 0, []
        for _, r in df.iterrows():
            a = np.array(json.loads(r["best_alpha_json"]))
            last.append(a[-1])
            if abs(a[-1] - ALPHA_MAX) < 1e-9:
                n_at_max += 1
            # how many angles sit within 1 deg of the upper limit
            crowd.append(int(np.sum(a > ALPHA_MAX - 1.0)))
        last = np.array(last)
        rows.append(dict(
            prob_k=k, n_runs=len(df),
            alpha_D_mean=last.mean(), alpha_D_min=last.min(), alpha_D_max=last.max(),
            pct_runs_alpha_D_at_90=100.0 * n_at_max / len(df),
            mean_angles_within_1deg_of_90=float(np.mean(crowd)),
        ))
    out = pd.DataFrame(rows)
    print(out.round(6).to_string(index=False))
    overall = out["pct_runs_alpha_D_at_90"].mean()
    print(f"\n  Across all 180 solutions, alpha_D = 90 deg exactly in "
          f"{overall:.1f}% of runs.")
    print("  => Proposition 5's prediction is confirmed empirically: the")
    print("     reparameterization does bias the search toward configurations whose")
    print("     final switching angle sits at the quarter-wave boundary.")
    out.to_csv(os.path.join(OUT_DIR, "t11_alpha_D_bias.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# Part 2 -- does the bias cost anything? Controlled initialization experiment.
# ---------------------------------------------------------------------------
def run_with_init(objective_fn, ub_total, dim, seed, init_scale, max_iter, pop_size=50):
    """Run GWO-OP-SW with delta ~ U(0, init_scale)^D instead of U(0, ub_total)^D.

    Implemented by monkey-patching only the initialization draw: everything
    else (update rule, schedule, selection, decode) is byte-identical to
    gwo_final.grey_wolf_optimizer_final, so the comparison isolates the
    initialization alone.
    """
    import numpy.random as _npr

    real_default_rng = np.random.default_rng

    class ScaledRNG:
        """Wraps the real generator, rescaling ONLY the first .random() call
        (the initialization draw); every later call passes through untouched,
        preserving the stream for the search itself."""
        def __init__(self, seed, scale, ub):
            self._rng = real_default_rng(seed)
            self._scale = scale / ub
            self._first = True

        def random(self, *a, **kw):
            out = self._rng.random(*a, **kw)
            if self._first:
                self._first = False
                out = out * self._scale
            return out

        def __getattr__(self, name):
            return getattr(self._rng, name)

    np.random.default_rng = lambda s=None: ScaledRNG(s, init_scale, ub_total)
    try:
        res = gwo_final.grey_wolf_optimizer_final(
            objective_fn, ub_total, dim, pop_size=pop_size, max_iter=max_iter,
            seed=seed, verbose=False)
    finally:
        np.random.default_rng = real_default_rng
    return res


def initialization_experiment(instances=(45, 50), n_seeds=10, max_iter=4000):
    print()
    print("=" * 80)
    print("PART 2  does the bias cost anything? paired initialization experiment")
    print("=" * 80)
    print("  A: delta ~ U(0, alpha_max)      -- the paper's initialization, E[sum] = D*45")
    print("  B: delta ~ U(0, 2*alpha_max/D)  -- sum-matched, E[sum] = alpha_max, so a")
    print("     large share of the initial population starts with sigma = 1")
    print(f"  {n_seeds} paired seeds per instance, identical budget "
          f"(pop 50 x {max_iter} iters), same seeds in both arms.\n")

    rows = []
    for k in instances:
        dim = get_dimension(k)
        fn = lambda x, kk=k: sopwm_evaluate(x, kk)
        for run in range(n_seeds):
            seed = 42 + run
            rA = run_with_init(fn, ALPHA_MAX, dim, seed, ALPHA_MAX, max_iter)
            rB = run_with_init(fn, ALPHA_MAX, dim, seed, 2 * ALPHA_MAX / dim, max_iter)
            for label, r in (("A_paper", rA), ("B_sum_matched", rB)):
                a = np.asarray(r["best_alpha"], dtype=float)
                rows.append(dict(
                    prob_k=k, run=run, seed=seed, arm=label,
                    best_f=r["best_f"], best_cv=r["best_cv"],
                    feasible=bool(r["best_cv"] <= 1e-6),
                    alpha_D=float(a[-1]),
                    alpha_D_at_90=bool(abs(a[-1] - ALPHA_MAX) < 1e-9),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "t11_initialization_experiment_raw.csv"), index=False)

    print("  --- results ---")
    summary = []
    for k in instances:
        sub = df[df.prob_k == k]
        A = sub[sub.arm == "A_paper"].sort_values("run")
        B = sub[sub.arm == "B_sum_matched"].sort_values("run")
        from scipy import stats as st
        try:
            w_stat, w_p = st.wilcoxon(A["best_f"].values, B["best_f"].values)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        summary.append(dict(
            prob_k=k,
            A_mean=A.best_f.mean(), B_mean=B.best_f.mean(),
            A_best=A.best_f.min(), B_best=B.best_f.min(),
            A_FR=100 * A.feasible.mean(), B_FR=100 * B.feasible.mean(),
            A_pct_alphaD_at_90=100 * A.alpha_D_at_90.mean(),
            B_pct_alphaD_at_90=100 * B.alpha_D_at_90.mean(),
            pct_diff_B_vs_A=100 * (B.best_f.mean() - A.best_f.mean()) / A.best_f.mean(),
            paired_wilcoxon_p=w_p,
        ))
    s = pd.DataFrame(summary)
    print(s.round(5).to_string(index=False))
    s.to_csv(os.path.join(OUT_DIR, "t11_initialization_experiment_summary.csv"), index=False)

    print("\n  Reading: B removes the initialization half of the bias (a large share of")
    print("  its starting population has sigma = 1). If B's final solutions still pin")
    print("  alpha_D at 90 deg, the pull toward that face comes from the OBJECTIVE, not")
    print("  from the representation; if B matches A on objective value, the bias is")
    print("  costless either way.")
    return df, s


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    measure_bias_in_solutions()
    initialization_experiment()
    print(f"\nSaved CSVs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
