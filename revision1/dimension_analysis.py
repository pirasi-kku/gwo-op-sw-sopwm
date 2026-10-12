"""
dimension_analysis.py  (T8)

Quantifies how feasibility probability and objective quality relate to
problem dimension / number of ordering constraints -- answers Reviewer B
comment B7. Pure analysis of existing data (results/raw_results.csv, 8
algorithms x 6 instances x 30 runs, corrected EPS_MARGIN=1e-6 dataset),
plus the ablation datasets as a robustness check. No optimizer runs.

Structural note that governs the whole analysis (and must be stated in the
manuscript): across RC45-RC50 the structural parameters are

    RC45: D=25, j=1  (3-level)     RC48: D=30, j=4  (9-level)
    RC46: D=25, j=2  (5-level)     RC49: D=30, j=5  (11-level)
    RC47: D=25, j=3  (7-level)     RC50: D=30, j=6  (13-level)

so the number of switching angles D (25 -> 30), the number of ordering
constraints (D-1 = 24 -> 29) and the number of DC sources j (1 -> 6) all
increase together and are perfectly rank-correlated over these 6 instances.
They therefore CANNOT be statistically separated with this benchmark set --
any "difficulty increases with dimension" statement is really "difficulty
increases with this joint structural size". This script reports the
relationships and states that confound explicitly rather than pretending to
isolate one variable.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from sopwm_objective import _SOPWM_PARAMS, BEST_KNOWN

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
RAW_CSV = os.path.join(ROOT_DIR, "results", "raw_results.csv")
ABLATION_TUNING = os.path.join(ROOT_DIR, "results", "ablation", "tuning_raw_results.csv")
OUT_DIR = os.path.join(_SCRIPT_DIR, "results_revision", "engineering")
os.makedirs(OUT_DIR, exist_ok=True)

INSTANCES = [45, 46, 47, 48, 49, 50]


# ---------------------------------------------------------------------------
# 0. structural table
# ---------------------------------------------------------------------------
def structural_table():
    rows = []
    for k in INSTANCES:
        p = _SOPWM_PARAMS[k]
        D, j, m = p["D"], p["j"], p["m"]
        rows.append(dict(
            prob_k=k, D=D, n_ordering_constraints=D - 1, j_dc_sources=j,
            level_count=2 * j + 1, m=m, equality_target_j_times_m=j * m,
            best_known=BEST_KNOWN[k],
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. feasibility & quality vs structural size, per algorithm
# ---------------------------------------------------------------------------
def per_algorithm_trends(df, struct):
    rows = []
    for algo in sorted(df.algorithm.unique()):
        fr, ratio = [], []
        for k in INSTANCES:
            sub = df[(df.algorithm == algo) & (df.prob_k == k)]
            fr.append(100.0 * sub["feasible"].mean())
            feas = sub[sub["feasible"] == True]  # noqa: E712
            ratio.append(feas["best_f"].mean() / BEST_KNOWN[k] if len(feas) else np.nan)

        j = struct["j_dc_sources"].values.astype(float)
        fr = np.array(fr)
        ratio = np.array(ratio)

        # Spearman over the 6 instances (n=6 -- weak power, reported with that caveat)
        if len(set(fr)) > 1:
            rho_fr, p_fr = stats.spearmanr(j, fr)
        else:
            rho_fr, p_fr = np.nan, np.nan  # constant feasibility (e.g. 100% everywhere)
        ok = ~np.isnan(ratio)
        if ok.sum() >= 3 and len(set(ratio[ok])) > 1:
            rho_q, p_q = stats.spearmanr(j[ok], ratio[ok])
        else:
            rho_q, p_q = np.nan, np.nan

        rows.append(dict(
            algorithm=algo,
            FR_RC45=fr[0], FR_RC46=fr[1], FR_RC47=fr[2],
            FR_RC48=fr[3], FR_RC49=fr[4], FR_RC50=fr[5],
            FR_drop_first_to_last=fr[0] - fr[5],
            spearman_rho_FR_vs_size=rho_fr, spearman_p_FR_vs_size=p_fr,
            mean_ratio_RC45=ratio[0], mean_ratio_RC50=ratio[5],
            spearman_rho_quality_vs_size=rho_q, spearman_p_quality_vs_size=p_q,
            n_instances_with_feasible_runs=int(ok.sum()),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. run-level logistic regression of feasibility on structural size
# ---------------------------------------------------------------------------
def logistic_fit(x, y):
    """MLE logistic regression y ~ 1 + x. Returns (intercept, slope, se_slope)
    or None on perfect separation / no variation in y."""
    if len(set(y)) < 2:
        return None
    X = np.column_stack([np.ones_like(x, dtype=float), x.astype(float)])

    def nll(b):
        z = X @ b
        # numerically stable log-likelihood
        return np.sum(np.logaddexp(0, z) - y * z)

    res = minimize(nll, np.zeros(2), method="BFGS")
    b = res.x
    # Wald SE from the analytic Fisher information
    p = 1.0 / (1.0 + np.exp(-(X @ b)))
    W = p * (1 - p)
    info = X.T @ (X * W[:, None])
    try:
        cov = np.linalg.inv(info)
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se = float("nan")
    # A huge |slope| or a huge SE both mean the likelihood is (quasi-)separated:
    # the MLE is running off toward +/-inf and the Wald CI is meaningless.
    if not np.isfinite(se) or abs(b[1]) > 10 or se > 5:
        return dict(intercept=float(b[0]), slope=float(b[1]), se_slope=float("nan"),
                    separated=True)
    return dict(intercept=float(b[0]), slope=float(b[1]), se_slope=se, separated=False)


def feasibility_regression(df, struct):
    j_map = dict(zip(struct.prob_k, struct.j_dc_sources))
    rows = []
    for algo in sorted(df.algorithm.unique()):
        sub = df[df.algorithm == algo]
        x = sub["prob_k"].map(j_map).values.astype(float)
        y = sub["feasible"].astype(int).values
        fit = logistic_fit(x, y)
        if fit is None:
            rows.append(dict(algorithm=algo, n_runs=len(sub),
                             overall_FR_pct=100.0 * y.mean(),
                             slope_log_odds_per_unit_size=np.nan, se=np.nan,
                             ci95_lo=np.nan, ci95_hi=np.nan,
                             odds_ratio_per_unit_size=np.nan,
                             note="feasibility constant across all runs -- no slope estimable"))
            continue
        if fit["separated"]:
            rows.append(dict(algorithm=algo, n_runs=len(sub),
                             overall_FR_pct=100.0 * y.mean(),
                             slope_log_odds_per_unit_size=fit["slope"], se=np.nan,
                             ci95_lo=np.nan, ci95_hi=np.nan,
                             odds_ratio_per_unit_size=np.nan,
                             note="quasi-complete separation -- slope not reliably estimable"))
            continue
        lo = fit["slope"] - 1.96 * fit["se_slope"]
        hi = fit["slope"] + 1.96 * fit["se_slope"]
        rows.append(dict(algorithm=algo, n_runs=len(sub),
                         overall_FR_pct=100.0 * y.mean(),
                         slope_log_odds_per_unit_size=fit["slope"], se=fit["se_slope"],
                         ci95_lo=lo, ci95_hi=hi,
                         odds_ratio_per_unit_size=float(np.exp(fit["slope"])),
                         note=""))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. quality (feasible runs only) vs structural size
# ---------------------------------------------------------------------------
def quality_regression(df, struct):
    j_map = dict(zip(struct.prob_k, struct.j_dc_sources))
    rows = []
    for algo in sorted(df.algorithm.unique()):
        sub = df[(df.algorithm == algo) & (df["feasible"] == True)].copy()  # noqa: E712
        if len(sub) < 10 or sub["prob_k"].nunique() < 3:
            rows.append(dict(algorithm=algo, n_feasible_runs=len(sub),
                             n_instances=sub["prob_k"].nunique(),
                             slope_log_ratio_per_unit_size=np.nan, p_value=np.nan,
                             note="too few feasible runs/instances for a meaningful fit"))
            continue
        sub["ratio"] = sub.apply(lambda r: r["best_f"] / BEST_KNOWN[r["prob_k"]], axis=1)
        x = sub["prob_k"].map(j_map).values.astype(float)
        y = np.log(sub["ratio"].values)
        lr = stats.linregress(x, y)
        rows.append(dict(algorithm=algo, n_feasible_runs=len(sub),
                         n_instances=sub["prob_k"].nunique(),
                         slope_log_ratio_per_unit_size=lr.slope, p_value=lr.pvalue,
                         pct_change_per_unit_size=100.0 * (np.exp(lr.slope) - 1),
                         note="feasible runs only -- survivorship bias where FR is low"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. robustness check on the ablation dataset (GWO-OP-SW family, 50 runs/cell)
# ---------------------------------------------------------------------------
def ablation_check(struct):
    if not os.path.exists(ABLATION_TUNING):
        return None
    ab = pd.read_csv(ABLATION_TUNING)
    j_map = dict(zip(struct.prob_k, struct.j_dc_sources))
    rows = []
    for cfg in sorted(ab.config.unique()):
        sub = ab[ab.config == cfg]
        fr = [100.0 * sub[sub.prob_k == k]["feasible"].mean() for k in INSTANCES]
        feas = sub[sub["feasible"] == True]  # noqa: E712
        ratio = [feas[feas.prob_k == k]["best_f"].mean() / BEST_KNOWN[k] for k in INSTANCES]
        j = np.array([j_map[k] for k in INSTANCES], dtype=float)
        rho, p = stats.spearmanr(j, ratio)
        rows.append(dict(config=cfg, min_FR_pct=min(fr), max_FR_pct=max(fr),
                         mean_ratio_RC45=ratio[0], mean_ratio_RC50=ratio[5],
                         spearman_rho_quality_vs_size=rho, spearman_p=p))
    return pd.DataFrame(rows)


def main():
    struct = structural_table()
    df = pd.read_csv(RAW_CSV)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("=" * 100)
    print("0. STRUCTURAL PARAMETERS of the 6 benchmark instances")
    print("=" * 100)
    print(struct.to_string(index=False))
    rho_dj, _ = stats.spearmanr(struct.D, struct.j_dc_sources)
    print(f"\nSpearman rho between D and j across the 6 instances = {rho_dj:.4f}")
    print("=> D, the ordering-constraint count (D-1), j and the level count all increase")
    print("   together. They are CONFOUNDED in this benchmark set and cannot be separated;")
    print("   report the relationship as 'structural size' rather than 'dimension' alone.")

    print()
    print("=" * 100)
    print("1. FEASIBILITY & QUALITY TRENDS per algorithm (n=6 instances -- weak power)")
    print("=" * 100)
    trends = per_algorithm_trends(df, struct)
    print(trends.round(4).to_string(index=False))

    print()
    print("=" * 100)
    print("2. RUN-LEVEL LOGISTIC REGRESSION: P(feasible) ~ structural size (n=180 runs/algo)")
    print("=" * 100)
    logit = feasibility_regression(df, struct)
    print(logit.round(4).to_string(index=False))
    print("\nNote: runs cluster into only 6 distinct structural sizes, so the Wald CIs are")
    print("approximate (they treat runs as independent). Direction and magnitude are the")
    print("interpretable part, not the exact p-value.")

    print()
    print("=" * 100)
    print("3. QUALITY vs structural size (feasible runs only, log(f / best_known))")
    print("=" * 100)
    qual = quality_regression(df, struct)
    print(qual.round(4).to_string(index=False))
    print("\nCaution: algorithms with low feasibility rates are scored only on the runs")
    print("that happened to stay feasible -- a survivorship bias that FLATTERS them here.")
    print("Feasibility (section 2) and quality (this section) must be read together.")

    print()
    print("=" * 100)
    print("4. ROBUSTNESS CHECK on the ablation dataset (5 SW configs, 50 runs/cell)")
    print("=" * 100)
    ab = ablation_check(struct)
    if ab is not None:
        print(ab.round(4).to_string(index=False))
    else:
        print("(ablation tuning dataset not found -- skipped)")

    struct.to_csv(os.path.join(OUT_DIR, "dimension_structural_table.csv"), index=False)
    trends.to_csv(os.path.join(OUT_DIR, "dimension_trends_per_algorithm.csv"), index=False)
    logit.to_csv(os.path.join(OUT_DIR, "dimension_feasibility_logistic.csv"), index=False)
    qual.to_csv(os.path.join(OUT_DIR, "dimension_quality_regression.csv"), index=False)
    if ab is not None:
        ab.to_csv(os.path.join(OUT_DIR, "dimension_ablation_check.csv"), index=False)
    print(f"\nSaved CSVs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
