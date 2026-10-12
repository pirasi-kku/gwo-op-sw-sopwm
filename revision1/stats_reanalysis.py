"""
stats_reanalysis.py  (T6)

Deeper statistics on the EXISTING result files (no rerun needed):
  - Per (algorithm, instance) descriptive stats: n, n_feasible, feasibility
    rate, mean/median/IQR/SD of best_f (feasible runs only), 95% bootstrap
    CI of the mean.                                                    -> A7
  - Per-instance pairwise comparison GWO-OP-SW vs every other algorithm
    (Mann-Whitney U on feasible best_f, independent samples), with a
    rank-biserial effect size and Holm-Bonferroni-adjusted p-values
    within each instance's family of 7 comparisons.                    -> C8
  - Updated aggregate Friedman + paired Wilcoxon-signed-rank-by-instance-
    rank test (now including COLSHADE, 8 algorithms), with Holm-adjusted
    p-values, and an explicit statement of the n=6 statistical-power
    limitation.                                                        -> A7, C8
  - Ablation: combo vs. short-high SW-schedule configuration, paired
    Wilcoxon signed-rank (same seeds per run across configs) + effect
    size, per instance and pooled, both raw and Holm-adjusted.         -> B6
  - Penalty-multiplier sensitivity re-analysis: since raw_results.csv only
    stores best_f/best_cv for FEASIBLE runs as the reported value (the
    original run_all_experiments.py protocol does not apply a fixed
    numeric penalty to infeasible runs in the stored CSV -- feasibility is
    tracked as a separate boolean), the "10x worst-feasible" penalty
    Reviewer B refers to is a *reporting-time* convention, not baked into
    this CSV. This script reconstructs it explicitly with 3 alternative
    multipliers (5x, 10x, 20x) to show the ranking is stable regardless of
    the exact multiplier chosen.                                       -> B5
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "..", "results")
RAW_CSV = os.path.join(RESULTS_DIR, "raw_results.csv")
ABLATION_RAW_CSV = os.path.join(RESULTS_DIR, "ablation", "tuning_raw_results.csv")
OUT_DIR = os.path.join(_SCRIPT_DIR, "results_revision", "engineering")
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(0)
N_BOOT = 5000
ALPHA = 0.05


def bootstrap_ci_mean(values, n_boot=N_BOOT, alpha=ALPHA):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (np.nan, np.nan)
    boot_means = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        sample = values[RNG.integers(0, n, size=n)]
        boot_means[b] = sample.mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (lo, hi)


def rank_biserial_from_mannwhitney(x, y):
    """Effect size for Mann-Whitney U: r = 1 - 2U/(n1*n2), range [-1,1].
    Positive r means x tends to be smaller than y (here: x=GWO-OP-SW is better/lower)."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return 1.0 - 2.0 * u / (n1 * n2)


def holm_adjust(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the ORIGINAL order of `pvals`."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((n - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


# ============================================================================
# 1. Descriptive stats per (algorithm, instance)
# ============================================================================
def descriptive_stats(df):
    rows = []
    for (algo, k), sub in df.groupby(["algorithm", "prob_k"]):
        n_total = len(sub)
        feas = sub[sub.feasible]
        n_feas = len(feas)
        vals = feas["best_f"].values
        if n_feas > 0:
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            mean_, sd_ = vals.mean(), vals.std(ddof=1) if n_feas > 1 else 0.0
            ci_lo, ci_hi = bootstrap_ci_mean(vals)
        else:
            q1 = med = q3 = mean_ = sd_ = ci_lo = ci_hi = np.nan
        rows.append(dict(
            algorithm=algo, prob_k=k, n_total=n_total, n_feasible=n_feas,
            feasibility_pct=100.0 * n_feas / n_total,
            mean_f=mean_, median_f=med, q1_f=q1, q3_f=q3, iqr_f=(q3 - q1) if n_feas > 0 else np.nan,
            sd_f=sd_, ci95_lo=ci_lo, ci95_hi=ci_hi,
        ))
    return pd.DataFrame(rows)


# ============================================================================
# 2. Per-instance pairwise: GWO-OP-SW vs each other algorithm (feasible-only)
# ============================================================================
def pairwise_per_instance(df):
    algos = sorted(a for a in df.algorithm.unique() if a != "GWO-OP-SW")
    all_rows = []
    for k in sorted(df.prob_k.unique()):
        ref = df[(df.algorithm == "GWO-OP-SW") & (df.prob_k == k) & df.feasible]["best_f"].values
        pvals, rows_this_instance = [], []
        for algo in algos:
            other = df[(df.algorithm == algo) & (df.prob_k == k) & df.feasible]["best_f"].values
            if len(ref) < 2 or len(other) < 2:
                rows_this_instance.append(dict(
                    prob_k=k, algorithm=algo, n_ref_feasible=len(ref), n_other_feasible=len(other),
                    p_value=np.nan, effect_size_rank_biserial=np.nan,
                    note="insufficient feasible samples for a valid test",
                ))
                pvals.append(np.nan)
                continue
            u_stat, p = stats.mannwhitneyu(ref, other, alternative="two-sided")
            r = rank_biserial_from_mannwhitney(ref, other)
            rows_this_instance.append(dict(
                prob_k=k, algorithm=algo, n_ref_feasible=len(ref), n_other_feasible=len(other),
                p_value=p, effect_size_rank_biserial=r, note="",
            ))
            pvals.append(p)
        valid_mask = ~np.isnan(pvals)
        if valid_mask.any():
            adj = holm_adjust(np.array(pvals)[valid_mask])
            it = iter(adj)
            for i, row in enumerate(rows_this_instance):
                if valid_mask[i]:
                    row["p_value_holm_adjusted"] = next(it)
                else:
                    row["p_value_holm_adjusted"] = np.nan
        all_rows.extend(rows_this_instance)
    return pd.DataFrame(all_rows)


# ============================================================================
# 3. Aggregate: Friedman + paired Wilcoxon-signed-rank on per-instance ranks
# ============================================================================
def aggregate_rank_tests(df):
    # mean best_f per (algorithm, instance), feasible-only; algorithms with
    # zero feasible runs on an instance get rank = worst (n_algos) for that
    # instance, consistent with "cannot solve this instance at all"
    algos = sorted(df.algorithm.unique())
    pivot = df[df.feasible].groupby(["algorithm", "prob_k"])["best_f"].mean().unstack()
    pivot = pivot.reindex(index=algos)
    ranks = pivot.rank(axis=0, method="average", na_option="bottom")

    friedman_stat, friedman_p = stats.friedmanchisquare(*[ranks.loc[a].values for a in algos])

    other_algos = [a for a in algos if a != "GWO-OP-SW"]
    wil_rows, pvals = [], []
    for algo in other_algos:
        stat, p = stats.wilcoxon(ranks.loc["GWO-OP-SW"], ranks.loc[algo])
        wil_rows.append(dict(comparison=f"GWO-OP-SW vs {algo}", statistic=stat, p_value=p))
        pvals.append(p)
    adj = holm_adjust(pvals)
    for row, a in zip(wil_rows, adj):
        row["p_value_holm_adjusted"] = a

    avg_rank = ranks.mean(axis=1).sort_values()

    summary_lines = [
        f"Friedman chi-square across {ranks.shape[1]} problem instances (n=6), {len(algos)} algorithms: "
        f"statistic={friedman_stat:.4f}, p={friedman_p:.4e}",
        "",
        "IMPORTANT LIMITATION: n=6 problem instances gives very limited statistical power for the "
        "paired/Friedman-style tests above -- p-values from only 6 paired observations should be read as "
        "indicative, not confirmatory. The per-instance Mann-Whitney tests (section 2, n=30 runs each) are "
        "the statistically better-powered comparison and should be given more weight in the response letter.",
    ]
    return (pd.DataFrame(wil_rows), avg_rank.reset_index().rename(columns={0: "avg_rank", "index": "algorithm"}),
            "\n".join(summary_lines))


# ============================================================================
# 4. Ablation: combo vs short_high (paired by seed/run index)
# ============================================================================
def ablation_combo_vs_shorthigh():
    df = pd.read_csv(ABLATION_RAW_CSV)
    sub = df[df.config.isin(["combo", "short_high"])]
    rows, pvals = [], []
    for k in sorted(sub.prob_k.unique()):
        piv = sub[sub.prob_k == k].pivot(index="run", columns="config", values="best_f").dropna()
        combo, short_high = piv["combo"].values, piv["short_high"].values
        stat, p = stats.wilcoxon(combo, short_high)
        diff_pct = 100.0 * (combo.mean() - short_high.mean()) / short_high.mean()
        # matched-pairs rank-biserial effect size
        d = combo - short_high
        r_rb = (np.sum(d < 0) - np.sum(d > 0)) / np.sum(d != 0) if np.any(d != 0) else 0.0
        rows.append(dict(prob_k=k, n_pairs=len(piv), combo_mean=combo.mean(),
                          short_high_mean=short_high.mean(), mean_diff_pct=diff_pct,
                          wilcoxon_statistic=stat, p_value=p, effect_size_matched_rank_biserial=r_rb))
        pvals.append(p)
    adj = holm_adjust(pvals)
    for row, a in zip(rows, adj):
        row["p_value_holm_adjusted"] = a
    return pd.DataFrame(rows)


# ============================================================================
# 5. Penalty-multiplier sensitivity (B5)
# ============================================================================
def penalty_sensitivity(df):
    rows = []
    for k in sorted(df.prob_k.unique()):
        sub = df[df.prob_k == k]
        worst_feasible = sub[sub.feasible]["best_f"].max()
        for mult in [5, 10, 20]:
            penalty_val = mult * worst_feasible
            scored = sub.copy()
            scored["scored_f"] = np.where(scored.feasible, scored.best_f, penalty_val)
            ranking = scored.groupby("algorithm")["scored_f"].mean().sort_values()
            rows.append(dict(prob_k=k, multiplier=mult, worst_feasible=worst_feasible,
                              ranking_top1=ranking.index[0], ranking_top1_value=ranking.iloc[0],
                              gwo_op_sw_rank=int(np.where(ranking.index == "GWO-OP-SW")[0][0]) + 1))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(RAW_CSV)

    desc = descriptive_stats(df)
    desc.to_csv(os.path.join(OUT_DIR, "descriptive_stats_per_instance.csv"), index=False)

    pairwise = pairwise_per_instance(df)
    pairwise.to_csv(os.path.join(OUT_DIR, "pairwise_mannwhitney_per_instance.csv"), index=False)

    wil_df, avg_rank_df, limitation_note = aggregate_rank_tests(df)
    wil_df.to_csv(os.path.join(OUT_DIR, "aggregate_friedman_wilcoxon.csv"), index=False)
    avg_rank_df.to_csv(os.path.join(OUT_DIR, "aggregate_avg_rank.csv"), index=False)

    ablation_df = ablation_combo_vs_shorthigh()
    ablation_df.to_csv(os.path.join(OUT_DIR, "ablation_combo_vs_shorthigh.csv"), index=False)

    penalty_df = penalty_sensitivity(df)
    penalty_df.to_csv(os.path.join(OUT_DIR, "penalty_sensitivity.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)

    print("=" * 100)
    print("1. DESCRIPTIVE STATS per (algorithm, instance) -- median/IQR/SD/95% bootstrap CI")
    print("=" * 100)
    print(desc.round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("2. PAIRWISE GWO-OP-SW vs each algorithm, per instance (Mann-Whitney U, feasible-only)")
    print("=" * 100)
    print(pairwise.round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("3. AGGREGATE Friedman + paired Wilcoxon-on-ranks (8 algorithms incl. COLSHADE)")
    print("=" * 100)
    print(wil_df.round(6).to_string(index=False))
    print()
    print(avg_rank_df.round(3).to_string(index=False))
    print()
    print(limitation_note)

    print("\n" + "=" * 100)
    print("4. ABLATION: combo vs short_high SW-schedule config (paired Wilcoxon, n=50 runs/instance)")
    print("=" * 100)
    print(ablation_df.round(4).to_string(index=False))
    pooled_diff = 100.0 * (ablation_df.combo_mean.mean() - ablation_df.short_high_mean.mean()) / ablation_df.short_high_mean.mean()
    print(f"\nPooled mean diff across 6 instances: combo vs short_high = {pooled_diff:.3f}%")
    n_sig = (ablation_df.p_value_holm_adjusted < ALPHA).sum()
    print(f"Instances where combo vs short_high difference is significant after Holm correction (p<0.05): "
          f"{n_sig}/6")

    print("\n" + "=" * 100)
    print("5. PENALTY-MULTIPLIER SENSITIVITY (B5) -- does the ranking change with the multiplier?")
    print("=" * 100)
    print(penalty_df.round(4).to_string(index=False))
    stable = (penalty_df.groupby("prob_k")["ranking_top1"].nunique() == 1).all()
    print(f"\nTop-ranked algorithm identical across all 3 multipliers (5x/10x/20x) for every instance: {stable}")

    print(f"\nAll CSVs saved in {OUT_DIR}/")


if __name__ == "__main__":
    main()
