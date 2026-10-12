"""
schedule_param_justification.py  (T9)

Documents and stress-tests how the square-wave (SW) convergence-factor
schedule and its parameters were selected -- answers Reviewer A comment A6
("were the SW parameters tuned on the same instances used for final
reporting?").

FACTUAL STARTING POINT (read directly from the code, not assumed):
  - run_ablation_tuning.py  : PROB_LIST = range(45, 51)  -> RC45..RC50
        CONFIGS = baseline       (periods=10, duty_high=0.5, a_low=0.10)
                  more_periods   (periods=20, duty_high=0.5, a_low=0.10)
                  short_high     (periods=10, duty_high=0.3, a_low=0.10)
                  deeper_exploit (periods=10, duty_high=0.5, a_low=0.01)
                  combo          (periods=15, duty_high=0.3, a_low=0.01)  <- adopted
        N_RUNS=50, POP_SIZE=50, MAX_ITER=4000 (= 200,000 FEs), seed=42+run
  - run_ablation_schedule.py: PROB_LIST = range(45, 51)  -> RC45..RC50
        SCHEDULES = linear / cosine / oscillating / square_wave / adaptive_switch
        N_RUNS=50, same protocol

So the honest answer to A6 is: YES -- both the schedule family and its
parameters were selected on RC45-RC50, i.e. the *same* six instances used
for final reporting. There is no separate tuning set. This script does not
hide that; it quantifies how much that matters, via:

  (1) leave-one-instance-out (LOIO) selection: for each held-out instance,
      re-select the winning config/schedule using ONLY the other five, then
      score that choice on the held-out instance. If the same choice keeps
      winning out-of-sample, the in-sample selection did not manufacture the
      reported advantage.
  (2) how large the spread between candidate settings is at all (if every
      candidate performs within a couple of percent, the selection decision
      cannot be inflating the headline results).
  (3) per-instance ranks of the adopted setting, so the paper can state
      plainly where it does and does not win.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
TUNING_CSV = os.path.join(ROOT_DIR, "results", "ablation", "tuning_raw_results.csv")
SCHEDULE_CSV = os.path.join(ROOT_DIR, "results", "ablation", "schedule_raw_results.csv")
OUT_DIR = os.path.join(_SCRIPT_DIR, "results_revision", "engineering")
os.makedirs(OUT_DIR, exist_ok=True)

INSTANCES = [45, 46, 47, 48, 49, 50]
ADOPTED_CONFIG = "combo"
ADOPTED_SCHEDULE = "square_wave"


def cell_stat(df, key_col, key, prob_k, criterion="mean"):
    """Objective statistic over feasible runs for one (setting, instance) cell.

    criterion="mean": mean of the per-run best_f  (robust, distribution-level)
    criterion="best": min of the per-run best_f   (best-of-N -- the criterion
                      gwo_final.py's docstring says was actually used to pick
                      'combo'; reported here so the selection can be judged on
                      its own stated terms as well as on the more robust one)
    """
    sub = df[(df[key_col] == key) & (df.prob_k == prob_k) & (df["feasible"] == True)]  # noqa: E712
    if len(sub) == 0:
        return np.nan
    return sub["best_f"].mean() if criterion == "mean" else sub["best_f"].min()


def cell_mean(df, key_col, key, prob_k):
    return cell_stat(df, key_col, key, prob_k, "mean")


def per_instance_ranks(df, key_col, adopted, criterion="mean"):
    """Rank every candidate setting within each instance (1 = best)."""
    keys = sorted(df[key_col].unique())
    rows = []
    for k in INSTANCES:
        means = {key: cell_stat(df, key_col, key, k, criterion) for key in keys}
        order = sorted(keys, key=lambda key: means[key])
        rank_of = {key: i + 1 for i, key in enumerate(order)}
        spread_pct = 100.0 * (max(means.values()) - min(means.values())) / min(means.values())
        row = dict(prob_k=k, winner=order[0],
                   adopted=adopted, adopted_rank=rank_of[adopted],
                   adopted_mean=means[adopted], best_mean=means[order[0]],
                   adopted_vs_best_pct=100.0 * (means[adopted] - means[order[0]]) / means[order[0]],
                   spread_worst_vs_best_pct=spread_pct)
        for key in keys:
            row[f"mean_{key}"] = means[key]
        rows.append(row)
    return pd.DataFrame(rows)


def loio_selection(df, key_col, adopted, criterion="mean"):
    """Leave-one-instance-out selection: choose the setting with the best mean
    rank over the other 5 instances, then score it on the held-out one."""
    keys = sorted(df[key_col].unique())
    # rank matrix: instance x setting
    rank_mat = pd.DataFrame(index=INSTANCES, columns=keys, dtype=float)
    mean_mat = pd.DataFrame(index=INSTANCES, columns=keys, dtype=float)
    for k in INSTANCES:
        means = {key: cell_stat(df, key_col, key, k, criterion) for key in keys}
        mean_mat.loc[k] = pd.Series(means)
        rank_mat.loc[k] = pd.Series(means).rank()

    rows = []
    for held in INSTANCES:
        train = [k for k in INSTANCES if k != held]
        avg_rank_train = rank_mat.loc[train].mean()
        selected = avg_rank_train.idxmin()

        held_means = mean_mat.loc[held]
        held_rank = held_means.rank()

        # paired comparison on the held-out instance: selected vs adopted
        if selected != adopted:
            a = df[(df[key_col] == selected) & (df.prob_k == held)].sort_values("run")
            b = df[(df[key_col] == adopted) & (df.prob_k == held)].sort_values("run")
            merged = a.merge(b, on=["prob_k", "run", "seed"], suffixes=("_sel", "_ado"))
            both_feas = merged[(merged["feasible_sel"] == True) &  # noqa: E712
                               (merged["feasible_ado"] == True)]   # noqa: E712
            if len(both_feas) >= 10:
                try:
                    w_stat, w_p = stats.wilcoxon(both_feas["best_f_sel"], both_feas["best_f_ado"])
                except ValueError:
                    w_stat, w_p = np.nan, np.nan
            else:
                w_stat, w_p = np.nan, np.nan
            n_pairs = len(both_feas)
        else:
            w_stat, w_p, n_pairs = np.nan, np.nan, np.nan

        rows.append(dict(
            held_out_instance=held,
            loio_selected=selected,
            same_as_adopted=bool(selected == adopted),
            avg_rank_of_selected_on_train=float(avg_rank_train[selected]),
            avg_rank_of_adopted_on_train=float(avg_rank_train[adopted]),
            heldout_mean_selected=float(held_means[selected]),
            heldout_mean_adopted=float(held_means[adopted]),
            heldout_pct_diff_adopted_vs_selected=float(
                100.0 * (held_means[adopted] - held_means[selected]) / held_means[selected]),
            heldout_rank_of_adopted=float(held_rank[adopted]),
            paired_wilcoxon_p_on_heldout=w_p, n_paired=n_pairs,
        ))
    return pd.DataFrame(rows), mean_mat, rank_mat


def report(title, df, key_col, adopted, criterion="mean"):
    print("=" * 100)
    print(f"{title}   [selection criterion: {criterion}]")
    print("=" * 100)

    ranks = per_instance_ranks(df, key_col, adopted, criterion)
    print(f"\n-- per-instance ranks (1 = best {criterion} f among feasible runs), "
          f"adopted = '{adopted}' --")
    show_cols = ["prob_k", "winner", "adopted_rank", "adopted_mean", "best_mean",
                 "adopted_vs_best_pct", "spread_worst_vs_best_pct"]
    print(ranks[show_cols].round(4).to_string(index=False))
    print(f"\nAdopted setting was the per-instance winner on "
          f"{int((ranks.adopted_rank == 1).sum())}/6 instances; "
          f"mean rank = {ranks.adopted_rank.mean():.2f} of {df[key_col].nunique()} candidates.")
    print(f"Worst-vs-best spread between candidate settings, averaged over instances: "
          f"{ranks.spread_worst_vs_best_pct.mean():.2f}% "
          f"(range {ranks.spread_worst_vs_best_pct.min():.2f}% - "
          f"{ranks.spread_worst_vs_best_pct.max():.2f}%)")

    loio, mean_mat, rank_mat = loio_selection(df, key_col, adopted, criterion)
    print("\n-- leave-one-instance-out selection (select on 5, score on the 6th) --")
    print(loio.round(4).to_string(index=False))
    n_same = int(loio.same_as_adopted.sum())
    print(f"\nLOIO re-selected '{adopted}' in {n_same}/6 folds.")
    if n_same < 6:
        diffs = loio[~loio.same_as_adopted]["heldout_pct_diff_adopted_vs_selected"]
        print(f"In the {6 - n_same} fold(s) where a different setting was selected, the adopted "
              f"setting was {diffs.mean():+.2f}% (mean) on the held-out instance relative to "
              f"that selection (positive = adopted is worse).")
    print(f"\n-- {criterion} f per (instance x setting) --")
    print(mean_mat.astype(float).round(5).to_string())
    print("\n-- Friedman test across settings (n=6 instances, on per-instance means) --")
    try:
        fs, fp = stats.friedmanchisquare(*[mean_mat[c].values.astype(float)
                                           for c in mean_mat.columns])
        print(f"Friedman chi-square = {fs:.4f}, p = {fp:.6f} "
              f"({'settings differ' if fp < 0.05 else 'no detectable difference between settings'})")
    except Exception as exc:  # pragma: no cover
        print(f"Friedman test not computable: {exc}")
    print()
    return ranks, loio, mean_mat


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    print("SELECTION PROTOCOL AS IMPLEMENTED (read from run_ablation_tuning.py /")
    print("run_ablation_schedule.py -- both use PROB_LIST = RC45..RC50, N_RUNS=50,")
    print("POP_SIZE=50, MAX_ITER=4000 = 200,000 FEs, seed = 42 + run):")
    print("  => The SW schedule family AND its parameters were selected on the SAME six")
    print("     instances used for final reporting. There is no held-out tuning set.")
    print("     This must be disclosed in the manuscript (A6). The analyses below")
    print("     quantify how much that selection could have inflated the results.")
    print()

    tuning = pd.read_csv(TUNING_CSV)
    sched = pd.read_csv(SCHEDULE_CSV)

    print("NOTE ON THE SELECTION CRITERION: gwo_final.py's own docstring states that")
    print("'combo' was chosen because it beat the runner-up on the BEST-of-N metric on")
    print("every problem (~20% better on average) while tying on the MEAN. Both criteria")
    print("are therefore evaluated below -- 'mean' (robust, distribution-level) and")
    print("'best' (best-of-N, the criterion actually used). Where they disagree, that")
    print("disagreement is itself the thing the manuscript needs to disclose.")
    print()

    outputs = {}
    for crit in ("mean", "best"):
        t_ranks, t_loio, t_means = report(
            "A. SW PARAMETER SELECTION -- 5 configs (periods / duty_high / a_low)",
            tuning, "config", ADOPTED_CONFIG, crit)
        s_ranks, s_loio, s_means = report(
            "B. SCHEDULE-FAMILY SELECTION -- 5 a-decay schedules",
            sched, "schedule", ADOPTED_SCHEDULE, crit)
        outputs[crit] = (t_ranks, t_loio, t_means, s_ranks, s_loio, s_means)

        t_ranks.to_csv(os.path.join(OUT_DIR, f"t9_config_ranks_per_instance_{crit}.csv"), index=False)
        t_loio.to_csv(os.path.join(OUT_DIR, f"t9_config_loio_selection_{crit}.csv"), index=False)
        t_means.to_csv(os.path.join(OUT_DIR, f"t9_config_matrix_{crit}.csv"))
        s_ranks.to_csv(os.path.join(OUT_DIR, f"t9_schedule_ranks_per_instance_{crit}.csv"), index=False)
        s_loio.to_csv(os.path.join(OUT_DIR, f"t9_schedule_loio_selection_{crit}.csv"), index=False)
        s_means.to_csv(os.path.join(OUT_DIR, f"t9_schedule_matrix_{crit}.csv"))

    print("=" * 100)
    print("VERDICT SUMMARY")
    print("=" * 100)
    for crit in ("mean", "best"):
        t_ranks, t_loio, _, s_ranks, s_loio, _ = outputs[crit]
        print(f"[criterion = {crit}]")
        print(f"  config '{ADOPTED_CONFIG}': per-instance wins {int((t_ranks.adopted_rank==1).sum())}/6, "
              f"mean rank {t_ranks.adopted_rank.mean():.2f}/5, "
              f"LOIO re-selects it {int(t_loio.same_as_adopted.sum())}/6 folds")
        print(f"  schedule '{ADOPTED_SCHEDULE}': per-instance wins {int((s_ranks.adopted_rank==1).sum())}/6, "
              f"mean rank {s_ranks.adopted_rank.mean():.2f}/5, "
              f"LOIO re-selects it {int(s_loio.same_as_adopted.sum())}/6 folds")
    print(f"\nSaved 12 CSVs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
