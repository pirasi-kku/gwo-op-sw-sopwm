"""
computational_overhead.py  (T4)

Quantifies the computational overhead of GWO-OP-SW vs. standard GWO
(decoding, cumulative-sum reparameterization, scaling, extra constraint
evaluation) -- answers Reviewer C comment C5. Uses `elapsed_sec` already
logged per run in results/raw_results.csv for every algorithm under the
identical protocol (MaxFEs=200,000, same hardware/process), so this is pure
analysis -- no rerun needed. COLSHADE is included too for extra context
even though C5 only asks for standard-GWO vs GWO-OP-SW.
"""
import os
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "..", "results")
RAW_CSV = os.path.join(RESULTS_DIR, "raw_results.csv")
OUT_DIR = os.path.join(_SCRIPT_DIR, "results_revision", "engineering")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(RAW_CSV)

    # --- per-instance overhead: GWO-OP-SW vs standard GWO ---
    rows = []
    for k in sorted(df.prob_k.unique()):
        gwo = df[(df.algorithm == "GWO") & (df.prob_k == k)]["elapsed_sec"]
        ops = df[(df.algorithm == "GWO-OP-SW") & (df.prob_k == k)]["elapsed_sec"]
        rows.append(dict(
            prob_k=k,
            GWO_mean_sec=gwo.mean(), GWO_std_sec=gwo.std(),
            GWO_OP_SW_mean_sec=ops.mean(), GWO_OP_SW_std_sec=ops.std(),
            overhead_sec=ops.mean() - gwo.mean(),
            overhead_pct=100.0 * (ops.mean() - gwo.mean()) / gwo.mean(),
        ))
    per_instance = pd.DataFrame(rows)

    # --- overall (pooled across all 180 runs each) ---
    gwo_all = df[df.algorithm == "GWO"]["elapsed_sec"]
    ops_all = df[df.algorithm == "GWO-OP-SW"]["elapsed_sec"]
    overall = dict(
        GWO_mean_sec=gwo_all.mean(), GWO_OP_SW_mean_sec=ops_all.mean(),
        overhead_sec=ops_all.mean() - gwo_all.mean(),
        overhead_pct=100.0 * (ops_all.mean() - gwo_all.mean()) / gwo_all.mean(),
    )

    # --- all algorithms, for context (mean elapsed_sec per instance) ---
    all_algo = df.groupby(["algorithm", "prob_k"])["elapsed_sec"].mean().unstack().round(2)

    per_instance.to_csv(os.path.join(OUT_DIR, "computational_overhead_per_instance.csv"), index=False)
    all_algo.to_csv(os.path.join(OUT_DIR, "elapsed_sec_all_algorithms.csv"))

    pd.set_option("display.width", 160)
    print("=== GWO-OP-SW vs standard GWO -- overhead per instance ===")
    print(per_instance.round(4).to_string(index=False))
    print()
    print("=== Overall (pooled 180 runs each) ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")
    print()
    print("=== Mean elapsed_sec, all algorithms x instance (context) ===")
    print(all_algo.to_string())
    print(f"\nSaved: computational_overhead_per_instance.csv, elapsed_sec_all_algorithms.csv in {OUT_DIR}/")


if __name__ == "__main__":
    main()
