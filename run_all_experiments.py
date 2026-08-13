"""
run_all_experiments.py  (v2 — เพิ่ม DE, GA, HHO + แก้บั๊ก statistical test + เพิ่ม
literature baseline table)

รันครบ 7 อัลกอริทึม (GWO-OP-SW ของเรา + GWO, PSO, WOA, DE, GA, HHO เป็น
same-protocol baseline) x 6 ปัญหา (RC45-RC50) x 30 runs พร้อม checkpoint/resume,
CSV, กราฟ, และ Friedman + Wilcoxon test ที่แก้ปัญหา NaN แล้ว

โครงสร้างไฟล์ผลลัพธ์:
  results/raw_results.csv          <- ผลดิบทุก run (checkpoint หลักที่ใช้ resume)
  results/summary_table.csv        <- ตาราง best/mean/std/worst/FR/SR ต่ออัลกอริทึม-ปัญหา
  results/summary_with_literature.csv <- summary_table.csv + แถวอ้างอิงจาก Kumar et al. 2020
  results/scalability_plot.png     <- กราฟเปรียบเทียบ scalability ข้าม 6 instance
  results/boxplot_RC<K>.png        <- boxplot เปรียบเทียบอัลกอริทึมต่อปัญหา (6 ไฟล์)
  results/statistical_tests.csv    <- ผล Friedman + Wilcoxon (แก้ NaN แล้ว)

วิธีใช้: รันไฟล์นี้ตรงๆ ใน Spyder ได้เลย รันซ้ำได้ทุกเมื่อ (checkpoint ข้ามของเดิม)

หมายเหตุเวลา: 7 algo x 6 prob x 30 runs = 1,260 runs ที่ MaxFEs=200,000/run
ใช้เวลานานหลายชั่วโมง แบ่งรันเป็นช่วงได้เพราะมี checkpoint
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sopwm_objective import sopwm_evaluate, get_bounds, get_dimension, BEST_KNOWN
from gwo_op_sw import grey_wolf_optimizer_final
from gwo_baseline import grey_wolf_optimizer
from pso_baseline import particle_swarm_optimizer
from woa_baseline import whale_optimizer
from de_baseline import differential_evolution
from ga_baseline import genetic_algorithm
from hho_baseline import harris_hawks_optimizer

# ---------------------------------------------------------------------------
# การตั้งค่า
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
RAW_CSV = os.path.join(RESULTS_DIR, "raw_results.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary_table.csv")
SUMMARY_LIT_CSV = os.path.join(RESULTS_DIR, "summary_with_literature.csv")
STATS_CSV = os.path.join(RESULTS_DIR, "statistical_tests.csv")

POP_SIZE = 50
MAX_ITER = 4000       # 50*4000 = 200,000 FEs ตรงมาตรฐาน CEC2020 (D=25-30)
N_RUNS = 30
SEED_BASE = 42
PROB_LIST = list(range(45, 51))
ALGORITHMS = ["GWO-OP-SW", "GWO", "PSO", "WOA", "DE", "GA", "HHO"]

# ผลจาก Kumar et al. (2020) Table 9 — สำหรับใส่ในตารางเปรียบเทียบสุดท้าย (ไม่ต้องรันซ้ำ)
LITERATURE_RESULTS = {
    # (algorithm, prob_k): dict(best, mean, worst, std, FR_pct)
    ("IUDE", 45): dict(best=5.52e-02, mean=7.66e-02, worst=1.17e-01, std=2.41e-02, FR_pct=100),
    ("eMAgES", 45): dict(best=3.80e-02, mean=5.01e-02, worst=6.42e-02, std=9.33e-03, FR_pct=100),
    ("eiLSHADE", 45): dict(best=7.71e-02, mean=1.08e-01, worst=1.64e-01, std=2.97e-02, FR_pct=100),
    ("IUDE", 46): dict(best=4.31e-02, mean=5.48e-02, worst=6.71e-02, std=9.07e-03, FR_pct=100),
    ("eMAgES", 46): dict(best=2.36e-02, mean=2.91e-02, worst=3.37e-02, std=4.13e-03, FR_pct=100),
    ("eiLSHADE", 46): dict(best=6.67e-02, mean=9.20e-02, worst=2.04e-01, std=4.45e-02, FR_pct=100),
    ("IUDE", 47): dict(best=3.44e-02, mean=6.44e-02, worst=2.05e-01, std=5.64e-02, FR_pct=92),
    ("eMAgES", 47): dict(best=1.51e-02, mean=1.98e-02, worst=2.46e-02, std=2.99e-03, FR_pct=100),
    ("eiLSHADE", 47): dict(best=2.71e-02, mean=4.57e-02, worst=6.98e-02, std=1.12e-02, FR_pct=100),
    ("IUDE", 48): dict(best=4.50e-02, mean=6.53e-02, worst=2.09e-01, std=5.38e-02, FR_pct=100),
    ("eMAgES", 48): dict(best=1.68e-02, mean=1.74e-02, worst=2.24e-02, std=1.87e-03, FR_pct=100),
    ("eiLSHADE", 48): dict(best=4.71e-02, mean=2.23e-01, worst=4.94e-01, std=1.62e-01, FR_pct=100),
    ("IUDE", 49): dict(best=2.85e-02, mean=5.50e-02, worst=9.58e-02, std=2.33e-02, FR_pct=100),
    ("eMAgES", 49): dict(best=9.83e-03, mean=3.06e-02, worst=5.96e-02, std=1.69e-02, FR_pct=100),
    ("eiLSHADE", 49): dict(best=6.78e-02, mean=1.61e-01, worst=3.20e-01, std=8.89e-02, FR_pct=100),
    ("IUDE", 50): dict(best=6.23e-02, mean=2.53e-01, worst=3.78e-01, std=1.13e-01, FR_pct=36),
    ("eMAgES", 50): dict(best=1.56e-02, mean=2.94e-02, worst=7.77e-02, std=2.34e-02, FR_pct=100),
    ("eiLSHADE", 50): dict(best=2.62e-01, mean=3.07e-01, worst=3.53e-01, std=3.17e-02, FR_pct=0),
}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Wrapper ให้ทุกอัลกอริทึมเรียกด้วย signature เดียวกัน
# ---------------------------------------------------------------------------
def run_one(algo_name, prob_k, seed):
    dim = get_dimension(prob_k)
    lb, ub = get_bounds(prob_k)

    def objective_fn(x, prob_k=prob_k):
        return sopwm_evaluate(x, prob_k)

    kwargs = dict(pop_size=POP_SIZE, max_iter=MAX_ITER, seed=seed, verbose=False)

    if algo_name == "GWO-OP-SW":
        res = grey_wolf_optimizer_final(objective_fn, 90.0, dim, **kwargs)
    elif algo_name == "GWO":
        res = grey_wolf_optimizer(objective_fn, lb, ub, dim, **kwargs)
    elif algo_name == "PSO":
        res = particle_swarm_optimizer(objective_fn, lb, ub, dim, **kwargs)
    elif algo_name == "WOA":
        res = whale_optimizer(objective_fn, lb, ub, dim, **kwargs)
    elif algo_name == "DE":
        res = differential_evolution(objective_fn, lb, ub, dim, **kwargs)
    elif algo_name == "GA":
        res = genetic_algorithm(objective_fn, lb, ub, dim, **kwargs)
    elif algo_name == "HHO":
        res = harris_hawks_optimizer(objective_fn, lb, ub, dim, **kwargs)
    else:
        raise ValueError(algo_name)

    return res["best_f"], res["best_cv"]


# ---------------------------------------------------------------------------
# ขั้น 1: รันทุก (algo, prob_k, run) พร้อม checkpoint/resume
# ---------------------------------------------------------------------------
def load_done_set():
    if not os.path.exists(RAW_CSV):
        return set()
    df = pd.read_csv(RAW_CSV)
    return set(zip(df["algorithm"], df["prob_k"], df["run"]))


def append_row(row):
    df_row = pd.DataFrame([row])
    header = not os.path.exists(RAW_CSV)
    df_row.to_csv(RAW_CSV, mode="a", header=header, index=False)


def run_all():
    done = load_done_set()
    total = len(ALGORITHMS) * len(PROB_LIST) * N_RUNS
    print(f"งานทั้งหมด: {total} runs | เสร็จแล้ว: {len(done)} runs | เหลือ: {total - len(done)} runs")

    count = 0
    for algo_name in ALGORITHMS:
        for prob_k in PROB_LIST:
            bk = BEST_KNOWN[prob_k]
            for run in range(N_RUNS):
                key = (algo_name, prob_k, run)
                count += 1
                if key in done:
                    continue

                seed = SEED_BASE + run
                t0 = time.time()
                best_f, best_cv = run_one(algo_name, prob_k, seed)
                elapsed = time.time() - t0

                feasible = bool(best_cv <= 1e-6)
                row = dict(algorithm=algo_name, prob_k=prob_k, run=run, seed=seed,
                           best_f=best_f if feasible else np.nan, best_cv=best_cv,
                           feasible=feasible, elapsed_sec=elapsed)
                append_row(row)
                done.add(key)

                ratio = (best_f / bk) if feasible else float("nan")
                print(f"[{count}/{total}] {algo_name:>10} RC{prob_k} run{run:02d}: "
                      f"f={best_f:.4e} ({ratio:.2f}x) feasible={feasible} ({elapsed:.1f}s)")

    print("\nรันครบทุก run แล้ว ✅")


# ---------------------------------------------------------------------------
# ขั้น 2: สรุปตาราง summary (best/mean/std/worst/FR/SR)
# ---------------------------------------------------------------------------
def build_summary():
    df = pd.read_csv(RAW_CSV)
    rows = []
    for algo_name in ALGORITHMS:
        for prob_k in PROB_LIST:
            sub = df[(df.algorithm == algo_name) & (df.prob_k == prob_k)]
            bk = BEST_KNOWN[prob_k]
            feasible_f = sub.loc[sub.feasible, "best_f"]
            fr = sub.feasible.mean() * 100
            sr = (sub.loc[sub.feasible, "best_f"] <= bk + 1e-8).mean() * 100 if len(feasible_f) else 0.0
            rows.append(dict(
                algorithm=algo_name, prob_k=prob_k, best_known=bk,
                best=feasible_f.min() if len(feasible_f) else np.nan,
                mean=feasible_f.mean() if len(feasible_f) else np.nan,
                std=feasible_f.std() if len(feasible_f) else np.nan,
                worst=feasible_f.max() if len(feasible_f) else np.nan,
                FR_pct=fr, SR_pct=sr, n_runs=len(sub),
            ))
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nบันทึกตารางสรุปที่ {SUMMARY_CSV}")

    # รวม literature baseline เข้าไปในตารางเดียวกันเพื่อเทียบครบทุกแหล่ง
    lit_rows = []
    for (algo, prob_k), vals in LITERATURE_RESULTS.items():
        lit_rows.append(dict(algorithm=algo, prob_k=prob_k, best_known=BEST_KNOWN[prob_k],
                              best=vals["best"], mean=vals["mean"], std=vals["std"],
                              worst=vals["worst"], FR_pct=vals["FR_pct"], SR_pct=np.nan,
                              n_runs="25 (Kumar et al. 2020)"))
    summary_with_lit = pd.concat([summary, pd.DataFrame(lit_rows)], ignore_index=True)
    summary_with_lit.to_csv(SUMMARY_LIT_CSV, index=False)
    print(f"บันทึกตารางรวม literature ที่ {SUMMARY_LIT_CSV}")
    return summary


# ---------------------------------------------------------------------------
# ขั้น 3: กราฟ
# ---------------------------------------------------------------------------
def make_plots(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    levels = {45: "3L", 46: "5L", 47: "7L", 48: "9L", 49: "11L", 50: "13L"}
    for algo_name in ALGORITHMS:
        sub = summary[summary.algorithm == algo_name].sort_values("prob_k")
        ratios = sub["mean"] / sub["best_known"]
        ax.plot([levels[p] for p in sub["prob_k"]], ratios, marker="o", label=algo_name)
    ax.set_xlabel("SOPWM instance (inverter level)")
    ax.set_ylabel("mean(f) / best-known (lower is better, 1.0 = matches optimum)")
    ax.set_title("Scalability comparison across SOPWM instances (RC45-RC50)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "scalability_plot.png"), dpi=150)
    plt.close(fig)

    df = pd.read_csv(RAW_CSV)
    for prob_k in PROB_LIST:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [df[(df.algorithm == a) & (df.prob_k == prob_k) & (df.feasible)]["best_f"].values
                for a in ALGORITHMS]
        ax.boxplot(data, tick_labels=ALGORITHMS, showmeans=True)
        ax.axhline(BEST_KNOWN[prob_k], color="red", linestyle="--", label="best-known")
        ax.set_title(f"RC{prob_k} - distribution across {N_RUNS} runs")
        ax.set_ylabel("f(x)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"boxplot_RC{prob_k}.png"), dpi=150)
        plt.close(fig)

    print(f"บันทึกกราฟทั้งหมดในโฟลเดอร์ {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
# ขั้น 4: Friedman + Wilcoxon test (แก้ปัญหา NaN แล้ว)
# ---------------------------------------------------------------------------
def run_statistical_tests():
    df = pd.read_csv(RAW_CSV)

    # matrix: แถว=ปัญหา, คอลัมน์=อัลกอริทึม, ค่า=mean(best_f) จาก feasible runs เท่านั้น
    mat = pd.DataFrame(index=PROB_LIST, columns=ALGORITHMS, dtype=float)
    for p in PROB_LIST:
        for a in ALGORITHMS:
            sub = df[(df.algorithm == a) & (df.prob_k == p) & (df.feasible)]
            mat.loc[p, a] = sub["best_f"].mean() if len(sub) else np.nan

    # แก้ NaN (ไม่เจอ feasible เลย = แพ้ที่สุดในปัญหานั้นโดยอัตโนมัติ)
    # แทนด้วยค่าที่แย่กว่าค่าจริงที่แย่ที่สุดในแถวนั้น 10 เท่า เพื่อให้ rank แย่สุดเสมอ
    mat_fixed = mat.copy()
    for p in PROB_LIST:
        row = mat.loc[p]
        if row.isna().any():
            worst_real = row.max(skipna=True)
            sentinel = worst_real * 10 if not np.isnan(worst_real) else 1e6
            mat_fixed.loc[p] = row.fillna(sentinel)

    rows = []

    stat, p = stats.friedmanchisquare(*[mat_fixed[a].values for a in ALGORITHMS])
    rows.append(dict(test="Friedman (across 6 problems)", comparison="all algorithms",
                      statistic=stat, p_value=p))

    for algo_name in ALGORITHMS:
        if algo_name == "GWO-OP-SW":
            continue
        try:
            w_stat, w_p = stats.wilcoxon(mat_fixed["GWO-OP-SW"], mat_fixed[algo_name])
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        rows.append(dict(test="Wilcoxon signed-rank", comparison=f"GWO-OP-SW vs {algo_name}",
                          statistic=w_stat, p_value=w_p))

    # average rank ต่ออัลกอริทึม (ใช้รายงานในเปเปอร์)
    ranks = mat_fixed.rank(axis=1)
    for algo_name in ALGORITHMS:
        rows.append(dict(test="Average rank", comparison=algo_name,
                          statistic=ranks[algo_name].mean(), p_value=np.nan))

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(STATS_CSV, index=False)
    print(f"\nบันทึกผล statistical test ที่ {STATS_CSV}")
    print(stats_df.to_string(index=False))
    return stats_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_all()
    summary = build_summary()
    print(summary.to_string(index=False))
    make_plots(summary)
    run_statistical_tests()
    print("\nเสร็จสมบูรณ์ ✅ ดูผลได้ที่โฟลเดอร์ results/")
