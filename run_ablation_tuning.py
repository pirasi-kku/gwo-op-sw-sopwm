"""
run_ablation_tuning.py

รัน ablation เปรียบเทียบ duty cycle / a_low ของ square_wave 5 config
(baseline/more_periods/short_high/deeper_exploit/combo) จาก
gwo_squarewave_tuning.py แบบเก็บผลลง CSV จริง (ต้นฉบับ print อย่างเดียว)

เพิ่มจากต้นฉบับ: checkpoint/resume แบบเดียวกับ run_all_experiments.py

วิธีใช้: รันตรงๆ ใน Spyder จากโฟลเดอร์โปรเจกต์นี้ได้เลย

ผลลัพธ์:
  results/ablation/tuning_raw_results.csv
  results/ablation/tuning_summary.csv

เวลาโดยประมาณ: ~3-3.5 วินาที/run x 6 ปัญหา x 5 config x N_RUNS
  N_RUNS=10 (ค่าดั้งเดิมในสคริปต์ต้นฉบับ) -> ~15-18 นาที
  N_RUNS=30 (ค่าที่ใช้ตอนยืนยันผลจริงในเปเปอร์) -> ~45-55 นาที
"""

import os
import time
import numpy as np
import pandas as pd

from sopwm_objective import sopwm_evaluate, get_dimension, BEST_KNOWN
from gwo_op_sw_tuning import grey_wolf_optimizer_op

# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join("results", "ablation")
os.makedirs(RESULTS_DIR, exist_ok=True)
RAW_CSV = os.path.join(RESULTS_DIR, "tuning_raw_results.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "tuning_summary.csv")

POP_SIZE = 50
MAX_ITER = 4000
N_RUNS = 50          # ปรับได้: 10 (เร็ว) / 30 (ยืนยันผลเต็ม)
SEED_BASE = 42
UB_TOTAL = 90.0
PROB_LIST = list(range(45, 51))
CONFIGS = {
    "baseline":       dict(periods=10, duty_high=0.5, a_low=0.10),
    "more_periods":   dict(periods=20, duty_high=0.5, a_low=0.10),
    "short_high":     dict(periods=10, duty_high=0.3, a_low=0.10),
    "deeper_exploit": dict(periods=10, duty_high=0.5, a_low=0.01),
    "combo":          dict(periods=15, duty_high=0.3, a_low=0.01),
}


def load_done_set():
    if not os.path.exists(RAW_CSV):
        return set()
    df = pd.read_csv(RAW_CSV)
    return set(zip(df["prob_k"], df["config"], df["run"]))


def append_row(row):
    df_row = pd.DataFrame([row])
    header = not os.path.exists(RAW_CSV)
    df_row.to_csv(RAW_CSV, mode="a", header=header, index=False)


def run_all():
    done = load_done_set()
    total = len(PROB_LIST) * len(CONFIGS) * N_RUNS
    print(f"งานทั้งหมด: {total} runs | เสร็จแล้ว: {len(done)} runs | เหลือ: {total - len(done)} runs")

    count = 0
    t_start = time.time()
    for prob_k in PROB_LIST:
        dim = get_dimension(prob_k)

        def objective_fn(alpha, prob_k=prob_k):
            return sopwm_evaluate(alpha, prob_k)

        for cfg_name, cfg in CONFIGS.items():
            for run in range(N_RUNS):
                count += 1
                key = (prob_k, cfg_name, run)
                if key in done:
                    continue

                seed = SEED_BASE + run
                t0 = time.time()
                res = grey_wolf_optimizer_op(
                    objective_fn, UB_TOTAL, dim,
                    pop_size=POP_SIZE, max_iter=MAX_ITER,
                    seed=seed, verbose=False, **cfg,
                )
                elapsed = time.time() - t0

                feasible = bool(res["best_cv"] <= 1e-6)
                row = dict(prob_k=prob_k, config=cfg_name, run=run, seed=seed,
                           best_f=res["best_f"] if feasible else np.nan,
                           best_cv=res["best_cv"], feasible=feasible, elapsed_sec=elapsed)
                append_row(row)
                done.add(key)

                elapsed_total = time.time() - t_start
                print(f"[{count}/{total}] RC{prob_k} {cfg_name:>15} run{run:02d}: "
                      f"f={res['best_f']:.4e} feasible={feasible} "
                      f"({elapsed:.1f}s, รวม {elapsed_total/60:.1f} นาที)")

    print("\nรันครบทุก run แล้ว ✅")


def build_summary():
    df = pd.read_csv(RAW_CSV)
    rows = []
    for prob_k in PROB_LIST:
        bk = BEST_KNOWN[prob_k]
        for cfg_name in CONFIGS:
            sub = df[(df.prob_k == prob_k) & (df.config == cfg_name)]
            feas = sub[sub.feasible]
            rows.append(dict(
                prob_k=prob_k, config=cfg_name, best_known=bk,
                best=feas["best_f"].min() if len(feas) else np.nan,
                mean=feas["best_f"].mean() if len(feas) else np.nan,
                std=feas["best_f"].std() if len(feas) else np.nan,
                worst=feas["best_f"].max() if len(feas) else np.nan,
                FR_pct=sub.feasible.mean() * 100 if len(sub) else 0.0,
                n_runs=len(sub),
            ))
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nบันทึกตารางสรุปที่ {SUMMARY_CSV}")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run_all()
    build_summary()
    print("\nเสร็จสมบูรณ์ ✅ ดูผลได้ที่โฟลเดอร์ results/ablation/")
