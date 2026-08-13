"""
run_ablation_schedule.py

รัน ablation เปรียบเทียบ a-decay 5 schedule (linear/cosine/oscillating/
square_wave/adaptive_switch) จาก gwo_modified_v2.py แบบเก็บผลลง CSV จริง
(ต้นฉบับ gwo_modified_v2.py print อย่างเดียว ไม่ save)

เพิ่มจากต้นฉบับ:
  - checkpoint/resume แบบเดียวกับ run_all_experiments.py (รันซ้ำได้ ข้ามของเดิม)
  - เก็บ convergence trace + a_used_log ของ run แรก (run=0) ของทุก (prob_k, schedule)
    ไว้ใน JSON สำหรับ plot_ablation_convergence.py

วิธีใช้: รันตรงๆ ใน Spyder จากโฟลเดอร์โปรเจกต์นี้ได้เลย (working directory ต้องเป็น
โฟลเดอร์นี้ เพราะ import sopwm_objective / gwo_modified_v2 แบบ relative)

ผลลัพธ์:
  results/ablation/schedule_raw_results.csv
  results/ablation/schedule_summary.csv
  results/ablation/schedule_convergence_traces.json

เวลาโดยประมาณ: ~3-4 วินาที/run x 6 ปัญหา x 5 schedule x N_RUNS
  N_RUNS=10  -> ~15-20 นาที
  N_RUNS=30  -> ~45-60 นาที
  N_RUNS=50 (ค่าดั้งเดิมใน gwo_modified_v2.py) -> ~75-100 นาที
ปรับ N_RUNS ด้านล่างได้ตามเวลาที่มี ค่อยรันเพิ่มทีหลังได้ (checkpoint ข้ามของเดิม)
"""

import os
import json
import time
import numpy as np
import pandas as pd

from sopwm_objective import sopwm_evaluate, get_dimension, BEST_KNOWN
from gwo_op_schedules import grey_wolf_optimizer_op

# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join("results", "ablation")
os.makedirs(RESULTS_DIR, exist_ok=True)
RAW_CSV = os.path.join(RESULTS_DIR, "schedule_raw_results.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "schedule_summary.csv")
CONV_JSON = os.path.join(RESULTS_DIR, "schedule_convergence_traces.json")

POP_SIZE = 50
MAX_ITER = 4000
N_RUNS = 50          # ปรับได้: 10 (เร็ว, ~15-20 นาที) / 30 / 50 (ค่าดั้งเดิม)
SEED_BASE = 42
UB_TOTAL = 90.0
PROB_LIST = list(range(45, 51))
SCHEDULES = ["linear", "cosine", "oscillating", "square_wave", "adaptive_switch"]


def load_done_set():
    if not os.path.exists(RAW_CSV):
        return set()
    df = pd.read_csv(RAW_CSV)
    return set(zip(df["prob_k"], df["schedule"], df["run"]))


def append_row(row):
    df_row = pd.DataFrame([row])
    header = not os.path.exists(RAW_CSV)
    df_row.to_csv(RAW_CSV, mode="a", header=header, index=False)


def load_convergence_store():
    if os.path.exists(CONV_JSON):
        with open(CONV_JSON) as f:
            return json.load(f)
    return {}


def run_all():
    done = load_done_set()
    convergence_store = load_convergence_store()

    total = len(PROB_LIST) * len(SCHEDULES) * N_RUNS
    print(f"งานทั้งหมด: {total} runs | เสร็จแล้ว: {len(done)} runs | เหลือ: {total - len(done)} runs")

    count = 0
    t_start = time.time()
    for prob_k in PROB_LIST:
        dim = get_dimension(prob_k)

        def objective_fn(alpha, prob_k=prob_k):
            return sopwm_evaluate(alpha, prob_k)

        for sched in SCHEDULES:
            for run in range(N_RUNS):
                count += 1
                key = (prob_k, sched, run)
                if key in done:
                    continue

                seed = SEED_BASE + run
                t0 = time.time()
                res = grey_wolf_optimizer_op(
                    objective_fn, UB_TOTAL, dim,
                    pop_size=POP_SIZE, max_iter=MAX_ITER, a_schedule=sched,
                    seed=seed, verbose=False,
                )
                elapsed = time.time() - t0

                feasible = bool(res["best_cv"] <= 1e-6)
                row = dict(prob_k=prob_k, schedule=sched, run=run, seed=seed,
                           best_f=res["best_f"] if feasible else np.nan,
                           best_cv=res["best_cv"], feasible=feasible,
                           max_stagnation=res["max_stagnation"], elapsed_sec=elapsed)
                append_row(row)
                done.add(key)

                # เก็บ convergence trace ของ run=0 ไว้ใช้ plot (ไม่เก็บทุก run กัน JSON ใหญ่เกิน)
                if run == 0:
                    ck = f"{prob_k}|{sched}"
                    convergence_store[ck] = dict(
                        convergence=[None if (isinstance(v, float) and np.isnan(v)) else v
                                     for v in res["convergence"]],
                        a_used_log=res["a_used_log"],
                    )
                    with open(CONV_JSON, "w") as f:
                        json.dump(convergence_store, f)

                elapsed_total = time.time() - t_start
                print(f"[{count}/{total}] RC{prob_k} {sched:>15} run{run:02d}: "
                      f"f={res['best_f']:.4e} feasible={feasible} "
                      f"({elapsed:.1f}s, รวม {elapsed_total/60:.1f} นาที)")

    print("\nรันครบทุก run แล้ว ✅")


def build_summary():
    df = pd.read_csv(RAW_CSV)
    rows = []
    for prob_k in PROB_LIST:
        bk = BEST_KNOWN[prob_k]
        for sched in SCHEDULES:
            sub = df[(df.prob_k == prob_k) & (df.schedule == sched)]
            feas = sub[sub.feasible]
            rows.append(dict(
                prob_k=prob_k, schedule=sched, best_known=bk,
                best=feas["best_f"].min() if len(feas) else np.nan,
                mean=feas["best_f"].mean() if len(feas) else np.nan,
                std=feas["best_f"].std() if len(feas) else np.nan,
                worst=feas["best_f"].max() if len(feas) else np.nan,
                FR_pct=sub.feasible.mean() * 100 if len(sub) else 0.0,
                mean_max_stagnation=sub.max_stagnation.mean() if len(sub) else np.nan,
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
    print("ขั้นต่อไป: รัน plot_ablation_convergence.py เพื่อสร้างกราฟ convergence")
