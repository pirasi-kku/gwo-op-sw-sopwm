"""
run_colshade_experiments.py

รัน COLSHADE 30 runs x 6 instance (RC45-RC50) ด้วย protocol เดียวกับ
run_all_experiments.py เดิมทุกประการ (MaxFEs=200,000, seed=42+run)
แล้วเขียนต่อท้าย results/raw_results.csv ไฟล์เดียวกับ 7 อัลกอริทึมเดิม
(algorithm="COLSHADE") — ทำให้ build_summary()/make_plots()/
run_statistical_tests() เดิมใน run_all_experiments.py ใช้ต่อได้ทันที
แค่เพิ่ม "COLSHADE" เข้า ALGORITHMS list ก่อนเรียกฟังก์ชันเหล่านั้น

นอกจากนี้ยังเก็บ best_alpha + convergence trace แยกไว้ต่างหาก (ที่
raw_results.csv เดิมไม่มีช่องเก็บ) สำหรับใช้ทำ engineering validation
(waveform/THD) และ convergence curve เช่นเดียวกับที่ทำให้ GWO-OP-SW ใน
rerun_gwo_opsw_for_revision.py

รันซ้ำได้ปลอดภัย ข้าม (prob_k, run) ที่เสร็จแล้วอัตโนมัติ
"""

import os
import json
import time
import numpy as np
import pandas as pd

from sopwm_objective import sopwm_evaluate, get_bounds, get_dimension, BEST_KNOWN
from colshade_baseline import colshade_op_sw

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "..", "results")  # FIX (revision1): ชี้กลับไป root/results ตัวจริง (เดิมเป็น "results" เฉยๆ ซึ่งถ้ารันจากโฟลเดอร์ revision1 จะไปสร้างไฟล์ใหม่แยกต่างหาก ไม่ต่อท้ายผลจริงของ 7 อัลกอริทึมเดิม)
RAW_CSV = os.path.join(RESULTS_DIR, "raw_results.csv")

RAW_DIR_REVISION = os.path.join(_SCRIPT_DIR, "results_revision", "raw")  # FIX (revision1): anchor กับตำแหน่งไฟล์สคริปต์
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RAW_DIR_REVISION, exist_ok=True)

MAX_FES = 200_000
N_RUNS = 30
SEED_BASE = 42
PROB_LIST = list(range(45, 51))
ALGO_NAME = "COLSHADE"


def convergence_npy_path(prob_k):
    return os.path.join(RAW_DIR_REVISION, f"colshade_convergence_RC{prob_k}.npy")


def load_or_init_convergence(prob_k):
    path = convergence_npy_path(prob_k)
    if os.path.exists(path):
        arr = np.load(path)
        if arr.shape[0] == N_RUNS:
            return arr
    return np.full((N_RUNS, 200), np.nan)  # 200 checkpoint points ตามที่ colshade_op_sw ใช้


def load_done_set():
    if not os.path.exists(RAW_CSV):
        return set()
    df = pd.read_csv(RAW_CSV)
    sub = df[df.algorithm == ALGO_NAME]
    return set(zip(sub["prob_k"], sub["run"]))


def append_row(row):
    df_row = pd.DataFrame([row])
    header = not os.path.exists(RAW_CSV)
    df_row.to_csv(RAW_CSV, mode="a", header=header, index=False)


def run_all():
    done = load_done_set()
    total = len(PROB_LIST) * N_RUNS
    count = 0

    for prob_k in PROB_LIST:
        dim = get_dimension(prob_k)
        lb, ub = get_bounds(prob_k)
        bk = BEST_KNOWN[prob_k]

        def objective_fn(x, prob_k=prob_k):
            return sopwm_evaluate(x, prob_k)

        conv_arr = load_or_init_convergence(prob_k)
        alpha_path = os.path.join(RAW_DIR_REVISION, f"colshade_alpha_RC{prob_k}.json")
        alpha_store = {}
        if os.path.exists(alpha_path):
            with open(alpha_path) as f:
                alpha_store = json.load(f)

        for run in range(N_RUNS):
            count += 1
            key = (prob_k, run)
            if key in done:
                continue

            seed = SEED_BASE + run
            t0 = time.time()
            res = colshade_op_sw(objective_fn, lb, ub, dim, max_fes=MAX_FES,
                                  seed=seed, verbose=False)
            elapsed = time.time() - t0

            row = dict(algorithm=ALGO_NAME, prob_k=prob_k, run=run, seed=seed,
                       best_f=res["best_f"], best_cv=res["best_cv"],
                       feasible=res["feasible"], elapsed_sec=elapsed)
            append_row(row)
            done.add(key)

            # เก็บ convergence + best_alpha แยก (raw_results.csv หลักไม่มีช่องนี้)
            hist_f = [h[1] for h in res["history"]]  # best_f ต่อ checkpoint
            n_pts = min(len(hist_f), conv_arr.shape[1])
            conv_arr[run, :n_pts] = hist_f[:n_pts]
            np.save(convergence_npy_path(prob_k), conv_arr)

            alpha_store[str(run)] = res["best_x"].tolist() if res["best_x"] is not None else None
            with open(alpha_path, "w") as f:
                json.dump(alpha_store, f)

            ratio = (res["best_f"] / bk) if res["feasible"] else float("nan")
            print(f"[{count}/{total}] {ALGO_NAME} RC{prob_k} run{run:02d}: "
                  f"f={res['best_f']:.4e} ({ratio:.2f}x) feasible={res['feasible']} "
                  f"({elapsed:.1f}s)")

    print(f"\nรัน {ALGO_NAME} ครบทุก run แล้ว ✅")
    print(f"เพิ่มเข้า {RAW_CSV} แล้ว (algorithm='{ALGO_NAME}') — เพิ่ม 'COLSHADE' เข้า")
    print("ALGORITHMS list ใน run_all_experiments.py แล้วเรียก build_summary()/")
    print("make_plots()/run_statistical_tests() ต่อได้ทันที")
    print(f"best_alpha + convergence เก็บแยกไว้ที่ {RAW_DIR_REVISION}/")


if __name__ == "__main__":
    run_all()
