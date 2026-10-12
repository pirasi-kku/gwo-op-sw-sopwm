"""
rerun_gwo_opsw_for_revision.py

รัน GWO-OP-SW ใหม่เฉพาะตัวเดียว (ไม่แตะ 6 baseline เดิม) 30 runs x 6 instance
(RC45-RC50) ด้วย gwo_final.py ที่ EPS_MARGIN = 1e-5 (= delta_min ในเปเปอร์ Sec. 2.3)
ผลที่ได้ต้องตรงกับ results/raw_results.csv เดิมแบบ bit-exact (seed เดียวกัน)
จุดประสงค์คือเก็บ best_alpha + convergence trace ที่ pipeline เดิมไม่ได้เก็บไว้

เหตุผลที่ต้องรันใหม่ (ไม่ใช่แค่วิเคราะห์ raw_results.csv เดิม):
  raw_results.csv จาก run_all_experiments.py เดิม เก็บแค่
  algorithm, prob_k, run, seed, best_f, best_cv, feasible, elapsed_sec
  --> ไม่มี best_alpha (คำตอบเต็ม) และไม่มี convergence trace ทั้งที่
  gwo_final.py คำนวณทั้งสองอย่างอยู่แล้วภายใน (res["best_alpha"],
  res["convergence"]) แค่ run_one() เดิมไม่ได้ดึงออกมาเก็บ

ข้อมูลที่ได้จากไฟล์นี้ใช้ต่อกับ:
  - engineering_analysis.py     (waveform/THD/FFT ต้องใช้ best_alpha)
  - verify_rc50.py              (ต้องใช้ best_alpha ของ RC50)
  - convergence curve ทั้ง 6 instance (Reviewer A, C)

ผลลัพธ์:
  results_revision/raw/gwo_op_sw_RC{k}.csv        <- summary ต่อ run (checkpoint/resume)
  results_revision/raw/gwo_op_sw_convergence_RC{k}.npy   <- shape (30, 4000) convergence trace

รันซ้ำได้ปลอดภัย ข้าม run ที่เสร็จแล้วอัตโนมัติ (เช็คจาก summary CSV)
"""

import os
import json
import time
import numpy as np
import pandas as pd

from sopwm_objective import sopwm_evaluate, get_bounds, get_dimension, BEST_KNOWN
from gwo_final import grey_wolf_optimizer_final

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results_revision_1e-5")  # FIX (revision1): anchor กับตำแหน่งไฟล์สคริปต์ ไม่พึ่ง cwd
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

POP_SIZE = 50
MAX_ITER = 4000
N_RUNS = 30
SEED_BASE = 42
UB_TOTAL = 90.0
PROB_LIST = list(range(45, 51))


def summary_csv_path(prob_k):
    return os.path.join(RAW_DIR, f"gwo_op_sw_RC{prob_k}.csv")


def convergence_npy_path(prob_k):
    return os.path.join(RAW_DIR, f"gwo_op_sw_convergence_RC{prob_k}.npy")


def load_completed_runs(prob_k):
    path = summary_csv_path(prob_k)
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df["run"].tolist())


def load_or_init_convergence(prob_k):
    path = convergence_npy_path(prob_k)
    if os.path.exists(path):
        arr = np.load(path)
        if arr.shape == (N_RUNS, MAX_ITER):
            return arr
    return np.full((N_RUNS, MAX_ITER), np.nan)


def append_summary_row(prob_k, row: dict):
    path = summary_csv_path(prob_k)
    df_row = pd.DataFrame([row])
    header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=header, index=False)


def run_all():
    total = len(PROB_LIST) * N_RUNS
    count = 0

    for prob_k in PROB_LIST:
        dim = get_dimension(prob_k)
        bk = BEST_KNOWN[prob_k]

        def objective_fn(alpha, prob_k=prob_k):
            return sopwm_evaluate(alpha, prob_k)

        completed = load_completed_runs(prob_k)
        conv_arr = load_or_init_convergence(prob_k)

        for run in range(N_RUNS):
            count += 1
            if run in completed:
                continue

            seed = SEED_BASE + run
            t0 = time.time()
            res = grey_wolf_optimizer_final(
                objective_fn, UB_TOTAL, dim,
                pop_size=POP_SIZE, max_iter=MAX_ITER,
                seed=seed, verbose=False,
            )
            elapsed = time.time() - t0

            feasible = bool(res["best_cv"] <= 1e-6)
            ratio = (res["best_f"] / bk) if feasible else float("nan")

            row = dict(
                run=run, seed=seed,
                best_f=res["best_f"] if feasible else np.nan,
                best_cv=res["best_cv"],
                feasible=feasible,
                elapsed_sec=elapsed,
                best_alpha_json=json.dumps(res["best_alpha"].tolist()),
            )
            append_summary_row(prob_k, row)

            conv = np.asarray(res["convergence"], dtype=float)
            conv_arr[run, :len(conv)] = conv
            np.save(convergence_npy_path(prob_k), conv_arr)

            print(f"[{count}/{total}] RC{prob_k} run{run:02d}: "
                  f"f={res['best_f']:.4e} ({ratio:.2f}x) feasible={feasible} "
                  f"({elapsed:.1f}s)")

    print("\nรัน GWO-OP-SW ใหม่ครบทุก run แล้ว (EPS_MARGIN=1e-5 ตรงกับเปเปอร์) ✅")
    print(f"ผลลัพธ์อยู่ที่ {RAW_DIR}/")


if __name__ == "__main__":
    run_all()
