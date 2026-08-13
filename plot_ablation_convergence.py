"""
plot_ablation_convergence.py

สร้างกราฟ convergence curve (alpha_f ต่อ iteration) เปรียบเทียบ 5 a-decay
schedule จาก results/ablation/schedule_convergence_traces.json
(ต้องรัน run_ablation_schedule.py ให้เสร็จอย่างน้อย 1 run ต่อ schedule ก่อน)

สร้าง 2 อย่างต่อปัญหาที่เลือก (ค่าเริ่มต้น: RC45 กับ RC50):
  1. กราฟ convergence: alpha_f (log scale) vs iteration, เส้นละ schedule
  2. กราฟ a-value trace: ค่า a ที่แต่ละ schedule ใช้จริงต่อ iteration
     (ให้เห็นภาพว่า square_wave สลับ high/low ต่างจาก linear/cosine ยังไง)

ผลลัพธ์: results/ablation/convergence_RC<k>.png, results/ablation/adecay_trace_RC<k>.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("results", "ablation")
CONV_JSON = os.path.join(RESULTS_DIR, "schedule_convergence_traces.json")

SCHEDULES = ["linear", "cosine", "oscillating", "square_wave", "adaptive_switch"]
PROBS_TO_PLOT = [45, 50]   # เปลี่ยนเป็น range(45, 51) ถ้าอยากได้ครบทุกปัญหา


def load_traces():
    if not os.path.exists(CONV_JSON):
        raise FileNotFoundError(
            f"ไม่พบ {CONV_JSON} — ต้องรัน run_ablation_schedule.py ให้เสร็จอย่างน้อย "
            f"1 run ต่อ schedule ก่อน (สคริปต์เก็บ trace ของ run=0 ให้อัตโนมัติ)"
        )
    with open(CONV_JSON) as f:
        return json.load(f)


def plot_convergence(traces, prob_k):
    fig, ax = plt.subplots(figsize=(8, 5))
    for sched in SCHEDULES:
        key = f"{prob_k}|{sched}"
        if key not in traces:
            continue
        conv = traces[key]["convergence"]
        y = [v if v is not None else np.nan for v in conv]
        ax.plot(y, label=sched, linewidth=1.3)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("alpha_f (feasible best, log scale)")
    ax.set_title(f"RC{prob_k}: convergence comparison across a-decay schedules")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"convergence_RC{prob_k}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"บันทึก {out_path}")


def plot_a_trace(traces, prob_k):
    fig, ax = plt.subplots(figsize=(8, 4))
    for sched in SCHEDULES:
        key = f"{prob_k}|{sched}"
        if key not in traces:
            continue
        a_log = traces[key]["a_used_log"]
        ax.plot(a_log, label=sched, linewidth=1.0, alpha=0.85)
    ax.set_xlabel("iteration")
    ax.set_ylabel("a (exploration-exploitation coefficient)")
    ax.set_title(f"RC{prob_k}: a-decay schedule trace")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"adecay_trace_RC{prob_k}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"บันทึก {out_path}")


if __name__ == "__main__":
    traces = load_traces()
    for prob_k in PROBS_TO_PLOT:
        plot_convergence(traces, prob_k)
        plot_a_trace(traces, prob_k)
    print("\nเสร็จสมบูรณ์ ✅")
