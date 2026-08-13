"""
gwo_modified_v2.py

GWO-OP + Nonlinear a-decay (ablation เทียบ 5 schedule)
=========================================================
ต่อยอดจาก gwo_modified.py (มีกลไก order-preserving reparameterization อยู่แล้ว)
เพิ่มตัวเลือก a-decay schedule 5 แบบ เพื่อเทียบกันตรงๆ ว่าแบบไหนแก้ premature
convergence ได้ดีกว่า (ปัญหาที่เจอใน gwo_modified.py: alpha ค้างนิ่งหลายร้อย iteration)

Schedule ที่มีให้เลือก (a_schedule):
  'linear'          : a(t) = 2 - 2*(t/T)                              (มาตรฐานเดิม, control)
  'cosine'          : a(t) = 2*cos((pi/2)*(t/T))                      (ลดช้าตอนต้น เร่งช่วงท้าย)
  'oscillating'      : a(t) = linear(t) + A*(1-t/T)*sin(2*pi*t/period) (burst แบบต่อเนื่อง
                                                                        เป็นคาบคงที่)
  'square_wave'      : a(t) สลับ a_high/a_low แบบ step function ตามคาบเวลาคงที่
                        (ไม่สนใจสถานะการค้นหาจริง เอาไว้เป็น control เทียบกับ adaptive_switch)
  'adaptive_switch'  : a ลดตาม linear schedule ตามปกติ แต่ถ้าตรวจพบ stagnation
                        (alpha_f ไม่ขยับเกิน stagnation_threshold รอบ) จะบังคับ a = a_high
                        ค้างไว้ boost_duration รอบเพื่อฉีด exploration กลับเข้าไป แล้วค่อย
                        กลับไปลดตาม schedule ปกติ — ตอบสนองต่อสถานะจริงของการค้นหา
                        ต่างจาก square_wave ที่สลับตามเวลาอย่างเดียว

ทุกอย่างอื่นเหมือน gwo_modified.py ทุกประการ (Deb's feasibility rule, order-preserving
decode, boundary handling) ต่างแค่ตรงฟังก์ชันคำนวณ a เท่านั้น เพื่อให้ผลต่างที่วัดได้
มาจาก a-schedule เพียงอย่างเดียว
"""

import numpy as np
import time
from sopwm_objective import sopwm_evaluate, get_bounds, get_dimension, \
    mean_constraint_violation, BEST_KNOWN

EPS_MARGIN = 1e-5


def is_better(f_a, cv_a, f_b, cv_b, eps_feasible=1e-6):
    a_feasible = cv_a <= eps_feasible
    b_feasible = cv_b <= eps_feasible
    if a_feasible and not b_feasible:
        return True
    if not a_feasible and b_feasible:
        return False
    if a_feasible and b_feasible:
        return f_a < f_b
    return cv_a < cv_b


def decode_delta_to_alpha(delta: np.ndarray, ub_total: float) -> np.ndarray:
    delta = np.clip(delta, EPS_MARGIN, None)
    raw_alpha = np.cumsum(delta, axis=1)
    last = raw_alpha[:, -1]
    scale = np.where(last > ub_total, ub_total / last, 1.0)
    return raw_alpha * scale[:, np.newaxis]


# ---------------------------------------------------------------------------
# a-decay schedules (stateless แบบ: linear, cosine, oscillating, square_wave)
# ---------------------------------------------------------------------------
def compute_a(t: int, max_iter: int, schedule: str,
              a_high: float = 2.0, a_low: float = 0.1,
              osc_amplitude: float = 0.5, osc_periods: int = 10,
              square_periods: int = 10) -> float:
    frac = t / max_iter
    if schedule == "linear":
        a = 2.0 - 2.0 * frac
    elif schedule == "cosine":
        a = 2.0 * np.cos((np.pi / 2.0) * frac)
    elif schedule == "oscillating":
        base = 2.0 - 2.0 * frac
        period = max_iter / osc_periods
        osc = osc_amplitude * (1.0 - frac) * np.sin(2 * np.pi * t / period)
        a = base + osc
    elif schedule == "square_wave":
        period = max_iter / square_periods
        a = a_high if (t % period) < (period / 2.0) else a_low
    else:
        raise ValueError(f"schedule '{schedule}' ต้องคำนวณแบบ stateful (adaptive_switch) "
                          f"ไม่ใช้ compute_a() โดยตรง")
    return float(np.clip(a, 0.0, 2.0))


# ---------------------------------------------------------------------------
# GWO-OP core พร้อม a_schedule ที่เลือกได้ (5 แบบ)
# ---------------------------------------------------------------------------
def grey_wolf_optimizer_op(objective_fn, ub_total, dim, pop_size=30, max_iter=500,
                            a_schedule="linear", constraint_epsilon=1e-4,
                            a_high=2.0, a_low=0.1,
                            stagnation_threshold=30, boost_duration=50,
                            seed=None, verbose=True):
    rng = np.random.default_rng(seed)
    delta_lb = np.zeros(dim)
    delta_ub = np.full(dim, ub_total)

    positions = delta_lb + rng.random((pop_size, dim)) * (delta_ub - delta_lb)

    def evaluate_population(delta_pop):
        alpha_pop = decode_delta_to_alpha(delta_pop, ub_total)
        f, g, h = objective_fn(alpha_pop)
        if g is None:
            g = np.zeros((alpha_pop.shape[0], 0))
        if h is None:
            h = np.zeros((alpha_pop.shape[0], 0))
        cv = mean_constraint_violation(g, h, epsilon=constraint_epsilon)
        return f, cv, g

    f_vals, cv_vals, g_vals = evaluate_population(positions)

    def rank_indices(f_vals, cv_vals):
        idx = list(range(len(f_vals)))
        idx.sort(key=lambda i: (cv_vals[i] > 1e-6, cv_vals[i] if cv_vals[i] > 1e-6 else f_vals[i]))
        return idx

    order = rank_indices(f_vals, cv_vals)
    alpha_idx, beta_idx, delta_idx = order[0], order[1], order[2]
    alpha_pos, alpha_f, alpha_cv = positions[alpha_idx].copy(), f_vals[alpha_idx], cv_vals[alpha_idx]
    beta_pos, beta_f, beta_cv = positions[beta_idx].copy(), f_vals[beta_idx], cv_vals[beta_idx]
    delta_pos, delta_f, delta_cv = positions[delta_idx].copy(), f_vals[delta_idx], cv_vals[delta_idx]

    convergence = []
    stagnation_counter = 0     # จำนวนรอบติดกันที่ alpha_f ไม่ขยับ (ใช้กับ adaptive_switch)
    boost_countdown = 0        # จำนวนรอบที่เหลือของการบังคับ a=a_high (เฉพาะ adaptive_switch)
    last_alpha_f = alpha_f
    a_used_log = []

    for t in range(max_iter):
        # ---- คำนวณ a ของรอบนี้ ----
        if a_schedule == "adaptive_switch":
            if boost_countdown > 0:
                a = a_high
                boost_countdown -= 1
            else:
                a = compute_a(t, max_iter, "linear")
                if stagnation_counter >= stagnation_threshold:
                    boost_countdown = boost_duration
                    stagnation_counter = 0
                    a = a_high
        else:
            a = compute_a(t, max_iter, a_schedule, a_high=a_high, a_low=a_low)
        a_used_log.append(a)

        r1 = rng.random((pop_size, dim)); r2 = rng.random((pop_size, dim))
        A1 = 2 * a * r1 - a; C1 = 2 * r2
        X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - positions)

        r1 = rng.random((pop_size, dim)); r2 = rng.random((pop_size, dim))
        A2 = 2 * a * r1 - a; C2 = 2 * r2
        X2 = beta_pos - A2 * np.abs(C2 * beta_pos - positions)

        r1 = rng.random((pop_size, dim)); r2 = rng.random((pop_size, dim))
        A3 = 2 * a * r1 - a; C3 = 2 * r2
        X3 = delta_pos - A3 * np.abs(C3 * delta_pos - positions)

        positions = (X1 + X2 + X3) / 3.0
        positions = np.clip(positions, delta_lb, delta_ub)

        f_vals, cv_vals, g_vals = evaluate_population(positions)

        all_pos = np.vstack([positions, alpha_pos[None, :], beta_pos[None, :], delta_pos[None, :]])
        all_f = np.concatenate([f_vals, [alpha_f, beta_f, delta_f]])
        all_cv = np.concatenate([cv_vals, [alpha_cv, beta_cv, delta_cv]])

        order = rank_indices(all_f, all_cv)
        alpha_idx, beta_idx, delta_idx = order[0], order[1], order[2]
        alpha_pos, alpha_f, alpha_cv = all_pos[alpha_idx].copy(), all_f[alpha_idx], all_cv[alpha_idx]
        beta_pos, beta_f, beta_cv = all_pos[beta_idx].copy(), all_f[beta_idx], all_cv[beta_idx]
        delta_pos, delta_f, delta_cv = all_pos[delta_idx].copy(), all_f[delta_idx], all_cv[delta_idx]

        if alpha_cv <= 1e-6 and abs(alpha_f - last_alpha_f) < 1e-12:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        last_alpha_f = alpha_f

        convergence.append(alpha_f if alpha_cv <= 1e-6 else np.nan)

        if verbose and (t % 200 == 0 or t == max_iter - 1):
            feas_str = "feasible" if alpha_cv <= 1e-6 else f"infeasible (CV={alpha_cv:.3e})"
            print(f"  iter {t:4d}/{max_iter} [{a_schedule:>15}]: a={a:.3f}  "
                  f"alpha_f={alpha_f:.6e} [{feas_str}]  stagnant_for={stagnation_counter}  "
                  f"boost_left={boost_countdown}")

    best_alpha = decode_delta_to_alpha(alpha_pos[None, :], ub_total)[0]
    max_stagnation = max(convergence_stagnation_lengths(convergence))
    return dict(best_delta=alpha_pos, best_alpha=best_alpha, best_f=alpha_f,
                best_cv=alpha_cv, convergence=convergence, max_stagnation=max_stagnation,
                a_used_log=a_used_log)


def convergence_stagnation_lengths(convergence):
    """คำนวณความยาวของช่วงที่ค่าไม่เปลี่ยน (นิ่ง) ที่ยาวที่สุด สำหรับวัดระดับ premature convergence"""
    lengths = []
    run = 0
    prev = None
    for v in convergence:
        if prev is not None and not np.isnan(v) and not np.isnan(prev) and abs(v - prev) < 1e-12:
            run += 1
        else:
            if run > 0:
                lengths.append(run)
            run = 0
        prev = v
    lengths.append(run)
    return lengths if lengths else [0]


# ---------------------------------------------------------------------------
# Main: ablation เทียบ 5 a-schedule บน SOPWM ทั้ง 6 instance
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    POP_SIZE = 50
    MAX_ITER = 4000     # 50*4000 = 200,000 FEs ตรงมาตรฐาน CEC2020
    N_RUNS = 50          # ลดเป็นเลขน้อยกว่านี้ (เช่น 5) ก่อนถ้าอยากเทสเร็วๆ ก่อนรันเต็ม
    SEED_BASE = 42
    UB_TOTAL = 90.0
    SCHEDULES = ["linear", "cosine", "oscillating", "square_wave", "adaptive_switch"]

    all_results = {}

    for prob_k in range(45, 51):
        dim = get_dimension(prob_k)

        def objective_fn(alpha, prob_k=prob_k):
            return sopwm_evaluate(alpha, prob_k)

        print(f"\n{'=' * 70}\nRC{prob_k} (D={dim}), best-known f(x*) = {BEST_KNOWN[prob_k]:.6e}\n{'=' * 70}")

        for sched in SCHEDULES:
            run_bests = []
            max_stag_list = []
            t0 = time.time()
            for run in range(N_RUNS):
                res = grey_wolf_optimizer_op(
                    objective_fn, UB_TOTAL, dim,
                    pop_size=POP_SIZE, max_iter=MAX_ITER, a_schedule=sched,
                    seed=SEED_BASE + run, verbose=(run == 0),
                )
                run_bests.append(res["best_f"] if res["best_cv"] <= 1e-6 else np.nan)
                max_stag_list.append(res["max_stagnation"])
            elapsed = time.time() - t0

            run_bests = np.array(run_bests, dtype=float)
            feasible_runs = ~np.isnan(run_bests)
            print(f"[{sched:>15}] {feasible_runs.sum()}/{N_RUNS} feasible, "
                  f"best={np.nanmin(run_bests):.6e}, mean={np.nanmean(run_bests):.6e}, "
                  f"worst={np.nanmax(run_bests):.6e}, "
                  f"mean_max_stagnation={np.mean(max_stag_list):.0f} iters, "
                  f"time={elapsed:.1f}s")

            all_results[(prob_k, sched)] = dict(
                best=float(np.nanmin(run_bests)), mean=float(np.nanmean(run_bests)),
                worst=float(np.nanmax(run_bests)), mean_max_stagnation=float(np.mean(max_stag_list)),
            )

    print(f"\n{'=' * 70}\nสรุปทั้งหมด (เทียบ 5 schedule)\n{'=' * 70}")
    for prob_k in range(45, 51):
        print(f"\nRC{prob_k} (best-known={BEST_KNOWN[prob_k]:.6e}):")
        for sched in SCHEDULES:
            r = all_results[(prob_k, sched)]
            print(f"  {sched:>15}: mean={r['mean']:.6e}  best={r['best']:.6e}  "
                  f"mean_max_stagnation={r['mean_max_stagnation']:.0f} iters")
