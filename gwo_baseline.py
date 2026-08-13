"""
gwo_baseline.py

Grey Wolf Optimizer (GWO) มาตรฐาน (Mirjalili et al., 2014) สำหรับแก้ปัญหา constrained
optimization โดยใช้ Deb's feasibility rules ในการจัดการ constraint (ไม่ใช้ penalty function)

ใช้คู่กับ sopwm_objective.py (ต้องอยู่ไดเรกทอรีเดียวกัน) เพื่อทดสอบกับปัญหา SOPWM (RC45-RC50)
แต่ตัว GWO เขียนให้ generic รับ objective function ใดๆ ที่มี signature แบบเดียวกันได้

รัน:
    python gwo_baseline.py
จะรัน GWO กับทั้ง 6 instance ของ SOPWM (RC45-RC50) และพิมพ์ผลเทียบกับ best-known
"""

import numpy as np
import time
from sopwm_objective import sopwm_evaluate, get_bounds, get_dimension, \
    mean_constraint_violation, BEST_KNOWN


# ---------------------------------------------------------------------------
# Deb's feasibility rules สำหรับเปรียบเทียบคำตอบ 2 ตัว (a ดีกว่า b หรือไม่)
# ---------------------------------------------------------------------------
def is_better(f_a, cv_a, f_b, cv_b, eps_feasible=1e-6):
    """
    คืนค่า True ถ้าคำตอบ a ดีกว่าคำตอบ b ตาม Deb's feasibility rules:
      1) ถ้า a feasible และ b infeasible -> a ดีกว่า
      2) ถ้าทั้งคู่ feasible -> เทียบค่า objective ตรงๆ
      3) ถ้าทั้งคู่ infeasible -> เทียบค่า constraint violation (น้อยกว่าดีกว่า)
    """
    a_feasible = cv_a <= eps_feasible
    b_feasible = cv_b <= eps_feasible

    if a_feasible and not b_feasible:
        return True
    if not a_feasible and b_feasible:
        return False
    if a_feasible and b_feasible:
        return f_a < f_b
    return cv_a < cv_b  # ทั้งคู่ infeasible


# ---------------------------------------------------------------------------
# GWO core
# ---------------------------------------------------------------------------
def grey_wolf_optimizer(objective_fn, lb, ub, dim, pop_size=30, max_iter=500,
                         constraint_epsilon=1e-4, seed=None, verbose=True):
    """
    Parameters
    ----------
    objective_fn : callable
        รับ x shape (ps, dim) คืนค่า (f, g, h)
        f shape (ps,), g shape (ps, ng) หรือ None, h shape (ps,) หรือ (ps, nh) หรือ None
    lb, ub : np.ndarray shape (dim,)
        ขอบเขตล่าง/บนของแต่ละมิติ
    dim : int
        จำนวนมิติของปัญหา
    pop_size : int
        ขนาดประชากร (จำนวนหมาป่า)
    max_iter : int
        จำนวนรอบสูงสุด
    constraint_epsilon : float
        tolerance สำหรับ equality constraint ตอนคำนวณ constraint violation
    seed : int หรือ None
        random seed สำหรับทำซ้ำได้

    Returns
    -------
    dict ที่มี key: 'best_x', 'best_f', 'best_cv', 'convergence' (list ของ best_f ต่อรอบ)
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    # ---- initialize population ----
    positions = lb + rng.random((pop_size, dim)) * (ub - lb)

    def evaluate_population(x):
        f, g, h = objective_fn(x)
        if g is None:
            g = np.zeros((x.shape[0], 0))
        if h is None:
            h = np.zeros((x.shape[0], 0))
        cv = mean_constraint_violation(g, h, epsilon=constraint_epsilon)
        return f, cv

    f_vals, cv_vals = evaluate_population(positions)

    # ---- เลือก alpha, beta, delta เริ่มต้นตาม feasibility rules ----
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

    for t in range(max_iter):
        a = 2.0 - t * (2.0 / max_iter)  # ลดจาก 2 -> 0 เชิงเส้น

        for i in range(pop_size):
            for leader_pos in (alpha_pos, beta_pos, delta_pos):
                pass  # placeholder เพื่อความชัดเจนของโครงสร้าง (อัปเดตจริงด้านล่างแบบ vectorized)

        # --- อัปเดตตำแหน่งแบบ vectorized เทียบกับ alpha, beta, delta ---
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = np.abs(C1 * alpha_pos - positions)
        X1 = alpha_pos - A1 * D_alpha

        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = np.abs(C2 * beta_pos - positions)
        X2 = beta_pos - A2 * D_beta

        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = np.abs(C3 * delta_pos - positions)
        X3 = delta_pos - A3 * D_delta

        positions = (X1 + X2 + X3) / 3.0

        # boundary handling: clip กลับเข้า [lb, ub]
        positions = np.clip(positions, lb, ub)

        # ประเมินผลใหม่
        f_vals, cv_vals = evaluate_population(positions)

        # อัปเดต alpha, beta, delta จากประชากรทั้งหมด (รวมของเก่า เพื่อไม่ให้ leader แย่ลง)
        all_pos = np.vstack([positions, alpha_pos[None, :], beta_pos[None, :], delta_pos[None, :]])
        all_f = np.concatenate([f_vals, [alpha_f, beta_f, delta_f]])
        all_cv = np.concatenate([cv_vals, [alpha_cv, beta_cv, delta_cv]])

        order = rank_indices(all_f, all_cv)
        alpha_idx, beta_idx, delta_idx = order[0], order[1], order[2]
        alpha_pos, alpha_f, alpha_cv = all_pos[alpha_idx].copy(), all_f[alpha_idx], all_cv[alpha_idx]
        beta_pos, beta_f, beta_cv = all_pos[beta_idx].copy(), all_f[beta_idx], all_cv[beta_idx]
        delta_pos, delta_f, delta_cv = all_pos[delta_idx].copy(), all_f[delta_idx], all_cv[delta_idx]

        convergence.append(alpha_f if alpha_cv <= 1e-6 else np.nan)

        if verbose and (t % 50 == 0 or t == max_iter - 1):
            feas_str = "feasible" if alpha_cv <= 1e-6 else f"infeasible (CV={alpha_cv:.3e})"
            print(f"  iter {t:4d}/{max_iter}: alpha_f={alpha_f:.6e} [{feas_str}]")

    return dict(best_x=alpha_pos, best_f=alpha_f, best_cv=alpha_cv, convergence=convergence)


# ---------------------------------------------------------------------------
# Main: ทดสอบกับ SOPWM ทั้ง 6 instance
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    POP_SIZE = 50
    MAX_ITER = 4000     # pop=50 * iter=4000 = 200,000 FEs ตรงตามมาตรฐาน CEC2020 Eq.(60) สำหรับ 10<D<=30
    N_RUNS = 30         # ต้องเท่ากับ gwo_modified.py เพื่อให้เปรียบเทียบแบบ apple-to-apple (ablation)
    SEED_BASE = 42

    results_summary = []

    for prob_k in range(45, 51):
        dim = get_dimension(prob_k)
        lb, ub = get_bounds(prob_k)

        def objective_fn(x, prob_k=prob_k):
            return sopwm_evaluate(x, prob_k)

        print(f"\n=== RC{prob_k} (D={dim}), best-known f(x*) = {BEST_KNOWN[prob_k]:.6e} ===")

        run_bests = []
        t0 = time.time()
        for run in range(N_RUNS):
            res = grey_wolf_optimizer(
                objective_fn, lb, ub, dim,
                pop_size=POP_SIZE, max_iter=MAX_ITER,
                seed=SEED_BASE + run, verbose=(run == 0),
            )
            run_bests.append(res["best_f"] if res["best_cv"] <= 1e-6 else np.nan)
            print(f"  run {run + 1}/{N_RUNS}: best_f = {res['best_f']:.6e}, "
                  f"CV = {res['best_cv']:.3e}, feasible = {res['best_cv'] <= 1e-6}")
        elapsed = time.time() - t0

        run_bests = np.array(run_bests, dtype=float)
        feasible_runs = ~np.isnan(run_bests)
        print(f"  --> {feasible_runs.sum()}/{N_RUNS} feasible runs, "
              f"mean(feasible best_f) = {np.nanmean(run_bests):.6e}, "
              f"time = {elapsed:.1f}s")

        results_summary.append(dict(prob_k=prob_k, best_known=BEST_KNOWN[prob_k],
                                     run_bests=run_bests.tolist()))

    print("\n=== สรุปทั้งหมด ===")
    for r in results_summary:
        print(r)
