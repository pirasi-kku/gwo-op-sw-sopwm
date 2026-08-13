"""
woa_baseline.py

Whale Optimization Algorithm (WOA) มาตรฐาน — same-protocol baseline สำหรับเทียบกับ
WOA-OP-SW (woa_op_sw.py) ค้นหาตรงในปริภูมิมุมสวิตชิ่ง [0, 90] องศาเลย (ไม่มี
order-preserving reparameterization) และใช้ linear a-decay มาตรฐาน 2 -> 0
(ไม่มี square-wave) ใช้ Deb's feasibility rule จัดการ constraint เหมือนกับ
baseline อื่นทุกไฟล์ในโปรเจกต์ เพื่อให้เทียบกันแบบยุติธรรม (same protocol,
ต่างแค่ backbone ไม่มี OP-SW)

อ้างอิง: Mirjalili & Lewis (2016) — WOA update equations มาตรฐาน ไม่มีการดัดแปลง
"""

import numpy as np
import time
from sopwm_objective import sopwm_evaluate, get_bounds, get_dimension, \
    mean_constraint_violation, BEST_KNOWN


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


def whale_optimizer(objective_fn, lb, ub, dim, pop_size=50, max_iter=4000,
                     constraint_epsilon=1e-4, seed=None, verbose=True):
    """
    WOA มาตรฐาน — ค้นหาตรงในปริภูมิ x (มุมสวิตชิ่งจริง [0,90]) ไม่มี reparameterization
    ใด ๆ ordering constraint (x1<x2<...<xD) ปล่อยให้เป็นหน้าที่ของ inequality
    constraint g ในการ penalize เอง (เหมือน gwo_baseline.py / pso_baseline.py)
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    positions = lb + rng.random((pop_size, dim)) * (ub - lb)

    def evaluate(x):
        f, g, h = objective_fn(x)
        if g is None:
            g = np.zeros((x.shape[0], 0))
        if h is None:
            h = np.zeros((x.shape[0], 0))
        cv = mean_constraint_violation(g, h, epsilon=constraint_epsilon)
        return f, cv

    f_vals, cv_vals = evaluate(positions)

    best_idx = 0
    for i in range(1, pop_size):
        if is_better(f_vals[i], cv_vals[i], f_vals[best_idx], cv_vals[best_idx]):
            best_idx = i
    best_pos = positions[best_idx].copy()
    best_f, best_cv = f_vals[best_idx], cv_vals[best_idx]

    b_const = 1.0  # spiral shape constant (มาตรฐานของ Mirjalili & Lewis, 2016)

    for t in range(max_iter):
        a = 2.0 - t * (2.0 / max_iter)  # <- linear decay มาตรฐาน (ไม่ใช่ square-wave)

        new_positions = np.empty_like(positions)
        for i in range(pop_size):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            A_vec = 2 * a * r1 - a
            C_vec = 2 * r2
            p = rng.random()

            if p < 0.5:
                if np.linalg.norm(A_vec) < 1.0:
                    D_vec = np.abs(C_vec * best_pos - positions[i])
                    new_positions[i] = best_pos - A_vec * D_vec
                else:
                    rand_idx = rng.integers(0, pop_size)
                    rand_pos = positions[rand_idx]
                    D_vec = np.abs(C_vec * rand_pos - positions[i])
                    new_positions[i] = rand_pos - A_vec * D_vec
            else:
                l_vec = rng.uniform(-1.0, 1.0, size=dim)
                D_prime = np.abs(best_pos - positions[i])
                new_positions[i] = D_prime * np.exp(b_const * l_vec) * np.cos(2 * np.pi * l_vec) + best_pos

        new_positions = np.clip(new_positions, lb, ub)

        new_f, new_cv = evaluate(new_positions)
        positions, f_vals, cv_vals = new_positions, new_f, new_cv

        for i in range(pop_size):
            if is_better(f_vals[i], cv_vals[i], best_f, best_cv):
                best_pos, best_f, best_cv = positions[i].copy(), f_vals[i], cv_vals[i]

        if verbose and (t % 500 == 0 or t == max_iter - 1):
            feas_str = "feasible" if best_cv <= 1e-6 else f"infeasible (CV={best_cv:.3e})"
            print(f"  iter {t:4d}/{max_iter}: best_f={best_f:.6e} [{feas_str}]")

    return dict(best_x=best_pos, best_f=best_f, best_cv=best_cv)


if __name__ == "__main__":
    POP_SIZE = 50
    MAX_ITER = 4000
    N_RUNS = 30
    SEED_BASE = 42

    all_results = {}
    for prob_k in range(45, 51):
        dim = get_dimension(prob_k)
        lb, ub = get_bounds(prob_k)

        def objective_fn(x_pop, prob_k=prob_k):
            return sopwm_evaluate(x_pop, prob_k)

        print(f"\n=== RC{prob_k} (D={dim}), best-known f(x*) = {BEST_KNOWN[prob_k]:.6e} ===")
        run_bests = []
        run_feasible = []
        t0 = time.time()
        for run in range(N_RUNS):
            res = whale_optimizer(objective_fn, lb, ub, dim, pop_size=POP_SIZE,
                                   max_iter=MAX_ITER, seed=SEED_BASE + run,
                                   verbose=(run == 0))
            run_bests.append(res["best_f"])
            run_feasible.append(res["best_cv"] <= 1e-6)
        elapsed = time.time() - t0
        run_bests = np.array(run_bests, dtype=float)
        run_feasible = np.array(run_feasible, dtype=bool)
        bk = BEST_KNOWN[prob_k]

        feas_bests = run_bests[run_feasible]
        if feas_bests.size > 0:
            print(f"[WOA] {run_feasible.sum()}/{N_RUNS} feasible, "
                  f"best={feas_bests.min():.6e} ({feas_bests.min()/bk:.2f}x), "
                  f"mean(feasible)={feas_bests.mean():.6e} ({feas_bests.mean()/bk:.2f}x), "
                  f"time={elapsed:.1f}s")
        else:
            print(f"[WOA] 0/{N_RUNS} feasible, time={elapsed:.1f}s")

        all_results[prob_k] = dict(f=run_bests.tolist(), feasible=run_feasible.tolist())

    print("\n=== สรุปทั้งหมด (WOA baseline) ===")
    for k, v in all_results.items():
        n_feas = sum(v["feasible"])
        print(f"RC{k}: {n_feas}/{N_RUNS} feasible")
