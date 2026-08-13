"""
pso_baseline.py

Particle Swarm Optimization มาตรฐาน — same-protocol baseline สำหรับเทียบกับ
PSO-OP-SW (pso_op_sw.py) ค้นหาตรงในปริภูมิมุมสวิตชิ่ง [0, 90] องศาเลย (ไม่มี
order-preserving reparameterization) และใช้ linear inertia-weight decay
มาตรฐาน 0.9 -> 0.4 (ไม่มี square-wave) ใช้ Deb's feasibility rule เหมือนกับ
baseline อื่นทุกไฟล์

อ้างอิง: Shi & Eberhart (1998) — linear decreasing inertia weight มาตรฐาน
"""

import numpy as np
import time
from sopwm_objective import sopwm_evaluate, get_bounds, get_dimension, \
    mean_constraint_violation, BEST_KNOWN

W_MAX = 0.9
W_MIN = 0.4
C1 = 2.0
C2 = 2.0


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


def particle_swarm_optimizer(objective_fn, lb, ub, dim, pop_size=50, max_iter=4000,
                              c1=C1, c2=C2, constraint_epsilon=1e-4, seed=None,
                              verbose=True):
    """PSO มาตรฐาน (linear inertia decay) — ค้นหาตรงในปริภูมิ x (มุมสวิตชิ่งจริง [0,90])"""
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    positions = lb + rng.random((pop_size, dim)) * (ub - lb)
    v_range = 0.2 * (ub - lb)
    velocities = -v_range + rng.random((pop_size, dim)) * (2 * v_range)

    def evaluate(x):
        f, g, h = objective_fn(x)
        if g is None:
            g = np.zeros((x.shape[0], 0))
        if h is None:
            h = np.zeros((x.shape[0], 0))
        cv = mean_constraint_violation(g, h, epsilon=constraint_epsilon)
        return f, cv

    f_vals, cv_vals = evaluate(positions)
    pbest_pos = positions.copy()
    pbest_f, pbest_cv = f_vals.copy(), cv_vals.copy()

    best_idx = 0
    for i in range(1, pop_size):
        if is_better(f_vals[i], cv_vals[i], f_vals[best_idx], cv_vals[best_idx]):
            best_idx = i
    gbest_pos = positions[best_idx].copy()
    gbest_f, gbest_cv = f_vals[best_idx], cv_vals[best_idx]

    for t in range(max_iter):
        w = W_MAX - t * (W_MAX - W_MIN) / max_iter  # <- linear decay มาตรฐาน

        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos - positions))
        velocities = np.clip(velocities, -v_range, v_range)

        positions = positions + velocities
        positions = np.clip(positions, lb, ub)

        f_vals, cv_vals = evaluate(positions)

        for i in range(pop_size):
            if is_better(f_vals[i], cv_vals[i], pbest_f[i], pbest_cv[i]):
                pbest_pos[i] = positions[i].copy()
                pbest_f[i], pbest_cv[i] = f_vals[i], cv_vals[i]
            if is_better(pbest_f[i], pbest_cv[i], gbest_f, gbest_cv):
                gbest_pos = pbest_pos[i].copy()
                gbest_f, gbest_cv = pbest_f[i], pbest_cv[i]

        if verbose and (t % 500 == 0 or t == max_iter - 1):
            feas_str = "feasible" if gbest_cv <= 1e-6 else f"infeasible (CV={gbest_cv:.3e})"
            print(f"  iter {t:4d}/{max_iter}: best_f={gbest_f:.6e} [{feas_str}]")

    return dict(best_x=gbest_pos, best_f=gbest_f, best_cv=gbest_cv)


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
        run_bests, run_feasible = [], []
        t0 = time.time()
        for run in range(N_RUNS):
            res = particle_swarm_optimizer(objective_fn, lb, ub, dim, pop_size=POP_SIZE,
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
            print(f"[PSO] {run_feasible.sum()}/{N_RUNS} feasible, "
                  f"best={feas_bests.min():.6e} ({feas_bests.min()/bk:.2f}x), "
                  f"mean(feasible)={feas_bests.mean():.6e} ({feas_bests.mean()/bk:.2f}x), "
                  f"time={elapsed:.1f}s")
        else:
            print(f"[PSO] 0/{N_RUNS} feasible, time={elapsed:.1f}s")

        all_results[prob_k] = dict(f=run_bests.tolist(), feasible=run_feasible.tolist())

    print("\n=== สรุปทั้งหมด (PSO baseline) ===")
    for k, v in all_results.items():
        print(f"RC{k}: {sum(v['feasible'])}/{N_RUNS} feasible")
