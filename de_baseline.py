"""
de_baseline.py

Differential Evolution (DE/rand/1/bin) มาตรฐาน — same-protocol baseline สำหรับ
เทียบกับ DE-OP-SW (de_op_sw.py) ค้นหาตรงในปริภูมิมุมสวิตชิ่ง [0, 90] องศาเลย
(ไม่มี order-preserving reparameterization) และใช้ F คงที่มาตรฐาน (ไม่มี
square-wave decay) ใช้ Deb's feasibility rule เหมือนกับ baseline อื่นทุกไฟล์

อ้างอิง: Storn & Price (1997) — DE/rand/1/bin มาตรฐาน ไม่มีการดัดแปลง
"""

import numpy as np
import time
from sopwm_objective import sopwm_evaluate, get_bounds, get_dimension, \
    mean_constraint_violation, BEST_KNOWN

F_CONST = 0.5   # scale factor มาตรฐาน (Storn & Price, 1997) — คงที่ ไม่ decay
CR_CONST = 0.9  # crossover rate มาตรฐาน — คงที่


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


def differential_evolution(objective_fn, lb, ub, dim, pop_size=50, max_iter=4000,
                            f_const=F_CONST, cr=CR_CONST, constraint_epsilon=1e-4,
                            seed=None, verbose=True):
    """DE/rand/1/bin มาตรฐาน — ค้นหาตรงในปริภูมิ x (มุมสวิตชิ่งจริง [0,90])"""
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

    for t in range(max_iter):
        trials = np.empty_like(positions)
        for i in range(pop_size):
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)

            mutant = positions[r1] + f_const * (positions[r2] - positions[r3])
            mutant = np.clip(mutant, lb, ub)

            j_rand = rng.integers(0, dim)
            cross_mask = rng.random(dim) < cr
            cross_mask[j_rand] = True

            trials[i] = np.where(cross_mask, mutant, positions[i])

        trial_f, trial_cv = evaluate(trials)

        for i in range(pop_size):
            if is_better(trial_f[i], trial_cv[i], f_vals[i], cv_vals[i]):
                positions[i] = trials[i]
                f_vals[i] = trial_f[i]
                cv_vals[i] = trial_cv[i]

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
        run_bests, run_feasible = [], []
        t0 = time.time()
        for run in range(N_RUNS):
            res = differential_evolution(objective_fn, lb, ub, dim, pop_size=POP_SIZE,
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
            print(f"[DE] {run_feasible.sum()}/{N_RUNS} feasible, "
                  f"best={feas_bests.min():.6e} ({feas_bests.min()/bk:.2f}x), "
                  f"mean(feasible)={feas_bests.mean():.6e} ({feas_bests.mean()/bk:.2f}x), "
                  f"time={elapsed:.1f}s")
        else:
            print(f"[DE] 0/{N_RUNS} feasible, time={elapsed:.1f}s")

        all_results[prob_k] = dict(f=run_bests.tolist(), feasible=run_feasible.tolist())

    print("\n=== สรุปทั้งหมด (DE baseline) ===")
    for k, v in all_results.items():
        print(f"RC{k}: {sum(v['feasible'])}/{N_RUNS} feasible")
