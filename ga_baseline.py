"""
ga_baseline.py

Real-coded Genetic Algorithm มาตรฐาน — same-protocol baseline สำหรับเทียบกับ
GA-OP-SW (ga_op_sw.py) ค้นหาตรงในปริภูมิมุมสวิตชิ่ง [0, 90] องศาเลย (ไม่มี
order-preserving reparameterization)

Operators มาตรฐาน:
  - Tournament selection (k=2)
  - SBX crossover (Simulated Binary Crossover, eta=15, pc=0.9)
  - Polynomial mutation (eta_m=20, pm คงที่ = 1/dim ตามธรรมเนียม — ไม่มี decay)
  - Elitism = 1 (เก็บตัวที่ดีที่สุดไว้เสมอ)

ใช้ Deb's feasibility rule จัดการ constraint เหมือนไฟล์ baseline อื่นทุกประการ
"""

import numpy as np
import time
from sopwm_objective import sopwm_evaluate, get_bounds, get_dimension, \
    mean_constraint_violation, BEST_KNOWN

ETA_C = 15      # SBX distribution index มาตรฐาน
PC = 0.9        # crossover probability มาตรฐาน
ETA_M = 20      # polynomial mutation distribution index มาตรฐาน


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


def tournament_select(rng, positions, f_vals, cv_vals, k=2):
    idxs = rng.integers(0, len(positions), size=k)
    best = idxs[0]
    for i in idxs[1:]:
        if is_better(f_vals[i], cv_vals[i], f_vals[best], cv_vals[best]):
            best = i
    return positions[best].copy()


def sbx_crossover(rng, p1, p2, lb, ub, eta=ETA_C, pc=PC):
    if rng.random() > pc:
        return p1.copy(), p2.copy()
    u = rng.random(len(p1))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)),
                     (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, lb, ub), np.clip(c2, lb, ub)


def polynomial_mutation(rng, x, lb, ub, eta=ETA_M, pm=None):
    dim = len(x)
    if pm is None:
        pm = 1.0 / dim
    x = x.copy()
    for j in range(dim):
        if rng.random() < pm:
            u = rng.random()
            delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 \
                else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            x[j] = x[j] + delta * (ub[j] - lb[j])
    return np.clip(x, lb, ub)


def genetic_algorithm(objective_fn, lb, ub, dim, pop_size=50, max_iter=4000,
                       constraint_epsilon=1e-4, seed=None, verbose=True):
    """Real-coded GA มาตรฐาน — ค้นหาตรงในปริภูมิ x (มุมสวิตชิ่งจริง [0,90])"""
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
    best_x, best_f, best_cv = positions[best_idx].copy(), f_vals[best_idx], cv_vals[best_idx]

    for t in range(max_iter):
        offspring = np.empty_like(positions)
        for i in range(0, pop_size, 2):
            p1 = tournament_select(rng, positions, f_vals, cv_vals)
            p2 = tournament_select(rng, positions, f_vals, cv_vals)
            c1, c2 = sbx_crossover(rng, p1, p2, lb, ub)
            c1 = polynomial_mutation(rng, c1, lb, ub)
            c2 = polynomial_mutation(rng, c2, lb, ub)
            offspring[i] = c1
            if i + 1 < pop_size:
                offspring[i + 1] = c2

        new_f, new_cv = evaluate(offspring)

        # elitism: แทนที่ตัวที่แย่ที่สุดของ offspring ด้วย best ของรุ่นก่อน
        worst_idx = 0
        for i in range(1, pop_size):
            if is_better(new_f[worst_idx], new_cv[worst_idx], new_f[i], new_cv[i]):
                worst_idx = i
        offspring[worst_idx] = best_x.copy()
        new_f[worst_idx], new_cv[worst_idx] = best_f, best_cv

        positions, f_vals, cv_vals = offspring, new_f, new_cv

        for i in range(pop_size):
            if is_better(f_vals[i], cv_vals[i], best_f, best_cv):
                best_x, best_f, best_cv = positions[i].copy(), f_vals[i], cv_vals[i]

        if verbose and (t % 500 == 0 or t == max_iter - 1):
            feas_str = "feasible" if best_cv <= 1e-6 else f"infeasible (CV={best_cv:.3e})"
            print(f"  iter {t:4d}/{max_iter}: best_f={best_f:.6e} [{feas_str}]")

    return dict(best_x=best_x, best_f=best_f, best_cv=best_cv)


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
            res = genetic_algorithm(objective_fn, lb, ub, dim, pop_size=POP_SIZE,
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
            print(f"[GA] {run_feasible.sum()}/{N_RUNS} feasible, "
                  f"best={feas_bests.min():.6e} ({feas_bests.min()/bk:.2f}x), "
                  f"mean(feasible)={feas_bests.mean():.6e} ({feas_bests.mean()/bk:.2f}x), "
                  f"time={elapsed:.1f}s")
        else:
            print(f"[GA] 0/{N_RUNS} feasible, time={elapsed:.1f}s")

        all_results[prob_k] = dict(f=run_bests.tolist(), feasible=run_feasible.tolist())

    print("\n=== สรุปทั้งหมด (GA baseline) ===")
    for k, v in all_results.items():
        print(f"RC{k}: {sum(v['feasible'])}/{N_RUNS} feasible")
