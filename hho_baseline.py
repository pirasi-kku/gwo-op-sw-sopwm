"""
hho_baseline.py

Harris Hawks Optimization (HHO) มาตรฐาน (Heidari et al., 2019) — same-protocol
baseline สำหรับเทียบกับ HHO-OP-SW (hho_op_sw.py) ค้นหาตรงในปริภูมิมุมสวิตชิ่ง
[0, 90] องศาเลย (ไม่มี order-preserving reparameterization) และใช้ escaping
energy E แบบ linear envelope มาตรฐาน (ไม่มี square-wave)

อ้างอิง: Heidari, A. A. et al. (2019). Harris hawks optimization: Algorithm
and applications. Future Generation Computer Systems. — ใช้สูตรมาตรฐานทั้ง
4 กลไก (soft besiege, hard besiege, soft besiege w/ rapid dives,
hard besiege w/ rapid dives) ไม่มีการดัดแปลง
"""

import math
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


def levy_flight(rng, dim, beta=1.5):
    """Levy flight step มาตรฐานที่ HHO ใช้ตอน rapid dive"""
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, size=dim)
    v = rng.normal(0, 1, size=dim)
    return u / (np.abs(v) ** (1 / beta))


def harris_hawks_optimizer(objective_fn, lb, ub, dim, pop_size=50, max_iter=4000,
                            constraint_epsilon=1e-4, seed=None, verbose=True):
    """HHO มาตรฐาน (Heidari et al., 2019) — ค้นหาตรงในปริภูมิ x (มุมสวิตชิ่งจริง [0,90])"""
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
    rabbit_pos = positions[best_idx].copy()
    rabbit_f, rabbit_cv = f_vals[best_idx], cv_vals[best_idx]

    for t in range(max_iter):
        E1 = 2.0 * (1.0 - t / max_iter)  # <- linear decay มาตรฐานของ escaping energy envelope

        new_positions = positions.copy()
        for i in range(pop_size):
            E0 = 2 * rng.random() - 1
            E = E1 * E0
            q = rng.random()

            if abs(E) >= 1:
                # exploration phase
                if q >= 0.5:
                    rand_idx = rng.integers(0, pop_size)
                    X_rand = positions[rand_idx]
                    r1, r2 = rng.random(), rng.random()
                    new_positions[i] = X_rand - r1 * np.abs(X_rand - 2 * r2 * positions[i])
                else:
                    mean_pos = positions.mean(axis=0)
                    r3, r4 = rng.random(), rng.random()
                    new_positions[i] = (rabbit_pos - mean_pos) - r4 * (lb + r3 * (ub - lb))
            else:
                # exploitation phase (4 กลไกมาตรฐาน)
                r = rng.random()
                if r >= 0.5 and abs(E) >= 0.5:
                    # soft besiege
                    J = 2 * (1 - rng.random())
                    delta_x = rabbit_pos - positions[i]
                    new_positions[i] = delta_x - E * np.abs(J * rabbit_pos - positions[i])
                elif r >= 0.5 and abs(E) < 0.5:
                    # hard besiege
                    new_positions[i] = rabbit_pos - E * np.abs(rabbit_pos - positions[i])
                elif r < 0.5 and abs(E) >= 0.5:
                    # soft besiege with progressive rapid dives
                    J = 2 * (1 - rng.random())
                    Y = rabbit_pos - E * np.abs(J * rabbit_pos - positions[i])
                    Y = np.clip(Y, lb, ub)
                    Z = Y + rng.random(dim) * levy_flight(rng, dim)
                    Z = np.clip(Z, lb, ub)
                    f_y, cv_y = evaluate(Y.reshape(1, -1))
                    f_z, cv_z = evaluate(Z.reshape(1, -1))
                    if is_better(f_y[0], cv_y[0], f_vals[i], cv_vals[i]):
                        new_positions[i] = Y
                    elif is_better(f_z[0], cv_z[0], f_vals[i], cv_vals[i]):
                        new_positions[i] = Z
                    else:
                        new_positions[i] = positions[i]
                else:
                    # hard besiege with progressive rapid dives
                    J = 2 * (1 - rng.random())
                    mean_pos = positions.mean(axis=0)
                    Y = rabbit_pos - E * np.abs(J * rabbit_pos - mean_pos)
                    Y = np.clip(Y, lb, ub)
                    Z = Y + rng.random(dim) * levy_flight(rng, dim)
                    Z = np.clip(Z, lb, ub)
                    f_y, cv_y = evaluate(Y.reshape(1, -1))
                    f_z, cv_z = evaluate(Z.reshape(1, -1))
                    if is_better(f_y[0], cv_y[0], f_vals[i], cv_vals[i]):
                        new_positions[i] = Y
                    elif is_better(f_z[0], cv_z[0], f_vals[i], cv_vals[i]):
                        new_positions[i] = Z
                    else:
                        new_positions[i] = positions[i]

        new_positions = np.clip(new_positions, lb, ub)
        new_f, new_cv = evaluate(new_positions)
        positions, f_vals, cv_vals = new_positions, new_f, new_cv

        for i in range(pop_size):
            if is_better(f_vals[i], cv_vals[i], rabbit_f, rabbit_cv):
                rabbit_pos, rabbit_f, rabbit_cv = positions[i].copy(), f_vals[i], cv_vals[i]

        if verbose and (t % 500 == 0 or t == max_iter - 1):
            feas_str = "feasible" if rabbit_cv <= 1e-6 else f"infeasible (CV={rabbit_cv:.3e})"
            print(f"  iter {t:4d}/{max_iter}: best_f={rabbit_f:.6e} [{feas_str}]")

    return dict(best_x=rabbit_pos, best_f=rabbit_f, best_cv=rabbit_cv)


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
            res = harris_hawks_optimizer(objective_fn, lb, ub, dim, pop_size=POP_SIZE,
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
            print(f"[HHO] {run_feasible.sum()}/{N_RUNS} feasible, "
                  f"best={feas_bests.min():.6e} ({feas_bests.min()/bk:.2f}x), "
                  f"mean(feasible)={feas_bests.mean():.6e} ({feas_bests.mean()/bk:.2f}x), "
                  f"time={elapsed:.1f}s")
        else:
            print(f"[HHO] 0/{N_RUNS} feasible, time={elapsed:.1f}s")

        all_results[prob_k] = dict(f=run_bests.tolist(), feasible=run_feasible.tolist())

    print("\n=== สรุปทั้งหมด (HHO baseline) ===")
    for k, v in all_results.items():
        print(f"RC{k}: {sum(v['feasible'])}/{N_RUNS} feasible")
