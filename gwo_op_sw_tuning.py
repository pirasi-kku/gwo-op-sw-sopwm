"""
gwo_squarewave_tuning.py

Tune duty cycle / a_low ของ square_wave schedule (ตัวชนะจาก gwo_modified_v2.py)
ก่อนโปรโมทเป็น gwo_final.py

ทดสอบ 5 config (เปลี่ยนแค่ periods, duty_high, a_low ของ square_wave — โครงสร้างอื่น
เหมือน gwo_modified_v2.py ทุกประการ: order-preserving reparameterization, Deb's rule):

  baseline        : periods=10, duty_high=0.5, a_low=0.10   (ค่าเดิมที่ชนะใน ablation ก่อนหน้า)
  more_periods    : periods=20, duty_high=0.5, a_low=0.10   (สลับถี่ขึ้น 2 เท่า)
  short_high      : periods=10, duty_high=0.3, a_low=0.10   (explore สั้นลง, exploit ยาวขึ้น)
  deeper_exploit  : periods=10, duty_high=0.5, a_low=0.01   (a_low ต่ำกว่าเดิม 10 เท่า)
  combo           : periods=15, duty_high=0.3, a_low=0.01   (รวมทุกไอเดีย)
"""

import numpy as np
import time
from sopwm_objective import sopwm_evaluate, get_bounds, get_dimension, \
    mean_constraint_violation, BEST_KNOWN

EPS_MARGIN = 1e-5


def decode_delta_to_alpha(delta: np.ndarray, ub_total: float) -> np.ndarray:
    delta = np.clip(delta, EPS_MARGIN, None)
    raw_alpha = np.cumsum(delta, axis=1)
    last = raw_alpha[:, -1]
    scale = np.where(last > ub_total, ub_total / last, 1.0)
    return raw_alpha * scale[:, np.newaxis]


def compute_a_square_wave(t, max_iter, periods, duty_high, a_high=2.0, a_low=0.1):
    period = max_iter / periods
    phase = t % period
    a = a_high if phase < (duty_high * period) else a_low
    return float(a)


def grey_wolf_optimizer_op(objective_fn, ub_total, dim, pop_size=30, max_iter=500,
                            periods=10, duty_high=0.5, a_high=2.0, a_low=0.1,
                            constraint_epsilon=1e-4, seed=None, verbose=True):
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
        return f, cv

    f_vals, cv_vals = evaluate_population(positions)

    def rank_indices(f_vals, cv_vals):
        idx = list(range(len(f_vals)))
        idx.sort(key=lambda i: (cv_vals[i] > 1e-6, cv_vals[i] if cv_vals[i] > 1e-6 else f_vals[i]))
        return idx

    order = rank_indices(f_vals, cv_vals)
    alpha_idx, beta_idx, delta_idx = order[0], order[1], order[2]
    alpha_pos, alpha_f, alpha_cv = positions[alpha_idx].copy(), f_vals[alpha_idx], cv_vals[alpha_idx]
    beta_pos, beta_f, beta_cv = positions[beta_idx].copy(), f_vals[beta_idx], cv_vals[beta_idx]
    delta_pos, delta_f, delta_cv = positions[delta_idx].copy(), f_vals[delta_idx], cv_vals[delta_idx]

    for t in range(max_iter):
        a = compute_a_square_wave(t, max_iter, periods, duty_high, a_high, a_low)

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

        f_vals, cv_vals = evaluate_population(positions)

        all_pos = np.vstack([positions, alpha_pos[None, :], beta_pos[None, :], delta_pos[None, :]])
        all_f = np.concatenate([f_vals, [alpha_f, beta_f, delta_f]])
        all_cv = np.concatenate([cv_vals, [alpha_cv, beta_cv, delta_cv]])

        order = rank_indices(all_f, all_cv)
        alpha_idx, beta_idx, delta_idx = order[0], order[1], order[2]
        alpha_pos, alpha_f, alpha_cv = all_pos[alpha_idx].copy(), all_f[alpha_idx], all_cv[alpha_idx]
        beta_pos, beta_f, beta_cv = all_pos[beta_idx].copy(), all_f[beta_idx], all_cv[beta_idx]
        delta_pos, delta_f, delta_cv = all_pos[delta_idx].copy(), all_f[delta_idx], all_cv[delta_idx]

    return dict(best_f=alpha_f, best_cv=alpha_cv)


if __name__ == "__main__":
    POP_SIZE = 50
    MAX_ITER = 4000
    N_RUNS = 10          # ลดจาก 30 เพื่อเทสเร็วขึ้นก่อน ค่อยรัน 30 เต็มกับ config ที่ชนะทีหลัง
    SEED_BASE = 42
    UB_TOTAL = 90.0

    CONFIGS = {
        "baseline":        dict(periods=10, duty_high=0.5, a_low=0.10),
        "more_periods":    dict(periods=20, duty_high=0.5, a_low=0.10),
        "short_high":      dict(periods=10, duty_high=0.3, a_low=0.10),
        "deeper_exploit":  dict(periods=10, duty_high=0.5, a_low=0.01),
        "combo":           dict(periods=15, duty_high=0.3, a_low=0.01),
    }

    all_results = {}

    for prob_k in range(45, 51):
        dim = get_dimension(prob_k)

        def objective_fn(alpha, prob_k=prob_k):
            return sopwm_evaluate(alpha, prob_k)

        print(f"\n{'=' * 70}\nRC{prob_k} (D={dim}), best-known f(x*) = {BEST_KNOWN[prob_k]:.6e}\n{'=' * 70}")

        for cfg_name, cfg in CONFIGS.items():
            run_bests = []
            t0 = time.time()
            for run in range(N_RUNS):
                res = grey_wolf_optimizer_op(
                    objective_fn, UB_TOTAL, dim,
                    pop_size=POP_SIZE, max_iter=MAX_ITER,
                    seed=SEED_BASE + run, verbose=False, **cfg,
                )
                run_bests.append(res["best_f"] if res["best_cv"] <= 1e-6 else np.nan)
            elapsed = time.time() - t0

            run_bests = np.array(run_bests, dtype=float)
            feasible_runs = ~np.isnan(run_bests)
            mean_f = np.nanmean(run_bests)
            best_f = np.nanmin(run_bests)
            print(f"[{cfg_name:>15}] {feasible_runs.sum()}/{N_RUNS} feasible, "
                  f"best={best_f:.6e} ({best_f/BEST_KNOWN[prob_k]:.2f}x), "
                  f"mean={mean_f:.6e} ({mean_f/BEST_KNOWN[prob_k]:.2f}x), "
                  f"time={elapsed:.1f}s")

            all_results[(prob_k, cfg_name)] = dict(best=float(best_f), mean=float(mean_f))

    print(f"\n{'=' * 70}\nสรุปทั้งหมด\n{'=' * 70}")
    for prob_k in range(45, 51):
        print(f"\nRC{prob_k} (best-known={BEST_KNOWN[prob_k]:.6e}):")
        for cfg_name in CONFIGS:
            r = all_results[(prob_k, cfg_name)]
            print(f"  {cfg_name:>15}: mean={r['mean']:.6e} ({r['mean']/BEST_KNOWN[prob_k]:.2f}x)  "
                  f"best={r['best']:.6e} ({r['best']/BEST_KNOWN[prob_k]:.2f}x)")
