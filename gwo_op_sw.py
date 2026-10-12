"""
gwo_final.py

GWO-OP-SW: Grey Wolf Optimizer with Order-Preserving reparameterization
           and Square-Wave a-decay — algoritm สุดท้ายสำหรับเปเปอร์
=========================================================================

ที่มาของการออกแบบ (สรุปจาก ablation study ทั้งหมด):

1. GWO มาตรฐาน (position เป็นมุมสวิตชิ่งตรงๆ) ทำ ordering constraint พัง
   แทบทุก iteration → infeasible ตลอด (ดู gwo_baseline.py)

2. แก้ด้วย Order-Preserving Reparameterization: ค้นหาในปริภูมิ "ค่าผลต่าง"
   delta_1..delta_N (ไม่ติดลบ) แล้ว decode เป็นมุมจริงด้วย cumulative sum +
   rescale ให้ไม่เกิน upper bound → ordering constraint เป็นจริงโดยโครงสร้าง
   เสมอ ไม่ต้องพึ่ง repair/penalty อีกต่อไป (ดู gwo_modified.py)
   ผลลัพธ์: feasible 100% ทุก run แต่ยังพบ premature convergence รุนแรง
   (alpha ค้างนิ่งหลายร้อย iteration)

3. ทดสอบ a-decay schedule 5 แบบเพื่อแก้ premature convergence
   (linear, cosine, oscillating, square_wave, adaptive_switch) — square_wave
   ชนะทุกปัญหา ทุกเมตริก อย่างชัดเจน (ดู gwo_modified_v2.py)
   ข้อสังเกตที่น่าสนใจ: adaptive_switch (ปรับ a ตามการตรวจจับ stagnation)
   กลับแย่ที่สุด เพราะ threshold ต่ำเกินไปทำให้ a ค้างสูงตลอด ไม่เคย exploit
   จริงจัง (negative result ที่มีค่าสำหรับอภิปรายในเปเปอร์)

4. Tune duty cycle และ a_low ของ square_wave (5 config) แล้วยืนยันด้วย
   30 runs เต็มเทียบ 2 ตัวเต็ง → "combo" (periods=15, duty_high=0.3,
   a_low=0.01) ชนะ "best" ทุกปัญหา (เฉลี่ยดีกว่า ~20%) และเสมอกับ
   short_high บน "mean" (ต่างกันแค่ noise level) → เลือก combo เป็นค่า
   สุดท้าย (ดู gwo_squarewave_tuning.py)

ผลลัพธ์สุดท้าย (30 runs, MaxFEs=200,000 ตามมาตรฐาน CEC2020):
  RC45: best=1.24x, mean=1.99x ของ best-known
  RC46: best=1.24x, mean=2.81x
  RC47: best=1.58x, mean=2.81x
  RC48: best=1.73x, mean=3.10x
  RC49: best=2.14x, mean=4.78x
  RC50: best=1.00x (!), mean=1.78x  <- แตะ best-known พอดี

Feasibility: 30/30 runs ทุกปัญหา (100%), ordering constraint เป็นจริงเสมอ
โดยโครงสร้าง — ต่างจาก baseline algorithms เดิม (IUDE/eMAgES/eiLSHADE)
ที่มี Success Rate เกือบ 0% บนปัญหากลุ่มนี้
"""

import numpy as np
import time
from sopwm_objective import sopwm_evaluate, get_bounds, get_dimension, \
    mean_constraint_violation, BEST_KNOWN

EPS_MARGIN = 1e-5  # delta_min in Eq.(5) of the manuscript. DELIBERATELY LARGER than the
# ordering-constraint margin eps=1e-6 in sopwm_objective.py -- do NOT "fix" it to 1e-6.
# Rationale (manuscript Sec. 2.3, verified numerically): the decode rescales by
# sigma = min(1, alpha_max/sum(delta)), which contracts separations to sigma*delta_min.
# With delta_min=1e-5 the ordering constraints hold exactly while sum(delta) <= 900 deg,
# and the worst case the search space admits gives mean violation 1.78e-7 at D=30 --
# a factor 5.6 below the 1e-6 feasibility threshold. Setting delta_min=1e-6 shrinks that
# exact region to 90 deg and the safety factor to 1.5x, weakening the guarantee for no gain.


# ---- พารามิเตอร์สุดท้ายของ square_wave a-decay (จาก tuning ablation) ----
A_HIGH = 2.0
A_LOW = 0.01
PERIODS = 15
DUTY_HIGH = 0.3


def decode_delta_to_alpha(delta: np.ndarray, ub_total: float) -> np.ndarray:
    """Order-preserving decode: delta-space (ไม่ติดลบ) -> alpha-space (มุมสวิตชิ่งจริง)
    รับประกัน 0 < alpha_1 < alpha_2 < ... < alpha_D <= ub_total เสมอโดยโครงสร้าง"""
    delta = np.clip(delta, EPS_MARGIN, None)
    raw_alpha = np.cumsum(delta, axis=1)
    last = raw_alpha[:, -1]
    scale = np.where(last > ub_total, ub_total / last, 1.0)
    return raw_alpha * scale[:, np.newaxis]


def compute_a(t: int, max_iter: int, periods: int = PERIODS, duty_high: float = DUTY_HIGH,
              a_high: float = A_HIGH, a_low: float = A_LOW) -> float:
    """Square-wave a-decay: สลับ a_high/a_low ตามคาบเวลาคงที่"""
    period = max_iter / periods
    phase = t % period
    return a_high if phase < (duty_high * period) else a_low


def grey_wolf_optimizer_final(objective_fn, ub_total, dim, pop_size=50, max_iter=4000,
                               constraint_epsilon=1e-4, seed=None, verbose=True):
    """
    GWO-OP-SW — อัลกอริทึมสุดท้ายที่รวม order-preserving reparameterization
    กับ square-wave a-decay (พารามิเตอร์ที่ tune แล้ว)

    Parameters
    ----------
    objective_fn : callable รับ alpha shape (ps, dim) คืนค่า (f, g, h)
    ub_total : ขอบเขตบนของมุมสุดท้าย (90 องศา สำหรับ SOPWM)
    dim : จำนวนมุมสวิตชิ่ง (D)
    pop_size, max_iter : ขนาดประชากรและจำนวนรอบ (default ตรงมาตรฐาน CEC2020: 50*4000=200,000 FEs)

    Returns
    -------
    dict: best_alpha, best_f, best_cv, convergence
    """
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

    convergence = []

    for t in range(max_iter):
        a = compute_a(t, max_iter)

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

        convergence.append(alpha_f if alpha_cv <= 1e-6 else np.nan)

        if verbose and (t % 500 == 0 or t == max_iter - 1):
            feas_str = "feasible" if alpha_cv <= 1e-6 else f"infeasible (CV={alpha_cv:.3e})"
            print(f"  iter {t:4d}/{max_iter}: a={a:.3f}  alpha_f={alpha_f:.6e} [{feas_str}]")

    best_alpha = decode_delta_to_alpha(alpha_pos[None, :], ub_total)[0]
    return dict(best_alpha=best_alpha, best_f=alpha_f, best_cv=alpha_cv, convergence=convergence)


# ---------------------------------------------------------------------------
# Main: รันเต็มรูปแบบทั้ง 6 instance (30 runs, 200,000 FEs) — ผลลัพธ์สุดท้ายสำหรับเปเปอร์
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    POP_SIZE = 50
    MAX_ITER = 4000
    N_RUNS = 30
    SEED_BASE = 42
    UB_TOTAL = 90.0

    all_results = {}

    for prob_k in range(45, 51):
        dim = get_dimension(prob_k)

        def objective_fn(alpha, prob_k=prob_k):
            return sopwm_evaluate(alpha, prob_k)

        print(f"\n{'=' * 70}\nRC{prob_k} (D={dim}), best-known f(x*) = {BEST_KNOWN[prob_k]:.6e}\n{'=' * 70}")

        run_bests, run_cvs = [], []
        t0 = time.time()
        for run in range(N_RUNS):
            res = grey_wolf_optimizer_final(
                objective_fn, UB_TOTAL, dim,
                pop_size=POP_SIZE, max_iter=MAX_ITER,
                seed=SEED_BASE + run, verbose=(run == 0),
            )
            run_bests.append(res["best_f"] if res["best_cv"] <= 1e-6 else np.nan)
            run_cvs.append(res["best_cv"])
        elapsed = time.time() - t0

        run_bests = np.array(run_bests, dtype=float)
        feasible_runs = ~np.isnan(run_bests)
        bk = BEST_KNOWN[prob_k]
        print(f"[GWO-OP-SW] {feasible_runs.sum()}/{N_RUNS} feasible, "
              f"best={np.nanmin(run_bests):.6e} ({np.nanmin(run_bests)/bk:.2f}x), "
              f"mean={np.nanmean(run_bests):.6e} ({np.nanmean(run_bests)/bk:.2f}x), "
              f"std={np.nanstd(run_bests):.6e}, worst={np.nanmax(run_bests):.6e}, "
              f"time={elapsed:.1f}s")

        all_results[prob_k] = dict(
            best=float(np.nanmin(run_bests)), mean=float(np.nanmean(run_bests)),
            std=float(np.nanstd(run_bests)), worst=float(np.nanmax(run_bests)),
            feasible_rate=float(feasible_runs.mean()),
        )

    print(f"\n{'=' * 70}\nสรุปผลสุดท้าย GWO-OP-SW (สำหรับตารางในเปเปอร์)\n{'=' * 70}")
    print(f"{'Prob':6}{'best-known':>14}{'best':>14}{'mean':>14}{'std':>14}{'worst':>14}{'FR%':>8}")
    for prob_k, r in all_results.items():
        bk = BEST_KNOWN[prob_k]
        print(f"RC{prob_k:<4}{bk:14.6e}{r['best']:14.6e}{r['mean']:14.6e}"
              f"{r['std']:14.6e}{r['worst']:14.6e}{r['feasible_rate']*100:8.0f}")
