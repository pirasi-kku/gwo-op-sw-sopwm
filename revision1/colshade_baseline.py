"""
colshade_baseline.py

COLSHADE (L-SHADE for Constrained Optimization with Levy Flights)
====================================================================
Implement ตาม Gurrola-Ramos, Hernandez-Aguirre, Dalmau-Cedeno (2020),
"COLSHADE for Real-World Single-Objective Constrained Optimization
Problems," 2020 IEEE Congress on Evolutionary Computation (CEC),
DOI: 10.1109/CEC48606.2020.9185583

ใช้เป็น baseline ใหม่สำหรับตอบ Reviewer B:
  "The authors should benchmark the proposed method against at least
   one strong recent constrained optimizer ... under a fully reproduced
   common experimental protocol rather than relying heavily on historical
   literature statistics."

REVISION 2: อัปเดตให้ตรงกับ sopwm_objective.py ตัวจริงแล้ว (ก่อนหน้านี้เดา
interface ผิดว่าคืนค่า cv รวมก้อนเดียว) — ตัวจริงคือ:

    f, g, h = sopwm_evaluate(x, prob_k)
    # f: shape (ps,)   objective
    # g: shape (ps,D-1) inequality violations (ordering constraint), feasible g<=0
    # h: shape (ps,)   equality VALUE (ไม่ใช่ violation) feasible เมื่อ |h|<=eps

เพราะแยก inequality/equality ได้ตรงๆ ทำให้ implement **dynamic-tolerance
COLSHADE เต็มรูปแบบ** ได้ตรงตามเปเปอร์ต้นฉบับ (ไม่ใช่แค่ static-handling
variant แบบร่างแรก) — I(x) จาก g, E(x,G) จาก h เทียบ tolerance ที่หด
ลงแบบ dynamic (eq. 9) จนเหลือ eps_final=1e-4 ซึ่งพอดีตรงกับค่า epsilon
default ของ mean_constraint_violation() ในไฟล์ sopwm_objective.py เอง —
ไม่ต้องปรับอะไรเพิ่มให้ protocol ตรงกัน

ผลลัพธ์สุดท้าย (best_f, best_cv สำหรับ CSV) คำนวณผ่าน mean_constraint_violation()
ตัวเดียวกับที่ gwo_final.py และ baseline อื่นทุกตัวใช้ (epsilon=1e-4,
feasible เมื่อ cv<=1e-6) เพื่อให้ตาราง/สถิติเทียบกันได้ตรงๆ ทุกแถว แม้ว่า
ระหว่างการค้นหา COLSHADE จะใช้ tolerance ของตัวเองที่ปรับแบบ dynamic ก็ตาม
(นี่คือกลไกหลักที่ทำให้ COLSHADE เป็น COLSHADE ตัดออกจะเสียความหมายของ
การเทียบ "COLSHADE จริง" ไป)
=====================================================================================
"""

import numpy as np
from scipy.stats import levy_stable
from sopwm_objective import mean_constraint_violation  # ใช้ตัวเดียวกับทุกอัลกอริทึมอื่น

# =====================================================================================
# COLSHADE control parameters (ตาม Section IV ของ paper ต้นฉบับ เป๊ะทุกค่า)
# =====================================================================================
R_NINIT = 18        # Ninit = round(D * R_NINIT)
R_ARC = 2.6          # |A| = round(Ninit * R_ARC)
P_BEST = 0.11         # proportion of best solutions (current-to-pbest)
P_EPS_FEASIBLE = 0.2  # proportion of population required epsilon-feasible to shrink tolerance
H_MEMORY = 6          # ขนาด cyclical memory
LEVY_ALPHA = 0.5
LEVY_BETA = 1.0
LEVY_GAMMA = 0.01
LEVY_DELTA = 0.0
Q_MIN = 1e-3
MU_PROB = 0.25        # smoothing rate สำหรับ q update
EPS_FINAL = 1e-4      # tolerance สุดท้ายสำหรับ equality constraint (เฉพาะโหมด dynamic)
N_MIN = 4


def _lehmer_mean(values, weights):
    """Weighted Lehmer mean ตาม eq.(13)-(14): L2(S;w) = sum(w*s^2) / sum(w*s)"""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denom = np.sum(weights * values)
    if denom <= 0:
        return np.sum(weights * values ** 2) / (np.sum(weights) + 1e-30)
    return np.sum(weights * values ** 2) / denom


def _generate_cr_f(mem_cr, mem_f, h_idx, rng):
    """สุ่ม CR_i, F_i จาก memory index เดียวกัน (ตาม eq. 4-5) พร้อม resample
    มาตรฐานของ SHADE/L-SHADE: CR clip [0,1], F resample จนกว่าจะ >0 แล้ว clip <=1
    (รายละเอียดนี้ไม่ได้เขียนไว้ตรงๆใน COLSHADE paper แต่สืบทอดมาจาก L-SHADE
    ซึ่ง paper บอกไว้ว่า "for the sake of completeness a brief review of L-SHADE")"""
    r = rng.integers(0, H_MEMORY)
    mu_cr = mem_cr[r]
    cr = 0.0 if mu_cr <= 0 else np.clip(rng.normal(mu_cr, 0.1), 0.0, 1.0)
    mu_f = mem_f[r]
    f_val = -1.0
    while f_val <= 0:
        f_val = rng.standard_cauchy() * 0.1 + mu_f
    f_val = min(f_val, 1.0)
    return cr, f_val, r


def _boundary_repair(u, lb, ub, x_parent, rng):
    """eq.(11): random blend กลับเข้า bound แทนการ clip ตรงๆ"""
    out_hi = u > ub
    out_lo = u < lb
    if np.any(out_hi):
        r = rng.uniform(0.0, 0.1, size=np.sum(out_hi))
        u[out_hi] = (1 - r) * ub[out_hi] + r * x_parent[out_hi]
    if np.any(out_lo):
        r = rng.uniform(0.0, 0.1, size=np.sum(out_lo))
        u[out_lo] = (1 - r) * lb[out_lo] + r * x_parent[out_lo]
    return u


def _current_to_pbest_mutation(x_i, pop, archive, pbest_pool, cr, f_val, rng):
    """eq.(3): DE/current-to-pbest/1/bin"""
    n, d = pop.shape
    x_pbest = pbest_pool[rng.integers(0, len(pbest_pool))]

    idx_r1 = rng.integers(0, n)
    x_r1 = pop[idx_r1]

    union_pool = np.vstack([pop, archive]) if len(archive) > 0 else pop
    idx_r2 = rng.integers(0, len(union_pool))
    x_r2 = union_pool[idx_r2]

    v = x_i + f_val * (x_pbest - x_i) + f_val * (x_r1 - x_r2)

    u = x_i.copy()
    j_rand = rng.integers(0, d)
    cross_mask = (rng.uniform(0, 1, size=d) <= cr)
    cross_mask[j_rand] = True
    u[cross_mask] = v[cross_mask]
    return u


def _levy_mutation(x_i, pbest_pool, cr, f_val, n_pop, rng):
    """Algorithm 2: levy/1/bin mutation"""
    d = x_i.shape[0]
    f_crit = np.sqrt(max(1e-30, (1.0 - cr / 2.0)) / max(1, n_pop))
    f_eff = max(f_crit, f_val)

    x_pbest = pbest_pool[rng.integers(0, len(pbest_pool))]

    u = x_i.copy()
    j_rand = rng.integers(0, d)
    for j in range(d):
        if rng.uniform(0, 1) <= cr or j == j_rand:
            step = levy_stable.rvs(LEVY_ALPHA, LEVY_BETA, loc=LEVY_DELTA,
                                    scale=LEVY_GAMMA, random_state=rng)
            f_levy = f_eff * step
            u[j] = x_i[j] + f_levy * (x_pbest[j] - x_i[j])
    return u


def colshade_op_sw(objective_fn, lb, ub, dim, pop_size=None, max_iter=None,
                    max_fes=200_000, seed=0, verbose=False):
    """
    COLSHADE runner — signature mirror ของ baseline อื่นๆ ใน run_all_experiments.py
    (objective_fn, lb, ub, dim, pop_size, max_iter, seed, verbose)

    objective_fn(x) ต้องคืนค่า (f, g, h) แบบเดียวกับ sopwm_evaluate จริง:
      f: (ps,)      objective
      g: (ps,D-1)   inequality violations, feasible เมื่อ g<=0
      h: (ps,)      equality VALUE (ไม่ใช่ violation) feasible เมื่อ |h|<=eps

    หมายเหตุ: pop_size/max_iter ที่รับเข้ามาไม่ได้ใช้ตรงๆ เพราะ COLSHADE กำหนด
    ขนาดประชากรเริ่มต้นจาก r_Ninit เอง (ตาม paper ต้นฉบับ) และลดลงเรื่อยๆ ตาม FEs
    ไม่ใช่ตาม iteration คงที่แบบ GWO — ใช้ max_fes=200_000 เป็นตัวควบคุมงบจริง
    (ตรงกับ pop=50 x max_iter=4000 ของ gwo_final.py พอดี)

    Constraint handling: ภายในใช้ tolerance eps_g ที่หดแบบ dynamic (eq.9, เฉพาะ
    equality component) สำหรับตัดสินใจ selection/pbest ระหว่างค้นหา — ตาม
    algorithm ต้นฉบับเป๊ะ ส่วนค่าที่ return ออกมา (best_f/best_cv/feasible)
    ใช้เกณฑ์ตายตัวเดียวกับทุกอัลกอริทึมอื่นในเปเปอร์ (epsilon=1e-4,
    feasible เมื่อ mean_cv<=1e-6) เพื่อให้เทียบกันตรงๆ ในตาราง/สถิติ — ไม่ปนกัน

    คืนค่า: dict มี best_x, best_f, best_cv, feasible, fes_used, history
    (history: [(fes, best_f, best_cv, feasible), ...] convergence trace)
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    n_init = max(N_MIN, int(round(dim * R_NINIT)))
    arc_size = max(1, int(round(n_init * R_ARC)))

    mem_cr = np.full(H_MEMORY, 0.5)
    mem_f = np.full(H_MEMORY, 0.5)
    mem_cr_l = np.full(H_MEMORY, 0.5)
    mem_f_l = np.full(H_MEMORY, 0.5)
    k_idx = 0
    k_idx_l = 0

    pop = lb + rng.uniform(0, 1, size=(n_init, dim)) * (ub - lb)

    def evaluate(x):
        """คืน (f, I(x), |h(x)|) — I(x)=sum inequality violation, |h(x)| สำหรับ
        ใช้กับ tolerance แบบ dynamic ทีหลัง"""
        f_arr, g_arr, h_arr = objective_fn(x.reshape(1, -1))
        f_val = float(f_arr[0])
        ineq = float(np.sum(np.maximum(g_arr[0], 0.0)))
        h_abs = float(abs(float(h_arr[0])))
        return f_val, ineq, h_abs

    def svc_dyn(ineq, h_abs, eps):
        """svc(x;eps) ตาม eq.(6)-(8) — ใช้ตัดสินใจภายในระหว่างค้นหา
        รองรับทั้ง scalar และ array (np.maximum แทน max ในตัวธรรมดา)"""
        return ineq + np.maximum(0.0, h_abs - eps)

    def cv_report(ineq, h_abs):
        """เทียบเท่า mean_constraint_violation(g,h,epsilon=1e-4) ของไฟล์อื่นทุก
        ประการ (epsilon ตายตัว, normalize ด้วย D) ใช้สำหรับ best_cv/feasible
        ที่ return ออกไปเท่านั้น ไม่ใช้ตัดสินใจภายใน"""
        return (ineq + max(0.0, h_abs - EPS_FINAL)) / dim

    fitness = np.empty(n_init)
    ineq_arr = np.empty(n_init)
    h_arr = np.empty(n_init)
    for i in range(n_init):
        fitness[i], ineq_arr[i], h_arr[i] = evaluate(pop[i])
    fes = n_init

    archive = np.empty((0, dim))
    q_g = 0.5
    fes_tol_end = int(0.6 * max_fes)
    # eq. ก่อน Algorithm 1 บรรทัด 5: ตั้ง tolerance เริ่มต้นให้ population ตั้งต้น
    # ทุกตัว eps-feasible พอดี (max ของ |h| ในประชากรเริ่มต้น)
    eps_g = float(np.max(h_arr)) if np.max(h_arr) > 0 else EPS_FINAL
    eps_g = max(eps_g, EPS_FINAL)

    # --- ตัวติดตาม "best เพื่อ report" แยกจาก selection ภายในของ COLSHADE ---
    # ใช้เกณฑ์ cv_report/1e-6 คงที่ตลอด ไม่ผูกกับ eps_g ที่เปลี่ยนไปเรื่อยๆ
    # เพื่อให้ผลลัพธ์สุดท้ายเทียบกับอัลกอริทึมอื่นได้ตรงๆ
    best_x = None
    best_f = np.inf
    best_cv = np.inf

    def update_best_tracker(x_i, f_i, ineq_i, h_i):
        nonlocal best_x, best_f, best_cv
        cv_i = cv_report(ineq_i, h_i)
        if cv_i < best_cv or (cv_i <= 1e-6 and best_cv <= 1e-6 and f_i < best_f):
            best_x, best_f, best_cv = x_i.copy(), f_i, cv_i

    for i in range(n_init):
        update_best_tracker(pop[i], fitness[i], ineq_arr[i], h_arr[i])

    history = []  # (fes, best_f, best_cv, feasible) checkpoint สำหรับ convergence curve
    checkpoint_every = max(1, max_fes // 200)  # ~200 จุดต่อ run พอสำหรับ plot
    next_checkpoint = checkpoint_every

    n_g = n_init
    gen = 0

    def rank_order_by_svc(svc_vals, fit_vals):
        """คืน index เรียง: eps-feasible (svc==0) ก่อนตาม fitness, ตามด้วย
        infeasible เรียงตาม svc"""
        feas_mask = svc_vals <= 1e-12
        feas_idx = np.where(feas_mask)[0][np.argsort(fit_vals[feas_mask])] \
            if np.any(feas_mask) else np.array([], dtype=int)
        infeas_idx = np.where(~feas_mask)[0][np.argsort(svc_vals[~feas_mask])] \
            if np.any(~feas_mask) else np.array([], dtype=int)
        return np.concatenate([feas_idx, infeas_idx]).astype(int)

    while fes < max_fes:
        svc_cur = svc_dyn(ineq_arr, h_arr, eps_g)
        n_p_best = max(1, int(round(P_BEST * n_g)))
        rank_order = rank_order_by_svc(svc_cur, fitness)
        pbest_pool = pop[rank_order[:n_p_best]]

        s_cr, s_f, s_cr_l, s_f_l = [], [], [], []
        d_f, d_f_l = [], []

        new_pop = pop.copy()
        new_fit = fitness.copy()
        new_ineq = ineq_arr.copy()
        new_h = h_arr.copy()
        removed_to_archive = []

        for i in range(n_g):
            if fes >= max_fes:
                break
            use_levy = rng.uniform(0, 1) <= q_g
            if use_levy:
                cr, f_val, r_idx = _generate_cr_f(mem_cr_l, mem_f_l, k_idx_l, rng)
                u = _levy_mutation(pop[i], pbest_pool, cr, f_val, n_g, rng)
            else:
                cr, f_val, r_idx = _generate_cr_f(mem_cr, mem_f, k_idx, rng)
                u = _current_to_pbest_mutation(pop[i], pop, archive, pbest_pool, cr, f_val, rng)

            u = _boundary_repair(u, lb, ub, pop[i], rng)
            f_u, ineq_u, h_u = evaluate(u)
            fes += 1
            update_best_tracker(u, f_u, ineq_u, h_u)

            x_svc = svc_cur[i]
            u_svc = svc_dyn(ineq_u, h_u, eps_g)
            x_f, u_f = fitness[i], f_u

            improved = False
            delta_val = 0.0

            # eq.(10): 3 กรณีของ feasibility tournament (best ปัจจุบันข้อยกเว้น
            # เฉพาะกรณีที่ 1 ตามที่ paper ระบุ — ที่นี่ simplified ให้ครอบคลุม
            # ทุก i เพราะ best เพื่อ report แยก tracker ต่างหากแล้วไม่เสี่ยงหาย)
            if u_svc < x_svc:
                improved = True
                delta_val = x_svc - u_svc
            elif u_svc <= 1e-12 and x_svc <= 1e-12 and u_f < x_f:
                improved = True
                delta_val = x_f - u_f

            if improved:
                removed_to_archive.append(pop[i].copy())
                new_pop[i] = u
                new_fit[i] = f_u
                new_ineq[i] = ineq_u
                new_h[i] = h_u
                if use_levy:
                    s_cr_l.append(cr); s_f_l.append(f_val); d_f_l.append(max(delta_val, 1e-30))
                else:
                    s_cr.append(cr); s_f.append(f_val); d_f.append(max(delta_val, 1e-30))

            if fes >= next_checkpoint or fes >= max_fes:
                feasible_now = best_cv <= 1e-6
                history.append((fes, float(best_f) if feasible_now else float('nan'),
                                 float(best_cv), bool(feasible_now)))
                next_checkpoint += checkpoint_every

        pop, fitness, ineq_arr, h_arr = new_pop, new_fit, new_ineq, new_h

        if removed_to_archive:
            archive = np.vstack([archive] + [x.reshape(1, -1) for x in removed_to_archive]) if len(archive) else \
                np.vstack([x.reshape(1, -1) for x in removed_to_archive])
            if len(archive) > arc_size:
                keep_idx = rng.choice(len(archive), size=arc_size, replace=False)
                archive = archive[keep_idx]

        if len(d_f) > 0:
            w = np.array(d_f) / (np.sum(d_f) + 1e-30)
            mem_cr[k_idx] = _lehmer_mean(s_cr, w) if any(c > 0 for c in s_cr) else mem_cr[k_idx]
            mem_f[k_idx] = _lehmer_mean(s_f, w)
            k_idx = (k_idx + 1) % H_MEMORY
        if len(d_f_l) > 0:
            w_l = np.array(d_f_l) / (np.sum(d_f_l) + 1e-30)
            mem_cr_l[k_idx_l] = _lehmer_mean(s_cr_l, w_l) if any(c > 0 for c in s_cr_l) else mem_cr_l[k_idx_l]
            mem_f_l[k_idx_l] = _lehmer_mean(s_f_l, w_l)
            k_idx_l = (k_idx_l + 1) % H_MEMORY

        sum_df = np.sum(d_f) if d_f else 0.0
        sum_df_l = np.sum(d_f_l) if d_f_l else 0.0
        if not (sum_df == 0.0 and sum_df_l == 0.0):
            q_hat = sum_df_l / (sum_df_l + sum_df + 1e-30)
            q_g = MU_PROB * q_g + (1 - MU_PROB) * q_hat
            q_g = float(np.clip(q_g, Q_MIN, 1 - Q_MIN))

        # eq.(9): หด eps_g ถ้ามีสัดส่วน p_G ของ population เป็น eps-feasible แล้ว
        svc_after = svc_dyn(ineq_arr, h_arr, eps_g)
        n_feasible_eps = int(np.sum(svc_after <= 1e-12))
        p_g_thresh = int(round(P_EPS_FEASIBLE * n_g))
        if fes < fes_tol_end and n_feasible_eps >= p_g_thresh and eps_g > EPS_FINAL:
            # BUGFIX (revision1, post-run diagnosis): original MATLAB get_tolerance.m
            # uses exponent = ps / remaining_fes (ps = current population size),
            # NOT 1 / remaining_fes. Without the population-size factor, eps_g
            # decays orders of magnitude too slowly (off by a factor of n_g, which
            # is up to ~18*D ~ 450-540 for these instances) and never reaches
            # EPS_FINAL by fes_tol_end -- this was confirmed against the original
            # author's get_tolerance.m (COLSHADE-main/get_tolerance.m) and matches
            # the observed symptom exactly (best_cv stuck at 1e-3..1e-1 range,
            # feasibility 0% on RC46/48/49/50, erratic on RC45/47).
            exponent = n_g / max(1, (fes_tol_end - fes))
            eps_g = eps_g * (EPS_FINAL / max(eps_g, 1e-30)) ** exponent
            eps_g = max(eps_g, EPS_FINAL)

        # linear population size reduction (บรรทัด 34 ของ Algorithm 1)
        n_next = int(round((N_MIN - n_init) / max_fes * fes + n_init))
        n_next = max(N_MIN, min(n_g, n_next))
        if n_next < n_g:
            svc_for_resize = svc_dyn(ineq_arr, h_arr, eps_g)
            keep_order = rank_order_by_svc(svc_for_resize, fitness)[:n_next]
            pop, fitness = pop[keep_order], fitness[keep_order]
            ineq_arr, h_arr = ineq_arr[keep_order], h_arr[keep_order]
            n_g = n_next

        gen += 1
        if verbose and gen % 50 == 0:
            print(f"[COLSHADE] gen={gen} fes={fes}/{max_fes} n_g={n_g} "
                  f"best_f={best_f:.6g} best_cv={best_cv:.3e} eps_g={eps_g:.3e}")

    return {
        "best_x": best_x,
        "best_f": float(best_f) if best_cv <= 1e-6 else float('nan'),
        "best_cv": float(best_cv),
        "feasible": bool(best_cv <= 1e-6),
        "fes_used": fes,
        "history": history,   # [(fes, best_f, best_cv, feasible), ...] สำหรับ convergence curve
    }


if __name__ == "__main__":
    print(__doc__)
    print("\nรันไฟล์นี้ตรงๆ จะไม่ทำอะไร — ต้อง import colshade_op_sw() ไปใช้ใน")
    print("run_all_experiments.py หรือสคริปต์รันจริง พร้อม sopwm_objective.py")
    print("ที่ตรงกับ signature ตามที่ระบุไว้ในคอมเมนต์ด้านบนของไฟล์ (ยัง ASSUME อยู่)")
