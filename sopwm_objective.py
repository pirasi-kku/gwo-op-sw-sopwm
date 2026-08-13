"""
sopwm_objective.py

Objective function สำหรับปัญหา Synchronous Optimal PWM (SOPWM), instance RC45-RC50
จาก CEC2020 Real-World Constrained Optimization benchmark suite (Kumar et al., 2020)

ยืนยันสูตร/ค่าคงที่กับโค้ด MATLAB ต้นฉบับ (cec20_func.m, Cal_par.m) แล้ว:
  - มุมสวิตชิ่งอยู่ในหน่วย "องศา" [0, 90] ไม่ใช่เรเดียนตรงๆ (แปลงหน่วยตอนคำนวณ)
  - ordering margin ของ inequality constraint คือ 1e-6
  - RC47 (7-level) มี modulation index m=0.36 ต่างจาก instance อื่น
"""

import numpy as np

# k ลำดับฮาร์มอนิกที่พิจารณา (ข้ามพหุคูณของ 3 ตามคุณสมบัติระบบไฟฟ้า 3 เฟส)
K_HARMONICS = np.array([5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 35, 37, 41, 43,
                         47, 49, 53, 55, 59, 61, 65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97])

# นิยามค่าคงที่ของแต่ละ instance: (D, m, j, s)
_SOPWM_PARAMS = {
    45: dict(D=25, m=0.32, j=1,
             s=np.array([(-1) ** (l + 1) for l in range(1, 26)])),  # 3-level
    46: dict(D=25, m=0.32, j=2,
             s=np.array([1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1,
                          1, -1, 1, -1, -1, 1, -1, 1, 1, -1])),  # 5-level
    47: dict(D=25, m=0.36, j=3,
             s=np.array([1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1,
                          -1, -1, -1, 1, 1, -1, -1, 1, 1, 1])),  # 7-level
    48: dict(D=30, m=0.32, j=4,
             s=np.array([1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1,
                          1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1])),  # 9-level
    49: dict(D=30, m=0.3333, j=5,
             s=np.array([1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1,
                          -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1])),  # 11-level
    50: dict(D=30, m=0.32, j=6,
             s=np.array([1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1,
                          -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1])),  # 13-level
}

# best-known objective values ที่รายงานใน Kumar et al. (2020), Table 3 (สำหรับ validate)
BEST_KNOWN = {
    45: 3.8029250566e-02,
    46: 2.1215000000e-02,
    47: 1.5164538375e-02,
    48: 1.6787535766e-02,
    49: 9.3118741800e-03,
    50: 1.5096451396e-02,
}


def get_bounds(prob_k: int):
    """คืนค่า (lower_bound, upper_bound) เป็นอาร์เรย์ขนาด D, หน่วยองศา"""
    D = _SOPWM_PARAMS[prob_k]["D"]
    return np.zeros(D), np.full(D, 90.0)


def get_dimension(prob_k: int) -> int:
    return _SOPWM_PARAMS[prob_k]["D"]


def sopwm_evaluate(x: np.ndarray, prob_k: int):
    """
    ประเมินปัญหา SOPWM (RC45-RC50)

    Parameters
    ----------
    x : np.ndarray, shape (ps, D) หรือ (D,)
        มุมสวิตชิ่ง หน่วย "องศา" อยู่ในช่วง [0, 90]
        แถวคือ population, คอลัมน์คือมิติ (ต้องเรียง x[:,0] < x[:,1] < ... เพื่อ feasible)
    prob_k : int
        45..50 ระบุ instance (3,5,7,9,11,13-level ตามลำดับ)

    Returns
    -------
    f : np.ndarray, shape (ps,)      ค่า objective (ยิ่งน้อยยิ่งดี, >=0)
    g : np.ndarray, shape (ps, D-1)  inequality constraints, feasible เมื่อ g <= 0
    h : np.ndarray, shape (ps,)      equality constraint, feasible เมื่อ |h| <= epsilon (เช่น 1e-4)
    """
    if prob_k not in _SOPWM_PARAMS:
        raise ValueError(f"prob_k={prob_k} ไม่ใช่ปัญหา SOPWM (ต้องเป็น 45-50)")

    params = _SOPWM_PARAMS[prob_k]
    D_expected = params["D"]
    m = params["m"]
    j = params["j"]
    s = params["s"]

    x = np.atleast_2d(np.asarray(x, dtype=float))
    ps, D = x.shape
    if D != D_expected:
        raise ValueError(f"prob_k={prob_k} ต้องการ D={D_expected} แต่ได้ x ที่มี D={D}")

    x_rad = x * np.pi / 180.0  # แปลงองศา -> เรเดียนตอนคำนวณ (ตามโค้ดต้นฉบับ)

    # objective: f = sqrt( sum_k ( sum_l s(l)*cos(k*x_l) )^2 / k^4 ) / ( j * sqrt(sum_k 1/k^4) )
    su = np.zeros(ps)
    for k_val in K_HARMONICS:
        su2 = np.sum(s[np.newaxis, :] * np.cos(k_val * x_rad), axis=1)  # shape (ps,)
        su += su2 ** 2 / (k_val ** 4)
    denom = np.sqrt(np.sum(1.0 / (K_HARMONICS.astype(float) ** 4)))
    f = np.sqrt(su) / (j * denom)

    # inequality constraints: ordering, g_i = x_i - x_{i+1} + 1e-6 <= 0
    g = x[:, :-1] - x[:, 1:] + 1e-6

    # equality constraint: h = sum(s*cos(x_rad)) - j*m
    h = np.sum(s[np.newaxis, :] * np.cos(x_rad), axis=1) - j * m

    return f, g, h


def mean_constraint_violation(g: np.ndarray, h: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """
    Mean Constraint Violation ตามนิยามใน Kumar et al. (2020) Eq. (61)
    รับ g shape (ps, ng), h shape (ps,) หรือ (ps, nh) -> คืนค่า shape (ps,)
    """
    g = np.atleast_2d(g)
    h = np.atleast_2d(h.reshape(-1, 1) if h.ndim == 1 else h)
    viol_g = np.sum(np.maximum(g, 0), axis=1)
    viol_h = np.sum(np.maximum(np.abs(h) - epsilon, 0), axis=1)
    m_total = g.shape[1] + h.shape[1]
    return (viol_g + viol_h) / m_total


if __name__ == "__main__":
    # ทดสอบเบื้องต้น: random feasible-ish (เรียงมุมจากน้อยไปมาก) แล้วดูว่า f คำนวณได้ไม่ error
    rng = np.random.default_rng(0)
    for prob_k in range(45, 51):
        D = get_dimension(prob_k)
        lb, ub = get_bounds(prob_k)
        x = np.sort(rng.uniform(lb, ub, size=(3, D)), axis=1)
        f, g, h = sopwm_evaluate(x, prob_k)
        mcv = mean_constraint_violation(g, h)
        print(f"RC{prob_k}: D={D}, best_known={BEST_KNOWN[prob_k]:.6e}")
        print(f"  f (random)   = {f}")
        print(f"  mean CV      = {mcv}")
