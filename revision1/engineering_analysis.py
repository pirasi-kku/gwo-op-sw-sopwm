"""
engineering_analysis.py

T3 (revision1 task list) — converts the optimized switching angles into
actual multilevel-inverter engineering quantities: reconstructed output
waveform, analytic Fourier/harmonic spectrum, THD, fundamental-voltage
error (validates the equality constraint physically), and switching-count
implications.

Answers reviewer comments A1, C1, C7 (and doubles as a sanity check tied to
C4's request to explain the sign vector / DC-source count j / modulation
index m physically).

Input: revision1/results_revision/raw/gwo_op_sw_RC{45..50}.csv
       (produced by rerun_gwo_opsw_for_revision.py — has best_alpha_json
       per run). For each instance, the run with the lowest best_f among
       feasible runs is taken as the representative solution.

Physical model (confirmed against sopwm_objective.py + the original CEC2020
MATLAB source, cec20_func.m / Cal_par.m):
  - A quarter-wave & half-wave symmetric staircase voltage waveform, built
    from a cascaded H-bridge multilevel inverter with j DC sources
    (level count = 2j+1), switched D times per quarter cycle at angles
    x_1 < x_2 < ... < x_D (deg), each transition stepping the output by
    s_l in {+1,-1} (one H-bridge unit, V_dc = 1 p.u.).
  - Odd-harmonic Fourier coefficients (only odd n survive; even n are
    exactly zero by half-wave symmetry, which the plotted spectrum also
    confirms numerically):
        V_n = (4 / (n*pi)) * sum_l s_l * cos(n * x_l_rad)          [p.u.]
  - The equality constraint used during optimization (h = sum(s*cos(x)) -
    j*m = 0) is exactly the n=1 case above rescaled: at feasibility,
        V_1 = (4/pi) * j * m                                        [p.u.]
    so recomputing V_1 from the angles and comparing to this target is a
    fully independent check of feasibility (Reviewer A/C's concern about
    verifying results independently, not just trusting the optimizer's own
    constraint-violation bookkeeping).
  - THD is computed over the same non-triplen odd-harmonic family the
    paper's own objective targets (skips multiples of 3, standard for
    3-phase systems where triplens cancel line-to-line), extended out to
    n=301 for numerical completeness (the objective itself only weights
    harmonics up to n=97; this script's THD therefore is a strictly more
    complete accounting, not the same finite sum reused blindly).
"""

import ast
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sopwm_objective import _SOPWM_PARAMS, BEST_KNOWN  # noqa: reuse the same instance table

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(_SCRIPT_DIR, "results_revision_1e-5", "raw")
OUT_DIR = os.path.join(_SCRIPT_DIR, "results_revision", "engineering")
os.makedirs(OUT_DIR, exist_ok=True)

INSTANCES = [45, 46, 47, 48, 49, 50]
HIGHLIGHT = INSTANCES  # generate figures for all 6 instances (was 47,49,50 only)
N_HARM_MAX = 301  # upper bound for the THD summation (odd, non-triplen)


def non_triplen_odd_harmonics(n_max):
    return np.array([n for n in range(1, n_max + 1, 2) if n % 3 != 0])


def load_best_alpha(prob_k):
    """Load the representative (lowest best_f among feasible) run for this instance."""
    path = os.path.join(RAW_DIR, f"gwo_op_sw_RC{prob_k}.csv")
    df = pd.read_csv(path)
    feas = df[df["feasible"] == True]  # noqa: E712
    if len(feas) == 0:
        raise RuntimeError(f"RC{prob_k}: no feasible run found in {path}")
    best_row = feas.loc[feas["best_f"].idxmin()]
    alpha = json.loads(best_row["best_alpha_json"]) if isinstance(best_row["best_alpha_json"], str) \
        else ast.literal_eval(best_row["best_alpha_json"])
    return np.asarray(alpha, dtype=float), best_row


def build_waveform(angles_deg, s, n_points_per_deg=20):
    """Reconstruct one full electrical cycle (0-360 deg) of the staircase
    output level (in H-bridge units, V_dc=1 p.u.), using quarter-wave and
    half-wave symmetry."""
    angles_deg = np.sort(angles_deg)

    def level_first_quadrant(theta):
        # cumulative sum of s at all switching angles <= theta
        idx = np.searchsorted(angles_deg, theta, side="right")
        return np.cumsum(s)[idx - 1] if idx > 0 else 0.0

    theta_full = np.linspace(0, 360, int(360 * n_points_per_deg), endpoint=False)
    level = np.zeros_like(theta_full)
    for i, th in enumerate(theta_full):
        if th <= 90:
            level[i] = level_first_quadrant(th)
        elif th <= 180:
            level[i] = level_first_quadrant(180 - th)
        elif th <= 270:
            level[i] = -level_first_quadrant(th - 180)
        else:
            level[i] = -level_first_quadrant(360 - th)
    return theta_full, level


def fourier_coeffs(angles_deg, s, harmonics):
    """Analytic Fourier coefficients V_n (p.u.) for the odd harmonics in
    `harmonics`, via V_n = (4/(n*pi)) * sum_l s_l * cos(n * x_l_rad)."""
    x_rad = angles_deg * np.pi / 180.0
    out = {}
    for n in harmonics:
        out[n] = (4.0 / (n * np.pi)) * np.sum(s * np.cos(n * x_rad))
    return out


def numeric_fft_crosscheck(theta_full, level, n_target):
    """Numerical FFT of the sampled staircase waveform, as an independent
    cross-check of the analytic Fourier coefficient for one harmonic."""
    n_samples = len(theta_full)
    spec = np.fft.rfft(level) / n_samples * 2.0
    # bin index for harmonic n_target of the fundamental (1 full cycle sampled)
    return abs(spec[n_target])


def main():
    rows = []
    for prob_k in INSTANCES:
        params = _SOPWM_PARAMS[prob_k]
        j, m, s = params["j"], params["m"], params["s"]
        level_count = 2 * j + 1
        D = params["D"]

        angles, best_row = load_best_alpha(prob_k)
        theta_full, waveform = build_waveform(angles, s)

        harmonics = non_triplen_odd_harmonics(N_HARM_MAX)
        coeffs = fourier_coeffs(angles, s, harmonics)

        v1 = coeffs[1]
        v1_target = (4.0 / np.pi) * j * m
        v1_err = v1 - v1_target

        thd_num = math.sqrt(sum(v ** 2 for n, v in coeffs.items() if n != 1))
        thd_pct = 100.0 * thd_num / abs(v1)

        # cross-check V1 and V5 against a numeric FFT of the sampled waveform
        v1_fft = numeric_fft_crosscheck(theta_full, waveform, 1)
        v5_fft = numeric_fft_crosscheck(theta_full, waveform, 5)
        v1_fft_diff = abs(v1_fft - abs(v1))
        v5_fft_diff = abs(v5_fft - abs(coeffs[5]))

        transitions_per_cycle = 4 * D
        transitions_per_cell_per_cycle = transitions_per_cycle / j

        rows.append(dict(
            prob_k=prob_k, level=level_count, D=D, j=j, m=m,
            best_f=best_row["best_f"], best_cv=best_row["best_cv"],
            V1_pu=v1, V1_target_pu=v1_target, V1_error_pu=v1_err,
            THD_percent=thd_pct,
            V1_fft_crosscheck_diff=v1_fft_diff, V5_fft_crosscheck_diff=v5_fft_diff,
            transitions_per_cycle=transitions_per_cycle,
            transitions_per_cell_per_cycle=transitions_per_cell_per_cycle,
        ))

        if prob_k in HIGHLIGHT:
            # --- waveform figure ---
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(theta_full, waveform, drawstyle="steps-post", linewidth=1.2)
            ax.set_xlabel("Electrical angle (deg)")
            ax.set_ylabel("Output level (V_dc units)")
            ax.set_title(f"RC{prob_k}: {level_count}-level staircase output waveform "
                         f"(D={D} switching angles/quarter, j={j} DC sources)")
            ax.set_xlim(0, 360)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"waveform_RC{prob_k}.png"), dpi=150)
            plt.close(fig)

            # --- harmonic spectrum figure (normalized to V1 = 100%) ---
            plot_harm = non_triplen_odd_harmonics(99)
            mags_pct = [100.0 * abs(coeffs.get(n, fourier_coeffs(angles, s, [n])[n])) / abs(v1)
                        for n in plot_harm]
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.bar(plot_harm, mags_pct, width=1.5)
            ax.set_xlabel("Harmonic order n")
            ax.set_ylabel("|V_n| / V_1 (%)")
            ax.set_title(f"RC{prob_k}: harmonic spectrum ({level_count}-level, THD={thd_pct:.3f}%)")
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"spectrum_RC{prob_k}.png"), dpi=150)
            plt.close(fig)

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "engineering_summary.csv")
    summary.to_csv(out_csv, index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Figures for RC{HIGHLIGHT} saved in {OUT_DIR}/")


if __name__ == "__main__":
    main()
