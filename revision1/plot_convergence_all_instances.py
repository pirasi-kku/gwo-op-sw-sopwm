"""
plot_convergence_all_instances.py -- Figure 3 of the manuscript: feasible
best-so-far convergence of GWO-OP-SW on RC45-RC50 (median, interquartile band
and best of the 30 runs).

Reads the per-iteration traces stored by rerun_gwo_opsw_for_revision.py in
results_revision_1e-5/raw/gwo_op_sw_convergence_RC<k>.npy (30 x 4000 arrays,
NaN until a run first becomes feasible).

Output: results_revision/engineering/convergence_all_instances.png

HOW TO RUN
  python revision1/plot_convergence_all_instances.py [output.png]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(_HERE, "results_revision_1e-5", "raw")
OUT = (sys.argv[1] if len(sys.argv) > 1 else
       os.path.join(_HERE, "results_revision", "engineering",
                    "convergence_all_instances.png"))
LEVEL = {45: 3, 46: 5, 47: 7, 48: 9, 49: 11, 50: 13}


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, k in zip(axes.flat, range(45, 51)):
        tr = np.load(os.path.join(RAW_DIR, "gwo_op_sw_convergence_RC%d.npy" % k))
        it = np.arange(1, tr.shape[1] + 1)
        q1, med, q3 = np.nanpercentile(tr, [25, 50, 75], axis=0)
        best = np.nanmin(tr, axis=0)   # best of the 30 runs at each iteration
        ax.fill_between(it, q1, q3, alpha=0.25, label="IQR (%d runs)" % tr.shape[0])
        ax.plot(it, med, lw=1.5, label="median")
        ax.plot(it, best, "--", lw=1.0, label="best run")
        ax.set_yscale("log")
        ax.set_xlim(0, tr.shape[1])
        ax.set_title("RC%d (%d-level)" % (k, LEVEL[k]), fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("feasible best-so-far $f$")
    for ax in axes[1, :]:
        ax.set_xlabel("iteration")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("Written to %s" % OUT)


if __name__ == "__main__":
    main()
