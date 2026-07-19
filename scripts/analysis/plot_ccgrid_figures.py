#!/usr/bin/env python
"""Plot CCGrid 2027 paper figures from the scaling / optimization experiments.
Runs INSIDE the deepmd-5090 container (matplotlib available there).

Produces into experiments/figures/ccgrid_2027/ (both .svg and .png):
  fig2_motivation     - baseline GPU underutilization (SM ~4%)
  fig3_scaling        - strong scaling: wall & parallel efficiency vs #GPUs
  fig5_launch_bound   - root cause: launch-bound (cudaLaunchKernel 84.6%)
  fig6_optimization   - batch x MPS mitigation matrix (SM 4.4% -> 38%)

Scaling is read from CSV; the rest are inlined from measured results
(each value annotated with its source file).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/data/zft/deepmd-al-hpc")
OUT = BASE / "experiments/figures/ccgrid_2027"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")


# ============ Fig.3: strong scaling (read CSV) ============
# source: experiments/scaling/strong_n8_ethanol.csv (ethanol, N=8, 200 steps)
rows = list(csv.DictReader(open(BASE / "experiments/scaling/strong_n8_ethanol.csv")))
rows.sort(key=lambda r: int(r["n_gpus"]))
G = [int(r["n_gpus"]) for r in rows]
wall = [float(r["total_wall_s"]) for r in rows]
eff = [float(r["parallel_eff"]) for r in rows]

fig, ax1 = plt.subplots(figsize=(5.2, 3.6))
ax1.plot(G, wall, "o-", color="C0", lw=2, ms=8, label="wall time")
ax1.plot(G, [wall[0] / g for g in G], "--", color="0.55", lw=1.5, label="ideal linear")
ax1.set_xlabel("Number of GPUs")
ax1.set_ylabel("Wall time (s)", color="C0")
ax1.set_xticks(G)
ax1.tick_params(axis="y", labelcolor="C0")
ax2 = ax1.twinx()
ax2.plot(G, [e * 100 for e in eff], "s--", color="C3", lw=1.5, ms=7, label="efficiency")
ax2.set_ylabel("Parallel efficiency (%)", color="C3")
ax2.set_ylim(0, 108)
ax2.tick_params(axis="y", labelcolor="C3")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center",
           bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False)
ax1.set_title("Strong scaling (ethanol, N=8 committee, 200 steps)")
fig.tight_layout()
save(fig, "fig3_scaling")
plt.close(fig)


# ============ Fig.2: motivation - baseline GPU underutilization ============
# source: experiments/scaling/wave_n4_g4/gpu_dmon.log (per-GPU SM avg ~4.1%)
sm_busy = 4.1
fig, ax = plt.subplots(figsize=(4.2, 3.8))
vals = [sm_busy, 100 - sm_busy]
bars = ax.bar(["SM busy", "idle"], vals, color=["C3", "0.8"])
ax.set_ylabel("GPU time (%)")
ax.set_ylim(0, 112)
ax.set_title("Baseline GPU underutilization\n(ethanol, batch=8, 1 model/GPU)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
fig.tight_layout()
save(fig, "fig2_motivation")
plt.close(fig)


# ============ Fig.5: launch-bound root cause (Nsight) ============
# source: experiments/nsight_prof/ethanol_30.nsys-rep -> nsys stats cuda_api_sum
labels = ["cudaLaunchKernel", "cudaMemcpyAsync", "cuLaunchKernel", "other"]
api_pct = [84.6, 6.7, 3.3, 5.4]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
ax1.bar(range(len(labels)), api_pct, color=["C3", "C1", "C2", "0.7"])
ax1.set_xticks(range(len(labels)))
ax1.set_xticklabels(labels, rotation=18, ha="right")
ax1.set_ylabel("Share of CPU CUDA-API time (%)")
ax1.set_title("CPU side: launch-dominated")
ax1.set_ylim(0, 95)
for i, v in enumerate(api_pct):
    ax1.text(i, v + 1.5, f"{v}%", ha="center", fontsize=8)
ax2.bar(["GPU kernel\nexecution", "cudaLaunchKernel\n(CPU)"], [0.28, 2.04], color=["C0", "C3"])
ax2.set_ylabel("Time (s)")
ax2.set_title("GPU works 0.28 s; CPU spends 2.04 s launching")
ax2.set_ylim(0, 2.3)
fig.tight_layout()
save(fig, "fig5_launch_bound")
plt.close(fig)


# ============ Fig.6: batch x MPS optimization matrix ============
# sources:
#   baseline / batch=256 single-model : experiments/scaling/batch_ablation/summary.json
#   MPS batch=8 / batch=256           : experiments/scaling/mps_n4_g1{,_bs256}/
configs = ["baseline\n(batch=8,\n1 model)", "batch=256\n(1 model)",
           "MPS\n(batch=8,\n4 models)", "MPS + batch=256\n(4 models)"]
sm = [4.4, 25.6, 14.2, 38.0]
colors = ["0.6", "C0", "C1", "C2"]
fig, ax = plt.subplots(figsize=(6.2, 4))
bars = ax.bar(range(len(configs)), sm, color=colors)
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(configs)
ax.set_ylabel("GPU SM utilization avg (%)")
ax.set_ylim(0, 45)
ax.set_title("Launch-bound mitigation: batch size x MPS (4 models, 1 GPU)")
for b, v in zip(bars, sm):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v}%", ha="center")
fig.tight_layout()
save(fig, "fig6_optimization")
plt.close(fig)


print(f"Figures saved to {OUT}")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}")
