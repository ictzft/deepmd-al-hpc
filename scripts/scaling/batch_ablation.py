#!/usr/bin/env python
"""Batch-size ablation: vary batch_size for single-model ethanol training and
measure GPU SM utilization. Validates the launch-bound hypothesis from the
Nsight analysis (larger batch -> larger kernels -> launch overhead占比 drops ->
SM rises). Runs INSIDE the deepmd-5090 container.

Usage: bash scripts/docker/run_in_5090.sh 0 -- python scripts/scaling/batch_ablation.py
"""
import json
import subprocess
import time
from pathlib import Path

TEMPLATE = "/data/zft/deepmd-al-hpc/configs/deepmd/pt_rmd17_ethanol.json"
EXP = Path("/data/zft/deepmd-al-hpc/experiments/scaling/batch_ablation")
EXP.mkdir(parents=True, exist_ok=True)

STEPS = 100
BATCHES = [8, 64, 256]

base = json.load(open(TEMPLATE))
results = []

for bs in BATCHES:
    cfg = json.loads(json.dumps(base))  # deep copy
    cfg["training"]["training_data"]["batch_size"] = bs
    cfg["training"]["validation_data"]["batch_size"] = bs
    cfg["training"]["numb_steps"] = STEPS
    cfg["training"]["save_freq"] = STEPS
    cfg_path = EXP / f"bs{bs}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    work = EXP / f"bs{bs}"
    work.mkdir(exist_ok=True)
    dmon_log = work / "dmon.log"

    print(f"\n=== batch_size={bs} ({STEPS} steps) ===")
    dmon = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-d", "1"],
        stdout=open(dmon_log, "w"), stderr=subprocess.STDOUT)

    env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES="0")
    t0 = time.time()
    rc = subprocess.run(
        ["dp", "-b", "pytorch", "train", str(cfg_path)],
        cwd=str(work), env=env,
        stdout=open(work / "train.log", "w"),
        stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0

    dmon.terminate()
    dmon.wait()

    # parse GPU0 SM%
    sm_vals = []
    for line in open(dmon_log):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "0" and parts[1].isdigit():
            sm_vals.append(int(parts[1]))
    sm_avg = sum(sm_vals) / len(sm_vals) if sm_vals else 0.0
    sm_max = max(sm_vals) if sm_vals else 0
    results.append({"batch": bs, "wall_s": round(wall, 2),
                    "sm_avg": round(sm_avg, 1), "sm_max": sm_max,
                    "rc": rc, "n_samples": len(sm_vals)})
    print(f"  wall={wall:.1f}s  SM avg={sm_avg:.1f}%  max={sm_max:.0f}%  rc={rc}")

print("\n=== batch ablation summary ===")
print(f"{'batch':>6} {'wall_s':>8} {'SM_avg':>8} {'SM_max':>8}")
for r in results:
    print(f"{r['batch']:>6} {r['wall_s']:>8.1f} {r['sm_avg']:>7.1f}% {r['sm_max']:>7.0f}%")
(EXP / "summary.json").write_text(json.dumps(results, indent=2))
print(f"\nsummary -> {EXP / 'summary.json'}")
