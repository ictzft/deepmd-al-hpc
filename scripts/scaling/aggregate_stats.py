#!/usr/bin/env python
"""Aggregate repeated stat_* runs into mean ± std for the DP-GEN comparison.

Reads wall (summary.json) and GPU0 SM avg (gpu_dmon.log) for each config across
repeats, prints a mean±std table. Stdlib only.
"""
import json
import statistics
import sys
from pathlib import Path

repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
root = Path("experiments/scaling")


def parse_sm(dmon_path, gpu=0):
    vals = []
    for line in open(dmon_path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) >= 2 and p[0] == str(gpu) and p[1].isdigit():
            vals.append(int(p[1]))
    return statistics.mean(vals) if vals else 0.0


configs = [
    ("DP-GEN-style (wave)", "stat_wave_n4_r{}_g4/summary.json",
     "stat_wave_n4_r{}_g4/gpu_dmon.log"),
    ("Ours MPS b8",         "stat_mps_b8_r{}/summary.json",
     "stat_mps_b8_r{}/gpu_dmon.log"),
    ("Ours MPS b256",       "stat_mps_b256_r{}/summary.json",
     "stat_mps_b256_r{}/gpu_dmon.log"),
]

print(f"\n=== DP-GEN comparison: mean ± std over {repeats} runs (ethanol N=4) ===")
print(f"{'config':<24} {'wall (s)':>20} {'SM avg (%)':>18} {'GPU·s':>12}")
print("-" * 78)
results = []
for name, spat, dpat in configs:
    walls, sms = [], []
    for r in range(1, repeats + 1):
        sj = root / spat.format(r)
        dm = root / dpat.format(r)
        if sj.exists():
            walls.append(json.load(open(sj))["total_wall_s"])
        if dm.exists():
            sms.append(parse_sm(dm))
    if not walls:
        print(f"{name:<24}  (no data)")
        continue
    wm = statistics.mean(walls)
    ws = statistics.stdev(walls) if len(walls) > 1 else 0.0
    smm = statistics.mean(sms) if sms else 0.0
    sms_ = statistics.stdev(sms) if len(sms) > 1 else 0.0
    # GPU-seconds: wave uses 4 GPUs, MPS uses 1
    ngpu = 4 if "wave" in name else 1
    gpu_s = wm * ngpu
    print(f"{name:<24} {wm:>8.1f} ± {ws:>5.1f}     {smm:>5.1f} ± {sms_:>4.1f}    {gpu_s:>8.1f}")
    results.append({"config": name, "wall_mean": wm, "wall_std": ws,
                    "sm_mean": smm, "sm_std": sms_, "gpu_seconds": gpu_s, "ngpu": ngpu})

if results:
    (root / "stat_summary.json").write_text(json.dumps(results, indent=2))
    # resource-efficiency ratio vs wave
    base = next((r["gpu_seconds"] for r in results if "wave" in r["config"]), None)
    if base:
        print("\n=== resource efficiency (GPU·s) vs DP-GEN-style ===")
        for r in results:
            ratio = base / r["gpu_seconds"]
            print(f"  {r['config']:<24} {ratio:.2f}×")
