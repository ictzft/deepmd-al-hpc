#!/usr/bin/env python3
"""Extract and summarize training wall times from all train.log files."""
import re, statistics
from pathlib import Path

PROJ = Path("/data/zft/deepmd-al-hpc")
baselines = PROJ / "experiments" / "baselines"

all_data = []
for log_file in sorted(baselines.rglob("train.log")):
    text = log_file.read_text()
    wall = re.findall(r"wall time: ([\d.]+) s", text)
    if not wall:
        continue
    all_data.append({
        "path": str(log_file.relative_to(PROJ)),
        "wall_s": float(wall[-1]),
    })

if not all_data:
    print("No train.log files found.")
    exit(0)

walls = [d["wall_s"] for d in all_data]

print(f"Training wall time summary")
print(f"=========================")
print(f"Total models: {len(all_data)}")
print(f"Mean:  {statistics.mean(walls):.1f}s")
print(f"Std:   {statistics.stdev(walls):.1f}s")
print(f"Min:   {min(walls):.1f}s")
print(f"Max:   {max(walls):.1f}s")
print(f"Median:{statistics.median(walls):.1f}s")
print()

# 2xV100 parallel: 4 models, 2 waves -> total ~2 * mean
print(f"Estimated 2xV100 parallel (4 models): ~{2 * statistics.mean(walls):.0f}s per round")
print(f"Estimated end-to-end per round: ~{2 * statistics.mean(walls) + 10:.0f}s (train + predict + I/O)")

# Write summary CSV
csv_path = PROJ / "experiments" / "profiling" / "training_wall_time_summary.csv"
with csv_path.open("w") as f:
    f.write("path,wall_time_s\n")
    for d in sorted(all_data, key=lambda x: x["wall_s"]):
        f.write(f"{d['path']},{d['wall_s']:.3f}\n")
print(f"\nWrote: {csv_path}")
