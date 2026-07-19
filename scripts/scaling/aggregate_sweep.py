#!/usr/bin/env python
"""Aggregate the MPS (N x batch) sweep into a red利曲线 + energy table.

Reads each sweep_nN_bsBS/ : summary.json (wall) + gpu.csv (util, power).
Outputs SM%, avg power (W), energy (J) = power x wall, and MPS speedup
(wall vs single-model baseline) to spot the saturation point. Stdlib only.
"""
import csv
import json
import statistics
from pathlib import Path

root = Path("experiments/scaling")
Ns = [2, 4, 8]
BSs = [8, 64, 256]

rows = []
for N in Ns:
    for BS in BSs:
        d = root / f"sweep_n{N}_bs{BS}"
        sj = d / "summary.json"
        gc = d / "gpu.csv"
        if not sj.exists():
            continue
        wall = json.load(open(sj))["total_wall_s"]
        sms, pws = [], []
        if gc.exists():
            for row in csv.reader(open(gc)):
                if len(row) < 4:
                    continue
                idx = row[1].strip()
                if idx != "0":
                    continue
                try:
                    u = row[2].strip().split()[0]   # "0 %" -> "0"
                    p = row[3].strip().split()[0]   # "20.43 W" -> "20.43"
                    sms.append(float(u))
                    pws.append(float(p))
                except (ValueError, IndexError):
                    continue
        sm = statistics.mean(sms) if sms else 0.0
        pw = statistics.mean(pws) if pws else 0.0
        energy = pw * wall
        rows.append({"N": N, "batch": BS, "wall": wall, "sm": sm,
                     "power": pw, "energy": energy})

print("\n=== MPS sweep: SM / power / energy vs (N, batch), 1 GPU shared ===")
print(f"{'N':>3} {'batch':>6} {'wall_s':>8} {'SM%':>7} {'power_W':>9} {'energy_J':>10}")
print("-" * 50)
for r in rows:
    print(f"{r['N']:>3} {r['batch']:>6} {r['wall']:>8.1f} {r['sm']:>6.1f}% "
          f"{r['power']:>8.1f} {r['energy']:>10.0f}")

# saturation analysis: where does SM plateau / wall spike?
print("\n=== saturation readout ===")
for BS in BSs:
    sub = [r for r in rows if r["batch"] == BS]
    if len(sub) < 2:
        continue
    sm_max = max(r["sm"] for r in sub)
    n_sat = next((r["N"] for r in sub if r["sm"] > 0.9 * sm_max), "?")
    print(f"  batch={BS}: SM peaks at N={n_sat} ({sm_max:.1f}%)")

(root / "sweep_summary.json").write_text(json.dumps(rows, indent=2))
print(f"\nsummary -> {root/'sweep_summary.json'}")
