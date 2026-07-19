#!/usr/bin/env python
"""Collect strong-scaling summary.json files into one CSV and print a table.

Reads experiments/scaling/<tag>_g*/summary.json produced by concurrent_runner
(via run_strong_scaling.sh), computes:
  - speedup   = wall(smallest G) / wall(G)
  - parallel efficiency = speedup * (base G) / G
and writes <root>/<tag>.csv. Stdlib only (no numpy needed on host).
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", help="experiment tag, e.g. strong_n8_ethanol")
    ap.add_argument("--root", default="experiments/scaling",
                    help="root dir holding <tag>_g* subdirs")
    args = ap.parse_args()

    root = Path(args.root)
    records = []
    for d in sorted(root.glob(f"{args.tag}_g*")):
        sp = d / "summary.json"
        if not sp.exists():
            print(f"[skip] {d} (no summary.json)", file=sys.stderr)
            continue
        records.append(json.loads(sp.read_text()))

    if not records:
        sys.exit(f"no summaries found under {root}/{args.tag}_g*")

    records.sort(key=lambda s: s["n_gpus"])
    base_wall = records[0]["total_wall_s"]
    base_g = records[0]["n_gpus"]

    out_csv = root / f"{args.tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_models", "n_gpus", "n_waves", "total_wall_s",
                    "throughput_models_per_s", "mean_model_wall_s",
                    f"speedup_vs_g{base_g}", "parallel_eff"])
        for s in records:
            g = s["n_gpus"]
            wall = s["total_wall_s"]
            speedup = base_wall / wall
            eff = speedup * base_g / g
            w.writerow([s["n_models"], g, s["n_waves"], wall,
                        s["throughput_models_per_s"], s["mean_model_wall_s"],
                        f"{speedup:.3f}", f"{eff:.3f}"])

    print(f"=== {args.tag} (baseline G={base_g}, wall={base_wall:.2f}s) ===")
    hdr = f"{'G':>3} {'waves':>5} {'wall_s':>9} {'models/s':>10} {'speedup':>8} {'eff':>6}"
    print(hdr)
    print("-" * len(hdr))
    for s in records:
        g = s["n_gpus"]
        wall = s["total_wall_s"]
        speedup = base_wall / wall
        eff = speedup * base_g / g
        print(f"{g:>3} {s['n_waves']:>5} {wall:>9.2f} "
              f"{s['throughput_models_per_s']:>10.4f} {speedup:>8.3f} {eff:>6.3f}")
    print(f"\nCSV -> {out_csv}")


if __name__ == "__main__":
    main()
