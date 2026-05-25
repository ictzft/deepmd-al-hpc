#!/usr/bin/env python3
"""Apply a selection strategy to an existing committee prediction NPZ.

Supports: uncertainty (top-K), random, uncertainty-diversity, trust-level.

Usage:
  python scripts/selection/select_by_strategy.py \
    --predictions experiments/baselines/random_seed0_round001_committee_prediction/committee_predictions.npz \
    --strategy uncertainty-diversity \
    --top-k 10 --top-m 30 \
    --output experiments/baselines/diversity_round001/selected_topk.json

The output JSON is compatible with the existing selected_topk.json format,
so downstream scripts (merge_selected_frames, prepare_random_baseline_round)
can consume it without changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from al.selector import select_by_strategy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Select frames from committee predictions.")
    parser.add_argument("--predictions", required=True, help="Path to committee_predictions.npz")
    parser.add_argument("--strategy", required=True,
                        choices=["uncertainty", "random", "uncertainty-diversity", "trust-level"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="Only for random strategy.")
    parser.add_argument("--top-m", type=int, default=30, help="For uncertainty-diversity: top-M pool size.")
    parser.add_argument("--descriptor", default="pairwise-distance", help="For uncertainty-diversity.")
    parser.add_argument("--low-pct", type=float, default=50.0, help="For trust-level: accurate/candidate boundary.")
    parser.add_argument("--high-pct", type=float, default=90.0, help="For trust-level: candidate/failed boundary.")
    parser.add_argument("--candidate-ratio", type=float, default=0.8, help="For trust-level: fraction from candidate region.")
    parser.add_argument("--output", required=True, help="Path to output selected_topk.json")
    parser.add_argument("--summary-csv", default=None, help="Optional CSV summary file.")
    args = parser.parse_args()

    # Load predictions
    data = np.load(args.predictions, allow_pickle=True)
    scores = data["force_dev_max"]
    coords = data.get("coord", None)
    n_frames = len(scores)

    print(f"Predictions: {args.predictions}")
    print(f"n_frames:    {n_frames}")
    print(f"Strategy:    {args.strategy}")
    print(f"top_k:       {args.top_k}")

    # Apply strategy
    result = select_by_strategy(
        scores=scores,
        k=args.top_k,
        strategy=args.strategy,
        seed=args.seed,
        coords=coords,
        top_m=args.top_m,
        descriptor=args.descriptor,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
        candidate_ratio=args.candidate_ratio,
    )

    # Parse result
    if isinstance(result, dict):
        selected_indices = result["selected_indices"]
        extra = {k: v for k, v in result.items() if k != "selected_indices"}
    else:
        selected_indices = result
        extra = {}

    # Build selected_frames records
    force_dev_max = scores[selected_indices]
    force_dev_mean = data["force_dev_mean"][selected_indices] if "force_dev_mean" in data else np.zeros(len(selected_indices))
    energy_dev = data["energy_dev"][selected_indices] if "energy_dev" in data else np.zeros(len(selected_indices))

    selected_frames = []
    for rank, idx in enumerate(selected_indices):
        selected_frames.append({
            "rank": rank + 1,
            "frame_index": int(idx),
            "force_dev_max": float(force_dev_max[rank]),
            "force_dev_mean": float(force_dev_mean[rank]),
            "energy_dev": float(energy_dev[rank]),
        })

    # Build output
    output: dict = {
        "n_frames": int(data.get("n_frames", n_frames)),
        "top_k": args.top_k,
        "selection_strategy": args.strategy,
        "selection_policy": args.strategy,
        "selected_indices": [int(x) for x in selected_indices],
        "selected_frames": selected_frames,
    }
    if args.strategy == "random":
        output["seed"] = args.seed
    if args.strategy == "uncertainty-diversity":
        output["top_m"] = args.top_m
        output["descriptor"] = args.descriptor
    if args.strategy == "trust-level":
        output.update({k: v for k, v in extra.items() if k != "selected_force_dev_max"})

    # Write JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {out_path}")

    # Print summary
    print(f"\n===== {args.strategy} selection summary =====")
    print(f"Selected {len(selected_indices)} frames:")
    for sf in selected_frames:
        print(f"  rank={sf['rank']:02d}  frame={sf['frame_index']:3d}  "
              f"force_dev_max={sf['force_dev_max']:.6e}  "
              f"energy_dev={sf['energy_dev']:.6e}")

    if extra:
        print("\nExtra info:")
        for k, v in extra.items():
            if isinstance(v, (list, np.ndarray)):
                print(f"  {k}: {v}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.6e}")
            else:
                print(f"  {k}: {v}")

    # Optional CSV
    if args.summary_csv:
        import csv
        csv_path = Path(args.summary_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "strategy", "top_k", "n_frames",
                "selected_force_dev_max_mean", "selected_force_dev_max_std",
                "selected_force_dev_max_min", "selected_force_dev_max_max",
            ])
            w.writeheader()
            w.writerow({
                "strategy": args.strategy,
                "top_k": args.top_k,
                "n_frames": n_frames,
                "selected_force_dev_max_mean": f"{float(np.mean(force_dev_max)):.6e}",
                "selected_force_dev_max_std": f"{float(np.std(force_dev_max)):.6e}",
                "selected_force_dev_max_min": f"{float(np.min(force_dev_max)):.6e}",
                "selected_force_dev_max_max": f"{float(np.max(force_dev_max)):.6e}",
            })
        print(f"\nWrote CSV: {csv_path}")


if __name__ == "__main__":
    main()
