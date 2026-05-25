#!/usr/bin/env python3
"""Summarize random baseline retraining metrics across seed0/seed1/seed2 for one round.

Generalized from summarize_random_round001_baselines.py to support any round ID.

Usage:
  python scripts/analysis/summarize_random_round_baselines.py --round-id 1
  python scripts/analysis/summarize_random_round_baselines.py --round-id 2
  python scripts/analysis/summarize_random_round_baselines.py --round-id 3

Input expectations:
  experiments/baselines/random_{seed}_round{XXX}_metrics_summary.csv
  experiments/baselines/random_{seed}_round{XXX}_prediction_summary.csv

Outputs:
  experiments/baselines/random_round{XXX}_baseline_summary.csv
  experiments/baselines/random_round{XXX}_baseline_summary.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "experiments" / "baselines"

# train = 200 + round_id * 10, candidate = 50 - round_id * 10
# round 1: 210 / 40, round 2: 220 / 30, round 3: 230 / 20


def _train_frames(round_id: int) -> int:
    return 200 + round_id * 10


def _candidate_frames(round_id: int) -> int:
    return 50 - round_id * 10


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6e}"
    return str(x)


def read_metrics_full(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model", row.get("Model", "")).strip()
            if model.lower() == "mean":
                result["energy_rmse_mean"] = float(row["energy_rmse"])
                result["force_rmse_mean"] = float(row["force_rmse"])
            if model.lower() == "std":
                result["energy_rmse_std"] = float(row["energy_rmse"])
                result["force_rmse_std"] = float(row["force_rmse"])
    return result


def read_prediction(path: Path, seed_label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows in {path}")

    round_tag = f"random_{seed_label}_round"

    # seed0 format: has "run" column, may contain multiple rows
    if "run" in rows[0]:
        for row in rows:
            run_name = row.get("run", "")
            if round_tag in run_name:
                return {
                    "candidate_force_dev_max_mean": float(row["force_dev_max_mean"]),
                    "candidate_force_dev_max_max": float(row["force_dev_max_max"]),
                    "candidate_force_dev_max_min": float(row["force_dev_max_min"]),
                    "candidate_energy_dev_mean": float(row.get("energy_dev_mean", "nan")),
                }
        raise ValueError(f"Could not find {round_tag} row in {path}")

    # seed1/seed2 format: single row, no "run" column
    row = rows[0]
    return {
        "candidate_force_dev_max_mean": float(row["force_dev_max_mean"]),
        "candidate_force_dev_max_max": float(row["force_dev_max_max"]),
        "candidate_force_dev_max_min": float(row["force_dev_max_min"]),
        "candidate_energy_dev_mean": float(row.get("energy_dev_mean", "nan")),
    }


def run(round_id: int) -> None:
    round_str = f"{round_id:03d}"
    train_frames = _train_frames(round_id)
    candidate_frames = _candidate_frames(round_id)

    metrics_files = {
        seed: PROJECT_ROOT / "experiments" / "baselines"
        / f"random_{seed}_round{round_str}_metrics_summary.csv"
        for seed in ["seed0", "seed1", "seed2"]
    }

    prediction_files = {
        seed: PROJECT_ROOT / "experiments" / "baselines"
        / f"random_{seed}_round{round_str}_prediction_summary.csv"
        for seed in ["seed0", "seed1", "seed2"]
    }

    # Validate inputs exist
    missing = [p for p in list(metrics_files.values()) + list(prediction_files.values()) if not p.exists()]
    if missing:
        print("ERROR: Missing input files:")
        for p in missing:
            print(f"  {p}")
        print("\nRun training, testing, and prediction for this round first.")
        raise SystemExit(1)

    rows: list[dict[str, Any]] = []
    for seed_label in ["seed0", "seed1", "seed2"]:
        metrics = read_metrics_full(metrics_files[seed_label])
        prediction = read_prediction(prediction_files[seed_label], seed_label)

        rows.append({
            "seed": seed_label,
            "round_id": round_str,
            "train_frames": train_frames,
            "candidate_frames": candidate_frames,
            "energy_rmse_mean": metrics.get("energy_rmse_mean"),
            "energy_rmse_std": metrics.get("energy_rmse_std"),
            "force_rmse_mean": metrics.get("force_rmse_mean"),
            "force_rmse_std": metrics.get("force_rmse_std"),
            **prediction,
        })

    # Cross-seed aggregate
    energy_rmse_vals = [r["energy_rmse_mean"] for r in rows if r["energy_rmse_mean"] is not None]
    force_rmse_vals = [r["force_rmse_mean"] for r in rows if r["force_rmse_mean"] is not None]
    candidate_dev_vals = [r["candidate_force_dev_max_mean"] for r in rows]

    cross_seed = {
        "seed": "mean",
        "round_id": round_str,
        "train_frames": train_frames,
        "candidate_frames": candidate_frames,
        "energy_rmse_mean": mean(energy_rmse_vals) if energy_rmse_vals else None,
        "energy_rmse_std": stdev(energy_rmse_vals) if len(energy_rmse_vals) >= 2 else None,
        "force_rmse_mean": mean(force_rmse_vals) if force_rmse_vals else None,
        "force_rmse_std": stdev(force_rmse_vals) if len(force_rmse_vals) >= 2 else None,
        "candidate_force_dev_max_mean": mean(candidate_dev_vals) if candidate_dev_vals else None,
        "candidate_force_dev_max_max": None,
        "candidate_force_dev_max_min": None,
        "candidate_energy_dev_mean": None,
    }
    rows.append(cross_seed)

    fieldnames = [
        "seed", "round_id", "train_frames", "candidate_frames",
        "energy_rmse_mean", "energy_rmse_std",
        "force_rmse_mean", "force_rmse_std",
        "candidate_force_dev_max_mean", "candidate_force_dev_max_max",
        "candidate_force_dev_max_min", "candidate_energy_dev_mean",
    ]

    summary_csv = OUT_DIR / f"random_round{round_str}_baseline_summary.csv"
    summary_md = OUT_DIR / f"random_round{round_str}_baseline_summary.md"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(v) for k, v in row.items()})

    # Markdown
    md_lines = [
        f"# Random Round{round_str} Baseline Summary",
        "",
        f"## Multi-seed retraining metrics (Round {round_str})",
        "",
        "| Seed | Train frames | Candidate frames | Energy RMSE mean / eV | Energy RMSE std / eV | Force RMSE mean / eV/Å | Force RMSE std / eV/Å |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['seed']} | {row['train_frames']} | {row['candidate_frames']} | "
            f"{fmt(row['energy_rmse_mean'])} | {fmt(row['energy_rmse_std'])} | "
            f"{fmt(row['force_rmse_mean'])} | {fmt(row['force_rmse_std'])} |"
        )

    md_lines.extend([
        "",
        f"## Candidate pool uncertainty after Round {round_str} retraining",
        "",
        "| Seed | force_dev_max mean | force_dev_max max | force_dev_max min | energy_dev mean |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in rows:
        if row["seed"] == "mean":
            md_lines.append(
                f"| {row['seed']} | {fmt(row['candidate_force_dev_max_mean'])} | — | — | — |"
            )
        else:
            md_lines.append(
                f"| {row['seed']} | {fmt(row['candidate_force_dev_max_mean'])} | "
                f"{fmt(row['candidate_force_dev_max_max'])} | "
                f"{fmt(row['candidate_force_dev_max_min'])} | "
                f"{fmt(row['candidate_energy_dev_mean'])} |"
            )

    md_lines.extend([
        "",
        "## Notes",
        "",
        f"- Random Round{round_str} multi-seed baseline (seed0/seed1/seed2).",
        "- Cross-seed mean and std are reported for energy_rmse, force_rmse, and candidate force_dev_max.",
        "- This is still a toy H2 workflow validation and does not represent realistic first-principles material systems.",
        "",
    ])
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"========== Random Round{round_str} baseline summary generated ==========")
    print(f"csv: {summary_csv}")
    print(f"md:  {summary_md}")
    print()
    for row in rows:
        print(
            f"{row['seed']:6s} | "
            f"E_RMSE={fmt(row['energy_rmse_mean']):>12s} | "
            f"F_RMSE={fmt(row['force_rmse_mean']):>12s} | "
            f"dev_max_mean={fmt(row['candidate_force_dev_max_mean']):>12s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize random baseline multi-seed metrics for one round."
    )
    parser.add_argument("--round-id", type=int, required=True, choices=[1, 2, 3],
                        help="Round ID (1, 2, or 3).")
    args = parser.parse_args()
    run(args.round_id)


if __name__ == "__main__":
    main()
