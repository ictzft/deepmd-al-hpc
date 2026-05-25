#!/usr/bin/env python3
"""Summarize uncertainty vs random baseline across all available rounds.

Reads uncertainty branch results from experiments/al_rounds_summary.csv and
random baseline results from experiments/baselines/random_round001_baseline_summary.csv
(and future round002/003 files when available).

Outputs:
  experiments/baselines/random_vs_uncertainty_summary.csv
  experiments/baselines/random_vs_uncertainty_summary.md

Field naming convention:
  - energy_rmse_mean/std → validation set RMSE (committee mean ± cross-model std)
  - force_rmse_mean/std  → validation set Force RMSE
  - force_dev_max_mean    → for uncertainty branch: top-K selected frames' force_dev_max mean
                             for random branch: remaining candidate-pool force_dev_max mean
  - force_dev_max_std     → cross-seed std of force_dev_max_mean (random mean row only)
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

UNCERTAINTY_CSV = PROJECT_ROOT / "experiments" / "al_rounds_summary.csv"

RANDOM_INPUTS: list[Path] = []  # auto-discovered below


def _discover_random_inputs() -> list[Path]:
    """Auto-discover available random_round*_baseline_summary.csv files."""
    baselines_dir = PROJECT_ROOT / "experiments" / "baselines"
    candidates = sorted(baselines_dir.glob("random_round*_baseline_summary.csv"),
                        key=lambda p: p.name)
    if not candidates:
        print("Warning: No random_round*_baseline_summary.csv files found.")
    return candidates


RANDOM_INPUTS = _discover_random_inputs()

# Per-round prediction summaries for candidate-pool uncertainty comparison
# Auto-discovered from available random_seed*_round*_prediction_summary.csv files
RANDOM_PREDICTION_INPUTS: dict[str, list[Path]] = {}


def _discover_prediction_inputs() -> dict[str, list[Path]]:
    """Auto-discover prediction summary files grouped by round."""
    baselines_dir = PROJECT_ROOT / "experiments" / "baselines"
    by_round: dict[str, list[Path]] = {}
    for p in sorted(baselines_dir.glob("random_seed*_round*_prediction_summary.csv")):
        # Extract round number from filename like random_seed0_round001_prediction_summary.csv
        import re
        m = re.search(r"round(\d{3})", p.name)
        if m:
            by_round.setdefault(m.group(1), []).append(p)
    return by_round


RANDOM_PREDICTION_INPUTS = _discover_prediction_inputs()

OUT_DIR = PROJECT_ROOT / "experiments" / "baselines"
OUT_CSV = OUT_DIR / "random_vs_uncertainty_summary.csv"
OUT_MD = OUT_DIR / "random_vs_uncertainty_summary.md"


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6e}"
    return str(x)


def read_uncertainty_rows() -> list[dict[str, Any]]:
    """Read uncertainty branch results."""
    rows = []
    with UNCERTAINTY_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "round": int(row["round"]),
                "method": "uncertainty",
                "seed": "—",
                "train_frames": int(row["train_frames"]),
                "candidate_frames": int(row["candidate_frames"]),
                "energy_rmse_mean": float(row["energy_rmse_mean"]),
                "force_rmse_mean": float(row["force_rmse_mean"]),
                "force_dev_max_mean": float(row["force_dev_max_mean"]),
            })
    return rows


def read_random_rows() -> list[dict[str, Any]]:
    """Read random baseline results from available summary files."""
    rows = []
    for path in RANDOM_INPUTS:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["seed"] == "mean":
                    continue  # skip aggregate row, we'll recompute
                rows.append({
                    "round": int(row["round_id"]),
                    "method": "random",
                    "seed": row["seed"],
                    "train_frames": int(row["train_frames"]),
                    "candidate_frames": int(row["candidate_frames"]),
                    "energy_rmse_mean": float(row["energy_rmse_mean"]),
                    "energy_rmse_std": float(row["energy_rmse_std"]),
                    "force_rmse_mean": float(row["force_rmse_mean"]),
                    "force_rmse_std": float(row["force_rmse_std"]),
                    "force_dev_max_mean": float(row["candidate_force_dev_max_mean"]),
                })
    return rows


def aggregate_random(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute mean±std across seed0/seed1/seed2 per round."""
    by_round: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_round.setdefault(row["round"], []).append(row)

    agg = []
    for rnd in sorted(by_round):
        group = by_round[rnd]
        energy_vals = [r["energy_rmse_mean"] for r in group]
        force_vals = [r["force_rmse_mean"] for r in group]
        dev_vals = [r["force_dev_max_mean"] for r in group]

        agg.append({
            "round": rnd,
            "method": "random",
            "seed": "mean",
            "train_frames": group[0]["train_frames"],
            "candidate_frames": group[0]["candidate_frames"],
            "energy_rmse_mean": mean(energy_vals) if energy_vals else None,
            "energy_rmse_std": stdev(energy_vals) if len(energy_vals) >= 2 else None,
            "force_rmse_mean": mean(force_vals) if force_vals else None,
            "force_rmse_std": stdev(force_vals) if len(force_vals) >= 2 else None,
            "force_dev_max_mean": mean(dev_vals) if dev_vals else None,
            "force_dev_max_std": stdev(dev_vals) if len(dev_vals) >= 2 else None,
        })
    return agg


def build_comparison_rows() -> list[dict[str, Any]]:
    """Build per-method per-round comparison rows."""
    uncertainty = read_uncertainty_rows()
    random_all = read_random_rows()
    random_agg = aggregate_random(random_all)

    comparison = []
    for u_row in uncertainty:
        comparison.append({
            "round": u_row["round"],
            "method": "uncertainty",
            "seed": "—",
            "train_frames": u_row["train_frames"],
            "candidate_frames": u_row["candidate_frames"],
            "energy_rmse_mean": u_row["energy_rmse_mean"],
            "energy_rmse_std": None,
            "force_rmse_mean": u_row["force_rmse_mean"],
            "force_rmse_std": None,
            "force_dev_max_mean": u_row["force_dev_max_mean"],
            "force_dev_max_std": None,
        })

    for row in random_all:
        comparison.append(row)

    for row in random_agg:
        comparison.append(row)

    return comparison


def write_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "round", "method", "seed",
        "train_frames", "candidate_frames",
        "energy_rmse_mean", "energy_rmse_std",
        "force_rmse_mean", "force_rmse_std",
        "force_dev_max_mean", "force_dev_max_std",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(v) for k, v in row.items()})


def write_md(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Uncertainty vs Random Baseline Comparison",
        "",
        "## Per-round summary (all methods)",
        "",
        "| Round | Method | Seed | Train | Candidate | Energy RMSE | Force RMSE | force_dev_max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['round']} | {row['method']} | {row['seed']} | "
            f"{row['train_frames']} | {row['candidate_frames']} | "
            f"{fmt(row['energy_rmse_mean'])} | {fmt(row['force_rmse_mean'])} | "
            f"{fmt(row['force_dev_max_mean'])} |"
        )

    # Separate section for aggregate comparison (uncertainty vs random mean)
    lines.extend([
        "",
        "## Round-level comparison (uncertainty vs random mean ± std)",
        "",
        "| Round | Method | Energy RMSE mean | Energy RMSE std | Force RMSE mean | Force RMSE std | force_dev_max mean | force_dev_max std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    # Collect uncertainty and random_mean rows
    uncertainty_rows = [r for r in rows if r["method"] == "uncertainty"]
    random_mean_rows = [r for r in rows if r["method"] == "random" and r["seed"] == "mean"]

    for rnd in sorted(set(r["round"] for r in rows)):
        u = next((r for r in uncertainty_rows if r["round"] == rnd), None)
        rr = next((r for r in random_mean_rows if r["round"] == rnd), None)

        if u:
            lines.append(
                f"| {rnd} | uncertainty | {fmt(u['energy_rmse_mean'])} | — | "
                f"{fmt(u['force_rmse_mean'])} | — | {fmt(u['force_dev_max_mean'])} | — |"
            )
        if rr:
            lines.append(
                f"| {rnd} | random | {fmt(rr['energy_rmse_mean'])} | {fmt(rr['energy_rmse_std'])} | "
                f"{fmt(rr['force_rmse_mean'])} | {fmt(rr['force_rmse_std'])} | "
                f"{fmt(rr['force_dev_max_mean'])} | {fmt(rr['force_dev_max_std'])} |"
            )

    lines.extend([
        "",
        "## Notes",
        "",
        "- Uncertainty branch uses `force_dev_max` top-K selection across Round 0–3.",
        "- Random baseline currently has Round 001 multi-seed data (seed0/seed1/seed2).",
        "- Random Round 002/003 data is pending — scripts and configs are prepared for reproducibility.",
        "- This is a toy H2 workflow validation. Real DFT/AIMD datasets and H100 scaling are not yet included.",
        "",
        "**Field meaning note:** `force_dev_max_mean` has different semantics per branch:",
        "- For **uncertainty**: mean `force_dev_max` of the top-K **selected** frames from committee prediction.",
        "- For **random**: mean `force_dev_max` of the **remaining candidate pool** after retraining.",
        "- These are not directly comparable; for a fair remaining candidate-pool comparison,",
        "  see the per-seed `random_seed*_round001_prediction_summary.csv` files (which include",
        "  the uncertainty_round001 remaining candidate-pool row).",
        "",
    ])

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_comparison_rows()
    write_csv(rows)
    write_md(rows)

    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    print(f"\nTotal rows: {len(rows)}")
    for row in rows:
        print(
            f"  round={row['round']} method={row['method']:12s} seed={row['seed']:6s} "
            f"E_RMSE={fmt(row['energy_rmse_mean']):>12s} "
            f"F_RMSE={fmt(row['force_rmse_mean']):>12s} "
            f"dev={fmt(row['force_dev_max_mean']):>12s}"
        )


if __name__ == "__main__":
    main()
