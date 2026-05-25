from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Any

METRICS_FILES = {
    "seed0": Path("experiments/baselines/random_seed0_round001_metrics_summary.csv"),
    "seed1": Path("experiments/baselines/random_seed1_round001_metrics_summary.csv"),
    "seed2": Path("experiments/baselines/random_seed2_round001_metrics_summary.csv"),
}

PREDICTION_FILES = {
    "seed0": Path("experiments/baselines/random_seed0_round001_prediction_summary.csv"),
    "seed1": Path("experiments/baselines/random_seed1_round001_prediction_summary.csv"),
    "seed2": Path("experiments/baselines/random_seed2_round001_prediction_summary.csv"),
}

OUT_DIR = Path("experiments/baselines")
SUMMARY_CSV = OUT_DIR / "random_round001_baseline_summary.csv"
SUMMARY_MD = OUT_DIR / "random_round001_baseline_summary.md"

ROUND_ID = "001"
TRAIN_FRAMES = 210
CANDIDATE_FRAMES = 40


def read_metrics(path: Path) -> dict[str, float]:
    """Read a metrics CSV and return the summary row (mean/std)."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row.get("model", row.get("Model", "")).strip().lower()
            if model == "mean":
                return {
                    "energy_rmse_mean": float(row["energy_rmse"]),
                    "force_rmse_mean": float(row["force_rmse"]),
                }
            if model == "std":
                return {
                    "energy_rmse_mean": float("nan"),
                    "force_rmse_mean": float("nan"),
                }
    raise ValueError(f"Could not find mean/std row in {path}")


def read_metrics_full(path: Path) -> dict[str, float]:
    """Read a metrics CSV and extract mean AND std rows."""
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
    """Read a prediction summary CSV, handling both formats."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows in {path}")

    # seed0 format: has "run" column, may contain multiple runs
    if "run" in rows[0]:
        for row in rows:
            run_name = row.get("run", "")
            if f"random_{seed_label}_round001" in run_name:
                return {
                    "candidate_force_dev_max_mean": float(row["force_dev_max_mean"]),
                    "candidate_force_dev_max_max": float(row["force_dev_max_max"]),
                    "candidate_force_dev_max_min": float(row["force_dev_max_min"]),
                    "candidate_energy_dev_mean": float(row.get("energy_dev_mean", "nan")),
                }
        raise ValueError(f"Could not find random_{seed_label}_round001 row in {path}")

    # seed1/seed2 format: single row, no "run" column
    row = rows[0]
    return {
        "candidate_force_dev_max_mean": float(row["force_dev_max_mean"]),
        "candidate_force_dev_max_max": float(row["force_dev_max_max"]),
        "candidate_force_dev_max_min": float(row["force_dev_max_min"]),
        "candidate_energy_dev_mean": float(row.get("energy_dev_mean", "nan")),
    }


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6e}"
    return str(x)


def main() -> None:
    rows: list[dict[str, Any]] = []

    for seed_label in ["seed0", "seed1", "seed2"]:
        metrics = read_metrics_full(METRICS_FILES[seed_label])
        prediction = read_prediction(PREDICTION_FILES[seed_label], seed_label)

        rows.append(
            {
                "seed": seed_label,
                "round_id": ROUND_ID,
                "train_frames": TRAIN_FRAMES,
                "candidate_frames": CANDIDATE_FRAMES,
                "energy_rmse_mean": metrics.get("energy_rmse_mean"),
                "energy_rmse_std": metrics.get("energy_rmse_std"),
                "force_rmse_mean": metrics.get("force_rmse_mean"),
                "force_rmse_std": metrics.get("force_rmse_std"),
                **prediction,
            }
        )

    # Compute cross-seed mean and std
    energy_rmse_vals = [r["energy_rmse_mean"] for r in rows if r["energy_rmse_mean"] is not None]
    force_rmse_vals = [r["force_rmse_mean"] for r in rows if r["force_rmse_mean"] is not None]
    candidate_dev_vals = [r["candidate_force_dev_max_mean"] for r in rows]

    cross_seed = {
        "seed": "mean",
        "round_id": ROUND_ID,
        "train_frames": TRAIN_FRAMES,
        "candidate_frames": CANDIDATE_FRAMES,
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
        "seed",
        "round_id",
        "train_frames",
        "candidate_frames",
        "energy_rmse_mean",
        "energy_rmse_std",
        "force_rmse_mean",
        "force_rmse_std",
        "candidate_force_dev_max_mean",
        "candidate_force_dev_max_max",
        "candidate_force_dev_max_min",
        "candidate_energy_dev_mean",
    ]

    # Write CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(v) for k, v in row.items()})

    # Write Markdown
    md_lines = [
        "# Random Round001 Baseline Summary",
        "",
        "## Multi-seed retraining metrics (Round 001)",
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

    md_lines.extend(
        [
            "",
            "## Candidate pool uncertainty after Round 001 retraining",
            "",
            "| Seed | force_dev_max mean | force_dev_max max | force_dev_max min | energy_dev mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for row in rows:
        if row["seed"] == "mean":
            md_lines.append(
                f"| {row['seed']} | {fmt(row['candidate_force_dev_max_mean'])} | "
                f"— | — | — |"
            )
        else:
            md_lines.append(
                f"| {row['seed']} | {fmt(row['candidate_force_dev_max_mean'])} | "
                f"{fmt(row['candidate_force_dev_max_max'])} | "
                f"{fmt(row['candidate_force_dev_max_min'])} | "
                f"{fmt(row['candidate_energy_dev_mean'])} |"
            )

    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Current random Round001 baseline has been extended from single-seed (seed0) to multi-seed (seed0 / seed1 / seed2).",
            "- The cross-seed mean and std are reported for energy_rmse, force_rmse, and candidate force_dev_max.",
            "- This is still a toy H2 workflow validation and does not represent realistic first-principles material systems.",
            "- The next step is to complete random Round002 / Round003 retraining for a full multi-round learning curve.",
            "",
        ]
    )

    SUMMARY_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print("========== Random Round001 baseline summary generated ==========")
    print(f"csv: {SUMMARY_CSV}")
    print(f"md:  {SUMMARY_MD}")
    print()
    for row in rows:
        print(
            f"{row['seed']:6s} | "
            f"E_RMSE={fmt(row['energy_rmse_mean']):>12s} | "
            f"F_RMSE={fmt(row['force_rmse_mean']):>12s} | "
            f"dev_max_mean={fmt(row['candidate_force_dev_max_mean']):>12s}"
        )


if __name__ == "__main__":
    main()
