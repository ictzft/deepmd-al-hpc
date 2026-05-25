#!/usr/bin/env python3
"""Generate metrics_summary.csv and prediction_summary.csv for one round and seed."""
import re
import csv
import statistics
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path("/data/zft/deepmd-al-hpc")


def generate_metrics(seed: str, round_id: int) -> None:
    round_str = f"{round_id:03d}"
    models_dir = PROJECT_ROOT / "experiments" / "baselines" / f"random_{seed}_round{round_str}_committee_models"

    rows = []
    for m_id in ["model_000", "model_001", "model_002", "model_003"]:
        test_log = models_dir / m_id / "test.log"
        if not test_log.exists():
            print(f"WARNING: {test_log} not found")
            continue
        text = test_log.read_text()
        e_rmse_all = re.findall(r"Energy RMSE\s+:\s+(\S+) eV", text)
        f_rmse_all = re.findall(r"Force  RMSE\s+:\s+(\S+) eV", text)
        e_mae_all = re.findall(r"Energy MAE\s+:\s+(\S+) eV", text)
        f_mae_all = re.findall(r"Force  MAE\s+:\s+(\S+) eV", text)
        if e_rmse_all and f_rmse_all:
            # Take the last (weighted average) values
            rows.append({
                "model": m_id,
                "energy_mae": float(e_mae_all[-1]),
                "energy_rmse": float(e_rmse_all[-1]),
                "force_mae": float(f_mae_all[-1]),
                "force_rmse": float(f_rmse_all[-1]),
            })

    if not rows:
        print(f"ERROR: No metrics extracted for {seed} round {round_str}")
        return

    mean_row = {
        "model": "mean",
        "energy_mae": statistics.mean(r["energy_mae"] for r in rows),
        "energy_rmse": statistics.mean(r["energy_rmse"] for r in rows),
        "force_mae": statistics.mean(r["force_mae"] for r in rows),
        "force_rmse": statistics.mean(r["force_rmse"] for r in rows),
    }
    std_row = {
        "model": "std",
        "energy_mae": statistics.stdev(r["energy_mae"] for r in rows),
        "energy_rmse": statistics.stdev(r["energy_rmse"] for r in rows),
        "force_mae": statistics.stdev(r["force_mae"] for r in rows),
        "force_rmse": statistics.stdev(r["force_rmse"] for r in rows),
    }
    rows.append(mean_row)
    rows.append(std_row)

    # Write metrics_summary.csv
    baselines_dir = PROJECT_ROOT / "experiments" / "baselines"
    csv_path = baselines_dir / f"random_{seed}_round{round_str}_metrics_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "energy_mae", "energy_rmse", "force_mae", "force_rmse"])
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.8e}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"Wrote {csv_path}")

    # Write metrics_summary.md
    md_path = baselines_dir / f"random_{seed}_round{round_str}_metrics_summary.md"
    lines = [
        f"# Random {seed} Round{round_str} Committee Test Metrics",
        "",
        "| Model | Energy RMSE / eV | Force RMSE / eV/Å |",
        "|---|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['energy_rmse']:.6e} | {r['force_rmse']:.6e} |")
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")


def generate_prediction(seed: str, round_id: int) -> None:
    round_str = f"{round_id:03d}"
    pred_dir = PROJECT_ROOT / "experiments" / "baselines" / f"random_{seed}_round{round_str}_committee_prediction"
    npz_path = pred_dir / "committee_predictions.npz"

    if not npz_path.exists():
        print(f"WARNING: {npz_path} not found")
        return

    data = np.load(npz_path, allow_pickle=True)
    force_dev_max = data["force_dev_max"]
    energy_dev = data["energy_dev"]

    baselines_dir = PROJECT_ROOT / "experiments" / "baselines"
    pred_csv = baselines_dir / f"random_{seed}_round{round_str}_prediction_summary.csv"
    with pred_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_frames", "force_dev_max_mean", "force_dev_max_max", "force_dev_max_min", "energy_dev_mean"])
        w.writeheader()
        w.writerow({
            "n_frames": len(force_dev_max),
            "force_dev_max_mean": f"{float(np.mean(force_dev_max)):.6e}",
            "force_dev_max_max": f"{float(np.max(force_dev_max)):.6e}",
            "force_dev_max_min": f"{float(np.min(force_dev_max)):.6e}",
            "energy_dev_mean": f"{float(np.mean(energy_dev)):.6e}",
        })
    print(f"Wrote {pred_csv}")

    pred_md = baselines_dir / f"random_{seed}_round{round_str}_prediction_summary.md"
    pred_md.write_text(
        f"# Random {seed} Round{round_str} Candidate-pool Prediction\n\n"
        f"n_frames: {len(force_dev_max)}\n\n"
        f"force_dev_max_mean: {np.mean(force_dev_max):.6e}\n"
        f"force_dev_max_max: {np.max(force_dev_max):.6e}\n"
        f"force_dev_max_min: {np.min(force_dev_max):.6e}\n"
        f"energy_dev_mean: {np.mean(energy_dev):.6e}\n"
    )
    print(f"Wrote {pred_md}")


if __name__ == "__main__":
    round_id = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    for seed in ["seed0", "seed1", "seed2"]:
        print(f"\n===== {seed} Round{round_id:03d} =====")
        generate_metrics(seed, round_id)
        generate_prediction(seed, round_id)
    print("\n===== Done =====")
