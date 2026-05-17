from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path("/data/zft")


ROUNDS = [
    {
        "round": 0,
        "train_dirs": [
            "data/toy_h2/train",
            "data/toy_h2/initial_train",
            "data/toy_h2/round_000_train",
        ],
        "candidate_dirs": [
            "data/toy_h2/candidate",
            "data/toy_h2/candidate_pool",
            "data/toy_h2/round_000_candidate",
        ],
        "model_dir": "experiments/exp_004_committee_models",
        "pred_dir": "experiments/exp_005_committee_prediction",
    },
    {
        "round": 1,
        "train_dirs": ["data/toy_h2/round_001_train"],
        "candidate_dirs": ["data/toy_h2/round_001_candidate"],
        "model_dir": "experiments/exp_007_round001_committee_models",
        "pred_dir": "experiments/exp_008_round001_committee_prediction",
    },
    {
        "round": 2,
        "train_dirs": ["data/toy_h2/round_002_train"],
        "candidate_dirs": ["data/toy_h2/round_002_candidate"],
        "model_dir": "experiments/exp_009_round002_committee_models",
        "pred_dir": "experiments/exp_010_round002_committee_prediction",
    },
    {
        "round": 3,
        "train_dirs": ["data/toy_h2/round_003_train"],
        "candidate_dirs": ["data/toy_h2/round_003_candidate"],
        "model_dir": "experiments/exp_011_round003_committee_models",
        "pred_dir": "experiments/exp_012_round003_committee_prediction",
    },
]


def first_existing(paths: list[str]) -> Path | None:
    for p in paths:
        full = ROOT / p
        if full.exists():
            return full
    return None


def count_frames(data_dir: Path | None) -> int | None:
    if data_dir is None or not data_dir.exists():
        return None

    total = 0
    found = False
    for set_dir in sorted(data_dir.glob("set.*")):
        coord = set_dir / "coord.npy"
        if coord.exists():
            arr = np.load(coord)
            total += int(arr.shape[0])
            found = True

    return total if found else None


def parse_test_log(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None

    text = path.read_text(errors="replace")

    def last_float(pattern: str) -> float | None:
        vals = re.findall(pattern, text)
        if not vals:
            return None
        return float(vals[-1])

    energy_rmse = last_float(r"Energy RMSE\s*:\s*([0-9.eE+-]+)\s*eV")
    energy_rmse_natoms = last_float(r"Energy RMSE/Natoms\s*:\s*([0-9.eE+-]+)\s*eV")
    force_rmse = last_float(r"Force\s+RMSE\s*:\s*([0-9.eE+-]+)\s*eV/")

    if energy_rmse is None and force_rmse is None:
        return None

    return {
        "energy_rmse": energy_rmse,
        "energy_rmse_natoms": energy_rmse_natoms,
        "force_rmse": force_rmse,
    }


def parse_models(model_dir: Path) -> list[dict[str, object]]:
    rows = []

    for model_path in sorted(model_dir.glob("model_*")):
        if not model_path.is_dir():
            continue

        test_log = model_path / "test.log"
        metrics = parse_test_log(test_log)
        if metrics is None:
            continue

        rows.append(
            {
                "model": model_path.name,
                "energy_rmse": metrics["energy_rmse"],
                "energy_rmse_natoms": metrics["energy_rmse_natoms"],
                "force_rmse": metrics["force_rmse"],
            }
        )

    return rows


def parse_prediction(pred_dir: Path) -> dict[str, object]:
    selected_json = pred_dir / "selected_topk.json"
    if not selected_json.exists():
        return {
            "n_models": None,
            "n_frames": None,
            "n_atoms": None,
            "top_k": None,
            "force_dev_max_max": None,
            "force_dev_max_mean": None,
            "force_dev_max_min": None,
            "energy_dev_mean": None,
        }

    data = json.loads(selected_json.read_text(errors="replace"))
    selected = data.get("selected_frames", [])

    force_vals = [float(x["force_dev_max"]) for x in selected if "force_dev_max" in x]
    energy_vals = [float(x["energy_dev"]) for x in selected if "energy_dev" in x]

    return {
        "n_models": data.get("n_models"),
        "n_frames": data.get("n_frames"),
        "n_atoms": data.get("n_atoms"),
        "top_k": data.get("top_k", len(selected)),
        "force_dev_max_max": max(force_vals) if force_vals else None,
        "force_dev_max_mean": mean(force_vals) if force_vals else None,
        "force_dev_max_min": min(force_vals) if force_vals else None,
        "energy_dev_mean": mean(energy_vals) if energy_vals else None,
    }


def parse_elapsed_seconds(log_dir: Path, kind: str) -> int | None:
    patterns = []
    if kind == "train":
        patterns = [
            "run*train*.log",
            "run*.log",
        ]
    elif kind == "prediction":
        patterns = [
            "run*prediction*.log",
            "run*.log",
        ]

    for pat in patterns:
        for log in sorted(log_dir.glob(pat)):
            text = log.read_text(errors="replace")
            m = re.findall(r"elapsed_seconds=([0-9]+)", text)
            if m:
                return int(m[-1])

    return None


def fmt(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6e}"
    return str(x)


def main() -> None:
    out_dir = ROOT / "experiments"
    model_rows = []
    round_rows = []

    for cfg in ROUNDS:
        r = cfg["round"]

        train_dir = first_existing(cfg["train_dirs"])
        candidate_dir = first_existing(cfg["candidate_dirs"])

        model_dir = ROOT / cfg["model_dir"]
        pred_dir = ROOT / cfg["pred_dir"]

        models = parse_models(model_dir)
        pred = parse_prediction(pred_dir)

        energy_vals = [x["energy_rmse"] for x in models if x["energy_rmse"] is not None]
        energy_natom_vals = [x["energy_rmse_natoms"] for x in models if x["energy_rmse_natoms"] is not None]
        force_vals = [x["force_rmse"] for x in models if x["force_rmse"] is not None]

        for m in models:
            row = {"round": r, **m}
            model_rows.append(row)

        candidate_count = count_frames(candidate_dir)
        if candidate_count is None:
            candidate_count = pred["n_frames"]

        round_rows.append(
            {
                "round": r,
                "train_frames": count_frames(train_dir),
                "candidate_frames": candidate_count,
                "n_models": len(models),
                "energy_rmse_mean": mean(energy_vals) if energy_vals else None,
                "energy_rmse_min": min(energy_vals) if energy_vals else None,
                "energy_rmse_max": max(energy_vals) if energy_vals else None,
                "energy_rmse_natoms_mean": mean(energy_natom_vals) if energy_natom_vals else None,
                "force_rmse_mean": mean(force_vals) if force_vals else None,
                "force_rmse_min": min(force_vals) if force_vals else None,
                "force_rmse_max": max(force_vals) if force_vals else None,
                "prediction_n_frames": pred["n_frames"],
                "top_k": pred["top_k"],
                "force_dev_max_max": pred["force_dev_max_max"],
                "force_dev_max_mean": pred["force_dev_max_mean"],
                "force_dev_max_min": pred["force_dev_max_min"],
                "energy_dev_mean": pred["energy_dev_mean"],
                "train_elapsed_s": parse_elapsed_seconds(model_dir, "train"),
                "prediction_elapsed_s": parse_elapsed_seconds(pred_dir, "prediction"),
            }
        )

    round_csv = out_dir / "al_rounds_summary.csv"
    model_csv = out_dir / "al_model_level_summary.csv"
    md_path = out_dir / "al_rounds_summary.md"

    round_fields = list(round_rows[0].keys())
    with round_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=round_fields)
        writer.writeheader()
        writer.writerows(round_rows)

    model_fields = ["round", "model", "energy_rmse", "energy_rmse_natoms", "force_rmse"]
    with model_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=model_fields)
        writer.writeheader()
        writer.writerows(model_rows)

    with md_path.open("w") as f:
        f.write("# Active Learning Round Summary\n\n")
        f.write("## Round-Level Summary\n\n")
        f.write("| " + " | ".join(round_fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(round_fields)) + " |\n")
        for row in round_rows:
            f.write("| " + " | ".join(fmt(row[k]) for k in round_fields) + " |\n")

        f.write("\n## Model-Level Summary\n\n")
        f.write("| " + " | ".join(model_fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(model_fields)) + " |\n")
        for row in model_rows:
            f.write("| " + " | ".join(fmt(row[k]) for k in model_fields) + " |\n")

    print(f"wrote {round_csv}")
    print(f"wrote {model_csv}")
    print(f"wrote {md_path}")

    print("\nRound-level summary:")
    for row in round_rows:
        print(row)


if __name__ == "__main__":
    main()
