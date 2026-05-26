"""Generate round-level and model-level summary for rMD17 ethanol active learning."""

import csv
import json
import re
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path("/data/guyida/deepmd-al-hpc")

ROUNDS = [
    {
        "round": 0,
        "train_dir": "data/rmd17/ethanol/train",
        "candidate_dir": "data/rmd17/ethanol/candidate",
        "model_dir": "experiments/rmd17_ethanol_round000_committee_models",
        "pred_dir": "experiments/rmd17_ethanol_round000_committee_prediction",
        "selected_json": "selected_uncertainty_1000.json",
    },
    {
        "round": 1,
        "train_dir": "data/rmd17/ethanol/train_round001_uncertainty",
        "candidate_dir": "data/rmd17/ethanol/candidate_round001",
        "model_dir": "experiments/rmd17_ethanol_round001_committee_models",
        "pred_dir": "experiments/rmd17_ethanol_round001_committee_prediction",
        "selected_json": "selected_uncertainty_1000.json",
    },
    {
        "round": 2,
        "train_dir": "data/rmd17/ethanol/train_round002_uncertainty",
        "candidate_dir": "data/rmd17/ethanol/candidate_round002",
        "model_dir": "experiments/rmd17_ethanol_round002_committee_models",
        "pred_dir": "experiments/rmd17_ethanol_round002_committee_prediction",
        "selected_json": "selected_uncertainty_1000.json",
    },
    {
        "round": 3,
        "train_dir": "data/rmd17/ethanol/train_round003_uncertainty",
        "candidate_dir": "data/rmd17/ethanol/candidate_round003",
        "model_dir": "experiments/rmd17_ethanol_round003_committee_models",
        "pred_dir": "experiments/rmd17_ethanol_round003_committee_prediction",
        "selected_json": "selected_uncertainty_1000.json",
    },
]


def count_frames(data_dir: Path) -> int | None:
    if not data_dir.exists():
        return None
    total = 0
    found = False
    for set_dir in sorted(data_dir.glob("set.*")):
        coord = set_dir / "coord.npy"
        if coord.exists():
            total += int(np.load(coord).shape[0])
            found = True
    return total if found else None


def parse_test_log(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")

    def last_float(pattern: str) -> float | None:
        vals = re.findall(pattern, text)
        return float(vals[-1]) if vals else None

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


def parse_prediction(selected_json_path: Path) -> dict:
    if not selected_json_path.exists():
        return {}

    data = json.loads(selected_json_path.read_text(errors="replace"))
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


def fmt(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6e}"
    return str(x)


def main() -> None:
    out_dir = ROOT / "experiments/rmd17_ethanol_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rows = []
    round_rows = []

    for cfg in ROUNDS:
        r = cfg["round"]
        train_dir = ROOT / cfg["train_dir"]
        candidate_dir = ROOT / cfg["candidate_dir"]
        model_dir = ROOT / cfg["model_dir"]
        pred_dir = ROOT / cfg["pred_dir"]
        selected_json = pred_dir / cfg["selected_json"]

        # Parse models
        models = []
        for model_path in sorted(model_dir.glob("model_*")):
            if not model_path.is_dir():
                continue
            metrics = parse_test_log(model_path / "test.log")
            if metrics is None:
                continue
            models.append({"model": model_path.name, **metrics})

        pred = parse_prediction(selected_json)

        energy_vals = [x["energy_rmse"] for x in models if x["energy_rmse"] is not None]
        force_vals = [x["force_rmse"] for x in models if x["force_rmse"] is not None]
        energy_natom_vals = [x["energy_rmse_natoms"] for x in models if x["energy_rmse_natoms"] is not None]

        for m in models:
            model_rows.append({"round": r, **m})

        candidate_count = count_frames(candidate_dir)
        if candidate_count is None:
            candidate_count = pred.get("n_frames")

        round_rows.append({
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
            "prediction_n_frames": pred.get("n_frames"),
            "top_k": pred.get("top_k"),
            "force_dev_max_max": pred.get("force_dev_max_max"),
            "force_dev_max_mean": pred.get("force_dev_max_mean"),
            "force_dev_max_min": pred.get("force_dev_max_min"),
            "energy_dev_mean": pred.get("energy_dev_mean"),
        })

    round_fields = [
        "round", "train_frames", "candidate_frames", "n_models",
        "energy_rmse_mean", "energy_rmse_min", "energy_rmse_max",
        "energy_rmse_natoms_mean",
        "force_rmse_mean", "force_rmse_min", "force_rmse_max",
        "prediction_n_frames", "top_k",
        "force_dev_max_max", "force_dev_max_mean", "force_dev_max_min",
        "energy_dev_mean",
    ]
    model_fields = ["round", "model", "energy_rmse", "energy_rmse_natoms", "force_rmse"]

    round_csv = out_dir / "al_rounds_summary.csv"
    model_csv = out_dir / "al_model_level_summary.csv"
    md_path = out_dir / "al_rounds_summary.md"

    with round_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=round_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(round_rows)

    with model_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=model_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(model_rows)

    with md_path.open("w") as f:
        f.write("# rMD17 Ethanol Active Learning Round Summary\n\n")
        f.write(f"Dataset: rMD17 ethanol (C₂H₅OH, 9 atoms). 2×Tesla V100-16GB.\n\n")
        f.write("## Round-Level Summary\n\n")
        f.write("| " + " | ".join(round_fields) + " |\n")
        f.write("| " + " | ".join(["---:"] * len(round_fields)) + " |\n")
        for row in round_rows:
            f.write("| " + " | ".join(fmt(row.get(k)) for k in round_fields) + " |\n")

        f.write("\n## Model-Level Summary\n\n")
        f.write("| " + " | ".join(model_fields) + " |\n")
        f.write("| " + " | ".join(["---:"] * len(model_fields)) + " |\n")
        for row in model_rows:
            f.write("| " + " | ".join(fmt(row.get(k)) for k in model_fields) + " |\n")

    print(f"wrote {round_csv}")
    print(f"wrote {model_csv}")
    print(f"wrote {md_path}")

    print("\nRound-level summary:")
    for row in round_rows:
        print(row)


if __name__ == "__main__":
    main()
