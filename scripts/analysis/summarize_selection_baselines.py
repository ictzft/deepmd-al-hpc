from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROUND_DIRS = [
    ("round_000", Path("experiments/exp_005_committee_prediction")),
    ("round_001", Path("experiments/exp_008_round001_committee_prediction")),
]

OUT_DIR = Path("experiments/baselines")
RUNS_CSV = OUT_DIR / "selection_baseline_runs.csv"
SUMMARY_CSV = OUT_DIR / "selection_baseline_summary.csv"
SUMMARY_MD = OUT_DIR / "selection_baseline_summary.md"


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def safe_stdev(values: list[float]) -> float | None:
    return stdev(values) if len(values) >= 2 else None


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.8g}"
    return str(x)


def load_selected_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_one(round_name: str, path: Path) -> dict[str, Any]:
    data = load_selected_json(path)
    selected = data.get("selected_frames", [])

    if not selected:
        raise ValueError(f"No selected_frames found in {path}")

    force_dev_max = [float(x["force_dev_max"]) for x in selected]
    force_dev_mean = [float(x["force_dev_mean"]) for x in selected]
    energy_dev = [float(x["energy_dev"]) for x in selected]
    frame_indices = [int(x["frame_index"]) for x in selected]

    strategy = data.get("selection_strategy")
    if strategy is None:
        if path.name == "selected_topk.json":
            strategy = "uncertainty_legacy"
        else:
            strategy = "unknown"

    seed = data.get("seed")

    return {
        "round": round_name,
        "file": str(path),
        "strategy": strategy,
        "policy": data.get("selection_policy", ""),
        "seed": "" if seed is None else seed,
        "n_frames": data.get("n_frames", ""),
        "top_k": data.get("top_k", len(selected)),
        "force_dev_max_mean": safe_mean(force_dev_max),
        "force_dev_max_max": max(force_dev_max),
        "force_dev_max_min": min(force_dev_max),
        "force_dev_mean_mean": safe_mean(force_dev_mean),
        "energy_dev_mean": safe_mean(energy_dev),
        "energy_dev_max": max(energy_dev),
        "energy_dev_min": min(energy_dev),
        "selected_indices": ";".join(str(x) for x in frame_indices),
    }


def collect_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for round_name, round_dir in ROUND_DIRS:
        if not round_dir.exists():
            print(f"Warning: missing directory: {round_dir}")
            continue

        # Keep the newly generated baseline files.
        # Skip selected_topk.json to avoid duplicating selected_uncertainty.json.
        paths = sorted(round_dir.glob("selected_*.json"))
        for path in paths:
            if path.name == "selected_topk.json":
                continue
            rows.append(summarize_one(round_name, path))

    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in rows:
        key = (str(row["round"]), str(row["strategy"]))
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []

    for (round_name, strategy), items in sorted(groups.items()):
        force_means = [float(x["force_dev_max_mean"]) for x in items]
        energy_means = [float(x["energy_dev_mean"]) for x in items]

        summary_rows.append(
            {
                "round": round_name,
                "strategy": strategy,
                "n_runs": len(items),
                "top_k": items[0]["top_k"],
                "force_dev_max_mean_avg": safe_mean(force_means),
                "force_dev_max_mean_std": safe_stdev(force_means),
                "energy_dev_mean_avg": safe_mean(energy_means),
                "energy_dev_mean_std": safe_stdev(energy_means),
            }
        )

    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(v) for k, v in row.items()})


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Selection-level Baseline Summary",
        "",
        "This table summarizes selection-level behavior for uncertainty sampling and random sampling.",
        "",
        "| Round | Strategy | Runs | Top-K | Avg selected force_dev_max | Std | Avg selected energy_dev | Std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| {round} | {strategy} | {n_runs} | {top_k} | {force_avg} | {force_std} | {energy_avg} | {energy_std} |".format(
                round=row["round"],
                strategy=row["strategy"],
                n_runs=row["n_runs"],
                top_k=row["top_k"],
                force_avg=fmt(row["force_dev_max_mean_avg"]),
                force_std=fmt(row["force_dev_max_mean_std"]),
                energy_avg=fmt(row["energy_dev_mean_avg"]),
                energy_std=fmt(row["energy_dev_mean_std"]),
            )
        )

    lines.extend(
        [
            "",
            "Note: This is a selection-level baseline summary. It does not yet represent retrained model accuracy.",
            "The next step is to merge selected frames, retrain committee models, and compare Energy/Force RMSE curves.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = collect_runs()
    summary_rows = aggregate(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RUNS_CSV, rows)
    write_csv(SUMMARY_CSV, summary_rows)
    write_markdown(SUMMARY_MD, summary_rows)

    print("========== Selection baseline summary generated ==========")
    print(f"runs_csv:    {RUNS_CSV}")
    print(f"summary_csv: {SUMMARY_CSV}")
    print(f"summary_md:  {SUMMARY_MD}")
    print()
    for row in summary_rows:
        print(
            f"{row['round']} | {row['strategy']} | "
            f"n_runs={row['n_runs']} | "
            f"force_dev_max_mean_avg={fmt(row['force_dev_max_mean_avg'])}"
        )


if __name__ == "__main__":
    main()
