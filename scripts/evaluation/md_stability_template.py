#!/usr/bin/env python3
"""Template for MD stability validation using LAMMPS + DeePMD model.

Usage:
  python scripts/evaluation/md_stability_template.py \
    --model experiments/strategy_comparison_toy_h2/uncertainty/seed0/round_003/frozen_model.pb \
    --initial-structure data/toy_h2/valid \
    --output experiments/strategy_comparison_toy_h2/uncertainty/seed0/round_003/md_stability_summary.json

This is a TEMPLATE that describes what MD stability validation should check.
Actual MD runs require LAMMPS with DeePMD plugin (lmp -m deepmd).
If LAMMPS is unavailable, outputs status=not_run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _check_lmp() -> bool:
    for path in os.environ.get("PATH", "").split(os.pathsep):
        for name in ["lmp", "lammps", "lmp_mpi"]:
            if (Path(path) / name).exists():
                return True
    return False


def _check_model(pb_path: Path) -> bool:
    return pb_path.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="MD stability validation template.")
    parser.add_argument("--model", required=True, help="Path to frozen_model.pb.")
    parser.add_argument("--initial-structure", required=True, help="DeePMD-format data dir for initial structure.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--steps", type=int, default=10000, help="Number of MD steps.")
    parser.add_argument("--timestep", type=float, default=0.5, help="Timestep in fs.")
    parser.add_argument("--temperature", type=float, default=300, help="Target temperature in K.")
    parser.add_argument("--ensemble", default="nvt", choices=["nve", "nvt", "npt"])
    args = parser.parse_args()

    model_path = Path(args.model)
    struct_path = Path(args.initial_structure)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "model": str(model_path),
        "initial_structure": str(struct_path),
        "steps": args.steps,
        "timestep_fs": args.timestep,
        "temperature_k": args.temperature,
        "ensemble": args.ensemble,
        "status": "not_run",
        "reason": "",
        "energy_drift_per_atom_per_ps": None,
        "temperature_mean": None,
        "temperature_std": None,
        "exploded": None,
        "total_energy_conserved": None,
    }

    if not _check_model(model_path):
        result["reason"] = f"frozen_model.pb not found: {model_path}"
    elif not _check_lmp():
        result["reason"] = "LAMMPS executable not found on PATH. Install LAMMPS with DEEPMD plugin."
    elif not struct_path.exists():
        result["reason"] = f"Initial structure directory not found: {struct_path}"
    else:
        # In a real run, this would:
        # 1. Write a LAMMPS input file using pair_style deepmd
        # 2. Run lmp -i in.lammps
        # 3. Parse the log.lammps for energy/temperature
        # 4. Check for explosion or drift
        result["reason"] = (
            "LAMMPS MD stability validation requires manual setup: "
            "1) write LAMMPS input with pair_style deepmd, "
            "2) run NVE/NVT simulation, "
            "3) check energy conservation and temperature stability. "
            "See docs/real_dataset_plan.md for details."
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Status: {result['status']}")
    print(f"Reason: {result['reason']}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
