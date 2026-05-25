#!/usr/bin/env python3
"""Template for independent-test evaluation of trained models.

Usage:
  python scripts/evaluation/evaluate_independent_test_template.py \
    --models experiments/strategy_comparison_toy_h2/uncertainty/seed0/round_003/models \
    --test-data data/real_datasets/small_system/independent_test \
    --output experiments/strategy_comparison_toy_h2/uncertainty/seed0/round_003/independent_test_metrics.json

This script attempts to run DeepMD-kit dp test on an independent test set.
If dp/models/data are unavailable, it outputs status=not_run with a reason.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _find_frozen_models(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("**/frozen_model.pb"))


def _check_dp() -> bool:
    for path in os.environ.get("PATH", "").split(os.pathsep):
        if (Path(path) / "dp").exists():
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent test evaluation template.")
    parser.add_argument("--models", required=True, help="Directory containing frozen_model.pb files.")
    parser.add_argument("--test-data", required=True, help="Path to independent test DeePMD data.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--n-test", type=int, default=200, help="Number of test frames to evaluate.")
    args = parser.parse_args()

    models_dir = Path(args.models)
    test_data = Path(args.test_data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "models_dir": str(models_dir),
        "test_data": str(test_data),
        "status": "not_run",
        "reason": "",
        "energy_rmse": None,
        "force_rmse": None,
        "n_models": 0,
    }

    # Check prerequisites
    if not _check_dp():
        result["reason"] = "DeepMD-kit dp executable not found on PATH."
    elif not test_data.exists():
        result["reason"] = f"Test data directory not found: {test_data}"
    else:
        models = _find_frozen_models(models_dir)
        result["n_models"] = len(models)
        if not models:
            result["reason"] = f"No frozen_model.pb found in {models_dir}"
        else:
            # Attempt dp test
            try:
                for i, pb in enumerate(models):
                    cmd = [
                        "dp", "-b", "tensorflow", "test",
                        "-m", str(pb),
                        "-s", str(test_data),
                        "-n", str(min(args.n_test, 50)),
                    ]
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True,
                        cwd=str(PROJECT_ROOT), timeout=120,
                    )
                    # Parse output for RMSE
                    import re
                    e_match = re.findall(r"Energy RMSE\s+:\s+(\S+) eV", proc.stdout)
                    f_match = re.findall(r"Force  RMSE\s+:\s+(\S+) eV", proc.stdout)
                    if e_match and f_match:
                        result["status"] = "completed"
                        result["energy_rmse"] = float(e_match[-1])
                        result["force_rmse"] = float(f_match[-1])
                        break  # Use first successful model
            except FileNotFoundError:
                result["reason"] = "dp executable not found."
            except subprocess.TimeoutExpired:
                result["reason"] = "dp test timed out."
            except Exception as e:
                result["reason"] = str(e)[:200]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Status: {result['status']}")
    if result["reason"]:
        print(f"Reason: {result['reason']}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
