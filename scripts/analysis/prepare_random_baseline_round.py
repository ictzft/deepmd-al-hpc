#!/usr/bin/env python3
"""Prepare data and configs for a random baseline active learning round.

This script automates the transition from one random baseline round to the next.
It handles: selection → merge → remaining candidate → committee configs.

Usage:
  # Generate Round 002 data + configs for random seed0
  python scripts/analysis/prepare_random_baseline_round.py \
    --seed-label seed0 --round-id 2

  # Generate Round 003 data + configs for random seed0
  python scripts/analysis/prepare_random_baseline_round.py \
    --seed-label seed0 --round-id 3

  # Batch: all seeds for Round 002 and 003
  for seed in seed0 seed1 seed2; do
    for round_id in 2 3; do
      python scripts/analysis/prepare_random_baseline_round.py \
        --seed-label $seed --round-id $round_id
    done
  done

Input expectations:
  - Round 002 reads from: experiments/baselines/random_{seed}_round001_committee_prediction/
  - Round 003 reads from: experiments/baselines/random_{seed}_round002_committee_prediction/
  - Data source: data/toy_h2/random_{seed}_round_{N-1}_train and _candidate
  - Output data: data/toy_h2/random_{seed}_round_{N}_train and _candidate

Note: This script does NOT run DeePMD training or inference. It only prepares data
and configs. Training must be run separately via scripts/train/train_round_committee_models.sh.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model seeds distinguish rounds but are identical across seed0/seed1/seed2
# (variation in random baseline comes from which frames are selected, not model init)
ROUND_BASE_SEEDS = {1: 1101, 2: 1201, 3: 1301}

TOP_K = 10
N_MODELS = 4


def resolve_script(name: str) -> Path:
    """Find a script relative to the project root."""
    candidates = [
        PROJECT_ROOT / "scripts" / "data" / name,
        PROJECT_ROOT / "scripts" / "config" / name,
        PROJECT_ROOT / "scripts" / "active_learning" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find script: {name}")


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}")
    print(f"[{desc}]")
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with code {result.returncode}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare data and configs for a random baseline round."
    )
    parser.add_argument(
        "--seed-label", required=True, choices=["seed0", "seed1", "seed2"],
        help="Random seed label."
    )
    parser.add_argument(
        "--round-id", type=int, required=True, choices=[2, 3],
        help="Target round ID (2 or 3)."
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Frames to select per round (default: {TOP_K})."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing."
    )
    args = parser.parse_args()

    seed_label = args.seed_label
    round_id = args.round_id
    prev_round = round_id - 1
    top_k = args.top_k
    base_seed = ROUND_BASE_SEEDS[round_id]

    # --- Paths ---
    # Source: previous round's committee prediction
    src_pred_dir = (
        PROJECT_ROOT / "experiments" / "baselines"
        / f"random_{seed_label}_round{prev_round:03d}_committee_prediction"
    )
    src_selected_json = src_pred_dir / "selected_topk.json"

    # Data: previous round's train/candidate
    src_train = PROJECT_ROOT / "data" / "toy_h2" / f"random_{seed_label}_round_{prev_round:03d}_train"
    src_candidate = PROJECT_ROOT / "data" / "toy_h2" / f"random_{seed_label}_round_{prev_round:03d}_candidate"

    # Output data
    out_train = PROJECT_ROOT / "data" / "toy_h2" / f"random_{seed_label}_round_{round_id:03d}_train"
    out_candidate = PROJECT_ROOT / "data" / "toy_h2" / f"random_{seed_label}_round_{round_id:03d}_candidate"

    # Output configs
    out_config_dir = PROJECT_ROOT / "configs" / "deepmd" / f"random_{seed_label}_round_{round_id:03d}_committee"

    # Output prediction dir (placeholder — will be populated after training)
    out_pred_dir = (
        PROJECT_ROOT / "experiments" / "baselines"
        / f"random_{seed_label}_round{round_id:03d}_committee_prediction"
    )
    out_models_dir = (
        PROJECT_ROOT / "experiments" / "baselines"
        / f"random_{seed_label}_round{round_id:03d}_committee_models"
    )

    # --- Validate inputs ---
    if not src_selected_json.exists():
        print(f"ERROR: Source selected_topk.json not found: {src_selected_json}")
        print(f"  This file should exist from the previous round's committee prediction.")
        print(f"  Run committee prediction for Round {prev_round} first:")
        print(f"  python scripts/inference/predict_committee_models.py \\")
        print(f"    --data data/toy_h2/random_{seed_label}_round_{prev_round:03d}_candidate \\")
        print(f"    --models ... \\")
        print(f"    --output {src_pred_dir}/committee_predictions.npz \\")
        print(f"    --selected-json {src_selected_json} \\")
        print(f"    --top-k {top_k}")
        sys.exit(1)

    if not src_train.exists():
        print(f"ERROR: Source train dir not found: {src_train}")
        sys.exit(1)

    if not src_candidate.exists():
        print(f"ERROR: Source candidate dir not found: {src_candidate}")
        sys.exit(1)

    # --- Print plan ---
    print("=" * 60)
    print(f"Random Baseline Round {round_id} Preparation")
    print(f"  seed_label:  {seed_label}")
    print(f"  round_id:    {round_id}")
    print(f"  prev_round:  {prev_round}")
    print(f"  top_k:       {top_k}")
    print(f"  base_seed:   {base_seed}")
    print(f"  src_train:   {src_train}")
    print(f"  src_cand:    {src_candidate}")
    print(f"  src_pred:    {src_selected_json}")
    print(f"  out_train:   {out_train}")
    print(f"  out_cand:    {out_candidate}")
    print(f"  out_config:  {out_config_dir}")
    print(f"  out_models:  {out_models_dir}")
    print(f"  out_pred:    {out_pred_dir}")

    if args.dry_run:
        print("\n[DRY RUN] No commands executed.")
        print("\nTo run the full pipeline after data+configs are ready:")
        print(f"  bash scripts/train/train_round_committee_models.sh \\")
        print(f"    {round_id:03d} \\")
        print(f"    configs/deepmd/random_{seed_label}_round_{round_id:03d}_committee \\")
        print(f"    experiments/baselines/random_{seed_label}_round{round_id:03d}_committee_models \\")
        print(f"    /data/zft/deepmd-al-hpc/data/toy_h2/valid")
        print()
        print(f"  python scripts/inference/predict_committee_models.py \\")
        print(f"    --data data/toy_h2/random_{seed_label}_round_{round_id:03d}_candidate \\")
        print(f"    --models \\")
        for m in range(N_MODELS):
            print(f"      {out_models_dir}/model_{m:03d}/frozen_model.pb \\")
        print(f"    --output {out_pred_dir}/committee_predictions.npz \\")
        print(f"    --selected-json {out_pred_dir}/selected_topk.json \\")
        print(f"    --top-k {top_k}")
        return

    # --- Step 1: Merge selected frames into training set ---
    run(
        [
            sys.executable, str(resolve_script("merge_selected_frames.py")),
            "--train", str(src_train),
            "--candidate", str(src_candidate),
            "--selection", str(src_selected_json),
            "--output", str(out_train),
            "--overwrite",
        ],
        f"Step 1/3: Merge → {out_train.name}",
    )

    # --- Step 2: Create remaining candidate ---
    run(
        [
            sys.executable, str(resolve_script("make_remaining_candidate.py")),
            "--candidate", str(src_candidate),
            "--selection", str(src_selected_json),
            "--output", str(out_candidate),
            "--overwrite",
        ],
        f"Step 2/3: Remaining candidate → {out_candidate.name}",
    )

    # --- Step 3: Generate committee configs ---
    train_abs = out_train.resolve()
    valid_abs = (PROJECT_ROOT / "data" / "toy_h2" / "valid").resolve()
    run(
        [
            sys.executable, str(resolve_script("make_round_committee_configs.py")),
            "--base", str(PROJECT_ROOT / "configs" / "deepmd" / "toy_h2_input.json"),
            "--output-dir", str(out_config_dir),
            "--train-system", str(train_abs),
            "--valid-system", str(valid_abs),
            "--round-id", str(round_id),
            "--n-models", str(N_MODELS),
            "--base-seed", str(base_seed),
        ],
        f"Step 3/3: Committee configs → {out_config_dir.name}",
    )

    # --- Summary ---
    n_selected = top_k
    src_train_count = _count_frames(src_train)
    new_train_count = (src_train_count + n_selected) if src_train_count else None

    print(f"\n{'='*60}")
    print("Preparation complete. Summary:")
    print(f"  seed_label:    {seed_label}")
    print(f"  round_id:      {round_id}")
    print(f"  base_seed:     {base_seed}")
    print(f"  train frames:  {src_train_count} → {new_train_count}")
    print(f"  candidate:     {_count_frames(out_candidate)} remaining")
    print(f"  configs:       {out_config_dir}")
    print(f"\nNext: run training then prediction:")
    print(f"  bash scripts/train/train_round_committee_models.sh \\")
    print(f"    {round_id:03d} \\")
    print(f"    configs/deepmd/random_{seed_label}_round_{round_id:03d}_committee \\")
    print(f"    experiments/baselines/random_{seed_label}_round{round_id:03d}_committee_models \\")
    print(f"    /data/zft/deepmd-al-hpc/data/toy_h2/valid")
    print(f"\nThen run prediction:")
    print(f"  python scripts/inference/predict_committee_models.py \\")
    print(f"    --data data/toy_h2/random_{seed_label}_round_{round_id:03d}_candidate \\")
    print(f"    --models {' '.join(f'{out_models_dir}/model_{m:03d}/frozen_model.pb' for m in range(N_MODELS))} \\")
    print(f"    --output {out_pred_dir}/committee_predictions.npz \\")
    print(f"    --selected-json {out_pred_dir}/selected_topk.json \\")
    print(f"    --top-k {top_k}")


def _count_frames(data_dir: Path) -> int | None:
    import numpy as np
    total = 0
    found = False
    for set_dir in sorted(data_dir.glob("set.*")):
        coord = set_dir / "coord.npy"
        if coord.exists():
            total += int(np.load(coord).shape[0])
            found = True
    return total if found else None


if __name__ == "__main__":
    main()
