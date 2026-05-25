#!/usr/bin/env python3
"""Template for splitting a real DFT/AIMD DeePMD dataset into active-learning splits.

Usage (dry-run):
  python scripts/data/prepare_real_dataset_template.py \
    --source data/real_datasets/my_system/all_data \
    --output data/real_datasets/my_system \
    --initial-train 100 --candidate-pool 1000 --validation 100 --test 100 \
    --seed 0 \
    --dry-run

This script is a TEMPLATE. It does NOT copy or move large data files by default.
It creates a metadata.json and prints the commands needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def count_frames(data_dir: Path) -> int:
    total = 0
    for set_dir in sorted(data_dir.glob("set.*")):
        coord = set_dir / "coord.npy"
        if coord.exists():
            import numpy as np
            total += int(np.load(coord).shape[0])
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare real DFT/AIMD dataset splits for active learning."
    )
    parser.add_argument("--source", required=True,
                        help="Path to source DeePMD-format data directory (contains set.000/, etc.).")
    parser.add_argument("--output", required=True,
                        help="Output root directory for split data.")
    parser.add_argument("--initial-train", type=int, default=100,
                        help="Number of frames for initial training set.")
    parser.add_argument("--candidate-pool", type=int, default=1000,
                        help="Number of frames for candidate pool.")
    parser.add_argument("--validation", type=int, default=100,
                        help="Number of frames for validation set.")
    parser.add_argument("--test", type=int, default=100,
                        help="Number of frames for independent test set.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for splitting.")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Print plan without copying data (default: True).")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    print("=" * 60)
    print("Real Dataset Preparation Template")
    print("=" * 60)
    print(f"  source:         {source}")
    print(f"  output:         {output}")
    print(f"  initial_train:  {args.initial_train} frames")
    print(f"  candidate_pool: {args.candidate_pool} frames")
    print(f"  validation:     {args.validation} frames")
    print(f"  test:           {args.test} frames")
    print(f"  seed:           {args.seed}")
    print(f"  dry_run:        {args.dry_run}")
    print()

    if not source.exists():
        print(f"ERROR: Source directory not found: {source}")
        print("Please provide a valid DeePMD-format data directory.")
        sys.exit(1)

    try:
        n_total = count_frames(source)
        print(f"Source contains ~{n_total} frames in set.*/ subdirectories.")
    except Exception as e:
        print(f"Could not count frames: {e}")
        n_total = 0

    n_needed = args.initial_train + args.candidate_pool + args.validation + args.test
    if n_total > 0 and n_total < n_needed:
        print(f"WARNING: Source has {n_total} frames, need {n_needed}.")
        print("Reduce split sizes or use a larger dataset.")

    # Plan the splits
    splits = {
        "initial_train": args.initial_train,
        "candidate_pool": args.candidate_pool,
        "validation": args.validation,
        "independent_test": args.test,
    }

    metadata: dict[str, Any] = {
        "dataset_name": source.name,
        "source": str(source.resolve()),
        "split": splits,
        "seed": args.seed,
        "n_total_frames": n_total,
        "notes": "Template metadata. Actual data split requires running on V100 with full dataset.",
    }

    # Create output dirs
    if args.dry_run:
        print("\n[Dry-run] Would create:")
        for name in splits:
            d = output / name
            print(f"  {d}/")
        print(f"  {output}/metadata.json")
        print(f"    -> {json.dumps(metadata, indent=2)}")
    else:
        for name in splits:
            (output / name).mkdir(parents=True, exist_ok=True)
        meta_path = output / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nCreated directory structure and {meta_path}")

    print("\n" + "=" * 60)
    print("NEXT STEPS (manual, on V100):")
    print("=" * 60)
    print(f"1. Split source data into the 4 directories under {output}/:")
    print(f"   - initial_train/  ({args.initial_train} frames)")
    print(f"   - candidate_pool/ ({args.candidate_pool} frames)")
    print(f"   - validation/     ({args.validation} frames)")
    print(f"   - independent_test/ ({args.test} frames)")
    print(f"2. Each directory must be in DeePMD format (type.raw + set.000/coord.npy, etc.)")
    print(f"3. Run active learning pipeline using data/real_datasets/<name>/ as base")
    print(f"4. initial_train -> used for Round 0 committee training")
    print(f"5. candidate_pool -> treated as the unlabeled pool for selection")
    print(f"6. validation -> used for model selection / hyperparameter tuning")
    print(f"7. independent_test -> final evaluation only (DO NOT use for selection)")
    print(f"\nCRITICAL: independent_test must remain separate from all active learning rounds.")
    print(f"Do NOT commit the split data to Git (add to .gitignore).")


if __name__ == "__main__":
    main()
