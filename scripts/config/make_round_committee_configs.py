#!/usr/bin/env python3
# normalized with LF line endings
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def set_nested_seed(config: dict, seed: int) -> None:
    config.setdefault("model", {})
    config["model"].setdefault("descriptor", {})
    config["model"].setdefault("fitting_net", {})
    config.setdefault("training", {})

    config["model"]["descriptor"]["seed"] = seed
    config["model"]["fitting_net"]["seed"] = seed
    config["training"]["seed"] = seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DeePMD committee configs for a new active learning round."
    )

    parser.add_argument("--base", default="configs/deepmd/toy_h2_input.json")
    parser.add_argument("--output-dir", default="configs/deepmd/round_001_committee")
    parser.add_argument("--train-system", default="/data/zft/data/toy_h2/round_001_train")
    parser.add_argument("--valid-system", default="/data/zft/data/toy_h2/valid")
    parser.add_argument("--round-id", type=int, default=1)
    parser.add_argument("--n-models", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=1101)

    args = parser.parse_args()

    base_path = Path(args.base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = json.loads(base_path.read_text(encoding="utf-8"))

    for model_id in range(args.n_models):
        cfg = deepcopy(base_config)
        seed = args.base_seed + model_id

        cfg["training"]["training_data"]["systems"] = [args.train_system]
        cfg["training"]["validation_data"]["systems"] = [args.valid_system]

        set_nested_seed(cfg, seed)

        out_path = output_dir / f"toy_h2_round{args.round_id:03d}_model_{model_id:03d}.json"
        out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        print(f"wrote {out_path} with seed={seed}")


if __name__ == "__main__":
    main()
