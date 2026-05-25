from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.al.selector import select_by_strategy


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Template json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(x: Any) -> float:
    return float(np.asarray(x).item())


def build_selected_frames(
    selected_indices: np.ndarray,
    force_dev_max: np.ndarray,
    force_dev_mean: np.ndarray,
    energy_dev: np.ndarray,
) -> list[dict[str, Any]]:
    selected_frames: list[dict[str, Any]] = []

    for rank, idx in enumerate(selected_indices, start=1):
        idx = int(idx)
        selected_frames.append(
            {
                "rank": rank,
                "frame_index": idx,
                "force_dev_max": to_float(force_dev_max[idx]),
                "force_dev_mean": to_float(force_dev_mean[idx]),
                "energy_dev": to_float(energy_dev[idx]),
            }
        )

    return selected_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select candidate frames from committee prediction npz using "
            "uncertainty, random, uncertainty-diversity, or trust-level sampling."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to committee_predictions.npz.",
    )
    parser.add_argument(
        "--strategy",
        default="uncertainty",
        choices=["uncertainty", "topk", "top-k", "random",
                 "uncertainty-diversity", "diversity",
                 "trust-level", "trust", "dpgen"],
        help="Selection strategy.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of frames to select.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Only used by random strategy.",
    )
    parser.add_argument(
        "--top-m",
        type=int,
        default=30,
        help="For diversity: top-M high-uncertainty pool size before FPS.",
    )
    parser.add_argument(
        "--low-pct",
        type=float,
        default=50.0,
        help="For trust-level: percentile for accurate/candidate boundary.",
    )
    parser.add_argument(
        "--high-pct",
        type=float,
        default=90.0,
        help="For trust-level: percentile for candidate/failed boundary.",
    )
    parser.add_argument(
        "--candidate-ratio",
        type=float,
        default=0.8,
        help="For trust-level: fraction from candidate region.",
    )
    parser.add_argument(
        "--template-json",
        type=Path,
        default=None,
        help=(
            "Optional existing selected_topk.json used as a metadata template. "
            "This keeps fields such as data_dir and models compatible."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output selected json file.",
    )

    args = parser.parse_args()

    if not args.predictions.exists():
        raise FileNotFoundError(f"Prediction npz not found: {args.predictions}")

    data = np.load(args.predictions, allow_pickle=True)

    required_keys = ["force_dev_max", "force_dev_mean", "energy_dev"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key in prediction npz: {key}")

    force_dev_max = np.asarray(data["force_dev_max"])
    force_dev_mean = np.asarray(data["force_dev_mean"])
    energy_dev = np.asarray(data["energy_dev"])

    if force_dev_max.ndim != 1:
        raise ValueError(f"force_dev_max should be 1D, got {force_dev_max.shape}")

    n_frames = int(force_dev_max.shape[0])
    top_k = min(int(args.top_k), n_frames)

    # Extract coords if available (needed for diversity strategy)
    coords = data.get("coord", None)
    if coords is not None:
        coords = np.asarray(coords)

    result = select_by_strategy(
        scores=force_dev_max,
        k=top_k,
        strategy=args.strategy,
        seed=args.seed,
        coords=coords,
        top_m=args.top_m,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
        candidate_ratio=args.candidate_ratio,
    )

    # Handle dict return from trust-level strategy
    extra_meta: dict[str, Any] = {}
    if isinstance(result, dict):
        selected_indices = result.pop("selected_indices")
        extra_meta = {k: v for k, v in result.items()}
    else:
        selected_indices = result

    template = load_json(args.template_json)

    output: dict[str, Any] = dict(template)
    policy = {
        "random": "random sampling",
        "uncertainty": "top-k by force_dev_max",
        "topk": "top-k by force_dev_max",
        "top-k": "top-k by force_dev_max",
        "uncertainty-diversity": "top-M uncertainty + FPS diversity",
        "diversity": "top-M uncertainty + FPS diversity",
        "trust-level": "DP-GEN-style trust-level sampling",
        "trust": "DP-GEN-style trust-level sampling",
        "dpgen": "DP-GEN-style trust-level sampling",
    }.get(args.strategy, args.strategy)

    output.update(
        {
            "n_frames": n_frames,
            "top_k": int(len(selected_indices)),
            "output_npz": str(args.predictions),
            "selection_strategy": args.strategy,
            "selection_policy": policy,
            "score_key": "force_dev_max",
            "seed": int(args.seed) if args.strategy == "random" else None,
            "selected_frames": build_selected_frames(
                selected_indices=selected_indices,
                force_dev_max=force_dev_max,
                force_dev_mean=force_dev_mean,
                energy_dev=energy_dev,
            ),
        }
    )
    if extra_meta:
        output.update(extra_meta)

    # If the npz contains force predictions, infer model and atom counts.
    if "forces" in data:
        forces = np.asarray(data["forces"])
        if forces.ndim == 4:
            output["n_models"] = int(forces.shape[0])
            output["n_atoms"] = int(forces.shape[2])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("========== Selection finished ==========")
    print(f"predictions: {args.predictions}")
    print(f"strategy:    {args.strategy}")
    print(f"top_k:       {len(selected_indices)}")
    print(f"seed:        {args.seed if args.strategy == 'random' else 'N/A'}")
    print(f"output:      {args.output}")
    print()
    print("selected frame indices:")
    print([int(x) for x in selected_indices])


if __name__ == "__main__":
    main()
