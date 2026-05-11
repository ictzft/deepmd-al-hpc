import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-json",
        type=str,
        default="experiments/exp_005_committee_prediction/selected_topk.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/exp_006_offline_active_learning/round_001_selection.json",
    )
    parser.add_argument("--initial-train-frames", type=int, default=200)
    parser.add_argument("--round-id", type=int, default=1)
    args = parser.parse_args()

    selected_path = Path(args.selected_json)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(selected_path.read_text(encoding="utf-8"))

    n_frames = int(data["n_frames"])
    top_k = int(data["top_k"])
    selected_frames = data["selected_frames"]

    selected_indices = [item["frame_index"] for item in selected_frames]
    selected_set = set(selected_indices)

    remaining_indices = [i for i in range(n_frames) if i not in selected_set]

    result = {
        "round_id": args.round_id,
        "selection_policy": "top-k by force_dev_max",
        "source_prediction": str(selected_path),
        "candidate_pool": {
            "data_dir": data["data_dir"],
            "n_frames_before_selection": n_frames,
            "n_frames_selected": top_k,
            "n_frames_after_selection": len(remaining_indices),
        },
        "training_set": {
            "n_frames_before_selection": args.initial_train_frames,
            "n_frames_added": top_k,
            "n_frames_after_selection": args.initial_train_frames + top_k,
        },
        "committee": {
            "n_models": data["n_models"],
            "n_atoms": data["n_atoms"],
            "models": data["models"],
        },
        "selected_indices": selected_indices,
        "remaining_candidate_indices": remaining_indices,
        "selected_frames": selected_frames,
    }

    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("========== Offline Active Learning Round ==========")
    print(f"round_id: {args.round_id}")
    print(f"selection_policy: top-k by force_dev_max")
    print(f"candidate frames before selection: {n_frames}")
    print(f"selected frames: {top_k}")
    print(f"candidate frames after selection: {len(remaining_indices)}")
    print(f"training frames before selection: {args.initial_train_frames}")
    print(f"training frames after selection: {args.initial_train_frames + top_k}")
    print(f"selected indices: {selected_indices}")
    print(f"saved to: {output_path}")


if __name__ == "__main__":
    main()
