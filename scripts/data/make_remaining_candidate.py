#!/usr/bin/env python3
# normalized with LF line endings
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = ["coord.npy", "box.npy", "energy.npy", "force.npy"]
OPTIONAL_ARRAYS = ["virial.npy"]
RAW_FILES = ["type.raw", "type_map.raw"]


def load_selected_indices(selection_json: Path) -> list[int]:
    data = json.loads(selection_json.read_text(encoding="utf-8"))

    if "selected_indices" in data:
        indices = [int(i) for i in data["selected_indices"]]
    elif "selected_frame_indices" in data:
        indices = [int(i) for i in data["selected_frame_indices"]]
    elif "selected_frames" in data:
        indices = []
        for item in data["selected_frames"]:
            if isinstance(item, dict):
                if "frame_index" in item:
                    indices.append(int(item["frame_index"]))
                elif "index" in item:
                    indices.append(int(item["index"]))
                else:
                    raise KeyError(f"Cannot find frame index in selected_frames item: {item}")
            else:
                indices.append(int(item))
    else:
        raise KeyError(
            "Cannot find selected indices. Expected one of: "
            "selected_indices, selected_frame_indices, selected_frames"
        )

    return sorted(set(indices))


def get_set_dirs(system_dir: Path) -> list[Path]:
    set_dirs = sorted(
        p for p in system_dir.iterdir()
        if p.is_dir() and p.name.startswith("set.")
    )
    if not set_dirs:
        raise FileNotFoundError(f"No set.* directories found in {system_dir}")
    return set_dirs


def load_array(system_dir: Path, filename: str, required: bool) -> np.ndarray | None:
    arrays = []
    for set_dir in get_set_dirs(system_dir):
        path = set_dir / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing required file: {path}")
            return None
        arrays.append(np.load(path))
    return np.concatenate(arrays, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create remaining candidate DeepMD system by excluding selected frame indices."
    )
    parser.add_argument("--candidate", required=True, help="Original candidate DeepMD system.")
    parser.add_argument("--selection", required=True, help="Selection JSON file.")
    parser.add_argument("--output", required=True, help="Output remaining candidate system.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    candidate_dir = Path(args.candidate).resolve()
    selection_json = Path(args.selection).resolve()
    output_dir = Path(args.output).resolve()
    output_set = output_dir / "set.000"

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}. Use --overwrite.")
        shutil.rmtree(output_dir)

    output_set.mkdir(parents=True, exist_ok=True)

    selected_indices = load_selected_indices(selection_json)

    coord = load_array(candidate_dir, "coord.npy", required=True)
    assert coord is not None
    n_frames = coord.shape[0]

    selected_set = set(selected_indices)
    remaining_indices = [i for i in range(n_frames) if i not in selected_set]

    for idx in selected_indices:
        if idx < 0 or idx >= n_frames:
            raise IndexError(f"Selected index out of range: {idx}, n_frames={n_frames}")

    for name in RAW_FILES:
        src = candidate_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing raw file: {src}")
        shutil.copy2(src, output_dir / name)

    saved_arrays = {}

    for name in REQUIRED_ARRAYS:
        arr = load_array(candidate_dir, name, required=True)
        assert arr is not None
        remain = arr[remaining_indices]
        np.save(output_set / name, remain)
        saved_arrays[name] = remain.shape

    for name in OPTIONAL_ARRAYS:
        arr = load_array(candidate_dir, name, required=False)
        if arr is not None:
            remain = arr[remaining_indices]
            np.save(output_set / name, remain)
            saved_arrays[name] = remain.shape

    metadata = {
        "candidate_dir": str(candidate_dir),
        "selection_json": str(selection_json),
        "output_dir": str(output_dir),
        "n_original": n_frames,
        "selected_indices": selected_indices,
        "n_selected": len(selected_indices),
        "remaining_indices": remaining_indices,
        "n_remaining": len(remaining_indices),
        "saved_arrays": {k: list(v) for k, v in saved_arrays.items()},
    }

    (output_dir / "remaining_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("========== Remaining candidate finished ==========")
    print(f"candidate   : {candidate_dir}")
    print(f"selection   : {selection_json}")
    print(f"output      : {output_dir}")
    print(f"n_original  : {n_frames}")
    print(f"n_selected  : {len(selected_indices)}")
    print(f"n_remaining : {len(remaining_indices)}")
    for name, shape in saved_arrays.items():
        print(f"{name:10s}: {shape}")


if __name__ == "__main__":
    main()
