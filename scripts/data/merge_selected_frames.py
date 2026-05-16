#!/usr/bin/env python3
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

    unique_indices = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            unique_indices.append(idx)
            seen.add(idx)

    return unique_indices


def get_set_dirs(system_dir: Path) -> list[Path]:
    set_dirs = sorted(
        p for p in system_dir.iterdir()
        if p.is_dir() and p.name.startswith("set.")
    )
    if not set_dirs:
        raise FileNotFoundError(f"No set.* directories found in {system_dir}")
    return set_dirs


def load_array_from_system(system_dir: Path, filename: str, required: bool) -> np.ndarray | None:
    arrays = []

    for set_dir in get_set_dirs(system_dir):
        path = set_dir / filename

        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing required file: {path}")
            return None

        arrays.append(np.load(path))

    return np.concatenate(arrays, axis=0)


def copy_raw_files(train_dir: Path, candidate_dir: Path, output_dir: Path) -> None:
    for name in RAW_FILES:
        train_file = train_dir / name
        candidate_file = candidate_dir / name

        if not train_file.exists():
            raise FileNotFoundError(f"Missing raw file in train set: {train_file}")
        if not candidate_file.exists():
            raise FileNotFoundError(f"Missing raw file in candidate set: {candidate_file}")

        train_text = train_file.read_text(encoding="utf-8").strip()
        candidate_text = candidate_file.read_text(encoding="utf-8").strip()

        if train_text != candidate_text:
            raise ValueError(
                f"{name} mismatch between train and candidate systems.\n"
                f"train: {train_text}\n"
                f"candidate: {candidate_text}"
            )

        shutil.copy2(train_file, output_dir / name)


def merge_array(
    train_dir: Path,
    candidate_dir: Path,
    selected_indices: list[int],
    filename: str,
    required: bool,
) -> np.ndarray | None:
    train_array = load_array_from_system(train_dir, filename, required=required)
    candidate_array = load_array_from_system(candidate_dir, filename, required=required)

    if train_array is None or candidate_array is None:
        return None

    n_candidate = candidate_array.shape[0]
    bad_indices = [idx for idx in selected_indices if idx < 0 or idx >= n_candidate]

    if bad_indices:
        raise IndexError(
            f"Selected indices out of range for {filename}. "
            f"n_candidate={n_candidate}, bad_indices={bad_indices}"
        )

    selected_array = candidate_array[selected_indices]
    merged = np.concatenate([train_array, selected_array], axis=0)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge selected candidate frames into a DeepMD npy training system."
    )
    parser.add_argument("--train", required=True, help="Original DeepMD training system.")
    parser.add_argument("--candidate", required=True, help="Candidate DeepMD system.")
    parser.add_argument("--selection", required=True, help="Selection JSON file.")
    parser.add_argument("--output", required=True, help="Output merged DeepMD training system.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")

    args = parser.parse_args()

    train_dir = Path(args.train).resolve()
    candidate_dir = Path(args.candidate).resolve()
    selection_json = Path(args.selection).resolve()
    output_dir = Path(args.output).resolve()
    output_set = output_dir / "set.000"

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)

    output_set.mkdir(parents=True, exist_ok=True)

    selected_indices = load_selected_indices(selection_json)

    copy_raw_files(train_dir, candidate_dir, output_dir)

    saved_arrays: dict[str, tuple[int, ...]] = {}

    for filename in REQUIRED_ARRAYS:
        merged = merge_array(
            train_dir=train_dir,
            candidate_dir=candidate_dir,
            selected_indices=selected_indices,
            filename=filename,
            required=True,
        )
        assert merged is not None
        np.save(output_set / filename, merged)
        saved_arrays[filename] = merged.shape

    for filename in OPTIONAL_ARRAYS:
        merged = merge_array(
            train_dir=train_dir,
            candidate_dir=candidate_dir,
            selected_indices=selected_indices,
            filename=filename,
            required=False,
        )
        if merged is not None:
            np.save(output_set / filename, merged)
            saved_arrays[filename] = merged.shape

    metadata = {
        "train_dir": str(train_dir),
        "candidate_dir": str(candidate_dir),
        "selection_json": str(selection_json),
        "output_dir": str(output_dir),
        "selected_indices": selected_indices,
        "n_selected": len(selected_indices),
        "saved_arrays": {k: list(v) for k, v in saved_arrays.items()},
    }

    (output_dir / "merge_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("========== Merge finished ==========")
    print(f"train      : {train_dir}")
    print(f"candidate  : {candidate_dir}")
    print(f"selection  : {selection_json}")
    print(f"output     : {output_dir}")
    print(f"n_selected : {len(selected_indices)}")

    for name, shape in saved_arrays.items():
        print(f"{name:10s}: {shape}")


if __name__ == "__main__":
    main()
