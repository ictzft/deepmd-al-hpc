import argparse
import json
from pathlib import Path

import numpy as np
from deepmd.infer import DeepPot


def load_deepmd_system(system_dir: Path):
    """
    Load a simple DeepMD npy system.

    Expected structure:
    system_dir/
      type.raw
      type_map.raw
      set.000/
        coord.npy
        force.npy
        energy.npy
        box.npy
    """
    system_dir = Path(system_dir)

    type_path = system_dir / "type.raw"
    if not type_path.exists():
        raise FileNotFoundError(f"Missing type.raw: {type_path}")

    atype = np.loadtxt(type_path, dtype=np.int32)
    atype = np.atleast_1d(atype).astype(np.int32)
    natoms = int(atype.shape[0])

    set_dirs = sorted([p for p in system_dir.iterdir() if p.is_dir() and p.name.startswith("set.")])
    if not set_dirs:
        raise FileNotFoundError(f"No set.* directories found in {system_dir}")

    coords = []
    boxes = []
    ref_energies = []
    ref_forces = []

    for set_dir in set_dirs:
        coord_path = set_dir / "coord.npy"
        box_path = set_dir / "box.npy"

        if not coord_path.exists():
            raise FileNotFoundError(f"Missing coord.npy: {coord_path}")
        if not box_path.exists():
            raise FileNotFoundError(f"Missing box.npy: {box_path}")

        coords.append(np.load(coord_path))
        boxes.append(np.load(box_path))

        energy_path = set_dir / "energy.npy"
        force_path = set_dir / "force.npy"

        if energy_path.exists():
            ref_energies.append(np.load(energy_path))
        if force_path.exists():
            ref_forces.append(np.load(force_path))

    coord = np.concatenate(coords, axis=0)
    box = np.concatenate(boxes, axis=0)

    if coord.ndim != 2 or coord.shape[1] != natoms * 3:
        raise ValueError(f"Unexpected coord shape {coord.shape}, expected [nframes, {natoms * 3}]")

    if box.ndim != 2 or box.shape[1] != 9:
        raise ValueError(f"Unexpected box shape {box.shape}, expected [nframes, 9]")

    ref_energy = np.concatenate(ref_energies, axis=0) if ref_energies else None
    ref_force = np.concatenate(ref_forces, axis=0) if ref_forces else None

    return coord, box, atype, ref_energy, ref_force


def eval_one_model(model_path: Path, coord: np.ndarray, box: np.ndarray, atype: np.ndarray):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    dp = DeepPot(str(model_path))

    # DeepPot normally accepts:
    # coord: [nframes, natoms * 3]
    # box:   [nframes, 9]
    # atype: [natoms]
    result = dp.eval(coord, box, atype)

    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(f"Unexpected DeepPot.eval output type: {type(result)}")

    energy = np.asarray(result[0])
    force = np.asarray(result[1])

    if len(result) >= 3:
        virial = np.asarray(result[2])
    else:
        virial = None

    nframes = coord.shape[0]
    natoms = atype.shape[0]

    energy = energy.reshape(nframes, -1)
    if energy.shape[1] == 1:
        energy = energy[:, 0]

    force = force.reshape(nframes, natoms, 3)

    if virial is not None:
        virial = virial.reshape(nframes, -1)

    return energy, force, virial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/toy_h2/valid",
        help="Candidate DeepMD system directory.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "experiments/exp_004_committee_models/model_000/frozen_model.pb",
            "experiments/exp_004_committee_models/model_001/frozen_model.pb",
            "experiments/exp_004_committee_models/model_002/frozen_model.pb",
            "experiments/exp_004_committee_models/model_003/frozen_model.pb",
        ],
        help="Frozen DeePMD model paths.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/exp_005_committee_prediction/committee_predictions.npz",
        help="Output npz file.",
    )
    parser.add_argument(
        "--selected-json",
        type=str,
        default="experiments/exp_005_committee_prediction/selected_topk.json",
        help="Output top-K selection json file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of high-uncertainty frames to select.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_path = Path(args.output)
    selected_json_path = Path(args.selected_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_json_path.parent.mkdir(parents=True, exist_ok=True)

    coord, box, atype, ref_energy, ref_force = load_deepmd_system(data_dir)

    nframes = coord.shape[0]
    natoms = atype.shape[0]
    nmodels = len(args.models)

    print("========== Candidate data ==========")
    print(f"data_dir: {data_dir}")
    print(f"nframes : {nframes}")
    print(f"natoms  : {natoms}")
    print(f"coord   : {coord.shape}")
    print(f"box     : {box.shape}")
    print(f"atype   : {atype.tolist()}")

    all_energies = []
    all_forces = []
    all_virials = []

    print("========== Committee prediction ==========")
    for i, model_path in enumerate(args.models):
        print(f"[{i}] model: {model_path}")
        energy, force, virial = eval_one_model(Path(model_path), coord, box, atype)

        print(f"    energy shape: {energy.shape}")
        print(f"    force  shape: {force.shape}")
        if virial is not None:
            print(f"    virial shape: {virial.shape}")

        all_energies.append(energy)
        all_forces.append(force)
        if virial is not None:
            all_virials.append(virial)

    energies = np.stack(all_energies, axis=0)
    forces = np.stack(all_forces, axis=0)
    virials = np.stack(all_virials, axis=0) if len(all_virials) == nmodels else None

    # Deviation calculation
    # forces: [nmodels, nframes, natoms, 3]
    force_std = np.std(forces, axis=0)                 # [nframes, natoms, 3]
    force_dev_per_atom = np.linalg.norm(force_std, axis=-1)  # [nframes, natoms]
    force_dev_max = np.max(force_dev_per_atom, axis=1)        # [nframes]
    force_dev_mean = np.mean(force_dev_per_atom, axis=1)      # [nframes]

    # energies: [nmodels, nframes]
    energy_dev = np.std(energies, axis=0)              # [nframes]

    top_k = min(args.top_k, nframes)
    selected_indices = np.argsort(force_dev_max)[::-1][:top_k]

    print("========== Prediction tensors ==========")
    print(f"energies shape: {energies.shape}")
    print(f"forces shape  : {forces.shape}")
    if virials is not None:
        print(f"virials shape : {virials.shape}")

    print("========== Top-K high uncertainty frames ==========")
    for rank, idx in enumerate(selected_indices, start=1):
        print(
            f"rank={rank:02d}, frame={int(idx)}, "
            f"force_dev_max={force_dev_max[idx]:.6e}, "
            f"force_dev_mean={force_dev_mean[idx]:.6e}, "
            f"energy_dev={energy_dev[idx]:.6e}"
        )

    save_dict = {
        "energies": energies,
        "forces": forces,
        "force_dev_max": force_dev_max,
        "force_dev_mean": force_dev_mean,
        "energy_dev": energy_dev,
        "selected_indices": selected_indices,
        "coord": coord,
        "box": box,
        "atype": atype,
    }

    if virials is not None:
        save_dict["virials"] = virials
    if ref_energy is not None:
        save_dict["ref_energy"] = ref_energy
    if ref_force is not None:
        save_dict["ref_force"] = ref_force

    np.savez(output_path, **save_dict)

    selected_records = []
    for rank, idx in enumerate(selected_indices, start=1):
        selected_records.append(
            {
                "rank": rank,
                "frame_index": int(idx),
                "force_dev_max": float(force_dev_max[idx]),
                "force_dev_mean": float(force_dev_mean[idx]),
                "energy_dev": float(energy_dev[idx]),
            }
        )

    summary = {
        "data_dir": str(data_dir),
        "models": [str(p) for p in args.models],
        "n_models": nmodels,
        "n_frames": nframes,
        "n_atoms": natoms,
        "top_k": top_k,
        "output_npz": str(output_path),
        "selected_frames": selected_records,
    }

    selected_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("========== Saved ==========")
    print(f"npz : {output_path}")
    print(f"json: {selected_json_path}")


if __name__ == "__main__":
    main()
