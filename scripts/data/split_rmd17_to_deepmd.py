import argparse
import os
import json
import numpy as np

Z_TO_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}

def write_deepmd_system(out_dir, z, coords, energies, forces, indices, type_map):
    os.makedirs(out_dir, exist_ok=True)
    set_dir = os.path.join(out_dir, "set.000")
    os.makedirs(set_dir, exist_ok=True)

    symbols = [Z_TO_SYMBOL[int(x)] for x in z]
    symbol_to_type = {s: i for i, s in enumerate(type_map)}
    atom_types = np.array([symbol_to_type[s] for s in symbols], dtype=int)

    sub_coords = coords[indices]
    sub_energies = energies[indices]
    sub_forces = forces[indices]

    nframes, natoms, _ = sub_coords.shape

    with open(os.path.join(out_dir, "type.raw"), "w") as f:
        f.write(" ".join(map(str, atom_types.tolist())) + "\n")

    with open(os.path.join(out_dir, "type_map.raw"), "w") as f:
        f.write("\n".join(type_map) + "\n")

    open(os.path.join(out_dir, "nopbc"), "w").close()

    # non-periodic molecule: generate dummy cubic box (20 A)
    box_size = 20.0
    box = np.tile(np.diag([box_size, box_size, box_size]), (nframes, 1))
    np.save(os.path.join(set_dir, "box.npy"), box.astype(np.float64))

    np.save(os.path.join(set_dir, "coord.npy"), sub_coords.reshape(nframes, natoms * 3))
    np.save(os.path.join(set_dir, "energy.npy"), sub_energies.reshape(nframes))
    np.save(os.path.join(set_dir, "force.npy"), sub_forces.reshape(nframes, natoms * 3))
    np.save(os.path.join(out_dir, "frame_indices.npy"), indices)

    print(out_dir)
    print("  frames:", nframes)
    print("  atoms:", natoms)
    print("  type_map:", type_map)
    print("  coord:", sub_coords.reshape(nframes, natoms * 3).shape)
    print("  energy:", sub_energies.reshape(nframes).shape)
    print("  force:", sub_forces.reshape(nframes, natoms * 3).shape)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--candidate", type=int, default=60000)
    parser.add_argument("--valid", type=int, default=5000)
    parser.add_argument("--test", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--type-map", nargs="+", default=["C", "H", "O"])
    args = parser.parse_args()

    data = np.load(args.input)
    z = data["nuclear_charges"].astype(int)
    coords = data["coords"].astype(np.float64)
    energies = data["energies"].astype(np.float64)
    forces = data["forces"].astype(np.float64)

    nframes = coords.shape[0]
    need = args.train + args.candidate + args.valid + args.test
    if need > nframes:
        raise ValueError(f"Need {need} frames, but only {nframes} available.")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(nframes)

    train_idx = perm[:args.train]
    candidate_idx = perm[args.train:args.train + args.candidate]
    valid_idx = perm[args.train + args.candidate:args.train + args.candidate + args.valid]
    test_idx = perm[args.train + args.candidate + args.valid:need]

    os.makedirs(args.output, exist_ok=True)

    split_info = {
        "input": args.input,
        "seed": args.seed,
        "type_map": args.type_map,
        "train": args.train,
        "candidate": args.candidate,
        "valid": args.valid,
        "test": args.test,
        "nframes_total": int(nframes),
    }

    with open(os.path.join(args.output, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    write_deepmd_system(os.path.join(args.output, "train"), z, coords, energies, forces, train_idx, args.type_map)
    write_deepmd_system(os.path.join(args.output, "candidate"), z, coords, energies, forces, candidate_idx, args.type_map)
    write_deepmd_system(os.path.join(args.output, "valid"), z, coords, energies, forces, valid_idx, args.type_map)
    write_deepmd_system(os.path.join(args.output, "test"), z, coords, energies, forces, test_idx, args.type_map)

if __name__ == "__main__":
    main()
