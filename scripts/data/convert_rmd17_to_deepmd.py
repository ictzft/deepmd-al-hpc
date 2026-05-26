import argparse
import os
import numpy as np

Z_TO_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}

def convert(input_npz, output_dir, train_frames=None):
    data = np.load(input_npz)

    z = data["nuclear_charges"].astype(int)
    coords = data["coords"].astype(np.float64)
    energies = data["energies"].astype(np.float64)
    forces = data["forces"].astype(np.float64)

    nframes, natoms, _ = coords.shape

    if train_frames is not None:
        n = min(train_frames, nframes)
        coords = coords[:n]
        energies = energies[:n]
        forces = forces[:n]
        nframes = n

    os.makedirs(output_dir, exist_ok=True)
    set_dir = os.path.join(output_dir, "set.000")
    os.makedirs(set_dir, exist_ok=True)

    unique_z = []
    for item in z:
        if item not in unique_z:
            unique_z.append(item)

    type_map = [Z_TO_SYMBOL[int(item)] for item in unique_z]
    z_to_type = {item: i for i, item in enumerate(unique_z)}
    atom_types = np.array([z_to_type[item] for item in z], dtype=int)

    with open(os.path.join(output_dir, "type.raw"), "w") as f:
        f.write(" ".join(map(str, atom_types.tolist())) + "\n")

    with open(os.path.join(output_dir, "type_map.raw"), "w") as f:
        f.write("\n".join(type_map) + "\n")

    # non-periodic molecule
    open(os.path.join(output_dir, "nopbc"), "w").close()

    np.save(os.path.join(set_dir, "coord.npy"), coords.reshape(nframes, natoms * 3))
    np.save(os.path.join(set_dir, "energy.npy"), energies.reshape(nframes))
    np.save(os.path.join(set_dir, "force.npy"), forces.reshape(nframes, natoms * 3))

    print("Converted:", input_npz)
    print("Output:", output_dir)
    print("Frames:", nframes)
    print("Atoms:", natoms)
    print("Type map:", type_map)
    print("coord.npy:", coords.reshape(nframes, natoms * 3).shape)
    print("energy.npy:", energies.reshape(nframes).shape)
    print("force.npy:", forces.reshape(nframes, natoms * 3).shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    convert(args.input, args.output, args.frames)

