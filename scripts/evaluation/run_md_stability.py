#!/usr/bin/env python3
"""NVE MD stability validation for rMD17 ethanol models using DeepPot Python API.

Runs Velocity Verlet NVE MD from a validation-set starting structure,
tracks energy conservation and temperature stability across rounds.
No LAMMPS required — uses deepmd.infer.DeepPot directly.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Constants
FS_TO_AU = 1.0 / 0.024188842  # femtosecond to atomic time unit
ANG_TO_AU = 1.0 / 0.52917721   # Angstrom to Bohr (atomic unit)
EV_TO_AU = 1.0 / 27.211386     # eV to Hartree
AMU_TO_AU = 1822.888486         # amu to electron mass

# Atomic masses for ethanol (C2H5OH) in amu
# type_map from data: C, H, O
MASSES_AMU = np.array([12.0107, 1.00794, 15.999])  # C, H, O


def load_system(system_dir: Path):
    """Load DeePMD-format system: coords, box, forces, energies, type_map, atype."""
    set_dirs = sorted(p for p in system_dir.iterdir()
                      if p.is_dir() and p.name.startswith("set."))
    if not set_dirs:
        raise FileNotFoundError(f"No set.* in {system_dir}")

    set_dir = set_dirs[0]  # Use first set
    coord = np.load(set_dir / "coord.npy")      # (nframes, natoms*3)
    box = np.load(set_dir / "box.npy")           # (nframes, 9)

    # type.raw: list of atom type indices
    type_path = system_dir / "type.raw"
    atype = np.loadtxt(type_path, dtype=np.int32).flatten()

    # type_map.raw: list of element symbols
    type_map_path = system_dir / "type_map.raw"
    if type_map_path.exists():
        type_map = type_map_path.read_text().strip().split()
    else:
        type_map = ["C", "H", "O"]

    return coord, box, atype, type_map


def build_mass_array(atype, type_map):
    """Build mass array (in amu) from atom types and type_map."""
    masses = np.zeros(len(atype))
    for i, t in enumerate(atype):
        elem = type_map[t] if t < len(type_map) else type_map[0]
        if elem == "C":
            masses[i] = 12.0107
        elif elem == "O":
            masses[i] = 15.999
        else:  # H
            masses[i] = 1.00794
    return masses


def velocity_verlet_nve(deep_pot, coords0, box, atype, masses_amu, timestep_fs, nsteps):
    """Run NVE MD with Velocity Verlet integrator.

    Args:
        deep_pot: DeepPot instance
        coords0: initial coordinates (natoms, 3) in Angstrom
        box: cell vectors (3, 3) in Angstrom
        atype: atom type indices (natoms,)
        masses_amu: masses in amu (natoms,)
        timestep_fs: timestep in femtoseconds
        nsteps: number of MD steps

    Returns:
        Dict with trajectory data and stability metrics.
    """
    dt_au = timestep_fs / FS_TO_AU  # timestep in atomic units
    masses_au = masses_amu * AMU_TO_AU  # masses in electron mass units
    natoms = len(atype)

    # Convert to atomic units for integration
    coords = coords0.copy() / ANG_TO_AU  # to Bohr
    box_au = box.reshape(3, 3) / ANG_TO_AU

    # Get initial forces and energy (DeepPot returns (1,1) for energy, (1,natoms,3) for forces)
    coord_flat = coords0.reshape(1, -1)
    cell = box.copy()
    e0, f0, _ = deep_pot.eval(coord_flat, cell, atype)
    forces = f0.reshape(natoms, 3)  # (natoms, 3) in eV/Ang

    # Initialize velocities from Maxwell-Boltzmann at low temperature for stability
    kb_au = 3.1668114e-6  # Boltzmann constant in Hartree/K
    temp_init = 100.0  # K — start cold, let system thermalize
    np.random.seed(42)
    vel_au = np.zeros((natoms, 3))
    for i in range(natoms):
        sigma = np.sqrt(kb_au * temp_init / masses_au[i])
        vel_au[i] = np.random.normal(0, sigma, 3)
    # Remove center-of-mass velocity
    v_com = np.sum(vel_au * masses_au[:, None], axis=0) / np.sum(masses_au)
    vel_au -= v_com

    # Store trajectory data
    times_fs = np.zeros(nsteps + 1)
    energies_total = np.zeros(nsteps + 1)
    energies_pot = np.zeros(nsteps + 1)
    energies_kin = np.zeros(nsteps + 1)
    temperatures = np.zeros(nsteps + 1)

    # Initial state
    coord_flat = coords.reshape(1, -1) * ANG_TO_AU
    cell_flat = box.copy()
    e_pot, f, _ = deep_pot.eval(coord_flat, cell_flat, atype)
    forces = f.reshape(natoms, 3)
    forces_au = forces * (EV_TO_AU / ANG_TO_AU)

    e_pot_au = float(e_pot[0][0]) * EV_TO_AU
    e_kin_au = 0.0
    for i in range(natoms):
        e_kin_au += 0.5 * masses_au[i] * np.sum(vel_au[i]**2)

    energies_total[0] = e_pot_au + e_kin_au
    energies_pot[0] = e_pot_au
    energies_kin[0] = e_kin_au
    dof = 3 * natoms - 6
    temperatures[0] = 2.0 * e_kin_au / (dof * kb_au) if dof > 0 else 0.0
    times_fs[0] = 0.0

    for step in range(1, nsteps + 1):
        # Velocity Verlet half-step: v += 0.5 * a * dt
        for i in range(natoms):
            vel_au[i] += 0.5 * (forces_au[i] / masses_au[i]) * dt_au

        # Position update: r += v * dt
        coords += vel_au * dt_au

        # Get new forces
        coord_flat = coords.reshape(1, -1) * ANG_TO_AU
        e_pot, f, _ = deep_pot.eval(coord_flat, cell_flat, atype)
        forces = f.reshape(natoms, 3)
        forces_au = forces * (EV_TO_AU / ANG_TO_AU)

        # Velocity Verlet second half-step: v += 0.5 * a * dt
        for i in range(natoms):
            vel_au[i] += 0.5 * (forces_au[i] / masses_au[i]) * dt_au

        # Compute energies
        e_pot_au = float(e_pot[0][0]) * EV_TO_AU
        e_kin_au = 0.0
        for i in range(natoms):
            e_kin_au += 0.5 * masses_au[i] * np.sum(vel_au[i]**2)

        energies_total[step] = e_pot_au + e_kin_au
        energies_pot[step] = e_pot_au
        energies_kin[step] = e_kin_au
        times_fs[step] = step * timestep_fs

        # Temperature: k_B * T * (3N-6)/2 = E_kin (nonlinear molecule)
        dof = 3 * natoms - 6  # nonlinear molecule
        if dof > 0:
            temperatures[step] = 2.0 * e_kin_au / (dof * kb_au)

    # Compute stability metrics over equilibrated portion (last 50% of trajectory)
    e_total_ev_all = energies_total / EV_TO_AU
    equil_start = nsteps // 2
    e_total_ev = e_total_ev_all[equil_start:]

    # Energy drift: linear fit over equilibrated portion
    t = np.arange(len(e_total_ev))
    A = np.vstack([t, np.ones(len(e_total_ev))]).T
    slope, _ = np.linalg.lstsq(A, e_total_ev, rcond=None)[0]
    drift_per_step = slope  # eV per step
    drift_per_ps = drift_per_step / (timestep_fs * 1e-3)  # eV per ps
    drift_per_atom_per_ps = drift_per_ps / natoms

    # Check for explosion (catastrophic energy divergence)
    equil_e_range = e_total_ev.max() - e_total_ev.min()
    total_e_range = e_total_ev_all.max() - e_total_ev_all.min()
    exploded = equil_e_range > 5.0  # More than 5 eV drift in equilibrated portion

    # Temperature over equilibrated portion
    temp_equil = temperatures[equil_start:]

    return {
        "times_fs": times_fs.tolist(),
        "energy_total_ev": e_total_ev_all.tolist(),
        "energy_pot_ev": (energies_pot / EV_TO_AU).tolist(),
        "energy_kin_ev": (energies_kin / EV_TO_AU).tolist(),
        "temperatures_k": temperatures.tolist(),
        "energy_drift_per_atom_per_ps_ev": float(drift_per_atom_per_ps),
        "energy_drift_per_ps_ev": float(drift_per_ps),
        "exploded": bool(exploded),
        "total_energy_range_ev": float(total_e_range),
        "equilibrated_energy_range_ev": float(equil_e_range),
        "temperature_mean_k": float(np.mean(temp_equil)),
        "temperature_std_k": float(np.std(temp_equil)),
        "initial_temperature_k": float(temperatures[20]),
    }


def main():
    parser = argparse.ArgumentParser(description="NVE MD stability validation")
    parser.add_argument("--model", required=True, help="Path to frozen_model.pb")
    parser.add_argument("--structure", required=True, help="DeePMD system dir for initial structure")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--steps", type=int, default=5000, help="Number of MD steps")
    parser.add_argument("--timestep", type=float, default=0.5, help="Timestep in fs")
    parser.add_argument("--frame", type=int, default=0, help="Frame index from structure")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from deepmd.infer import DeepPot

    # Load model and structure
    dp = DeepPot(str(model_path))
    coords, box, atype, type_map = load_system(Path(args.structure))
    masses = build_mass_array(atype, type_map)

    frame = min(args.frame, coords.shape[0] - 1)
    coord0 = coords[frame].reshape(-1, 3)
    box0 = box[frame]

    print(f"Model: {model_path}")
    print(f"Natoms: {len(atype)}, Type map: {type_map}")
    print(f"Steps: {args.steps}, Timestep: {args.timestep} fs")
    print(f"Simulation time: {args.steps * args.timestep / 1000:.1f} ps")

    result = velocity_verlet_nve(dp, coord0, box0, atype, masses,
                                 args.timestep, args.steps)

    result["model"] = str(model_path)
    result["steps"] = args.steps
    result["timestep_fs"] = args.timestep
    result["natoms"] = int(len(atype))
    result["exploded"] = bool(result["exploded"])

    with output_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"Status: {'EXPLODED' if result['exploded'] else 'stable'}")
    print(f"Energy drift (equil): {result['energy_drift_per_atom_per_ps_ev']:.6e} eV/atom/ps")
    print(f"Temperature (equil): {result['temperature_mean_k']:.1f} ± {result['temperature_std_k']:.1f} K")
    print(f"Total E range: {result['total_energy_range_ev']:.4f} eV")
    print(f"Equil E range: {result['equilibrated_energy_range_ev']:.4f} eV")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
