"""Quick MD stability test for rMD17 ethanol models.
Runs short NVE at 10K to check if the model can maintain structural integrity.
"""

import json, sys, numpy as np
from pathlib import Path
from deepmd.infer import DeepPot

MODELS = {
    "uncertainty_round0": "experiments/rmd17_ethanol_round000_committee_models/model_002/frozen_model.pb",
    "uncertainty_round1": "experiments/rmd17_ethanol_round001_committee_models/model_003/frozen_model.pb",
    "uncertainty_round2": "experiments/rmd17_ethanol_round002_committee_models/model_000/frozen_model.pb",
    "uncertainty_round3": "experiments/rmd17_ethanol_round003_committee_models/model_000/frozen_model.pb",
    "random_round3": "experiments/baselines/random_seed0_round003_committee_models/model_001/frozen_model.pb",
}

FS_TO_AU = 1.0 / 0.024188842
ANG_TO_AU = 1.0 / 0.52917721
EV_TO_AU = 1.0 / 27.211386
AMU_TO_AU = 1822.888486
kb_au = 3.1668114e-6

# Load reference structure
data_dir = Path("data/rmd17/ethanol/valid")
coord = np.load(data_dir / "set.000/coord.npy")
box = np.load(data_dir / "set.000/box.npy")
atype = np.loadtxt(data_dir / "type.raw", dtype=np.int32).flatten()
type_map_raw = data_dir / "type_map.raw"
type_map = type_map_raw.read_text().strip().split() if type_map_raw.exists() else ["C", "H", "O"]

MASS_AMU = {"C": 12.0107, "H": 1.00794, "O": 15.999}

def run_md(model_path, label, timestep_fs=0.1, nsteps=2000, temp_init=10.0):
    dp = DeepPot(str(model_path))
    natoms = len(atype)
    masses_amu = np.array([MASS_AMU[type_map[t]] for t in atype])
    masses_au = masses_amu * AMU_TO_AU

    coord0 = coord[0].reshape(natoms, 3) / ANG_TO_AU
    dt_au = timestep_fs / FS_TO_AU

    # Initialize velocities
    np.random.seed(42)
    vel_au = np.array([np.random.normal(0, np.sqrt(kb_au * temp_init / m), 3) for m in masses_au])
    vel_au -= np.sum(vel_au * masses_au[:, None], axis=0) / np.sum(masses_au)

    energies_ev = np.zeros(nsteps)
    for step in range(nsteps):
        coord_flat = (coord0 * ANG_TO_AU).reshape(1, -1)
        e_pot, force, _ = dp.eval(coord_flat, box[0:1], atype)
        f_au = force.reshape(natoms, 3) * (EV_TO_AU / ANG_TO_AU)

        # Velocity Verlet
        vel_au += 0.5 * (f_au / masses_au[:, None]) * dt_au
        coord0 += vel_au * dt_au
        coord_flat = (coord0 * ANG_TO_AU).reshape(1, -1)
        e_pot, force, _ = dp.eval(coord_flat, box[0:1], atype)
        f_au = force.reshape(natoms, 3) * (EV_TO_AU / ANG_TO_AU)
        vel_au += 0.5 * (f_au / masses_au[:, None]) * dt_au

        e_pot_au = float(e_pot[0][0]) * EV_TO_AU
        e_kin_au = 0.5 * np.sum(masses_au[:, None] * vel_au**2)
        energies_ev[step] = (e_pot_au + e_kin_au) / EV_TO_AU

    # Check stability
    equil_start = nsteps // 2
    e_equil = energies_ev[equil_start:]
    drift = (e_equil[-1] - e_equil[0]) / (len(e_equil) * timestep_fs * 1e-3)  # eV/ps
    e_range = e_equil.max() - e_equil.min()
    total_range = energies_ev.max() - energies_ev.min()
    stable = e_range < 1.0 and abs(drift) < 1.0

    return {
        "label": label,
        "stable": stable,
        "drift_ev_per_ps": float(drift),
        "equil_e_range_ev": float(e_range),
        "total_e_range_ev": float(total_range),
        "e_first_ev": float(energies_ev[0]),
        "e_last_ev": float(energies_ev[-1]),
    }


results = []
for label, model_path in MODELS.items():
    r = run_md(model_path, label)
    r["stable"] = bool(r["stable"])
    results.append(r)
    status = "STABLE" if r["stable"] else "DISSOCIATED"
    print(f"{label:25s} {status:12s} drift={r['drift_ev_per_ps']:+.4f} eV/ps  E_range={r['equil_e_range_ev']:.4f} eV")

# Save
import json as json_mod
out_path = Path("experiments/rmd17_ethanol_summary/md_stability/md_summary.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(str(out_path), "w") as f:
    json_mod.dump(results, f, indent=2)

print("\nWrote md_summary.json")
