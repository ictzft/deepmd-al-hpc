from typing import Optional

import numpy as np


def force_model_deviation(forces: np.ndarray) -> np.ndarray:
    """
    计算 committee models 的 force model deviation。

    Parameters
    ----------
    forces:
        shape = [n_models, n_frames, n_atoms, 3]

    Returns
    -------
    frame_deviation:
        shape = [n_frames]
    """
    if forces.ndim != 4:
        raise ValueError(
            f"forces should have shape [n_models, n_frames, n_atoms, 3], got {forces.shape}"
        )

    force_std = np.std(forces, axis=0)
    atom_std_norm = np.linalg.norm(force_std, axis=-1)
    frame_deviation = np.max(atom_std_norm, axis=-1)

    return frame_deviation


def energy_model_deviation(energies: np.ndarray) -> np.ndarray:
    """
    计算 energy model deviation。

    Parameters
    ----------
    energies:
        shape = [n_models, n_frames]

    Returns
    -------
    deviation:
        shape = [n_frames]
    """
    if energies.ndim != 2:
        raise ValueError(
            f"energies should have shape [n_models, n_frames], got {energies.shape}"
        )

    return np.std(energies, axis=0)


def combined_deviation(
    force_dev: np.ndarray,
    energy_dev: Optional[np.ndarray] = None,
    force_weight: float = 1.0,
    energy_weight: float = 0.0,
) -> np.ndarray:
    """
    计算组合不确定性分数。

    当前阶段以 force deviation 为主。
    后续可以加入 energy deviation、virial deviation、结构多样性等。
    """
    score = force_weight * force_dev

    if energy_dev is not None:
        score = score + energy_weight * energy_dev

    return score
