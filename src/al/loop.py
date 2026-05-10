import numpy as np

from src.metrics.deviation import force_model_deviation, combined_deviation
from src.al.selector import select_top_k


def run_one_offline_al_round(
    committee_forces: np.ndarray,
    select_k: int,
) -> dict:
    """
    运行一轮简化版 offline active learning。

    当前阶段不真正调用 DeepMD-kit，只输入 committee forces，
    用于检查主动学习框架是否跑通。

    Parameters
    ----------
    committee_forces:
        shape = [n_models, n_frames, n_atoms, 3]

    select_k:
        本轮选择多少个候选构型加入训练集。

    Returns
    -------
    result:
        包含 force deviation、selected indices 和统计信息。
    """
    force_dev = force_model_deviation(committee_forces)
    score = combined_deviation(force_dev=force_dev)

    selected_indices = select_top_k(score, k=select_k)

    result = {
        "n_models": int(committee_forces.shape[0]),
        "n_frames": int(committee_forces.shape[1]),
        "n_atoms": int(committee_forces.shape[2]),
        "select_k": int(select_k),
        "max_score": float(np.max(score)),
        "min_score": float(np.min(score)),
        "mean_score": float(np.mean(score)),
        "selected_indices": selected_indices.tolist(),
        "selected_scores": score[selected_indices].tolist(),
    }

    return result
