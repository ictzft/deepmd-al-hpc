import numpy as np


def select_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """
    根据不确定性分数选择 top-K 构型。

    Parameters
    ----------
    scores:
        shape = [n_frames]
        每个候选构型的不确定性分数。

    k:
        选择的构型数量。

    Returns
    -------
    selected_indices:
        shape = [k]
        被选中的构型索引，按照分数从高到低排序。
    """
    if scores.ndim != 1:
        raise ValueError(f"scores should be 1D, got {scores.shape}")

    if k <= 0:
        raise ValueError("k should be positive")

    k = min(k, len(scores))

    selected_indices = np.argsort(scores)[::-1][:k]
    return selected_indices


def select_by_threshold(scores: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    根据分数区间筛选构型。

    这个函数后续可以模拟 DP-GEN 中 trust level 的思想：
    - 太低：模型已经确定，不选
    - 中等：有价值，选择
    - 太高：可能是异常结构，谨慎处理
    """
    if scores.ndim != 1:
        raise ValueError(f"scores should be 1D, got {scores.shape}")

    mask = (scores >= lower) & (scores <= upper)
    return np.where(mask)[0]
