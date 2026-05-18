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
    scores = np.asarray(scores)

    if scores.ndim != 1:
        raise ValueError(f"scores should be 1D, got {scores.shape}")

    if k <= 0:
        raise ValueError("k should be positive")

    if len(scores) == 0:
        raise ValueError("scores should not be empty")

    k = min(k, len(scores))

    selected_indices = np.argsort(scores)[::-1][:k]
    return selected_indices.astype(int)


def select_random_k(n_items: int, k: int, seed: int | None = 0) -> np.ndarray:
    """
    随机选择 K 个候选构型。

    这个函数用于构建 Random Sampling baseline。
    为了保证实验可复现，默认使用固定随机种子。

    Parameters
    ----------
    n_items:
        候选池中的构型数量。

    k:
        选择的构型数量。

    seed:
        随机种子。不同 seed 可用于多次重复实验。

    Returns
    -------
    selected_indices:
        shape = [k]
        被随机选中的构型索引。
    """
    if n_items <= 0:
        raise ValueError("n_items should be positive")

    if k <= 0:
        raise ValueError("k should be positive")

    k = min(k, n_items)

    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(n_items, size=k, replace=False)
    return selected_indices.astype(int)


def select_by_strategy(
    scores: np.ndarray,
    k: int,
    strategy: str = "uncertainty",
    seed: int | None = 0,
) -> np.ndarray:
    """
    根据指定策略选择构型。

    Supported strategies
    --------------------
    uncertainty / topk / top-k:
        按 scores 从大到小选择 top-K。
        当前默认 scores 一般为 force_dev_max。

    random:
        随机选择 K 个构型，用于 random sampling baseline。
    """
    scores = np.asarray(scores)
    strategy = strategy.lower().replace("_", "-")

    if strategy in {"uncertainty", "topk", "top-k", "force-dev-max"}:
        return select_top_k(scores=scores, k=k)

    if strategy == "random":
        return select_random_k(n_items=len(scores), k=k, seed=seed)

    raise ValueError(
        f"Unknown selection strategy: {strategy}. "
        "Supported strategies: uncertainty, topk, top-k, random."
    )


def select_by_threshold(scores: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    根据分数区间筛选构型。

    这个函数后续可以模拟 DP-GEN 中 trust level 的思想：
    - 太低：模型已经确定，不选
    - 中等：有价值，选择
    - 太高：可能是异常结构，谨慎处理
    """
    scores = np.asarray(scores)

    if scores.ndim != 1:
        raise ValueError(f"scores should be 1D, got {scores.shape}")

    mask = (scores >= lower) & (scores <= upper)
    return np.where(mask)[0].astype(int)
