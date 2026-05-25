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


def select_uncertainty_diversity(
    scores: np.ndarray,
    coords: np.ndarray,
    k: int = 10,
    top_m: int = 30,
    descriptor: str = "pairwise-distance",
) -> np.ndarray:
    """Select top-K diverse frames from top-M high-uncertainty candidates.

    Algorithm:
      1. Sort by scores (descending), keep top-M.
      2. Compute a per-frame descriptor from coords.
      3. Farthest Point Sampling (FPS) on the descriptor space to pick k
         diverse frames.

    Parameters
    ----------
    scores: [n_frames] uncertainty scores (e.g. force_dev_max).
    coords: [n_frames, n_atoms * 3] flattened atomic coordinates.
    k: number of frames to select.
    top_m: number of high-uncertainty candidates to consider before FPS.
    descriptor: descriptor type. Currently only "pairwise-distance" is supported
                (flattened pairwise distance vector per frame).

    Returns
    -------
    selected_indices: [k] indices into the ORIGINAL candidate pool.
    """
    scores = np.asarray(scores)
    coords = np.asarray(coords)
    n_frames = len(scores)
    n_coords_per_frame = coords.shape[1]
    n_atoms = n_coords_per_frame // 3

    top_m = min(top_m, n_frames)
    k = min(k, top_m)

    # Step 1: top-M by uncertainty
    candidate_pool = np.argsort(scores)[::-1][:top_m]
    candidate_scores = scores[candidate_pool]
    candidate_coords = coords[candidate_pool]

    # Step 2: build descriptor
    if descriptor == "pairwise-distance":
        desc = _pairwise_distance_descriptor(candidate_coords, n_atoms)
    else:
        desc = _pairwise_distance_descriptor(candidate_coords, n_atoms)

    # Step 3: FPS
    selected_local = _farthest_point_sampling(desc, k)

    return candidate_pool[selected_local].astype(int)


def _pairwise_distance_descriptor(coords: np.ndarray, n_atoms: int) -> np.ndarray:
    """Compute flattened pairwise distance vector per frame."""
    n_frames = coords.shape[0]
    coords_reshaped = coords.reshape(n_frames, n_atoms, 3)
    desc_list = []
    for i in range(n_frames):
        diff = coords_reshaped[i, :, None, :] - coords_reshaped[i, None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        triu = dist[np.triu_indices(n_atoms, k=1)]
        desc_list.append(triu)
    return np.array(desc_list)


def _farthest_point_sampling(descriptors: np.ndarray, k: int) -> np.ndarray:
    """Select k diverse points via farthest point sampling in descriptor space."""
    n = len(descriptors)
    k = min(k, n)

    selected = [0]
    dist_to_selected = np.full(n, np.inf)

    for _ in range(1, k):
        last = selected[-1]
        d = np.sum((descriptors - descriptors[last]) ** 2, axis=1)
        dist_to_selected = np.minimum(dist_to_selected, d)
        next_idx = int(np.argmax(dist_to_selected))
        selected.append(next_idx)

    return np.array(selected, dtype=int)


def select_trust_level(
    scores: np.ndarray,
    k: int = 10,
    low_pct: float = 50.0,
    high_pct: float = 90.0,
    candidate_ratio: float = 0.8,
) -> dict:
    """DP-GEN-style trust-level selection prototype.

    Classifies frames into three regions based on force_dev_max percentiles:
      - accurate:  scores < low_threshold   → skip
      - candidate: low_threshold ≤ scores ≤ high_threshold → select from here
      - failed:    scores > high_threshold  → record but skip (or keep a few)

    Parameters
    ----------
    scores: [n_frames] uncertainty scores.
    k: target number of frames to select.
    low_pct: percentile for accurate/candidate boundary (default 50).
    high_pct: percentile for candidate/failed boundary (default 90).
    candidate_ratio: fraction of k to take from candidate region (default 0.8).
                     Remaining (1 - candidate_ratio) * k from failed region.

    Returns
    -------
    dict with keys:
      selected_indices, accurate_count, candidate_count, failed_count,
      low_threshold, high_threshold, n_from_candidate, n_from_failed,
      selected_force_dev_max
    """
    scores = np.asarray(scores)
    n_frames = len(scores)

    low_threshold = float(np.percentile(scores, low_pct))
    high_threshold = float(np.percentile(scores, high_pct))

    accurate_mask = scores < low_threshold
    candidate_mask = (scores >= low_threshold) & (scores <= high_threshold)
    failed_mask = scores > high_threshold

    accurate_idx = np.where(accurate_mask)[0]
    candidate_idx = np.where(candidate_mask)[0]
    failed_idx = np.where(failed_mask)[0]

    n_from_candidate = min(int(k * candidate_ratio), len(candidate_idx))
    n_from_failed = min(k - n_from_candidate, len(failed_idx))

    # Select top-uncertainty within each region
    if len(candidate_idx) > 0:
        cand_order = np.argsort(scores[candidate_idx])[::-1]
        selected_candidate = candidate_idx[cand_order[:n_from_candidate]]
    else:
        selected_candidate = np.array([], dtype=int)

    if len(failed_idx) > 0:
        fail_order = np.argsort(scores[failed_idx])[::-1]
        selected_failed = failed_idx[fail_order[:n_from_failed]]
    else:
        selected_failed = np.array([], dtype=int)

    # If still short, fill from remaining high-uncertainty
    selected = np.concatenate([selected_candidate, selected_failed])
    if len(selected) < k:
        remaining = np.setdiff1d(np.argsort(scores)[::-1], selected)
        needed = k - len(selected)
        selected = np.concatenate([selected, remaining[:needed]])

    selected = selected[:k]

    return {
        "selected_indices": selected.astype(int),
        "accurate_count": int(np.sum(accurate_mask)),
        "candidate_count": int(np.sum(candidate_mask)),
        "failed_count": int(np.sum(failed_mask)),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "n_from_candidate": len(selected_candidate),
        "n_from_failed": len(selected_failed),
        "selected_force_dev_max": scores[selected].tolist(),
    }


def select_by_strategy(
    scores: np.ndarray,
    k: int,
    strategy: str = "uncertainty",
    seed: int | None = 0,
    coords: np.ndarray | None = None,
    top_m: int = 30,
    descriptor: str = "pairwise-distance",
    low_pct: float = 50.0,
    high_pct: float = 90.0,
    candidate_ratio: float = 0.8,
):
    """根据指定策略选择构型。

    Supported strategies
    --------------------
    uncertainty / topk / top-k:
        按 scores 从大到小选择 top-K。

    random:
        随机选择 K 个构型。

    uncertainty-diversity / diversity:
        从 top-M 高不确定性候选中用 FPS 选择 K 个多样性构型。
        需要提供 coords 参数。

    trust-level / trust / dpgen:
        DP-GEN-style trust-level sampling。
        按分位数分为 accurate / candidate / failed 三区。
    """
    scores = np.asarray(scores)
    strategy = strategy.lower().replace("_", "-")

    if strategy in {"uncertainty", "topk", "top-k", "force-dev-max"}:
        return select_top_k(scores=scores, k=k)

    if strategy == "random":
        return select_random_k(n_items=len(scores), k=k, seed=seed)

    if strategy in {"uncertainty-diversity", "diversity"}:
        if coords is None:
            raise ValueError("uncertainty-diversity strategy requires coords.")
        return select_uncertainty_diversity(
            scores=scores, coords=coords, k=k, top_m=top_m, descriptor=descriptor
        )

    if strategy in {"trust-level", "trust", "dpgen"}:
        return select_trust_level(
            scores=scores, k=k, low_pct=low_pct, high_pct=high_pct,
            candidate_ratio=candidate_ratio,
        )

    raise ValueError(
        f"Unknown selection strategy: {strategy}. "
        "Supported: uncertainty, topk, random, uncertainty-diversity, trust-level."
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
