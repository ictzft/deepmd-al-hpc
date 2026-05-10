from dataclasses import dataclass


@dataclass
class ModelTask:
    model_id: int
    seed: int
    gpu_id: int


def build_committee_schedule(
    n_models: int = 4,
    n_gpus: int = 2,
    base_seed: int = 2026,
) -> list[list[ModelTask]]:
    """
    构建 committee models 在多 GPU 上的训练计划。

    在 2×V100 上，4 个模型分两批训练：

    batch 1:
      GPU 0 -> model 0
      GPU 1 -> model 1

    batch 2:
      GPU 0 -> model 2
      GPU 1 -> model 3
    """
    if n_models <= 0:
        raise ValueError("n_models should be positive")

    if n_gpus <= 0:
        raise ValueError("n_gpus should be positive")

    batches: list[list[ModelTask]] = []
    current_batch: list[ModelTask] = []

    for model_id in range(n_models):
        gpu_id = model_id % n_gpus
        task = ModelTask(
            model_id=model_id,
            seed=base_seed + model_id,
            gpu_id=gpu_id,
        )
        current_batch.append(task)

        if len(current_batch) == n_gpus:
            batches.append(current_batch)
            current_batch = []

    if current_batch:
        batches.append(current_batch)

    return batches
