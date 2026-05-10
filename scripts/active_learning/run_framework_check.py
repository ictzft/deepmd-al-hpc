import argparse
import numpy as np

from src.al.loop import run_one_offline_al_round
from src.al.scheduler import build_committee_schedule
from src.utils.io import load_json, save_json
from src.utils.logger import log
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/active_learning/framework_check.json",
        help="Path to active learning framework check config.",
    )
    args = parser.parse_args()

    config = load_json(args.config)

    experiment_name = config["experiment_name"]
    n_models = int(config["n_models"])
    n_frames = int(config["n_frames"])
    n_atoms = int(config["n_atoms"])
    select_k = int(config["select_k"])
    seed = int(config["seed"])
    output_path = config["output_path"]

    log(f"Start framework check: {experiment_name}")
    log(f"n_models={n_models}, n_frames={n_frames}, n_atoms={n_atoms}, select_k={select_k}")

    set_seed(seed)

    # 构建 2×V100 上的 committee model 训练计划
    schedule = build_committee_schedule(
        n_models=n_models,
        n_gpus=2,
        base_seed=seed,
    )

    schedule_as_dict = [
        [
            {
                "model_id": task.model_id,
                "seed": task.seed,
                "gpu_id": task.gpu_id,
            }
            for task in batch
        ]
        for batch in schedule
    ]

    log(f"Committee training schedule: {schedule_as_dict}")

    # 使用随机数模拟 4 个 committee models 对候选构型的 force 预测
    # shape = [n_models, n_frames, n_atoms, 3]
    committee_forces = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=(n_models, n_frames, n_atoms, 3),
    )

    result = run_one_offline_al_round(
        committee_forces=committee_forces,
        select_k=select_k,
    )

    result["training_schedule"] = schedule_as_dict

    save_json(result, output_path)

    log(f"Framework check finished.")
    log(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
