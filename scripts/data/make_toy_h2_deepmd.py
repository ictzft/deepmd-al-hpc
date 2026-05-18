import argparse
from pathlib import Path

import numpy as np


def generate_h2_dataset(
    output_dir: Path,
    n_frames: int,
    seed: int,
    r0: float = 0.74,
    k: float = 10.0,
    box_size: float = 10.0,
) -> None:
    """
    生成一个最小 H2 toy DeepMD 数据集。

    说明：
    - 体系包含 2 个 H 原子；
    - 能量使用简单谐振子势 E = 0.5 * k * (r - r0)^2；
    - 力由该势能对坐标求导得到；
    - 该数据只用于验证 DeePMD 训练流程，不用于真实科学结论。
    """
    rng = np.random.default_rng(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    set_dir = output_dir / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)

    natoms = 2

    coords = np.zeros((n_frames, natoms * 3), dtype=np.float64)
    forces = np.zeros((n_frames, natoms * 3), dtype=np.float64)
    energies = np.zeros((n_frames,), dtype=np.float64)
    boxes = np.zeros((n_frames, 9), dtype=np.float64)

    for i in range(n_frames):
        # H2 分子中心放在盒子中心附近
        center = np.array([box_size / 2, box_size / 2, box_size / 2])

        # 键长在 r0 附近扰动
        r = r0 + rng.normal(0.0, 0.08)
        r = max(0.45, min(1.10, r))

        # 给键方向加一点随机性
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)

        pos0 = center - 0.5 * r * direction
        pos1 = center + 0.5 * r * direction

        d = pos1 - pos0
        dist = np.linalg.norm(d)
        unit = d / dist

        energy = 0.5 * k * (dist - r0) ** 2

        # F1 = -dE/dR1, F2 = -dE/dR2
        force_on_1 = k * (dist - r0) * unit
        force_on_2 = -force_on_1

        coords[i, :] = np.concatenate([pos0, pos1])
        forces[i, :] = np.concatenate([force_on_1, force_on_2])
        energies[i] = energy
        boxes[i, :] = np.array(
            [
                box_size, 0.0, 0.0,
                0.0, box_size, 0.0,
                0.0, 0.0, box_size,
            ],
            dtype=np.float64,
        )

    # DeepMD raw files
    # type.raw: 每个原子的类型编号
    # type_map.raw: 类型编号对应元素名称
    (output_dir / "type.raw").write_text("0 0\n", encoding="utf-8")
    (output_dir / "type_map.raw").write_text("H\n", encoding="utf-8")

    np.save(set_dir / "coord.npy", coords)
    np.save(set_dir / "force.npy", forces)
    np.save(set_dir / "energy.npy", energies)
    np.save(set_dir / "box.npy", boxes)

    print(f"Generated {n_frames} frames in {output_dir}")
    print(f"coord shape:  {coords.shape}")
    print(f"force shape:  {forces.shape}")
    print(f"energy shape: {energies.shape}")
    print(f"box shape:    {boxes.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    generate_h2_dataset(
        output_dir=Path(args.output),
        n_frames=args.n_frames,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
