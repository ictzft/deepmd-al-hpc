# H100 Profiling 计划

本文档预留给未来的 H100 / 多 GPU scaling 实验。
**当前仓库中未报告任何 H100 结果。**

当前状态（2026-05-25）：
- V100 wall-clock profiling 已完成（见 `docs/profiling_v100.md`）。
- H100 实验尚未进行。

## 计划工作

1. 将已验证的 V100 流水线迁移到 H100。
2. 对比 V100 vs H100 单模型训练 wall-clock time。
3. 如有资源可用，评估 1/2/4/8 GPU scaling。
4. 测量 committee training throughput（models/hour）。
5. 测量 candidate prediction throughput（frames/second）。
6. 通过 nvidia-smi dmon 报告 GPU utilization 和 memory usage。
7. 对比端到端 active learning round time（V100 vs H100）。

## 前提条件

- 具有 DeepMD-kit Docker 环境的 H100 节点访问权限。
- 使用相同训练配置（现有 `configs/deepmd/` JSON 文件）以保证公平对比。
- 使用相同 toy H2 数据集进行受控基准测试。

## 与 V100 profiling 的关系

当前 V100 baseline 数据见 `docs/profiling_v100.md`，包括：
- 单模型训练 wall time：mean=10.9s（1000 steps, 8.7ms/batch）
- 2×V100 并行加速比：1.97×
- 代表性 GPU utilization 和 memory 数据

H100 profiling 应使用相同方法并报告相同指标，以便直接对比。
