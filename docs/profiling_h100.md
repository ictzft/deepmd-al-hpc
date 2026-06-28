# H100 / RTX 5090 Profiling 计划

本文档预留给未来的 H100 / RTX 5090 多 GPU scaling 实验。
**当前仓库中未报告任何 H100 结果。**

当前状态（2026-06-28）：
- V100 wall-clock profiling 已完成（见 `docs/profiling_v100.md`）。
- H100 实验尚未进行。
- RTX 5090 环境已就绪，实验待执行。

## RTX 5090 硬件环境

```text
GPU：8×NVIDIA GeForce RTX 5090 (32GB, Blackwell sm_120)
驱动：570.124.06，CUDA 12.8
镜像：deepmd-5090:latest（基于 nvcr.io/nvidia/pytorch:25.06-py3，CUDA 12.9 Minor Version Compat）
DeepMD-kit：v3.1.3 PyTorch 后端（从源码编译）
单卡 toy H2 baseline：0.039 s/batch（1000 steps, ~40s total）
```

## 计划工作

### RTX 5090（优先）

1. 单卡 V100 vs 5090 speedup 对比（toy H2 相同配置）。
2. 多 GPU scaling：1/2/4/8 GPU committee training throughput。
3. Bfloat16 / mixed-precision training（5090 原生支持 bf16，V100 不支持）。
4. Candidate prediction throughput vs batch size scaling。
5. 端到端 active learning round time 对比。
6. GPU utilization / memory 曲线（nvidia-smi dmon）。

### H100（待资源就绪）

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
