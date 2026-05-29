# Random Baseline 完成后的下一步（更新于 2026-05-28）

> toy H2 和 rMD17 ethanol 上的 random baseline 及四策略工作均已完成。本文档描述剩余工作。

---

## 已完成（toy H2）

- Random baseline seed0/seed1/seed2 Round 001–003（2026-05-25 完成）
- Multi-seed random mean ± std（Round 001/002/003）
- Uncertainty-diversity Round 002–003（已完成）
- Trust-level Round 002–003（已完成）
- 统一口径四策略对比（已完成）
- 系统化端到端 V100 profiling（132 models, unified CSV）（已完成）

## 已完成（rMD17 ethanol）

- Uncertainty branch Round 0–3（已完成）
- Random baseline 3 seeds × 3 rounds（已完成）
- Diversity / trust_level baselines（已完成）
- Four-strategy comparison（已完成）
- Independent test evaluation（已完成）
- MD stability 10K/100K NVE（已完成）
- Pipeline profiling（52 models, all stages）（已完成）

## 待完成

1. rMD17 benzene：diversity / trust_level baselines + MD stability
2. 全流程 GPU utilization/memory 曲线（nvidia-smi dmon）
3. H100 / 多 GPU scaling 实验
4. rMD17 之外的更多分子或材料体系
