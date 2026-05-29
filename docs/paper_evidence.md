# 论文证据清单

本文档跟踪 `deepmd-al-hpc` 项目的当前证据和待完成实验。最后更新：2026-05-28。

---

## 1. 当前已支持的结论

1. 本仓库在 2×V100 上实现了一个可复现的 toy H2 offline active learning 原型。
2. 4-model committee training、prediction 和 model deviation 已端到端实现。
3. 基于 uncertainty 的 top-K selection 一致选中比 random sampling 更高不确定性的构型。
4. uncertainty branch 中，selected top-K `force_dev_max_mean` 在 Round 0–3 单调下降（0.441 → 0.170）。
5. Random sampling baseline 已大规模完成：3 seeds × 3 rounds = 9 次独立运行。
6. Multi-seed random mean ± std 已在 Round 001/002/003 上可用。
7. Uncertainty vs random 对比图已在所有轮次生成。
8. 2×V100 model-level parallel training 实现 ~1.97× 加速比。
9. 单模型训练 wall time ~10.9s（1000 steps）用于 toy H2 模型。
10. 四种 selection strategy 已实现并完成 full multi-seed multi-round 对比：uncertainty / random / diversity / trust_level 均已完成 seed0/seed1/seed2 Round 001–003（2026-05-25, 2×V100）。
11. 统一口径四策略对比表使用跨 seed mean ± std 和一致的 "remaining candidate-pool" 指标。
12. V100 training wall-time profiling：132 models, mean=11.0s, 2×V100 parallel ~22s/round。
13. 结构多样性分析：diversity（FPS）在 toy H2 上实现 3.1x 结构覆盖度提升（vs uncertainty top-K）。
14. rMD17 ethanol uncertainty branch Round 0–3 active learning 闭环完成；Force RMSE 在 validation（0.374→0.354）和 independent test（0.344→0.327 eV/Å）上均单调下降。
15. rMD17 ethanol random baseline（3 seeds × 3 rounds）：uncertainty Force RMSE 单调改善，random 方差大（Round 3: 0.354 vs 0.607 ± 0.385 eV/Å）。在该单体系实验中，uncertainty 表现出比 random 更稳定的改善趋势。
16. rMD17 ethanol 四策略对比：三种 active strategy（uncertainty/diversity/trust_level）Force RMSE 相似（0.354–0.362 eV/Å, 1σ 内），均低于 random（0.607 eV/Å）；但 random 跨 seed 方差大（std=0.683），不能声称严格统计显著性。与 toy H2 结论一致。
17. MD stability test（NVE 10K）：所有模型稳定，drift ~0.035 eV/ps；100K+ 解离表明当前 Force RMSE ~0.35 eV/Å 不足以支撑高温 MD。
18. 本仓库提供了文档化的脚本、配置和轻量摘要，用于复现主要实验工作流，前提是具备所需的本地数据集和 DeepMD-kit 环境。
19. rMD17 benzene uncertainty branch Round 000–003 完成（4 rounds × 4 models, top-1000 per round, 12 原子分子, 60000 帧候选池）。

---

## 2. 部分支持、仍需验证的结论

1. **Uncertainty sampling 比 random sampling 更有效地降低剩余候选池不确定性。**
   - *证据（toy H2）：* Round 001 剩余候选池对比中，uncertainty_round001（0.126）< 三个 random seed（0.355, 0.488, 0.332）。
   - *证据（rMD17）：* 四策略 Round 3 对比显示三种 active strategy（0.354–0.362）在 validation 和 independent test 上均低于 random（0.607 eV/Å）；但 random 跨 seed 方差大（std=0.683）。
   - *差距：* active strategy 之间的差异在两个数据集上均在 1σ 内；benzene baselines 待补充以进行多体系确认。

2. **Uncertainty-diversity sampling 在不严重降低模型质量的前提下改善结构覆盖度。**
   - *证据：* toy H2 上 multi-seed Round 001–003 显示 diversity F_RMSE 与 random 相当。rMD17 ethanol 上 diversity F_RMSE（0.3555）与 uncertainty（0.3537）竞争力相当。Selection-level 对比确认了更广的结构覆盖（toy H2 上 3.1x）。
   - *差距：* 结构多样性 descriptor 分析仅限于 toy H2 上的 H-H 距离；更丰富的 descriptor（SOAP, ACSF）尚未在真实分子体系上测试。

3. **DP-GEN-style trust-level sampling 在 committee model 框架中可行。**
   - *证据：* Trust-level 正确将 50 帧候选池划分为 25 accurate / 20 candidate / 5 failed。Multi-seed Round 001–003 F_RMSE（1.35e-01, 1.49e-01, 1.78e-01）具有竞争力。
   - *差距：* 仅使用 force_dev_max；完整 DP-GEN 使用 force 和 energy deviation。

---

## 3. 尚不支持的结论

1. 该方法在多个数据集或多轮实验中显著优于 random sampling（ethanol 证据可用；benzene uncertainty + random + independent test 已完成，diversity/trust_level 待补充）。
2. 该方法已在多个真实 DFT/AIMD 系统上验证（rMD17 ethanol 完全验证；rMD17 benzene uncertainty + random + independent test 已完成，diversity/trust_level 待补充）。
3. 该框架已展示 H100 或多节点 scaling。
4. 该框架已通过高温 MD stability 测试验证（10K NVE 稳定；100K+ 解离）。
5. 该框架已达到生产就绪或满足 CCF-B 投稿标准。
6. 该 active learning 工作流相比 random 或系统采样在真实体系上降低了 DFT labeling 总成本（offline AL 模拟 labeling；需要 online 验证）。

---

## 4. 当前限制

1. Toy H2 数据集（2 atoms, 250 frames）——不能代表真实材料体系。
2. Toy H2：valid set 同时作为 candidate pool（无 independent test）。rMD17 ethanol：independent test 可用（10000 frames）。
3. rMD17 ethanol 四策略对比已完成；rMD17 benzene uncertainty + random + independent test 已完成，diversity/trust_level 待补充；多体系验证部分进行中。
4. Uncertainty-diversity（FPS）和 trust-level（DP-GEN-style）已在 toy H2 和 rMD17 ethanol 上实现并验证。
5. 无 H100 或多节点 scaling 实验。
6. MD stability 仅在 10K NVE 验证；100K+ 解离——高温 MD stability 尚未实现。
7. 全流程 GPU utilization 曲线未记录（有代表性样本）。
8. Profiling 覆盖 124 models（所有策略）；GPU utilization 曲线和分阶段 I/O latency 待补充。

---

## 5. 下一步实验

1. **统一对比指标** — 已完成（aligned_comparison.csv 使用一致的 remaining candidate-pool 指标）。
2. **添加 GPU 监控曲线** — 在完整一轮中运行 nvidia-smi dmon（代表性样本已完成，完整曲线待补充）。
3. **添加 uncertainty-diversity selection** — 已完成（FPS + pairwise-distance descriptor, 3.1x 结构覆盖度）。
4. **迁移到真实 DFT/AIMD 数据集** — rMD17 ethanol 四策略 multi-seed multi-round 已完成；independent test 已完成；MD stability 已完成。rMD17 benzene uncertainty + random + independent test 已完成。下一步：benzene diversity/trust_level baselines + MD stability，然后是更多体系。
5. **运行 H100 / 多 GPU scaling** — 测试训练 throughput 和端到端 round time。
6. **MD stability 测试** — 10K NVE 已完成（稳定, drift ~0.035 eV/ps）；高温（>100K）MD stability 待补充。
