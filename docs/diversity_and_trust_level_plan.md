# Diversity 和 Trust-Level Baseline 计划

---

## 1. 状态

Diversity 和 trust-level 策略已**完全实现并验证**（multi-seed Round 001–003, 2026-05-25, 2×V100）。本文档保留作为原始计划和实现细节的参考。

- **Uncertainty-diversity**：减少 top-K uncertainty selection 中的结构冗余
- **Trust-level（DP-GEN-style）**：按 model deviation 将候选池划分为 accurate / candidate / failed 区域

---

## 2. 当前可用组件

| 组件 | 位置 | 状态 |
|---|---|---|
| Uncertainty-diversity selector | `src/al/selector.py:select_uncertainty_diversity` | 已实现 |
| Trust-level selector | `src/al/selector.py:select_trust_level` | 已实现 |
| Strategy CLI | `scripts/selection/select_by_strategy.py` | 已实现 |
| 支持 diversity/trust 的 Selector | `scripts/active_learning/select_from_predictions.py` | 已更新 |
| Diversity multi-seed Round 001–003 | `experiments/baselines/diversity_*` | 已完成（2026-05-25） |
| Trust-level multi-seed Round 001–003 | `experiments/baselines/trust_level_*` | 已完成（2026-05-25） |
| Selection-level 对比 | `experiments/baselines/strategy_comparison_round000.csv` | 已完成 |
| 统一口径四策略对比 | `experiments/baselines/aligned_comparison.csv` | 已完成（2026-05-25） |
| Diversity descriptor 分析 | `experiments/baselines/diversity_analysis.md` | 已完成（3.1× 覆盖度） |

---

## 3. Uncertainty-Diversity Selection

**算法**：按 force_dev_max 选 top-M → pairwise-distance descriptor → Farthest Point Sampling 选出 K 个多样性帧。

**默认参数**：top_k=10, top_m=30, descriptor=pairwise-distance。

**用法**：
```bash
python scripts/selection/select_by_strategy.py \
  --predictions <committee_predictions.npz> \
  --strategy uncertainty-diversity \
  --top-k 10 --top-m 30 \
  --output <output.json>
```

**Multi-seed Round 001–003 结果（3-seed mean ± std）**：

| Round | F_RMSE | F_RMSE std |
|---:|---:|---:|
| 1 | 2.052e-01 | 5.789e-02 |
| 2 | 1.738e-01 | 9.290e-03 |
| 3 | 1.759e-01 | 4.082e-02 |

完整对比见 `experiments/baselines/aligned_comparison.md`。

---

## 4. Trust-Level / DP-GEN-style Selection

**算法**：Force_dev_max 分位数划分：
- accurate（<P50）：跳过
- candidate（P50–P90）：从此区域选择（80%）
- failed（>P90）：少量保留（20%）

**默认参数**：low_pct=50, high_pct=90, candidate_ratio=0.8。

**用法**：
```bash
python scripts/selection/select_by_strategy.py \
  --predictions <committee_predictions.npz> \
  --strategy trust-level \
  --top-k 10 --low-pct 50 --high-pct 90 \
  --output <output.json>
```

**Multi-seed Round 001–003 结果（3-seed mean ± std）**：

| Round | F_RMSE | F_RMSE std |
|---:|---:|---:|
| 1 | 1.353e-01 | 2.761e-02 |
| 2 | 1.491e-01 | 2.256e-02 |
| 3 | 1.782e-01 | 6.470e-03 |

Accurate/Candidate/Failed 划分（初始 50 帧候选池）：25/20/5。

完整对比见 `experiments/baselines/aligned_comparison.md`。

---

## 5. Round 002–003 执行（已完成）

Diversity 和 trust-level 策略的 Round 002–003 已于 2026-05-25 完成（2×V100）。执行流程与 `scripts/run_random_baseline_round.sh` 相同：
1. 使用 `merge_selected_frames.py` + `make_remaining_candidate.py` 准备数据
2. 使用 `make_round_committee_configs.py` 生成配置
3. 使用 `train_round_committee_models.sh` 训练
4. 使用 `predict_committee_models.py` 预测
5. 使用 `select_by_strategy.py` 选择下一轮帧

命令模板见 `docs/random_baseline_execution_checklist.md`。

---

## 6. 生成的输出文件

```
experiments/baselines/diversity_round002_*/（训练模型 + 预测）— 已完成
experiments/baselines/diversity_round003_*/ — 已完成
experiments/baselines/trust_level_round002_*/ — 已完成
experiments/baselines/trust_level_round003_*/ — 已完成
experiments/baselines/aligned_comparison.csv（4 strategies × 3 rounds）— 已完成
```

---

## 7. 对比表设计

| 策略 | Round | E_RMSE | F_RMSE | Cand force_dev_max | Selected force_dev_max |
|---|---:|---:|---:|---:|---:|
| uncertainty | 1 | ... | ... | ... | ... |
| random (mean) | 1 | ... | ... | ... | ... |
| diversity | 1 | ... | ... | ... | ... |
| trust_level | 1 | ... | ... | ... | ... |

---

## 8. 限制

1. 仅在 toy H2（2 atoms）上测试——结构多样性 descriptor 仅为 H-H bond length。真实体系需要更丰富的 descriptor（SOAP, ACSF）。
2. Trust-level 仅使用 force_dev_max；完整 DP-GEN 使用 force 和 energy deviation。
3. Cross-model variance 限制了单次运行对比的可靠性。
4. 所有结论仍需在真实 DFT/AIMD 数据集上验证。
