# Active Learning Selection Strategies

本文档说明 `deepmd-al-hpc` 中已实现的候选构型选择策略。所有策略均基于 committee model prediction 的输出（`force_dev_max`、`force_dev_mean`、`energy_dev`）进行选择。

当前所有实验均在 toy H2 原型数据集上完成，不代表真实材料体系。

---

## 1. Random Sampling

**策略**：从 candidate pool 中随机选择 K 个构型。

**参数**：`--seed` 控制随机种子。

**用途**：作为最简单的 baseline，验证其他 selection strategy 是否优于随机。

**当前状态**：已完成 seed0/seed1/seed2 Round 001–003 full retraining baseline。

**输入**：无（仅需要 candidate pool 大小）

**输出**：随机选中的 frame indices

---

## 2. Uncertainty Top-K

**策略**：按 `force_dev_max` 从高到低排序，选择 top-K。

**参数**：`--top-k` 控制选择数量。

**用途**：选择模型最不确定的构型（高 force model deviation），这些构型加入训练集后最有可能提高模型在欠拟合区域的精度。

**当前状态**：已完成 uncertainty branch Round 0–3 full retraining。

**输入**：`force_dev_max` 数组

**输出**：top-K frame indices

---

## 3. Uncertainty-Diversity Sampling

**策略**：两步法 — 先按 `force_dev_max` 选 top-M 个高不确定性候选，再对这些候选的 pairwise-distance descriptor 做 Farthest Point Sampling (FPS) 选出 K 个多样性构型。

**参数**：
- `--top-k`：最终选择数量（默认 10）
- `--top-m`：高不确定性候选池大小（默认 30）
- `--descriptor`：结构 descriptor 类型（当前仅 `pairwise-distance`）

**设计动机**：pure uncertainty top-K 容易选出结构相似的构型（例如 H2 中 bond length 相近的帧），导致信息冗余。uncertainty-diversity 在高不确定性候选中增加结构覆盖度。

**当前状态**：策略已实现，Round 000 selection-level 验证已完成，Round 001 retraining 已完成（单次 run）。

**输入**：`force_dev_max` + `coord` (from committee_predictions.npz)

**输出**：K 个兼顾高不确定性和结构多样性的 frame indices

**Round 001 结果（toy H2）**：

```text
E_RMSE mean = 0.877 eV
F_RMSE mean = 0.269 eV/Å
Remaining candidate force_dev_max mean = 0.218
```

---

## 4. DP-GEN-Style Trust-Level Sampling

**策略**：基于 `force_dev_max` 分位数将候选池分为三个区域：

- **accurate**：`force_dev_max < low_threshold` → 模型已准确，不选
- **candidate**：`low_threshold ≤ force_dev_max ≤ high_threshold` → 优先选
- **failed**：`force_dev_max > high_threshold` → 可能是异常构型，少量保留

默认阈值：`low_threshold = P50`，`high_threshold = P90`。

**参数**：
- `--top-k`：目标选择数量（默认 10）
- `--low-pct`：accurate/candidate 边界分位数（默认 50）
- `--high-pct`：candidate/failed 边界分位数（默认 90）
- `--candidate-ratio`：从 candidate 区域选择的比例（默认 0.8）

**设计动机**：DP-GEN 中使用模型偏差的 trust level 来确定哪些构型需要重新标注。本实现为简化原型，只使用 `force_dev_max` 一个指标（完整 DP-GEN 使用 `max_devi_f` 和 `max_devi_e`）。

**当前状态**：策略已实现，Round 000 selection-level 验证已完成，Round 001 retraining 已完成（单次 run）。

**输入**：`force_dev_max` 数组

**输出**：selected indices + 三区统计信息

**Round 001 结果（toy H2）**：

```text
E_RMSE mean = 0.549 eV
F_RMSE mean = 0.136 eV/Å
Remaining candidate force_dev_max mean = 0.160
Accurate/Candidate/Failed split: 25/20/5 (from 50-frame initial pool)
```

---

## 5. Strategy Comparison (Round 001 retraining)

| Strategy | Energy RMSE / eV | Force RMSE / eV/Å | Cand force_dev_max mean |
|---|---:|---:|---:|
| uncertainty | 0.729 | 0.162 | 0.126 |
| random (mean of 3) | 0.456 | 0.211 | 0.392 |
| diversity | 0.877 | 0.269 | 0.218 |
| trust_level | 0.549 | 0.136 | 0.160 |

**注意**：
- Random baseline 是 3-seed mean ± std，其他策略是单次 run
- Toy H2 数据规模小，cross-model variance 大，单次 run 结果不具统计意义
- Trust level Round 001 在 Energy RMSE 和 Force RMSE 上都较低，但这可能是特定 committee model 初始化的随机效应
- 完整结论需要 multi-seed multi-round retraining + 真实数据集

---

## 6. 如何运行

### 6.1 Selection-level（只选择，不训练）

```bash
# 从已有 committee prediction 选择
python scripts/selection/select_by_strategy.py \
  --predictions experiments/exp_005_committee_prediction/committee_predictions.npz \
  --strategy uncertainty-diversity \
  --top-k 10 --top-m 30 \
  --output experiments/baselines/diversity_round000/selected_topk.json

python scripts/selection/select_by_strategy.py \
  --predictions experiments/exp_005_committee_prediction/committee_predictions.npz \
  --strategy trust-level \
  --top-k 10 --low-pct 50 --high-pct 90 \
  --output experiments/baselines/trust_level_round000/selected_topk.json
```

### 6.2 完整 Round（数据准备 + 训练 + 预测）

参见 `scripts/run_random_baseline_round.sh` 的命令模式，替换 strategy 相关参数。

---

## 7. 当前限制

1. All strategies tested only on toy H2 (2 atoms, 250 frames).
2. Diversity descriptor is limited to pairwise-distance; more sophisticated descriptors (SOAP, ACSF) are not yet implemented.
3. Trust-level uses only `force_dev_max`; full DP-GEN uses both force and energy deviation.
4. New strategies have only single-run Round 001 results; multi-seed multi-round retraining is pending.
5. Cross-model variance in committee models limits the reliability of per-run comparisons.
