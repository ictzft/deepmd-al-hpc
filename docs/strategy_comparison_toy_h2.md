# Toy H2 四策略对比

---

## 1. 为什么需要四策略对比

仅靠 random baseline 只能说明"uncertainty 优于 random"。更强的论文需要将 uncertainty 与其他合理 selection strategy 进行对比：

1. **Random** — 最简单的 baseline，验证 selection strategy 是否真的有效
2. **Uncertainty top-K** — 标准 committee-based active learning
3. **Uncertainty-diversity** — 减少高不确定性区域的结构冗余
4. **Trust-level（DP-GEN-style）** — 按 model deviation 阈值划分候选池

---

## 2. 策略描述

| 策略 | 算法 | 参数 |
|---|---|---|
| random | 均匀随机采样 | `seed` |
| uncertainty | 按 `force_dev_max` 降序排序，取 top-K | `top_k` |
| uncertainty-diversity | 按不确定性选 top-M → pairwise-distance descriptor → Farthest Point Sampling | `top_k`, `top_m`, `descriptor` |
| trust-level | 基于分位数的 accurate/candidate/failed 区域划分 | `top_k`, `low_pct`, `high_pct`, `candidate_ratio` |

详见 `docs/selection_strategies.md`。

---

## 3. 如何 dry-run

```bash
for strategy in random uncertainty uncertainty-diversity trust-level; do
  bash scripts/experiments/run_toy_h2_strategy_comparison.sh \
    --strategy $strategy \
    --start-round 1 --end-round 3 \
    --top-k 10 \
    --dry-run
done
```

dry-run 打印所有命令但不执行。用于在 V100 上运行前验证路径和参数。

---

## 4. 如何在 V100 上运行

1. 进入 DeepMD-kit Docker 容器
2. 验证数据路径存在（`data/toy_h2/train`, `data/toy_h2/valid`）
3. 去掉 `--dry-run` 执行：

```bash
# Docker 容器内：
bash scripts/experiments/run_toy_h2_strategy_comparison.sh \
  --strategy uncertainty-diversity \
  --start-round 1 --end-round 3 \
  --top-k 10

bash scripts/experiments/run_toy_h2_strategy_comparison.sh \
  --strategy trust-level \
  --start-round 1 --end-round 3 \
  --top-k 10
```

注意：random 和 uncertainty 策略的结果已在 `experiments/baselines/` 和 `experiments/exp_*/` 中。对比框架可以引用这些已有结果。

---

## 5. 如何汇总结果

```bash
python scripts/analysis/summarize_strategy_comparison.py
```

输出：
- `experiments/strategy_comparison_toy_h2/strategy_summary.csv`
- `experiments/strategy_comparison_toy_h2/strategy_summary.md`

---

## 6. Toy H2 的局限性

- 2 原子体系：结构多样性 descriptor 仅为 H-H bond length
- 小数据集（250 frames）：cross-model variance 大
- Valid set 同时作为 candidate pool：无 independent test set
- 结果表明工作流可行性，不代表真实材料体系性能
- 对比应视为原型框架，而非最终论文表格

---

## 7. 迁移到真实 DFT/AIMD 数据集

将此对比框架迁移到真实 DFT/AIMD 数据集的计划见 `docs/real_dataset_plan.md`，包括正确的 train/candidate/validation/test 划分。
