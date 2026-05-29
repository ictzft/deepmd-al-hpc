# Random Baseline 执行清单

**状态：全部已完成（2026-05-25, 2×V100）。** 本文档记录已完成的操作和复现方式。

---

## 1. 已完成项

| Seed | Round 001 | Round 002 | Round 003 |
|---|---|---|---|
| seed0 | 已完成 | 已完成 | 已完成 |
| seed1 | 已完成 | 已完成 | 已完成 |
| seed2 | 已完成 | 已完成 | 已完成 |

每个 (seed, round) 运行包括：数据准备 → 4-model committee training → committee prediction → metrics summary。

---

## 2. 生成的摘要文件

| 文件 | 内容 |
|---|---|
| `experiments/baselines/random_round001_baseline_summary.csv` | seed0/1/2 Round 001 指标 + candidate uncertainty |
| `experiments/baselines/random_round002_baseline_summary.csv` | seed0/1/2 Round 002 指标 |
| `experiments/baselines/random_round003_baseline_summary.csv` | seed0/1/2 Round 003 指标 |
| `experiments/baselines/random_vs_uncertainty_summary.csv` | 16 行：uncertainty R0-3 + random R1-3（all seeds + mean） |

---

## 3. 生成的图表

```
experiments/figures/
  random_vs_uncertainty_force_rmse.svg
  random_vs_uncertainty_energy_rmse.svg
  random_vs_uncertainty_candidate_force_dev.svg
  random_vs_uncertainty_dataset_size.svg
```

---

## 4. 复现命令

### 快速模式（每个 seed/round 一条命令）

```bash
# DeepMD-kit Docker 容器内：
bash scripts/run_random_baseline_round.sh 002 seed0
bash scripts/run_random_baseline_round.sh 002 seed1
bash scripts/run_random_baseline_round.sh 002 seed2
bash scripts/run_random_baseline_round.sh 003 seed0
bash scripts/run_random_baseline_round.sh 003 seed1
bash scripts/run_random_baseline_round.sh 003 seed2
```

### 汇总结果

```bash
python scripts/analysis/summarize_random_round_baselines.py --round-id 2
python scripts/analysis/summarize_random_round_baselines.py --round-id 3
python scripts/analysis/summarize_random_vs_uncertainty.py
python scripts/analysis/plot_random_vs_uncertainty.py
```

---

## 5. 未被 Git 跟踪的文件

- `data/toy_h2/random_seed*_round_*_train/` — 训练数据（.npy arrays）
- `data/toy_h2/random_seed*_round_*_candidate/` — 候选数据
- `experiments/baselines/random_seed*_round*_committee_models/` — frozen models, logs, checkpoints
- `experiments/baselines/random_seed*_round*_committee_prediction/committee_predictions.npz` — 预测数组

---

## 6. 后续任务

见 `docs/random_baseline_next_steps.md` 了解 random baseline 完成后的工作。
