# Random Baseline 执行记录

> 本文档原为 random baseline 后续实验计划。截至 2026-05-25，Round 002/003 已在 2×V100 上全部执行完毕。本文档现作为历史执行记录保留。

**当前状态：所有计划实验已完成。** 执行命令清单见 `docs/random_baseline_execution_checklist.md`（已全部标记 done）。

---

## 1. 当前已完成内容

### 1.1 Selection-level baseline

```text
Round 0: random seed0 / seed1 / seed2 selection JSON 已生成
Round 1: random seed0 / seed1 / seed2 selection JSON 已生成
汇总:    experiments/baselines/selection_baseline_summary.csv / .md
```

Selection-level baseline 只比较不同 selection strategy 选出的构型不确定性特征，不涉及 retraining。

结果：uncertainty top-K 选出的构型平均 `force_dev_max` 高于 random sampling 选出的构型。

### 1.2 Random Round 001 retraining baseline

| Seed | Status | Metrics | Prediction |
|---|---|---|---|
| seed0 | 已完成 | `random_seed0_round001_metrics_summary.csv` | `random_seed0_round001_prediction_summary.csv` |
| seed1 | 已完成 | `random_seed1_round001_metrics_summary.csv` | `random_seed1_round001_prediction_summary.csv` |
| seed2 | 已完成 | `random_seed2_round001_metrics_summary.csv` | `random_seed2_round001_prediction_summary.csv` |

三 seed 汇总：

```text
experiments/baselines/random_round001_baseline_summary.csv
experiments/baselines/random_round001_baseline_summary.md
experiments/baselines/random_round001_comparison.csv
experiments/baselines/random_round001_comparison.md
```

### 1.3 Uncertainty vs random 初步对比

```text
experiments/baselines/random_vs_uncertainty_summary.csv
experiments/baselines/random_vs_uncertainty_summary.md
experiments/figures/random_vs_uncertainty_force_rmse.svg
experiments/figures/random_vs_uncertainty_energy_rmse.svg
experiments/figures/random_vs_uncertainty_candidate_force_dev.svg
experiments/figures/random_vs_uncertainty_dataset_size.svg
```

说明：当前对比图中 random baseline 只有 Round 001 一个数据点。完整 learning curve 需要 Round 002/003 数据。

---

## 2. 当前缺失内容

### 2.1 Random Round 002 retraining（未执行）

需要完成以下三个 seed 的 Round 002：

```text
random seed0 Round 002: train → config → committee models → prediction → summary
random seed1 Round 002: train → config → committee models → prediction → summary
random seed2 Round 002: train → config → committee models → prediction → summary
```

### 2.2 Random Round 003 retraining（未执行）

依赖 Round 002 完成后执行：

```text
random seed0 Round 003: train → config → committee models → prediction → summary
random seed1 Round 003: train → config → committee models → prediction → summary
random seed2 Round 003: train → config → committee models → prediction → summary
```

### 2.3 完整 multi-round 汇总（未完成）

所有 round 完成后需要更新：

```text
experiments/baselines/random_vs_uncertainty_summary.csv  → 增加 Round 002/003 行
experiments/figures/random_vs_uncertainty_*.svg           → 增加 Round 1/2/3 数据点
```

---

## 3. 为什么需要补 Round 002 和 Round 003

1. **单个 round 不能构成 learning curve**：当前只有 Round 001 一个数据点，无法观察 random baseline 的 Force RMSE 和 candidate deviation 随轮次的变化趋势。
2. **多 seed 需要多轮才能体现统计意义**：seed0/seed1/seed2 在 Round 001 的 Energy RMSE 方差较大（0.249–0.691 eV），需要更多轮次来观察方差是否收敛。
3. **与 uncertainty branch 的可比性**：uncertainty branch 有 Round 0–3 四个数据点，random baseline 至少需要 Round 001–003 三个数据点才能进行有意义的 learning curve 对比。
4. **论文级别证据链**：CCF-B 级别的主动学习论文通常要求 multi-round multi-seed baseline comparison。

---

## 4. 目录组织规范

### 4.1 数据目录

```text
data/toy_h2/
  random_seed0_round_001_train/       # 210 frames（已完成）
  random_seed0_round_001_candidate/   # 40 frames （已完成）
  random_seed0_round_002_train/       # 220 frames（待生成）
  random_seed0_round_002_candidate/   # 30 frames （待生成）
  random_seed0_round_003_train/       # 230 frames（待生成）
  random_seed0_round_003_candidate/   # 20 frames （待生成）
  random_seed1_round_001_train/       # 同上（已完成）
  ...
  random_seed2_round_003_candidate/   # 同上（待生成）
```

### 4.2 配置目录

```text
configs/deepmd/
  random_seed0_round_001_committee/   # base_seed=1101（已完成）
  random_seed0_round_002_committee/   # base_seed=1201（待生成）
  random_seed0_round_003_committee/   # base_seed=1301（待生成）
  random_seed1_round_001_committee/   # base_seed=1101（已完成）
  ...
```

### 4.3 实验输出目录

```text
experiments/baselines/
  random_seed0_round001_committee_models/       # 4 frozen_model.pb（已完成）
  random_seed0_round001_committee_prediction/   # selected_topk.json + .npz（已完成）
  random_seed0_round002_committee_models/       # 待训练
  random_seed0_round002_committee_prediction/   # 待预测
  ...
```

---

## 5. 已有可复用脚本

当前 `scripts/` 目录下已有完整工具链，无需新增脚本：

| 任务 | 已有脚本 | 状态 |
|---|---|---|
| 构造下一轮 train set | `scripts/data/merge_selected_frames.py` | 已可用 |
| 构造下一轮 candidate pool | `scripts/data/make_remaining_candidate.py` | 已可用 |
| 生成 committee configs | `scripts/config/make_round_committee_configs.py` | 已可用 |
| 训练 committee models | `scripts/train/train_round_committee_models.sh` | 已可用 |
| Committee prediction | `scripts/inference/predict_committee_models.py` | 已可用 |
| 一站式数据+配置准备 | `scripts/analysis/prepare_random_baseline_round.py` | 已可用 |
| Round 001 三 seed 汇总 | `scripts/analysis/summarize_random_round001_baselines.py` | 已可用 |
| 全轮次对比汇总 | `scripts/analysis/summarize_random_vs_uncertainty.py` | 已可用 |
| 对比图生成 | `scripts/analysis/plot_random_vs_uncertainty.py` | 已可用 |

---

## 6. 执行步骤

### Step 1: 生成 Round 002 数据和配置

```bash
# 在 DeepMD-kit Docker 外即可执行（只操作数据和 JSON 配置）
for seed in seed0 seed1 seed2; do
  python scripts/analysis/prepare_random_baseline_round.py \
    --seed-label $seed --round-id 2
done
```

此脚本自动完成：
1. 从 Round 001 committee prediction 读取 selected indices
2. 调用 `merge_selected_frames.py` 生成 `random_{seed}_round_002_train`
3. 调用 `make_remaining_candidate.py` 生成 `random_{seed}_round_002_candidate`
4. 调用 `make_round_committee_configs.py` 生成 4 个 committee configs

### Step 2: 训练 Round 002 committee models

```bash
# 在 DeepMD-kit Docker 内执行（需要 dp 命令和 GPU）
bash scripts/train/train_round_committee_models.sh \
  002 \
  configs/deepmd/random_seed0_round_002_committee \
  experiments/baselines/random_seed0_round002_committee_models \
  /data/zft/deepmd-al-hpc/data/toy_h2/valid

# 对 seed1、seed2 重复
```

每次训练输出 4 个 frozen_model.pb 和 test.log。

### Step 3: Round 002 committee prediction

```bash
# 在 DeepMD-kit Docker 内执行
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/random_seed0_round_002_candidate \
  --models \
    experiments/baselines/random_seed0_round002_committee_models/model_000/frozen_model.pb \
    experiments/baselines/random_seed0_round002_committee_models/model_001/frozen_model.pb \
    experiments/baselines/random_seed0_round002_committee_models/model_002/frozen_model.pb \
    experiments/baselines/random_seed0_round002_committee_models/model_003/frozen_model.pb \
  --output experiments/baselines/random_seed0_round002_committee_prediction/committee_predictions.npz \
  --selected-json experiments/baselines/random_seed0_round002_committee_prediction/selected_topk.json \
  --top-k 10

# 对 seed1、seed2 重复
```

### Step 4: Round 003 数据和训练

```bash
# 生成 Round 003 数据和配置（依赖 Round 002 prediction 完成）
for seed in seed0 seed1 seed2; do
  python scripts/analysis/prepare_random_baseline_round.py \
    --seed-label $seed --round-id 3
done

# 训练 Round 003（同 Step 2，替换 round-id）
# 预测 Round 003（同 Step 3，替换路径）
```

### Step 5: 全轮次汇总

```bash
# 更新汇总 CSV/MD（需先更新脚本中的 RANDOM_INPUTS 路径，添加 round002/003）
python scripts/analysis/summarize_random_vs_uncertainty.py

# 重新生成对比图
python scripts/analysis/plot_random_vs_uncertainty.py
```

---

## 7. 最终需要输出的文件

### 7.1 每个 seed 每轮

```text
random_{seed}_round{XXX}_metrics_summary.csv   # Energy RMSE, Force RMSE
random_{seed}_round{XXX}_metrics_summary.md
random_{seed}_round{XXX}_prediction_summary.csv # force_dev_max, energy_dev
random_{seed}_round{XXX}_prediction_summary.md
```

### 7.2 跨 seed 汇总

```text
random_round001_baseline_summary.csv  # 已完成
random_round002_baseline_summary.csv  # 待生成
random_round003_baseline_summary.csv  # 待生成
```

### 7.3 最终对比

```text
random_vs_uncertainty_summary.csv           # 含 Round 0–3 完整数据
random_vs_uncertainty_summary.md
random_vs_uncertainty_force_rmse.svg        # uncertainty vs random mean ± std
random_vs_uncertainty_energy_rmse.svg
random_vs_uncertainty_candidate_force_dev.svg
random_vs_uncertainty_dataset_size.svg
```

---

## 8. 预计耗时

基于 V100 上的已有经验（Round 001 约 1–2 分钟/model）：

| 阶段 | 模型数 | 预计耗时 |
|---|---|---|
| Round 002 训练 | 3 seeds × 4 models = 12 | ~15–25 min（2×V100 并行） |
| Round 002 预测 | 3 seeds × 1 run = 3 | ~1–2 min |
| Round 003 训练 | 3 seeds × 4 models = 12 | ~15–25 min（2×V100 并行） |
| Round 003 预测 | 3 seeds × 1 run = 3 | ~1–2 min |
| **总计** | | **~35–55 min** |

---

## 9. 注意事项

1. 每个 seed 的 Round 002 必须基于该 seed 自己的 Round 001 train/candidate 继续，不要从初始 candidate pool 重新选择。
2. Round 003 同理，基于 Round 002 的结果继续。
3. 训练配置（learning rate、batch size、n_steps）应与 uncertainty branch 和 Round 001 保持一致。
4. 每个 seed 的 committee prediction 使用相同的 `--top-k 10`。
5. 所有 `.npz`、`.pb`、`checkpoint`、`model.ckpt*` 不要提交到 Git。
6. 轻量结果（`.csv`、`.md`、`selected_topk.json`）可以提交。
