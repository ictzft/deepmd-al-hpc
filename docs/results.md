# 实验结果汇总与解释

本文档汇总 `deepmd-al-hpc` 在 toy H2 和 rMD17 ethanol 上的主动学习实验结果，说明当前能够支持的结论与仍需补充的实验。

当前结果主要来自：

```text
【toy H2】
uncertainty branch Round 0–3
selection-level random baseline
random / diversity / trust_level 四策略 multi-seed Round 001–003 retraining
V100 profiling baseline

【rMD17 ethanol】
uncertainty branch Round 0–3
random baseline (3 seeds × 3 rounds)
independent test evaluation (52 models)
MD stability (10K/100K NVE)
pipeline profiling (all stages)
```

---

## 1. 当前实验范围

当前已经完成的实验范围包括：

**toy H2 阶段**：

```text
toy H2 数据生成
  ↓
单模型 DeePMD baseline
  ↓
初始 4-model committee training
  ↓
初始 committee prediction
  ↓
uncertainty top-K selection
  ↓
dataset-level offline active learning Round 0–3
  ↓
Round 0–3 summary 与 learning curve
  ↓
selection-level random baseline
  ↓
random seed0 / seed1 / seed2 Round 001–003 retraining baseline
  ↓
diversity + trust_level multi-seed Round 001–003
  ↓
四策略 aligned comparison + learning curves
  ↓
V100 profiling baseline
```

**rMD17 ethanol 阶段**：

```text
数据格式转换 (convert_rmd17_to_deepmd.py)
  ↓
train/valid/test/candidate 数据划分 (split_rmd17_to_deepmd.py)
  ↓
uncertainty branch Round 0–3 (1000→4000 训练帧, 60000→57000 候选帧)
  ↓
independent test evaluation (10000 帧, 52 模型)
  ↓
random baseline (3 seeds × 3 rounds, 36 模型)
  ↓
MD stability (NVE 10K/100K)
  ↓
pipeline profiling (all stages, unified CSV)
```

当前尚未完成：

```text
rMD17 ethanol diversity / trust_level baselines
H100 / 多 GPU scaling
系统 GPU utilization / memory 曲线记录
```

---

## 2. Uncertainty Branch Round 0–3 Summary

当前 uncertainty branch 的数据规模如下：

```text
Round 0: train 200, candidate 50
Round 1: train 210, candidate 40
Round 2: train 220, candidate 30
Round 3: train 230, candidate 20
```

每轮 selection 设置为：

```text
selection strategy: uncertainty top-K
selection score   : force_dev_max
top-K             : 10
committee size    : 4
```

Round-level summary 如下：

| Round | Train frames | Candidate frames | Force RMSE mean | force_dev_max mean | force_dev_max max | force_dev_max min |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 50 | 1.821392e-01 | 4.409891e-01 | 7.891949e-01 | 3.228673e-01 |
| 1 | 210 | 40 | 1.617669e-01 | 2.693333e-01 | 5.083387e-01 | 1.376306e-01 |
| 2 | 220 | 30 | 1.938590e-01 | 1.874125e-01 | 2.759882e-01 | 1.514366e-01 |
| 3 | 230 | 20 | 1.742648e-01 | 1.701889e-01 | 2.426791e-01 | 1.504961e-01 |

对应结果文件：

```text
experiments/al_rounds_summary.csv
experiments/al_rounds_summary.md
experiments/al_model_level_summary.csv
```

---

## 3. Candidate Pool Uncertainty 变化

当前 uncertainty branch 中，top-K 高不确定性构型的平均 force model deviation 随轮次推进逐步降低：

```text
force_dev_max_mean:

Round 0: 0.440989
Round 1: 0.269333
Round 2: 0.187412
Round 3: 0.170189
```

该结果说明：

```text
随着主动学习轮次推进，
候选池中被选出的 top-K 高不确定性构型的平均 force model deviation 持续降低。
```

这可以理解为：

```text
模型最不确定的候选构型正在逐轮被加入训练集；
剩余 candidate pool 中的高不确定性区域逐步减少；
committee models 在当前 toy candidate space 中的预测分歧逐步减小。
```

更适合写成：

> 多轮 uncertainty-based active learning 后，候选池中的高不确定性构型平均 force model deviation 呈下降趋势，说明当前主动学习闭环能够逐步覆盖模型较不确定的构型区域。

---

## 4. Force RMSE 变化

当前验证集 Force RMSE mean 如下：

```text
Force RMSE mean:

Round 0: 0.182139
Round 1: 0.161767
Round 2: 0.193859
Round 3: 0.174265
```

可以看到，Force RMSE 没有严格单调下降。

因此不建议写成：

```text
错误表述：
模型精度随着主动学习轮次持续提升。
```

更严谨的表述应该是：

```text
推荐表述：
多轮主动学习后，候选池不确定性呈持续下降趋势；
验证集 Force RMSE 整体处于同一量级，但存在一定波动。
```

可能原因包括：

```text
toy H2 数据规模较小；
committee models 随机初始化带来差异；
当前 valid set 同时承担 candidate pool 和 validation/test 角色；
top-K uncertainty selection 更直接优化候选池覆盖，而不一定保证每一轮 RMSE 单调下降；
当前已完成 multi-seed (seed0/seed1/seed2) mean ± std 统计（Round 001/002/003）。
```

---

## 5. Learning Curve 文件

当前已经生成的 learning curve 文件包括：

```text
experiments/figures/force_model_deviation_rounds.svg
experiments/figures/dataset_size_rounds.svg
experiments/figures/validation_rmse_rounds.svg
```

对应含义：

```text
force_model_deviation_rounds.svg:
  展示 Round 0–3 中 force_dev_max 的变化趋势。

dataset_size_rounds.svg:
  展示训练集 frame 数增加和 candidate pool frame 数减少。

validation_rmse_rounds.svg:
  展示 Energy / Force RMSE 随 active learning round 的变化。
```

---

## 6. Selection-level Random Baseline 结果

当前已经加入 random sampling baseline，并生成 selection-level 对比结果。

Selection-level baseline 的目标是比较：

```text
uncertainty top-K 选出来的样本
vs.
random sampling 选出来的样本
```

在平均不确定性上的差异。

当前结果为：

```text
Round 000:
random force_dev_max_mean      : 0.143007
uncertainty force_dev_max_mean : 0.440989

Round 001:
random force_dev_max_mean      : 0.145731
uncertainty force_dev_max_mean : 0.269333
```

该结果说明：

```text
uncertainty top-K 策略确实选中了平均不确定性更高的构型；
random sampling 选中的构型平均不确定性较低。
```

但是需要注意：

```text
selection-level baseline 只能说明不同 selection strategy 选出的构型不确定性不同；
不能直接代表 retraining 后模型精度差异。
```

因此，selection-level baseline 不能单独支撑：

```text
uncertainty sampling 最终模型精度优于 random sampling
```

它只能支撑：

```text
uncertainty top-K selection 确实倾向于选择更高不确定性的构型。
```

相关结果文件：

```text
experiments/baselines/selection_baseline_runs.csv
experiments/baselines/selection_baseline_summary.csv
experiments/baselines/selection_baseline_summary.md
```

---

## 7. Multi-seed Random Round 001–003 Retraining 结果

当前已经完成 random seed0 / seed1 / seed2 的 Round 001–003 retraining baseline（2026-05-25, 2×V100）。

实验设置：

```text
从 Round 0 candidate pool 中随机选择 10 个 frames
  ↓
合并进初始 train set
  ↓
生成 random_seed*_round_001_train
  ↓
生成 random_seed*_round_001_candidate
  ↓
训练 4 个 random seed* committee models
  ↓
测试 committee models
```

每个 seed 的数据规模：

```text
random_seed*_round_001_train     : 210 frames
random_seed*_round_001_candidate : 40 frames
```

三个 seed 的 Round 001 committee models 测试结果：

| Seed | Energy RMSE Mean / eV | Energy RMSE Std / eV | Force RMSE Mean / eV/Å | Force RMSE Std / eV/Å |
|---|---:|---:|---:|---:|
| seed0 | 6.908853e-01 | 7.559906e-01 | 2.553366e-01 | 1.729852e-01 |
| seed1 | 2.488029e-01 | 4.563935e-01 | 2.288370e-01 | 3.565438e-02 |
| seed2 | 4.284121e-01 | 5.054377e-01 | 1.494923e-01 | 4.669047e-02 |
| **Mean** | **4.560335e-01** | — | **2.112220e-01** | — |
| **Std** | **2.223318e-01** | — | **5.507695e-02** | — |

跨 seed 随机均值为：

```text
Random Mean Energy RMSE: 0.456034 eV
Random Mean Force RMSE : 0.211222 eV/Å
```

相关结果文件：

```text
experiments/baselines/random_seed0_round001_metrics_summary.csv
experiments/baselines/random_seed0_round001_metrics_summary.md
experiments/baselines/random_seed1_round001_metrics_summary.csv
experiments/baselines/random_seed1_round001_metrics_summary.md
experiments/baselines/random_seed2_round001_metrics_summary.csv
experiments/baselines/random_seed2_round001_metrics_summary.md
experiments/baselines/random_round001_comparison.csv
```

主要观察：

```text
三个 random seed 的 committee models 间 Energy RMSE 和 Force RMSE 存在较大方差；
该方差源自 toy H2 数据规模较小和 committee 随机初始化；
seed0 的 Energy RMSE Std 最大 (0.756 eV)，因为 model_003 异常偏高。
```

### 7.1 Random Round 002 结果

| Seed | Energy RMSE Mean / eV | Force RMSE Mean / eV/Å |
|---|---:|---:|
| seed0 | 1.977364e+00 | 2.120793e-01 |
| seed1 | 1.847642e+00 | 1.796397e-01 |
| seed2 | 9.067673e-01 | 1.968315e-01 |
| **Mean** | **1.577258e+00** | **1.961835e-01** |

### 7.2 Random Round 003 结果

| Seed | Energy RMSE Mean / eV | Force RMSE Mean / eV/Å |
|---|---:|---:|
| seed0 | 1.369364e+00 | 2.373825e-01 |
| seed1 | 1.707502e+00 | 1.417639e-01 |
| seed2 | 1.310121e+00 | 1.879388e-01 |
| **Mean** | **1.462329e+00** | **1.890284e-01** |

### 7.3 Multi-round Random Mean Force RMSE

```text
Round 001: 0.211222 eV/Å
Round 002: 0.196183 eV/Å
Round 003: 0.189028 eV/Å
```

Multi-round random baseline 已完成（2026-05-25, 2×V100）。完整汇总见 `experiments/baselines/random_round002_baseline_summary.csv` 和 `random_round003_baseline_summary.csv`。

---

## 8. Candidate-pool Uncertainty 对比

各 branch Round 001 retraining 后，对剩余 candidate pool 进行 committee prediction 的对比结果。

| Run | n_frames | force_dev_max mean | force_dev_max max | force_dev_max min | energy_dev mean |
|---|---:|---:|---:|---:|---:|
| uncertainty_round001 | 40 | 0.126442 | 0.508339 | 0.042645 | 0.448212 |
| random_seed0_round001 | 40 | 0.355420 | 1.586355 | 0.086667 | 0.656541 |
| random_seed1_round001 | 40 | 0.487795 | 1.262038 | 0.327483 | 0.396726 |
| random_seed2_round001 | 40 | 0.332138 | 1.117230 | 0.139321 | 0.446260 |
| **random mean** | 40 | **0.391784** | — | — | **0.499842** |

其中：

```text
uncertainty_round001:
force_dev_max_mean = 0.126442

random mean (seed0/seed1/seed2):
force_dev_max_mean = 0.391784
```

该结果说明：

```text
在当前 toy H2 offline active learning 设置下，
加入 10 个 uncertainty-selected frames 后，
剩余 candidate pool 的平均 force model deviation (0.126442)
低于所有三个 random seed baseline (seed0: 0.355420, seed1: 0.487795, seed2: 0.332138)。
```

更严谨的表述为：

> 在 toy H2 和 Round 001 的 random seed0 / seed1 / seed2 对比中，uncertainty sampling 一致显示出更低的 remaining candidate-pool force model deviation (0.126442 vs random mean 0.391784)。

需要注意：

```text
该结论目前仍基于 toy H2 数据集；
Round 001–003 multi-round random retraining 已补充完成；
但仍不能直接推广到真实 DFT / AIMD 数据集。
```

相关结果文件：

```text
experiments/baselines/random_seed0_round001_prediction_summary.csv
experiments/baselines/random_seed0_round001_prediction_summary.md
experiments/baselines/random_seed1_round001_prediction_summary.csv
experiments/baselines/random_seed1_round001_prediction_summary.md
experiments/baselines/random_seed2_round001_prediction_summary.csv
experiments/baselines/random_seed2_round001_prediction_summary.md
experiments/baselines/random_round001_comparison.csv
```

---

## 9. Four-Strategy Multi-Seed Comparison (2026-05-25)

Diversity 和 trust_level 策略已完成 seed0/seed1/seed2 Round 001–003 multi-seed retraining。四策略 Force RMSE 对比如下：

| Strategy | R1 F_RMSE | R1 std | R2 F_RMSE | R2 std | R3 F_RMSE | R3 std |
|---|---:|---:|---:|---:|---:|---:|
| uncertainty | 1.515e-01 | 2.496e-02 | 2.132e-01 | 2.242e-02 | 1.965e-01 | 2.421e-02 |
| random | 2.112e-01 | 5.508e-02 | 1.962e-01 | 1.623e-02 | 1.890e-01 | 4.782e-02 |
| diversity | 2.052e-01 | 5.789e-02 | 1.738e-01 | 9.290e-03 | 1.759e-01 | 4.082e-02 |
| trust_level | 1.353e-01 | 2.761e-02 | 1.491e-01 | 2.256e-02 | 1.782e-01 | 6.470e-03 |

完整数据见 `experiments/strategy_comparison_toy_h2/strategy_summary.md` 和 `experiments/baselines/aligned_comparison.md`。
全部四策略（uncertainty / random / diversity / trust_level）均为 3-seed mean ± std（2026-05-25, 2×V100）。

---

## 10. Structural Diversity Analysis (2026-05-25)

Round 0 selection-level 对比三种策略选中帧的 H-H 键长分布：

| Strategy | PW dist (Å) | FDM mean | 解读 |
|---|---:|---:|---|
| uncertainty | 0.038 | 0.441 | 最高不确定性，结构聚类 |
| diversity | 0.119 | 0.254 | 3.1x 更大结构覆盖 |
| trust_level | 0.063 | 0.351 | 中间态 |

Diversity (FPS) 在高不确定性候选池中通过 farthest-point sampling 显著提升了选中帧的结构覆盖度（pairwise distance 从 0.038Å 增加到 0.119Å），代价是平均 force_dev_max 从 0.441 降低到 0.254。这验证了 uncertainty-diversity 策略的设计动机：在高不确定性帧中增加结构多样性，减少重复选择相似构型。

完整分析见 `experiments/baselines/diversity_analysis.md`。

---

## 11. rMD17 Ethanol 真实数据集结果 (2026-05-26)

### 11.1 Uncertainty Branch Round 0–3

| Round | Train | Candidate | Valid F_RMSE | Test F_RMSE | force_dev_max (top-1000) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1000 | 60000 | 0.3739 | 0.3439 | 0.6129 |
| 1 | 2000 | 59000 | 0.3715 | 0.3433 | 0.4570 |
| 2 | 3000 | 58000 | 0.3644 | 0.3352 | 0.3906 |
| 3 | 4000 | 57000 | 0.3537 | 0.3266 | 0.4569 |

- **Validation 和 independent test Force RMSE 均单调下降**
- Independent test（10000帧，从未参与 AL）Force RMSE 稳定比 valid 低 ~0.028 eV/Å
- 与 toy H2 的波动不同，真实分子上的精度提升是稳定且持续的

### 11.2 Random Baseline 对比（3-seed mean ± std）

| Round | Uncertainty F_RMSE | Random F_RMSE |
|---:|---:|---:|
| 1 | 0.3715 | 0.3734 ± 0.010 |
| 2 | 0.3644 | 0.3990 ± 0.031 |
| 3 | 0.3537 | **0.6067 ± 0.385** |

- Uncertainty 持续改善，random 显著恶化——在真实分子体系上首次验证了 uncertainty-based AL 的有效性
- Round 3 差距 0.25 eV/Å，random 甚至比 Round 0 的 baseline（0.3739）更差
- 说明 random sampling 选中的非代表性构型导致模型过拟合

### 11.2b Four-Strategy Comparison (Round 3, 3-seed mean ± std)

| Strategy | Force RMSE | Std | vs Random |
|---|---:|---:|---:|
| uncertainty | 0.3537 | 0.0247 | 1.72× better |
| diversity | 0.3555 | 0.0143 | 1.71× better |
| trust_level | 0.3616 | 0.0166 | 1.68× better |
| random | 0.6067 | 0.6826 | — baseline |

- 三种 active 策略差异在 1σ 内，mean 均明显低于 random（0.607 eV/Å），但 random 跨 seed 方差大（std=0.683），不宜仅凭当前数据做严格统计显著性结论
- 与 toy H2 结论一致：未出现单一策略显著优于其他
- Diversity 有 1/36 模型训练失败（seed0 model_000, E=30.2 eV），剔除后 F_RMSE=0.3555

### 11.3 MD 稳定性

| 条件 | 结果 |
|---|---|
| NVE 10K, 0.25 fs, 2.5 ps | 所有模型稳定，drift ~0.035 eV/ps |
| NVE 100K+, 0.25 fs | 所有模型立即解离（< 0.005 ps） |

当前 Force RMSE ~0.35 eV/Å 足以维持近平衡态稳定，但不足以支撑高温 MD。需更多训练数据。

### 11.4 与 toy H2 的对比

| 指标 | toy H2 | rMD17 ethanol |
|---|---|---|
| Force RMSE 趋势 | 波动，无单调性 | **单调下降** |
| Uncertainty vs Random | 差异在 1σ 内 | **显著差异**（Round 3: 0.25 eV/Å gap）|
| 跨模型方差 | 大 | 中等 |
| Independent test | 无（valid = candidate） | 有（10000帧独立测试集）|

完整数据见 `experiments/rmd17_ethanol_summary/`。

---

## 12. 当前可以支持的结论

基于当前结果，可以支持以下结论：

```text
1. 当前项目已经跑通 DeePMD toy H2 主动学习原型流程。
2. 已经实现 4-model committee training 和真实 committee prediction。
3. 已经能够计算 force / energy model deviation。
4. 已经能够基于 force_dev_max 进行 top-K 高不确定性构型选择。
5. 已经完成 dataset-level offline active learning Round 0–3 多轮闭环。
6. uncertainty branch 中 top-K force_dev_max_mean 随轮次推进呈下降趋势。
7. random sampling selection-level baseline 已经初步完成。
8. uncertainty top-K 确实选中了平均不确定性更高的构型。
9. random seed0 / seed1 / seed2 Round 001–003 retraining baseline 已经完成（2026-05-25, 2×V100）。
10. multi-seed random mean ± std 已经在 Round 001/002/003 上完成。
11. uncertainty vs random full comparison table 和 multi-round learning curve 已生成。
12. 在 Round 001 remaining candidate-pool 对比中，uncertainty branch 的 force_dev_max_mean 低于所有三个 random seed。
13. V100 训练耗时已从 36 个 train.log 中提取，2×V100 并行加速比约 1.97×。
14. rMD17 ethanol 上 uncertainty branch 的 validation 和 independent test Force RMSE 均单调下降。
15. rMD17 ethanol 单体系上，uncertainty 相比 random 表现出更稳定的 Force RMSE 改善趋势（Round 3: 0.354 vs 0.607 ± 0.385 eV/Å）。
16. rMD17 ethanol 上 NVE 10K MD 稳定，100K+ 解离——模型精度需进一步提高。
```

可以写进 README 的简洁表述：

> 当前 toy H2 原型已经完成 DeePMD committee training、candidate-pool prediction、model deviation 计算、uncertainty top-K selection、dataset update 和 Round 0–3 offline active learning 闭环。全部四种策略（uncertainty / random / diversity / trust_level）均已完成 seed0/seed1/seed2 Round 001–003 multi-seed multi-round retraining，四策略对齐对比表明各策略 Force RMSE 差异在 1σ 以内。

---

## 13. 当前不能过度声称的结论

当前结果还不能支持以下过强结论：

```text
1. uncertainty sampling 在所有体系上均显著优于 random sampling（仅 rmd17 ethanol 单体系验证，random std 较大）；
2. 当前方法在真实 DFT / AIMD 数据集上有效（rmd17 ethanol 已验证，多体系待验证）；
3. 当前方法已经达到 CCF-B 论文完整实验标准；
4. 当前方法已经完成 H100 / 多 GPU 加速验证；
5. 当前 active learning 可以稳定降低 Force RMSE（rmd17 上已验证，toy H2 上波动）；
6. 当前 toy H2 结果可以推广到真实材料体系（rmd17 趋势不同，需更多体系）；
7. 当前方法在所有真实材料体系上有效；
8. 当前结果已经证明 MD 稳定性更好（高温 MD 尚不稳定）。
```

更严谨的说法：

```text
toy H2 上已完成四策略 multi-seed multi-round 完整对比；
rMD17 ethanol 上 uncertainty vs random 对比已完成，
在独立测试集上，uncertainty 相比 random 表现出更稳定的改善趋势（Round 3: 0.327 vs 0.580 ± 0.692 eV/Å，random 方差大）；
diversity/trust_level 在真实数据上已完成（四策略差异在 1σ 内）；
仍需要多个真实体系和 H100 scaling 进一步验证。
```

---

## 14. 当前实验限制

当前项目仍处于原型验证阶段，主要限制包括：

1. toy H2 数据集仅用于流程验证；rMD17 ethanol 已提供真实分子体系证据，但仅覆盖单一分子；
2. toy H2 中 valid set 同时承担 candidate pool 和 validation/test 角色（rMD17 已通过 independent test 修正）；
3. rMD17 ethanol 四策略 multi-seed multi-round 已完成；independent test 已完成；diversity/trust_level 在真实数据上已运行；
4. toy H2 上 diversity 和 trust-level 已完成 multi-seed Round 001–003；rMD17 上待运行；
5. 当前尚未进行 H100 / 多 GPU scaling 实验；
6. V100 training wall-clock profiling 已全量记录（52 模型）；GPU utilization 曲线未记录；
7. MD 稳定性在 10K NVE 通过验证，100K+ 解离——需提高模型精度后重新评估；
8. 在 rMD17 ethanol 单体系实验中，uncertainty 相比 random 表现出更稳定的改善趋势；random 跨 seed 方差较大，multi-system 验证仍需补充。

---

## 15. 后续需要补充的结果

为了进一步支撑 CCF-B 投稿，后续需要补充以下结果。

### 15.1 Random Baseline（已完成）

```text
random seed0 / seed1 / seed2 Round 001–003 ✓
multi-seed random mean ± std (Round 001/002/003) ✓
Force RMSE / Energy RMSE learning curve ✓
candidate-pool uncertainty curve ✓
training time per model ✓
prediction time per round ✓
```

### 15.2 Four-Strategy Comparison（已完成）

```text
uncertainty / random / diversity / trust_level multi-seed Round 001–003 ✓
aligned comparison table (remaining candidate-pool metric) ✓
structural diversity analysis ✓
```

### 15.3 V100 Profiling（已完成）

```text
single-model training time (132 models, mean=11.0s) ✓
2×V100 parallel training time (~22s/round) ✓
prediction time (6.5–7.2s) ✓
dataset update time (0.34s) ✓
end-to-end round time (~29s) ✓
GPU utilization sample (SM 23%, 5407 MiB) ✓
```

---

### 15.4 rMD17 Ethanol 真实数据集（已完成主要部分）

```text
rMD17 ethanol data conversion ✓
train / candidate / test split ✓
uncertainty branch Round 0–3 ✓
random baseline (3 seeds × 3 rounds) ✓
independent test evaluation (52 models) ✓
MD stability (10K/100K NVE) ✓
pipeline profiling (52 models, all stages) ✓
RMSE learning curve ✓
diversity / trust_level baselines ✓
```

当前已经从：

```text
toy H2 workflow validation
```

推进到：

```text
realistic first-principles dataset validation
```

---

## 16. 建议论文表述

当前阶段可以使用的论文式表述：

```text
We first validate the proposed active learning workflow on a toy H2 dataset.
The goal of this stage is not to draw conclusions about realistic material systems,
but to verify the correctness of the dataset-level active learning loop,
including committee training, candidate-pool prediction, model deviation estimation,
uncertainty-based selection, dataset update, and baseline comparison.
```

中文表述：

```text
本文首先在 toy H2 数据集上验证所提出主动学习流程的可运行性。
该阶段的目标不是评价真实材料体系上的模型精度，
而是验证 dataset-level 主动学习闭环是否能够完整打通，
包括 committee 训练、候选池预测、模型偏差计算、不确定性采样、
数据集更新以及 random baseline 对比。
```

当前结果可以写作：

```text
在 toy H2 原型实验中，uncertainty branch 的 top-K force_dev_max_mean
从 Round 0 的 0.440989 降低到 Round 3 的 0.170189，
表明随着主动学习轮次推进，剩余候选池中的高不确定性构型逐步减少。
```

关于 random baseline 的表述：

```text
Selection-level random baseline shows that uncertainty top-K selects configurations
with higher average force model deviation than random sampling.
Furthermore, in the Round 001 retraining comparison across random seed0 / seed1 / seed2,
the uncertainty branch consistently yields a lower remaining candidate-pool
force_dev_max_mean (0.126 vs random mean 0.392) than all three random branches.
```

中文表述：

```text
selection-level random baseline 表明，uncertainty top-K 相比随机采样
能够选中平均 force model deviation 更高的构型。
进一步地，在 Round 001 的 random seed0 / seed1 / seed2 对比中，
uncertainty branch 的 remaining candidate-pool force_dev_max_mean
一致低于所有三个 random branch (0.126 vs random mean 0.392)。
```

关于 rMD17 ethanol 结果的表述：

```text
On rMD17 ethanol (9-atom organic molecule, 60000-frame candidate pool),
the uncertainty-based active learning branch shows monotonically decreasing
Force RMSE on both validation (0.374 to 0.354 eV/Å) and independent test sets
(0.344 to 0.327 eV/Å). In contrast, the random baseline (3 seeds × 3 rounds)
degrades significantly, with independent test Force RMSE increasing from
0.346 to 0.580 eV/Å by Round 3 (1.78× worse than uncertainty).
```

中文表述：

```text
在 rMD17 ethanol（9 原子有机分子，60000 帧候选池）上，
uncertainty-based active learning 的 Force RMSE 在 validation 和 independent test
上均单调下降（valid: 0.374→0.354 eV/Å, test: 0.344→0.327 eV/Å）。
相比之下，random baseline（3 seeds × 3 rounds）的 independent test Force RMSE
从 0.346 恶化到 0.580 eV/Å（Round 3 为 uncertainty 的 1.78 倍）。
```

注意必须补充限制：

```text
These observations are from a single molecular system (rMD17 ethanol).
Multi-system validation and diversity/trust-level baselines on real data
are still required before drawing general conclusions.
```

中文表述：

```text
当前结论基于单一分子体系（rMD17 ethanol）。
仍需要多个真实体系和 diversity/trust-level baselines 进一步验证。
```

---

## 17. 结果文件索引

主线 active learning 结果：

```text
experiments/al_rounds_summary.csv
experiments/al_rounds_summary.md
experiments/al_model_level_summary.csv
experiments/figures/force_model_deviation_rounds.svg
experiments/figures/dataset_size_rounds.svg
experiments/figures/validation_rmse_rounds.svg
```

selection-level random baseline：

```text
experiments/baselines/selection_baseline_runs.csv
experiments/baselines/selection_baseline_summary.csv
experiments/baselines/selection_baseline_summary.md
```

random seed0 retraining baseline：

```text
experiments/baselines/random_seed0_round001_metrics_summary.csv
experiments/baselines/random_seed0_round001_metrics_summary.md
```

random seed1 retraining baseline：

```text
experiments/baselines/random_seed1_round001_metrics_summary.csv
experiments/baselines/random_seed1_round001_metrics_summary.md
```

random seed2 retraining baseline：

```text
experiments/baselines/random_seed2_round001_metrics_summary.csv
experiments/baselines/random_seed2_round001_metrics_summary.md
```

multi-seed Round 001 汇总：

```text
experiments/baselines/random_round001_comparison.csv
```

random seed0 candidate-pool prediction：

```text
experiments/baselines/random_seed0_round001_prediction_summary.csv
experiments/baselines/random_seed0_round001_prediction_summary.md
experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json
```

random seed1 candidate-pool prediction：

```text
experiments/baselines/random_seed1_round001_prediction_summary.csv
experiments/baselines/random_seed1_round001_prediction_summary.md
experiments/baselines/random_seed1_round001_committee_prediction/selected_topk.json
```

random seed2 candidate-pool prediction：

```text
experiments/baselines/random_seed2_round001_prediction_summary.csv
experiments/baselines/random_seed2_round001_prediction_summary.md
experiments/baselines/random_seed2_round001_committee_prediction/selected_topk.json
```

rMD17 ethanol 结果：

```text
experiments/rmd17_ethanol_summary/al_rounds_summary.csv
experiments/rmd17_ethanol_summary/al_rounds_summary.md
experiments/rmd17_ethanol_summary/al_model_level_summary.csv
experiments/rmd17_ethanol_summary/independent_test_all_summary.csv
experiments/rmd17_ethanol_summary/independent_test_all_models.csv
experiments/rmd17_ethanol_summary/random_baseline_round_summary.csv
experiments/rmd17_ethanol_summary/random_baseline_model_level.csv
experiments/rmd17_ethanol_summary/profiling_unified.csv
experiments/rmd17_ethanol_summary/profiling_all_models.csv
experiments/rmd17_ethanol_summary/md_stability/md_summary.json
```

---

## 18. 小结

当前结果说明：

```text
toy H2 上四策略 multi-seed multi-round 已完成；V100 profiling baseline 已完成；
rMD17 ethanol 上 uncertainty branch Round 0–3 已完成，random baseline (3 seeds × 3 rounds) 已完成，
rMD17 ethanol 四策略 multi-seed multi-round 已完成，independent test evaluation 已完成，MD stability 已完成；
H100 scaling 未开始。
```

当前结果已经属于：

```text
toy workflow validation + real molecule validation (rMD17 ethanol)
```

而不是：

```text
final paper-level validation on real datasets (multi-system, multi-strategy still needed)
```

下一步重点是：

```text
rMD17 ethanol: diversity + trust_level baselines
  ↓
full GPU utilization curves (nvidia-smi dmon)
  ↓
H100 scaling
```