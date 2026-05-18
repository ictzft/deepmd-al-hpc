# 实验结果汇总与解释

本文档用于汇总 `deepmd-al-hpc` 当前 toy H2 offline active learning 原型实验结果，并说明当前能够支持的结论与仍需补充的实验。

当前结果主要来自：

```text
uncertainty branch Round 0–3
selection-level random baseline
random seed0 Round 001 retraining baseline
uncertainty_round001 vs random_seed0_round001 candidate-pool uncertainty comparison
```

需要说明的是：

```text
当前 toy H2 数据集只用于流程验证，
不能代表真实材料或分子体系上的最终模型性能。
```

---

## 1. 当前实验范围

当前已经完成的实验范围包括：

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
random seed0 Round 001 retraining baseline
  ↓
uncertainty branch vs random seed0 candidate-pool comparison
```

当前尚未完成：

```text
random seed1 / seed2 retraining baseline
random Round 0–3 多轮 retraining
random mean ± std
完整 RMSE learning curve 对比
真实 DFT / AIMD 数据集
H100 / 多 GPU scaling
MD 稳定性验证
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
当前尚未进行多 seed mean ± std 统计。
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

## 7. Random seed0 Round 001 Retraining 结果

当前已经完成 random seed0 的 Round 001 retraining baseline。

实验设置：

```text
从 Round 0 candidate pool 中随机选择 10 个 frames
  ↓
合并进初始 train set
  ↓
生成 random_seed0_round_001_train
  ↓
生成 random_seed0_round_001_candidate
  ↓
训练 4 个 random seed0 committee models
  ↓
测试 committee models
```

数据规模：

```text
random_seed0_round_001_train     : 210 frames
random_seed0_round_001_candidate : 40 frames
```

4 个 random seed0 Round 001 committee models 的测试结果为：

| Model | Energy RMSE / eV | Force RMSE / eV/Å |
|---|---:|---:|
| model_000 | 8.813657e-01 | 2.551970e-01 |
| model_001 | 9.782892e-03 | 1.827423e-01 |
| model_002 | 1.965928e-01 | 8.939603e-02 |
| model_003 | 1.675800e+00 | 4.940109e-01 |
| Mean | 6.908853e-01 | 2.553366e-01 |
| Std | 7.559906e-01 | 1.729852e-01 |

相关结果文件：

```text
experiments/baselines/random_seed0_round001_metrics_summary.csv
experiments/baselines/random_seed0_round001_metrics_summary.md
```

主要观察：

```text
random seed0 committee models 的 Energy RMSE 和 Force RMSE 方差较大；
单个 random seed 结果不够稳定；
后续需要继续补充 random seed1 / seed2，并报告 random mean ± std。
```

---

## 8. Candidate-pool Uncertainty 对比

random seed0 Round 001 retraining 后，对剩余 candidate pool 进行 committee prediction，并与 uncertainty branch Round 001 进行对比。

对比结果如下：

| Run | Candidate Pool | n_frames | force_dev_max mean | force_dev_max max | force_dev_max min | energy_dev mean |
|---|---|---:|---:|---:|---:|---:|
| uncertainty_round001 | data/toy_h2/round_001_candidate | 40 | 0.126442 | 0.508339 | 0.042645 | 0.448212 |
| random_seed0_round001 | data/toy_h2/random_seed0_round_001_candidate | 40 | 0.355420 | 1.586355 | 0.086667 | 0.656541 |

其中：

```text
uncertainty_round001:
force_dev_max_mean = 0.126442

random_seed0_round001:
force_dev_max_mean = 0.355420
```

该结果说明：

```text
在当前 toy H2 offline active learning 设置下，
加入 10 个 uncertainty-selected frames 后，
剩余 candidate pool 的平均 force model deviation 低于 random seed0 baseline。
```

更严谨的表述为：

> 在 toy H2 和 random seed0 Round 001 的初步对比中，uncertainty sampling 相比 random seed0 baseline 显示出更低的 remaining candidate-pool force model deviation。

需要注意：

```text
该结论目前仍基于 toy H2 和单个 random seed0；
不能直接推广到所有 random seeds；
也不能直接推广到真实 DFT / AIMD 数据集。
```

相关结果文件：

```text
experiments/baselines/random_seed0_round001_prediction_summary.csv
experiments/baselines/random_seed0_round001_prediction_summary.md
experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json
```

---

## 9. 当前可以支持的结论

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
9. random seed0 Round 001 retraining baseline 已经完成。
10. 在 random seed0 Round 001 对比中，uncertainty branch 的 remaining candidate-pool force_dev_max_mean 更低。
```

可以写进 README 的简洁表述：

> 当前 toy H2 原型已经完成 DeePMD committee training、candidate-pool prediction、model deviation 计算、uncertainty top-K selection、dataset update 和 Round 0–3 offline active learning 闭环。初步 random baseline 结果显示，uncertainty top-K 能选中平均不确定性更高的构型，并在 random seed0 Round 001 对比中显示出更低的 remaining candidate-pool force model deviation。

---

## 10. 当前不能过度声称的结论

当前结果还不能支持以下过强结论：

```text
1. uncertainty sampling 一定显著优于 random sampling；
2. 当前方法在真实 DFT / AIMD 数据集上有效；
3. 当前方法已经达到 CCF-B 论文完整实验标准；
4. 当前方法已经完成 H100 / 多 GPU 加速验证；
5. 当前 active learning 可以稳定降低 Force RMSE；
6. 当前 toy H2 结果可以推广到真实材料体系；
7. 当前 random baseline 已经完整完成；
8. 当前结果已经证明 MD 稳定性更好。
```

更严谨的说法：

```text
当前结果说明主动学习闭环和 baseline 对比流程可行；
但仍需要多 seed random baseline、真实 DFT / AIMD 数据、系统 profiling 和 H100 scaling 进一步验证。
```

---

## 11. 当前实验限制

当前项目仍处于原型验证阶段，主要限制包括：

1. toy H2 数据集仅用于流程验证，不能代表真实材料或分子体系；
2. 当前 valid set 同时承担 candidate pool 和 validation/test 的角色；
3. 当前尚未引入真实 DFT / AIMD 数据集；
4. random sampling baseline 尚未完整完成，目前 retraining baseline 只完成 random seed0；
5. 当前尚未完成 random seed1 / seed2 和多轮 random retraining；
6. 当前尚未报告 random mean ± std；
7. 当前尚未形成完整 RMSE learning curve 对比；
8. 当前尚未加入结构多样性选择策略；
9. 当前尚未进行 H100 / 多 GPU scaling 实验；
10. 当前尚未系统记录端到端 active learning wall-clock time；
11. 当前尚未进行 MD 稳定性验证；
12. 当前结果更适合证明主动学习闭环可运行，尚不足以作为完整论文级结论。

---

## 12. 后续需要补充的结果

为了进一步支撑 CCF-B 投稿，后续需要补充以下结果。

### 12.1 完整 Random Baseline

需要补充：

```text
random seed1 Round 001
random seed2 Round 001
random seed0 Round 002 / Round 003
random seed1 Round 002 / Round 003
random seed2 Round 002 / Round 003
```

最终报告：

```text
Force RMSE mean ± std
Energy RMSE mean ± std
candidate-pool uncertainty mean ± std
selected uncertainty mean ± std
training time mean ± std
prediction time mean ± std
```

---

### 12.2 Full RMSE Learning Curve

需要生成：

```text
uncertainty branch Force RMSE curve
random mean Force RMSE curve
random ± std shaded region
uncertainty branch Energy RMSE curve
random mean Energy RMSE curve
```

目标是回答：

```text
uncertainty sampling 是否在多轮 retraining 后比 random sampling 更有效？
```

---

### 12.3 Candidate-pool Uncertainty Curve

需要生成：

```text
uncertainty branch candidate-pool force_dev_max_mean curve
random mean candidate-pool force_dev_max_mean curve
random ± std shaded region
```

目标是回答：

```text
uncertainty sampling 是否更快降低 remaining candidate pool uncertainty？
```

---

### 12.4 Profiling 与 Wall-clock Time

需要补充：

```text
single-model training time
4-model committee training time
committee prediction time
model deviation calculation time
selection time
dataset update time
end-to-end active learning round time
```

目标是回答：

```text
主动学习闭环的主要耗时瓶颈在哪里？
多模型并行和 H100 加速能否降低端到端 wall-clock time？
```

---

### 12.5 真实 DFT / AIMD 数据集

需要迁移到真实数据集：

```text
real DFT / AIMD configurations
DeepMD npy format conversion
train / candidate / test split
uncertainty branch
random baseline
RMSE learning curve
candidate-pool uncertainty curve
```

目标是从：

```text
toy H2 workflow validation
```

推进到：

```text
realistic first-principles dataset validation
```

---

## 13. 建议论文表述

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
Furthermore, in the preliminary random seed0 Round 001 retraining comparison,
the uncertainty branch yields a lower remaining candidate-pool force_dev_max_mean
than the random seed0 branch.
```

中文表述：

```text
selection-level random baseline 表明，uncertainty top-K 相比随机采样
能够选中平均 force model deviation 更高的构型。
进一步地，在 random seed0 Round 001 的初步 retraining 对比中，
uncertainty branch 的 remaining candidate-pool force_dev_max_mean
低于 random seed0 branch。
```

注意必须补充限制：

```text
However, this observation is still based on a toy H2 dataset and a single random seed.
More random seeds, complete random retraining branches, real DFT / AIMD datasets,
and systematic profiling are needed before drawing general conclusions.
```

中文表述：

```text
然而，该观察目前仍基于 toy H2 数据集和单个 random seed。
在得出一般性结论之前，仍需补充多随机种子 random baseline、
完整 random retraining branch、真实 DFT / AIMD 数据集以及系统性能分析。
```

---

## 14. 结果文件索引

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

random seed0 candidate-pool prediction：

```text
experiments/baselines/random_seed0_round001_prediction_summary.csv
experiments/baselines/random_seed0_round001_prediction_summary.md
experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json
```

---

## 15. 小结

当前结果说明：

```text
deepmd-al-hpc 已经完成 toy H2 上的 dataset-level offline active learning 原型验证；
uncertainty branch 已经跑通 Round 0–3；
random sampling baseline 已经从 selection-level 推进到 random seed0 Round 001 retraining；
当前结果初步显示 uncertainty sampling 能够更有效地降低 remaining candidate-pool force model deviation。
```

但当前结果仍然属于：

```text
toy workflow validation
```

而不是：

```text
final paper-level validation
```

下一步重点是：

```text
random seed1 / seed2
  ↓
random Round 0–3 retraining
  ↓
random mean ± std
  ↓
full RMSE learning curve
  ↓
real DFT / AIMD dataset
  ↓
profiling and H100 scaling
```