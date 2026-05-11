# exp_006_offline_active_learning

## 1. 实验目的

本实验用于验证一个最小 offline active learning round 是否能够基于真实 committee prediction 和 model deviation 完成构型选择。

该实验承接前面的实验：

```text
exp_003_single_model_baseline
  ↓
exp_004_committee_models
  ↓
exp_005_committee_prediction
  ↓
exp_006_offline_active_learning
```

其中，`exp_005` 已经完成 4 个真实 DeePMD frozen models 对同一个 candidate pool 的预测，并基于 `force_dev_max` 选出了 top-10 高不确定性构型。

本实验进一步将该选择结果整理为一轮 offline active learning 的 round 记录。

---

## 2. 实验流程

```text
初始训练集
  ↓
candidate pool
  ↓
committee models prediction
  ↓
force / energy model deviation
  ↓
top-K 高不确定性构型选择
  ↓
模拟加入训练集
  ↓
生成 round_001_selection.json
```

---

## 3. 输入数据

本实验使用 `exp_005` 的构型筛选结果：

```text
experiments/exp_005_committee_prediction/selected_topk.json
```

candidate pool 来自：

```text
data/toy_h2/valid
```

---

## 4. Round 1 设置

| 项目 | 数值 |
|---|---:|
| round_id | 1 |
| selection policy | top-k by force_dev_max |
| initial training frames | 200 |
| candidate frames before selection | 50 |
| selected frames | 10 |
| candidate frames after selection | 40 |
| training frames after selection | 210 |
| n_models | 4 |
| n_atoms | 2 |

---

## 5. 选择结果

本轮基于 `force_dev_max` 选择 top-10 高不确定性构型：

| Rank | Frame index | force_dev_max | force_dev_mean | energy_dev |
|---:|---:|---:|---:|---:|
| 1 | 39 | 7.891949e-01 | 7.891949e-01 | 8.625447e-01 |
| 2 | 18 | 4.544573e-01 | 4.544573e-01 | 8.399900e-01 |
| 3 | 12 | 4.490663e-01 | 4.490663e-01 | 8.393389e-01 |
| 4 | 9 | 4.369525e-01 | 4.369525e-01 | 8.346730e-01 |
| 5 | 14 | 4.351531e-01 | 4.351531e-01 | 8.352293e-01 |
| 6 | 19 | 4.060467e-01 | 4.060467e-01 | 8.295973e-01 |
| 7 | 13 | 3.985356e-01 | 3.985356e-01 | 8.294061e-01 |
| 8 | 23 | 3.835185e-01 | 3.835185e-01 | 8.290560e-01 |
| 9 | 44 | 3.340988e-01 | 3.340988e-01 | 8.799229e-01 |
| 10 | 4 | 3.228673e-01 | 3.228673e-01 | 8.277992e-01 |

---

## 6. 输出文件

```text
experiments/exp_006_offline_active_learning/round_001_selection.json
```

该文件记录了：

```text
round_id
selection_policy
candidate_pool 状态
training_set 状态
committee models 信息
selected_indices
remaining_candidate_indices
selected_frames
```

---

## 7. 实验结论

本实验完成了一个最小 offline active learning round 的选择记录。

当前已实现：

```text
真实 committee models
  ↓
真实 committee prediction
  ↓
真实 force / energy deviation
  ↓
top-K 高不确定性构型选择
  ↓
模拟更新训练集规模
```

该实验说明项目已经从单模型 baseline、committee training、committee prediction 进一步推进到了主动学习闭环的 round-level 管理阶段。

需要说明的是，当前实验只是 offline active learning 的最小 round 记录，还没有真正把 selected frames 合并进新的训练集并重新训练 committee models。

下一步工作是：

```text
构造 expanded training dataset
  ↓
将 selected top-K frames 加入训练集
  ↓
重新训练 4 个 committee models
  ↓
比较 Round 0 和 Round 1 的测试误差
```
