# exp_005_committee_prediction

## 1. 实验目的

本实验用于验证 4 个真实 DeePMD committee models 能否对同一个 candidate pool 进行批量预测，并基于预测结果计算真实 model deviation。

该实验是从 exp_004 的真实 committee models 训练，进一步推进到真实 committee prediction 和不确定性评估的关键步骤。

实验流程如下：

```text
4 个 frozen DeePMD models
  ↓
读取同一个 toy H2 candidate pool
  ↓
分别预测 energy / force / virial
  ↓
整理 committee prediction tensor
  ↓
计算 force deviation 和 energy deviation
  ↓
选择 top-K 高不确定性构型
```

---

## 2. 输入数据

candidate pool 使用 toy H2 valid 数据集：

```text
data/toy_h2/valid
```

数据规模：

| 项目 | 数值 |
|---|---:|
| n_models | 4 |
| n_frames | 50 |
| n_atoms | 2 |
| top_k | 10 |

---

## 3. 使用的 frozen models

```text
experiments/exp_004_committee_models/model_000/frozen_model.pb
experiments/exp_004_committee_models/model_001/frozen_model.pb
experiments/exp_004_committee_models/model_002/frozen_model.pb
experiments/exp_004_committee_models/model_003/frozen_model.pb
```

---

## 4. 运行脚本

推理脚本：

```text
scripts/inference/predict_committee_models.py
```

运行命令：

```bash
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/valid \
  --output experiments/exp_005_committee_prediction/committee_predictions.npz \
  --selected-json experiments/exp_005_committee_prediction/selected_topk.json \
  --top-k 10
```

---

## 5. 输出张量形状

本实验成功生成 committee prediction 结果：

| Tensor | Shape | 含义 |
|---|---:|---|
| energies | (4, 50) | 4 个模型对 50 帧构型的能量预测 |
| forces | (4, 50, 2, 3) | 4 个模型对 50 帧、2 个原子的力预测 |
| virials | (4, 50, 9) | 4 个模型对 50 帧构型的 virial 预测 |
| force_dev_max | (50,) | 每帧最大 force deviation |
| force_dev_mean | (50,) | 每帧平均 force deviation |
| energy_dev | (50,) | 每帧 energy deviation |
| selected_indices | (10,) | top-10 高不确定性构型索引 |

---

## 6. Top-10 高不确定性构型

本实验基于 `force_dev_max` 选择 top-10 高不确定性构型：

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

## 7. 结果说明

本实验说明 4 个真实 DeePMD frozen models 已经能够对同一个 candidate pool 进行预测，并成功整理为统一的 committee prediction 张量。

其中：

```text
energies shape = [n_models, n_frames]
forces shape   = [n_models, n_frames, n_atoms, 3]
```

这正是后续主动学习中计算 model deviation 所需要的核心数据结构。

本实验中的 candidate pool 仍然是 toy H2 valid 数据集，因此结果仅用于验证工程流程，不用于真实材料或分子体系的科学结论。

---

## 8. 实验结论

本实验完成了从真实 committee models 到真实 committee prediction 的关键步骤。

已完成内容包括：

```text
加载 4 个 frozen_model.pb
  ↓
读取 toy H2 candidate pool
  ↓
预测 energy / force / virial
  ↓
生成 committee prediction tensors
  ↓
计算 force / energy model deviation
  ↓
选择 top-10 高不确定性构型
```

该实验标志着项目已经从“真实 committee models baseline”进一步推进到“真实 model deviation 计算与构型筛选”。

下一步工作是：

```text
exp_006_offline_active_learning
  ↓
构建初始训练集 / candidate pool 划分
  ↓
基于 model deviation 选择新构型
  ↓
模拟 labeling
  ↓
重新训练 committee models
  ↓
形成最小 offline active learning 闭环
```
