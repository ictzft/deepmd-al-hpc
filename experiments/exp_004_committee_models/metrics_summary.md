# exp_004_committee_models

## 1. 实验目的

本实验用于验证 toy H2 数据集上的 4 个真实 DeePMD committee models 是否能够完成训练、冻结和测试流程。

该实验的重点不是追求 toy H2 模型精度，而是验证以下流程是否跑通：

```text
4 个不同随机种子 DeePMD 模型
  ↓
2×V100 分两批并行训练
  ↓
dp train
  ↓
dp freeze
  ↓
dp test
  ↓
汇总 Energy / Force 误差
```

该实验是从单模型 baseline 进入真实 committee models 的关键步骤，为后续真实 committee prediction 和 model deviation 计算做准备。

---

## 2. 实验环境

| 项目 | 内容 |
|---|---|
| 服务器 | shared-v100 |
| 工作目录 | /data/zft |
| GPU | 2 × V100 |
| DeepMD-kit | v3.1.4.dev81+geab341973 |
| 数据集 | toy H2 |
| 训练集 | /data/zft/data/toy_h2/train |
| 测试集 | /data/zft/data/toy_h2/valid |
| 测试帧数 | 50 |

---

## 3. Committee 配置

4 个模型采用相同的 DeePMD 网络结构和训练数据，但使用不同随机种子。

| Model | Seed | Config file |
|---|---:|---|
| model_000 | 1001 | configs/deepmd/committee/toy_h2_model_000.json |
| model_001 | 1002 | configs/deepmd/committee/toy_h2_model_001.json |
| model_002 | 1003 | configs/deepmd/committee/toy_h2_model_002.json |
| model_003 | 1004 | configs/deepmd/committee/toy_h2_model_003.json |

每个配置文件中同步修改了以下 3 个 seed：

```text
model.descriptor.seed
model.fitting_net.seed
training.seed
```

---

## 4. 训练调度方式

当前使用 2×V100，因此 4 个 committee models 分两批训练：

```text
Batch 1:
GPU 0 → model_000
GPU 1 → model_001

Batch 2:
GPU 0 → model_002
GPU 1 → model_003
```

运行脚本：

```bash
bash scripts/train/train_committee_models.sh
```

---

## 5. 训练结果摘要

| Model | Seed | Wall time | Average training time | Frozen model |
|---|---:|---:|---:|---|
| model_000 | 1001 | 10.250 s | 0.0081 s/batch | generated |
| model_001 | 1002 | 10.388 s | 0.0083 s/batch | generated |
| model_002 | 1003 | 10.514 s | 0.0083 s/batch | generated |
| model_003 | 1004 | 10.185 s | 0.0081 s/batch | generated |

4 个模型均成功生成 frozen model：

```text
experiments/exp_004_committee_models/model_000/frozen_model.pb
experiments/exp_004_committee_models/model_001/frozen_model.pb
experiments/exp_004_committee_models/model_002/frozen_model.pb
experiments/exp_004_committee_models/model_003/frozen_model.pb
```

每个 frozen model 大小约为 125 KB。

---

## 6. 测试结果

测试数据为：

```text
/data/zft/data/toy_h2/valid
```

测试帧数为 50。

| Model | Energy MAE (eV) | Energy RMSE (eV) | Energy MAE/Natoms (eV) | Energy RMSE/Natoms (eV) | Force MAE (eV/Å) | Force RMSE (eV/Å) |
|---|---:|---:|---:|---:|---:|---:|
| model_000 | 9.256462e-01 | 9.256704e-01 | 4.628231e-01 | 4.628352e-01 | 8.225294e-02 | 1.864015e-01 |
| model_001 | 1.187703e-02 | 1.193425e-02 | 5.938516e-03 | 5.967126e-03 | 3.372384e-02 | 1.486410e-01 |
| model_002 | 2.003550e+00 | 2.003717e+00 | 1.001775e+00 | 1.001859e+00 | 1.459888e-01 | 2.977629e-01 |
| model_003 | 1.121596e-03 | 1.708328e-03 | 5.607978e-04 | 8.541642e-04 | 3.641681e-02 | 9.575121e-02 |

---

## 7. 结果说明

本实验说明 4 个真实 DeePMD committee models 的训练、冻结和测试流程已经跑通。

不同随机种子下，4 个模型的测试误差存在明显差异。其中 model_001 和 model_003 的 Energy 误差较低，model_000 和 model_002 的 Energy 误差较高。这说明在当前 toy H2 小数据集、短训练步数和小网络设置下，不同初始化会带来较明显的训练结果差异。

由于该实验的主要目标是验证 committee models 的工程流程，因此当前结果不用于评价真实材料或分子体系上的模型精度。

从主动学习角度看，committee models 之间存在差异是后续计算 model deviation 的基础。下一阶段需要让这 4 个 frozen models 对同一个 candidate pool 进行预测，并将预测结果整理为：

```text
forces shape   = [n_models, n_frames, n_atoms, 3]
energies shape = [n_models, n_frames]
```

然后进一步计算真实 force / energy model deviation。

---

## 8. 实验结论

本实验完成了 toy H2 数据集上的真实 4-model DeePMD committee baseline。

已完成内容包括：

```text
4 个不同 seed 的 DeePMD 配置
  ↓
2×V100 分两批训练
  ↓
4 个模型分别 dp train
  ↓
4 个模型分别 dp freeze
  ↓
4 个模型分别 dp test
  ↓
生成 4 个 frozen_model.pb
  ↓
汇总训练时间和测试误差
```

该实验标志着项目已经从“单模型 DeePMD baseline”推进到“真实 committee models baseline”。

下一步工作是：

```text
exp_005_committee_prediction
  ↓
使用 4 个 frozen models 对同一 candidate pool 进行预测
  ↓
整理 energy / force prediction
  ↓
计算真实 model deviation
  ↓
选择 top-K 高不确定性构型
```
