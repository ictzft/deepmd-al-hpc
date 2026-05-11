# exp_003_single_model_baseline

## 1. 实验目的

本实验用于验证 toy H2 数据集上的最小 DeePMD 单模型训练流程是否能够跑通。

该实验重点不是追求模型精度，而是确认以下流程可以完整执行：

```text
toy H2 DeepMD 数据集
  ↓
dp train
  ↓
dp freeze
  ↓
dp test
  ↓
Energy / Force error summary
```

当前 toy H2 数据集仅用于流程验证，不用于真实分子体系或材料体系的科学精度评价。

---

## 2. 数据集信息

| 项目 | 内容 |
|---|---:|
| 体系 | toy H2 |
| 元素类型 | H |
| 原子数 | 2 |
| 训练集帧数 | 200 |
| 验证集帧数 | 50 |
| 训练集路径 | data/toy_h2/train |
| 验证集路径 | data/toy_h2/valid |

---

## 3. 数据生成方式

toy H2 数据集由以下脚本生成：

```text
scripts/data/make_toy_h2_deepmd.py
```

在 Docker 容器中执行：

```bash
python scripts/data/make_toy_h2_deepmd.py \
  --output data/toy_h2/train \
  --n-frames 200 \
  --seed 2026

python scripts/data/make_toy_h2_deepmd.py \
  --output data/toy_h2/valid \
  --n-frames 50 \
  --seed 2027
```

生成后的 DeepMD 数据文件包括：

```text
type.raw
type_map.raw
set.000/coord.npy
set.000/force.npy
set.000/energy.npy
set.000/box.npy
```

数据形状如下：

| Split | coord | force | energy | box |
|---|---:|---:|---:|---:|
| train | (200, 6) | (200, 6) | (200,) | (200, 9) |
| valid | (50, 6) | (50, 6) | (50,) | (50, 9) |

---

## 4. 训练配置

DeePMD 训练配置文件：

```text
configs/deepmd/toy_h2_input.json
```

实验目录：

```text
experiments/exp_003_single_model_baseline/
```

---

## 5. 训练结果摘要

| 指标 | 数值 |
|---|---:|
| Wall time | 9.411 s |
| Average training time | 0.0075 s/batch |

---

## 6. 测试结果

| 指标 | 数值 |
|---|---:|
| Energy MAE | 3.815557e-01 eV |
| Energy RMSE | 3.815592e-01 eV |
| Energy MAE / Natoms | 1.907779e-01 eV |
| Energy RMSE / Natoms | 1.907796e-01 eV |
| Force MAE | 2.702034e-02 eV/Å |
| Force RMSE | 7.977260e-02 eV/Å |

---

## 7. 实验结论

本实验说明 toy H2 数据集上的 DeePMD 单模型流程已经跑通，包括：

```text
数据生成
  ↓
模型训练
  ↓
模型冻结
  ↓
模型测试
  ↓
误差结果记录
```

当前结果只能证明 DeePMD train / freeze / test 工程链路可用，不能作为真实材料或分子体系上的模型精度结论。

下一阶段将在此基础上推进到 4 个不同随机种子的 DeePMD committee models 训练。
