# toy H2 主线实验

本文档用于记录 `deepmd-al-hpc` 当前 toy H2 原型实验的主线流程，包括：

```text
主动学习 skeleton 检查
  ↓
toy H2 DeepMD 数据生成
  ↓
单模型 DeePMD baseline
  ↓
初始 4-model committee training
  ↓
初始 committee prediction
  ↓
force / energy model deviation
  ↓
Round 0 selected_topk.json
```

该流程的目标不是得到真实材料体系上的科学结论，而是验证 DeePMD 主动学习闭环中最基础的工程链路是否可运行。

---

## 1. 实验目标

toy H2 主线实验主要用于验证：

```text
DeepMD-kit 是否能正常训练 toy 数据
toy H2 数据是否能生成 DeepMD npy 格式
单模型 DeePMD baseline 是否能 train / freeze / test
4 个 DeePMD committee models 是否能正常训练
多个 frozen models 是否能对 candidate pool 做 prediction
是否能计算 force / energy model deviation
是否能基于 force_dev_max 选出 top-K 高不确定性构型
```

当前 toy H2 数据集只用于流程验证：

```text
不代表真实材料体系
不代表真实 DFT / AIMD 数据复杂度
不用于得出真实物理或化学结论
```

---

## 2. 前置条件

在运行本文档命令前，请先完成环境配置。

推荐阅读：

```text
docs/setup.md
```

进入项目目录：

```bash
cd /data/zft
```

建议在 DeepMD Docker 容器中运行本文档中的大多数命令：

```bash
bash scripts/docker/enter_deepmd_container.sh
```

进入容器后确认路径：

```bash
pwd
```

预期输出：

```text
/data/zft
```

---

## 3. 主动学习 Skeleton 检查

该步骤使用模拟 committee forces，验证主动学习框架是否打通。

运行命令：

```bash
PYTHONPATH=. python scripts/active_learning/run_framework_check.py \
  --config configs/active_learning/framework_check.json
```

输出文件：

```text
experiments/exp_002_framework_check/result.json
```

说明：

```text
exp_002 中的 committee forces 来自随机数模拟；
该实验只用于验证 deviation 计算、top-K 选择和调度逻辑。
```

真实 DeePMD committee prediction 在后续 `exp_005_committee_prediction` 中完成。

---

## 4. 生成 toy H2 DeepMD 数据

当前 toy H2 数据只用于流程验证。

### 4.1 生成初始训练集

```bash
python scripts/data/make_toy_h2_deepmd.py \
  --output data/toy_h2/train \
  --n-frames 200 \
  --seed 2026
```

输出目录：

```text
data/toy_h2/train
```

该目录作为初始训练集，包含：

```text
200 frames
2 atoms per frame
coord.npy
box.npy
energy.npy
force.npy
```

---

### 4.2 生成初始 candidate pool / validation set

```bash
python scripts/data/make_toy_h2_deepmd.py \
  --output data/toy_h2/valid \
  --n-frames 50 \
  --seed 2027
```

输出目录：

```text
data/toy_h2/valid
```

该目录当前同时作为：

```text
initial candidate pool
validation / test set
```

说明：

```text
当前 toy H2 阶段为了简化流程，将 valid 同时用于 candidate pool 和测试；
后续真实 DFT / AIMD 数据集阶段，应明确区分 train / candidate / test。
```

---

## 5. 检查 toy H2 数据形状

运行：

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

for d in [Path("data/toy_h2/train"), Path("data/toy_h2/valid")]:
    print(f"\n--- {d} ---")
    for name in ["coord.npy", "box.npy", "energy.npy", "force.npy"]:
        p = d / "set.000" / name
        arr = np.load(p)
        print(name, arr.shape)
PY
```

预期输出形状：

```text
data/toy_h2/train:
coord.npy  : (200, 6)
box.npy    : (200, 9)
energy.npy : (200,)
force.npy  : (200, 6)

data/toy_h2/valid:
coord.npy  : (50, 6)
box.npy    : (50, 9)
energy.npy : (50,)
force.npy  : (50, 6)
```

含义：

```text
coord.npy:
  2 个 H 原子，每个原子 3 个坐标，因此每帧 6 维

box.npy:
  每帧 3×3 box matrix，展平为 9 维

energy.npy:
  每帧一个总能量

force.npy:
  2 个 H 原子，每个原子 3 个力分量，因此每帧 6 维
```

---

## 6. 单模型 DeePMD Baseline

该步骤用于验证最基础的 DeePMD train / freeze / test 链路。

训练配置文件：

```text
configs/deepmd/toy_h2_input.json
```

---

### 6.1 训练单模型

```bash
bash scripts/train/train_single_model.sh
```

---

### 6.2 冻结模型

```bash
bash scripts/eval/freeze_model.sh
```

---

### 6.3 测试模型

```bash
bash scripts/eval/test_single_model.sh
```

---

### 6.4 输出目录

对应实验目录：

```text
experiments/exp_003_single_model_baseline/
```

轻量结果摘要：

```text
experiments/exp_003_single_model_baseline/README.md
experiments/exp_003_single_model_baseline/metrics_summary.md
```

---

### 6.5 当前单模型 baseline 测试结果

| 指标 | 数值 |
|---|---:|
| Energy MAE | 3.815557e-01 eV |
| Energy RMSE | 3.815592e-01 eV |
| Energy MAE/Natoms | 1.907779e-01 eV |
| Energy RMSE/Natoms | 1.907796e-01 eV |
| Force MAE | 2.702034e-02 eV/Å |
| Force RMSE | 7.977260e-02 eV/Å |

说明：

```text
该结果只用于验证 DeePMD 训练流程是否打通，
不是本文最终要强调的科学结论。
```

---

## 7. 初始 4-model DeePMD Committee Training

主动学习需要通过多个模型之间的预测分歧估计不确定性。

当前初始 committee 包含 4 个 DeePMD 模型：

```text
model_000
model_001
model_002
model_003
```

每个模型使用不同随机种子训练。

---

### 7.1 初始 committee 配置目录

```text
configs/deepmd/committee/
```

包含 4 个配置文件：

```text
toy_h2_model_000.json
toy_h2_model_001.json
toy_h2_model_002.json
toy_h2_model_003.json
```

初始 4-model committee 配置：

| 模型 | 随机种子 | 配置文件 |
|---|---:|---|
| model_000 | 1001 | configs/deepmd/committee/toy_h2_model_000.json |
| model_001 | 1002 | configs/deepmd/committee/toy_h2_model_001.json |
| model_002 | 1003 | configs/deepmd/committee/toy_h2_model_002.json |
| model_003 | 1004 | configs/deepmd/committee/toy_h2_model_003.json |

---

### 7.2 训练 4-model committee

运行：

```bash
bash scripts/train/train_committee_models.sh
```

当前在 2×V100 上采用两批训练：

```text
Batch 1:
GPU 0 → model_000
GPU 1 → model_001

Batch 2:
GPU 0 → model_002
GPU 1 → model_003
```

---

### 7.3 输出目录

对应实验目录：

```text
experiments/exp_004_committee_models/
```

轻量结果摘要：

```text
experiments/exp_004_committee_models/metrics_summary.md
```

训练完成后，每个模型目录中应包含：

```text
frozen_model.pb
checkpoint
model.ckpt*
train.log
test.log
lcurve.out
```

其中：

```text
frozen_model.pb、checkpoint、model.ckpt*、log 文件默认不提交 GitHub
metrics_summary.md 可以提交 GitHub
```

---

## 8. 初始 Committee Prediction

该步骤使用 4 个 frozen DeePMD models 对初始 candidate pool 进行真实推理。

推理脚本：

```text
scripts/inference/predict_committee_models.py
```

---

### 8.1 运行初始 committee prediction

```bash
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/valid \
  --models \
    experiments/exp_004_committee_models/model_000/frozen_model.pb \
    experiments/exp_004_committee_models/model_001/frozen_model.pb \
    experiments/exp_004_committee_models/model_002/frozen_model.pb \
    experiments/exp_004_committee_models/model_003/frozen_model.pb \
  --output experiments/exp_005_committee_prediction/committee_predictions.npz \
  --selected-json experiments/exp_005_committee_prediction/selected_topk.json \
  --top-k 10
```

---

### 8.2 输出目录

对应实验目录：

```text
experiments/exp_005_committee_prediction/
```

输出文件：

```text
experiments/exp_005_committee_prediction/committee_predictions.npz
experiments/exp_005_committee_prediction/selected_topk.json
experiments/exp_005_committee_prediction/metrics_summary.md
```

说明：

```text
committee_predictions.npz 属于中间数组文件，不提交到 GitHub。
selected_topk.json 可以提交 GitHub。
metrics_summary.md 可以提交 GitHub。
```

---

## 9. Committee Prediction 输出内容

`committee_predictions.npz` 中主要包含：

```text
energies
forces
virials
force_dev_max
force_dev_mean
energy_dev
selected_indices
coord
box
atype
ref_energy
ref_force
```

各字段含义：

```text
energies:
  多个 committee models 对 candidate frames 的 energy prediction

forces:
  多个 committee models 对 candidate frames 的 force prediction

virials:
  多个 committee models 对 candidate frames 的 virial prediction

force_dev_max:
  每个 candidate frame 上最大的 force model deviation

force_dev_mean:
  每个 candidate frame 上的平均 force model deviation

energy_dev:
  多个 committee models 之间的 energy prediction deviation

selected_indices:
  基于 force_dev_max 选择出的 top-K frame indices

coord / box / atype:
  candidate frame 的结构信息

ref_energy / ref_force:
  toy dataset 中已有的参考 energy / force
```

---

## 10. Round 0 Top-K 高不确定性构型选择

当前 toy H2 原型中，主要使用：

```text
force_dev_max
```

作为高不确定性构型筛选指标。

筛选策略：

```text
按 force_dev_max 从大到小排序
选择前 top-K 个 frames
```

当前设置：

```text
top-K = 10
```

输出：

```text
experiments/exp_005_committee_prediction/selected_topk.json
```

该文件记录：

```text
selected frame indices
force_dev_max
energy deviation
selection strategy
top-K size
```

这些 selected frames 后续会被合并进训练集，用于 Round 1 committee retraining。

---

## 11. 当前阶段对应实验目录

| 实验编号 | 实验名称 | 状态 | 说明 |
|---|---|---|---|
| exp_002 | framework_check | 已完成 | 基于模拟 committee forces 验证 deviation 和 top-K 选择 |
| exp_003 | single_model_baseline | 已完成 | toy H2 单模型 DeePMD train / freeze / test |
| exp_004 | committee_models | 已完成 | 4 个真实 DeePMD committee models 训练、冻结和测试 |
| exp_005 | committee_prediction | 已完成 | 4 个 frozen models 真实预测、deviation 计算和 top-K 筛选 |

---

## 12. 当前阶段输出文件汇总

### 12.1 数据文件

```text
data/toy_h2/train
data/toy_h2/valid
```

说明：

```text
data/ 目录为本地数据目录，默认不提交到 GitHub。
```

---

### 12.2 单模型结果

```text
experiments/exp_003_single_model_baseline/README.md
experiments/exp_003_single_model_baseline/metrics_summary.md
```

---

### 12.3 初始 Committee 训练结果

```text
experiments/exp_004_committee_models/metrics_summary.md
```

大文件不提交：

```text
experiments/exp_004_committee_models/model_*/frozen_model.pb
experiments/exp_004_committee_models/model_*/checkpoint
experiments/exp_004_committee_models/model_*/model.ckpt*
experiments/exp_004_committee_models/model_*/train.log
experiments/exp_004_committee_models/model_*/test.log
experiments/exp_004_committee_models/model_*/lcurve.out
```

---

### 12.4 初始 Committee Prediction 结果

```text
experiments/exp_005_committee_prediction/metrics_summary.md
experiments/exp_005_committee_prediction/selected_topk.json
```

大文件不提交：

```text
experiments/exp_005_committee_prediction/committee_predictions.npz
```

---

## 13. 当前阶段完成后，下一步做什么？

完成本文档流程后，应继续阅读：

```text
docs/uncertainty_rounds.md
```

下一步将基于：

```text
experiments/exp_005_committee_prediction/selected_topk.json
```

执行：

```text
selected frames 合并进训练集
candidate pool 更新
Round 1 committee config 生成
Round 1 committee retraining
Round 1 candidate prediction
Round 2 / Round 3 多轮闭环
```

也就是从：

```text
selection-level top-K result
```

推进到：

```text
dataset-level offline active learning multi-round closed loop
```

---

## 14. 注意事项

1. 本文档中的 toy H2 数据只用于流程验证；
2. 当前 valid set 同时作为 candidate pool 和 validation set，后续真实数据实验中应进一步拆分；
3. `committee_predictions.npz` 文件较大，不提交 GitHub；
4. `frozen_model.pb`、checkpoint、log 文件不提交 GitHub；
5. `selected_topk.json` 和 `metrics_summary.md` 属于轻量结果，可以提交；
6. 如果命令中出现 DeepPot import 失败，请确认已经进入 DeepMD Docker 容器；
7. 如果 Python 命令失败，可确认当前环境是否为 `/opt/deepmd-kit/bin/python`；
8. 如果 GPU 不可见，请检查 Docker 是否使用 `--gpus all`。

---

## 15. 小结

本文档完成了 toy H2 主线实验的前半部分：

```text
framework check
  ↓
toy H2 data generation
  ↓
single-model DeePMD baseline
  ↓
initial 4-model committee training
  ↓
initial committee prediction
  ↓
force / energy model deviation
  ↓
Round 0 uncertainty top-K selection
```

该阶段完成后，项目已经具备继续执行 multi-round offline active learning 的基础。

下一阶段重点是：

```text
Round 0 selected frames
  ↓
merge into train set
  ↓
update candidate pool
  ↓
retrain Round 1 committee models
  ↓
repeat until Round 3
```

相关文档：

```text
docs/uncertainty_rounds.md
```