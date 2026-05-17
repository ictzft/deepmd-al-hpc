# Reproduce Guide

本文档用于复现 `deepmd-al-hpc` 当前 toy H2 offline active learning 原型实验流程。

当前复现目标不是得到真实材料体系上的科学结论，而是验证以下工程链路：

```text
DeepMD-kit 环境验证
  ↓
toy H2 数据生成
  ↓
单模型 DeePMD baseline
  ↓
4-model committee training
  ↓
committee prediction
  ↓
force / energy model deviation
  ↓
top-K 高不确定性构型筛选
  ↓
selected frames 合并进训练集
  ↓
candidate pool 更新
  ↓
下一轮 committee retraining
  ↓
Round 0–3 summary 与 learning curve 汇总
```

当前 toy H2 主线实验已经形成如下闭环：

```text
Round 0: train 200, candidate 50
Round 1: train 210, candidate 40
Round 2: train 220, candidate 30
Round 3: train 230, candidate 20
```

当前最核心的实验现象是：

```text
force_dev_max_mean:
Round 0: 0.440989
Round 1: 0.269333
Round 2: 0.187412
Round 3: 0.170189
```

说明：本文档中的 toy H2 数据集只用于流程验证，不用于评价真实材料或分子体系上的模型精度。

---

## 1. 项目目录

宿主机上的项目目录为：

```bash
cd /data/zft
```

检查仓库状态：

```bash
git status
git log --oneline -5
```

建议在每次运行实验前确认工作区干净：

```bash
git status --short
```

---

## 2. 推荐运行方式

本项目采用宿主机代码目录挂载到 Docker 容器中的方式运行：

```text
宿主机 /data/zft
  ↕ 实时同步
Docker 容器 /data/zft
```

因此可以在宿主机中修改代码，在 Docker 容器中运行代码，并将实验结果同步保存回宿主机目录。

建议分工如下：

```text
宿主机：
- git status
- git add
- git commit
- git push
- README / docs 编辑

Docker 容器：
- dp train
- dp freeze
- dp test
- DeepPot prediction
- Python 实验脚本
```

---

## 3. DeepMD-kit Docker 环境

本项目真实 DeePMD 训练、冻结、测试和推理依赖 DeepMD-kit Docker 环境。

推荐使用仓库脚本进入容器：

```bash
bash scripts/docker/enter_deepmd_container.sh
```

如果需要手动启动容器，可以使用：

```bash
cd /data/zft

docker run --rm -it \
  --gpus all \
  --user $(id -u):$(id -g) \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp \
  -v /data/zft:/data/zft \
  -w /data/zft \
  ghcr.io/deepmodeling/deepmd-kit:master \
  bash
```

已验证的环境信息：

```text
镜像：ghcr.io/deepmodeling/deepmd-kit:master
DeepMD-kit：v3.1.3-81-geab34197
Python：/opt/deepmd-kit/bin/python
dp：/opt/deepmd-kit/bin/dp
lmp：/opt/deepmd-kit/bin/lmp
```

进入容器后验证：

```bash
dp -h
lmp -h
python -c "import deepmd; print('deepmd import ok')"
python -c "from deepmd.infer import DeepPot; print('DeepPot import ok')"
python -c "import numpy as np; print(np.__version__)"
nvidia-smi
```

如果在宿主机上执行 Python 脚本且没有 `python` 命令，可以使用：

```bash
python3
```

---

## 4. Torch 基础开发环境

Torch 环境主要用于运行主动学习框架 skeleton、model deviation 计算和部分 Python 工程脚本。

```text
镜像：cuda-torch:cuda11.3-cudnn8-ubuntu18.04-torch2.4
用途：
- 运行 Python 主动学习框架；
- 测试 force model deviation；
- 测试 top-K 构型筛选；
- 验证 2×V100 调度逻辑。
```

进入 Torch 容器：

```bash
bash scripts/docker/enter_torch_container.sh
```

---

## 5. 数据与实验文件约定

当前仓库采用如下约定：

```text
data/                         # 本地数据，不提交到 GitHub
configs/                      # 配置文件，提交到 GitHub
scripts/                      # 运行脚本，提交到 GitHub
src/                          # 框架源码，提交到 GitHub
experiments/                  # 只提交轻量实验摘要
docs/                         # 文档
```

不提交到 GitHub 的内容包括：

```text
原始数据集
处理后的大规模数据
DeePMD 训练 checkpoint
frozen_model.pb
*.npy
*.npz
*.log
lcurve.out
out.json
input_v2_compat.json
LAMMPS / MD 轨迹文件
Python 缓存文件
```

可以提交的轻量结果包括：

```text
metrics_summary.md
selected_topk.json
round*_summary.md
summary.csv
summary.md
small figures
```

---

## 6. Step 0：主动学习 skeleton 检查

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

真实 committee prediction 在 `exp_005_committee_prediction` 中完成。

---

## 7. Step 1：生成 toy H2 DeepMD 数据

当前 toy H2 数据只用于流程验证。

生成训练集：

```bash
python scripts/data/make_toy_h2_deepmd.py \
  --output data/toy_h2/train \
  --n-frames 200 \
  --seed 2026
```

生成验证集 / 初始 candidate pool：

```bash
python scripts/data/make_toy_h2_deepmd.py \
  --output data/toy_h2/valid \
  --n-frames 50 \
  --seed 2027
```

当前默认约定：

```text
data/toy_h2/train    # 初始训练集，200 frames
data/toy_h2/valid    # 初始 candidate pool / validation set，50 frames
```

---

## 8. Step 2：单模型 DeePMD baseline

训练配置文件：

```text
configs/deepmd/toy_h2_input.json
```

运行单模型训练：

```bash
bash scripts/train/train_single_model.sh
```

冻结模型：

```bash
bash scripts/eval/freeze_model.sh
```

测试模型：

```bash
bash scripts/eval/test_single_model.sh
```

对应实验目录：

```text
experiments/exp_003_single_model_baseline/
```

轻量结果摘要：

```text
experiments/exp_003_single_model_baseline/README.md
experiments/exp_003_single_model_baseline/metrics_summary.md
```

当前单模型 baseline 测试结果摘要：

| Metric | Value |
|---|---:|
| Energy MAE | 3.815557e-01 eV |
| Energy RMSE | 3.815592e-01 eV |
| Energy MAE/Natoms | 1.907779e-01 eV |
| Energy RMSE/Natoms | 1.907796e-01 eV |
| Force MAE | 2.702034e-02 eV/Å |
| Force RMSE | 7.977260e-02 eV/Å |

说明：该结果只用于验证 DeePMD 训练流程是否打通。

---

## 9. Step 3：训练初始 4-model committee

初始 committee 配置目录：

```text
configs/deepmd/committee/
```

包含 4 个不同随机种子的模型配置：

```text
toy_h2_model_000.json
toy_h2_model_001.json
toy_h2_model_002.json
toy_h2_model_003.json
```

运行 4-model committee 训练：

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

对应实验目录：

```text
experiments/exp_004_committee_models/
```

轻量结果摘要：

```text
experiments/exp_004_committee_models/metrics_summary.md
```

初始 4-model committee 配置：

| Model | Seed | Config file |
|---|---:|---|
| model_000 | 1001 | configs/deepmd/committee/toy_h2_model_000.json |
| model_001 | 1002 | configs/deepmd/committee/toy_h2_model_001.json |
| model_002 | 1003 | configs/deepmd/committee/toy_h2_model_002.json |
| model_003 | 1004 | configs/deepmd/committee/toy_h2_model_003.json |

---

## 10. Step 4：初始 committee prediction

使用 4 个 frozen DeePMD models 对初始 candidate pool 进行真实推理。

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

输出张量包括：

```text
energies
forces
virials
force_dev_max
force_dev_mean
energy_dev
selected_indices
```

对应实验目录：

```text
experiments/exp_005_committee_prediction/
```

轻量结果摘要：

```text
experiments/exp_005_committee_prediction/metrics_summary.md
experiments/exp_005_committee_prediction/selected_topk.json
```

说明：

```text
committee_predictions.npz
```

属于中间数组文件，不提交到 GitHub。

---

## 11. Step 5：生成 offline active learning selection 记录

将 `exp_005` 的 top-K 选择结果整理为一轮 offline active learning selection 记录。

运行：

```bash
PYTHONPATH=. python scripts/active_learning/run_offline_al_round.py \
  --prediction-json experiments/exp_005_committee_prediction/selected_topk.json \
  --output experiments/exp_006_offline_active_learning/round_001_selection.json
```

对应实验目录：

```text
experiments/exp_006_offline_active_learning/
```

轻量结果：

```text
experiments/exp_006_offline_active_learning/round_001_selection.json
experiments/exp_006_offline_active_learning/metrics_summary.md
```

---

## 12. Step 6：Round 0 → Round 1 数据更新

### 12.1 合并 selected frames，生成 Round 1 训练集

将 Round 0 选出的 top-10 high-uncertainty frames 合并到训练集。

```bash
python scripts/data/merge_selected_frames.py \
  --base data/toy_h2/train \
  --candidate data/toy_h2/valid \
  --selected-json experiments/exp_005_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_001_train
```

生成：

```text
data/toy_h2/round_001_train       # 210 frames
```

### 12.2 更新 candidate pool，生成 Round 1 candidate

从初始 candidate pool 中移除已经被选入训练集的 frames。

```bash
python scripts/data/make_remaining_candidate.py \
  --candidate data/toy_h2/valid \
  --selected-json experiments/exp_005_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_001_candidate
```

生成：

```text
data/toy_h2/round_001_candidate   # 40 frames
```

### 12.3 生成 Round 1 committee 配置

```bash
python scripts/config/make_round_committee_configs.py \
  --base-config configs/deepmd/toy_h2_input.json \
  --train-data data/toy_h2/round_001_train \
  --output-dir configs/deepmd/round_001_committee \
  --round-id 1 \
  --seed-base 1101
```

生成：

```text
configs/deepmd/round_001_committee/
```

### 12.4 训练 Round 1 committee models

```bash
bash scripts/train/train_round_committee_models.sh \
  configs/deepmd/round_001_committee \
  experiments/exp_007_round001_committee_models
```

对应实验目录：

```text
experiments/exp_007_round001_committee_models/
```

轻量结果：

```text
experiments/exp_007_round001_committee_models/metrics_summary.md
```

### 12.5 Round 1 committee prediction

```bash
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/round_001_candidate \
  --model-dir experiments/exp_007_round001_committee_models \
  --output experiments/exp_008_round001_committee_prediction/committee_predictions.npz \
  --selected-json experiments/exp_008_round001_committee_prediction/selected_topk.json \
  --top-k 10
```

对应实验目录：

```text
experiments/exp_008_round001_committee_prediction/
```

轻量结果：

```text
experiments/exp_008_round001_committee_prediction/selected_topk.json
```

---

## 13. Step 7：Round 1 → Round 2 数据更新

### 13.1 合并 Round 1 selected frames

```bash
python scripts/data/merge_selected_frames.py \
  --base data/toy_h2/round_001_train \
  --candidate data/toy_h2/round_001_candidate \
  --selected-json experiments/exp_008_round001_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_002_train
```

### 13.2 更新 Round 2 candidate pool

```bash
python scripts/data/make_remaining_candidate.py \
  --candidate data/toy_h2/round_001_candidate \
  --selected-json experiments/exp_008_round001_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_002_candidate
```

生成：

```text
data/toy_h2/round_002_train       # 220 frames
data/toy_h2/round_002_candidate   # 30 frames
```

### 13.3 生成 Round 2 committee 配置

```bash
python scripts/config/make_round_committee_configs.py \
  --base-config configs/deepmd/toy_h2_input.json \
  --train-data data/toy_h2/round_002_train \
  --output-dir configs/deepmd/round_002_committee \
  --round-id 2 \
  --seed-base 1201
```

### 13.4 训练 Round 2 committee models

```bash
bash scripts/train/train_round_committee_models.sh \
  configs/deepmd/round_002_committee \
  experiments/exp_009_round002_committee_models
```

对应实验目录：

```text
experiments/exp_009_round002_committee_models/
```

轻量结果：

```text
experiments/exp_009_round002_committee_models/metrics_summary.md
```

### 13.5 Round 2 committee prediction

```bash
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/round_002_candidate \
  --model-dir experiments/exp_009_round002_committee_models \
  --output experiments/exp_010_round002_committee_prediction/committee_predictions.npz \
  --selected-json experiments/exp_010_round002_committee_prediction/selected_topk.json \
  --top-k 10
```

对应实验目录：

```text
experiments/exp_010_round002_committee_prediction/
```

轻量结果：

```text
experiments/exp_010_round002_committee_prediction/selected_topk.json
```

---

## 14. Step 8：Round 2 → Round 3 数据更新

### 14.1 合并 Round 2 selected frames

```bash
python scripts/data/merge_selected_frames.py \
  --base data/toy_h2/round_002_train \
  --candidate data/toy_h2/round_002_candidate \
  --selected-json experiments/exp_010_round002_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_003_train
```

### 14.2 更新 Round 3 candidate pool

```bash
python scripts/data/make_remaining_candidate.py \
  --candidate data/toy_h2/round_002_candidate \
  --selected-json experiments/exp_010_round002_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_003_candidate
```

生成：

```text
data/toy_h2/round_003_train       # 230 frames
data/toy_h2/round_003_candidate   # 20 frames
```

### 14.3 生成 Round 3 committee 配置

```bash
python scripts/config/make_round_committee_configs.py \
  --base-config configs/deepmd/toy_h2_input.json \
  --train-data data/toy_h2/round_003_train \
  --output-dir configs/deepmd/round_003_committee \
  --round-id 3 \
  --seed-base 1301
```

### 14.4 训练 Round 3 committee models

```bash
bash scripts/train/train_round_committee_models.sh \
  configs/deepmd/round_003_committee \
  experiments/exp_011_round003_committee_models
```

对应实验目录：

```text
experiments/exp_011_round003_committee_models/
```

轻量结果：

```text
experiments/exp_011_round003_committee_models/metrics_summary.md
```

### 14.5 Round 3 committee prediction

```bash
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/round_003_candidate \
  --model-dir experiments/exp_011_round003_committee_models \
  --output experiments/exp_012_round003_committee_prediction/committee_predictions.npz \
  --selected-json experiments/exp_012_round003_committee_prediction/selected_topk.json \
  --top-k 10
```

对应实验目录：

```text
experiments/exp_012_round003_committee_prediction/
```

轻量结果：

```text
experiments/exp_012_round003_committee_prediction/selected_topk.json
experiments/exp_012_round003_committee_prediction/round003_summary.md
```

Round 3 prediction 结果：

```text
n_models: 4
n_frames: 20
n_atoms: 2
top_k: 10
prediction time: 7 s
```

---

## 15. Step 9：汇总 Round 0–3 learning curve

汇总脚本：

```bash
python scripts/analysis/summarize_al_rounds.py
```

绘图脚本：

```bash
python scripts/analysis/plot_al_rounds.py
```

输出文件：

```text
experiments/al_rounds_summary.csv
experiments/al_model_level_summary.csv
experiments/al_rounds_summary.md
experiments/figures/force_model_deviation_rounds.svg
experiments/figures/dataset_size_rounds.svg
experiments/figures/validation_rmse_rounds.svg
```

Round-level summary：

| Round | Train frames | Candidate frames | Force RMSE mean | force_dev_max mean | force_dev_max max | force_dev_max min |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 50 | 1.821392e-01 | 4.409891e-01 | 7.891949e-01 | 3.228673e-01 |
| 1 | 210 | 40 | 1.617669e-01 | 2.693333e-01 | 5.083387e-01 | 1.376306e-01 |
| 2 | 220 | 30 | 1.938590e-01 | 1.874125e-01 | 2.759882e-01 | 1.514366e-01 |
| 3 | 230 | 20 | 1.742648e-01 | 1.701889e-01 | 2.426791e-01 | 1.504961e-01 |

主要观察：

```text
随着主动学习轮次推进，训练集由 200 frames 增加到 230 frames，
候选池由 50 frames 缩减到 20 frames。

与此同时，top-K 高不确定性构型的平均 force model deviation
从 Round 0 的 0.440989 降低到 Round 3 的 0.170189。
```

验证集 Force RMSE 并没有严格单调下降：

```text
Round 0: 0.182139
Round 1: 0.161767
Round 2: 0.193859
Round 3: 0.174265
```

因此当前结果更适合表述为：

> 多轮主动学习后，候选池不确定性呈持续下降趋势；验证集 Force RMSE 整体处于同一量级，但受 toy 数据规模、随机初始化和 committee 模型差异影响，存在一定波动。

---

## 16. 当前实验目录说明

| 实验编号 | 实验名称 | 状态 | GitHub 轻量记录 | 说明 |
|---|---|---|---|---|
| exp_001 | env_check | 已完成 | deepmd_env_check.txt | 验证 Docker、DeepMD-kit、dp、lmp、Python import |
| exp_002 | framework_check | 已完成 | result.json | 基于模拟 committee forces 验证 deviation 和 top-K 选择 |
| exp_003 | single_model_baseline | 已完成 | README.md、metrics_summary.md | toy H2 单模型 DeePMD train / freeze / test |
| exp_004 | committee_models | 已完成 | metrics_summary.md | 4 个真实 DeePMD committee models 训练、冻结和测试 |
| exp_005 | committee_prediction | 已完成 | metrics_summary.md、selected_topk.json | 4 个 frozen models 真实预测、deviation 计算和 top-K 筛选 |
| exp_006 | offline_active_learning | 已完成 | metrics_summary.md、round_001_selection.json | 基于 top-K 结果形成一轮 offline AL selection 记录 |
| exp_007 | round001_committee_models | 已完成 | metrics_summary.md | 合并 Round 0 selected frames 后，重新训练 Round 1 的 4 个 committee models |
| exp_008 | round001_committee_prediction | 已完成 | selected_topk.json | 使用 Round 1 models 对 40 个剩余 candidate 进行预测和 top-K 筛选 |
| exp_009 | round002_committee_models | 已完成 | metrics_summary.md | 合并 Round 1 selected frames 后，重新训练 Round 2 的 4 个 committee models |
| exp_010 | round002_committee_prediction | 已完成 | selected_topk.json | 使用 Round 2 models 对 30 个剩余 candidate 进行预测和 top-K 筛选 |
| exp_011 | round003_committee_models | 已完成 | metrics_summary.md | 合并 Round 2 selected frames 后，重新训练 Round 3 的 4 个 committee models |
| exp_012 | round003_committee_prediction | 已完成 | round003_summary.md、selected_topk.json | 使用 Round 3 models 对 20 个剩余 candidate 进行预测和 top-K 筛选 |
| figures | learning curve figures | 已完成 | svg figures | 生成 Round 0–3 的 deviation、dataset size 和 RMSE 曲线 |

---

## 17. 代码与配置检查

Python 语法检查：

```bash
python3 -m py_compile \
  scripts/active_learning/run_framework_check.py \
  scripts/active_learning/run_offline_al_round.py \
  scripts/inference/predict_committee_models.py \
  scripts/data/merge_selected_frames.py \
  scripts/data/make_remaining_candidate.py \
  scripts/config/make_round_committee_configs.py \
  scripts/analysis/summarize_al_rounds.py \
  scripts/analysis/plot_al_rounds.py \
  src/metrics/deviation.py \
  src/al/scheduler.py \
  src/al/loop.py
```

Shell 语法检查：

```bash
bash -n \
  scripts/train/train_single_model.sh \
  scripts/train/train_committee_models.sh \
  scripts/train/train_round_committee_models.sh \
  scripts/eval/freeze_model.sh \
  scripts/eval/test_single_model.sh
```

JSON 配置检查：

```bash
python3 -m json.tool configs/deepmd/toy_h2_input.json > /tmp/check_toy_h2_input.json

for f in configs/deepmd/round_001_committee/*.json \
         configs/deepmd/round_002_committee/*.json \
         configs/deepmd/round_003_committee/*.json; do
  python3 -m json.tool "$f" > /tmp/check_round_config.json
done
```

Git 状态检查：

```bash
git status --short
git ls-files | grep -E "exp_008|exp_010|exp_007|exp_009|exp_011"
git ls-files | grep -E "深度势能|第一性原理|高性能加速" || true
```

预期结果：

```text
working tree clean
exp_008 / exp_010 selected_topk.json 已被追踪
exp_007 / exp_009 / exp_011 metrics_summary.md 已被追踪
根目录误生成的中文空文件不再被追踪
```

---

## 18. 当前已知限制

当前项目仍处于原型验证阶段，主要限制包括：

1. 当前 toy H2 数据集仅用于流程验证，不能代表真实材料或分子体系上的模型精度；
2. 当前已经完成 dataset-level offline active learning 多轮闭环，但尚未引入真实 DFT / AIMD 数据集；
3. 当前已经完成 Round 0–3 committee retraining、prediction 和 learning curve 汇总，但尚未加入 random sampling baseline；
4. 当前还不能证明 model deviation sampling 一定优于随机采样；
5. 当前尚未引入结构多样性选择策略，仅基于 `force_dev_max` 进行 top-K 选择；
6. 当前尚未进行 H100 多 GPU 加速实验；
7. 当前尚未进行真实 DFT labeling 或在线主动学习调度，仅使用已有 toy 数据模拟 offline labeling；
8. 当前 V100 profiling 只记录了 Round 3 的初步训练与预测耗时，尚未系统记录 Round 0–2 的端到端耗时。

---

## 19. GitHub 提交建议

提交前检查：

```bash
git status --short
git diff
```

轻量结果可以提交：

```text
metrics_summary.md
selected_topk.json
round*_summary.md
summary.csv
summary.md
small figures
```

大型文件不要提交：

```text
*.npy
*.npz
*.pb
model.ckpt*
checkpoint
*.log
lcurve.out
out.json
input_v2_compat.json
```

提交 `docs/reproduce.md`：

```bash
git add docs/reproduce.md
git commit -m "Update reproduction guide for active learning pipeline"
git push origin main
```

---

## 20. 后续迁移到 H100 时的复用方式

迁移到 H100 时，本文件仍然作为主流程参考。主要需要替换或补充：

```text
Docker / module 环境
GPU 数量
训练脚本中的 GPU 调度逻辑
多模型并行策略
profiling 统计方式
H100 mixed precision 设置
```

H100 阶段重点记录：

```text
单模型训练时间
4-model committee 总训练时间
candidate prediction 时间
model deviation 计算时间
单轮 active learning 端到端耗时
V100 vs H100 加速比
GPU 利用率与显存占用
```

---

## 21. 后续计划

下一阶段重点工作：

```text
1. 加入 random sampling baseline
2. 迁移到真实 DFT / AIMD 数据集
3. 补充 V100 profiling
4. 在 H100 多 GPU 平台上评估端到端加速效果
```

random sampling baseline 计划：

```text
model deviation sampling
vs.
random sampling
```

需要比较：

```text
Force RMSE
Energy RMSE
候选池不确定性下降趋势
训练耗时
推理耗时
端到端 active learning wall-clock time
```

预期输出：

```text
experiments/random_baseline/
experiments/random_vs_deviation_summary.csv
experiments/figures/random_vs_deviation.svg
```

---

## 22. 当前阶段总结

当前项目已经完成：

```text
环境验证
  ↓
主动学习 skeleton
  ↓
toy H2 单模型 DeePMD train / freeze / test
  ↓
4 个真实 DeePMD committee models 训练
  ↓
4 个 frozen models 真实预测
  ↓
真实 force / energy model deviation
  ↓
top-K 高不确定性构型选择
  ↓
offline active learning round 记录
  ↓
selected frames 合并进新训练集
  ↓
Round 1 committee models 重新训练
  ↓
Round 1 candidate pool 重新预测与筛选
  ↓
Round 2 committee models 重新训练
  ↓
Round 2 candidate pool 重新预测与筛选
  ↓
Round 3 committee models 重新训练
  ↓
Round 3 candidate pool 重新预测与筛选
  ↓
Round 0–3 summary 表格生成
  ↓
Round 0–3 learning curve 图生成
```

当前仓库已经从 **selection-level 记录** 推进到 **dataset-level offline active learning 多轮闭环原型 + 初步结果分析阶段**。

下一步关键是从：

```text
多轮流程跑通 + learning curve 初步整理
```

推进到：

```text
有对照实验、真实数据验证和高性能加速证据
```