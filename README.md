# deepmd-al-hpc

本项目是一个面向 **第一性原理机器学习势函数主动学习闭环** 的多模型并行与高性能训练原型系统。

项目不直接训练大语言模型，也不是直接复现 Megatron-LM，而是借鉴 Megatron 系列大规模训练框架中的 **多 GPU 并行、micro-batch、混合精度、批量推理、流水线调度和分布式实验管理思想**，将这些高性能训练思想迁移到 DeePMD / DeepMD-kit 机器学习势函数主动学习场景中。

当前阶段主要在 **2×Tesla V100 GPU** 平台上完成项目原型验证，包括 Docker 环境验证、DeepMD-kit 环境验证、主动学习框架 skeleton、toy H2 单模型 DeePMD baseline、4 个真实 DeePMD committee models 训练、真实 committee prediction、model deviation 计算、top-K 高不确定性构型筛选，以及 dataset-level offline active learning 多轮闭环验证。

截至 **2026-05-17**，项目已经从“单模型 baseline + 模拟 committee forces”推进到“真实 committee models + 真实 model deviation + dataset-level offline active learning 多轮闭环 + learning curve 结果分析”。当前已经完成 Round 0、Round 1、Round 2 和 Round 3 的数据扩展、committee model 训练、冻结、测试和候选构型筛选，并已整理 Round 0–3 的主动学习汇总表与 learning curve 图。

当前 toy H2 offline active learning 主线实验已经形成如下闭环：

```text
Round 0: train 200, candidate 50
Round 1: train 210, candidate 40
Round 2: train 220, candidate 30
Round 3: train 230, candidate 20
```

随着主动学习轮次推进，top-K 高不确定性构型的平均 force model deviation 从 Round 0 的 `0.440989` 降低到 Round 3 的 `0.170189`，说明 committee models 在候选构型空间上的预测分歧逐步减小。当前结果主要用于验证主动学习闭环和工程流程，不用于评价真实材料体系精度。后续将进一步加入 random sampling baseline、真实 DFT / AIMD 数据集迁移，以及 H100 多 GPU 加速实验。

---

## 一、研究方向

本项目属于 **AI for Science 与高性能计算交叉方向**，主要关注：

```text
第一性原理计算
  ↓
机器学习势函数
  ↓
主动学习采样
  ↓
多模型并行训练
  ↓
GPU / H100 高性能加速
```

暂定中文题目：

> 面向第一性原理机器学习势函数主动学习闭环的多模型并行训练框架研究

暂定英文题目：

> A Multi-Model Parallel Active Learning Framework for First-Principles Machine Learning Potentials

也可以进一步概括为：

> A High-Performance Active Learning Framework for First-Principles Machine Learning Potentials

---

## 二、项目目标

本项目面向第一性原理机器学习势函数的主动学习闭环训练过程，研究如何在保证势函数精度和分子动力学稳定性的前提下，提高训练、推理、不确定性评估和构型筛选的整体效率。

具体目标包括：

1. 基于 DeePMD / DeepMD-kit 跑通机器学习势函数基线训练流程；
2. 构建多个 committee models，用于模型不确定性估计；
3. 基于 force / energy / virial deviation 计算候选构型的不确定性；
4. 实现 offline active learning 闭环，模拟 DFT labeling 过程；
5. 借鉴 Megatron-style 的并行训练思想，实现多模型并行训练和批量推理；
6. 在 2×V100 平台上完成原型验证、结果汇总和初步性能 profiling；
7. 为后续 H100 多 GPU 平台上的主动学习闭环加速实验做准备。

---

## 三、为什么借鉴 Megatron-style 思想？

Megatron-LM / Megatron-Core 面向的是大语言模型的大规模分布式训练，其核心思想包括：

- 多 GPU 并行训练；
- micro-batch 切分；
- 混合精度训练；
- pipeline 调度；
- 分布式 checkpoint；
- 大规模训练任务管理。

本项目不直接使用 Megatron 训练 DeePMD，而是借鉴这些系统设计思想，解决机器学习势函数主动学习闭环中的类似问题：

```text
Megatron 面对的问题：
大模型训练太慢，需要多 GPU 并行和流水线调度。

本项目面对的问题：
主动学习势函数闭环太慢，需要多模型并行训练、批量推理和自动化调度。
```

在本项目中，Megatron-style 思想主要对应为：

| Megatron-style 思想 | 本项目中的对应设计 |
|---|---|
| Data Parallel | 多 GPU 训练 DeePMD 模型 |
| Expert / Multi-Model Parallel | committee models 并行训练 |
| Micro-batching | 候选构型池批量推理 |
| Pipeline Scheduling | 训练、推理、deviation、筛选的流水线化 |
| Mixed Precision | 后续在 A100 / H100 上探索 BF16 / FP16 / TF32 |
| Distributed Checkpoint | 管理每轮主动学习的模型、数据划分和筛选结果 |
| Profiling | 分析训练、推理、deviation 和数据处理耗时 |

---

## 四、主动学习闭环流程

本项目计划实现如下主动学习闭环：

```text
初始 DFT 数据集
  ↓
训练多个 DeePMD committee models
  ↓
在候选构型池上进行批量推理
  ↓
计算 force / energy / virial model deviation
  ↓
根据不确定性和结构多样性筛选高价值构型
  ↓
将筛选出的构型加入训练集
  ↓
重新训练 committee models
  ↓
进入下一轮主动学习
```

当前阶段优先实现 **offline active learning**：

```text
已有完整 DFT 数据集
  ↓
每轮主动学习只允许模型看到其中一部分标签
  ↓
被选中的构型才加入训练集
  ↓
用已有数据模拟真实 DFT labeling
```

这样可以避免前期大量真实 DFT 计算开销，先验证主动学习和高性能训练框架是否可行。

---

## 五、系统框架设计

本项目整体分为四层。

### 1. 模型训练层

负责调用 DeepMD-kit 训练单个 DeePMD 模型。

输入：

- DeepMD 配置文件；
- 训练集路径；
- 随机种子；
- GPU id。

输出：

- 模型 checkpoint；
- frozen model；
- 训练日志；
- loss 曲线；
- 能量、力和 virial 误差。

### 2. 多模型并行层

负责训练多个 committee models。

在 2×V100 阶段，采用两批训练：

```text
Batch 1:
GPU 0 → model_000
GPU 1 → model_001

Batch 2:
GPU 0 → model_002
GPU 1 → model_003
```

在后续 H100 阶段，计划扩展为：

```text
GPU 0 → model_000
GPU 1 → model_001
GPU 2 → model_002
GPU 3 → model_003
```

### 3. 批量推理与不确定性评估层

负责让多个 committee models 对候选构型池进行预测，并计算 model deviation。

候选构型池会被切分为多个 micro-batches：

```text
candidate_pool
  ↓
batch_000
batch_001
batch_002
...
```

多个模型分别预测：

- energy；
- force；
- virial。

然后计算：

- force deviation；
- energy deviation；
- virial deviation；
- combined uncertainty score。

当前 toy H2 原型中，已经实现了基于 4 个 frozen models 的真实 committee prediction，并基于 `force_dev_max` 选择 top-K 高不确定性构型。

### 4. 主动学习调度层

负责组织完整闭环：

```text
Round 0:
  构造初始训练集
  训练初始 committee models
  对 candidate pool 进行预测
  计算 model deviation
  筛选 top-K 构型

Round 1:
  将 selected frames 合并进训练集
  更新 remaining candidate pool
  重新训练 committee models
  继续推理和筛选

Round 2:
  继续扩展训练集
  继续更新 candidate pool
  继续重新训练 committee models
  继续推理和筛选

Round 3:
  继续扩展训练集
  继续更新 candidate pool
  继续重新训练 committee models
  对剩余 candidate pool 进行预测和筛选
```

当前已经从 selection-level 记录推进到 dataset-level 多轮闭环，完成了 Round 0–3 的 committee retraining、prediction、model deviation 计算和结果汇总。

---

## 六、项目目录结构

当前仓库结构如下：

```text
deepmd-al-hpc/
├── README.md
├── .gitignore
├── configs/
│   ├── active_learning/           # 主动学习配置
│   └── deepmd/                    # DeePMD input.json 与 committee 配置
│       ├── committee/
│       ├── round_001_committee/
│       ├── round_002_committee/
│       └── round_003_committee/
├── scripts/
│   ├── active_learning/           # 主动学习框架检查与 offline AL round 脚本
│   ├── analysis/                  # 多轮 AL 结果汇总与 learning curve 生成脚本
│   ├── config/                    # 多轮 committee 配置生成脚本
│   ├── data/                      # toy 数据生成、selected frames 合并与 candidate 更新脚本
│   ├── docker/                    # Docker 运行脚本
│   ├── eval/                      # freeze、test、误差评估脚本
│   ├── inference/                 # committee models 推理脚本
│   └── train/                     # 单模型、committee models 和 round retraining 脚本
├── src/
│   ├── al/                        # 主动学习循环、筛选和调度
│   ├── metrics/                   # model deviation 与误差指标
│   └── utils/                     # IO、日志、随机种子等工具
├── experiments/
│   ├── exp_001_env_check/
│   ├── exp_002_framework_check/
│   ├── exp_003_single_model_baseline/
│   ├── exp_004_committee_models/
│   ├── exp_005_committee_prediction/
│   ├── exp_006_offline_active_learning/
│   ├── exp_007_round001_committee_models/
│   ├── exp_008_round001_committee_prediction/
│   ├── exp_009_round002_committee_models/
│   ├── exp_010_round002_committee_prediction/
│   ├── exp_011_round003_committee_models/
│   ├── exp_012_round003_committee_prediction/
│   ├── figures/
│   ├── al_model_level_summary.csv
│   ├── al_rounds_summary.csv
│   └── al_rounds_summary.md
└── docs/
```

说明：`experiments/` 中的大型训练日志、`.pb` 模型、checkpoint、`.npy` 和 `.npz` 文件不提交到 GitHub；GitHub 中只保留代码、配置文件和轻量结果摘要。

后续计划补充：

```text
slurm/                         # Slurm 作业脚本
results/                       # 小型结果摘要
notebooks/                     # 绘图和分析 notebook
```

---

## 七、运行环境

### 1. Torch 基础开发环境

用于运行主动学习框架 skeleton、model deviation 计算和部分 Python 工程脚本。

```text
镜像：cuda-torch:cuda11.3-cudnn8-ubuntu18.04-torch2.4
用途：
- 运行 Python 主动学习框架；
- 测试 force model deviation；
- 测试 top-K 构型筛选；
- 验证 2×V100 调度逻辑。
```

进入容器：

```bash
bash scripts/docker/enter_torch_container.sh
```

### 2. DeepMD-kit 训练环境

用于真实 DeePMD 训练、冻结、测试和推理。

```text
镜像：ghcr.io/deepmodeling/deepmd-kit:master
DeepMD-kit：v3.1.3-81-geab34197
Python：/opt/deepmd-kit/bin/python
dp：/opt/deepmd-kit/bin/dp
lmp：/opt/deepmd-kit/bin/lmp
```

进入容器：

```bash
bash scripts/docker/enter_deepmd_container.sh
```

或者直接启动：

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

已验证命令：

```bash
dp -h
lmp -h
python -c "import deepmd; print('deepmd import ok')"
python -c "from deepmd.infer import DeepPot; print('DeepPot import ok')"
python -c "import numpy as np; print(np.__version__)"
nvidia-smi
```

环境验证日志：

```text
experiments/exp_001_env_check/deepmd_env_check.txt
```

---

## 八、Docker 目录同步方式

本项目采用宿主机代码目录挂载到 Docker 容器中的方式运行：

```bash
cd /data/zft

docker run --rm -it \
  --gpus all \
  --user $(id -u):$(id -g) \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e HOME=/tmp \
  -v /data/zft:/data/zft \
  -w /data/zft \
  <docker-image> \
  bash
```

实现效果：

```text
宿主机 /data/zft
  ↕ 实时同步
Docker 容器 /data/zft
```

因此可以在宿主机中修改代码，在 Docker 容器中运行代码，并将实验结果同步保存回宿主机目录。

为避免 Docker 生成 root 权限文件，容器启动脚本中默认使用：

```bash
--user $(id -u):$(id -g)
-e PYTHONDONTWRITEBYTECODE=1
```

如果容器中出现：

```text
I have no name!
```

通常是因为容器内没有宿主机用户 ID 对应的用户名记录，不影响代码运行和文件写入。但由于容器用户可能没有完整 Git 身份和 SSH 环境，建议在 **宿主机** 上执行 `git commit` 和 `git push`。

---

## 九、当前已完成内容

### 1. 第 1 周：环境与仓库初始化

状态：已完成。

已完成任务：

- [x] 初始化 GitHub 仓库；
- [x] 将仓库 clone 到 shared-v100 服务器 `/data/zft`；
- [x] 配置 GitHub SSH 推送；
- [x] 创建项目基础目录结构；
- [x] 编写中文 README；
- [x] 创建 `.gitignore`，避免提交缓存、日志、模型和数据文件；
- [x] 清理并移除已被 Git 跟踪的 `__pycache__` 和 `.pyc` 文件；
- [x] 验证 Docker 可以调用 2×V100 GPU；
- [x] 打通宿主机 `/data/zft` 与 Docker 容器 `/data/zft` 的目录同步；
- [x] 跑通主动学习 framework check；
- [x] 基于模拟 committee forces 验证 force model deviation 与 top-K 构型筛选逻辑；
- [x] 验证 DeepMD-kit Docker 环境；
- [x] 验证 `dp -h`、`lmp -h` 和 `python import deepmd`；
- [x] 完成代码 commit 并 push 到 GitHub。

### 2. 第 2 周：单模型 DeePMD baseline

状态：已完成最小闭环。

已完成任务：

- [x] 补充 toy H2 数据生成脚本；
- [x] 生成 toy H2 DeepMD 数据集；
- [x] 编写 DeePMD 训练配置 `configs/deepmd/toy_h2_input.json`；
- [x] 运行 `dp train`；
- [x] 运行 `dp freeze`；
- [x] 运行 `dp test`；
- [x] 输出 Energy / Force 误差；
- [x] 保存单模型 baseline 实验结果；
- [x] 编写 `experiments/exp_003_single_model_baseline/metrics_summary.md`。

当前 toy H2 数据集只用于流程验证，不用于真实科学结论。

### 3. 第 3 周：committee models 与真实 model deviation

状态：已完成真实 4-model committee baseline、committee prediction 和最小 offline AL selection 记录。

已完成任务：

- [x] 生成 4 份不同随机种子的 DeePMD 配置；
- [x] 编写 `scripts/train/train_committee_models.sh`；
- [x] 在 2×V100 上分两批训练 4 个 DeePMD 模型；
- [x] 分别执行 `dp freeze` 得到 4 个 `frozen_model.pb`；
- [x] 分别执行 `dp test`，汇总 Energy / Force 误差；
- [x] 编写 `experiments/exp_004_committee_models/metrics_summary.md`；
- [x] 编写 `scripts/inference/predict_committee_models.py`；
- [x] 使用 4 个 frozen models 对同一个 candidate pool 进行真实预测；
- [x] 整理 `energies`、`forces`、`virials` 等 committee prediction 张量；
- [x] 计算 `force_dev_max`、`force_dev_mean` 和 `energy_dev`；
- [x] 基于 `force_dev_max` 选择 top-10 高不确定性构型；
- [x] 编写 `experiments/exp_005_committee_prediction/metrics_summary.md`；
- [x] 编写 `scripts/active_learning/run_offline_al_round.py`；
- [x] 生成 `round_001_selection.json`；
- [x] 编写 `experiments/exp_006_offline_active_learning/metrics_summary.md`。

### 4. 第 4 周：dataset-level offline active learning 多轮闭环

状态：已完成 Round 1、Round 2 与 Round 3 数据闭环。

已完成任务：

- [x] 编写 `scripts/data/merge_selected_frames.py`；
- [x] 编写 `scripts/data/make_remaining_candidate.py`；
- [x] 编写 `scripts/config/make_round_committee_configs.py`；
- [x] 编写 `scripts/train/train_round_committee_models.sh`；
- [x] 生成 `data/toy_h2/round_001_train`，训练集从 200 frames 扩展到 210 frames；
- [x] 生成 `data/toy_h2/round_001_candidate`，候选池从 50 frames 缩减到 40 frames；
- [x] 生成 Round 1 的 4 个 DeePMD committee 配置；
- [x] 在 2×V100 上完成 Round 1 的 4 个 committee models 训练、冻结和测试；
- [x] 使用 Round 1 committee models 对 40 个剩余候选构型进行真实预测；
- [x] 计算 Round 1 candidate pool 的 force / energy model deviation；
- [x] 选择 Round 2 需要加入训练集的 top-10 高不确定性构型；
- [x] 生成 `data/toy_h2/round_002_train`，训练集从 210 frames 扩展到 220 frames；
- [x] 生成 `data/toy_h2/round_002_candidate`，候选池从 40 frames 缩减到 30 frames；
- [x] 生成 Round 2 的 4 个 DeePMD committee 配置；
- [x] 在 2×V100 上完成 Round 2 的 4 个 committee models 训练、冻结和测试；
- [x] 使用 Round 2 committee models 对 30 个剩余候选构型进行真实预测；
- [x] 选择 Round 3 需要加入训练集的 top-10 高不确定性构型；
- [x] 生成 `data/toy_h2/round_003_train`，训练集从 220 frames 扩展到 230 frames；
- [x] 生成 `data/toy_h2/round_003_candidate`，候选池从 30 frames 缩减到 20 frames；
- [x] 生成 Round 3 的 4 个 DeePMD committee 配置；
- [x] 在 2×V100 上完成 Round 3 的 4 个 committee models 训练、冻结和测试；
- [x] 使用 Round 3 committee models 对 20 个剩余候选构型进行真实预测；
- [x] 计算 Round 3 candidate pool 的 force / energy model deviation；
- [x] 选择 Round 3 prediction 中的 top-10 高不确定性构型。

### 5. V100 收尾实验：Round 0–3 汇总与 learning curve

状态：已完成。

已完成任务：

- [x] 编写 `scripts/analysis/summarize_al_rounds.py`；
- [x] 汇总 Round 0–3 的训练集规模；
- [x] 汇总 Round 0–3 的 candidate pool 规模；
- [x] 汇总 Round 0–3 的 4-model Energy / Force 测试误差；
- [x] 汇总每轮 top-K `force_dev_max` 的最大值、均值和最小值；
- [x] 输出 `experiments/al_rounds_summary.csv`；
- [x] 输出 `experiments/al_model_level_summary.csv`；
- [x] 输出 `experiments/al_rounds_summary.md`；
- [x] 编写 `scripts/analysis/plot_al_rounds.py`；
- [x] 生成 `experiments/figures/force_model_deviation_rounds.svg`；
- [x] 生成 `experiments/figures/dataset_size_rounds.svg`；
- [x] 生成 `experiments/figures/validation_rmse_rounds.svg`；
- [x] 将 Round 3 配置、轻量结果、summary 和 analysis 脚本 commit 并 push 到 GitHub。

---

## 十、实验进展总览

| 实验编号 | 实验名称 | 状态 | 说明 |
|---|---|---|---|
| exp_001 | env_check | 已完成 | 验证 Docker、DeepMD-kit、dp、lmp、Python import |
| exp_002 | framework_check | 已完成 | 基于模拟 committee forces 验证 deviation 和 top-K 选择 |
| exp_003 | single_model_baseline | 已完成 | toy H2 单模型 DeePMD train / freeze / test |
| exp_004 | committee_models | 已完成 | 4 个真实 DeePMD committee models 训练、冻结和测试 |
| exp_005 | committee_prediction | 已完成 | 4 个 frozen models 真实预测、deviation 计算和 top-K 筛选 |
| exp_006 | offline_active_learning | 已完成 | 基于 top-K 结果形成一轮 offline AL selection 记录 |
| exp_007 | round001_committee_models | 已完成 | 合并 top-10 selected frames 后，重新训练 Round 1 的 4 个 committee models |
| exp_008 | round001_committee_prediction | 已完成 | 使用 Round 1 models 对 40 个剩余 candidate 进行预测和 top-K 筛选 |
| exp_009 | round002_committee_models | 已完成 | 合并 Round 1 selected frames 后，重新训练 Round 2 的 4 个 committee models |
| exp_010 | round002_committee_prediction | 已完成 | 使用 Round 2 models 对 30 个剩余 candidate 进行预测和 top-K 筛选 |
| exp_011 | round003_committee_models | 已完成 | 合并 Round 2 selected frames 后，重新训练 Round 3 的 4 个 committee models |
| exp_012 | round003_committee_prediction | 已完成 | 使用 Round 3 models 对 20 个剩余 candidate 进行预测和 top-K 筛选 |
| figures | learning curve figures | 已完成 | 生成 Round 0–3 的 deviation、dataset size 和 RMSE 曲线 |

---

## 十一、主动学习 skeleton 验证结果

当前已经跑通最小主动学习框架：

```text
模拟 4 个 committee models 的 force prediction
  ↓
计算 force model deviation
  ↓
按照不确定性分数排序
  ↓
选择 top-K 高不确定性构型
  ↓
保存筛选结果
```

运行命令：

```bash
PYTHONPATH=. python scripts/active_learning/run_framework_check.py \
  --config configs/active_learning/framework_check.json
```

输出文件：

```text
experiments/exp_002_framework_check/result.json
```

需要说明的是：`exp_002` 中的 committee forces 来自随机数模拟，主要用于验证主动学习流程、deviation 计算和调度逻辑是否打通。真实 committee prediction 已在 `exp_005_committee_prediction` 中完成。

---

## 十二、单模型 DeePMD baseline 结果

当前使用一个最小 toy H2 数据集进行流程验证。

```text
体系：H2 toy system
元素类型：H
原子数：2
训练集帧数：200
验证集帧数：50

训练集路径：
/data/zft/data/toy_h2/train

验证集路径：
/data/zft/data/toy_h2/valid
```

数据生成脚本：

```text
scripts/data/make_toy_h2_deepmd.py
```

数据生成命令：

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

训练配置：

```text
configs/deepmd/toy_h2_input.json
```

实验目录：

```text
experiments/exp_003_single_model_baseline/
```

训练结果摘要：

```text
finished training
wall time: 9.411 s
average training time: 0.0075 s/batch
```

测试结果如下：

| Metric | Value |
|---|---:|
| Energy MAE | 3.815557e-01 eV |
| Energy RMSE | 3.815592e-01 eV |
| Energy MAE/Natoms | 1.907779e-01 eV |
| Energy RMSE/Natoms | 1.907796e-01 eV |
| Force MAE | 2.702034e-02 eV/Å |
| Force RMSE | 7.977260e-02 eV/Å |

说明：当前结果只用于验证 DeePMD 训练流程是否打通，不用于评价真实材料或分子体系上的模型精度。

---

## 十三、真实 4-model committee baseline 结果

在 `exp_004_committee_models` 中，项目训练了 4 个不同随机种子的 DeePMD 模型。

| Model | Seed | Config file |
|---|---:|---|
| model_000 | 1001 | configs/deepmd/committee/toy_h2_model_000.json |
| model_001 | 1002 | configs/deepmd/committee/toy_h2_model_001.json |
| model_002 | 1003 | configs/deepmd/committee/toy_h2_model_002.json |
| model_003 | 1004 | configs/deepmd/committee/toy_h2_model_003.json |

运行脚本：

```bash
bash scripts/train/train_committee_models.sh
```

训练调度方式：

```text
Batch 1:
GPU 0 → model_000
GPU 1 → model_001

Batch 2:
GPU 0 → model_002
GPU 1 → model_003
```

训练结果摘要：

| Model | Seed | Wall time | Average training time | Frozen model |
|---|---:|---:|---:|---|
| model_000 | 1001 | 10.250 s | 0.0081 s/batch | generated |
| model_001 | 1002 | 10.388 s | 0.0083 s/batch | generated |
| model_002 | 1003 | 10.514 s | 0.0083 s/batch | generated |
| model_003 | 1004 | 10.185 s | 0.0081 s/batch | generated |

测试结果：

| Model | Energy MAE (eV) | Energy RMSE (eV) | Energy MAE/Natoms (eV) | Energy RMSE/Natoms (eV) | Force MAE (eV/Å) | Force RMSE (eV/Å) |
|---|---:|---:|---:|---:|---:|---:|
| model_000 | 9.256462e-01 | 9.256704e-01 | 4.628231e-01 | 4.628352e-01 | 8.225294e-02 | 1.864015e-01 |
| model_001 | 1.187703e-02 | 1.193425e-02 | 5.938516e-03 | 5.967126e-03 | 3.372384e-02 | 1.486410e-01 |
| model_002 | 2.003550e+00 | 2.003717e+00 | 1.001775e+00 | 1.001859e+00 | 1.459888e-01 | 2.977629e-01 |
| model_003 | 1.121596e-03 | 1.708328e-03 | 5.607978e-04 | 8.541642e-04 | 3.641681e-02 | 9.575121e-02 |

说明：不同随机种子下模型误差存在明显差异，这为后续 committee model deviation 提供了基础。

---

## 十四、真实 committee prediction 与 model deviation

在 `exp_005_committee_prediction` 中，项目使用 4 个 frozen models 对同一个 toy H2 candidate pool 进行真实预测。

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

输出张量形状：

| Tensor | Shape | 含义 |
|---|---:|---|
| energies | (4, 50) | 4 个模型对 50 帧构型的能量预测 |
| forces | (4, 50, 2, 3) | 4 个模型对 50 帧、2 个原子的力预测 |
| virials | (4, 50, 9) | 4 个模型对 50 帧构型的 virial 预测 |
| force_dev_max | (50,) | 每帧最大 force deviation |
| force_dev_mean | (50,) | 每帧平均 force deviation |
| energy_dev | (50,) | 每帧 energy deviation |
| selected_indices | (10,) | top-10 高不确定性构型索引 |

Top-10 高不确定性构型：

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

说明：`committee_predictions.npz` 属于实验输出数据文件，不提交到 GitHub；`selected_topk.json` 和 `metrics_summary.md` 用于记录轻量结果。

---

## 十五、dataset-level offline active learning 多轮闭环

在 `exp_006_offline_active_learning` 中，项目首先将 `exp_005` 的 top-K 选择结果整理为一轮 offline active learning selection 记录。随后项目进一步从 selection-level 推进到 dataset-level：真正把 selected frames 合并进训练集，更新 candidate pool，并重新训练下一轮 committee models。

### 1. Round 0 → Round 1

Round 0 使用初始训练集与候选池：

| 项目 | 数值 |
|---|---:|
| initial training frames | 200 |
| initial candidate frames | 50 |
| selected frames | 10 |
| training frames after selection | 210 |
| candidate frames after selection | 40 |

新增脚本：

```text
scripts/data/merge_selected_frames.py
scripts/data/make_remaining_candidate.py
```

生成数据集：

```text
data/toy_h2/round_001_train       # 210 frames
data/toy_h2/round_001_candidate   # 40 frames
```

Round 1 committee 配置：

```text
configs/deepmd/round_001_committee/
```

Round 1 committee models：

```text
experiments/exp_007_round001_committee_models/
```

Round 1 prediction：

```text
experiments/exp_008_round001_committee_prediction/
```

Round 1 top-10 选择结果基于 `round_001_candidate` 的 frame index：

| Rank | Frame index | force_dev_max | energy_dev |
|---:|---:|---:|---:|
| 1 | 10 | 5.083387e-01 | 4.475698e-01 |
| 2 | 2 | 4.616380e-01 | 4.490363e-01 |
| 3 | 36 | 3.822970e-01 | 4.466175e-01 |
| 4 | 17 | 3.003161e-01 | 4.488546e-01 |
| 5 | 22 | 2.623882e-01 | 4.497980e-01 |
| 6 | 25 | 2.012675e-01 | 4.460668e-01 |
| 7 | 5 | 1.566108e-01 | 4.467225e-01 |
| 8 | 14 | 1.438941e-01 | 4.476540e-01 |
| 9 | 24 | 1.389516e-01 | 4.480902e-01 |
| 10 | 0 | 1.376306e-01 | 4.471448e-01 |

### 2. Round 1 → Round 2

Round 2 数据扩展结果：

| 项目 | 数值 |
|---|---:|
| previous training frames | 210 |
| previous candidate frames | 40 |
| selected frames | 10 |
| training frames after selection | 220 |
| candidate frames after selection | 30 |

生成数据集：

```text
data/toy_h2/round_002_train       # 220 frames
data/toy_h2/round_002_candidate   # 30 frames
```

Round 2 committee 配置：

```text
configs/deepmd/round_002_committee/
```

Round 2 committee models：

```text
experiments/exp_009_round002_committee_models/
```

Round 2 prediction：

```text
experiments/exp_010_round002_committee_prediction/
```

Round 2 top-10 选择结果基于 `round_002_candidate` 的 frame index：

| Rank | Frame index | force_dev_max | energy_dev |
|---:|---:|---:|---:|
| 1 | 13 | 2.759882e-01 | 1.548132e+00 |
| 2 | 12 | 2.491643e-01 | 1.549469e+00 |
| 3 | 6 | 2.040958e-01 | 1.551253e+00 |
| 4 | 9 | 1.746328e-01 | 1.551423e+00 |
| 5 | 24 | 1.746072e-01 | 1.551418e+00 |
| 6 | 8 | 1.724995e-01 | 1.551186e+00 |
| 7 | 26 | 1.621692e-01 | 1.552673e+00 |
| 8 | 16 | 1.559771e-01 | 1.556179e+00 |
| 9 | 25 | 1.535541e-01 | 1.551759e+00 |
| 10 | 20 | 1.514366e-01 | 1.553624e+00 |

### 3. Round 2 → Round 3

Round 3 数据扩展结果：

| 项目 | 数值 |
|---|---:|
| previous training frames | 220 |
| previous candidate frames | 30 |
| selected frames | 10 |
| training frames after selection | 230 |
| candidate frames after selection | 20 |

生成数据集：

```text
data/toy_h2/round_003_train       # 230 frames
data/toy_h2/round_003_candidate   # 20 frames
```

Round 3 committee 配置：

```text
configs/deepmd/round_003_committee/
```

Round 3 committee models：

```text
experiments/exp_011_round003_committee_models/
```

Round 3 prediction：

```text
experiments/exp_012_round003_committee_prediction/
```

Round 3 top-10 选择结果基于 `round_003_candidate` 的 frame index：

| Rank | Frame index | force_dev_max | energy_dev |
|---:|---:|---:|---:|
| 1 | 5 | 2.426791e-01 | 1.080004e+00 |
| 2 | 3 | 1.903137e-01 | 1.079452e+00 |
| 3 | 0 | 1.664500e-01 | 1.079340e+00 |
| 4 | 19 | 1.605250e-01 | 1.078957e+00 |
| 5 | 4 | 1.604581e-01 | 1.079509e+00 |
| 6 | 14 | 1.594691e-01 | 1.079818e+00 |
| 7 | 13 | 1.593928e-01 | 1.078720e+00 |
| 8 | 18 | 1.585817e-01 | 1.080024e+00 |
| 9 | 8 | 1.535236e-01 | 1.080936e+00 |
| 10 | 12 | 1.504961e-01 | 1.078207e+00 |

当前多轮 offline active learning 数据闭环状态：

| Round | Training frames | Candidate frames before selection | Selected frames | Candidate frames after selection | Committee retraining |
|---:|---:|---:|---:|---:|---|
| 0 | 200 | 50 | 10 | 40 | completed in exp_004 |
| 1 | 210 | 40 | 10 | 30 | completed in exp_007 |
| 2 | 220 | 30 | 10 | 20 | completed in exp_009 |
| 3 | 230 | 20 | 10 | - | completed in exp_011 |

说明：当前 toy H2 数据集只用于流程验证。由于训练步数较少、toy 数据规模极小，不同随机种子模型的 Energy / Force 误差仍存在波动，当前结果不用于评价真实材料体系精度。

---

## 十六、Round 0–3 learning curve 汇总结果

当前已经完成 Round 0–3 的主动学习结果汇总。

汇总脚本：

```text
scripts/analysis/summarize_al_rounds.py
```

绘图脚本：

```text
scripts/analysis/plot_al_rounds.py
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

### 1. Round-level summary

| Round | Train frames | Candidate frames | Force RMSE mean | force_dev_max mean | force_dev_max max | force_dev_max min |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 50 | 1.821392e-01 | 4.409891e-01 | 7.891949e-01 | 3.228673e-01 |
| 1 | 210 | 40 | 1.617669e-01 | 2.693333e-01 | 5.083387e-01 | 1.376306e-01 |
| 2 | 220 | 30 | 1.938590e-01 | 1.874125e-01 | 2.759882e-01 | 1.514366e-01 |
| 3 | 230 | 20 | 1.742648e-01 | 1.701889e-01 | 2.426791e-01 | 1.504961e-01 |

### 2. 主要观察

随着主动学习轮次推进，训练集由 200 frames 增加到 230 frames，候选池由 50 frames 缩减到 20 frames。与此同时，top-K 高不确定性构型的平均 force model deviation 从 Round 0 的 `0.440989` 降低到 Round 3 的 `0.170189`。

这说明基于 committee model deviation 的主动学习采样逐步降低了剩余候选池中高不确定性构型的模型分歧。

需要注意的是，验证集 Force RMSE 并没有严格单调下降：

```text
Round 0: 0.182139
Round 1: 0.161767
Round 2: 0.193859
Round 3: 0.174265
```

因此当前结果更适合表述为：

> 多轮主动学习后，候选池不确定性呈持续下降趋势；验证集 Force RMSE 整体处于同一量级，但受 toy 数据规模、随机初始化和 committee 模型差异影响，存在一定波动。

---

## 十七、Round 3 committee retraining 与 prediction 结果

Round 3 使用的数据集为：

```text
Training set:
data/toy_h2/round_003_train       # 230 frames

Candidate pool:
data/toy_h2/round_003_candidate   # 20 frames

Validation set:
data/toy_h2/valid                 # 50 frames
```

Round 3 committee 配置：

| Model | Seed | Config file |
|---|---:|---|
| model_000 | 1301 | configs/deepmd/round_003_committee/toy_h2_round003_model_000.json |
| model_001 | 1302 | configs/deepmd/round_003_committee/toy_h2_round003_model_001.json |
| model_002 | 1303 | configs/deepmd/round_003_committee/toy_h2_round003_model_002.json |
| model_003 | 1304 | configs/deepmd/round_003_committee/toy_h2_round003_model_003.json |

Round 3 测试结果：

| Model | Energy RMSE (eV) | Energy RMSE/Natoms (eV) | Force RMSE (eV/Å) |
|---|---:|---:|---:|
| model_000 | 7.858708e-01 | 3.929354e-01 | 3.768867e-01 |
| model_001 | 6.801369e-01 | 3.400684e-01 | 1.493383e-01 |
| model_002 | 6.787318e-01 | 3.393659e-01 | 1.070028e-01 |
| model_003 | 1.851586e+00 | 9.257932e-01 | 6.383155e-02 |

Round 3 prediction 结果：

```text
n_models: 4
n_frames: 20
n_atoms: 2
top_k: 10
prediction time: 7 s
```

说明：Round 3 中 `model_003` 的 Energy RMSE 明显偏高，但 Force RMSE 较低；同时 selected frames 的 energy deviation 基本集中在 1.078–1.081 附近。因此当前主动学习筛选主要按 force-based model deviation 解释，不将 energy deviation 作为主要筛选依据。

---

## 十八、当前已知限制

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

## 十九、项目不上传的内容

以下内容不应提交到 GitHub：

- 原始数据集；
- 处理后的大规模数据；
- DeePMD 训练 checkpoint；
- `.pb` 模型文件；
- 大型日志文件；
- 大规模实验结果；
- `.npy` / `.npz` 数据文件；
- 轨迹文件；
- LAMMPS 输出文件；
- 大型图片和中间结果；
- Python 缓存文件；
- Docker 临时文件；
- DeePMD 自动生成的 `lcurve.out`、`out.json`、`input_v2_compat.json`。

这些内容应保存在服务器本地或单独的数据目录中。GitHub 中只保留代码、配置文件和轻量实验摘要。

当前已提交到 GitHub 的轻量结果包括：

```text
configs/deepmd/round_003_committee/
experiments/al_rounds_summary.csv
experiments/al_model_level_summary.csv
experiments/al_rounds_summary.md
experiments/figures/
experiments/exp_012_round003_committee_prediction/selected_topk.json
experiments/exp_012_round003_committee_prediction/round003_summary.md
scripts/analysis/
```

未提交的大型或中间结果包括：

```text
committee_predictions.npz
run_round003_prediction.log
frozen_model.pb
model.ckpt
lcurve.out
```

---

## 二十、下一阶段计划

### 阶段 1：random sampling baseline

为了验证 model deviation sampling 的意义，需要增加随机采样对照组：

```text
model deviation sampling
vs.
random sampling
```

计划内容：

- [ ] 从相同 candidate pool 中随机选择 top-K 数量的构型；
- [ ] 构造 random sampling 的 round_001 / round_002 / round_003 train set；
- [ ] 训练 random baseline committee models；
- [ ] 比较 Force RMSE、Energy RMSE 和候选池不确定性下降趋势；
- [ ] 形成 model deviation sampling 与 random sampling 的对比曲线。

预期输出：

```text
experiments/random_baseline/
experiments/random_vs_deviation_summary.csv
experiments/figures/random_vs_deviation.svg
```

### 阶段 2：真实 DFT / AIMD 数据集迁移

在 toy H2 流程验证完成后，迁移到更接近真实应用的数据：

- [ ] 准备小规模真实 DFT / AIMD 数据；
- [ ] 转换为 DeepMD npy 格式；
- [ ] 设置真实体系的 DeePMD 输入配置；
- [ ] 复用当前 offline AL pipeline；
- [ ] 分析真实体系上的 model deviation 与构型筛选效果。

### 阶段 3：V100 profiling 补充

当前已记录 Round 3 的初步耗时：

```text
Round 3 committee training elapsed time: 76 s
Round 3 prediction elapsed time: 7 s
```

后续需要系统补充：

- [ ] 单模型训练耗时；
- [ ] 4-model committee 串行训练耗时；
- [ ] 2×V100 并行训练耗时；
- [ ] candidate prediction 耗时；
- [ ] model deviation 计算耗时；
- [ ] 单个 active learning round 的端到端耗时。

### 阶段 4：H100 多 GPU 加速实验

在 H100 多 GPU 平台上进一步评估：

- [ ] 训练时间；
- [ ] 推理吞吐；
- [ ] model deviation 计算时间；
- [ ] 多 GPU 加速比；
- [ ] 端到端 active learning wall-clock time。

---

## 二十一、代码与配置检查

在每次重要修改后，建议执行以下检查：

```bash
wc -l \
  scripts/active_learning/run_framework_check.py \
  scripts/active_learning/run_offline_al_round.py \
  scripts/inference/predict_committee_models.py \
  scripts/data/merge_selected_frames.py \
  scripts/data/make_remaining_candidate.py \
  scripts/config/make_round_committee_configs.py \
  scripts/analysis/summarize_al_rounds.py \
  scripts/analysis/plot_al_rounds.py \
  scripts/train/train_single_model.sh \
  scripts/train/train_committee_models.sh \
  scripts/train/train_round_committee_models.sh \
  scripts/eval/freeze_model.sh \
  scripts/eval/test_single_model.sh \
  src/metrics/deviation.py \
  src/al/scheduler.py \
  .gitignore \
  configs/deepmd/toy_h2_input.json

python -m py_compile \
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

bash -n \
  scripts/train/train_single_model.sh \
  scripts/train/train_committee_models.sh \
  scripts/train/train_round_committee_models.sh \
  scripts/eval/freeze_model.sh \
  scripts/eval/test_single_model.sh

python -m json.tool configs/deepmd/toy_h2_input.json > /tmp/check_toy_h2_input.json

for f in configs/deepmd/round_001_committee/*.json \
         configs/deepmd/round_002_committee/*.json \
         configs/deepmd/round_003_committee/*.json; do
  python -m json.tool "$f" > /tmp/check_round_config.json
done
```

说明：宿主机环境中可能没有 `python` 命令，因此这里默认建议在 DeepMD Docker 容器中执行检查；如果在宿主机上执行，可将 `python` 替换为 `python3`。

---

## 二十二、预期研究贡献

本项目后续希望形成以下几个方面的贡献：

1. **面向机器学习势函数主动学习闭环的多模型并行训练框架**  
   将 committee models 作为并行训练单元，提高多轮主动学习中模型训练效率。

2. **面向候选构型池的批量推理与 model deviation 评估方法**  
   通过 micro-batch 和多 GPU 推理，提高大规模候选构型不确定性评估效率。

3. **面向 DFT 标注节省的主动学习采样策略**  
   结合模型不确定性和结构多样性，减少冗余构型标注。

4. **面向 H100 平台的主动学习闭环加速实验**  
   从训练时间、推理吞吐、model deviation 计算时间、多 GPU 加速比和端到端 wall-clock time 等方面评估性能提升。

---

## 二十三、最终研究主线

本项目最终希望形成如下研究主线：

```text
DFT / AIMD 精度高但计算昂贵
  ↓
DeePMD / DeepMD-kit 学习第一性原理势能面
  ↓
主动学习减少 DFT 标注冗余
  ↓
committee models 和 model deviation 评估构型不确定性
  ↓
借鉴 Megatron-style 思想进行多模型并行训练与批量推理
  ↓
GPU / H100 加速训练、推理和主动学习闭环
  ↓
形成面向 AI for Science 的高性能主动学习势函数训练框架
```

---

## 二十四、当前阶段总结

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

当前仓库已经从 **第一阶段工程原型** 推进到 **dataset-level offline active learning 多轮闭环原型 + 初步结果分析阶段**。项目已经不再停留在 selection-level 记录，而是完成了 selected frames 合并、remaining candidate 更新、下一轮 committee retraining、下一轮不确定性筛选，以及多轮结果汇总。

当前已经形成如下最小多轮闭环：

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

这说明随着主动学习轮次推进，剩余候选池中的高不确定性构型平均 force model deviation 持续降低，committee models 在候选构型空间中的预测分歧逐步减小。

下一步关键是从：

```text
多轮流程跑通 + learning curve 初步整理
```

推进到：

```text
有对照实验、真实数据验证和高性能加速证据
```

也就是加入 random sampling baseline，迁移到真实 DFT / AIMD 数据集，并进一步在 H100 平台上评估训练、推理和主动学习闭环的端到端加速效果。

<!-- README updated after Round 3 V100 active learning summary on 2026-05-17. -->