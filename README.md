# deepmd-al-hpc

`deepmd-al-hpc` 是一个面向 **第一性原理机器学习势函数主动学习闭环** 的多模型并行与高性能训练原型系统。

项目不直接训练大语言模型，也不是直接复现 Megatron-LM，而是借鉴 Megatron 系列大规模训练框架中的 **多 GPU 并行、micro-batch、混合精度、批量推理、流水线调度和分布式实验管理思想**，将这些系统设计思想迁移到 DeePMD / DeepMD-kit 机器学习势函数主动学习场景中。

当前阶段主要在 **2×Tesla V100 GPU** 平台上完成 toy H2 原型验证，包括：

- DeepMD-kit Docker 环境验证；
- toy H2 单模型 DeePMD baseline；
- 4-model DeePMD committee training；
- committee prediction；
- force / energy model deviation 计算；
- top-K 高不确定性构型筛选；
- dataset-level offline active learning 多轮闭环；
- Round 0–3 summary 与 learning curve 结果分析。

当前项目已经从 **selection-level 主动学习记录** 推进到 **dataset-level offline active learning 多轮闭环原型**。

---

## 1. Research Direction

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

## 2. Motivation

第一性原理计算，例如 DFT / AIMD，精度较高但计算代价昂贵。机器学习势函数可以学习第一性原理势能面，从而显著提高分子动力学模拟效率。

然而，机器学习势函数训练通常依赖大量高质量标注构型，而真实 DFT 标注成本很高。因此，本项目关注：

```text
如何用主动学习减少 DFT 标注冗余？
如何用 committee models 估计构型不确定性？
如何用多 GPU / H100 加速多轮主动学习闭环？
```

本项目当前采用 **offline active learning** 方式进行原型验证：

```text
已有完整 toy 数据集
  ↓
每轮只允许模型看到其中一部分标签
  ↓
被选中的构型才加入训练集
  ↓
用已有数据模拟真实 DFT labeling
```

这样可以先验证主动学习闭环、数据更新逻辑和多模型并行训练框架是否可行，再迁移到真实 DFT / AIMD 数据集。

---

## 3. Core Workflow

当前主动学习闭环如下：

```text
初始 DFT 数据集
  ↓
训练多个 DeePMD committee models
  ↓
在 candidate pool 上进行批量推理
  ↓
计算 force / energy / virial model deviation
  ↓
基于不确定性筛选 top-K 构型
  ↓
将 selected frames 合并进训练集
  ↓
更新 remaining candidate pool
  ↓
重新训练下一轮 committee models
  ↓
进入下一轮 active learning
```

当前 toy H2 原型中，主要使用 `force_dev_max` 作为高不确定性构型筛选指标。

---

## 4. Current Status

截至 **2026-05-17**，项目已经完成：

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
selected frames 合并进新训练集
  ↓
remaining candidate pool 更新
  ↓
Round 1 committee retraining
  ↓
Round 1 candidate prediction and selection
  ↓
Round 2 committee retraining
  ↓
Round 2 candidate prediction and selection
  ↓
Round 3 committee retraining
  ↓
Round 3 candidate prediction and selection
  ↓
Round 0–3 summary 表格生成
  ↓
Round 0–3 learning curve 图生成
```

当前最小多轮闭环为：

```text
Round 0: train 200, candidate 50
Round 1: train 210, candidate 40
Round 2: train 220, candidate 30
Round 3: train 230, candidate 20
```

---

## 5. Main Result

当前 toy H2 offline active learning 主线实验中，top-K 高不确定性构型的平均 force model deviation 随轮次推进逐步降低：

```text
force_dev_max_mean:
Round 0: 0.440989
Round 1: 0.269333
Round 2: 0.187412
Round 3: 0.170189
```

这说明随着主动学习轮次推进，剩余候选池中的高不确定性构型平均 force model deviation 持续降低，committee models 在候选构型空间中的预测分歧逐步减小。

验证集 Force RMSE 并没有严格单调下降：

```text
Force RMSE mean:
Round 0: 0.182139
Round 1: 0.161767
Round 2: 0.193859
Round 3: 0.174265
```

因此当前结果更适合表述为：

> 多轮主动学习后，候选池不确定性呈持续下降趋势；验证集 Force RMSE 整体处于同一量级，但受 toy 数据规模、随机初始化和 committee 模型差异影响，存在一定波动。

说明：当前 toy H2 结果主要用于验证主动学习闭环和工程流程，不用于评价真实材料体系精度。

---

## 6. Repository Structure

```text
deepmd-al-hpc/
├── README.md
├── .gitignore
├── .gitattributes
├── configs/
│   ├── active_learning/               # 主动学习配置
│   └── deepmd/                        # DeePMD input.json 与 committee 配置
│       ├── committee/
│       ├── round_001_committee/
│       ├── round_002_committee/
│       ├── round_003_committee/
│       └── toy_h2_input.json
├── docs/
│   └── reproduce.md                   # 完整复现流程
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
├── scripts/
│   ├── active_learning/               # 主动学习框架检查与 offline AL round 脚本
│   ├── analysis/                      # 多轮 AL 结果汇总与 learning curve 生成脚本
│   ├── config/                        # 多轮 committee 配置生成脚本
│   ├── data/                          # toy 数据生成、selected frames 合并与 candidate 更新脚本
│   ├── docker/                        # Docker 运行脚本
│   ├── eval/                          # freeze、test、误差评估脚本
│   ├── inference/                     # committee models 推理脚本
│   └── train/                         # 单模型、committee models 和 round retraining 脚本
└── src/
    ├── al/                            # 主动学习循环、筛选和调度
    ├── metrics/                       # model deviation 与误差指标
    └── utils/                         # IO、日志、随机种子等工具
```

说明：

- `data/` 目录为服务器本地数据目录，默认不提交到 GitHub；
- `experiments/` 中只保留轻量实验摘要；
- `.pb` 模型、checkpoint、`.npy`、`.npz` 和大型日志文件不提交到 GitHub；
- 完整复现流程见 `docs/reproduce.md`。

---

## 7. Documentation

| 文档 / 文件 | 说明 |
|---|---|
| `docs/reproduce.md` | 完整复现实验流程，包括 Docker 环境、toy H2 数据生成、单模型 baseline、committee training、prediction、Round 1–3 数据闭环和 summary 生成 |
| `experiments/al_rounds_summary.md` | Round 0–3 主动学习结果汇总 |
| `experiments/al_rounds_summary.csv` | Round-level summary 表格 |
| `experiments/al_model_level_summary.csv` | Model-level summary 表格 |
| `experiments/figures/` | Round 0–3 learning curve 图 |

---

## 8. Quick Start

进入项目目录：

```bash
cd /data/zft
```

进入 DeepMD-kit Docker 环境：

```bash
bash scripts/docker/enter_deepmd_container.sh
```

完整复现实验流程请查看：

```text
docs/reproduce.md
```

该文档包含：

```text
1. DeepMD-kit Docker 环境验证
2. toy H2 数据生成
3. 单模型 DeePMD baseline
4. 4-model committee training
5. committee prediction
6. force / energy model deviation 计算
7. selected frames 合并
8. candidate pool 更新
9. Round 1–3 committee retraining
10. Round 0–3 learning curve 汇总
```

---

## 9. Experiment Overview

| 实验编号 | 实验名称 | 状态 | 说明 |
|---|---|---|---|
| exp_001 | env_check | 已完成 | 验证 Docker、DeepMD-kit、dp、lmp、Python import |
| exp_002 | framework_check | 已完成 | 基于模拟 committee forces 验证 deviation 和 top-K 选择 |
| exp_003 | single_model_baseline | 已完成 | toy H2 单模型 DeePMD train / freeze / test |
| exp_004 | committee_models | 已完成 | 4 个真实 DeePMD committee models 训练、冻结和测试 |
| exp_005 | committee_prediction | 已完成 | 4 个 frozen models 真实预测、deviation 计算和 top-K 筛选 |
| exp_006 | offline_active_learning | 已完成 | 基于 top-K 结果形成一轮 offline AL selection 记录 |
| exp_007 | round001_committee_models | 已完成 | 合并 Round 0 selected frames 后重新训练 Round 1 committee models |
| exp_008 | round001_committee_prediction | 已完成 | 使用 Round 1 models 对 40 个 candidate 进行预测和 top-K 筛选 |
| exp_009 | round002_committee_models | 已完成 | 合并 Round 1 selected frames 后重新训练 Round 2 committee models |
| exp_010 | round002_committee_prediction | 已完成 | 使用 Round 2 models 对 30 个 candidate 进行预测和 top-K 筛选 |
| exp_011 | round003_committee_models | 已完成 | 合并 Round 2 selected frames 后重新训练 Round 3 committee models |
| exp_012 | round003_committee_prediction | 已完成 | 使用 Round 3 models 对 20 个 candidate 进行预测和 top-K 筛选 |
| figures | learning curve figures | 已完成 | 生成 Round 0–3 的 deviation、dataset size 和 RMSE 曲线 |

---

## 10. System Design

本项目整体分为四层：

```text
模型训练层
  ↓
多模型并行层
  ↓
批量推理与不确定性评估层
  ↓
主动学习调度层
```

### 10.1 模型训练层

负责调用 DeepMD-kit 训练单个 DeePMD 模型，输出 checkpoint、frozen model、训练日志和测试误差。

### 10.2 多模型并行层

负责训练多个 committee models。

在当前 2×V100 阶段，采用两批训练：

```text
Batch 1:
GPU 0 → model_000
GPU 1 → model_001

Batch 2:
GPU 0 → model_002
GPU 1 → model_003
```

后续在 H100 平台上计划扩展为更多 GPU 上的并行训练与批量推理。

### 10.3 批量推理与不确定性评估层

负责让多个 committee models 对 candidate pool 进行预测，并计算：

```text
force deviation
energy deviation
virial deviation
combined uncertainty score
```

当前 toy H2 原型中，主要使用 `force_dev_max` 进行 top-K 高不确定性构型筛选。

### 10.4 主动学习调度层

负责组织完整闭环：

```text
train set update
candidate pool update
committee retraining
committee prediction
model deviation calculation
top-K selection
round summary
```

---

## 11. Version Management

以下内容不应提交到 GitHub：

```text
data/
datasets/
raw_data/
processed_data/
*.npy
*.npz
*.pb
model.ckpt*
checkpoint
*.log
lcurve.out
out.json
input_v2_compat.json
LAMMPS / MD trajectory files
```

当前 GitHub 中主要保留：

```text
代码
配置文件
轻量实验摘要
selected_topk.json
summary.csv / summary.md
learning curve figures
文档
```

---

## 12. Known Limitations

当前项目仍处于原型验证阶段，主要限制包括：

1. 当前 toy H2 数据集仅用于流程验证，不能代表真实材料或分子体系上的模型精度；
2. 当前已经完成 dataset-level offline active learning 多轮闭环，但尚未引入真实 DFT / AIMD 数据集；
3. 当前尚未加入 random sampling baseline，因此还不能证明 model deviation sampling 一定优于随机采样；
4. 当前尚未引入结构多样性选择策略，仅基于 `force_dev_max` 进行 top-K 选择；
5. 当前尚未进行 H100 多 GPU 加速实验；
6. 当前尚未进行真实 DFT labeling 或在线主动学习调度；
7. 当前 V100 profiling 只记录了 Round 3 的初步训练与预测耗时，尚未系统记录 Round 0–2 的端到端耗时。

---

## 13. Roadmap

下一阶段计划包括：

### 13.1 Random Sampling Baseline

增加随机采样对照组：

```text
model deviation sampling
vs.
random sampling
```

比较：

```text
Force RMSE
Energy RMSE
候选池不确定性下降趋势
训练耗时
推理耗时
端到端 active learning wall-clock time
```

### 13.2 Real DFT / AIMD Dataset

迁移到更接近真实应用的数据：

```text
真实 DFT / AIMD 数据
  ↓
DeepMD npy 格式转换
  ↓
offline active learning pipeline
  ↓
model deviation 与构型筛选分析
```

### 13.3 V100 Profiling

补充系统性能分析：

```text
单模型训练耗时
4-model committee 串行训练耗时
2×V100 并行训练耗时
candidate prediction 耗时
model deviation 计算耗时
单个 active learning round 端到端耗时
```

### 13.4 H100 Acceleration

在 H100 多 GPU 平台上评估：

```text
训练时间
推理吞吐
model deviation 计算时间
多 GPU 加速比
端到端 active learning wall-clock time
GPU 利用率与显存占用
```

---

## 14. Expected Contributions

本项目后续希望形成以下几个方面的贡献：

1. **面向机器学习势函数主动学习闭环的多模型并行训练框架**  
   将 committee models 作为并行训练单元，提高多轮主动学习中的模型训练效率。

2. **面向候选构型池的批量推理与 model deviation 评估方法**  
   通过 micro-batch 和多 GPU 推理，提高候选构型不确定性评估效率。

3. **面向 DFT 标注节省的主动学习采样策略**  
   结合模型不确定性和结构多样性，减少冗余构型标注。

4. **面向 H100 平台的主动学习闭环加速实验**  
   从训练时间、推理吞吐、model deviation 计算时间、多 GPU 加速比和端到端 wall-clock time 等方面评估性能提升。

---

## 15. Summary

当前项目已经完成：

```text
toy H2 单模型 baseline
  ↓
4-model DeePMD committee training
  ↓
真实 committee prediction
  ↓
force / energy model deviation
  ↓
top-K 高不确定性构型筛选
  ↓
dataset-level offline active learning 多轮闭环
  ↓
Round 0–3 summary 与 learning curve
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

<!-- README simplified after moving reproduction details to docs/reproduce.md on 2026-05-17. -->