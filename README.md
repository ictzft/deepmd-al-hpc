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
- Round 0–3 summary 与 learning curve 结果分析；
- random sampling baseline 的 selection-level 对比；
- random seed0 Round 001 retraining baseline；
- random seed0 candidate-pool committee prediction 与 uncertainty branch 对比。

当前项目已经从 **selection-level 主动学习记录** 推进到 **dataset-level offline active learning 多轮闭环原型**，并进一步补充了第一版 **random sampling baseline**。

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

随着 random baseline 和后续 uncertainty-diversity sampling 的加入，论文方向也可以进一步收敛为：

> A Multi-GPU Uncertainty-Diversity Active Learning Framework for Deep Potential Models

---

## 2. Motivation

第一性原理计算，例如 DFT / AIMD，精度较高但计算代价昂贵。机器学习势函数可以学习第一性原理势能面，从而显著提高分子动力学模拟效率。

然而，机器学习势函数训练通常依赖大量高质量标注构型，而真实 DFT 标注成本很高。因此，本项目关注：

```text
如何用主动学习减少 DFT 标注冗余？
如何用 committee models 估计构型不确定性？
如何通过 random baseline 验证 uncertainty sampling 的有效性？
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

这样可以先验证主动学习闭环、数据更新逻辑、多模型并行训练框架和 baseline 对比流程是否可行，再迁移到真实 DFT / AIMD 数据集。

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
基于采样策略筛选构型
      ├── uncertainty top-K
      └── random sampling baseline
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

目前已经支持的 selection strategy 包括：

```text
uncertainty / top-K:
  按 force_dev_max 从大到小选择构型

random:
  按固定随机种子随机选择构型，用于 random sampling baseline
```

---

## 4. Current Status

截至 **2026-05-18**，项目已经完成：

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
  ↓
random sampling selection-level baseline
  ↓
random seed0 Round 001 retraining baseline
  ↓
random seed0 Round 001 candidate-pool prediction summary
```

当前 uncertainty-sampling 主线的最小多轮闭环为：

```text
Round 0: train 200, candidate 50
Round 1: train 210, candidate 40
Round 2: train 220, candidate 30
Round 3: train 230, candidate 20
```

当前 random-sampling baseline 已完成：

```text
Selection-level baseline:
  Round 0 random seed0 / seed1 / seed2
  Round 1 random seed0 / seed1 / seed2

Retraining baseline:
  random seed0 Round 001
```

---

## 5. Main Results

### 5.1 Uncertainty-sampling multi-round result

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

---

### 5.2 Selection-level random baseline

当前已经加入 random sampling baseline，并生成 selection-level 对比结果。

```text
Round 000:
random force_dev_max_mean      : 0.143007
uncertainty force_dev_max_mean : 0.440989

Round 001:
random force_dev_max_mean      : 0.145731
uncertainty force_dev_max_mean : 0.269333
```

这说明 uncertainty top-K 策略确实选中了平均不确定性更高的构型，而 random sampling 选中的构型平均不确定性较低。

相关结果文件：

```text
experiments/baselines/selection_baseline_runs.csv
experiments/baselines/selection_baseline_summary.csv
experiments/baselines/selection_baseline_summary.md
```

---

### 5.3 Random seed0 Round 001 retraining baseline

当前已经完成 random seed0 的 Round 001 retraining baseline。

随机选择的 10 个 frames 来自 Round 0 candidate pool，然后合并进初始训练集：

```text
train:
data/toy_h2/train
  +
random seed0 selected 10 frames
  ↓
data/toy_h2/random_seed0_round_001_train

candidate:
data/toy_h2/valid
  -
random seed0 selected 10 frames
  ↓
data/toy_h2/random_seed0_round_001_candidate
```

数据规模为：

```text
random_seed0_round_001_train     : 210 frames
random_seed0_round_001_candidate : 40 frames
```

4 个 random seed0 Round 001 committee models 的测试结果为：

```text
Mean Energy RMSE: 6.908853e-01 eV
Std  Energy RMSE: 7.559906e-01 eV

Mean Force RMSE : 2.553366e-01 eV/Å
Std  Force RMSE : 1.729852e-01 eV/Å
```

该结果显示 random seed0 committee 的模型间差异较大，因此后续应报告 mean / std，并继续补充 seed1 和 seed2。

相关结果文件：

```text
experiments/baselines/random_seed0_round001_metrics_summary.csv
experiments/baselines/random_seed0_round001_metrics_summary.md
```

---

### 5.4 Candidate-pool uncertainty comparison

random seed0 Round 001 retraining 后，对剩余 candidate pool 进行 committee prediction，并与 uncertainty branch 的 Round 001 candidate-pool prediction 进行对比。

```text
uncertainty_round001:
force_dev_max_mean: 0.126442
force_dev_max_max : 0.508339
force_dev_max_min : 0.042645

random_seed0_round001:
force_dev_max_mean: 0.355420
force_dev_max_max : 1.586355
force_dev_max_min : 0.086667
```

该结果说明：

> 在 toy H2 offline active learning 设置下，加入 10 个 uncertainty-selected frames 后，剩余 candidate pool 的平均 force model deviation 低于 random seed0 baseline。这初步表明 uncertainty sampling 比该 random seed0 baseline 更有效地降低了候选池不确定性。

相关结果文件：

```text
experiments/baselines/random_seed0_round001_prediction_summary.csv
experiments/baselines/random_seed0_round001_prediction_summary.md
experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json
```

说明：当前该结论仍基于 toy H2 和单个 random seed0，后续需要补充 seed1 / seed2，并报告 random mean ± std。

---

## 6. Repository Structure

```text
deepmd-al-hpc/
├── README.md
├── .gitignore
├── .gitattributes
├── configs/
│   ├── active_learning/
│   └── deepmd/
│       ├── committee/
│       ├── round_001_committee/
│       ├── round_002_committee/
│       ├── round_003_committee/
│       ├── random_seed0_round_001_committee/
│       └── toy_h2_input.json
├── docs/
│   ├── reproduce.md
│   └── week2_single_model_baseline.md
├── experiments/
│   ├── baselines/
│   │   ├── selection_baseline_runs.csv
│   │   ├── selection_baseline_summary.csv
│   │   ├── selection_baseline_summary.md
│   │   ├── random_seed0_round001_metrics_summary.csv
│   │   ├── random_seed0_round001_metrics_summary.md
│   │   ├── random_seed0_round001_prediction_summary.csv
│   │   ├── random_seed0_round001_prediction_summary.md
│   │   └── random_seed0_round001_committee_prediction/
│   │       └── selected_topk.json
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
│   ├── active_learning/
│   │   ├── run_framework_check.py
│   │   ├── run_offline_al_round.py
│   │   └── select_from_predictions.py
│   ├── analysis/
│   │   ├── plot_al_rounds.py
│   │   ├── summarize_al_rounds.py
│   │   └── summarize_selection_baselines.py
│   ├── config/
│   ├── data/
│   ├── docker/
│   ├── eval/
│   ├── inference/
│   └── train/
└── src/
    ├── al/
    │   ├── loop.py
    │   ├── scheduler.py
    │   └── selector.py
    ├── metrics/
    └── utils/
```

说明：

- `data/` 目录为服务器本地数据目录，默认不提交到 GitHub；
- `experiments/` 中只保留轻量实验摘要和 selected JSON；
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
| `experiments/baselines/selection_baseline_summary.md` | random sampling 与 uncertainty sampling 的 selection-level 对比 |
| `experiments/baselines/random_seed0_round001_metrics_summary.md` | random seed0 Round 001 retraining 后的 committee 测试指标 |
| `experiments/baselines/random_seed0_round001_prediction_summary.md` | random seed0 与 uncertainty Round 001 candidate-pool uncertainty 对比 |
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

## 9. Random Baseline Reproduction

### 9.1 Selection-level baseline

从已有 committee prediction 结果生成 uncertainty / random selection JSON：

```bash
python scripts/active_learning/select_from_predictions.py \
  --predictions experiments/exp_005_committee_prediction/committee_predictions.npz \
  --strategy random \
  --top-k 10 \
  --seed 0 \
  --template-json experiments/exp_005_committee_prediction/selected_topk.json \
  --output experiments/exp_005_committee_prediction/selected_random_seed0.json
```

汇总 selection-level baseline：

```bash
python scripts/analysis/summarize_selection_baselines.py
```

输出文件：

```text
experiments/baselines/selection_baseline_runs.csv
experiments/baselines/selection_baseline_summary.csv
experiments/baselines/selection_baseline_summary.md
```

---

### 9.2 Random seed0 Round 001 retraining

构造 random seed0 Round 001 train / candidate 数据：

```bash
python scripts/data/merge_selected_frames.py \
  --train data/toy_h2/train \
  --candidate data/toy_h2/valid \
  --selection experiments/exp_005_committee_prediction/selected_random_seed0.json \
  --output data/toy_h2/random_seed0_round_001_train \
  --overwrite

python scripts/data/make_remaining_candidate.py \
  --candidate data/toy_h2/valid \
  --selection experiments/exp_005_committee_prediction/selected_random_seed0.json \
  --output data/toy_h2/random_seed0_round_001_candidate \
  --overwrite
```

训练 random seed0 Round 001 committee models：

```bash
bash scripts/train/train_round_committee_models.sh \
  001 \
  configs/deepmd/random_seed0_round_001_committee \
  experiments/baselines/random_seed0_round001_committee_models \
  /data/zft/data/toy_h2/valid
```

使用 random seed0 committee models 对剩余 candidate pool 做预测：

```bash
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/random_seed0_round_001_candidate \
  --models \
    experiments/baselines/random_seed0_round001_committee_models/model_000/frozen_model.pb \
    experiments/baselines/random_seed0_round001_committee_models/model_001/frozen_model.pb \
    experiments/baselines/random_seed0_round001_committee_models/model_002/frozen_model.pb \
    experiments/baselines/random_seed0_round001_committee_models/model_003/frozen_model.pb \
  --output experiments/baselines/random_seed0_round001_committee_prediction/committee_predictions.npz \
  --selected-json experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json \
  --top-k 10
```

说明：

- `committee_predictions.npz` 默认不提交到 GitHub；
- GitHub 中仅保留轻量 summary 和 `selected_topk.json`；
- 后续需要继续补充 random seed1 / seed2 retraining baseline。

---

## 10. Experiment Overview

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
| baselines/selection_baseline | 已完成 | random sampling 与 uncertainty sampling 的 selection-level 对比 |
| baselines/random_seed0_round001 | 已完成 | random seed0 Round 001 retraining、metrics summary 和 candidate-pool prediction summary |
| figures | 已完成 | 生成 Round 0–3 的 deviation、dataset size 和 RMSE 曲线 |

---

## 11. System Design

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

### 11.1 模型训练层

负责调用 DeepMD-kit 训练单个 DeePMD 模型，输出 checkpoint、frozen model、训练日志和测试误差。

### 11.2 多模型并行层

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

### 11.3 批量推理与不确定性评估层

负责让多个 committee models 对 candidate pool 进行预测，并计算：

```text
force deviation
energy deviation
virial deviation
combined uncertainty score
```

当前 toy H2 原型中，主要使用 `force_dev_max` 进行 top-K 高不确定性构型筛选。

### 11.4 主动学习调度层

负责组织完整闭环：

```text
train set update
candidate pool update
committee retraining
committee prediction
model deviation calculation
selection strategy
round summary
baseline comparison
```

当前已经支持：

```text
uncertainty top-K
random sampling baseline
```

后续将加入：

```text
uncertainty-diversity sampling
DP-GEN-style threshold sampling
```

---

## 12. Version Management

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

## 13. Known Limitations

当前项目仍处于原型验证阶段，主要限制包括：

1. 当前 toy H2 数据集仅用于流程验证，不能代表真实材料或分子体系上的模型精度；
2. 当前已经完成 dataset-level offline active learning 多轮闭环，但尚未引入真实 DFT / AIMD 数据集；
3. 当前已经加入 random sampling baseline，但 retraining baseline 只完成 random seed0，仍需补充 seed1 / seed2 并报告 mean ± std；
4. 当前尚未引入结构多样性选择策略，仅基于 `force_dev_max` 进行 top-K 选择；
5. 当前尚未进行 H100 多 GPU 加速实验；
6. 当前尚未进行真实 DFT labeling 或在线主动学习调度；
7. 当前 V100 profiling 只记录了部分训练与预测耗时，尚未系统记录所有 round 的端到端耗时；
8. 当前 committee models 在部分实验中存在较大方差，后续需要分析随机初始化、训练集选择和 toy 数据规模对结果稳定性的影响。

---

## 14. Roadmap

下一阶段计划包括：

### 14.1 Complete Random Sampling Baseline

继续补充随机采样对照组：

```text
random seed0
random seed1
random seed2
```

最终报告：

```text
Random mean ± std
vs.
Uncertainty sampling
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

---

### 14.2 Uncertainty-Diversity Sampling

在 uncertainty top-K 的基础上加入结构多样性约束：

```text
Step 1: 按 force_dev_max 选择 top-M 高不确定性构型
Step 2: 基于结构特征或距离统计进行聚类
Step 3: 每个 cluster 内选择 uncertainty 最高的构型
Step 4: 得到最终 top-K
```

目标是避免 uncertainty-only 选中大量相似构型。

---

### 14.3 Real DFT / AIMD Dataset

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

---

### 14.4 V100 Profiling

补充系统性能分析：

```text
单模型训练耗时
4-model committee 串行训练耗时
2×V100 并行训练耗时
candidate prediction 耗时
model deviation 计算耗时
单个 active learning round 端到端耗时
```

---

### 14.5 H100 Acceleration

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

## 15. Expected Contributions

本项目后续希望形成以下几个方面的贡献：

1. **面向机器学习势函数主动学习闭环的多模型并行训练框架**  
   将 committee models 作为并行训练单元，提高多轮主动学习中的模型训练效率。

2. **面向候选构型池的批量推理与 model deviation 评估方法**  
   通过 micro-batch 和多 GPU 推理，提高候选构型不确定性评估效率。

3. **面向 DFT 标注节省的主动学习采样策略**  
   从 uncertainty top-K 出发，进一步结合 random baseline、结构多样性和 uncertainty calibration，减少冗余构型标注。

4. **面向 H100 平台的主动学习闭环加速实验**  
   从训练时间、推理吞吐、model deviation 计算时间、多 GPU 加速比和端到端 wall-clock time 等方面评估性能提升。

---

## 16. Summary

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
  ↓
random sampling selection-level baseline
  ↓
random seed0 Round 001 retraining baseline
```

当前最核心的实验现象包括：

```text
Uncertainty branch force_dev_max_mean:
Round 0: 0.440989
Round 1: 0.269333
Round 2: 0.187412
Round 3: 0.170189
```

以及：

```text
Candidate-pool uncertainty after Round 001 retraining:

uncertainty_round001:
force_dev_max_mean = 0.126442

random_seed0_round001:
force_dev_max_mean = 0.355420
```

这说明在当前 toy H2 offline active learning 设置下，uncertainty sampling 比 random seed0 baseline 更有效地降低了剩余 candidate pool 的平均 force model deviation。

下一步关键是从：

```text
单 seed random baseline + toy H2 流程验证
```

推进到：

```text
多 seed random baseline
  ↓
uncertainty-diversity sampling
  ↓
真实 DFT / AIMD 数据集
  ↓
系统 profiling 与 H100 加速实验
```

也就是进一步补齐对照实验、真实数据验证和高性能加速证据。

<!-- README updated with random sampling baseline and random seed0 Round 001 results on 2026-05-18. -->