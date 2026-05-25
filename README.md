# deepmd-al-hpc

`deepmd-al-hpc` 是一个面向 **第一性原理机器学习势函数主动学习闭环** 的原型系统，重点关注 **DeePMD / DeepMD-kit 多模型 committee 训练**、**基于不确定性的构型筛选**、**random sampling baseline 对比**，以及后续面向 **多 GPU / H100 平台** 的高性能加速。

本项目属于 **AI for Science × 高性能计算** 交叉方向，目标是构建一个用于机器学习势函数主动学习的可复现实验框架。

当前仓库不是大语言模型训练项目，也不是 Megatron-LM 复现项目。本项目借鉴大规模训练系统中的多 GPU 并行、批量推理、micro-batch、混合精度和实验调度思想，并将其迁移到 DeePMD 主动学习场景中。

---

## 1. 当前状态

截至 **2026-05-25**，本项目已在 **2×Tesla V100 GPU** 平台上完成 toy H2 原型验证。

当前已经实现：

- DeepMD-kit Docker 环境验证；
- toy H2 数据生成脚本；
- toy H2 单模型 DeePMD baseline；
- 4-model DeePMD committee training；
- committee prediction；
- force / energy model deviation 计算；
- 基于 `force_dev_max` 的 top-K 高不确定性构型筛选；
- dataset-level offline active learning Round 0–3 多轮闭环；
- Round 0–3 summary 与 learning curve 结果分析；
- random sampling baseline 的 selection-level 对比；
- random seed0 / seed1 / seed2 Round 001 retraining baseline；
- multi-seed random baseline mean ± std (Round 001)；
- uncertainty vs random Round 001 初步对比和 learning curve 图；
- random Round 002/003 数据准备脚本和复现命令；
- 初步文档体系整理，包括环境配置、复现实验、结果说明、baseline、paper evidence、profiling 计划和 Git 数据管理规范。

当前项目已经从：

```text
selection-level active learning record
```

推进到：

```text
dataset-level offline active learning closed-loop prototype
```

并进一步补充了第一版 random sampling baseline。

需要说明的是：当前 random sampling baseline 已经完成 selection-level random baseline 和 random seed0 / seed1 / seed2 Round 001 retraining baseline；Round 002/003 的数据准备脚本和复现命令已就绪，等待实际训练执行；完整 RMSE learning curve 对比和端到端耗时仍需后续补充。

---

## 2. 项目动机

第一性原理计算，例如 DFT / AIMD，具有较高精度，但计算代价昂贵。机器学习势函数可以学习第一性原理势能面，从而显著提高分子动力学模拟效率。

然而，机器学习势函数通常依赖大量高质量标注构型，而真实 DFT 标注成本很高。因此，本项目关注以下问题：

```text
如何用主动学习减少 DFT 标注冗余？
如何用 committee models 估计候选构型的不确定性？
如何通过 random baseline 验证 uncertainty sampling 的有效性？
如何用多 GPU / H100 加速多轮主动学习闭环？
```

当前阶段采用 **offline active learning** 方式进行原型验证：

```text
已有完整 toy dataset
  ↓
每轮只允许模型使用其中一部分 labeled data
  ↓
被选中的 configurations 加入训练集
  ↓
用已有数据模拟真实 DFT labeling 过程
```

这种方式可以在不真正调用昂贵 DFT 标注程序的情况下，先验证主动学习闭环、数据更新逻辑、committee training、uncertainty selection 和 baseline 对比流程。

---

## 3. 主动学习流程

当前主动学习闭环如下：

```text
Initial labeled dataset
  ↓
Train multiple DeePMD committee models
  ↓
Run committee prediction on candidate pool
  ↓
Compute force / energy model deviation
  ↓
Select configurations
      ├── uncertainty top-K
      └── random sampling baseline
  ↓
Merge selected frames into training set
  ↓
Update remaining candidate pool
  ↓
Retrain committee models
  ↓
Start next active learning round
```

当前 toy H2 原型中，主要使用 `force_dev_max` 作为高不确定性构型筛选指标。

目前支持的 selection strategy：

```text
uncertainty / top-K:
  Select configurations with the largest force_dev_max.

random:
  Randomly select configurations with fixed random seeds.
  Used as random sampling baseline.
```

后续计划加入：

```text
uncertainty-diversity sampling:
  Select high-uncertainty configurations while reducing structural redundancy.

DP-GEN-style threshold sampling:
  Split configurations into accurate / candidate / failed regions using trust levels.
```

---

## 4. 关键实验结果

### 4.1 Uncertainty Sampling Round 0–3 结果

当前 toy H2 offline active learning 主线实验中，top-K 高不确定性构型的平均 force model deviation 随轮次推进逐步降低：

```text
force_dev_max_mean:

Round 0: 0.440989
Round 1: 0.269333
Round 2: 0.187412
Round 3: 0.170189
```

这说明随着主动学习轮次推进，剩余候选池中的高不确定性构型平均 force model deviation 持续降低，committee models 在候选构型空间中的预测分歧逐步减小。

验证集 Force RMSE 没有严格单调下降：

```text
Force RMSE mean:

Round 0: 0.182139
Round 1: 0.161767
Round 2: 0.193859
Round 3: 0.174265
```

因此当前结果更适合表述为：

> 多轮主动学习后，候选池不确定性呈下降趋势；验证集 Force RMSE 整体处于同一量级，但受 toy 数据规模、随机初始化和 committee 模型差异影响，存在一定波动。

---

### 4.2 Selection-level Random Baseline 结果

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

需要注意：selection-level baseline 只能说明不同 selection strategy 选出的构型不确定性不同，不能直接代表 retraining 后模型精度差异。

---

### 4.3 Multi-seed Random Round 001 Retraining Baseline 结果

当前已经完成 random seed0 / seed1 / seed2 的 Round 001 retraining baseline。

每个 seed 的数据规模为：

```text
random_seed*_round_001_train     : 210 frames
random_seed*_round_001_candidate : 40 frames
```

三个 seed 的 Round 001 committee models 测试结果：

| Seed | Energy RMSE Mean / eV | Energy RMSE Std / eV | Force RMSE Mean / eV/Å | Force RMSE Std / eV/Å |
|---|---:|---:|---:|---:|
| seed0 | 6.908853e-01 | 7.559906e-01 | 2.553366e-01 | 1.729852e-01 |
| seed1 | 2.488029e-01 | 4.563935e-01 | 2.288370e-01 | 3.565438e-02 |
| seed2 | 4.284121e-01 | 5.054377e-01 | 1.494923e-01 | 4.669047e-02 |
| **Mean** | **4.560335e-01** | — | **2.112220e-01** | — |
| **Std** | **2.223318e-01** | — | **5.507695e-02** | — |

跨 seed 随机均值为：

```text
Random Mean Energy RMSE: 0.456034 eV
Random Mean Force RMSE : 0.211222 eV/Å
```

该结果显示不同 random seed 的 committee 模型间存在较大方差，符合 toy 数据规模和 committee 随机初始化的预期。

相关结果文件：

```text
experiments/baselines/random_seed0_round001_metrics_summary.csv
experiments/baselines/random_seed0_round001_metrics_summary.md
experiments/baselines/random_seed1_round001_metrics_summary.csv
experiments/baselines/random_seed1_round001_metrics_summary.md
experiments/baselines/random_seed2_round001_metrics_summary.csv
experiments/baselines/random_seed2_round001_metrics_summary.md
experiments/baselines/random_round001_comparison.csv
```

---

### 4.4 Candidate Pool Uncertainty 对比结果

各 branch Round 001 retraining 后，对剩余 candidate pool 进行 committee prediction 的对比结果。

| Branch | force_dev_max mean | force_dev_max max | force_dev_max min | energy_dev mean |
|---|---:|---:|---:|---:|
| uncertainty_round001 | 0.126442 | 0.508339 | 0.042645 | 0.448212 |
| random_seed0_round001 | 0.355420 | 1.586355 | 0.086667 | 0.656541 |
| random_seed1_round001 | 0.487795 | 1.262038 | 0.327483 | 0.396726 |
| random_seed2_round001 | 0.332138 | 1.117230 | 0.139321 | 0.446260 |
| **random mean** | **0.391784** | — | — | **0.499842** |

该结果初步表明：在当前 toy H2 offline active learning 设置下，加入 10 个 uncertainty-selected frames 后，剩余 candidate pool 的平均 force model deviation (0.126442) 低于所有三个 random seed baseline (seed0: 0.355420, seed1: 0.487795, seed2: 0.332138)。

但该结论仍基于 toy H2 和 Round 001 单轮 retraining，后续需要补充 Round 002/003 多轮 random retraining 对比。

相关结果文件：

```text
experiments/baselines/random_seed0_round001_prediction_summary.csv
experiments/baselines/random_seed0_round001_prediction_summary.md
experiments/baselines/random_seed1_round001_prediction_summary.csv
experiments/baselines/random_seed1_round001_prediction_summary.md
experiments/baselines/random_seed2_round001_prediction_summary.csv
experiments/baselines/random_seed2_round001_prediction_summary.md
experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json
experiments/baselines/random_seed1_round001_committee_prediction/selected_topk.json
experiments/baselines/random_seed2_round001_committee_prediction/selected_topk.json
```

---

## 5. 仓库结构

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
│       ├── random_seed1_round_001_committee/
│       ├── random_seed2_round_001_committee/
│       └── toy_h2_input.json
├── docs/
│   ├── code_check.md
│   ├── data_and_git_policy.md
│   ├── profiling_h100.md
│   ├── paper_evidence.md
│   ├── random_baseline.md
│   ├── reproduce.md
│   ├── reproduce_legacy.md
│   ├── results.md
│   ├── setup.md
│   ├── toy_h2_pipeline.md
│   └── uncertainty_rounds.md
├── experiments/
│   ├── baselines/
│   │   ├── random_round001_*.csv
│   │   ├── random_vs_uncertainty_summary.csv
│   │   └── ...
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
│   │   ├── *_rounds.svg
│   │   ├── random_vs_uncertainty_*.svg
│   │   └── ...
│   ├── al_model_level_summary.csv
│   ├── al_rounds_summary.csv
│   └── al_rounds_summary.md
├── scripts/
│   ├── active_learning/
│   ├── analysis/
│   ├── config/
│   ├── data/
│   ├── docker/
│   ├── eval/
│   ├── inference/
│   └── train/
└── src/
    ├── al/
    ├── metrics/
    └── utils/
```

说明：

- `data/` 为服务器本地数据目录，默认不提交到 GitHub；
- `.pb` 模型、checkpoint、`.npy`、`.npz` 和大型日志文件不提交到 GitHub；
- `experiments/` 中主要保留轻量 summary、selected JSON 和 learning curve figures；
- `scripts/data/` 是数据生成与处理脚本目录，应该被 Git 正常跟踪；
- 完整复现流程见 `docs/reproduce.md`。

---

## 6. 文档说明

当前文档已经拆分为多个专题文件，便于维护和复现。

| 文档 | 作用 |
|---|---|
| `docs/setup.md` | 环境配置、Docker、DeepMD-kit 基础检查 |
| `docs/toy_h2_pipeline.md` | toy H2 数据生成、单模型训练和基础流程 |
| `docs/uncertainty_rounds.md` | uncertainty sampling Round 0–3 多轮闭环说明 |
| `docs/random_baseline.md` | random sampling baseline、seed0 retraining 和后续扩展计划 |
| `docs/results.md` | 当前实验结果、learning curve 和结果解释 |
| `docs/reproduce.md` | 当前主复现入口，整合主要实验命令 |
| `docs/reproduce_legacy.md` | 早期复现记录，作为历史版本保留 |
| `docs/data_and_git_policy.md` | 数据文件、模型文件、日志文件与 Git 管理规范 |
| `docs/code_check.md` | 提交前代码检查、状态检查和大文件检查 |
| `docs/profiling_h100.md` | V100 profiling、H100 迁移和多 GPU scaling 实验计划 |
| `docs/paper_evidence.md` | 论文证据清单，当前可支持/不可支持的结论 |
| `docs/random_baseline_next_steps.md` | random baseline 后续 Round 002/003 实验计划和执行清单 |
| `docs/profiling_v100.md` | V100 系统性能 profiling 方案、指标和输出格式 |

推荐阅读顺序：

```text
docs/setup.md
  ↓
docs/toy_h2_pipeline.md
  ↓
docs/uncertainty_rounds.md
  ↓
docs/random_baseline.md
  ↓
docs/results.md
  ↓
docs/reproduce.md
```

---

## 7. 核心模块

### 7.1 模型训练层

负责调用 DeepMD-kit 训练 DeePMD 模型，输出 checkpoint、frozen model、训练日志和测试误差。

当前已验证：

```text
single DeePMD model training
4-model committee training
round-wise committee retraining
random baseline committee retraining
```

---

### 7.2 Committee Prediction 层

负责让多个 committee models 对 candidate pool 进行预测，并保存：

```text
energy predictions
force predictions
model deviation statistics
selected frame indices
```

当前 toy H2 原型主要使用 `force_dev_max` 作为 selection score。

---

### 7.3 主动学习调度层

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

---

### 7.4 结果分析层

负责生成：

```text
round-level summary
model-level summary
selection-level baseline summary
candidate-pool uncertainty comparison
learning curve figures
```

---

## 8. 快速开始

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
11. random sampling baseline 对比
12. random seed0 Round 001 retraining baseline
```

---

## 9. 实验概览

| 实验 | 状态 | 说明 |
|---|---|---|
| `exp_001_env_check` | 已完成 | 验证 Docker、DeepMD-kit、`dp`、`lmp` 和 Python import |
| `exp_002_framework_check` | 已完成 | 基于模拟 committee forces 验证 deviation 计算和 top-K 选择 |
| `exp_003_single_model_baseline` | 已完成 | toy H2 单模型 DeePMD train / freeze / test |
| `exp_004_committee_models` | 已完成 | 4 个 DeePMD committee models 训练和测试 |
| `exp_005_committee_prediction` | 已完成 | 进行 committee prediction 和 top-K selection |
| `exp_006_offline_active_learning` | 已完成 | 形成一轮 offline active learning selection record |
| `exp_007_round001_committee_models` | 已完成 | 重新训练 Round 1 committee models |
| `exp_008_round001_committee_prediction` | 已完成 | 预测 Round 1 candidate pool 并选择 top-K |
| `exp_009_round002_committee_models` | 已完成 | 重新训练 Round 2 committee models |
| `exp_010_round002_committee_prediction` | 已完成 | 预测 Round 2 candidate pool 并选择 top-K |
| `exp_011_round003_committee_models` | 已完成 | 重新训练 Round 3 committee models |
| `exp_012_round003_committee_prediction` | 已完成 | 预测 Round 3 candidate pool 并选择 top-K |
| `baselines/selection_baseline` | 已完成 | random sampling 与 uncertainty sampling 的 selection-level 对比 |
| `baselines/random_seed0_round001` | 已完成 | random seed0 Round 001 retraining 和 prediction summary |
| `baselines/random_seed1_round001` | 已完成 | random seed1 Round 001 retraining 和 prediction summary |
| `baselines/random_seed2_round001` | 已完成 | random seed2 Round 001 retraining 和 prediction summary |
| `baselines/random_round001_comparison` | 已完成 | uncertainty vs random seed0/seed1/seed2 Round 001 汇总对比 |
| `baselines/random_round001_baseline_summary` | 已完成 | random 三 seed Round 001 独立汇总 |
| `baselines/random_vs_uncertainty_summary` | 已完成 | uncertainty vs random 全轮次汇总 CSV + MD |
| `figures/random_vs_uncertainty_*` | 已完成 | uncertainty vs random 对比 learning curve 图 |
| `figures` | 已完成 | 生成 Round 0–3 deviation、dataset size 和 RMSE 曲线 |

---

## 10. 版本管理原则

以下内容不应提交到 GitHub：

```text
/data/
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
source code
configuration files
lightweight experiment summaries
selected_topk.json
summary.csv / summary.md
learning curve figures
documentation
```

需要注意：

```text
/data/          means ignoring only the root-level data directory.
scripts/data/   contains data-processing scripts and should be tracked by Git.
```

因此 `.gitignore` 中应写：

```gitignore
/data/
```

而不应写：

```gitignore
data/
```

否则会误伤 `scripts/data/` 下的数据处理脚本。

可以用以下命令检查：

```bash
# 根目录 data 应该被忽略
git check-ignore -v data/toy_h2/train/set.000/coord.npy || true

# scripts/data 不应该被忽略
git check-ignore -v --no-index scripts/data/new_test_file.py || true
```

期望结果：

```text
data/toy_h2/... should be ignored.
scripts/data/... should not be ignored.
```

---

## 11. 当前限制

当前项目仍处于原型验证阶段，主要限制包括：

1. toy H2 数据集仅用于流程验证，不能代表真实材料或分子体系上的模型精度；
2. 当前尚未引入真实 DFT / AIMD 数据集；
3. random sampling baseline 已经完成 selection-level 对比和 Round 001 三 seed retraining，但尚未完成 Round 002/003 多轮 retraining；
4. selection-level random baseline 只能说明 uncertainty sampling 选出的构型平均不确定性更高，不能直接代表 retraining 后模型精度优势；
5. 当前尚未引入结构多样性选择策略，仅基于 `force_dev_max` 进行 top-K 选择；
6. 当前尚未进行 H100 多 GPU scaling 实验；
7. 当前尚未进行真实 DFT labeling 或在线主动学习调度；
8. 当前 V100 profiling 只记录了部分训练与预测耗时，尚未系统记录所有 round 的端到端耗时；
9. 当前 committee models 在部分实验中存在较大方差，后续需要分析随机初始化、训练集选择和 toy 数据规模对结果稳定性的影响；
10. 当前结果更适合证明主动学习闭环和 baseline 对比流程可行，尚不足以直接支撑完整 CCF-B 论文实验结论。

一句话概括：

> 当前仓库已经能证明“流程打通了”，但还不能证明“方法在真实体系上稳定有效且优于强基线”。

---

## 12. 后续计划

### 12.1 第一阶段：补完整 Random Sampling Baseline

当前 random baseline 进展：

```text
已完成:
  selection-level random baseline
  random seed0 / seed1 / seed2 Round 001 retraining
  random multi-seed mean ± std (Round 001)

待完成:
  random seed0 / seed1 / seed2 Round 002 retraining
  random seed0 / seed1 / seed2 Round 003 retraining
  Force RMSE / Energy RMSE learning curve
  candidate-pool uncertainty curve
  end-to-end active learning wall-clock time
```

目标是从：

```text
multi-seed Round 001 random baseline
```

推进到：

```text
multi-seed Round 0–3 random baseline with full learning curve
```

---

### 12.2 第二阶段：加入 Uncertainty-Diversity Sampling

在 uncertainty top-K 的基础上加入结构多样性约束：

```text
Step 1: Select top-M high-uncertainty configurations.
Step 2: Cluster or diversify selected candidates using structural descriptors.
Step 3: Select final top-K configurations.
```

目标是减少 uncertainty-only selection 中可能出现的相似构型重复选择问题。

---

### 12.3 第三阶段：迁移到真实 DFT / AIMD 数据集

迁移到更接近真实应用的数据集：

```text
real DFT / AIMD configurations
  ↓
DeepMD npy format conversion
  ↓
offline active learning pipeline
  ↓
model deviation and configuration selection analysis
```

---

### 12.4 第四阶段：补充 Profiling 与 H100 实验

补充系统性能分析：

```text
single-model training time
4-model committee training time
2×V100 parallel training time
candidate prediction time
model deviation calculation time
end-to-end active learning round time
H100 speedup and GPU utilization
```

---

### 12.5 第五阶段：整理成论文级实验体系

如果以 CCF-B 论文为目标，后续还需要补充：

```text
stronger baselines:
  random sampling
  DP-GEN-style threshold sampling
  naive retraining
  uncertainty-diversity sampling

real datasets:
  real DFT / AIMD configurations
  multiple molecular or material systems

system evaluation:
  training throughput
  inference throughput
  GPU utilization
  scaling efficiency
  end-to-end wall-clock time

scientific validation:
  force / energy RMSE
  MD stability
  trajectory stability
  physical property consistency
```

---

## 13. 预期贡献

本项目后续希望形成以下贡献：

1. **面向 DeePMD 模型的 dataset-level offline active learning 框架**  
   构建包含 dataset update、committee retraining、candidate prediction、model deviation calculation 和 selection 的可复现主动学习闭环。

2. **基于 committee model deviation 的不确定性评估与 baseline 对比**  
   使用 force / energy model deviation 筛选高不确定性构型，并与 random sampling baselines 进行对比。

3. **面向多模型训练和批量预测的并行化工作流**  
   将 committee models 组织为并行训练和预测单元，为后续多 GPU / H100 加速提供基础。

4. **面向机器学习势函数主动学习的高性能系统实验**  
   从训练时间、预测吞吐、model deviation 开销、GPU 利用率和端到端 wall-clock time 等角度评估系统性能。

---

## 14. 总结

当前项目已经完成 toy H2 上的 DeePMD 主动学习闭环原型：

```text
toy H2 data generation
  ↓
single-model DeePMD baseline
  ↓
4-model committee training
  ↓
committee prediction
  ↓
force / energy model deviation
  ↓
uncertainty top-K selection
  ↓
dataset-level offline active learning Round 0–3
  ↓
selection-level random baseline
  ↓
random seed0 / seed1 / seed2 Round 001 retraining baseline
  ↓
multi-seed random mean ± std (Round 001)
```

当前核心结论是：

> 在 toy H2 offline active learning 设置下，uncertainty sampling 能够更有效地选择高不确定性构型，并在 Round 001 的 random seed0 / seed1 / seed2 对比中一致显示出更低的剩余 candidate-pool force model deviation。

但该结论仍基于 toy H2 数据集，且尚未完成 Round 002/003 多轮 random retraining，不能直接推广到真实 DFT / AIMD 数据集，也不能直接作为完整论文级结论。

下一步重点是：

```text
fix and keep documentation consistent
  ↓
complete multi-seed random baseline
  ↓
add uncertainty-diversity sampling
  ↓
move to real DFT / AIMD datasets
  ↓
add systematic profiling
  ↓
run H100 / multi-GPU scaling experiments
```

<!-- README reorganized and updated on 2026-05-18. -->