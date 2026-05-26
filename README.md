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
- random seed0 / seed1 / seed2 Round 002 retraining baseline；
- random seed0 / seed1 / seed2 Round 003 retraining baseline；
- multi-seed random baseline mean ± std (Round 001/002/003)；
- uncertainty vs random 完整多轮 comparison table 和 learning curve 图；
- diversity 和 trust_level (DP-GEN-style) 四策略 multi-seed multi-round 完整对比 (2026-05-25)；
- V100 系统化 training profiling（132 models, mean=11.0s, 2×V100 parallel ~22s/round）；
- 结构多样性 descriptor 量化分析（diversity FPS vs uncertainty top-K, 3.1x 更大结构覆盖）；
- V100 profiling 方案和记录脚本；
- 文档体系整理，包括环境配置、复现实验、结果说明、baseline、paper evidence、profiling 计划和 Git 数据管理规范。

当前项目已经从：

```text
selection-level active learning record
```

推进到：

```text
dataset-level offline active learning closed-loop prototype
```

并进一步补充了第一版 random sampling baseline。

需要说明的是：当前全部四种 selection strategy（random / uncertainty / diversity / trust_level）均已完成 seed0/seed1/seed2 Round 001–003 multi-seed multi-round retraining baseline；四策略对齐对比表（aligned comparison）和 learning curve 图已生成；V100 training profiling 已完成（132 models, mean=11.0s/model），端到端 per-round 约 32s；预测和 I/O 分阶段耗时已记录；rMD17 ethanol 真实数据集数据管道已启动（数据转换、划分、configs 生成、Round 0–2 预测已完成）；H100 scaling 尚未开始。

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

uncertainty-diversity:
  Top-M high-uncertainty candidates + Farthest Point Sampling on
  pairwise-distance descriptor. Reduces structural redundancy.

trust-level (DP-GEN-style):
  Split candidate pool into accurate / candidate / failed regions
  by force_dev_max percentiles. Select primarily from candidate region.
```

详见 `docs/selection_strategies.md`。

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

### 4.3 Four-Strategy Multi-Seed Round 001–003 Comparison

当前全部四种策略均已完成 seed0/seed1/seed2 Round 001–003 multi-seed retraining baseline。
对齐后的 Force RMSE 对比（3-seed mean ± std）见 `experiments/baselines/aligned_comparison.md`。

每轮数据规模：

```text
Round 001: train 210, candidate 40
Round 002: train 220, candidate 30
Round 003: train 230, candidate 20
```

四策略对齐 Force RMSE 对比（3-seed mean ± std, toy H2, 2×V100）：

| Strategy | R1 F_RMSE | R2 F_RMSE | R3 F_RMSE |
|---|---:|---:|---:|
| uncertainty | 1.51e-01 ± 0.025 | 2.13e-01 ± 0.024 | 1.96e-01 ± 0.024 |
| random | 2.11e-01 ± 0.055 | 1.96e-01 ± 0.016 | 1.89e-01 ± 0.048 |
| diversity | 2.05e-01 ± 0.058 | 1.74e-01 ± 0.009 | 1.76e-01 ± 0.041 |
| trust_level | 1.35e-01 ± 0.028 | 1.49e-01 ± 0.023 | 1.78e-01 ± 0.006 |

所有策略使用统一的 "remaining candidate-pool" 指标，可直接对比。
完整数据见 `experiments/baselines/aligned_comparison.md` 和 `experiments/strategy_comparison_toy_h2/strategy_summary.md`。

相关结果文件：

```text
experiments/baselines/aligned_comparison.csv
experiments/baselines/aligned_comparison.md
experiments/baselines/random_round001_baseline_summary.csv
experiments/baselines/random_round002_baseline_summary.csv
experiments/baselines/random_round003_baseline_summary.csv
experiments/strategy_comparison_toy_h2/strategy_summary.md
```

---

### 4.4 Aligned Four-Strategy Comparison (Authoritative)

**统一口径**：全部策略使用同一指标 —— retraining 后 remaining candidate pool 的 `force_dev_max`。
旧版 `random_vs_uncertainty_summary.csv` 存在 selected-K vs remaining-candidate 混用，已标记 deprecated。

权威对比文件：

```text
experiments/baselines/aligned_comparison.csv     (authoritative — consistent metric)
experiments/baselines/aligned_comparison.md
```

Learning curve 图：

```text
experiments/figures/random_vs_uncertainty_force_rmse.svg
experiments/figures/random_vs_uncertainty_energy_rmse.svg
experiments/figures/random_vs_uncertainty_candidate_force_dev.svg
experiments/figures/random_vs_uncertainty_dataset_size.svg
```

该结果表明：
- uncertainty branch 的 selected top-K force_dev_max 随轮次稳步下降（0.440 → 0.170）
- 四策略 Force RMSE（3-seed mean ± std）差异在 1σ 以内，toy H2 上无法宣称某一策略显著优于其他
- 跨 seed 方差较大，符合 toy H2 数据规模和 committee 随机初始化的预期
- 完整结论仍需要真实 DFT/AIMD 数据集进一步验证

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
│   ├── paper_evidence.md
│   ├── profiling_h100.md
│   ├── profiling_v100.md
│   ├── random_baseline.md
│   ├── random_baseline_execution_checklist.md
│   ├── random_baseline_next_steps.md
│   ├── reproduce.md
│   ├── reproduce_legacy.md
│   ├── selection_strategies.md
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
│   ├── profiling/
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
│   ├── profiling/
│   ├── selection/
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
| `docs/profiling_h100.md` | H100 迁移和多 GPU scaling 实验计划（尚未执行） |
| `docs/paper_evidence.md` | 论文证据清单，当前可支持/不可支持的结论 |
| `docs/random_baseline_next_steps.md` | random baseline 完成后的下一步（profiling / diversity / trust-level / real data） |
| `docs/random_baseline_execution_checklist.md` | random baseline 完整执行清单和复现命令索引 |
| `docs/selection_strategies.md` | 四类 selection strategy 说明、参数、设计动机和 Round 001 对比 |
| `docs/diversity_and_trust_level_plan.md` | diversity 和 trust-level baseline 的后续多轮实验计划 |
| `docs/current_project_status.md` | 当前项目全局状态总览、已完成/待完成清单、claim boundary |
| `docs/profiling_v100.md` | V100 系统性能 profiling 方案、实测数据和操作指南 |
| `docs/profiling_h100.md` | H100 迁移计划（尚未执行） |

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
uncertainty-diversity (FPS)
trust-level (DP-GEN-style)
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
1.  DeepMD-kit Docker 环境验证
2.  toy H2 数据生成
3.  单模型 DeePMD baseline
4.  4-model committee training
5.  committee prediction
6.  force / energy model deviation 计算
7.  selected frames 合并
8.  candidate pool 更新
9.  Round 1–3 committee retraining
10. Round 0–3 learning curve 汇总
11. random sampling selection-level baseline (seed0/seed1/seed2)
12. random seed0/seed1/seed2 Round 001–003 retraining baseline
13. diversity + trust_level multi-seed Round 001–003
14. 四策略 aligned comparison + learning curves
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
| `baselines/random_seed0_round001–003` | 已完成 | random seed0 Round 001–003 retraining + prediction |
| `baselines/random_seed1_round001–003` | 已完成 | random seed1 Round 001–003 retraining + prediction |
| `baselines/random_seed2_round001–003` | 已完成 | random seed2 Round 001–003 retraining + prediction |
| `baselines/aligned_comparison` | 已完成 | 四策略统一口径对比 CSV + MD（authoritative） |
| `baselines/random_round001/002/003_baseline_summary` | 已完成 | random 三 seed Round 001–003 汇总 |
| `baselines/diversity_round001-003` | 已完成 | diversity 三 seed Round 001–003 |
| `baselines/trust_level_round001-003` | 已完成 | trust_level 三 seed Round 001–003 |
| `baselines/random_vs_uncertainty_summary` | 已完成（legacy） | 旧版汇总（selected-K vs remaining-candidate 混用，deprecated） |
| `figures/random_vs_uncertainty_*` | 已完成 | 对比 learning curve 图 |
| `figures` | 已完成 | Round 0–3 deviation、dataset size 和 RMSE 曲线 |

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
2. rMD17 ethanol 真实数据集数据管道已启动（转换/划分/configs/predictions），但完整的 multi-round active learning 尚未执行；
3. random sampling baseline 已完成 selection-level 对比和 seed0/seed1/seed2 Round 001–003 multi-round retraining (2026-05-25)；
4. selection-level random baseline 只能说明 uncertainty sampling 选出的构型平均不确定性更高，不能直接代表 retraining 后模型精度优势；
5. uncertainty-diversity（FPS）和 DP-GEN-style trust-level 策略已实现并完成 multi-seed Round 001–003 验证；
6. 当前尚未进行 H100 多 GPU scaling 实验；
7. 当前尚未进行真实 DFT labeling 或在线主动学习调度；
8. 当前 V100 profiling 已记录训练耗时和代表性 GPU 利用率，但 prediction 和端到端耗时仍需更精确的系统测量；
9. 当前 committee models 在部分实验中存在较大方差，后续需要分析随机初始化、训练集选择和 toy 数据规模对结果稳定性的影响；
10. 当前结果更适合证明主动学习闭环和 baseline 对比流程可行，尚不足以直接支撑完整 CCF-B 论文实验结论。

一句话概括：

> 当前仓库已经能证明“流程打通了”，但还不能证明“方法在真实体系上稳定有效且优于强基线”。

---

## 12. 后续计划

### 12.1 第一阶段：补完整 Random Sampling Baseline（已完成）

当前 random baseline 进展：

```text
已完成:
  selection-level random baseline
  random seed0 / seed1 / seed2 Round 001 retraining
  random seed0 / seed1 / seed2 Round 002 retraining
  random seed0 / seed1 / seed2 Round 003 retraining
  random multi-seed mean ± std (Round 001/002/003)
  Force RMSE / Energy RMSE learning curve 图
  candidate-pool uncertainty 对比图
  uncertainty vs random full comparison table

待完成:
  end-to-end active learning wall-clock time (profiling)
```

已从：

```text
multi-seed Round 001 random baseline
```

推进到：

```text
multi-seed Round 0–3 random baseline with full learning curve
```

---

### 12.2 第二阶段：加入 Uncertainty-Diversity 和 Trust-Level 策略（已完成）

uncertainty-diversity（FPS + pairwise-distance descriptor）和 DP-GEN-style trust-level selection 已实现并完成 multi-seed Round 001–003 验证，详见 `docs/selection_strategies.md`。

Diversity 策略在 toy H2 上将选中帧的结构覆盖度（pairwise distance）提升了 3.1×。

---

### 12.3 第三阶段：迁移到真实 DFT / AIMD 数据集（已启动）

迁移到更接近真实应用的数据集：

```text
rMD17 ethanol 数据集
  ↓
DeepMD npy format conversion (done: convert_rmd17_to_deepmd.py)
  ↓
train/valid/test/candidate split (done: split_rmd17_to_deepmd.py)
  ↓
Round 0–3 committee configs 生成 (done)
  ↓
Round 0–2 committee prediction (done)
  ↓
offline active learning pipeline (pending: multi-round training + selection loop)
  ↓
model deviation and configuration selection analysis
```

---

### 12.4 第四阶段：补充 Profiling 与 H100 实验

V100 profiling 已完成（training: 132 models, mean=11.0s; end-to-end: 36-round CSV; 2×V100 speedup: 1.97×）。H100 部分尚未开始：

```text
single-model training time (done: 132 models)
4-model committee training time (done)
2×V100 parallel training time (done: ~22s/round)
candidate prediction time (done: 6.5–7.2s)
end-to-end active learning round time (done: ~29s/round)
H100 speedup and GPU utilization (not started)
```

---

### 12.5 第五阶段：整理成论文级实验体系

如果以论文为目标，后续还需要补充：

```text
stronger baselines (done):
  random sampling
  uncertainty-diversity sampling
  DP-GEN-style trust-level sampling

real datasets (pending):
  real DFT / AIMD configurations
  multiple molecular or material systems

scientific validation (pending):
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
random seed0 / seed1 / seed2 Round 002 retraining baseline
  ↓
random seed0 / seed1 / seed2 Round 003 retraining baseline
  ↓
diversity + trust_level multi-seed Round 001–003
  ↓
four-strategy aligned comparison + learning curves
```

当前核心结论是：

> 在 toy H2 offline active learning 设置下，uncertainty sampling 能够更有效地选择高不确定性构型。四策略（uncertainty / random / diversity / trust_level）multi-seed multi-round 对齐对比已完成，全部策略均使用统一的 remaining candidate-pool 指标。

但该结论仍基于 toy H2 数据集，尚未在真实 DFT / AIMD 数据集上验证，不能直接推广到真实材料体系，也不能直接作为完整论文级结论。

下一步重点是：

```text
keep documentation consistent (done)
  ↓
complete multi-seed random baseline (done)
  ↓
add uncertainty-diversity + trust-level sampling (done)
  ↓
add systematic profiling (done: 36-round end-to-end CSV)
  ↓
move to real DFT / AIMD datasets (started: rMD17 ethanol data pipeline)
  ↓
run H100 / multi-GPU scaling experiments
```

<!-- README updated on 2026-05-25. -->