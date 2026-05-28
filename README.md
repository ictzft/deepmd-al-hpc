# deepmd-al-hpc

## 最新状态修正（2026-05-28）

当前仓库已经不再只是 toy H2 + rMD17 ethanol。最新进展如下：

- toy H2：四策略 multi-seed multi-round 已完成；
- rMD17 ethanol：uncertainty / random / diversity / trust_level 四策略 multi-seed multi-round 已完成，independent test 与 10K NVE stability 已完成；
- rMD17 benzene：uncertainty Round000–003 已完成，random baseline seed0/seed1/seed2 Round001–003 已完成，independent test 与 mean ± std 汇总已完成；
- V100：training / prediction / selection / dataset update 主流程已跑通，已有 profiling baseline；
- H100 / 多 GPU scaling：尚未开始；
- 真实在线 DFT/AIMD labeling：尚未接入，当前仍属于 offline active learning 原型；
- 长期 MD 稳定性：10K NVE 可作为短程 sanity check，100K+ 解离说明当前模型仍不能宣称长期 MD 稳定。

当前最稳妥的论文表述是：本项目已经在 toy H2、rMD17 ethanol 和 rMD17 benzene 上验证了 DeePMD committee-based offline active learning 原型流程，并在 ethanol 与 benzene 上完成 uncertainty selection 与 random baseline 的独立测试对照；H100 scaling、在线 DFT labeling、第三个真实分子体系和长期 MD 稳定性仍属于后续工作。

---


## 0. 最新实验状态（2026-05-28）

当前项目已经完成 toy H2 原型验证、rMD17 ethanol 单体系验证，以及 rMD17 benzene 第二真实体系验证。需要注意，本仓库当前阶段仍属于 offline active learning 原型系统：使用已有 rMD17 标注数据模拟 DFT labeling，不等同于已经接入实时 DFT/AIMD 标注闭环。

当前已完成：

- toy H2：uncertainty / random / diversity / trust_level 四策略 multi-seed multi-round 对比；
- rMD17 ethanol：uncertainty Round000-003，random baseline，多策略对比，independent test，10K NVE stability sanity check；
- rMD17 benzene：uncertainty Round000-003，random seed0/1/2 Round001-003，independent test，对照表与 mean ± std 汇总；
- V100：已完成基础训练与推理流程验证，已有 profiling baseline。

当前仍未完成：

- H100 / 多 GPU scaling 实验尚未开始；
- 真实在线 DFT/AIMD labeling 闭环尚未接入；
- 10K NVE 只能说明短程稳定性，100K+ 仍会出现解离，不能宣称长期 MD 稳定；
- uncertainty、diversity、trust_level 之间差异有限，当前不能声称某一 active strategy 统计显著优于其他 active strategy；
- V100 profiling 还需要补充 GPU utilization、throughput、end-to-end speedup 和 scaling 分析。

当前可支持的论文表述是：本项目实现并验证了一个面向 DeePMD committee models 的 offline active learning 原型系统，在 ethanol 和 benzene 两个 rMD17 分子体系上完成了 uncertainty selection 与 random baseline 的对照验证，并在 V100 平台上形成了初步性能 profiling；H100 scaling、在线 DFT 标注和长期 MD 稳定性仍作为后续工作。

---


`deepmd-al-hpc` 是一个面向 **第一性原理机器学习势函数主动学习闭环** 的原型系统，重点关注 **DeePMD / DeepMD-kit 多模型 committee 训练**、**基于不确定性的构型筛选**、**random sampling baseline 对比**，以及后续面向 **多 GPU / H100 平台** 的高性能加速。

本项目属于 **AI for Science × 高性能计算** 交叉方向，目标是构建一个用于机器学习势函数主动学习的可复现实验框架。

当前仓库不是大语言模型训练项目，也不是 Megatron-LM 复现项目。本项目借鉴大规模训练系统中的多 GPU 并行、批量推理、micro-batch、混合精度和实验调度思想，并将其迁移到 DeePMD 主动学习场景中。

---

## 1. 当前状态

截至 **2026-05-28**，本项目已在 **2×Tesla V100 GPU** 平台上完成 toy H2 原型验证，以及 rMD17 ethanol 和 rMD17 benzene 两个真实分子体系的主动学习验证。

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
- rMD17 benzene 真实分子体系 uncertainty branch Round 000–003 多轮闭环 + random baseline（seed0/1/2 Round 001–003）+ independent test（12 原子，60000 帧候选池）；
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

需要说明的是：toy H2 四策略 multi-seed multi-round 已完成；V100 profiling baseline 已完成；rMD17 ethanol 四策略（uncertainty / random / diversity / trust_level）multi-seed multi-round 已完成，independent test evaluation 已完成，10K NVE MD stability 已完成；rMD17 benzene uncertainty branch Round 000–003 和 random baseline（seed0/1/2 Round 001–003）已完成，independent test 已完成（diversity / trust_level baseline 待补充）；H100 scaling 未开始。

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
- 完整结论已通过 rMD17 ethanol 得到验证（见 4.5 节）

---

### 4.5 rMD17 Ethanol Uncertainty vs Random Baseline（2026-05-26, 2×V100）

rMD17 ethanol（C₂H₅OH, 9 原子）上已完成 uncertainty branch Round 0–3 和 random baseline（3 seeds × 3 rounds, 36 models）。

**Uncertainty branch 结果**：

| Round | Train | Candidate | Valid F_RMSE | Test F_RMSE | force_dev_max (top-1000) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1000 | 60000 | 0.3739 | **0.3439** | 0.6129 |
| 1 | 2000 | 59000 | 0.3715 | **0.3433** | 0.4570 |
| 2 | 3000 | 58000 | 0.3644 | **0.3352** | 0.3906 |
| 3 | 4000 | 57000 | 0.3537 | **0.3266** | 0.4569 |

- Validation 和 independent test（10000 帧，从未参与 AL）Force RMSE 均**单调下降**
- Independent test Force RMSE 稳定比 valid 低 ~0.028 eV/Å

**Random Baseline 对比（3-seed mean ± std, validation set）**：

| Round | Uncertainty F_RMSE | Random F_RMSE |
|---:|---:|---:|
| 1 | 0.3715 | 0.3734 ± 0.010 |
| 2 | 0.3644 | 0.3990 ± 0.031 |
| 3 | **0.3537** | **0.6067 ± 0.385** |

**Independent Test 对比（10000 帧，从未参与 AL）**：

| Round | Uncertainty F_RMSE | Random F_RMSE | Gap |
|---:|---:|---:|---:|
| 1 | 0.3433 | 0.3464 | +0.003 |
| 2 | 0.3352 | 0.3719 | +0.037 |
| 3 | **0.3266** | **0.5801** | **+0.254 (1.78x)** |

- Uncertainty 在 validation 和 independent test 上均单调改善
- Random 在 independent test 上较为波动，Round 3 均值为 uncertainty 的 1.78 倍（但 std = 0.385 较大）
- 在 rMD17 ethanol 单体系实验中，uncertainty 表现出比 random 更稳定的改善趋势

**四策略对比（Round 3, validation set, 3-seed mean ± std）**：

| Strategy | Force RMSE | Std |
|---|---:|---:|
| uncertainty | 0.3537 | 0.0247 |
| diversity | 0.3555 | 0.0143 |
| trust_level | 0.3616 | 0.0166 |
| random | 0.6067 | 0.6826 |

- 三种 active 策略 Force RMSE 差异在 1σ 内（0.354–0.362 eV/Å），mean 均明显低于 random（0.607 eV/Å），但 random 跨 seed 方差大（std=0.683），不宜仅凭当前数据做严格统计显著性结论
- 与 toy H2 结论一致：在 rMD17 上也未出现单一策略显著优于其他策略

**MD Stability**：10K NVE 下所有模型稳定（drift ~0.035 eV/ps），100K+ 解离——当前 Force RMSE ~0.35 eV/Å 不足以支撑高温 MD。

**Top-K Ablation（Round 3）**：K=250 (0.3408) 和 K=2000 (0.3315) 均优于 K=1000 baseline (0.3537)。

**Committee Size Ablation（Round 1）**：8-model (0.3392) > 2-model (0.3436) > 4-model (0.3715)，但 8-model 训练成本是 4-model 的 2 倍。

完整结果见 `experiments/rmd17_ethanol_summary/` 和 `docs/real_dataset_plan.md`。

---

### 4.6 rMD17 Benzene Uncertainty Branch（2026-05-27, 2×V100）

rMD17 benzene（C₆H₅OH, 12 原子）上已完成 uncertainty branch Round 000–003（4 rounds, 4 committee models per round）。

**数据规模**：

| Split | Frames |
|---|---:|
| Initial train | 1000 |
| Candidate | 60000 |
| Validation | 5000 |
| Test | 10000 |

**Uncertainty branch 结果**：

| Round | Train | Candidate | Selection | Status |
|---:|---:|---:|---|---|
| 000 | 1000 | 60000 | initial | done |
| 001 | 2000 | 59000 | uncertainty top-1000 | done |
| 002 | 3000 | 58000 | uncertainty top-1000 | done |
| 003 | 4000 | 57000 | uncertainty top-1000 | done |

- 与 rMD17 ethanol 使用相同的 active learning pipeline 和 selection 策略
- `DP_INFER_BATCH_SIZE=1800` 用于避免 V100 16GB OOM
- random baseline（seed0/1/2 Round 001–003）已完成，independent test 已完成
- diversity / trust_level baseline 和 MD stability 待补充

详细说明见 `docs/results/rmd17_benzene_active_learning.md`。

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
│   ├── current_project_status.md
│   ├── data_and_git_policy.md
│   ├── diversity_and_trust_level_plan.md
│   ├── paper_evidence.md
│   ├── profiling_h100.md
│   ├── profiling_v100.md
│   ├── random_baseline.md
│   ├── random_baseline_execution_checklist.md
│   ├── random_baseline_next_steps.md
│   ├── real_dataset_execution_guide.md
│   ├── real_dataset_plan.md
│   ├── reproduce.md
│   ├── reproduce_legacy.md
│   ├── results.md
│   ├── selection_strategies.md
│   ├── setup.md
│   ├── strategy_comparison_toy_h2.md
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
| `docs/random_baseline.md` | random sampling baseline、multi-seed retraining 和对比结果 |
| `docs/random_baseline_next_steps.md` | random baseline 完成后的下一步 |
| `docs/random_baseline_execution_checklist.md` | random baseline 完整执行清单和复现命令索引 |
| `docs/selection_strategies.md` | 四类 selection strategy 说明、参数、设计动机和对比结果 |
| `docs/diversity_and_trust_level_plan.md` | diversity 和 trust-level 实验计划与 multi-seed 结果 |
| `docs/results.md` | 实验结果汇总与解释（toy H2 + rMD17 ethanol + rMD17 benzene） |
| `docs/paper_evidence.md` | 论文证据清单，当前可支持/不可支持的结论 |
| `docs/current_project_status.md` | 当前项目全局状态总览、已完成/待完成清单、claim boundary |
| `docs/real_dataset_plan.md` | 真实数据集迁移计划与 rMD17 ethanol 实验结果 |
| `docs/real_dataset_execution_guide.md` | 真实数据集执行指南 |
| `docs/reproduce.md` | 当前主复现入口，整合主要实验命令 |
| `docs/reproduce_legacy.md` | 早期复现记录，作为历史版本保留 |
| `docs/data_and_git_policy.md` | 数据文件、模型文件、日志文件与 Git 管理规范 |
| `docs/code_check.md` | 提交前代码检查、状态检查和大文件检查 |
| `docs/profiling_v100.md` | V100 系统性能 profiling 方案、实测数据和操作指南 |
| `docs/profiling_h100.md` | H100 迁移计划（尚未执行） |
| `docs/strategy_comparison_toy_h2.md` | toy H2 策略对比说明 |
| `docs/results/rmd17_benzene_active_learning.md` | rMD17 benzene 主动学习实验说明 |
| `docs/results/rmd17_ethanol_active_learning.md` | rMD17 ethanol 主动学习实验说明 |

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
| `rmd17_ethanol/uncertainty_round0-3` | 已完成 | 真实分子体系 uncertainty branch 多轮闭环 |
| `rmd17_ethanol/random_baseline` | 已完成 | 3 seeds × 3 rounds random baseline (36 models) |
| `rmd17_ethanol/independent_test` | 已完成 | 10000-frame 独立测试集评估 |
| `rmd17_ethanol/md_stability` | 已完成 | NVE MD 稳定性测试（10K/100K） |
| `rmd17_benzene/uncertainty_round0-3` | 已完成 | benzene uncertainty branch Round 000–003 多轮闭环 |
| `rmd17_benzene/random_baseline` | 已完成 | benzene random baseline (3 seeds × 3 rounds) |
| `rmd17_benzene/independent_test` | 已完成 | benzene 独立测试集评估 |
| `rmd17_benzene/diversity_baseline` | 待完成 | benzene diversity baseline |
| `rmd17_benzene/trust_level_baseline` | 待完成 | benzene trust_level baseline |

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
2. rMD17 ethanol 四策略 multi-seed multi-round 已完成；independent test 已完成；MD stability (10K) 已完成；
3. rMD17 benzene uncertainty branch 和 random baseline 已完成，independent test 已完成，diversity / trust_level baseline 待补充；
4. rMD17 ethanol 上四策略（uncertainty / random / diversity / trust_level）multi-seed multi-round 对比已完成（Round 3: 0.354–0.362 vs random 0.607 eV/Å），benzene 上的多策略对比待验证；
5. 当前尚未进行 H100 多 GPU scaling 实验；
6. diversity（FPS）和 trust-level（DP-GEN-style）策略在 toy H2 和 rMD17 ethanol 上均已完成 multi-seed multi-round 验证；多体系泛化性待验证；
7. V100 profiling 已记录训练耗时和代表性 GPU 利用率，但全流程 GPU utilization 曲线未记录；
8. MD 稳定性仅在 10K NVE 验证通过，100K+ 解离——提高模型精度后需重新评估；
9. 当前 committee models 跨 seed 方差在部分实验中较大；
10. 在 rMD17 ethanol 单体系上，uncertainty 相比 random 表现出更稳定的改善趋势，但 random 跨 seed 方差大（Round 3 std=0.385），不宜过度推广为普遍结论。

一句话概括：

> 当前仓库已在 toy H2、rMD17 ethanol 和 rMD17 benzene 上验证了 uncertainty-based AL 闭环的可行性；ethanol 上 uncertainty 比 random baseline 表现出更稳定的改善趋势；benzene 多策略对比（diversity / trust_level）和高温 MD 稳定性仍需补充。

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
rMD17 ethanol 数据集 (C₂H₅OH, 9 atoms, 27 Cartesian force components)
  ↓
DeepMD npy format conversion (done: convert_rmd17_to_deepmd.py)
  ↓
train/valid/test/candidate split (done: split_rmd17_to_deepmd.py)
  ↓
Round 0 committee training (done: 4 models, train 1000, candidate 60000)
  ↓
Round 0 committee prediction + uncertainty top-K selection (done: 1000 frames selected)
  ↓
Round 1 committee training (done: 4 models, train 2000, candidate 59000)
  ↓
Round 1 committee prediction + selection (done)
  ↓
Round 2 committee training (done: 4 models, train 3000, candidate 58000)
  ↓
Round 2 committee prediction + selection (done)
  ↓
Round 3 committee training (done: 4 models, train 4000, candidate 57000)
  ↓
Round 3 committee prediction (done: 57000 frames, top-1000 selected)
  ↓
summary + learning curve (done: al_rounds_summary.csv + 3 SVG figures)
  ↓
independent test evaluation (done: 10000-frame test, Force RMSE 0.344→0.327 eV/Å monotonic)
  ↓
four-strategy comparison on ethanol (done: uncertainty/random/diversity/trust_level)
  ↓
MD stability (done: 10K stable, 100K+ dissociates)

rMD17 benzene 数据集 (C₆H₅OH, 12 atoms):
  ↓
train/valid/test/candidate split (done: 1000/5000/10000/60000)
  ↓
Round 000–003 uncertainty branch (done: 4 rounds, 4 models each, top-1000 per round)
  ↓
random baseline seed0/1/2 Round 001–003 (done)
  ↓
independent test evaluation (done)
  ↓
diversity / trust_level baseline (pending)
  ↓
MD stability (pending)
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
stronger baselines (partially done):
  random sampling (done on ethanol + benzene)
  uncertainty-diversity sampling (done on ethanol; pending on benzene)
  DP-GEN-style trust-level sampling (done on ethanol; pending on benzene)

real datasets (partially done):
  rMD17 ethanol (done: 4-strategy, independent test, MD stability)
  rMD17 benzene (done: uncertainty, random, independent test; diversity/trust_level pending)
  additional molecular or material systems (pending)

scientific validation (partially done):
  MD stability (done: 10K NVE on ethanol; pending on benzene)
  trajectory stability (pending)
  physical property consistency (pending)
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

> 在 rMD17 ethanol 单体系实验中，uncertainty / diversity / trust_level 三种 active 策略的 Force RMSE 差异在 1σ 内（0.354–0.362 eV/Å），mean 均明显低于 random baseline（0.607 eV/Å）；但 random 跨 seed 方差大（std=0.683），不能仅凭当前数据做严格统计显著性结论。这与 toy H2 四策略结论一致。rMD17 benzene 上 uncertainty branch 和 random baseline 已完成，diversity / trust_level baseline 待补充。

下一步重点是：

```text
keep documentation consistent (done)
  ↓
complete multi-seed random baseline (done: toy H2 + rMD17)
  ↓
add uncertainty-diversity + trust-level sampling (done: toy H2)
  ↓
add systematic profiling (done: 36-round end-to-end CSV)
  ↓
add independent test evaluation (done: rMD17 ethanol)
  ↓
add MD stability validation (done: 10K stable, 100K dissociates)
  ↓
run diversity + trust_level on rMD17 ethanol (done: four-strategy comparison complete)
  ↓
complete rMD17 benzene baselines (diversity / trust_level + MD stability)
  ↓
run H100 / multi-GPU scaling experiments
```

<!-- README updated on 2026-05-28. -->
