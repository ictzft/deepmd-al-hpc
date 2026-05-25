# 复现实验总入口

本文档是 `deepmd-al-hpc` 项目的复现导航页。

原始大型复现文档建议备份为：

```text
docs/reproduce_legacy.md
```

新的文档组织方式是：

```text
docs/
├── reproduce.md                 # 复现总入口
├── setup.md                     # 环境配置与 Docker 使用
├── toy_h2_pipeline.md           # toy H2 数据生成、单模型、初始 committee、prediction
├── uncertainty_rounds.md        # uncertainty branch Round 0–3 多轮闭环
├── random_baseline.md           # random sampling baseline 与 random retraining
├── results.md                   # 实验结果汇总与解释
├── data_and_git_policy.md       # 数据目录、Git 跟踪、大文件忽略规则
├── code_check.md                # Python / Shell / JSON / Git 检查命令
└── profiling_h100.md            # V100 profiling 与 H100 迁移计划
```

---

## 1. 文档定位

`docs/reproduce.md` 不再作为所有实验命令的合集，而是作为整个复现实验体系的总入口。

本文档主要回答：

```text
当前仓库能复现什么？
应该按照什么顺序复现？
每一部分应该查看哪个专题文档？
当前哪些内容已经完成？
当前哪些内容尚未完成？
主要结果文件在哪里？
```

具体命令分别维护在：

```text
docs/setup.md
docs/toy_h2_pipeline.md
docs/uncertainty_rounds.md
docs/random_baseline.md
docs/code_check.md
```

实验结果解释维护在：

```text
docs/results.md
```

数据管理、Git 提交和大文件忽略规则维护在：

```text
docs/data_and_git_policy.md
```

性能分析和 H100 迁移计划维护在：

```text
docs/profiling_h100.md
```

---

## 2. 当前可复现范围

当前仓库主要复现 toy H2 上的 offline active learning 原型实验。

已经完成的主线流程包括：

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
uncertainty top-K 构型筛选
  ↓
selected frames 合并进训练集
  ↓
candidate pool 更新
  ↓
下一轮 committee retraining
  ↓
dataset-level offline active learning Round 0–3
  ↓
Round 0–3 summary 与 learning curve
  ↓
random sampling selection-level baseline
  ↓
random seed0 Round 001 retraining baseline
  ↓
uncertainty branch vs random seed0 candidate-pool comparison
```

当前 toy H2 主线实验已经形成如下 uncertainty-sampling 多轮闭环：

```text
Round 0: train 200, candidate 50
Round 1: train 210, candidate 40
Round 2: train 220, candidate 30
Round 3: train 230, candidate 20
```

当前 random baseline 已完成：

```text
Selection-level baseline:
  Round 0 random seed0 / seed1 / seed2
  Round 1 random seed0 / seed1 / seed2

Retraining baseline:
  random seed0 / seed1 / seed2 Round 001
  multi-seed random Round 001 mean ± std
  random Round 001 candidate-pool prediction summary (seed0/seed1/seed2)
  uncertainty vs random Round 001 candidate-pool comparison
  uncertainty vs random comparison figures

Round 002/003:
  已执行完毕（2026-05-25），summary 和 comparison 已生成

当前可复现的 random baseline 范围为：
  Round 001–003 multi-seed multi-round comparison
  uncertainty vs random full learning curve
```

说明：

```text
本文档中的 toy H2 数据集只用于流程验证，
不用于评价真实材料或分子体系上的模型精度。
```

---

## 3. 推荐复现顺序

建议按照以下顺序复现实验：

```text
Step 0: 环境准备
Step 1: toy H2 数据生成
Step 2: 单模型 DeePMD baseline
Step 3: 初始 4-model committee training
Step 4: 初始 committee prediction
Step 5: uncertainty top-K selection
Step 6: uncertainty branch Round 0–3 多轮闭环
Step 7: Round 0–3 summary 与 learning curve
Step 8: random sampling selection-level baseline (seed0/seed1/seed2)
Step 9: random seed0 Round 001 retraining baseline
Step 10: random seed1/seed2 Round 001 retraining baseline
Step 11: multi-seed random Round 001 mean ± std 汇总
Step 12: uncertainty vs random Round 001 candidate-pool 对比
Step 13: 结果解释与当前限制分析
Step 14: 代码、配置和 Git 状态检查
Step 15: profiling 与 H100 迁移计划
```

对应文档如下：

| Step | 内容 | 文档 |
|---|---|---|
| Step 0 | 环境准备、Docker、GPU 检查 | `docs/setup.md` |
| Step 1–5 | toy H2 数据、单模型、初始 committee、prediction、top-K | `docs/toy_h2_pipeline.md` |
| Step 6 | uncertainty branch Round 0–3 多轮闭环 | `docs/uncertainty_rounds.md` |
| Step 7、13 | 结果汇总、learning curve、当前结论与限制 | `docs/results.md` |
| Step 8–12 | random sampling baseline、seed0/1/2 retraining、multi-seed 汇总、uncertainty vs random 对比 | `docs/random_baseline.md` |
| Step 14 | Python / Shell / JSON / Git 检查 | `docs/code_check.md` |
| Step 15 | V100 profiling 与 H100 迁移计划 | `docs/profiling_h100.md` |

---

## 4. 最小复现路径

如果只想快速验证 toy H2 uncertainty branch 主线流程，可以阅读和运行：

```text
1. docs/setup.md
2. docs/toy_h2_pipeline.md
3. docs/uncertainty_rounds.md
4. docs/results.md
```

该路径主要复现：

```text
toy H2 数据生成
  ↓
单模型 DeePMD baseline
  ↓
4-model committee training
  ↓
committee prediction
  ↓
uncertainty top-K selection
  ↓
Round 0–3 offline active learning
  ↓
learning curve summary
```

---

## 5. 完整复现路径

如果希望复现当前仓库中所有已完成内容，建议阅读和运行：

```text
1. docs/setup.md
2. docs/data_and_git_policy.md
3. docs/toy_h2_pipeline.md
4. docs/uncertainty_rounds.md
5. docs/random_baseline.md
6. docs/results.md
7. docs/code_check.md
```

该路径覆盖：

```text
环境配置
数据生成
单模型 baseline
committee training
committee prediction
uncertainty branch Round 0–3
selection-level random baseline (seed0/seed1/seed2)
random seed0/seed1/seed2 Round 001 retraining baseline
random multi-seed Round 001 mean ± std 汇总
uncertainty vs random Round 001 candidate-pool comparison
结果汇总
代码与配置检查
Git 状态检查
```

如果后续要补充系统性能实验和 H100 迁移，请继续阅读：

```text
8. docs/profiling_h100.md
```

---

## 6. 文档结构说明

### 6.1 `docs/setup.md`

用于记录运行环境和 Docker 使用方式，主要包括：

```text
宿主机与 Docker 容器路径关系
DeepMD-kit Docker 环境
Torch 基础开发环境
GPU 检查
dp / lmp / python 检查
DeepPot import 检查
```

---

### 6.2 `docs/toy_h2_pipeline.md`

用于记录 toy H2 主线实验的前半部分，主要包括：

```text
主动学习 skeleton 检查
toy H2 DeepMD 数据生成
单模型 DeePMD baseline
初始 4-model committee training
初始 committee prediction
force / energy model deviation
Round 0 selected_topk.json
```

该文档回答：

```text
如何从零开始得到初始 committee prediction 和 Round 0 top-K selection？
```

---

### 6.3 `docs/uncertainty_rounds.md`

用于记录 uncertainty top-K branch 的多轮主动学习闭环，主要包括：

```text
Round 0 → Round 1 数据更新
Round 1 committee retraining
Round 1 committee prediction
Round 1 → Round 2 数据更新
Round 2 committee retraining
Round 2 committee prediction
Round 2 → Round 3 数据更新
Round 3 committee retraining
Round 3 committee prediction
```

该文档回答：

```text
uncertainty top-K 主线如何从 Round 0 跑到 Round 3？
```

---

### 6.4 `docs/random_baseline.md`

用于记录 random sampling baseline，主要包括：

```text
selection-level random baseline
Round 0 random seed0 / seed1 / seed2
Round 1 random seed0 / seed1 / seed2
random seed0 / seed1 / seed2 Round 001 retraining
random seed0 / seed1 / seed2 candidate-pool prediction
multi-seed random mean ± std (Round 001)
uncertainty branch vs random_seed* Round 001 comparison
uncertainty vs random comparison figures
后续 random Round 002/003 计划
```

该文档回答：

```text
如何构造 random baseline，并与 uncertainty branch 进行 Round 001 初步对比？
后续如何扩展到 Round 002/003？
```

---

### 6.5 `docs/paper_evidence.md`

用于记录论文证据清单，主要包括：

```text
当前可支持的结论
当前不可支持的结论
已完成的实验
待补充的实验
```

---

### 6.6 `docs/results.md`

用于记录实验结果和结果解释，主要包括：

```text
Round-level summary
Force RMSE learning curve
force_dev_max_mean learning curve
dataset size curve
selection-level random baseline 结果
random seed0 retraining 结果
candidate-pool uncertainty 对比
当前结论
当前限制
```

该文档回答：

```text
当前实验结果说明了什么？哪些结论可以写？哪些结论还不能写？
```

---

### 6.7 `docs/data_and_git_policy.md`

用于记录数据管理和 Git 提交规则，主要包括：

```text
/data/ 与 scripts/data/ 的区别
哪些文件可以提交
哪些文件不能提交
.gitignore 规则
大文件误跟踪检查
GitHub 提交建议
```

---

### 6.8 `docs/code_check.md`

用于记录提交前检查命令，主要包括：

```text
Python 语法检查
Shell 语法检查
JSON 配置检查
Git 状态检查
大文件误跟踪检查
```

---

### 6.9 `docs/profiling_h100.md`

用于记录后续系统性能分析和 H100 迁移计划，主要包括：

```text
V100 profiling 当前状态
训练耗时统计
candidate prediction 耗时统计
model deviation 计算耗时统计
端到端 active learning wall-clock time
H100 迁移需要修改的内容
H100 / 多 GPU scaling 实验计划
```

---

## 7. 当前主要结果摘要

当前 uncertainty branch 的核心结果为：

```text
force_dev_max_mean:

Round 0: 0.440989
Round 1: 0.269333
Round 2: 0.187412
Round 3: 0.170189
```

这说明：

```text
随着主动学习轮次推进，
top-K 高不确定性构型的平均 force model deviation 持续降低。
```

验证集 Force RMSE 没有严格单调下降：

```text
Force RMSE mean:

Round 0: 0.182139
Round 1: 0.161767
Round 2: 0.193859
Round 3: 0.174265
```

因此当前更严谨的表述是：

> 多轮主动学习后，候选池不确定性呈持续下降趋势；验证集 Force RMSE 整体处于同一量级，但受 toy 数据规模、随机初始化和 committee 模型差异影响，存在一定波动。

当前 random Round 001 candidate-pool uncertainty 对比结果为：

```text
uncertainty_round001:
  force_dev_max_mean = 0.126442

random_seed0_round001:
  force_dev_max_mean = 0.355420

random_seed1_round001:
  force_dev_max_mean = 0.487795

random_seed2_round001:
  force_dev_max_mean = 0.332138

random mean (seed0/seed1/seed2):
  force_dev_max_mean = 0.391784
```

该结果初步表明：

```text
在 toy H2 offline active learning 和 Round 001 设置下，
uncertainty branch 的 remaining candidate-pool force_dev_max_mean
在 seed0 / seed1 / seed2 三个对比中均低于对应的 random branch。
```

但需要注意：

```text
Round 001 random baseline provides an initial multi-seed comparison;
uncertainty selection reduces remaining candidate-pool uncertainty more clearly
in the current toy H2 Round 001 comparison;
validation RMSE still fluctuates and requires more rounds for robust conclusion;
random Round 002/003 are still needed for complete learning-curve comparison.
```

---

## 8. 扩展 random baseline 到 Round 002/003

本节提供从 Round 001 扩展到 Round 002/003 的完整复现命令。

### 8.1 前提条件

每个 seed 需要先完成上一轮的 committee prediction。Round 001 已全部完成。

### 8.2 使用自动化脚本生成数据和配置

```bash
# 为所有 seed 生成 Round 002 数据和配置
for seed in seed0 seed1 seed2; do
  python scripts/analysis/prepare_random_baseline_round.py \
    --seed-label $seed --round-id 2
done

# Round 002 训练完成后，生成 Round 003 数据和配置
for seed in seed0 seed1 seed2; do
  python scripts/analysis/prepare_random_baseline_round.py \
    --seed-label $seed --round-id 3
done
```

`prepare_random_baseline_round.py` 自动完成：
1. 从上轮 committee prediction 读取 selected indices
2. 调用 `merge_selected_frames.py` 生成新 train set
3. 调用 `make_remaining_candidate.py` 生成新 candidate pool
4. 调用 `make_round_committee_configs.py` 生成 4-model configs

### 8.3 手动分步执行

以 seed0 Round 002 为例：

```bash
# Step 1: 生成 train set
python scripts/data/merge_selected_frames.py \
  --train data/toy_h2/random_seed0_round_001_train \
  --candidate data/toy_h2/random_seed0_round_001_candidate \
  --selection experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json \
  --output data/toy_h2/random_seed0_round_002_train --overwrite

# Step 2: 生成 remaining candidate
python scripts/data/make_remaining_candidate.py \
  --candidate data/toy_h2/random_seed0_round_001_candidate \
  --selection experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json \
  --output data/toy_h2/random_seed0_round_002_candidate --overwrite

# Step 3: 生成 committee configs
python scripts/config/make_round_committee_configs.py \
  --base configs/deepmd/toy_h2_input.json \
  --output-dir configs/deepmd/random_seed0_round_002_committee \
  --train-system /data/zft/deepmd-al-hpc/data/toy_h2/random_seed0_round_002_train \
  --valid-system /data/zft/deepmd-al-hpc/data/toy_h2/valid \
  --round-id 2 --n-models 4 --base-seed 1201

# Step 4: 训练
bash scripts/train/train_round_committee_models.sh \
  002 configs/deepmd/random_seed0_round_002_committee \
  experiments/baselines/random_seed0_round002_committee_models \
  /data/zft/deepmd-al-hpc/data/toy_h2/valid

# Step 5: 预测
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/random_seed0_round_002_candidate \
  --models \
    experiments/baselines/random_seed0_round002_committee_models/model_000/frozen_model.pb \
    experiments/baselines/random_seed0_round002_committee_models/model_001/frozen_model.pb \
    experiments/baselines/random_seed0_round002_committee_models/model_002/frozen_model.pb \
    experiments/baselines/random_seed0_round002_committee_models/model_003/frozen_model.pb \
  --output experiments/baselines/random_seed0_round002_committee_prediction/committee_predictions.npz \
  --selected-json experiments/baselines/random_seed0_round002_committee_prediction/selected_topk.json \
  --top-k 10
```

对 seed1/seed2 和 Round 003 替换对应的路径和参数。

### 8.4 汇总对比

所有 round 完成后运行：

```bash
python scripts/analysis/summarize_random_vs_uncertainty.py
python scripts/analysis/plot_random_vs_uncertainty.py
```

---

## 9. 主要结果文件索引

### 9.1 主线 Active Learning 结果

```text
experiments/al_rounds_summary.csv
experiments/al_rounds_summary.md
experiments/al_model_level_summary.csv
experiments/figures/
```

其中 learning curve 图包括：

```text
experiments/figures/force_model_deviation_rounds.svg
experiments/figures/dataset_size_rounds.svg
experiments/figures/validation_rmse_rounds.svg
```

---

### 9.2 Selection-level Random Baseline

```text
experiments/baselines/selection_baseline_runs.csv
experiments/baselines/selection_baseline_summary.csv
experiments/baselines/selection_baseline_summary.md
```

---

### 9.3 Multi-seed Random Round 001 Retraining Baseline

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

### 9.4 Random seed0 Candidate-pool Prediction

```text
experiments/baselines/random_seed0_round001_prediction_summary.csv
experiments/baselines/random_seed0_round001_prediction_summary.md
experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json
```

### 9.5 Random seed1 / seed2 Candidate-pool Prediction

```text
experiments/baselines/random_seed1_round001_prediction_summary.csv
experiments/baselines/random_seed1_round001_prediction_summary.md
experiments/baselines/random_seed1_round001_committee_prediction/selected_topk.json
experiments/baselines/random_seed2_round001_prediction_summary.csv
experiments/baselines/random_seed2_round001_prediction_summary.md
experiments/baselines/random_seed2_round001_committee_prediction/selected_topk.json
```

---

## 10. 当前尚未完成内容

当前仍需补充：

```text
真实 DFT / AIMD 数据集
全流程 GPU utilization 曲线记录 (nvidia-smi dmon)
H100 / 多 GPU scaling
MD 稳定性验证
independent_test 独立评估
```

其中近期优先级最高的是：

```text
真实 DFT / AIMD 数据集（框架已就绪，见 docs/real_dataset_execution_guide.md）
全流程 GPU monitoring 曲线记录
```

---

## 11. 当前限制

当前复现流程仍有以下限制：

1. toy H2 数据集仅用于流程验证，不能代表真实材料或分子体系；
2. 当前尚未引入真实 DFT / AIMD 数据集；
3. random sampling baseline 已完成 seed0/seed1/seed2 Round 001–003 multi-round retraining；
4. uncertainty vs random full RMSE learning curve 对比已生成；
5. diversity 和 trust-level 策略已完成 multi-seed Round 001–003 完整实验（2026-05-25, 2×V100）；
6. 当前尚未进行 H100 / 多 GPU scaling 实验；
7. V100 training wall-clock profiling 已记录；GPU utilization/memory 系统监测和端到端精确测量未完成；
8. 当前结果更适合证明主动学习闭环和 baseline 对比流程可行，尚不足以作为完整论文级结论。

---

## 12. 后续文档维护计划

下一阶段补充 random baseline 后，建议更新以下文档：

```text
docs/random_baseline.md:
  增加 random Round 002/003 多轮 branch
  增加 multi-round learning curve 对比

docs/results.md:
  增加多轮 uncertainty vs random mean ± std
  增加 full RMSE learning curve
  增加 candidate-pool uncertainty curve

docs/profiling_h100.md:
  增加每轮训练、预测、selection、dataset update 和端到端耗时

README.md:
  只保留精简版结果摘要和文档入口
```

后续迁移到真实 DFT / AIMD 数据集时，建议新增：

```text
docs/real_dataset.md
```

用于记录：

```text
真实数据来源
数据格式转换
训练 / candidate / test 划分
真实体系上的 active learning 设置
真实体系结果
```

---

## 13. 总结

新的文档组织原则是：

```text
README.md:
  项目首页，说明项目是什么、当前做到什么、核心结果是什么。

docs/reproduce.md:
  复现总入口，说明复现范围、推荐顺序和文档导航。

docs/setup.md:
  环境配置。

docs/toy_h2_pipeline.md:
  toy H2 初始实验。

docs/uncertainty_rounds.md:
  uncertainty branch 多轮主动学习闭环。

docs/random_baseline.md:
  random baseline 专题。

docs/results.md:
  实验结果汇总与解释。

docs/data_and_git_policy.md:
  数据与 Git 管理规则。

docs/code_check.md:
  代码和配置检查。

docs/profiling_h100.md:
  性能分析与 H100 迁移计划。
```

这样可以避免 `reproduce.md` 继续膨胀，也更符合论文型开源项目的文档组织方式。

<!-- reproduce.md reorganized as a reproduction index and documentation roadmap on 2026-05-18. -->