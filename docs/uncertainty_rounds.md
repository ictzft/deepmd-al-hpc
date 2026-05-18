# Uncertainty Branch 多轮主动学习闭环

本文档用于记录 `deepmd-al-hpc` 当前 toy H2 原型实验中的 uncertainty top-K branch 多轮主动学习闭环。

该阶段承接：

```text
docs/toy_h2_pipeline.md
```

即在已经完成：

```text
toy H2 数据生成
  ↓
单模型 DeePMD baseline
  ↓
初始 4-model committee training
  ↓
初始 committee prediction
  ↓
Round 0 selected_topk.json
```

之后，继续执行：

```text
selected frames 合并进训练集
  ↓
candidate pool 更新
  ↓
下一轮 committee config 生成
  ↓
下一轮 committee retraining
  ↓
下一轮 committee prediction
  ↓
继续下一轮 active learning
```

---

## 1. 当前主线轮次设置

当前 toy H2 uncertainty branch 已经完成 Round 0–3 多轮闭环。

数据规模如下：

```text
Round 0: train 200, candidate 50
Round 1: train 210, candidate 40
Round 2: train 220, candidate 30
Round 3: train 230, candidate 20
```

每轮选择：

```text
top-K = 10
```

当前使用的 selection score：

```text
force_dev_max
```

选择策略：

```text
按 force_dev_max 从大到小排序，选择前 10 个高不确定性 frames。
```

---

## 2. 当前闭环流程

完整 uncertainty branch 的逻辑如下：

```text
Round 0:
  initial train set = data/toy_h2/train
  initial candidate pool = data/toy_h2/valid
  initial committee prediction = experiments/exp_005_committee_prediction/
  selected frames = experiments/exp_005_committee_prediction/selected_topk.json
        ↓
  merge selected frames into train set
  update remaining candidate pool
        ↓
Round 1:
  train set = data/toy_h2/round_001_train
  candidate pool = data/toy_h2/round_001_candidate
  train Round 1 committee models
  predict Round 1 candidate pool
  select new top-K frames
        ↓
Round 2:
  train set = data/toy_h2/round_002_train
  candidate pool = data/toy_h2/round_002_candidate
  train Round 2 committee models
  predict Round 2 candidate pool
  select new top-K frames
        ↓
Round 3:
  train set = data/toy_h2/round_003_train
  candidate pool = data/toy_h2/round_003_candidate
  train Round 3 committee models
  predict Round 3 candidate pool
  select new top-K frames
```

说明：

```text
Round 0 是初始 committee prediction。
Round 1 / Round 2 / Round 3 是基于 selected frames 更新数据集后的 retraining rounds。
```

---

## 3. 第 0 步：生成 offline active learning selection 记录

在进入 Round 1 数据更新前，可以先将 `exp_005` 的 top-K 选择结果整理为一轮 offline active learning selection 记录。

运行：

```bash
PYTHONPATH=. python scripts/active_learning/run_offline_al_round.py \
  --selected-json experiments/exp_005_committee_prediction/selected_topk.json \
  --output experiments/exp_006_offline_active_learning/round_001_selection.json \
  --initial-train-frames 200 \
  --round-id 1
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

说明：

```text
该步骤主要用于生成 offline active learning selection record。
真正的数据集更新在后续 merge_selected_frames.py 和 make_remaining_candidate.py 中完成。
```

---

## 4. Round 0 到 Round 1 数据更新

Round 0 的输入包括：

```text
初始训练集：
data/toy_h2/train

初始 candidate pool：
data/toy_h2/valid

Round 0 top-K selection：
experiments/exp_005_committee_prediction/selected_topk.json
```

目标是生成：

```text
Round 1 训练集：
data/toy_h2/round_001_train

Round 1 candidate pool：
data/toy_h2/round_001_candidate
```

---

### 4.1 合并 selected frames，生成 Round 1 训练集

将 Round 0 选出的 top-10 high-uncertainty frames 合并到训练集：

```bash
python scripts/data/merge_selected_frames.py \
  --train data/toy_h2/train \
  --candidate data/toy_h2/valid \
  --selection experiments/exp_005_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_001_train \
  --overwrite
```

生成：

```text
data/toy_h2/round_001_train
```

数据规模：

```text
200 initial train frames + 10 selected frames = 210 frames
```

---

### 4.2 更新 candidate pool，生成 Round 1 candidate

从初始 candidate pool 中移除已经被选入训练集的 frames：

```bash
python scripts/data/make_remaining_candidate.py \
  --candidate data/toy_h2/valid \
  --selection experiments/exp_005_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_001_candidate \
  --overwrite
```

生成：

```text
data/toy_h2/round_001_candidate
```

数据规模：

```text
50 initial candidate frames - 10 selected frames = 40 frames
```

---

### 4.3 生成 Round 1 committee 配置

```bash
python scripts/config/make_round_committee_configs.py \
  --base-config configs/deepmd/toy_h2_input.json \
  --train-data data/toy_h2/round_001_train \
  --output-dir configs/deepmd/round_001_committee \
  --round-id 1 \
  --seed-base 1101
```

生成配置目录：

```text
configs/deepmd/round_001_committee/
```

该目录应包含 4 个 Round 1 committee 配置文件。

---

### 4.4 训练 Round 1 committee models

```bash
bash scripts/train/train_round_committee_models.sh \
  001 \
  configs/deepmd/round_001_committee \
  experiments/exp_007_round001_committee_models \
  /data/zft/data/toy_h2/valid
```

对应实验目录：

```text
experiments/exp_007_round001_committee_models/
```

轻量结果：

```text
experiments/exp_007_round001_committee_models/metrics_summary.md
```

说明：

```text
frozen_model.pb、checkpoint、model.ckpt*、train.log、test.log、lcurve.out 等大文件默认不提交 GitHub。
metrics_summary.md 可以提交 GitHub。
```

---

### 4.5 Round 1 committee prediction

使用 Round 1 committee models 对 Round 1 candidate pool 进行 prediction：

```bash
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/round_001_candidate \
  --models \
    experiments/exp_007_round001_committee_models/model_000/frozen_model.pb \
    experiments/exp_007_round001_committee_models/model_001/frozen_model.pb \
    experiments/exp_007_round001_committee_models/model_002/frozen_model.pb \
    experiments/exp_007_round001_committee_models/model_003/frozen_model.pb \
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

中间文件：

```text
experiments/exp_008_round001_committee_prediction/committee_predictions.npz
```

说明：

```text
committee_predictions.npz 属于中间数组文件，默认不提交 GitHub。
selected_topk.json 可以提交 GitHub。
```

---

## 5. Round 1 到 Round 2 数据更新

Round 1 的输入包括：

```text
Round 1 训练集：
data/toy_h2/round_001_train

Round 1 candidate pool：
data/toy_h2/round_001_candidate

Round 1 top-K selection：
experiments/exp_008_round001_committee_prediction/selected_topk.json
```

目标是生成：

```text
Round 2 训练集：
data/toy_h2/round_002_train

Round 2 candidate pool：
data/toy_h2/round_002_candidate
```

---

### 5.1 合并 Round 1 selected frames

```bash
python scripts/data/merge_selected_frames.py \
  --train data/toy_h2/round_001_train \
  --candidate data/toy_h2/round_001_candidate \
  --selection experiments/exp_008_round001_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_002_train \
  --overwrite
```

生成：

```text
data/toy_h2/round_002_train
```

数据规模：

```text
210 Round 1 train frames + 10 selected frames = 220 frames
```

---

### 5.2 更新 Round 2 candidate pool

```bash
python scripts/data/make_remaining_candidate.py \
  --candidate data/toy_h2/round_001_candidate \
  --selection experiments/exp_008_round001_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_002_candidate \
  --overwrite
```

生成：

```text
data/toy_h2/round_002_candidate
```

数据规模：

```text
40 Round 1 candidate frames - 10 selected frames = 30 frames
```

---

### 5.3 生成 Round 2 committee 配置

```bash
python scripts/config/make_round_committee_configs.py \
  --base-config configs/deepmd/toy_h2_input.json \
  --train-data data/toy_h2/round_002_train \
  --output-dir configs/deepmd/round_002_committee \
  --round-id 2 \
  --seed-base 1201
```

生成配置目录：

```text
configs/deepmd/round_002_committee/
```

---

### 5.4 训练 Round 2 committee models

```bash
bash scripts/train/train_round_committee_models.sh \
  002 \
  configs/deepmd/round_002_committee \
  experiments/exp_009_round002_committee_models \
  /data/zft/data/toy_h2/valid
```

对应实验目录：

```text
experiments/exp_009_round002_committee_models/
```

轻量结果：

```text
experiments/exp_009_round002_committee_models/metrics_summary.md
```

---

### 5.5 Round 2 committee prediction

```bash
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/round_002_candidate \
  --models \
    experiments/exp_009_round002_committee_models/model_000/frozen_model.pb \
    experiments/exp_009_round002_committee_models/model_001/frozen_model.pb \
    experiments/exp_009_round002_committee_models/model_002/frozen_model.pb \
    experiments/exp_009_round002_committee_models/model_003/frozen_model.pb \
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

中间文件：

```text
experiments/exp_010_round002_committee_prediction/committee_predictions.npz
```

---

## 6. Round 2 到 Round 3 数据更新

Round 2 的输入包括：

```text
Round 2 训练集：
data/toy_h2/round_002_train

Round 2 candidate pool：
data/toy_h2/round_002_candidate

Round 2 top-K selection：
experiments/exp_010_round002_committee_prediction/selected_topk.json
```

目标是生成：

```text
Round 3 训练集：
data/toy_h2/round_003_train

Round 3 candidate pool：
data/toy_h2/round_003_candidate
```

---

### 6.1 合并 Round 2 selected frames

```bash
python scripts/data/merge_selected_frames.py \
  --train data/toy_h2/round_002_train \
  --candidate data/toy_h2/round_002_candidate \
  --selection experiments/exp_010_round002_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_003_train \
  --overwrite
```

生成：

```text
data/toy_h2/round_003_train
```

数据规模：

```text
220 Round 2 train frames + 10 selected frames = 230 frames
```

---

### 6.2 更新 Round 3 candidate pool

```bash
python scripts/data/make_remaining_candidate.py \
  --candidate data/toy_h2/round_002_candidate \
  --selection experiments/exp_010_round002_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_003_candidate \
  --overwrite
```

生成：

```text
data/toy_h2/round_003_candidate
```

数据规模：

```text
30 Round 2 candidate frames - 10 selected frames = 20 frames
```

---

### 6.3 生成 Round 3 committee 配置

```bash
python scripts/config/make_round_committee_configs.py \
  --base-config configs/deepmd/toy_h2_input.json \
  --train-data data/toy_h2/round_003_train \
  --output-dir configs/deepmd/round_003_committee \
  --round-id 3 \
  --seed-base 1301
```

生成配置目录：

```text
configs/deepmd/round_003_committee/
```

---

### 6.4 训练 Round 3 committee models

```bash
bash scripts/train/train_round_committee_models.sh \
  003 \
  configs/deepmd/round_003_committee \
  experiments/exp_011_round003_committee_models \
  /data/zft/data/toy_h2/valid
```

对应实验目录：

```text
experiments/exp_011_round003_committee_models/
```

轻量结果：

```text
experiments/exp_011_round003_committee_models/metrics_summary.md
```

---

### 6.5 Round 3 committee prediction

```bash
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/round_003_candidate \
  --models \
    experiments/exp_011_round003_committee_models/model_000/frozen_model.pb \
    experiments/exp_011_round003_committee_models/model_001/frozen_model.pb \
    experiments/exp_011_round003_committee_models/model_002/frozen_model.pb \
    experiments/exp_011_round003_committee_models/model_003/frozen_model.pb \
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

当前 Round 3 prediction 结果：

```text
n_models: 4
n_frames: 20
n_atoms: 2
top_k: 10
prediction time: 7 s
```

---

## 7. Round 0–3 输出目录汇总

### 7.1 Round 0

```text
experiments/exp_005_committee_prediction/
experiments/exp_006_offline_active_learning/
```

主要轻量文件：

```text
experiments/exp_005_committee_prediction/selected_topk.json
experiments/exp_005_committee_prediction/metrics_summary.md
experiments/exp_006_offline_active_learning/round_001_selection.json
experiments/exp_006_offline_active_learning/metrics_summary.md
```

---

### 7.2 Round 1

```text
data/toy_h2/round_001_train
data/toy_h2/round_001_candidate
configs/deepmd/round_001_committee/
experiments/exp_007_round001_committee_models/
experiments/exp_008_round001_committee_prediction/
```

主要轻量文件：

```text
experiments/exp_007_round001_committee_models/metrics_summary.md
experiments/exp_008_round001_committee_prediction/selected_topk.json
```

---

### 7.3 Round 2

```text
data/toy_h2/round_002_train
data/toy_h2/round_002_candidate
configs/deepmd/round_002_committee/
experiments/exp_009_round002_committee_models/
experiments/exp_010_round002_committee_prediction/
```

主要轻量文件：

```text
experiments/exp_009_round002_committee_models/metrics_summary.md
experiments/exp_010_round002_committee_prediction/selected_topk.json
```

---

### 7.4 Round 3

```text
data/toy_h2/round_003_train
data/toy_h2/round_003_candidate
configs/deepmd/round_003_committee/
experiments/exp_011_round003_committee_models/
experiments/exp_012_round003_committee_prediction/
```

主要轻量文件：

```text
experiments/exp_011_round003_committee_models/metrics_summary.md
experiments/exp_012_round003_committee_prediction/selected_topk.json
experiments/exp_012_round003_committee_prediction/round003_summary.md
```

---

## 8. 大文件与轻量文件说明

以下内容默认不提交 GitHub：

```text
data/toy_h2/round_*_train/
data/toy_h2/round_*_candidate/
experiments/exp_007_round001_committee_models/model_*/frozen_model.pb
experiments/exp_009_round002_committee_models/model_*/frozen_model.pb
experiments/exp_011_round003_committee_models/model_*/frozen_model.pb
experiments/**/committee_predictions.npz
checkpoint
model.ckpt*
train.log
test.log
lcurve.out
out.json
input_v2_compat.json
```

以下轻量结果可以提交 GitHub：

```text
selected_topk.json
round_001_selection.json
metrics_summary.md
round003_summary.md
summary.csv
summary.md
small figures
config json
```

---

## 9. 当前主线结果摘要

当前 uncertainty branch 的核心现象是：

```text
force_dev_max_mean:

Round 0: 0.440989
Round 1: 0.269333
Round 2: 0.187412
Round 3: 0.170189
```

说明：

```text
随着主动学习轮次推进，top-K 高不确定性构型的平均 force model deviation 持续降低。
```

当前验证集 Force RMSE 没有严格单调下降：

```text
Force RMSE mean:

Round 0: 0.182139
Round 1: 0.161767
Round 2: 0.193859
Round 3: 0.174265
```

因此更严谨的表述是：

> 多轮主动学习后，候选池不确定性呈持续下降趋势；验证集 Force RMSE 整体处于同一量级，但受 toy 数据规模、随机初始化和 committee 模型差异影响，存在一定波动。

详细结果见：

```text
docs/results.md
```

---

## 10. 生成 Round 0–3 Summary 与 Learning Curve

完成 Round 0–3 prediction 后，可以运行 summary 和绘图脚本。

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

这些结果的详细解释见：

```text
docs/results.md
```

---

## 11. 注意事项

1. 本文档复现的是 uncertainty top-K branch；
2. random sampling baseline 不在本文档中记录，详见 `docs/random_baseline.md`；
3. toy H2 数据只用于流程验证，不能代表真实材料体系；
4. 当前 valid set 同时承担 candidate pool 和 validation/test 的角色，真实数据实验中应进一步拆分；
5. 每一轮 selected frames 合并后，candidate pool 必须同步更新；
6. 不要每一轮都从原始 `data/toy_h2/valid` 中重新选择，必须沿着当前 branch 的 candidate pool 往下走；
7. `committee_predictions.npz` 不提交 GitHub；
8. `frozen_model.pb` 和 checkpoint 不提交 GitHub；
9. `selected_topk.json`、`metrics_summary.md`、summary CSV / MD 可以提交 GitHub。

---

## 12. 下一步

完成 uncertainty branch Round 0–3 后，可以继续阅读：

```text
docs/random_baseline.md
```

用于复现：

```text
selection-level random baseline
random seed0 Round 001 retraining baseline
uncertainty branch vs random seed0 candidate-pool uncertainty comparison
```

后续第一阶段重点是补全：

```text
random seed1 / seed2
random Round 0–3 retraining
random mean ± std
full RMSE learning curve
```

---

## 13. 小结

本文档完成了 toy H2 uncertainty branch 的 dataset-level offline active learning 多轮闭环：

```text
Round 0 selected_topk.json
  ↓
Round 1 train / candidate update
  ↓
Round 1 committee retraining
  ↓
Round 1 committee prediction
  ↓
Round 2 train / candidate update
  ↓
Round 2 committee retraining
  ↓
Round 2 committee prediction
  ↓
Round 3 train / candidate update
  ↓
Round 3 committee retraining
  ↓
Round 3 committee prediction
  ↓
Round 0–3 summary and learning curve
```

该阶段说明当前项目已经从：

```text
selection-level active learning record
```

推进到：

```text
dataset-level offline active learning multi-round closed-loop prototype
```