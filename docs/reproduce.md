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

---

## 1. 进入项目目录

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

## 2. 进入 DeepMD-kit Docker 环境

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

验证 DeepMD-kit 环境：

```bash
dp -h
lmp -h
python -c "import deepmd; print('deepmd import ok')"
python -c "from deepmd.infer import DeepPot; print('DeepPot import ok')"
python -c "import numpy as np; print(np.__version__)"
nvidia-smi
```

如果在宿主机上执行 Python 脚本且没有 `python` 命令，可使用：

```bash
python3
```

---

## 3. 生成 toy H2 DeepMD 数据

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

说明：`data/` 目录默认不提交到 GitHub。

---

## 4. 跑单模型 DeePMD baseline

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

---

## 5. 训练 4-model committee

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

说明：checkpoint、frozen model、日志等大型输出不提交到 GitHub。

---

## 6. 做 committee prediction

使用 4 个 frozen DeePMD models 对 candidate pool 进行真实推理。

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

输出内容包括：

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

## 7. 生成 offline active learning selection 记录

将 `exp_005` 的 top-K 选择结果整理为一轮 offline active learning 记录。

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

## 8. 合并 selected frames，生成 Round 1 训练集

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

---

## 9. 更新 candidate pool，生成 Round 1 candidate

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

---

## 10. 生成 Round 1 committee 配置

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

---

## 11. 训练 Round 1 committee models

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

---

## 12. Round 1 committee prediction

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

## 13. 生成 Round 2 训练集与 candidate pool

合并 Round 1 selected frames：

```bash
python scripts/data/merge_selected_frames.py \
  --base data/toy_h2/round_001_train \
  --candidate data/toy_h2/round_001_candidate \
  --selected-json experiments/exp_008_round001_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_002_train
```

更新 candidate pool：

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

---

## 14. 生成并训练 Round 2 committee

生成配置：

```bash
python scripts/config/make_round_committee_configs.py \
  --base-config configs/deepmd/toy_h2_input.json \
  --train-data data/toy_h2/round_002_train \
  --output-dir configs/deepmd/round_002_committee \
  --round-id 2 \
  --seed-base 1201
```

训练：

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

---

## 15. Round 2 committee prediction

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

## 16. 生成 Round 3 训练集与 candidate pool

合并 Round 2 selected frames：

```bash
python scripts/data/merge_selected_frames.py \
  --base data/toy_h2/round_002_train \
  --candidate data/toy_h2/round_002_candidate \
  --selected-json experiments/exp_010_round002_committee_prediction/selected_topk.json \
  --output data/toy_h2/round_003_train
```

更新 candidate pool：

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

---

## 17. 生成并训练 Round 3 committee

生成配置：

```bash
python scripts/config/make_round_committee_configs.py \
  --base-config configs/deepmd/toy_h2_input.json \
  --train-data data/toy_h2/round_003_train \
  --output-dir configs/deepmd/round_003_committee \
  --round-id 3 \
  --seed-base 1301
```

训练：

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

---

## 18. Round 3 committee prediction

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

---

## 19. 汇总 Round 0–3 learning curve

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

当前 Round 0–3 summary：

```text
Round 0: train 200, candidate 50
Round 1: train 210, candidate 40
Round 2: train 220, candidate 30
Round 3: train 230, candidate 20
```

当前 force deviation 变化：

```text
force_dev_max_mean:
Round 0: 0.440989
Round 1: 0.269333
Round 2: 0.187412
Round 3: 0.170189
```

说明：当前结果主要用于验证 active learning pipeline 和工程流程，不用于评价真实材料体系上的模型精度。

---

## 20. 代码与配置检查

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
```

---

## 21. GitHub 提交建议

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

提交前检查：

```bash
git status --short
git diff
```

提交：

```bash
git add docs/reproduce.md
git commit -m "Add reproduction guide for active learning pipeline"
git push origin main
```

---

## 22. 后续迁移到 H100 时的复用方式

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