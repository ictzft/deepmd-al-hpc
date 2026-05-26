# Random Sampling Baseline

本文档用于记录 `deepmd-al-hpc` 当前 toy H2 原型实验中的 random sampling baseline，包括：

```text
selection-level random baseline
  ↓
random seed0 / seed1 / seed2 selection
  ↓
selection-level baseline summary
  ↓
random seed0 / seed1 / seed2 Round 001 retraining
  ↓
random seed0 / seed1 / seed2 candidate-pool committee prediction
  ↓
multi-seed random mean ± std (Round 001)
  ↓
uncertainty branch vs random_seed* comparison
```

该文档承接：

```text
docs/toy_h2_pipeline.md
docs/uncertainty_rounds.md
```

其中 `docs/uncertainty_rounds.md` 记录 uncertainty top-K 主线，而本文档记录 random sampling 对照组。

---

## 1. 文档目标

random sampling baseline 的目标是回答：

```text
uncertainty top-K selection 是否真的比随机选择更有效？
```

具体来说，需要比较：

```text
uncertainty sampling
vs.
random sampling
```

在以下方面的差异：

```text
选中构型的平均不确定性
retraining 后的 Force RMSE / Energy RMSE
retraining 后剩余 candidate pool 的不确定性
多轮 active learning learning curve
多 seed mean ± std
端到端 active learning wall-clock time
```

当前阶段已经完成：

```text
selection-level random baseline
random seed0 / seed1 / seed2 Round 001 retraining baseline
random seed0 / seed1 / seed2 candidate-pool prediction summary
random multi-seed mean ± std (Round 001)
```

已全部完成（2026-05-25）：

```text
random seed0 / seed1 / seed2 Round 002 retraining
random seed0 / seed1 / seed2 Round 003 retraining
full RMSE learning curve (uncertainty vs random, 4 SVG figures)
multi-round candidate-pool uncertainty comparison
```

仍待完成：

```text
end-to-end wall-clock time (systematic profiling with GPU monitoring)
```

---

## 2. 当前完成情况

当前 random baseline 已完成内容如下：

```text
Selection-level baseline:
  Round 0 random seed0 / seed1 / seed2
  Round 1 random seed0 / seed1 / seed2

Retraining baseline (all completed 2026-05-25 on 2×V100):
  random seed0 Round 001 / 002 / 003
  random seed1 Round 001 / 002 / 003
  random seed2 Round 001 / 002 / 003

Prediction comparison:
  random seed0 / seed1 / seed2 Round 001–003 candidate-pool committee prediction
  multi-seed random mean ± std (Round 001/002/003)
  uncertainty branch vs random multi-seed full comparison table + learning curves

Profiling:
  training wall time recorded per model from train.log (mean=10.9s/model, n=36)
  2×V100 parallel speedup: 1.97× (near-linear)
```

当前还不能直接声称：

```text
uncertainty sampling 显著优于 random sampling
```

更严谨的说法是：

```text
在 toy H2 和 Round 001 的 random seed0 / seed1 / seed2 对比中，
uncertainty sampling 一致显示出更低的 remaining candidate-pool force model deviation。
```

---

## 3. Selection-level Random Baseline

selection-level baseline 只比较不同 selection strategy 选出的构型特征，不进行 retraining。

该阶段主要回答：

```text
uncertainty top-K 是否确实选中了更高不确定性的构型？
random sampling 选中的构型平均不确定性是多少？
```

注意：

```text
selection-level baseline 不能直接代表 retraining 后模型精度差异。
```

---

## 4. Round 0 Selection-level Baseline

Round 0 使用初始 committee prediction 结果：

```text
experiments/exp_005_committee_prediction/committee_predictions.npz
```

模板文件：

```text
experiments/exp_005_committee_prediction/selected_topk.json
```

---

### 4.1 生成 Round 0 uncertainty selection JSON

```bash
python scripts/active_learning/select_from_predictions.py \
  --predictions experiments/exp_005_committee_prediction/committee_predictions.npz \
  --strategy uncertainty \
  --top-k 10 \
  --template-json experiments/exp_005_committee_prediction/selected_topk.json \
  --output experiments/exp_005_committee_prediction/selected_uncertainty.json
```

输出：

```text
experiments/exp_005_committee_prediction/selected_uncertainty.json
```

---

### 4.2 生成 Round 0 random seed0 selection JSON

```bash
python scripts/active_learning/select_from_predictions.py \
  --predictions experiments/exp_005_committee_prediction/committee_predictions.npz \
  --strategy random \
  --top-k 10 \
  --seed 0 \
  --template-json experiments/exp_005_committee_prediction/selected_topk.json \
  --output experiments/exp_005_committee_prediction/selected_random_seed0.json
```

输出：

```text
experiments/exp_005_committee_prediction/selected_random_seed0.json
```

---

### 4.3 生成 Round 0 random seed1 selection JSON

```bash
python scripts/active_learning/select_from_predictions.py \
  --predictions experiments/exp_005_committee_prediction/committee_predictions.npz \
  --strategy random \
  --top-k 10 \
  --seed 1 \
  --template-json experiments/exp_005_committee_prediction/selected_topk.json \
  --output experiments/exp_005_committee_prediction/selected_random_seed1.json
```

输出：

```text
experiments/exp_005_committee_prediction/selected_random_seed1.json
```

---

### 4.4 生成 Round 0 random seed2 selection JSON

```bash
python scripts/active_learning/select_from_predictions.py \
  --predictions experiments/exp_005_committee_prediction/committee_predictions.npz \
  --strategy random \
  --top-k 10 \
  --seed 2 \
  --template-json experiments/exp_005_committee_prediction/selected_topk.json \
  --output experiments/exp_005_committee_prediction/selected_random_seed2.json
```

输出：

```text
experiments/exp_005_committee_prediction/selected_random_seed2.json
```

---

## 5. Round 1 Selection-level Baseline

Round 1 使用 Round 1 committee prediction 结果：

```text
experiments/exp_008_round001_committee_prediction/committee_predictions.npz
```

模板文件：

```text
experiments/exp_008_round001_committee_prediction/selected_topk.json
```

---

### 5.1 生成 Round 1 uncertainty selection JSON

```bash
python scripts/active_learning/select_from_predictions.py \
  --predictions experiments/exp_008_round001_committee_prediction/committee_predictions.npz \
  --strategy uncertainty \
  --top-k 10 \
  --template-json experiments/exp_008_round001_committee_prediction/selected_topk.json \
  --output experiments/exp_008_round001_committee_prediction/selected_uncertainty.json
```

---

### 5.2 生成 Round 1 random seed0 selection JSON

```bash
python scripts/active_learning/select_from_predictions.py \
  --predictions experiments/exp_008_round001_committee_prediction/committee_predictions.npz \
  --strategy random \
  --top-k 10 \
  --seed 0 \
  --template-json experiments/exp_008_round001_committee_prediction/selected_topk.json \
  --output experiments/exp_008_round001_committee_prediction/selected_random_seed0.json
```

---

### 5.3 生成 Round 1 random seed1 selection JSON

```bash
python scripts/active_learning/select_from_predictions.py \
  --predictions experiments/exp_008_round001_committee_prediction/committee_predictions.npz \
  --strategy random \
  --top-k 10 \
  --seed 1 \
  --template-json experiments/exp_008_round001_committee_prediction/selected_topk.json \
  --output experiments/exp_008_round001_committee_prediction/selected_random_seed1.json
```

---

### 5.4 生成 Round 1 random seed2 selection JSON

```bash
python scripts/active_learning/select_from_predictions.py \
  --predictions experiments/exp_008_round001_committee_prediction/committee_predictions.npz \
  --strategy random \
  --top-k 10 \
  --seed 2 \
  --template-json experiments/exp_008_round001_committee_prediction/selected_topk.json \
  --output experiments/exp_008_round001_committee_prediction/selected_random_seed2.json
```

---

## 6. 汇总 Selection-level Baseline

生成完 Round 0 和 Round 1 的 uncertainty / random selection JSON 后，运行：

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

## 7. Selection-level Baseline 当前结果

当前 selection-level 对比结果如下：

```text
Round 000:
random force_dev_max_mean      : 0.143007
uncertainty force_dev_max_mean : 0.440989

Round 001:
random force_dev_max_mean      : 0.145731
uncertainty force_dev_max_mean : 0.269333
```

该结果说明：

```text
uncertainty top-K 策略确实选中了平均不确定性更高的构型；
random sampling 选中的构型平均不确定性较低。
```

但需要注意：

```text
selection-level baseline 只能说明不同 selection strategy 选出的构型不确定性不同；
不能直接说明 retraining 后模型精度存在优势。
```

要进一步证明主动学习效果，需要比较：

```text
retrained committee models 的 Force RMSE
retrained committee models 的 Energy RMSE
remaining candidate pool 的 force_dev_max_mean
不同 seed 下的 mean ± std
```

---

## 8. Random seed0 Round 001 Retraining Baseline

当前已经完成 random seed0 Round 001 retraining baseline。

该实验对应：

```text
从初始 candidate pool 中随机选择 10 个 frames
  ↓
合并进初始训练集
  ↓
构造 random seed0 Round 001 train set
  ↓
构造 random seed0 Round 001 remaining candidate pool
  ↓
重新训练 4 个 committee models
  ↓
测试 random seed0 committee models
```

---

## 9. 构造 random seed0 Round 001 Train Set

使用 Round 0 random seed0 selection：

```text
experiments/exp_005_committee_prediction/selected_random_seed0.json
```

构造 random seed0 Round 001 train set：

```bash
python scripts/data/merge_selected_frames.py \
  --train data/toy_h2/train \
  --candidate data/toy_h2/valid \
  --selection experiments/exp_005_committee_prediction/selected_random_seed0.json \
  --output data/toy_h2/random_seed0_round_001_train \
  --overwrite
```

输出目录：

```text
data/toy_h2/random_seed0_round_001_train
```

数据规模：

```text
200 initial train frames + 10 random-selected frames = 210 frames
```

检查结果：

```text
coord.npy  : (210, 6)
box.npy    : (210, 9)
energy.npy : (210,)
force.npy  : (210, 6)
```

---

## 10. 构造 random seed0 Round 001 Remaining Candidate

从初始 candidate pool 中移除 random seed0 已选中的 frames：

```bash
python scripts/data/make_remaining_candidate.py \
  --candidate data/toy_h2/valid \
  --selection experiments/exp_005_committee_prediction/selected_random_seed0.json \
  --output data/toy_h2/random_seed0_round_001_candidate \
  --overwrite
```

输出目录：

```text
data/toy_h2/random_seed0_round_001_candidate
```

数据规模：

```text
50 initial candidate frames - 10 random-selected frames = 40 frames
```

检查结果：

```text
coord.npy  : (40, 6)
box.npy    : (40, 9)
energy.npy : (40,)
force.npy  : (40, 6)
```

---

## 11. Random seed0 Round 001 Committee Config

random seed0 Round 001 config 目录：

```text
configs/deepmd/random_seed0_round_001_committee/
```

4 个配置文件均应指向：

```text
/data/zft/data/toy_h2/random_seed0_round_001_train
```

检查配置：

```bash
grep -R "training_data\|systems\|random_seed0_round_001_train" -n \
  configs/deepmd/random_seed0_round_001_committee
```

JSON 语法检查：

```bash
python - <<'PY'
import json
from pathlib import Path

config_dir = Path("configs/deepmd/random_seed0_round_001_committee")

for p in sorted(config_dir.glob("*.json")):
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print("OK", p)
    print("  train:", data["training"]["training_data"]["systems"])
    print("  valid:", data["training"]["validation_data"]["systems"])
PY
```

---

## 12. 训练 random seed0 Round 001 Committee Models

```bash
bash scripts/train/train_round_committee_models.sh \
  001 \
  configs/deepmd/random_seed0_round_001_committee \
  experiments/baselines/random_seed0_round001_committee_models \
  /data/zft/data/toy_h2/valid
```

输出目录：

```text
experiments/baselines/random_seed0_round001_committee_models/
```

该目录不提交到 GitHub，因为其中包含：

```text
frozen_model.pb
checkpoint
model.ckpt*
train.log
test.log
lcurve.out
```

检查 4 个 frozen models 是否生成：

```bash
find experiments/baselines/random_seed0_round001_committee_models \
  -maxdepth 2 \
  -name "frozen_model.pb" \
  -type f \
  | sort
```

预期输出：

```text
experiments/baselines/random_seed0_round001_committee_models/model_000/frozen_model.pb
experiments/baselines/random_seed0_round001_committee_models/model_001/frozen_model.pb
experiments/baselines/random_seed0_round001_committee_models/model_002/frozen_model.pb
experiments/baselines/random_seed0_round001_committee_models/model_003/frozen_model.pb
```

---

## 13. Random seed0 Round 001 Committee Metrics

当前 random seed0 Round 001 committee 测试结果：

| Model | Energy RMSE / eV | Force RMSE / eV/Å |
|---|---:|---:|
| model_000 | 8.813657e-01 | 2.551970e-01 |
| model_001 | 9.782892e-03 | 1.827423e-01 |
| model_002 | 1.965928e-01 | 8.939603e-02 |
| model_003 | 1.675800e+00 | 4.940109e-01 |
| Mean | 6.908853e-01 | 2.553366e-01 |
| Std | 7.559906e-01 | 1.729852e-01 |

相关轻量结果：

```text
experiments/baselines/random_seed0_round001_metrics_summary.csv
experiments/baselines/random_seed0_round001_metrics_summary.md
```

说明：

```text
random seed0 committee model 方差较大；
因此需要补充 seed1 / seed2，并报告 random mean ± std。
```

---

## 14. Random seed1 Round 001 Committee Metrics

当前 random seed1 Round 001 committee 测试结果：

| Model | Energy RMSE / eV | Force RMSE / eV/Å |
|---|---:|---:|
| model_000 | 9.330507e-01 | 2.555631e-01 |
| model_001 | 8.658584e-03 | 2.624681e-01 |
| model_002 | 4.101113e-02 | 1.900886e-01 |
| model_003 | 1.249115e-02 | 2.072284e-01 |
| Mean | 2.488029e-01 | 2.288370e-01 |
| Std | 4.563935e-01 | 3.565438e-02 |

相关轻量结果：

```text
experiments/baselines/random_seed1_round001_metrics_summary.csv
experiments/baselines/random_seed1_round001_metrics_summary.md
```

---

## 15. Random seed2 Round 001 Committee Metrics

当前 random seed2 Round 001 committee 测试结果：

| Model | Energy RMSE / eV | Force RMSE / eV/Å |
|---|---:|---:|
| model_000 | 1.034722e+00 | 1.450501e-01 |
| model_001 | 1.650412e-02 | 2.141395e-01 |
| model_002 | 6.554224e-01 | 1.356484e-01 |
| model_003 | 7.000031e-03 | 1.031312e-01 |
| Mean | 4.284121e-01 | 1.494923e-01 |
| Std | 5.054377e-01 | 4.669047e-02 |

相关轻量结果：

```text
experiments/baselines/random_seed2_round001_metrics_summary.csv
experiments/baselines/random_seed2_round001_metrics_summary.md
```

---

## 16. Multi-seed Random Round 001 汇总

三个 seed 的 Round 001 retraining 结果汇总：

| Seed | Energy RMSE Mean / eV | Energy RMSE Std / eV | Force RMSE Mean / eV/Å | Force RMSE Std / eV/Å |
|---|---:|---:|---:|---:|
| seed0 | 6.908853e-01 | 7.559906e-01 | 2.553366e-01 | 1.729852e-01 |
| seed1 | 2.488029e-01 | 4.563935e-01 | 2.288370e-01 | 3.565438e-02 |
| seed2 | 4.284121e-01 | 5.054377e-01 | 1.494923e-01 | 4.669047e-02 |
| **Mean** | **4.560335e-01** | — | **2.112220e-01** | — |
| **Std** | **2.223318e-01** | — | **5.507695e-02** | — |

说明：

```text
三个 random seed 的 committee model 间存在较大方差；
该方差源自 toy H2 数据规模较小和 committee 随机初始化；
Round 002/003 已完成（2026-05-25, 2×V100），完整 multi-round learning curve 已生成。
```

相关汇总文件：

```text
experiments/baselines/random_round001_comparison.csv
experiments/baselines/random_round001_baseline_summary.csv
experiments/baselines/random_round001_baseline_summary.md
```

---

## 17. Random seed1 / seed2 Round001 Reproducibility Commands

本节提供 random seed1 和 random seed2 Round001 retraining 的完整复现命令。

### 17.1 生成 random seed1 / seed2 selection JSON

```bash
python scripts/active_learning/select_from_predictions.py \
  --predictions experiments/exp_005_committee_prediction/committee_predictions.npz \
  --strategy random \
  --top-k 10 \
  --seed 1 \
  --template-json experiments/exp_005_committee_prediction/selected_topk.json \
  --output experiments/exp_005_committee_prediction/selected_random_seed1.json

python scripts/active_learning/select_from_predictions.py \
  --predictions experiments/exp_005_committee_prediction/committee_predictions.npz \
  --strategy random \
  --top-k 10 \
  --seed 2 \
  --template-json experiments/exp_005_committee_prediction/selected_topk.json \
  --output experiments/exp_005_committee_prediction/selected_random_seed2.json
```

### 17.2 构造 seed1 Round001 train / candidate

```bash
python scripts/data/merge_selected_frames.py \
  --train data/toy_h2/train \
  --candidate data/toy_h2/valid \
  --selection experiments/exp_005_committee_prediction/selected_random_seed1.json \
  --output data/toy_h2/random_seed1_round_001_train \
  --overwrite

python scripts/data/make_remaining_candidate.py \
  --candidate data/toy_h2/valid \
  --selection experiments/exp_005_committee_prediction/selected_random_seed1.json \
  --output data/toy_h2/random_seed1_round_001_candidate \
  --overwrite
```

### 17.3 构造 seed2 Round001 train / candidate

```bash
python scripts/data/merge_selected_frames.py \
  --train data/toy_h2/train \
  --candidate data/toy_h2/valid \
  --selection experiments/exp_005_committee_prediction/selected_random_seed2.json \
  --output data/toy_h2/random_seed2_round_001_train \
  --overwrite

python scripts/data/make_remaining_candidate.py \
  --candidate data/toy_h2/valid \
  --selection experiments/exp_005_committee_prediction/selected_random_seed2.json \
  --output data/toy_h2/random_seed2_round_001_candidate \
  --overwrite
```

### 17.4 生成 seed1 / seed2 committee configs

```bash
python scripts/config/make_round_committee_configs.py \
  --base configs/deepmd/toy_h2_input.json \
  --output-dir configs/deepmd/random_seed1_round_001_committee \
  --train-system /data/zft/deepmd-al-hpc/data/toy_h2/random_seed1_round_001_train \
  --valid-system /data/zft/deepmd-al-hpc/data/toy_h2/valid \
  --round-id 1 \
  --n-models 4 \
  --base-seed 2101

python scripts/config/make_round_committee_configs.py \
  --base configs/deepmd/toy_h2_input.json \
  --output-dir configs/deepmd/random_seed2_round_001_committee \
  --train-system /data/zft/deepmd-al-hpc/data/toy_h2/random_seed2_round_001_train \
  --valid-system /data/zft/deepmd-al-hpc/data/toy_h2/valid \
  --round-id 1 \
  --n-models 4 \
  --base-seed 2201
```

### 17.5 训练 seed1 / seed2 committee models

```bash
bash scripts/train/train_round_committee_models.sh \
  001 \
  configs/deepmd/random_seed1_round_001_committee \
  experiments/baselines/random_seed1_round001_committee_models \
  /data/zft/deepmd-al-hpc/data/toy_h2/valid

bash scripts/train/train_round_committee_models.sh \
  001 \
  configs/deepmd/random_seed2_round_001_committee \
  experiments/baselines/random_seed2_round001_committee_models \
  /data/zft/deepmd-al-hpc/data/toy_h2/valid
```

### 17.6 检查模型是否生成

```bash
find experiments/baselines/random_seed1_round001_committee_models \
  -maxdepth 2 -name "frozen_model.pb" -type f | sort

find experiments/baselines/random_seed2_round001_committee_models \
  -maxdepth 2 -name "frozen_model.pb" -type f | sort
```

每个 seed 应该有 4 个模型：

```text
model_000/frozen_model.pb
model_001/frozen_model.pb
model_002/frozen_model.pb
model_003/frozen_model.pb
```

### 17.7 对 seed1 / seed2 candidate pool 做 committee prediction

```bash
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/random_seed1_round_001_candidate \
  --models \
  experiments/baselines/random_seed1_round001_committee_models/model_000/frozen_model.pb \
  experiments/baselines/random_seed1_round001_committee_models/model_001/frozen_model.pb \
  experiments/baselines/random_seed1_round001_committee_models/model_002/frozen_model.pb \
  experiments/baselines/random_seed1_round001_committee_models/model_003/frozen_model.pb \
  --output experiments/baselines/random_seed1_round001_committee_prediction/committee_predictions.npz \
  --selected-json experiments/baselines/random_seed1_round001_committee_prediction/selected_topk.json \
  --top-k 10

python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/random_seed2_round_001_candidate \
  --models \
  experiments/baselines/random_seed2_round001_committee_models/model_000/frozen_model.pb \
  experiments/baselines/random_seed2_round001_committee_models/model_001/frozen_model.pb \
  experiments/baselines/random_seed2_round001_committee_models/model_002/frozen_model.pb \
  experiments/baselines/random_seed2_round001_committee_models/model_003/frozen_model.pb \
  --output experiments/baselines/random_seed2_round001_committee_prediction/committee_predictions.npz \
  --selected-json experiments/baselines/random_seed2_round001_committee_prediction/selected_topk.json \
  --top-k 10
```

### 17.8 生成三 seed 汇总

```bash
python scripts/analysis/summarize_random_round001_baselines.py
```

输出文件：

```text
experiments/baselines/random_round001_baseline_summary.csv
experiments/baselines/random_round001_baseline_summary.md
```

---

## 18. Random seed0 Candidate-pool Committee Prediction

使用 random seed0 Round 001 committee models 对剩余 candidate pool 进行 prediction。

运行：

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

输出文件：

```text
experiments/baselines/random_seed0_round001_committee_prediction/committee_predictions.npz
experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json
```

说明：

```text
committee_predictions.npz 不提交到 GitHub。
selected_topk.json 可以提交到 GitHub。
```

---

## 19. Random seed0 Candidate-pool Prediction 当前结果

当前 random seed0 candidate-pool prediction 结果：

```text
n_frames: 40

force_dev_max_mean: 0.355420
force_dev_max_max : 1.586355
force_dev_max_min : 0.086667

energy_dev_mean   : 0.656541
energy_dev_max    : 0.810345
energy_dev_min    : 0.642052
```

相关轻量结果：

```text
experiments/baselines/random_seed0_round001_prediction_summary.csv
experiments/baselines/random_seed0_round001_prediction_summary.md
```

---

## 20. Uncertainty Branch 与 Random Baseline 对比

对比对象：

```text
uncertainty_round001:
  data/toy_h2/round_001_candidate

random_seed0_round001:
  data/toy_h2/random_seed0_round_001_candidate

random_seed1_round001:
  data/toy_h2/random_seed1_round_001_candidate

random_seed2_round001:
  data/toy_h2/random_seed2_round_001_candidate
```

对比表：

| Run | n_frames | force_dev_max mean | force_dev_max max | force_dev_max min | energy_dev mean |
|---|---:|---:|---:|---:|---:|
| uncertainty_round001 | 40 | 0.126442 | 0.508339 | 0.042645 | 0.448212 |
| random_seed0_round001 | 40 | 0.355420 | 1.586355 | 0.086667 | 0.656541 |
| random_seed1_round001 | 40 | 0.487795 | 1.262038 | 0.327483 | 0.396726 |
| random_seed2_round001 | 40 | 0.332138 | 1.117230 | 0.139321 | 0.446260 |
| **random mean** | 40 | **0.391784** | — | — | **0.499842** |

主要观察：

```text
在 toy H2 offline active learning 设置下，
加入 10 个 uncertainty-selected frames 后，
剩余 candidate pool 的平均 force model deviation (0.126442)
低于所有三个 random seed baseline (seed0: 0.355420, seed1: 0.487795, seed2: 0.332138)。

在当前 toy H2 和 Round 001 的 setting 下，
uncertainty branch 的 remaining candidate-pool force_dev_max_mean
在 seed0 / seed1 / seed2 三个对比中均低于对应的 random branch。
这为后续多轮 retraining 对比提供了初步 baseline 参考。
```

注意：

```text
该结论目前仍基于 toy H2 数据集，Round 001–003 multi-round retraining 已补充完成。
Round 002/003 多轮 random retraining 已完成（2026-05-25），
并生成完整 RMSE learning curve 对比。
```

---

## 21. 当前输出文件汇总

### 21.1 Selection-level Baseline

```text
experiments/baselines/selection_baseline_runs.csv
experiments/baselines/selection_baseline_summary.csv
experiments/baselines/selection_baseline_summary.md
```

---

### 21.2 Random seed0 Round 001 Retraining Baseline

```text
experiments/baselines/random_seed0_round001_metrics_summary.csv
experiments/baselines/random_seed0_round001_metrics_summary.md
```

---

### 21.3 Random seed1 Round 001 Retraining Baseline

```text
experiments/baselines/random_seed1_round001_metrics_summary.csv
experiments/baselines/random_seed1_round001_metrics_summary.md
```

---

### 21.4 Random seed2 Round 001 Retraining Baseline

```text
experiments/baselines/random_seed2_round001_metrics_summary.csv
experiments/baselines/random_seed2_round001_metrics_summary.md
```

---

### 21.5 Multi-seed Round 001 Comparison

```text
experiments/baselines/random_round001_comparison.csv
```

---

### 21.6 Random seed0 Candidate-pool Prediction

```text
experiments/baselines/random_seed0_round001_prediction_summary.csv
experiments/baselines/random_seed0_round001_prediction_summary.md
experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json
```

---

### 21.7 Random seed1 Candidate-pool Prediction

```text
experiments/baselines/random_seed1_round001_prediction_summary.csv
experiments/baselines/random_seed1_round001_prediction_summary.md
experiments/baselines/random_seed1_round001_committee_prediction/selected_topk.json
```

---

### 21.8 Random seed2 Candidate-pool Prediction

```text
experiments/baselines/random_seed2_round001_prediction_summary.csv
experiments/baselines/random_seed2_round001_prediction_summary.md
experiments/baselines/random_seed2_round001_committee_prediction/selected_topk.json
```

---

## 22. 大文件与轻量文件说明

以下内容默认不提交 GitHub：

```text
data/toy_h2/random_seed0_round_001_train/
data/toy_h2/random_seed0_round_001_candidate/
experiments/baselines/random_seed0_round001_committee_models/
experiments/baselines/random_seed0_round001_committee_prediction/committee_predictions.npz
*.pb
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
selected_random_seed*.json
selected_uncertainty.json
selected_topk.json
selection_baseline_runs.csv
selection_baseline_summary.csv
selection_baseline_summary.md
random_seed0_round001_metrics_summary.csv
random_seed0_round001_metrics_summary.md
random_seed0_round001_prediction_summary.csv
random_seed0_round001_prediction_summary.md
```

---

## 23. 后续 Random Baseline 补充计划

已完成（2026-05-25, 2×V100）：

```text
random seed0 / seed1 / seed2 Round 001 retraining ✓
random seed0 / seed1 / seed2 Round 002 retraining ✓
random seed0 / seed1 / seed2 Round 003 retraining ✓
multi-seed random mean ± std (Round 001/002/003) ✓
```

最终已形成：

```text
random seed0 / seed1 / seed2
Round 0 / Round 1 / Round 2 / Round 3
Force RMSE mean ± std ✓
Energy RMSE mean ± std ✓
candidate-pool uncertainty mean ± std ✓
training time mean ± std ✓
prediction time mean ± std ✓
```

Multi-round random baseline 已全部完成，完整 learning curve 和 uncertainty vs random 对比已生成。

---

## 24. 后续结果文件建议

已生成（2026-05-25）：

```text
experiments/baselines/random_seed0_round002_metrics_summary.md
experiments/baselines/random_seed1_round002_metrics_summary.md
experiments/baselines/random_seed2_round002_metrics_summary.md
experiments/baselines/random_seed0_round003_metrics_summary.md
experiments/baselines/random_seed1_round003_metrics_summary.md
experiments/baselines/random_seed2_round003_metrics_summary.md
experiments/baselines/random_round002_baseline_summary.csv
experiments/baselines/random_round003_baseline_summary.csv
```

当前已生成：

```text
experiments/baselines/random_vs_uncertainty_summary.csv
experiments/baselines/random_vs_uncertainty_summary.md
experiments/figures/random_vs_uncertainty_force_rmse.svg
experiments/figures/random_vs_uncertainty_energy_rmse.svg
experiments/figures/random_vs_uncertainty_candidate_force_dev.svg
experiments/figures/random_vs_uncertainty_dataset_size.svg
```

---

## 25. 注意事项

1. selection-level baseline 只比较“选出来的样本”，不能证明 retraining 后效果；
2. retraining baseline 才能比较 Force RMSE、Energy RMSE 和 candidate-pool uncertainty；
3. random seed0 只是一个随机种子，不能代表 random sampling 的整体表现；
4. 后续至少需要 seed0 / seed1 / seed2；
5. 每个 random seed 应该形成独立的 active learning branch；
6. 不要每一轮都从初始 candidate pool 重新随机选择；
7. random Round 2 / Round 3 应该基于该 seed 自己上一轮更新后的 train / candidate；
8. 最终论文级结果应报告 mean ± std；
9. random baseline 的训练配置应与 uncertainty branch 保持一致；
10. 唯一区别应是 selection strategy 不同。

---

## 26. 小结

当前 random sampling baseline 已经全部完成（2026-05-25, 2×V100）：selection-level 对比、random seed0 / seed1 / seed2 Round 001–003 retraining baseline、multi-seed random mean ± std、以及 uncertainty vs random full comparison + learning curves 均已生成。

当前完成链路为：

```text
Round 0 committee prediction
  ↓
random seed0 / seed1 / seed2 selection
  ↓
selection-level summary
  ↓
random seed0 / seed1 / seed2 selected frames 合并进训练集
  ↓
random seed0 / seed1 / seed2 remaining candidate pool 更新
  ↓
random seed0 / seed1 / seed2 Round 001 committee retraining
  ↓
random seed0 / seed1 / seed2 candidate-pool prediction
  ↓
random seed0 / seed1 / seed2 Round 002 committee retraining
  ↓
random seed0 / seed1 / seed2 Round 002 candidate-pool prediction
  ↓
random seed0 / seed1 / seed2 Round 003 committee retraining
  ↓
random seed0 / seed1 / seed2 Round 003 candidate-pool prediction
  ↓
uncertainty vs random full multi-round comparison + learning curves
  ↓
四策略 aligned comparison (uncertainty / random / diversity / trust_level)
```

当前结论应谨慎表述为：

```text
在 toy H2 和 Round 001 的 random seed0 / seed1 / seed2 对比中，
uncertainty sampling 一致显示出更低的 remaining candidate-pool force model deviation
(0.126442 vs random mean 0.391784)。
```

下一阶段重点是：

```text
迁移到真实 DFT / AIMD 数据集（rMD17 ethanol 已启动）
  ↓
GPU utilization / memory 系统曲线记录
  ↓
H100 / 多 GPU scaling
```