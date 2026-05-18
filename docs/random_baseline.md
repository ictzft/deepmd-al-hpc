# Random Sampling Baseline

本文档用于记录 `deepmd-al-hpc` 当前 toy H2 原型实验中的 random sampling baseline，包括：

```text
selection-level random baseline
  ↓
random seed0 / seed1 / seed2 selection
  ↓
selection-level baseline summary
  ↓
random seed0 Round 001 retraining
  ↓
random seed0 candidate-pool committee prediction
  ↓
uncertainty branch vs random seed0 comparison
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
random seed0 Round 001 retraining baseline
random seed0 candidate-pool prediction summary
```

尚未完成：

```text
random seed1 Round 001 retraining
random seed2 Round 001 retraining
random seed0 / seed1 / seed2 Round 002 retraining
random seed0 / seed1 / seed2 Round 003 retraining
random mean ± std
full RMSE learning curve
end-to-end wall-clock time comparison
```

---

## 2. 当前完成情况

当前 random baseline 已完成内容如下：

```text
Selection-level baseline:
  Round 0 random seed0 / seed1 / seed2
  Round 1 random seed0 / seed1 / seed2

Retraining baseline:
  random seed0 Round 001

Prediction comparison:
  random seed0 Round 001 candidate-pool committee prediction
  uncertainty_round001 vs random_seed0_round001 candidate-pool uncertainty comparison
```

当前还不能直接声称：

```text
uncertainty sampling 显著优于 random sampling
```

更严谨的说法是：

```text
在 toy H2 和 random seed0 Round 001 的初步对比中，
uncertainty sampling 显示出更低的 remaining candidate-pool force model deviation。
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
因此后续需要继续补充 seed1 / seed2，并报告 random mean ± std。
```

---

## 14. Random seed0 Candidate-pool Committee Prediction

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

## 15. Random seed0 Candidate-pool Prediction 当前结果

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

## 16. Uncertainty Branch 与 Random seed0 对比

对比对象：

```text
uncertainty_round001:
  data/toy_h2/round_001_candidate

random_seed0_round001:
  data/toy_h2/random_seed0_round_001_candidate
```

对比表：

| Run | Candidate Pool | n_frames | force_dev_max mean | force_dev_max max | force_dev_max min | energy_dev mean |
|---|---|---:|---:|---:|---:|---:|
| uncertainty_round001 | data/toy_h2/round_001_candidate | 40 | 0.126442 | 0.508339 | 0.042645 | 0.448212 |
| random_seed0_round001 | data/toy_h2/random_seed0_round_001_candidate | 40 | 0.355420 | 1.586355 | 0.086667 | 0.656541 |

主要观察：

```text
在 toy H2 offline active learning 设置下，
加入 10 个 uncertainty-selected frames 后，
剩余 candidate pool 的平均 force model deviation 低于 random seed0 baseline。

这初步表明 uncertainty sampling 比 random seed0 baseline
更有效地降低了候选池不确定性。
```

注意：

```text
该结论目前仍基于 toy H2 和单个 random seed0。
后续必须补充 random seed1 / seed2，并报告 random mean ± std。
```

---

## 17. 当前输出文件汇总

### 17.1 Selection-level Baseline

```text
experiments/baselines/selection_baseline_runs.csv
experiments/baselines/selection_baseline_summary.csv
experiments/baselines/selection_baseline_summary.md
```

---

### 17.2 Random seed0 Round 001 Retraining Baseline

```text
experiments/baselines/random_seed0_round001_metrics_summary.csv
experiments/baselines/random_seed0_round001_metrics_summary.md
```

---

### 17.3 Random seed0 Candidate-pool Prediction

```text
experiments/baselines/random_seed0_round001_prediction_summary.csv
experiments/baselines/random_seed0_round001_prediction_summary.md
experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json
```

---

## 18. 大文件与轻量文件说明

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

## 19. 后续 Random Baseline 补充计划

下一步需要从：

```text
single-seed random baseline
```

推进到：

```text
multi-seed random baseline with mean ± std
```

需要补充：

```text
random seed1 Round 001 retraining
random seed2 Round 001 retraining
random seed0 Round 002 retraining
random seed1 Round 002 retraining
random seed2 Round 002 retraining
random seed0 Round 003 retraining
random seed1 Round 003 retraining
random seed2 Round 003 retraining
```

最终应形成：

```text
random seed0 / seed1 / seed2
Round 0 / Round 1 / Round 2 / Round 3
Force RMSE mean ± std
Energy RMSE mean ± std
candidate-pool uncertainty mean ± std
training time mean ± std
prediction time mean ± std
```

---

## 20. 后续结果文件建议

后续可以新增：

```text
experiments/baselines/random_seed1_round001_metrics_summary.md
experiments/baselines/random_seed2_round001_metrics_summary.md
experiments/baselines/random_seed0_round002_metrics_summary.md
experiments/baselines/random_seed1_round002_metrics_summary.md
experiments/baselines/random_seed2_round002_metrics_summary.md
experiments/baselines/random_vs_uncertainty_summary.csv
experiments/figures/random_vs_uncertainty_rmse.svg
experiments/figures/random_vs_uncertainty_candidate_uncertainty.svg
```

最终在 `docs/results.md` 中统一汇总：

```text
uncertainty branch
vs.
random mean ± std
```

---

## 21. 注意事项

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

## 22. 小结

当前 random sampling baseline 已经从 selection-level 对比推进到 random seed0 Round 001 retraining baseline。

当前完成链路为：

```text
Round 0 committee prediction
  ↓
random seed0 / seed1 / seed2 selection
  ↓
selection-level summary
  ↓
random seed0 selected frames 合并进训练集
  ↓
random seed0 remaining candidate pool 更新
  ↓
random seed0 Round 001 committee retraining
  ↓
random seed0 candidate-pool prediction
  ↓
uncertainty_round001 vs random_seed0_round001 comparison
```

当前结论应谨慎表述为：

```text
在 toy H2 和 random seed0 Round 001 的初步对比中，
uncertainty sampling 相比 random seed0 baseline
显示出更低的 remaining candidate-pool force model deviation。
```

下一阶段重点是：

```text
补充 random seed1 / seed2
  ↓
补充 random Round 0–3 retraining
  ↓
统计 random mean ± std
  ↓
生成 full RMSE learning curve
```