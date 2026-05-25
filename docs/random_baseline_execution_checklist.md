# Random Baseline Round 002/003 执行清单

本文档是面向实际执行的命令清单。所有命令需要在 DeepMD-kit Docker 容器内运行。

**当前状态：Round 001 已完成。Round 002/003 待执行。**

---

## 0. 前提条件

```bash
# 进入项目目录
cd /data/zft/deepmd-al-hpc

# 确认已进入 DeepMD-kit Docker
dp --version
python -c "import deepmd; print(deepmd.__version__)"
nvidia-smi
```

---

## 1. 快速执行（推荐）

使用 `run_random_baseline_round.sh` 一条命令完成一个 (seed, round)：

```bash
# ===== Round 002 =====
bash scripts/run_random_baseline_round.sh 002 seed0
bash scripts/run_random_baseline_round.sh 002 seed1
bash scripts/run_random_baseline_round.sh 002 seed2

# ===== Round 003 (需要 Round 002 完成后) =====
bash scripts/run_random_baseline_round.sh 003 seed0
bash scripts/run_random_baseline_round.sh 003 seed1
bash scripts/run_random_baseline_round.sh 003 seed2
```

该脚本自动完成：数据准备 → 训练 → 预测。

---

## 2. 分步执行（按需调试）

### 2.1 Round 002 — seed0

```bash
# Step 1: 准备数据和配置
python scripts/analysis/prepare_random_baseline_round.py --seed-label seed0 --round-id 2

# Step 2: 训练（2xV100，约 2 分钟）
bash scripts/train/train_round_committee_models.sh \
  002 \
  configs/deepmd/random_seed0_round_002_committee \
  experiments/baselines/random_seed0_round002_committee_models \
  /data/zft/deepmd-al-hpc/data/toy_h2/valid

# Step 3: 预测
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

# Step 4: 汇总（每个 seed 完成后可选）
python scripts/analysis/summarize_random_round_baselines.py --round-id 2
```

### 2.2 Round 002 — seed1

```bash
python scripts/analysis/prepare_random_baseline_round.py --seed-label seed1 --round-id 2
bash scripts/train/train_round_committee_models.sh \
  002 \
  configs/deepmd/random_seed1_round_002_committee \
  experiments/baselines/random_seed1_round002_committee_models \
  /data/zft/deepmd-al-hpc/data/toy_h2/valid
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/random_seed1_round_002_candidate \
  --models \
    experiments/baselines/random_seed1_round002_committee_models/model_000/frozen_model.pb \
    experiments/baselines/random_seed1_round002_committee_models/model_001/frozen_model.pb \
    experiments/baselines/random_seed1_round002_committee_models/model_002/frozen_model.pb \
    experiments/baselines/random_seed1_round002_committee_models/model_003/frozen_model.pb \
  --output experiments/baselines/random_seed1_round002_committee_prediction/committee_predictions.npz \
  --selected-json experiments/baselines/random_seed1_round002_committee_prediction/selected_topk.json \
  --top-k 10
```

### 2.3 Round 002 — seed2

```bash
python scripts/analysis/prepare_random_baseline_round.py --seed-label seed2 --round-id 2
bash scripts/train/train_round_committee_models.sh \
  002 \
  configs/deepmd/random_seed2_round_002_committee \
  experiments/baselines/random_seed2_round002_committee_models \
  /data/zft/deepmd-al-hpc/data/toy_h2/valid
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/random_seed2_round_002_candidate \
  --models \
    experiments/baselines/random_seed2_round002_committee_models/model_000/frozen_model.pb \
    experiments/baselines/random_seed2_round002_committee_models/model_001/frozen_model.pb \
    experiments/baselines/random_seed2_round002_committee_models/model_002/frozen_model.pb \
    experiments/baselines/random_seed2_round002_committee_models/model_003/frozen_model.pb \
  --output experiments/baselines/random_seed2_round002_committee_prediction/committee_predictions.npz \
  --selected-json experiments/baselines/random_seed2_round002_committee_prediction/selected_topk.json \
  --top-k 10
```

### 2.4 Round 003 — seed0

```bash
python scripts/analysis/prepare_random_baseline_round.py --seed-label seed0 --round-id 3
bash scripts/train/train_round_committee_models.sh \
  003 \
  configs/deepmd/random_seed0_round_003_committee \
  experiments/baselines/random_seed0_round003_committee_models \
  /data/zft/deepmd-al-hpc/data/toy_h2/valid
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/random_seed0_round_003_candidate \
  --models \
    experiments/baselines/random_seed0_round003_committee_models/model_000/frozen_model.pb \
    experiments/baselines/random_seed0_round003_committee_models/model_001/frozen_model.pb \
    experiments/baselines/random_seed0_round003_committee_models/model_002/frozen_model.pb \
    experiments/baselines/random_seed0_round003_committee_models/model_003/frozen_model.pb \
  --output experiments/baselines/random_seed0_round003_committee_prediction/committee_predictions.npz \
  --selected-json experiments/baselines/random_seed0_round003_committee_prediction/selected_topk.json \
  --top-k 10
```

### 2.5 Round 003 — seed1 / seed2

替换 seed 和路径后缀，与 2.4 模式相同。

---

## 3. 每轮完成后

### 3.1 检查模型

```bash
find experiments/baselines/random_${SEED}_round${ROUND}_committee_models \
  -maxdepth 2 -name "frozen_model.pb" -type f | sort
# 应该是 4 个文件
```

### 3.2 检查预测输出

```bash
ls experiments/baselines/random_${SEED}_round${ROUND}_committee_prediction/
# 应有: committee_predictions.npz, selected_topk.json
```

### 3.3 生成该轮汇总

```bash
python scripts/analysis/summarize_random_round_baselines.py --round-id ${ROUND}
```

这会生成：
- `experiments/baselines/random_round${ROUND}_baseline_summary.csv`
- `experiments/baselines/random_round${ROUND}_baseline_summary.md`

---

## 4. 全部完成后（Round 001/002/003 都有数据）

### 4.1 更新 uncertainty vs random 汇总

修改 `scripts/analysis/summarize_random_vs_uncertainty.py`，取消注释 Round 002/003 行：

```python
RANDOM_INPUTS = [
    PROJECT_ROOT / "experiments" / "baselines" / "random_round001_baseline_summary.csv",
    PROJECT_ROOT / "experiments" / "baselines" / "random_round002_baseline_summary.csv",  # 取消注释
    PROJECT_ROOT / "experiments" / "baselines" / "random_round003_baseline_summary.csv",  # 取消注释
]
```

然后运行：

```bash
python scripts/analysis/summarize_random_vs_uncertainty.py
python scripts/analysis/plot_random_vs_uncertainty.py
```

### 4.2 查看完整对比图

```bash
ls experiments/figures/random_vs_uncertainty_*.svg
```

---

## 5. Profiling（可选，推荐）

```bash
# 启动 GPU 监控后运行训练
nvidia-smi dmon -s pucvmet -d 2 > experiments/profiling/gpu_dmon_round002_seed0.log &
DMON_PID=$!

bash scripts/run_random_baseline_round.sh 002 seed0

kill $DMON_PID

# 记录预测阶段耗时
bash scripts/profiling/record_round_profiling.sh 002 seed0 random
```

---

## 6. 预计耗时

基于 V100 已有经验（约 1–2 分钟 / model）：

| 阶段 | 模型数 | 预计耗时 |
|---|---|---|
| Round 002 训练 | 3 seeds × 4 models = 12 | ~15–25 min（2×V100 并行） |
| Round 002 预测 | 3 seeds | ~2 min |
| Round 003 训练 | 3 seeds × 4 models = 12 | ~15–25 min（2×V100 并行） |
| Round 003 预测 | 3 seeds | ~2 min |
| **总计** | | **~35–55 min** |

---

## 7. 执行状态追踪

| Seed | Round 001 | Round 002 | Round 003 |
|---|---|---|---|
| seed0 | done | done | done |
| seed1 | done | done | done |
| seed2 | done | done | done |

所有 round 已执行完毕（2026-05-25，2×V100）。
