# V100 Profiling 方案与实操指南

本文档定义在 2×Tesla V100 GPU 平台上对 `deepmd-al-hpc` 主动学习闭环进行系统性能 profiling 的方案和具体操作命令。

**当前状态：方案阶段。** 以下大部分指标和命令模板已就绪，但尚未系统执行所有 round 的 profiling。当前只有少量零散耗时记录（见第 6 节）。本文档的目标是提供可立即执行的 profiling 操作指南。

## 1. 为什么 V100 阶段需要做 profiling

V100 阶段的 profiling 目标不是追求极限性能，而是：

- 建立一套可复现的系统性能记录方式；
- 明确主动学习闭环中各阶段的耗时占比；
- 识别主要性能瓶颈（training vs inference vs I/O）；
- 为后续 H100 迁移和多 GPU scaling 提供 baseline 对比数据。

当前 V100 profiling 只零散记录了部分训练和预测耗时（见 `experiments/al_rounds_summary.csv` 中的 `train_elapsed_s` 和 `prediction_elapsed_s` 列），尚未系统记录所有 round 和所有 seed 的完整性能数据。

---

## 2. 需要统计的性能指标

### 2.1 单次训练耗时

**指标**：single-model training wall-clock time

**测量方式**：在 `train_round_committee_models.sh` 中已有的 `date` 时间戳基础上，增加 `time` 命令精确记录。

**记录内容**：

```text
- model_id: model_000 ~ model_003
- round_id: 0, 1, 2, 3
- branch: uncertainty / random_seed0 / random_seed1 / random_seed2
- start_time: ISO 8601
- end_time: ISO 8601
- elapsed_seconds: 总耗时
- n_steps: 训练步数（从 lcurve.out 读取）
- GPU: V100-16GB / V100-32GB
```

**当前状态**：`train_round_committee_models.sh` 已记录 start/end time。`experiments/al_rounds_summary.csv` 中已有 `train_elapsed_s` 列但大多为空。

### 2.2 4-model committee serial training time

**指标**：串行训练 4 个模型的总 wall-clock time。

**测量方式**：4 个模型串行执行时，从第一个模型开始到第四个模型结束的总时间。

### 2.3 2×V100 model-level parallel training time

**指标**：2 个 GPU 各跑 2 个模型时的总 wall-clock time。

**测量方式**：当前 `train_round_committee_models.sh` 已实现 2-GPU 并行（GPU 0 跑 model_000/002，GPU 1 跑 model_001/003）。需要记录从并行启动到全部完成的 wall-clock time。

**对比意义**：

```text
serial_4_models vs parallel_2x2_models
→ 量化 model-level parallelism 的实际加速比
```

### 2.4 Candidate prediction time

**指标**：单次 committee prediction 的 wall-clock time。

**测量方式**：在 `predict_committee_models.py` 中增加计时逻辑，或在调用脚本中用 `time` 命令包裹。

**记录内容**：

```text
- n_candidate_frames: 候选池帧数
- n_committee_models: 4
- elapsed_seconds: 总耗时
- per_frame_ms: 每帧平均耗时
```

**当前状态**：`experiments/al_rounds_summary.csv` 中已有 `prediction_elapsed_s` 列，Round 3 记录了 7 秒。

### 2.5 Model deviation calculation time

**指标**：从 committee forces/energies 计算 `force_dev_max`、`force_dev_mean`、`energy_dev` 的耗时。

**测量方式**：此计算在 `predict_committee_models.py` 中完成（`np.std` + `np.linalg.norm` + `np.max`），可作为 prediction 的子步骤计时。

**记录内容**：

```text
- n_frames: 候选池帧数
- n_models: 4
- elapsed_ms: 计算耗时（通常远小于 prediction）
```

### 2.6 Top-K selection time

**指标**：`np.argsort(force_dev_max)[::-1][:k]` 的执行耗时。

**测量方式**：通常 < 1ms，可忽略或粗略记录。

### 2.7 Dataset update time

**指标**：`merge_selected_frames.py` 和 `make_remaining_candidate.py` 的执行耗时。

**测量方式**：用 `time` 命令包裹。

**记录内容**：

```text
- merge_elapsed_seconds
- remaining_elapsed_seconds
- n_frames_before / n_frames_after
```

### 2.8 End-to-end active learning round wall-clock time

**指标**：完整一轮主动学习的端到端耗时。

**包含**：

```text
1. committee training (主导耗时)
2. freeze models
3. test on validation set
4. committee prediction
5. model deviation calculation
6. top-K selection
7. dataset update (merge + remaining candidate)
```

**不含**：

```text
- Docker 启动时间
- 人工等待时间
- 数据下载/上传时间
```

---

## 3. Profiling 执行方式

### 3.1 训练阶段

在 `train_round_committee_models.sh` 中已有的时间戳基础上，增加精确计时：

```bash
START_TS=$(date +%s)
# ... 训练逻辑 ...
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "elapsed_seconds=${ELAPSED}" >> "${EXP_DIR_ABS}/train_elapsed.log"
```

**实操命令模板 (single-model training time)：**

```bash
# 记录单个模型的 wall-clock time
MODEL_DIR="experiments/baselines/random_seed0_round001_committee_models/model_000"
START_TS=$(date +%s)
dp train configs/deepmd/random_seed0_round_001_committee/toy_h2_round001_model_000.json
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "model_000,start=$(date -d @${START_TS} -Iseconds),end=$(date -d @${END_TS} -Iseconds),elapsed_s=${ELAPSED}" \
  >> "${MODEL_DIR}/train_elapsed.log"
```

**实操命令模板 (4-model serial training total time)：**

```bash
# 串行训练 4 个模型，记录总耗时
TOTAL_START=$(date +%s)
for i in 0 1 2 3; do
  MODEL_START=$(date +%s)
  dp train configs/deepmd/committee/toy_h2_model_$(printf "%03d" $i).json
  MODEL_END=$(date +%s)
  echo "model_$(printf "%03d" $i),elapsed_s=$((MODEL_END - MODEL_START))"
done
TOTAL_END=$(date +%s)
echo "serial_4_models,total_elapsed_s=$((TOTAL_END - TOTAL_START))"
```

**实操命令模板 (2×V100 model-level parallel training)：**

当前 `train_round_committee_models.sh` 已实现 GPU 分配：
- GPU 0: model_000, model_002
- GPU 1: model_001, model_003

```bash
# 并行训练，记录 wall-clock time
START_TS=$(date +%s)
bash scripts/train/train_round_committee_models.sh \
  001 \
  configs/deepmd/random_seed0_round_001_committee \
  experiments/baselines/random_seed0_round001_committee_models \
  /data/zft/deepmd-al-hpc/data/toy_h2/valid
END_TS=$(date +%s)
echo "parallel_2gpu_4models,elapsed_s=$((END_TS - START_TS))"
```

### 3.2 预测阶段

在调用 `predict_committee_models.py` 时使用 `time`：

```bash
time python scripts/inference/predict_committee_models.py \
  --data ... --models ... --output ... --selected-json ... --top-k 10 \
  2>&1 | tee prediction.log
```

**实操命令模板 (candidate prediction time)：**

```bash
# 记录 committee prediction 的完整耗时
START_TS=$(date +%s)
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
END_TS=$(date +%s)
echo "prediction,round=001,seed=seed0,elapsed_s=$((END_TS - START_TS))"
```

### 3.3 数据更新阶段

```bash
time python scripts/data/merge_selected_frames.py ...
time python scripts/data/make_remaining_candidate.py ...
```

**实操命令模板 (dataset update time)：**

```bash
# merge_selected_frames
START_TS=$(date +%s)
python scripts/data/merge_selected_frames.py \
  --train data/toy_h2/random_seed0_round_001_train \
  --candidate data/toy_h2/random_seed0_round_001_candidate \
  --selection experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json \
  --output data/toy_h2/random_seed0_round_002_train --overwrite
END_TS=$(date +%s)
echo "merge,round=001→002,seed=seed0,elapsed_s=$((END_TS - START_TS))"

# make_remaining_candidate
START_TS=$(date +%s)
python scripts/data/make_remaining_candidate.py \
  --candidate data/toy_h2/random_seed0_round_001_candidate \
  --selection experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json \
  --output data/toy_h2/random_seed0_round_002_candidate --overwrite
END_TS=$(date +%s)
echo "remaining_candidate,round=001→002,seed=seed0,elapsed_s=$((END_TS - START_TS))"
```

### 3.4 GPU 利用率监控

**使用 `nvidia-smi` 周期性记录：**

```bash
# 每隔 5 秒记录一次 GPU 利用率，输出到日志文件
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv -l 5 > experiments/profiling/gpu_monitor_round001_seed0.csv &

# 记录 PID，结束后 kill
NVIDIA_SMI_PID=$!
# ... 运行训练/预测 ...
kill $NVIDIA_SMI_PID
```

**使用 `nvidia-smi dmon` 简洁监控：**

```bash
# 每 2 秒输出一行，包含 GPU 利用率、显存、温度等
nvidia-smi dmon -s pucvmet -d 2 > experiments/profiling/gpu_dmon_round001.log &
DMON_PID=$!
# ... 运行训练/预测 ...
kill $DMON_PID
```

**使用 `nvidia-smi` 单次查询（适合脚本中插入）：**

```bash
# 训练前记录 GPU 状态
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader > experiments/profiling/gpu_before.log

# ... 训练 ...

# 训练后记录 GPU 状态
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader > experiments/profiling/gpu_after.log
```

### 3.5 End-to-end round wall-clock time

**实操命令模板：**

```bash
#!/bin/bash
# 完整一轮 active learning 的端到端计时
ROUND=002
SEED=seed0

LOG_DIR="experiments/profiling"
mkdir -p "${LOG_DIR}"

ROUND_START=$(date +%s)

# 1. Training (主导耗时)
echo "[$(date -Iseconds)] Training start" | tee -a "${LOG_DIR}/round_${ROUND}_${SEED}_timeline.log"
TRAIN_START=$(date +%s)
bash scripts/train/train_round_committee_models.sh \
  ${ROUND} \
  configs/deepmd/random_${SEED}_round_${ROUND}_committee \
  experiments/baselines/random_${SEED}_round${ROUND}_committee_models \
  /data/zft/deepmd-al-hpc/data/toy_h2/valid
TRAIN_END=$(date +%s)
TRAIN_ELAPSED=$((TRAIN_END - TRAIN_START))
echo "[$(date -Iseconds)] Training end, elapsed_s=${TRAIN_ELAPSED}" | tee -a "${LOG_DIR}/round_${ROUND}_${SEED}_timeline.log"

# 2-4. Prediction + deviation + selection
echo "[$(date -Iseconds)] Prediction start" | tee -a "${LOG_DIR}/round_${ROUND}_${SEED}_timeline.log"
PRED_START=$(date +%s)
python scripts/inference/predict_committee_models.py \
  --data data/toy_h2/random_${SEED}_round_${ROUND}_candidate \
  --models \
    experiments/baselines/random_${SEED}_round${ROUND}_committee_models/model_000/frozen_model.pb \
    experiments/baselines/random_${SEED}_round${ROUND}_committee_models/model_001/frozen_model.pb \
    experiments/baselines/random_${SEED}_round${ROUND}_committee_models/model_002/frozen_model.pb \
    experiments/baselines/random_${SEED}_round${ROUND}_committee_models/model_003/frozen_model.pb \
  --output experiments/baselines/random_${SEED}_round${ROUND}_committee_prediction/committee_predictions.npz \
  --selected-json experiments/baselines/random_${SEED}_round${ROUND}_committee_prediction/selected_topk.json \
  --top-k 10
PRED_END=$(date +%s)
PRED_ELAPSED=$((PRED_END - PRED_START))
echo "[$(date -Iseconds)] Prediction end, elapsed_s=${PRED_ELAPSED}" | tee -a "${LOG_DIR}/round_${ROUND}_${SEED}_timeline.log"

# 5. Dataset update (for next round)
echo "[$(date -Iseconds)] Dataset update start" | tee -a "${LOG_DIR}/round_${ROUND}_${SEED}_timeline.log"
UPDATE_START=$(date +%s)
NEXT_ROUND=$((ROUND + 1))
python scripts/data/merge_selected_frames.py \
  --train data/toy_h2/random_${SEED}_round_${ROUND}_train \
  --candidate data/toy_h2/random_${SEED}_round_${ROUND}_candidate \
  --selection experiments/baselines/random_${SEED}_round${ROUND}_committee_prediction/selected_topk.json \
  --output data/toy_h2/random_${SEED}_round_${NEXT_ROUND}_train --overwrite
python scripts/data/make_remaining_candidate.py \
  --candidate data/toy_h2/random_${SEED}_round_${ROUND}_candidate \
  --selection experiments/baselines/random_${SEED}_round${ROUND}_committee_prediction/selected_topk.json \
  --output data/toy_h2/random_${SEED}_round_${NEXT_ROUND}_candidate --overwrite
UPDATE_END=$(date +%s)
UPDATE_ELAPSED=$((UPDATE_END - UPDATE_START))
echo "[$(date -Iseconds)] Dataset update end, elapsed_s=${UPDATE_ELAPSED}" | tee -a "${LOG_DIR}/round_${ROUND}_${SEED}_timeline.log"

ROUND_END=$(date +%s)
ROUND_ELAPSED=$((ROUND_END - ROUND_START))

# 写入 CSV 行
echo "round=${ROUND},seed=${SEED},train_s=${TRAIN_ELAPSED},prediction_s=${PRED_ELAPSED},dataset_update_s=${UPDATE_ELAPSED},end_to_end_s=${ROUND_ELAPSED}" \
  >> "${LOG_DIR}/profiling_v100_rounds.csv"

echo ""
echo "========== Round ${ROUND} (${SEED}) profiling complete =========="
echo "  Training:       ${TRAIN_ELAPSED} s"
echo "  Prediction:     ${PRED_ELAPSED} s"
echo "  Dataset update: ${UPDATE_ELAPSED} s"
echo "  End-to-end:     ${ROUND_ELAPSED} s"
```

---
## 4. Profiling 输出格式

### 4.1 Per-round profiling CSV

建议输出文件：`experiments/profiling/profiling_v100_rounds.csv`

```csv
round_id,branch,seed,train_elapsed_s,freeze_elapsed_s,test_elapsed_s,prediction_elapsed_s,dev_calc_elapsed_s,selection_elapsed_s,dataset_update_elapsed_s,end_to_end_elapsed_s,notes
0,uncertainty,,360,,,,,,,
1,uncertainty,,380,5,2,7,0.01,0.01,2,397,
1,random,seed0,350,5,2,6,0.01,0.01,2,365,
1,random,seed1,,,,,,,,,
...
```

### 4.2 Profiling Markdown summary

建议输出文件：`experiments/profiling/profiling_v100_summary.md`

```markdown
# V100 Profiling Summary

## Per-round end-to-end time

| Round | Branch | Seed | Train / s | Prediction / s | End-to-end / s |
|---|---:|---|---:|---:|---:|
| 0 | uncertainty | — | 360 | 7 | 370 |
| 1 | uncertainty | — | 380 | 7 | 397 |
| 1 | random | seed0 | 350 | 6 | 365 |
| ... | | | | | |

## Breakdown by phase (uncertainty branch, average)

| Phase | Time / s | % of total |
|---|---|---|
| Training (4 models) | 370 | 93% |
| Prediction | 7 | 2% |
| Others (freeze, test, dataset update) | 20 | 5% |
```

---

## 5. 后续 H100 迁移如何复用

V100 profiling 建立的指标体系和 CSV 格式可以直接复用到 H100：

1. **相同指标**：elapsed_seconds、per_frame_ms、end-to-end time
2. **相同 CSV schema**：增加 `gpu_model` 列（V100 / H100）
3. **对比维度**：
   - V100 serial vs H100 serial
   - 2×V100 parallel vs 4×H100 parallel
   - per-frame prediction throughput
   - GPU 利用率（nvidia-smi 记录）

---

## 6. 当前 V100 profiling 已有数据

`experiments/al_rounds_summary.csv` 中：

| Round | train_elapsed_s | prediction_elapsed_s |
|---|---:|---:|
| 0 | — | — |
| 1 | — | — |
| 2 | — | — |
| 3 | 76 | 7 |

说明：目前只有 Round 3 记录了耗时。需要在后续实验中补充完整 profiling 数据。

---

## 7. 如何开始 V100 Profiling

### 第一步：创建 profiling 目录

```bash
mkdir -p experiments/profiling
```

### 第二步：对已完成的一轮补记录耗时

如果已完成的某轮还有 `train.log` 和 `test.log`，可以从中提取耗时数据：

```bash
# 从 train.log 提取训练耗时（DeePMD 在 train.log 末尾会打印 wall time）
for model_dir in experiments/baselines/random_seed0_round001_committee_models/model_*/; do
  echo "=== $(basename $model_dir) ==="
  tail -20 "${model_dir}/train.log" | grep -E "wall time|elapsed|total"
done
```

### 第三步：对后续 Round 启用完整 profiling

以 random seed0 Round 002 为例：

```bash
# 1. 启动 GPU 监控（后台）
nvidia-smi dmon -s pucvmet -d 2 > experiments/profiling/gpu_dmon_round002_seed0.log &
DMON_PID=$!

# 2. 运行完整 round（参考 3.5 节的端到端模板）
bash scripts/profiling/record_round.sh 002 seed0

# 3. 停止 GPU 监控
kill $DMON_PID
```

### 第四步：汇总到 profiling CSV

手动整理或使用脚本，将每轮的耗时数据追加到 `experiments/profiling/profiling_v100_rounds.csv`。

### 第五步：生成 profiling summary

参考 4.2 节的 Markdown 模板，将 CSV 数据整理为可读的 `experiments/profiling/profiling_v100_summary.md`。

### 第六步：作为 H100 scaling baseline

V100 profiling 完成后，记录：
```text
- 2×V100 serial training time per model
- 2×V100 parallel training time (2 models simultaneous)
- Per-frame prediction time
- End-to-end round time
- GPU utilization (average)
```

这些数据将成为后续 `docs/profiling_h100.md` 中 H100 加速比计算的 baseline。
