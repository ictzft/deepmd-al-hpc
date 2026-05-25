# V100 Profiling 方案

本文档定义在 2×Tesla V100 GPU 平台上对 `deepmd-al-hpc` 主动学习闭环进行系统性能 profiling 的方案。

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

### 3.2 预测阶段

在调用 `predict_committee_models.py` 时使用 `time`：

```bash
time python scripts/inference/predict_committee_models.py \
  --data ... --models ... --output ... --selected-json ... --top-k 10 \
  2>&1 | tee prediction.log
```

或在 Python 脚本内部增加 `import time; t0 = time.time(); ...; print(f"elapsed_seconds={time.time()-t0:.0f}")`。

### 3.3 数据更新阶段

```bash
time python scripts/data/merge_selected_frames.py ...
time python scripts/data/make_remaining_candidate.py ...
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

## 7. 建议的 Profiling 执行顺序

1. **先跑一轮 uncertainty branch**（例如 Round 0 重新训练），完整记录所有阶段耗时，验证 profiling 流程。
2. **对 random seed0 Round 001** 做 profiling，因为该轮已经完成，只需补记录（如果还有 test.log 和 train.log 可以从中提取）。
3. **后续所有 round** 自动启用 profiling，在训练和预测脚本中内置计时逻辑。
4. **汇总** 到 `experiments/profiling/profiling_v100_rounds.csv` 和 `.md`。
