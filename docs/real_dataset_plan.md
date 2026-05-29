# 真实 DFT/AIMD 数据集迁移计划

---

## 1. 为什么 toy H2 不够

Toy H2（2 atoms, 250 frames）验证了流水线可运行，但不能支撑关于真实材料体系的结论：

- 2 原子 H2 除 H-H bond length 外无结构多样性
- Committee model variance 因数据集过小而被放大
- Valid set 同时充当 candidate pool——无 independent evaluation
- 结果不能推广到多元素或周期性体系

真实 DFT/AIMD 数据集是论文投稿的最低要求。

---

## 2. 数据集要求

区分两个验证阶段：

**分子基准阶段**（当前：rMD17 ethanol, benzene）：
- 小有机分子，无 PBC
- Energy + force labels（无 virial）
- 足以验证 AL 工作流和策略对比

**周期性材料阶段**（未来）：
- 具有周期性边界条件的块体金属、氧化物或液体
- Energy, force, 和 virial labels
- 材料科学论文投稿所需

---

## 3. 推荐数据划分

| 划分 | 用途 | 典型大小 |
|---|---|---|
| `initial_train` | 训练 Round 0 committee models | 100–200 frames |
| `candidate_pool` | 未标注池，用于 active learning selection | 500–2000 frames |
| `validation` | 模型选择和超参数调优 | 100–200 frames |
| `independent_test` | 仅用于最终评估，永不参与 AL | 100–200 frames |

### 关键规则

1. `independent_test` 绝不能用于 selection
2. `candidate_pool` 模拟昂贵的 DFT labeling——每个被选中的帧被"标注"
3. `validation` 与 `independent_test` 分离
4. Toy H2 当前违反规则 1（valid = candidate_pool）。真实数据集流水线中必须修正。

---

## 4. 如何转换已有 DeePMD 数据

如果数据已是 DeePMD 格式（npy arrays with `set.*/` structure）：

```bash
python scripts/data/prepare_real_dataset_template.py \
  --source data/real_datasets/my_system/all_data \
  --output data/real_datasets/my_system \
  --initial-train 100 --candidate-pool 1000 \
  --validation 100 --test 100 \
  --seed 0
```

这会创建目录结构和 metadata，然后打印手动划分实际数据所需的步骤。

---

## 5. 如何复用四策略流水线

现有脚本适用于任何 DeePMD 格式数据：

1. 将 `data/toy_h2/` 路径替换为 `data/real_datasets/<name>/` 路径
2. 使用 `initial_train` 作为初始训练集
3. 使用 `candidate_pool` 作为初始候选池
4. 使用 `validation` 进行 `dp test` 评估
5. 仅在最终评估中使用 `independent_test`（永不输入 AL 轮次）

`scripts/experiments/run_toy_h2_strategy_comparison.sh` 模板可通过修改数据路径来适配。

---

## 6. Independent test 评估

所有 AL 轮次完成后：
1. 在完整训练集上训练最终模型
2. 在 `independent_test` 上计算 Energy RMSE, Force RMSE
3. 计算 model deviation 统计量
4. 与 baseline（无 AL, random selection）对比

`independent_test` 必须仅使用一次——在最后。

---

## 7. MD stability 验证（未来）

MD stability 是超越 RMSE 的重要物理验证：

1. 使用最终 frozen model 运行 LAMMPS MD
2. 检查能量守恒（NVE ensemble）
3. 检查 RDF / 结构属性与 DFT 参考一致
4. 检查无非物理键断裂

这推迟到真实数据集流水线验证之后。

---

## 8. 不应提交到 Git 的文件

```
data/real_datasets/**/*.npy
data/real_datasets/**/*.npz
data/real_datasets/**/coord.npy
data/real_datasets/**/force.npy
data/real_datasets/**/energy.npy
data/real_datasets/**/box.npy
```

`metadata.json` 和相关配置文件适合提交到 Git。

---

## 9. 下一步

1. 获取小型真实 DFT/AIMD 数据集（如来自公共数据库或合作者）
2. 如需要，转换为 DeePMD npy 格式
3. 运行 `prepare_real_dataset_template.py` 创建划分
4. 将策略对比运行器适配到真实数据路径
5. 在真实数据上运行四策略对比
6. 进行 independent test 评估
7. （未来）运行 MD stability 验证

---

## 10. 当前状态（2026-05-28）

rMD17 ethanol 流水线在当前 offline-AL 阶段已完成：uncertainty/random/diversity/trust_level、independent test、短时 NVE sanity check 和 V100 profiling 均已完成。

**数据**（C₂H₅OH, 9 atoms, 27 Cartesian force components）：
| 划分 | 帧数 | 路径 |
|---|---|---|
| 初始训练集 | 1000 | `data/rmd17/ethanol/train` |
| 验证集 | 5000 | `data/rmd17/ethanol/valid` |
| 测试集 | 10000 | `data/rmd17/ethanol/test` |
| 初始候选池 | 60000 | `data/rmd17/ethanol/candidate` |

**Active learning 轮次（uncertainty branch）**：
| Round | 训练帧 | 候选帧 | Training | Prediction |
|---|---:|---:|---|---|
| 0 | 1000 | 60000 | 已完成 | 已完成 |
| 1 | 2000 | 59000 | 已完成 | 已完成 |
| 2 | 3000 | 58000 | 已完成 | 已完成 |
| 3 | 4000 | 57000 | 已完成 | 已完成 |

每轮从候选池中选择 1000 个 uncertainty top-K 帧。

**已完成：**
- 数据格式转换脚本：`scripts/data/convert_rmd17_to_deepmd.py`
- 数据划分脚本：`scripts/data/split_rmd17_to_deepmd.py`
- Round 0–3 committee 配置（4 models × 4 rounds = 16 configs）
- Round 0–3 committee 模型训练（16 frozen models）
- Round 0–3 committee prediction + uncertainty top-K selection
- Round 0–3 summary CSV + MD + learning curve 图
- 统一 profiling CSV（52 models, all pipeline stages）

**端到端流水线 Profiling（2×V100）**：
| Round | 训练 (s) | 预测 (s) | 其他 (s) | 总计 (s) |
|---:|---:|---:|---:|---:|
| 0 | 87 | 185 | 21 | 293 |
| 1 | 104 | 182 | 21 | 307 |
| 2 | 107 | 179 | 21 | 307 |
| 3 | 106 | 176 | 21 | 303 |

- 训练：4 models / 2 GPUs parallel，mean 50.4s/model（uncertainty），56.7s/model（random）
- 预测：57k–60k frames × 4 models，~3 min/round
- 每轮总计：~5 min（uncertainty），~10 min（random, 3 seeds）
- 完整 uncertainty branch（Round 0–3）：~20 min
- 完整 random baseline（3 seeds × 3 rounds）：~29 min
- 数据：`experiments/rmd17_ethanol_summary/profiling_unified.csv`，`profiling_all_models.csv`

**关键结果（uncertainty branch）**：

| Round | 训练帧 | 候选帧 | Force RMSE mean | force_dev_max（selected top-1000） |
|---:|---:|---:|---:|---:|
| 0 | 1000 | 60000 | 3.739e-01 | 6.129e-01 |
| 1 | 2000 | 59000 | 3.715e-01 | 4.570e-01 |
| 2 | 3000 | 58000 | 3.644e-01 | 3.906e-01 |
| 3 | 4000 | 57000 | 3.537e-01 | 4.569e-01 |

- Force RMSE 单调下降（0.374 → 0.354 eV/Å），与 toy H2 不同
- top-1000 force_dev_max_mean 在 Round 0→2 下降（0.613 → 0.391），Round 3 回弹至 0.457
- Energy RMSE 在所有轮次稳定在 ~0.12–0.13 eV
- 摘要文件：`experiments/rmd17_ethanol_summary/`

**Independent Test 结果（10000 帧，从未参与 AL）**：
| Round | Force RMSE（test） | Force RMSE（valid） |
|---:|---:|---:|
| 0 | 0.343914 | 0.373912 |
| 1 | 0.343304 | 0.371493 |
| 2 | 0.335249 | 0.364440 |
| 3 | 0.326594 | 0.353702 |

- Test Force RMSE 单调下降（0.344→0.327 eV/Å），确认真实改善
- Test RMSE 一致比 validation 低 ~0.028 eV/Å

**Random Baseline（validation set, cross-seed mean ± std；std 为跨 3 个 seed 均值的标准差，非 12 个模型）**：
| Round | Uncertainty F_RMSE | Random F_RMSE（mean±std） |
|---:|---:|---:|
| 1 | 0.3715 | 0.3734 ± 0.010 |
| 2 | 0.3644 | 0.3990 ± 0.031 |
| 3 | 0.3537 | 0.6067 ± 0.385 |

- Uncertainty Force RMSE 单调下降，random 在 Round 3 恶化
- Random Round 3 mean（0.607）是 uncertainty（0.354）的 1.71 倍，但 random 跨 seed 方差大

**MD Stability（NVE, 10K）**：
- 所有模型在 10K 下稳定，drift ~0.035 eV/ps
- 100K+ 下所有模型立即解离（Force RMSE ~0.35 eV/Å 不足以支撑 MD）
- Uncertainty Round 3 具有最低 drift（-0.0338 eV/ps）

**四策略对比（Round 3, 3-seed mean ± std）**：
| 策略 | Force RMSE | Std |
|---|---:|---:|
| uncertainty | 0.3537 | 0.0247 |
| diversity | 0.3555 | 0.0143 |
| trust_level | 0.3616 | 0.0166 |
| random | 0.6067 | 0.6826 |

三种 active strategy 在 1σ 内，mean 均低于 random（0.607 ± 0.683 eV/Å）。但 random 方差大——不能声称严格统计显著性。与 toy H2 一致。

**Top-K Labeling Budget Ablation（Round 3, 3-seed mean ± std）**：
| K | Force RMSE | Std | vs K=1000 |
|---:|---:|---:|---:|
| 250 | 0.3408 | 0.0141 | 3.6% better |
| 500 | 0.4146 | 0.1790 | one outlier seed |
| 1000 | 0.3537 | 0.0247 | baseline |
| 2000 | 0.3315 | 0.0176 | 6.3% better |

- K=250（最严格选择）和 K=2000（最多数据）均优于 K=1000
- 更大的 K 受益于更多训练数据；更小的 K 受益于更高选择精度

**Committee Size Ablation（Round 1, 3-seed mean ± std）**：
| Committee | Force RMSE | Std | 训练时间/round |
|---:|---:|---:|---:|
| 2 models | 0.3436 | 0.0155 | ~55s（1 batch） |
| 4 models | 0.3715 | 0.0146 | ~110s（2 batches） |
| 8 models | 0.3392 | 0.0206 | ~220s（4 batches） |

- 8-model 最佳 Force RMSE 但训练成本是 4-model 的 2 倍
- 2-model 与 4-model 竞争力相当，成本减半
- 收益递减：4→8 models 仅 8.7% RMSE 改善，成本翻倍

---

## 11. rMD17 Benzene 结果（2026-05-27）

rMD17 benzene（C₆H₆, 12 atoms）是第二个被验证的真实分子体系。

**数据**：
| 划分 | 帧数 | 路径 |
|---|---|---|
| 初始训练集 | 1000 | `data/rmd17/benzene/train` |
| 验证集 | 5000 | `data/rmd17/benzene/valid` |
| 测试集 | 10000 | `data/rmd17/benzene/test` |
| 初始候选池 | 60000 | `data/rmd17/benzene/candidate` |

**Active learning 轮次（uncertainty branch）**：
| Round | 训练帧 | 候选帧 | 选择策略 | 状态 |
|---:|---:|---:|---|---|
| 000 | 1000 | 60000 | 初始 | 已完成 |
| 001 | 2000 | 59000 | uncertainty top-1000 | 已完成 |
| 002 | 3000 | 58000 | uncertainty top-1000 | 已完成 |
| 003 | 4000 | 57000 | uncertainty top-1000 | 已完成 |

- 每轮 4 个 committee models，`DP_INFER_BATCH_SIZE=1800` 避免 V100 OOM
- 与 rMD17 ethanol 使用相同的流水线和脚本

**已完成：**
- 数据格式转换和划分
- Round 000–003 committee training（4 models × 4 rounds = 16 models）
- Round 000–003 committee prediction + uncertainty top-1000 selection
- Random baseline（seed0/1/2 Round 001–003）
- Independent test evaluation

**待补充：**
- Diversity baseline（3 seeds × 3 rounds）
- Trust_level baseline（3 seeds × 3 rounds）
- MD stability（NVE 10K/100K）
- Four-strategy comparison
- Pipeline profiling

**待补充（通用）：**
- rMD17 之外的更多分子/材料体系
