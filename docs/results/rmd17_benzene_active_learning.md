# rMD17 Benzene 主动学习实验

## 数据集

- 数据集：rMD17 benzene
- 原始数据：`data/raw/rmd17_benzene.npz`
- DeePMD 格式数据：`data/rmd17/benzene/`
- Type map：`C H`

数据划分：

| 划分 | 帧数 |
|---|---:|
| 初始训练集 | 1000 |
| 候选池 | 60000 |
| 验证集 | 5000 |
| 测试集 | 10000 |

> 原始 `.npz`、处理后的 `.npy`、候选池、预测结果和 frozen models 不被 Git 跟踪。存储在训练服务器上。

---

## Uncertainty Branch（Round 000–003）

| Round | 训练帧 | 候选帧 | 选择策略 | 状态 |
|---:|---:|---:|---|---|
| 000 | 1000 | 60000 | 初始 | 已完成 |
| 001 | 2000 | 59000 | uncertainty top-1000 | 已完成 |
| 002 | 3000 | 58000 | uncertainty top-1000 | 已完成 |
| 003 | 4000 | 57000 | uncertainty top-1000 | 已完成 |

- 每轮 4 个 committee models，`DP_INFER_BATCH_SIZE=1800` 避免 V100 OOM
- 与 rMD17 ethanol 使用相同的 active learning 流水线

---

## 已完成实验

- 数据格式转换和划分
- Round 000–003 committee training（4 models × 4 rounds = 16 models）
- Round 000–003 committee prediction + uncertainty top-1000 selection
- Random baseline（seed0/1/2 Round 001–003）
- Independent test evaluation

## 已完成实验（更新于 2026-06-28）

- Diversity baseline（3 seeds × 3 rounds，36 模型）
- Trust_level baseline（3 seeds × 3 rounds，36 模型）
- Four-strategy comparison
- MD stability（NVE 2.5ps，10K）

### Four-Strategy Force RMSE（Round 3，validation set，3-seed mean ± std）

| Strategy | Force RMSE (eV/Å) | Std |
|---|---:|---:|
| uncertainty | 1.825e-01 | N/A |
| diversity | 1.891e-01 | ±2.476e-02 |
| trust_level | 1.966e-01 | ±2.682e-02 |
| random | 2.165e-01 | ±2.548e-02 |

### MD Stability（NVE 2.5ps, 0.5fs, ~10K）

全部 4 个策略的 Round 3 代表模型均稳定，drift < 0.002 eV/atom/ps：

| Model | Drift (eV/atom/ps) | Temp (K) |
|---|---:|---:|
| uncertainty_round3 | 1.808e-03 | 78.0 |
| random_seed0_round3 | 1.775e-03 | 77.7 |
| diversity_seed0_round3 | 1.946e-03 | 77.2 |
| trust_level_seed0_round3 | 1.920e-03 | 77.5 |

### 四策略分析文件

- `experiments/rmd17_benzene_summary/all_strategies_detail.csv`
- `experiments/rmd17_benzene_summary/four_strategy_comparison.csv`
- `experiments/rmd17_benzene_summary/md_stability/md_summary.json`
