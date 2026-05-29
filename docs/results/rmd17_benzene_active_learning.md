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

## 待补充实验

- Diversity baseline（3 seeds × 3 rounds）
- Trust_level baseline（3 seeds × 3 rounds）
- Four-strategy comparison
- MD stability（NVE 10K/100K）
- Pipeline profiling
