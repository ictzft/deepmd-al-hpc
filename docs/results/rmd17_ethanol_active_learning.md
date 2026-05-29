# rMD17 Ethanol 主动学习实验

## 数据集

- 数据集：rMD17 ethanol
- 原始数据：`data/raw/rmd17_ethanol.npz`
- DeePMD 格式数据：`data/rmd17/ethanol/`
- Type map：`C H O`

数据划分：

| 划分 | 帧数 |
|---|---:|
| 初始训练集 | 1000 |
| 候选池 | 60000 |
| 验证集 | 5000 |
| 测试集 | 10000 |

> 原始 `.npz`、处理后的 `.npy`、候选池、预测结果和 frozen models 不被 Git 跟踪。存储在训练服务器上。

---

## Uncertainty Branch（Round 0–3）

| Round | 训练帧 | 候选帧 | Valid F_RMSE | Test F_RMSE | force_dev_max (top-1000) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1000 | 60000 | 0.3739 | 0.3439 | 0.6129 |
| 1 | 2000 | 59000 | 0.3715 | 0.3433 | 0.4570 |
| 2 | 3000 | 58000 | 0.3644 | 0.3352 | 0.3906 |
| 3 | 4000 | 57000 | 0.3537 | 0.3266 | 0.4569 |

- Validation 和 independent test（10000 帧，从未参与 AL）Force RMSE 均单调下降
- Independent test Force RMSE 稳定比 valid 低 ~0.028 eV/Å

---

## Random Baseline 对比（validation set, cross-seed mean ± std）

| Round | Uncertainty F_RMSE | Random F_RMSE |
|---:|---:|---:|
| 1 | 0.3715 | 0.3734 ± 0.010 |
| 2 | 0.3644 | 0.3990 ± 0.031 |
| 3 | 0.3537 | 0.6067 ± 0.385 |

- Uncertainty 持续改善，random 在 Round 3 出现明显恶化

---

## 四策略对比（Round 3, validation set, 3-seed mean ± std）

| 策略 | Force RMSE | Std |
|---|---:|---:|
| uncertainty | 0.3537 | 0.0247 |
| diversity | 0.3555 | 0.0143 |
| trust_level | 0.3616 | 0.0166 |
| random | 0.6067 | 0.6826 |

- 三种 active strategy 差异在 1σ 内，mean 均低于 random
- 与 toy H2 结论一致：未出现单一策略显著优于其他

---

## MD Stability

| 条件 | 结果 |
|---|---|
| NVE 10K, 0.25 fs, 2.5 ps | 所有模型稳定，drift ~0.035 eV/ps |
| NVE 100K+, 0.25 fs | 所有模型立即解离（< 0.005 ps） |

当前 Force RMSE ~0.35 eV/Å 足以维持近平衡态稳定，但不足以支撑高温 MD。

---

## 详细结果

完整数据见 `experiments/rmd17_ethanol_summary/`。
