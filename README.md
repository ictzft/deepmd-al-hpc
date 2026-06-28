# deepmd-al-hpc

`deepmd-al-hpc` 是一个面向 **DeePMD committee-based offline active learning** 的可复现实验框架，在 toy H2、rMD17 ethanol 和 rMD17 benzene 上验证了主动学习闭环，并在 2×V100 上完成 profiling baseline。

---

## 当前状态（2026-06-28）

| 系统 | 状态 |
|---|---|
| toy H2 | 四策略 multi-seed multi-round 完成；V100 profiling 完成 |
| rMD17 ethanol | 四策略 + independent test + 10K NVE stability 完成 |
| rMD17 benzene | 四策略 + independent test + 2.5ps NVE stability 完成 |
| H100 scaling | 未开始 |

当前仍属于 **offline active learning 原型**：使用已有 rMD17 标注数据模拟 DFT labeling，不等同于已接入实时 DFT/AIMD 标注闭环。

详细状态见 [`docs/current_project_status.md`](docs/current_project_status.md)。

---

## 快速开始

```bash
cd /data/zft
bash scripts/docker/enter_deepmd_container.sh
```

完整复现流程见 [`docs/reproduce.md`](docs/reproduce.md)。

---

## 文档导航

| 文档 | 作用 |
|---|---|
| [`docs/current_project_status.md`](docs/current_project_status.md) | 项目全局状态、已完成/待完成清单 |
| [`docs/results.md`](docs/results.md) | 实验结果汇总（toy H2 + rMD17 ethanol + rMD17 benzene） |
| [`docs/paper_evidence.md`](docs/paper_evidence.md) | 论文证据清单、可支持/不可支持的结论 |
| [`docs/claim_boundary_2026_05_28.md`](docs/claim_boundary_2026_05_28.md) | claim boundary 与论文安全表述 |
| [`docs/reproduce.md`](docs/reproduce.md) | 复现总入口、推荐顺序与文档导航 |
| [`docs/setup.md`](docs/setup.md) | 环境配置、Docker、DeepMD-kit 基础检查 |
| [`docs/toy_h2_pipeline.md`](docs/toy_h2_pipeline.md) | toy H2 数据生成、单模型训练和基础流程 |
| [`docs/uncertainty_rounds.md`](docs/uncertainty_rounds.md) | uncertainty sampling Round 0–3 多轮闭环 |
| [`docs/random_baseline.md`](docs/random_baseline.md) | random sampling baseline、multi-seed retraining |
| [`docs/selection_strategies.md`](docs/selection_strategies.md) | 四类 selection strategy 说明与对比 |
| [`docs/diversity_and_trust_level_plan.md`](docs/diversity_and_trust_level_plan.md) | diversity / trust-level 实验计划与结果 |
| [`docs/real_dataset_plan.md`](docs/real_dataset_plan.md) | 真实数据集迁移计划与 rMD17 结果 |
| [`docs/profiling_v100.md`](docs/profiling_v100.md) | V100 profiling 方案与实测数据 |
| [`docs/profiling_h100.md`](docs/profiling_h100.md) | H100 迁移计划（尚未执行） |
| [`docs/data_and_git_policy.md`](docs/data_and_git_policy.md) | 数据与 Git 管理规范 |
| [`docs/code_check.md`](docs/code_check.md) | 提交前代码检查 |

推荐阅读顺序：

```text
docs/setup.md → docs/toy_h2_pipeline.md → docs/uncertainty_rounds.md → docs/random_baseline.md → docs/results.md → docs/reproduce.md
```

---

## Claim Boundary

**Can claim:**
- Reproducible offline active learning pipeline on toy H2, rMD17 ethanol, and rMD17 benzene (2×V100)
- Four-strategy multi-seed multi-round comparison completed on toy H2 and rMD17 ethanol
- On rMD17 ethanol, uncertainty-based and related active selection strategies show more stable improvement trends than random sampling (Round 3: 0.354–0.362 vs random 0.607 eV/Å); however, random cross-seed variance is large (std=0.683), so strict statistical significance cannot be claimed
- Uncertainty branch shows monotonically decreasing Force RMSE on both validation and independent test for rMD17 ethanol
- 2×V100 model-level parallel training achieves ~1.97× speedup

**Cannot claim (yet):**
- Results generalize broadly (only 2 rMD17 systems tested; more systems needed)
- One active strategy consistently outperforms others (differences within 1σ on both toy H2 and ethanol)
- High-temperature MD stability (100K+ dissociation on ethanol)
- H100 multi-GPU scaling results
- Online DFT/AIMD labeling

详细 claim boundary 见 [`docs/claim_boundary_2026_05_28.md`](docs/claim_boundary_2026_05_28.md) 和 [`docs/paper_evidence.md`](docs/paper_evidence.md)。

---

## 主要结果链接

**toy H2:**
- `experiments/al_rounds_summary.csv` — uncertainty Round 0–3
- `experiments/baselines/aligned_comparison.csv` — 四策略统一口径对比（authoritative）
- `experiments/figures/` — learning curve 图

**rMD17 ethanol:**
- `experiments/rmd17_ethanol_summary/al_rounds_summary.csv` — uncertainty branch
- `experiments/rmd17_ethanol_summary/independent_test_all_summary.csv` — independent test
- `experiments/rmd17_ethanol_summary/four_strategy_comparison.csv` — 四策略对比
- `experiments/rmd17_ethanol_summary/md_stability/md_summary.json` — MD stability

**rMD17 benzene:**
- `experiments/rmd17_benzene_round{000–003}_committee_prediction/selected_topk.json` — uncertainty branch
- `docs/results/rmd17_benzene_active_learning.md` — benzene 实验说明

---

## 版本管理

以下内容不提交到 GitHub：`/data/`、`*.pb`、`*.npy`、`*.npz`、`model.ckpt*`、`checkpoint`、`*.log`、LAMMPS trajectory 文件。

GitHub 中保留：source code、configuration files、lightweight experiment summaries（`selected_topk.json`、`summary.csv/md`）、learning curve figures、documentation。

详见 [`docs/data_and_git_policy.md`](docs/data_and_git_policy.md)。

<!-- README updated on 2026-06-28. -->
