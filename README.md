# deepmd-al-hpc

`deepmd-al-hpc` 是一个面向 **机器学习势函数（MLP）committee-based active learning 的 GPU 性能表征与优化** 的研究框架，目标投稿 **IPDPS 2027**（CCF-B）。

**核心问题**：committee AL 训练（DP-GEN 范式）在 GPU 上严重低效 —— 单模型训练 SM 利用率仅 **4–9%**。本项目用 Nsight 定位根因（**launch-bound**），并提出 MPS 多模型共享 + batch 调优：单卡利用率提升 **9.9×**、资源效率 **~4×**。

> 历史：仓库早期是 DeePMD committee offline active learning 的方法学验证（toy H2 + rMD17 四策略，2×V100），现已重新定位为系统性能论文；方法学结果保留为论文 appendix 与背景。

---

## 当前状态（2026-07-19）

### IPDPS 2027 系统论文（主线，8×RTX 5090）

| 模块 | 状态 |
|---|---|
| 强 scaling（N=8 ethanol, G=1/2/4/8）| ✅ 近线性，G=8 效率 85% |
| GPU 利用率根因（Nsight）| ✅ launch-bound（cudaLaunchKernel 占 CPU 84.6%）|
| MPS 多模型共享优化 | ✅ 资源效率 ~4× |
| batch×MPS 组合 | ✅ SM 4.0%→39.4%（9.9×）|
| DP-GEN 对比（Table 2）| ✅ wave 代理 + 源码佐证 |
| 论文 figure（Fig.2/3/5/6）| ✅ 定稿 |
| 统计补强（n=3 mean±std）| ✅ 4× std ≤0.1s |
| 长训练验证（2000 步）| ✅ MPS 239s ≈ wave 235s |
| 能耗 + 红利曲线 sweep | ✅ 每模型能耗 N=8 降到 N=2 的 1/3 |
| 大体系验证 | ⏳ 待补数据集 |
| 真跑 dpgen 完整闭环 | ⏳ future work（需 VASP）|
| 跨架构（V100/H100）| ⏳ future work |
| 论文正文 | ⏳ 未开始 |

### 方法学背景（历史，2×V100）
- toy H2 + rMD17 ethanol/benzene 四策略（uncertainty/random/diversity/trust_level）multi-seed multi-round 完成
- independent test + NVE MD stability 完成

详见 [`docs/current_project_status.md`](docs/current_project_status.md)。

---

## 硬件环境

| 环境 | GPU | 角色 |
|---|---|---|
| RTX 5090 | 8×RTX 5090-32GB (Blackwell) | 系统论文主平台（PyTorch 后端）|
| shared-v100 | 2×Tesla V100-16GB | 历史方法学实验 |

5090 环境配置见 [`docs/setup.md`](docs/setup.md)。**所有 GPU 操作走容器**（`scripts/docker/run_in_5090.sh`）。

---

## 核心结果（系统论文）

| 产出 | 内容 | 位置 |
|---|---|---|
| Fig.2 motivation | baseline SM 仅 4% | `experiments/figures/ccgrid_2027/` |
| Fig.3 强 scaling | N=8, G=8 效率 85% | 同上 |
| Fig.5 launch-bound 根因 | cudaLaunchKernel 84.6% | 同上 |
| Fig.6 优化矩阵 | batch×MPS, SM 9.9× | 同上 |
| Table 2 DP-GEN 对比 | 资源效率 4× | `docs/results/dpgen_comparison.md` |
| Sweep 红利曲线 | SM 随 N/batch 单调升、未饱和 | `experiments/scaling/sweep_summary.json` |

---

## 快速开始

```bash
# 所有 GPU 操作走 5090 容器（PyTorch 后端）
bash scripts/docker/run_in_5090.sh 0 -- <command>

# 例：MPS 多模型共享实验（4 模型共享 1 卡，batch=256）
bash scripts/scaling/run_mps.sh 4 1 100 256

# 强 scaling（N=8, 1/2/4/8 GPU）
bash scripts/scaling/run_strong_scaling.sh 8 ethanol 200
```

HPDC 路线图见 [`docs/ccgrid_2027_roadmap.md`](docs/ccgrid_2027_roadmap.md)，复现总入口见 [`docs/reproduce.md`](docs/reproduce.md)。

---

## 文档导航

**系统论文（主线）**：

| 文档 | 作用 |
|---|---|
| [`docs/ccgrid_2027_roadmap.md`](docs/ccgrid_2027_roadmap.md) | IPDPS 2027 投稿路线图（定位/gap/计划/时间线）|
| [`docs/results/dpgen_comparison.md`](docs/results/dpgen_comparison.md) | DP-GEN 对比（Table 2，含统计+长训练）|
| [`docs/current_project_status.md`](docs/current_project_status.md) | 项目全局状态 |

**方法学背景（历史）**：

| 文档 | 作用 |
|---|---|
| [`docs/results.md`](docs/results.md) | 方法学实验结果（toy H2 + rMD17）|
| [`docs/selection_strategies.md`](docs/selection_strategies.md) | 四类 selection strategy |
| [`docs/paper_evidence.md`](docs/paper_evidence.md) | 方法学证据清单 |
| [`docs/profiling_v100.md`](docs/profiling_v100.md) | V100 profiling |
| [`docs/reproduce.md`](docs/reproduce.md) | 复现总入口 |
| [`docs/setup.md`](docs/setup.md) | 环境配置 |

---

## Claim Boundary（系统论文）

**Can claim:**
- committee AL 训练（DP-GEN 范式）GPU 利用率仅 4–9%（实测，Fig.2）
- 根因是 launch-bound（cudaLaunchKernel 占 CPU 84.6%，Nsight，Fig.5）
- MPS 多模型共享 + batch：单卡 SM 4.0%→39.4%（9.9×），资源效率 ~4×（统计 n=3 + 长训练 2000 步双验证，Fig.6 + Table 2）
- 强 scaling 近线性（N=8, G=8 效率 85%，Fig.3）
- 每模型能耗随 committee 规模 N 降到 1/3（sweep）

**Cannot claim (yet):**
- 大体系（蛋白质/材料）结论（只 ethanol/benzene 小分子）—— 待补数据集
- 完整 DP-GEN AL 闭环对比（无 VASP，用源码佐证的训练调度代理）
- 单模型多卡 DDP 维度（只 model-level parallel）
- 跨架构（V100/H100）验证
- 高温 MD stability（方法学侧，100K+ 解离）

---

## 主要结果链接

**系统论文**：
- `experiments/figures/ccgrid_2027/` — Fig.2/3/5/6（SVG+PNG）
- `experiments/scaling/` — scaling/MPS/wave/long/stats/sweep 原始数据
- `experiments/nsight_prof/` — Nsight trace
- `docs/results/dpgen_comparison.md` — Table 2

**方法学（历史）**：
- `experiments/al_rounds_summary.csv` — toy H2 uncertainty
- `experiments/rmd17_ethanol_summary/` — ethanol 四策略
- `experiments/rmd17_benzene_round*/` — benzene

---

## 版本管理

不提交：`/data/`、`*.pb`、`*.npy`、`*.npz`、`model.ckpt*`、`checkpoint`、`*.log`、`*.nsys-rep`、LAMMPS trajectory。
保留：source code、configs、lightweight summaries（`summary.json` / `selected_topk.json` / `*.csv`）、figures、docs。

详见 [`docs/data_and_git_policy.md`](docs/data_and_git_policy.md)。

<!-- README updated 2026-07-19 (IPDPS 2027 systems paper repositioning). -->
