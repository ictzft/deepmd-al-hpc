# 数据与 Git 管理原则

本文档用于记录 `deepmd-al-hpc` 项目的数据目录约定、Git 跟踪规则、大文件忽略规则和提交建议。

当前项目中既包含代码、配置、轻量实验摘要，也包含本地数据、DeePMD 模型文件和中间预测数组。为了保持 GitHub 仓库轻量、可读、可复现，需要明确区分：

```text
应该提交到 GitHub 的内容
不应该提交到 GitHub 的内容
只保留在服务器本地的数据和模型文件
```

---

## 1. 仓库根目录

项目在服务器上的默认目录为：

```bash
cd /data/zft
```

建议每次开始修改前先检查当前 Git 状态：

```bash
git status -sb
git log --oneline -5
```

其中：

```text
git status -sb
  查看当前分支和工作区修改状态

git log --oneline -5
  查看最近 5 次提交
```

---

## 2. 目录约定

当前仓库建议采用如下目录约定：

```text
deepmd-al-hpc/
├── README.md
├── configs/                     # 配置文件，提交到 GitHub
├── docs/                        # 文档，提交到 GitHub
├── experiments/                 # 轻量实验摘要，部分提交到 GitHub
├── scripts/                     # 运行脚本，提交到 GitHub
│   ├── active_learning/
│   ├── analysis/
│   ├── config/
│   ├── data/                    # 数据处理脚本，提交到 GitHub
│   ├── docker/
│   ├── eval/
│   ├── inference/
│   └── train/
├── src/                         # 框架源码，提交到 GitHub
└── data/                        # 本地数据目录，不提交到 GitHub
```

需要特别注意：

```text
/data/         表示仓库根目录下的数据目录，不提交到 GitHub
scripts/data/  表示数据处理脚本目录，需要提交到 GitHub
```

---

## 3. `/data/` 与 `scripts/data/` 的区别

这两个目录容易混淆，必须明确区分。

### 3.1 `/data/`

仓库根目录下的：

```text
/data/
```

用于保存本地数据，例如：

```text
data/toy_h2/train
data/toy_h2/valid
data/toy_h2/round_001_train
data/toy_h2/round_001_candidate
data/toy_h2/random_seed0_round_001_train
data/toy_h2/random_seed0_round_001_candidate
```

这些内容通常包含：

```text
coord.npy
box.npy
energy.npy
force.npy
type.raw
type_map.raw
```

这些文件属于本地数据或中间数据，默认不提交 GitHub。

---

### 3.2 `scripts/data/`

脚本目录：

```text
scripts/data/
```

用于保存数据处理脚本，例如：

```text
scripts/data/make_toy_h2_deepmd.py
scripts/data/merge_selected_frames.py
scripts/data/make_remaining_candidate.py
```

这些是可复现实验所需的代码，必须提交到 GitHub。

---

## 4. `.gitignore` 关键规则

为了只忽略仓库根目录下的数据目录，而不误伤 `scripts/data/`，`.gitignore` 中应该写：

```gitignore
/data/
```

而不应该写：

```gitignore
data/
```

原因是：

```text
/data/:
  只忽略仓库根目录下的 data 目录

data/:
  可能误伤所有名为 data 的目录，包括 scripts/data/
```

如果误写成：

```gitignore
data/
```

可能导致：

```text
scripts/data/make_toy_h2_deepmd.py
scripts/data/merge_selected_frames.py
scripts/data/make_remaining_candidate.py
```

无法被 Git 正常跟踪。

---

## 5. 检查 `.gitignore`

查看 `.gitignore` 中与 data 相关的规则：

```bash
grep -n "data" .gitignore
```

期望看到：

```text
/data/
```

不希望看到：

```text
data/
```

如果发现错误规则，可以修改 `.gitignore`，然后重新检查：

```bash
grep -n "data" .gitignore
```

---

## 6. 应该提交到 GitHub 的内容

建议提交以下内容：

```text
README.md
docs/*.md
configs/**/*.json
scripts/**/*.py
scripts/**/*.sh
src/**/*.py
experiments/**/metrics_summary.md
experiments/**/selected_topk.json
experiments/**/selected_random_seed*.json
experiments/**/selected_uncertainty.json
experiments/**/round*_summary.md
experiments/**/*.csv
experiments/**/*.md
experiments/figures/*.svg
```

这些文件的特点是：

```text
体积小
可读性强
能够说明实验流程和结果
便于别人复现实验
不包含大型模型或中间数组
```

---

## 7. 不应该提交到 GitHub 的内容

以下内容默认不提交 GitHub：

```text
/data/
datasets/
raw_data/
processed_data/
*.npy
*.npz
*.pb
model.ckpt*
checkpoint
*.log
lcurve.out
out.json
input_v2_compat.json
LAMMPS / MD trajectory files
Python cache files
```

具体包括：

```text
coord.npy
box.npy
energy.npy
force.npy
committee_predictions.npz
frozen_model.pb
model.ckpt*
checkpoint
train.log
test.log
lcurve.out
out.json
input_v2_compat.json
__pycache__/
*.pyc
```

这些文件通常具有以下特点：

```text
体积较大
可由脚本重新生成
不适合放入 GitHub
可能导致仓库膨胀
```

---

## 8. 当前项目中常见的大文件位置

### 8.1 DeePMD 模型文件

例如：

```text
experiments/exp_004_committee_models/model_000/frozen_model.pb
experiments/exp_007_round001_committee_models/model_000/frozen_model.pb
experiments/baselines/random_seed0_round001_committee_models/model_000/frozen_model.pb
```

这些文件不提交 GitHub。

---

### 8.2 Committee prediction 中间数组

例如：

```text
experiments/exp_005_committee_prediction/committee_predictions.npz
experiments/exp_008_round001_committee_prediction/committee_predictions.npz
experiments/exp_010_round002_committee_prediction/committee_predictions.npz
experiments/exp_012_round003_committee_prediction/committee_predictions.npz
experiments/baselines/random_seed0_round001_committee_prediction/committee_predictions.npz
```

这些 `.npz` 文件不提交 GitHub。

---

### 8.3 训练日志与 checkpoint

例如：

```text
checkpoint
model.ckpt*
train.log
test.log
lcurve.out
out.json
input_v2_compat.json
```

这些文件不提交 GitHub。

---

## 9. 可以提交的轻量实验结果

虽然大文件不提交，但以下轻量文件可以提交：

```text
metrics_summary.md
selected_topk.json
selected_random_seed*.json
selected_uncertainty.json
round_001_selection.json
round003_summary.md
selection_baseline_runs.csv
selection_baseline_summary.csv
selection_baseline_summary.md
random_seed0_round001_metrics_summary.csv
random_seed0_round001_metrics_summary.md
random_seed0_round001_prediction_summary.csv
random_seed0_round001_prediction_summary.md
al_rounds_summary.csv
al_rounds_summary.md
al_model_level_summary.csv
small svg figures
```

这些文件用于说明：

```text
实验是否完成
选中了哪些 frames
每轮 RMSE / model deviation 结果
baseline 对比结果
learning curve 趋势
```

---

## 10. 检查脚本是否被 Git 跟踪

确认 `scripts/data/make_toy_h2_deepmd.py` 已被 Git 跟踪：

```bash
git ls-files scripts/data/make_toy_h2_deepmd.py
```

如果输出：

```text
scripts/data/make_toy_h2_deepmd.py
```

说明该脚本已被跟踪。

也可以检查远端 `origin/main` 中是否已有该脚本：

```bash
git ls-tree -r origin/main scripts/data | grep make_toy_h2_deepmd
```

预期输出类似：

```text
100644 blob ... scripts/data/make_toy_h2_deepmd.py
```

---

## 11. 检查 random baseline 轻量结果是否被跟踪

查看 random baseline 相关文件：

```bash
git ls-files | grep -E "random_seed0|selection_baseline|selected_random|selected_uncertainty"
```

期望看到类似：

```text
experiments/baselines/selection_baseline_runs.csv
experiments/baselines/selection_baseline_summary.csv
experiments/baselines/selection_baseline_summary.md
experiments/baselines/random_seed0_round001_metrics_summary.csv
experiments/baselines/random_seed0_round001_metrics_summary.md
experiments/baselines/random_seed0_round001_prediction_summary.csv
experiments/baselines/random_seed0_round001_prediction_summary.md
experiments/exp_005_committee_prediction/selected_random_seed0.json
experiments/exp_005_committee_prediction/selected_random_seed1.json
experiments/exp_005_committee_prediction/selected_random_seed2.json
experiments/exp_005_committee_prediction/selected_uncertainty.json
```

---

## 12. 检查是否误跟踪大文件

每次提交前建议运行：

```bash
git ls-files | grep -E "\.pb$|\.npz$|\.npy$|checkpoint|model.ckpt|train.log|test.log|lcurve.out|out.json"
```

如果没有输出，说明大文件没有被 Git 跟踪。

如果有输出，需要检查是否误提交了大文件。

---

## 13. 如果大文件已经被 Git 跟踪怎么办

如果发现某些大文件已经被 Git 跟踪，但你希望保留本地文件、只从 Git 中移除，可以使用：

```bash
git rm --cached <file>
```

例如：

```bash
git rm --cached experiments/exp_005_committee_prediction/committee_predictions.npz
```

如果是整个目录：

```bash
git rm -r --cached experiments/exp_004_committee_models/model_000
```

然后提交 `.gitignore` 和移除记录：

```bash
git add .gitignore
git commit -m "Remove generated large files from tracking"
```

注意：

```text
git rm --cached:
  只从 Git 索引中移除，不删除本地文件

rm:
  会删除本地文件
```

不要误用 `rm` 删除本地实验结果。

---

## 14. 提交前推荐检查顺序

每次提交前建议按以下顺序检查。

### 14.1 查看工作区状态

```bash
git status --short
```

常见状态含义：

```text
M   已修改并被 Git 跟踪的文件
??  新建但尚未被 Git 跟踪的文件
D   被删除的文件
```

---

### 14.2 查看具体修改

```bash
git diff
```

如果只想看某个文件：

```bash
git diff README.md
git diff docs/reproduce.md
```

---

### 14.3 查看 staged 修改

如果已经 `git add`，可以查看暂存区：

```bash
git diff --cached
```

---

### 14.4 检查大文件误跟踪

```bash
git ls-files | grep -E "\.pb$|\.npz$|\.npy$|checkpoint|model.ckpt|train.log|test.log|lcurve.out|out.json"
```

无输出为正常。

---

### 14.5 检查最近提交

```bash
git log --oneline -5
```

---

## 15. 文档拆分提交建议

如果本次主要是文档拆分，例如新增：

```text
docs/setup.md
docs/toy_h2_pipeline.md
docs/uncertainty_rounds.md
docs/random_baseline.md
docs/results.md
docs/data_and_git_policy.md
docs/code_check.md
docs/profiling_h100.md
```

建议先检查：

```bash
git status --short
```

然后添加文档：

```bash
git add docs/
```

查看暂存区：

```bash
git diff --cached --stat
```

提交：

```bash
git commit -m "Split reproduce guide into modular documentation"
```

推送：

```bash
git push origin main
```

---

## 16. README 与 docs 同时修改时的提交建议

如果同时修改了：

```text
README.md
docs/reproduce.md
docs/*.md
```

可以一起提交：

```bash
git add README.md docs/
git commit -m "Refine README and modularize reproduction docs"
git push origin main
```

如果修改内容较多，也可以拆成两个提交：

```bash
git add README.md
git commit -m "Simplify README overview"

git add docs/
git commit -m "Split reproduce guide into modular docs"

git push origin main
```

推荐原则：

```text
如果 README 和 docs 是同一轮文档重构，可以一起提交；
如果 README 是项目首页优化，docs 是复现文档拆分，也可以拆成两个提交。
```

---

## 17. 查看当前未跟踪文件

当前如果看到：

```text
?? docs/code_check.md
?? docs/data_and_git_policy.md
?? docs/profiling_h100.md
?? docs/random_baseline.md
?? docs/results.md
?? docs/setup.md
?? docs/toy_h2_pipeline.md
?? docs/uncertainty_rounds.md
```

说明这些文件已经创建，但还没有执行：

```bash
git add docs/
```

在确认内容无误后，可以添加：

```bash
git add docs/
```

---

## 18. 查看某个文件是否已经被跟踪

例如查看 `docs/setup.md` 是否已经被 Git 跟踪：

```bash
git ls-files docs/setup.md
```

如果有输出：

```text
docs/setup.md
```

说明已经被跟踪。

如果没有输出，说明还没有 `git add`。

---

## 19. 检查文档文件列表

查看当前 `docs/` 下的文件：

```bash
find docs -maxdepth 1 -type f | sort
```

期望包括：

```text
docs/code_check.md
docs/data_and_git_policy.md
docs/profiling_h100.md
docs/random_baseline.md
docs/reproduce.md
docs/reproduce_legacy.md
docs/results.md
docs/setup.md
docs/toy_h2_pipeline.md
docs/uncertainty_rounds.md
docs/week2_single_model_baseline.md
```

说明：

```text
docs/reproduce_legacy.md:
  旧版大型 reproduce 文档备份

docs/reproduce.md:
  新版复现总入口

其他 docs/*.md:
  拆分后的专题文档
```

---

## 20. 旧版 reproduce 文档备份

拆分文档前，建议将旧版大型 `reproduce.md` 备份为：

```text
docs/reproduce_legacy.md
```

如果还没有备份，可以执行：

```bash
git show HEAD:docs/reproduce.md > docs/reproduce_legacy.md
```

检查：

```bash
ls -lh docs/reproduce_legacy.md
head -n 20 docs/reproduce_legacy.md
```

预期：

```text
docs/reproduce_legacy.md 大小约几十 KB
开头为旧版复现实验说明文档
```

---

## 21. 推荐最终 Git 检查

完成文档迁移后，建议运行：

```bash
git status --short
git diff --stat
git ls-files | grep -E "\.pb$|\.npz$|\.npy$|checkpoint|model.ckpt|train.log|test.log|lcurve.out|out.json"
```

预期：

```text
git status --short:
  显示 README.md、docs/*.md 的修改或新增

git diff --stat:
  显示文档修改统计

大文件检查:
  无输出
```

确认无误后再提交。

---

## 22. 小结

本文档的核心原则是：

```text
代码、配置、文档、轻量 summary 可以提交；
本地数据、模型权重、中间预测数组、训练日志不提交。
```

尤其需要记住：

```text
.gitignore 中应写 /data/，而不是 data/。
```

这样可以保证：

```text
仓库保持轻量；
复现实验所需脚本被正确跟踪；
本地数据和模型文件不会误提交；
README 和 docs 能清楚说明实验流程和结果。
```