# 代码与配置检查

本文档用于记录 `deepmd-al-hpc` 项目提交前的代码、脚本、配置、Git 状态和大文件检查命令。

建议在每次提交前执行本文档中的检查，避免出现：

```text
Python 脚本语法错误
Shell 脚本语法错误
JSON 配置格式错误
.pb / .npz / .npy 大文件误提交
新建 docs 文档忘记 git add
random baseline 轻量结果没有被 Git 跟踪
```

---

## 1. 检查前准备

进入项目根目录：

```bash
cd /data/zft
```

查看当前分支和工作区状态：

```bash
git status -sb
git log --oneline -5
```

如果当前只是文档修改，可以优先检查：

```bash
git status --short
git diff --stat
```

如果涉及代码、脚本或配置修改，则建议完整执行本文档后续检查。

---

## 2. Python 语法检查

### 2.1 使用 `python`

在 DeepMD Docker 容器或合适的 Python 环境中运行：

```bash
python -m py_compile \
  scripts/active_learning/run_framework_check.py \
  scripts/active_learning/run_offline_al_round.py \
  scripts/active_learning/select_from_predictions.py \
  scripts/inference/predict_committee_models.py \
  scripts/data/make_toy_h2_deepmd.py \
  scripts/data/merge_selected_frames.py \
  scripts/data/make_remaining_candidate.py \
  scripts/config/make_round_committee_configs.py \
  scripts/analysis/summarize_al_rounds.py \
  scripts/analysis/plot_al_rounds.py \
  scripts/analysis/summarize_selection_baselines.py \
  scripts/analysis/summarize_random_round001_baselines.py \
  scripts/analysis/summarize_random_vs_uncertainty.py \
  scripts/analysis/plot_random_vs_uncertainty.py \
  scripts/analysis/prepare_random_baseline_round.py \
  scripts/analysis/summarize_random_round_baselines.py \
  src/metrics/deviation.py \
  src/al/scheduler.py \
  src/al/selector.py \
  src/al/loop.py
```

如果没有输出，说明这些 Python 文件语法检查通过。

---

### 2.2 使用 `python3`

如果宿主机没有 `python` 命令，可以改用：

```bash
python3 -m py_compile \
  scripts/active_learning/run_framework_check.py \
  scripts/active_learning/run_offline_al_round.py \
  scripts/active_learning/select_from_predictions.py \
  scripts/inference/predict_committee_models.py \
  scripts/data/make_toy_h2_deepmd.py \
  scripts/data/merge_selected_frames.py \
  scripts/data/make_remaining_candidate.py \
  scripts/config/make_round_committee_configs.py \
  scripts/analysis/summarize_al_rounds.py \
  scripts/analysis/plot_al_rounds.py \
  scripts/analysis/summarize_selection_baselines.py \
  scripts/analysis/summarize_random_round001_baselines.py \
  scripts/analysis/summarize_random_vs_uncertainty.py \
  scripts/analysis/plot_random_vs_uncertainty.py \
  scripts/analysis/prepare_random_baseline_round.py \
  scripts/analysis/summarize_random_round_baselines.py \
  src/metrics/deviation.py \
  src/al/scheduler.py \
  src/al/selector.py \
  src/al/loop.py
```

---

## 3. Shell 脚本语法检查

### 3.1 训练与评估脚本

运行：

```bash
bash -n \
  scripts/train/train_single_model.sh \
  scripts/train/train_committee_models.sh \
  scripts/train/train_round_committee_models.sh \
  scripts/eval/freeze_model.sh \
  scripts/eval/test_single_model.sh \
  scripts/run_random_baseline_round.sh \
  scripts/profiling/record_round_profiling.sh
```

如果没有输出，说明这些 Shell 脚本语法检查通过。

---

### 3.2 Docker 入口脚本

如果修改过 Docker 入口脚本，也建议检查：

```bash
bash -n \
  scripts/docker/check_deepmd_env.sh \
  scripts/docker/enter_deepmd_container.sh \
  scripts/docker/enter_torch_container.sh
```

---

## 4. JSON 配置检查

### 4.1 检查基础 DeepMD 配置

```bash
python -m json.tool configs/deepmd/toy_h2_input.json > /tmp/check_toy_h2_input.json
```

如果宿主机没有 `python` 命令，可以使用：

```bash
python3 -m json.tool configs/deepmd/toy_h2_input.json > /tmp/check_toy_h2_input.json
```

---

### 4.2 检查初始 committee 配置

```bash
for f in configs/deepmd/committee/*.json; do
  python -m json.tool "$f" > /tmp/check_committee_config.json
done
```

如果使用 `python3`：

```bash
for f in configs/deepmd/committee/*.json; do
  python3 -m json.tool "$f" > /tmp/check_committee_config.json
done
```

---

### 4.3 检查 Round 1–3 committee 配置

```bash
for f in configs/deepmd/round_001_committee/*.json \
         configs/deepmd/round_002_committee/*.json \
         configs/deepmd/round_003_committee/*.json; do
  python -m json.tool "$f" > /tmp/check_round_config.json
done
```

如果使用 `python3`：

```bash
for f in configs/deepmd/round_001_committee/*.json \
         configs/deepmd/round_002_committee/*.json \
         configs/deepmd/round_003_committee/*.json; do
  python3 -m json.tool "$f" > /tmp/check_round_config.json
done
```

---

### 4.4 检查 random seed0 配置

```bash
for f in configs/deepmd/random_seed0_round_001_committee/*.json; do
  python -m json.tool "$f" > /tmp/check_random_seed0_config.json
done
```

如果使用 `python3`：

```bash
for f in configs/deepmd/random_seed0_round_001_committee/*.json; do
  python3 -m json.tool "$f" > /tmp/check_random_seed0_config.json
done
```

---

### 4.5 检查 random seed1 配置

```bash
for f in configs/deepmd/random_seed1_round_001_committee/*.json; do
  python -m json.tool "$f" > /tmp/check_random_seed1_config.json
done
```

如果使用 `python3`：

```bash
for f in configs/deepmd/random_seed1_round_001_committee/*.json; do
  python3 -m json.tool "$f" > /tmp/check_random_seed1_config.json
done
```

---

### 4.6 检查 random seed2 配置

```bash
for f in configs/deepmd/random_seed2_round_001_committee/*.json; do
  python -m json.tool "$f" > /tmp/check_random_seed2_config.json
done
```

如果使用 `python3`：

```bash
for f in configs/deepmd/random_seed2_round_001_committee/*.json; do
  python3 -m json.tool "$f" > /tmp/check_random_seed2_config.json
done
```

---

## 5. 检查 DeepMD 配置中的训练数据路径

如果修改了 committee 配置，建议检查配置中的训练数据路径是否正确。

### 5.1 检查 Round 1 配置

```bash
grep -R "training_data\|systems\|round_001_train" -n \
  configs/deepmd/round_001_committee
```

预期应看到：

```text
data/toy_h2/round_001_train
```

---

### 5.2 检查 Round 2 配置

```bash
grep -R "training_data\|systems\|round_002_train" -n \
  configs/deepmd/round_002_committee
```

预期应看到：

```text
data/toy_h2/round_002_train
```

---

### 5.3 检查 Round 3 配置

```bash
grep -R "training_data\|systems\|round_003_train" -n \
  configs/deepmd/round_003_committee
```

预期应看到：

```text
data/toy_h2/round_003_train
```

---

### 5.4 检查 random seed0 配置

```bash
grep -R "training_data\|systems\|random_seed0_round_001_train" -n \
  configs/deepmd/random_seed0_round_001_committee
```

预期应看到：

```text
data/toy_h2/random_seed0_round_001_train
```

---

### 5.5 检查 random seed1 配置

```bash
grep -R "training_data\|systems\|random_seed1_round_001_train" -n \
  configs/deepmd/random_seed1_round_001_committee
```

预期应看到：

```text
data/toy_h2/random_seed1_round_001_train
```

---

### 5.6 检查 random seed2 配置

```bash
grep -R "training_data\|systems\|random_seed2_round_001_train" -n \
  configs/deepmd/random_seed2_round_001_committee
```

预期应看到：

```text
data/toy_h2/random_seed2_round_001_train
```

---

## 6. 检查 toy H2 数据生成脚本是否被跟踪

确认 `scripts/data/make_toy_h2_deepmd.py` 已经被 Git 跟踪：

```bash
git ls-files scripts/data/make_toy_h2_deepmd.py
```

预期输出：

```text
scripts/data/make_toy_h2_deepmd.py
```

如果没有输出，说明该文件尚未被 Git 跟踪，需要执行：

```bash
git add scripts/data/make_toy_h2_deepmd.py
```

也可以检查远端 `origin/main` 是否已有该文件：

```bash
git ls-tree -r origin/main scripts/data | grep make_toy_h2_deepmd
```

预期输出类似：

```text
100644 blob ... scripts/data/make_toy_h2_deepmd.py
```

---

## 7. 检查 random baseline 轻量结果

查看 random baseline 相关文件是否已经被 Git 跟踪：

```bash
git ls-files | grep -E "random_seed[012]|random_round|random_vs_uncertainty|selection_baseline|selected_random|selected_uncertainty"
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
experiments/baselines/random_seed1_round001_metrics_summary.csv
experiments/baselines/random_seed1_round001_metrics_summary.md
experiments/baselines/random_seed1_round001_prediction_summary.csv
experiments/baselines/random_seed1_round001_prediction_summary.md
experiments/baselines/random_seed2_round001_metrics_summary.csv
experiments/baselines/random_seed2_round001_metrics_summary.md
experiments/baselines/random_seed2_round001_prediction_summary.csv
experiments/baselines/random_seed2_round001_prediction_summary.md
experiments/baselines/random_round001_baseline_summary.csv
experiments/baselines/random_round001_baseline_summary.md
experiments/baselines/random_round001_comparison.csv
experiments/baselines/random_round001_comparison.md
experiments/baselines/random_vs_uncertainty_summary.csv
experiments/baselines/random_vs_uncertainty_summary.md
experiments/exp_005_committee_prediction/selected_random_seed0.json
experiments/exp_005_committee_prediction/selected_random_seed1.json
experiments/exp_005_committee_prediction/selected_random_seed2.json
experiments/exp_005_committee_prediction/selected_uncertainty.json
experiments/baselines/random_seed0_round001_committee_prediction/selected_topk.json
experiments/baselines/random_seed1_round001_committee_prediction/selected_topk.json
experiments/baselines/random_seed2_round001_committee_prediction/selected_topk.json
```

说明：

```text
这些是轻量 summary / selection 文件，可以提交到 GitHub。
```

---

## 8. 检查文档拆分文件

如果正在进行文档拆分，检查 `docs/` 目录：

```bash
find docs -maxdepth 1 -type f | sort
```

期望包括：

```text
docs/code_check.md
docs/data_and_git_policy.md
docs/paper_evidence.md
docs/profiling_h100.md
docs/profiling_v100.md
docs/random_baseline.md
docs/random_baseline_next_steps.md
docs/reproduce.md
docs/reproduce_legacy.md
docs/results.md
docs/setup.md
docs/toy_h2_pipeline.md
docs/uncertainty_rounds.md
```

其中：

```text
docs/reproduce.md
  新版复现总入口

docs/reproduce_legacy.md
  旧版大型 reproduce 文档备份

docs/setup.md
docs/toy_h2_pipeline.md
docs/uncertainty_rounds.md
docs/random_baseline.md
docs/random_baseline_next_steps.md
docs/paper_evidence.md
docs/results.md
docs/data_and_git_policy.md
docs/code_check.md
docs/profiling_v100.md
docs/profiling_h100.md
  拆分后的专题文档
```

---

## 9. 检查大文件是否被误跟踪

每次提交前建议运行：

```bash
git ls-files | grep -E "\.pb$|\.npz$|\.npy$|checkpoint|model.ckpt|train.log|test.log|lcurve.out|out.json|input_v2_compat.json"
```

如果没有输出，说明没有大文件被 Git 跟踪。

如果有输出，需要检查这些文件是否应当从 Git 中移除。

---

## 10. 如果大文件已经被 Git 跟踪

如果发现某个大文件已经被 Git 跟踪，但你希望保留本地文件、只从 Git 中移除，可以使用：

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

然后提交：

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

## 11. Git 状态检查

### 11.1 查看工作区状态

```bash
git status --short
```

常见状态含义：

```text
M   已修改并被 Git 跟踪的文件
??  新建但尚未被 Git 跟踪的文件
D   被删除的文件
```

例如：

```text
 M README.md
 M docs/reproduce.md
?? docs/code_check.md
?? docs/data_and_git_policy.md
?? docs/profiling_h100.md
?? docs/random_baseline.md
?? docs/results.md
?? docs/setup.md
?? docs/toy_h2_pipeline.md
?? docs/uncertainty_rounds.md
```

说明：

```text
README.md 和 docs/reproduce.md 已修改；
其他 docs/*.md 是新建但尚未 git add 的文件。
```

---

### 11.2 查看具体修改

查看所有未暂存修改：

```bash
git diff
```

查看指定文件修改：

```bash
git diff README.md
git diff docs/reproduce.md
git diff docs/setup.md
```

---

### 11.3 查看暂存区修改

如果已经执行 `git add`，可以查看暂存区：

```bash
git diff --cached
```

查看暂存区统计：

```bash
git diff --cached --stat
```

---

### 11.4 查看最近提交

```bash
git log --oneline -5
```

---

## 12. 文档修改后的检查

如果本次只修改文档，可以至少执行：

```bash
git status --short
git diff --stat
find docs -maxdepth 1 -type f | sort
```

如果想检查 Markdown 文件是否存在异常截断，可以看行数：

```bash
wc -l docs/*.md
```

也可以预览前几行：

```bash
for f in docs/*.md; do
  echo
  echo "===== $f ====="
  sed -n '1,20p' "$f"
done
```

---

## 13. 推荐提交前 Checklist

提交前建议按以下顺序执行。

### 13.1 代码语法检查

```bash
python -m py_compile \
  scripts/active_learning/run_framework_check.py \
  scripts/active_learning/run_offline_al_round.py \
  scripts/active_learning/select_from_predictions.py \
  scripts/inference/predict_committee_models.py \
  scripts/data/make_toy_h2_deepmd.py \
  scripts/data/merge_selected_frames.py \
  scripts/data/make_remaining_candidate.py \
  scripts/config/make_round_committee_configs.py \
  scripts/analysis/summarize_al_rounds.py \
  scripts/analysis/plot_al_rounds.py \
  scripts/analysis/summarize_selection_baselines.py \
  scripts/analysis/summarize_random_round001_baselines.py \
  scripts/analysis/summarize_random_vs_uncertainty.py \
  scripts/analysis/plot_random_vs_uncertainty.py \
  scripts/analysis/prepare_random_baseline_round.py \
  scripts/analysis/summarize_random_round_baselines.py \
  src/metrics/deviation.py \
  src/al/scheduler.py \
  src/al/selector.py \
  src/al/loop.py
```

---

### 13.2 Shell 脚本检查

```bash
bash -n \
  scripts/train/train_single_model.sh \
  scripts/train/train_committee_models.sh \
  scripts/train/train_round_committee_models.sh \
  scripts/eval/freeze_model.sh \
  scripts/eval/test_single_model.sh \
  scripts/run_random_baseline_round.sh \
  scripts/profiling/record_round_profiling.sh
```

---

### 13.3 JSON 配置检查

```bash
python -m json.tool configs/deepmd/toy_h2_input.json > /tmp/check_toy_h2_input.json

for f in configs/deepmd/committee/*.json \
         configs/deepmd/round_001_committee/*.json \
         configs/deepmd/round_002_committee/*.json \
         configs/deepmd/round_003_committee/*.json \
         configs/deepmd/random_seed0_round_001_committee/*.json \
         configs/deepmd/random_seed1_round_001_committee/*.json \
         configs/deepmd/random_seed2_round_001_committee/*.json; do
  python -m json.tool "$f" > /tmp/check_config.json
done
```

---

### 13.4 Git 状态检查

```bash
git status --short
git diff --stat
git log --oneline -5
```

---

### 13.5 大文件误跟踪检查

```bash
git ls-files | grep -E "\.pb$|\.npz$|\.npy$|checkpoint|model.ckpt|train.log|test.log|lcurve.out|out.json|input_v2_compat.json"
```

预期：

```text
无输出
```

---

## 14. 推荐提交方式

### 14.1 文档拆分提交

如果本次主要是拆分和完善文档：

```bash
git add README.md docs/
git diff --cached --stat
git commit -m "Refine README and split reproduce guide into modular docs"
git push origin main
```

---

### 14.2 只提交 docs

如果只修改了 `docs/`：

```bash
git add docs/
git diff --cached --stat
git commit -m "Update modular documentation"
git push origin main
```

---

### 14.3 代码与文档分开提交

如果同时修改代码和文档，建议分开提交：

```bash
git add scripts/ src/ configs/
git commit -m "Update active learning scripts"

git add README.md docs/
git commit -m "Update documentation"

git push origin main
```

---

## 15. 常见问题

### 15.1 `python` 命令不存在

如果宿主机没有 `python`，可以使用：

```bash
python3
```

例如：

```bash
python3 -m py_compile scripts/data/make_toy_h2_deepmd.py
```

---

### 15.2 JSON 检查失败

如果出现 JSON 解析错误，需要检查对应配置文件是否有：

```text
多余逗号
缺少括号
字符串引号错误
路径写错
```

可以用：

```bash
python -m json.tool <config.json>
```

定位错误。

---

### 15.3 Shell 检查失败

如果 `bash -n` 报错，通常说明脚本存在：

```text
if / fi 不匹配
for / done 不匹配
引号没有闭合
反斜杠续行错误
```

需要先修复脚本语法，再提交。

---

### 15.4 大文件检查有输出

如果大文件检查命令有输出，先不要提交。

需要判断：

```text
该文件是否应该提交？
是否应该加入 .gitignore？
是否应该 git rm --cached？
```

一般来说：

```text
.pb / .npz / .npy / checkpoint / log 文件不应该提交。
```

---

## 16. 小结

本文档的目标是保证每次提交前完成基本检查：

```text
Python 脚本可解析
Shell 脚本语法正确
JSON 配置格式正确
Git 状态清晰
大文件没有误跟踪
文档拆分文件完整
```

推荐最终提交前至少运行：

```bash
git status --short
git diff --stat
git ls-files | grep -E "\.pb$|\.npz$|\.npy$|checkpoint|model.ckpt|train.log|test.log|lcurve.out|out.json|input_v2_compat.json"
```

如果大文件检查无输出，且 `git diff --cached --stat` 只包含预期文件，再进行 commit 和 push。

---

## 17. rMD17 Benzene 验证

如果修改了 benzene 相关的脚本、配置或结果文件，建议执行以下额外检查。

### 17.1 benzene Python 脚本语法检查

```bash
python3 -m py_compile \
  scripts/experiments/run_rmd17_benzene_strategy_baseline.py \
  scripts/analysis/summarize_rmd17_benzene.py \
  scripts/analysis/plot_rmd17_benzene_comparison.py
```

### 17.2 benzene 配置完整性检查

```bash
# 检查 diversity config 目录
for seed in 0 1 2; do
  for rd in 1 2 3; do
    dir="configs/deepmd/rmd17_benzene_diversity_seed${seed}_round00${rd}_committee"
    n=$(ls "$dir"/*.json 2>/dev/null | wc -l)
    [ "$n" -ne 4 ] && echo "MISSING: $dir has $n files (expected 4)"
  done
done

# 检查 trust_level config 目录
for seed in 0 1 2; do
  for rd in 1 2 3; do
    dir="configs/deepmd/rmd17_benzene_trust_level_seed${seed}_round00${rd}_committee"
    n=$(ls "$dir"/*.json 2>/dev/null | wc -l)
    [ "$n" -ne 4 ] && echo "MISSING: $dir has $n files (expected 4)"
  done
done

echo "Config check done（无输出 = 全部通过）"
```

### 17.3 benzene 实验摘要完整性检查

```bash
# 检查四策略汇总 CSV
for f in experiments/rmd17_benzene_summary/all_strategies_detail.csv \
         experiments/rmd17_benzene_summary/four_strategy_comparison.csv; do
  [ -f "$f" ] && echo "✅ $f" || echo "❌ MISSING: $f"
done

# 检查 MD stability
for f in experiments/rmd17_benzene_summary/md_stability/md_summary.json; do
  [ -f "$f" ] && echo "✅ $f" || echo "❌ MISSING: $f"
done

# 检查 selected JSON
for strategy in diversity trust_level; do
  for seed in 0 1 2; do
    for rd in 1 2 3; do
      f="experiments/baselines/rmd17_benzene_${strategy}_seed${seed}_round00${rd}_committee_prediction/selected_${strategy}.json"
      [ -f "$f" ] || echo "❌ MISSING: $f"
    done
  done
done
echo "Summary check done（无 MISSING 输出 = 全部通过）"
```

### 17.4 benzene 四策略结果一致性检查

```bash
python3 -c "
import csv
with open('experiments/rmd17_benzene_summary/all_strategies_detail.csv') as f:
    rows = list(csv.DictReader(f))
# 检查 Round 3 有 4 个策略的结果
rd3 = [r for r in rows if r['round'] == '3']
strategies = set(r['strategy'] for r in rd3)
assert strategies == {'uncertainty', 'random', 'diversity', 'trust_level'}, f'Missing strategies: {strategies}'
print(f'Round 3 strategies: {strategies}')
for r in rd3:
    assert r['force_rmse_mean'], f'Missing force_rmse_mean for {r[\"strategy\"]}'
print('All strategies have valid force_rmse_mean')
print('✅ Consistency check passed')
"
```

### 17.5 benzene 大文件泄露检查

```bash
# benzene 相关目录不应有 pb/npz/npy 被 git 跟踪
git ls-files experiments/baselines/rmd17_benzene_* | grep -E "\.pb$|\.npz$|\.npy$" && echo "❌ LEAK" || echo "✅ No large files tracked"
git ls-files experiments/rmd17_benzene_* | grep -E "\.pb$|\.npz$|\.npy$" && echo "❌ LEAK" || echo "✅ No large files tracked"
```