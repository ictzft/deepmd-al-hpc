# Local Experiment Artifact Inventory

Generated: 2026-05-25 | Host: shared-v100

**Note**: Large files (`.pb`, `.npz`, train.log, checkpoint) are NOT committed to Git.
This inventory is for local reference only.

| Directory | frozen_model.pb | train.log | selected_topk.json | prediction_summary.csv | metrics_summary.csv |
|---|---|---|---|---|---|
| experiments/exp_001_env_check | — | — | — | — | — |
| experiments/exp_002_framework_check | — | — | — | — | — |
| experiments/exp_003_single_model_baseline | 1 | 1 | — | — | — |
| experiments/exp_004_committee_models | 4 | 4 | — | — | — |
| experiments/exp_005_committee_prediction | — | — | 2.3KB | — | — |
| experiments/exp_006_offline_active_learning | — | — | — | — | — |
| experiments/exp_007_round001_committee_models | 4 | 4 | — | — | — |
| experiments/exp_008_round001_committee_prediction | — | — | 2.4KB | — | — |
| experiments/exp_009_round002_committee_models | 4 | 4 | — | — | — |
| experiments/exp_010_round002_committee_prediction | — | — | 2.4KB | — | — |
| experiments/exp_011_round003_committee_models | 4 | 4 | — | — | — |
| experiments/exp_012_round003_committee_prediction | — | — | 2.4KB | — | — |
| experiments/baselines/diversity_round000 | — | — | 2.1KB | — | — |
| experiments/baselines/diversity_round001_committee_models | 4 | 4 | — | — | — |
| experiments/baselines/diversity_round001_committee_prediction | — | — | 2.5KB | — | — |
| experiments/baselines/random_seed0_round001_committee_models | 4 | 4 | — | — | — |
| experiments/baselines/random_seed0_round001_committee_prediction | — | — | 2.5KB | — | — |
| experiments/baselines/random_seed0_round002_committee_models | 4 | 4 | — | — | — |
| experiments/baselines/random_seed0_round002_committee_prediction | — | — | 2.5KB | — | — |
| experiments/baselines/random_seed0_round003_committee_models | 4 | 4 | — | — | — |
| experiments/baselines/random_seed0_round003_committee_prediction | — | — | 2.5KB | — | — |
| experiments/baselines/random_seed1_round001_committee_models | 4 | 4 | — | — | — |
| experiments/baselines/random_seed1_round001_committee_prediction | — | — | 2.5KB | — | — |
| experiments/baselines/random_seed1_round002_committee_models | 4 | 4 | — | — | — |
| experiments/baselines/random_seed1_round002_committee_prediction | — | — | 2.5KB | — | — |
| experiments/baselines/random_seed1_round003_committee_models | 4 | 4 | — | — | — |
| experiments/baselines/random_seed1_round003_committee_prediction | — | — | 2.5KB | — | — |
| experiments/baselines/random_seed2_round001_committee_models | 4 | 4 | — | — | — |
| experiments/baselines/random_seed2_round001_committee_prediction | — | — | 2.5KB | — | — |
| experiments/baselines/random_seed2_round002_committee_models | 4 | 4 | — | — | — |
| experiments/baselines/random_seed2_round002_committee_prediction | — | — | 2.5KB | — | — |
| experiments/baselines/random_seed2_round003_committee_models | 4 | 4 | — | — | — |
| experiments/baselines/random_seed2_round003_committee_prediction | — | — | 2.5KB | — | — |
| experiments/baselines/trust_level_round000 | — | — | 2.3KB | — | — |
| experiments/baselines/trust_level_round001_committee_models | 4 | 4 | — | — | — |
| experiments/baselines/trust_level_round001_committee_prediction | — | — | 2.5KB | — | — |
| experiments/baselines/uncertainty_topk_round000 | — | — | 2.1KB | — | — |

## Baseline summary CSVs

- `experiments/baselines/random_round001_baseline_summary.csv` (671B)
- `experiments/baselines/random_round001_comparison.csv` (770B)
- `experiments/baselines/random_round002_baseline_summary.csv` (671B)
- `experiments/baselines/random_round003_baseline_summary.csv` (671B)
- `experiments/baselines/random_seed0_round001_metrics_summary.csv` (459B)
- `experiments/baselines/random_seed0_round001_prediction_summary.csv` (387B)
- `experiments/baselines/random_seed0_round002_metrics_summary.csv` (466B)
- `experiments/baselines/random_seed0_round002_prediction_summary.csv` (137B)
- `experiments/baselines/random_seed0_round003_metrics_summary.csv` (466B)
- `experiments/baselines/random_seed0_round003_prediction_summary.csv` (137B)
- `experiments/baselines/random_seed1_round001_metrics_summary.csv` (418B)
- `experiments/baselines/random_seed1_round001_prediction_summary.csv` (137B)
- `experiments/baselines/random_seed1_round002_metrics_summary.csv` (466B)
- `experiments/baselines/random_seed1_round002_prediction_summary.csv` (137B)
- `experiments/baselines/random_seed1_round003_metrics_summary.csv` (466B)
- `experiments/baselines/random_seed1_round003_prediction_summary.csv` (137B)
- `experiments/baselines/random_seed2_round001_metrics_summary.csv` (418B)
- `experiments/baselines/random_seed2_round001_prediction_summary.csv` (137B)
- `experiments/baselines/random_seed2_round002_metrics_summary.csv` (466B)
- `experiments/baselines/random_seed2_round002_prediction_summary.csv` (137B)
- `experiments/baselines/random_seed2_round003_metrics_summary.csv` (466B)
- `experiments/baselines/random_seed2_round003_prediction_summary.csv` (137B)
- `experiments/baselines/random_vs_uncertainty_summary.csv` (1523B)
- `experiments/baselines/selection_baseline_runs.csv` (1957B)
- `experiments/baselines/selection_baseline_summary.csv` (359B)
- `experiments/baselines/strategy_comparison_round000.csv` (539B)

## Notes

- `frozen_model.pb`, `train.log`, `checkpoint`, and `committee_predictions.npz` are NOT committed to Git.
- `selected_topk.json`, `*_summary.csv`, and `*_summary.md` ARE committed (lightweight results).
- Generated by `scripts/analysis/check_local_experiment_artifacts.py`.