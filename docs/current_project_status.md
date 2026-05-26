# Current Project Status

Date: 2026-05-25 | Host: shared-v100 (2×Tesla V100-16GB) | Branch: main

---

## 1. One-sentence Summary

Toy H2 offline active learning prototype with 4-model DeePMD committee, uncertainty top-K selection, multi-seed multi-round random baseline, and preliminary V100 training profiling — all completed on 2×V100.

---

## 2. Completed Components

| # | Component | Rounds / Seeds | Status |
|---|---|---|---|
| 1 | DeepMD-kit Docker environment | — | done |
| 2 | toy H2 data generation | 250 frames | done |
| 3 | Single-model DeePMD baseline | — | done |
| 4 | 4-model committee training + prediction | — | done |
| 5 | force_dev_max / force_dev_mean / energy_dev | — | done |
| 6 | Uncertainty top-K offline active learning | Round 0–3 | done |
| 7 | Random sampling selection-level baseline | Round 0–1, seed0/1/2 | done |
| 8 | Random retraining baseline | Round 001–003, seed0/1/2 | done |
| 9 | Uncertainty multi-seed retraining | Round 001–003, seed0/1/2 | done |
| 10 | Diversity multi-seed retraining | Round 001–003, seed0/1/2 | done |
| 11 | Trust-level multi-seed retraining | Round 001–003, seed0/1/2 | done |
| 12 | Multi-seed mean ± std | all 4 strategies | done |
| 13 | Aligned four-strategy comparison | consistent metric | done |
| 14 | Comparison learning curve figures | 4 SVG | done |
| 15 | V100 training wall-time profiling | 132 train.log | done |
| 16 | V100 GPU utilization sample | representative | done |
| 17 | Diversity descriptor analysis | H-H bond length, 3 strategies | done |
| 18 | Profiling scripts + summary | profile_command, parse, summarize | done |
| 19 | rMD17 ethanol uncertainty branch | Round 0–3 training + prediction + summary done | done |

---

## 3. Available Data and Results

### 3.1 Uncertainty branch (al_rounds_summary.csv)

| Round | Train | Candidate | Force RMSE | force_dev_max (selected top-K) |
|---:|---:|---:|---:|---:|
| 0 | 200 | 50 | 0.182 | 0.441 |
| 1 | 210 | 40 | 0.162 | 0.269 |
| 2 | 220 | 30 | 0.194 | 0.187 |
| 3 | 230 | 20 | 0.174 | 0.170 |

### 3.2 Four-Strategy Aligned Force RMSE (3-seed mean ± std)

| Strategy | R1 | R2 | R3 |
|---|---:|---:|---:|
| uncertainty | 1.51e-01 ± 0.025 | 2.13e-01 ± 0.024 | 1.96e-01 ± 0.024 |
| random | 2.11e-01 ± 0.055 | 1.96e-01 ± 0.016 | 1.89e-01 ± 0.048 |
| diversity | 2.05e-01 ± 0.058 | 1.74e-01 ± 0.009 | 1.76e-01 ± 0.041 |
| trust_level | 1.35e-01 ± 0.028 | 1.49e-01 ± 0.023 | 1.78e-01 ± 0.006 |

See `experiments/baselines/aligned_comparison.md` for full aligned comparison.

### 3.3 Key result files

- `experiments/al_rounds_summary.csv` — uncertainty Round 0–3
- `experiments/baselines/random_round001_baseline_summary.csv`
- `experiments/baselines/random_round002_baseline_summary.csv`
- `experiments/baselines/random_round003_baseline_summary.csv`
- `experiments/baselines/aligned_comparison.csv` — 统一 remaining-candidate-pool 指标的四策略对比（authoritative）
- `experiments/baselines/random_vs_uncertainty_summary.csv` — 旧版（legacy，存在 selected-K vs remaining-candidate 指标混用）
- `experiments/baselines/strategy_comparison_round000.csv` — selection-level 4-strategy comparison

---

## 4. Available Scripts

| Category | Key Scripts |
|---|---|
| Data | `make_toy_h2_deepmd.py`, `merge_selected_frames.py`, `make_remaining_candidate.py` |
| Training | `train_single_model.sh`, `train_committee_models.sh`, `train_round_committee_models.sh` |
| Inference | `predict_committee_models.py` |
| Selection | `select_from_predictions.py`, `select_by_strategy.py` |
| Analysis | `summarize_al_rounds.py`, `summarize_random_round_baselines.py`, `summarize_random_vs_uncertainty.py`, `plot_random_vs_uncertainty.py`, `compare_strategies_selection.py` |
| Profiling | `record_round_profiling.sh` |
| Inventory | `check_local_data_inventory.py`, `check_local_experiment_artifacts.py` |
| Runner | `run_random_baseline_round.sh` |

---

## 5. Available Figures

```
experiments/figures/
  dataset_size_rounds.svg
  force_model_deviation_rounds.svg
  validation_rmse_rounds.svg
  random_vs_uncertainty_force_rmse.svg
  random_vs_uncertainty_energy_rmse.svg
  random_vs_uncertainty_candidate_force_dev.svg
  random_vs_uncertainty_dataset_size.svg
```

---

## 6. V100 Profiling Progress

**Done:**
- Training wall time: 132 models, mean=11.0s ± 0.5s, 8.7ms/batch (1000 steps)
- 2×V100 parallel speedup: 1.97× (near-linear), ~22s/round
- Estimated end-to-end: ~32s/round (train + predict + I/O)
- GPU utilization sample: SM 23%, memory 5407 MiB (33%)
- Per-model wall time CSV: `experiments/profiling/training_wall_time_summary.csv`
- Diversity descriptor analysis: FPS achieves 3.1x greater structural spread
- Environment: Tesla V100-SXM2-16GB, DeepMD-kit v3.1.4.dev81

**Not done:**
- Full GPU utilization curves across entire rounds
- Per-stage (prediction/I/O) exact wall time in real training

---

## 7. Remaining V100 Tasks

1. Systematic end-to-end profiling (prediction, I/O, dataset update)
2. GPU utilization/memory curves for a full training round
3. (done) Uncertainty-diversity multi-round (Round 002–003)
4. (done) Trust-level multi-round (Round 002–003)
5. (done) Full 4-strategy comparison (uncertainty / random / diversity / trust-level)

---

## 8. Tasks Deferred to H100

1. Multi-GPU scaling (4/8 GPU)
2. Bfloat16 / mixed-precision training
3. Large-batch committee training throughput
4. H100 vs V100 speedup comparison

---

## 9. Claim Boundary

**Can claim:**
- Toy H2 prototype with reproducible active learning pipeline
- Uncertainty top-K selects higher-uncertainty frames than random
- Multi-seed multi-round random baseline for comparison
- 2×V100 model-level parallel training is near-linear

**Cannot claim:**
- Method works on real DFT/AIMD systems
- Uncertainty sampling significantly outperforms random on realistic datasets
- CCF-B paper-level evidence
- H100 scaling results
- MD stability validation

---

## 10. Recommended Next Commit / Experiment Order

1. (done) Sync documentation
2. (now) Create local data/artifact inventory
3. Systematic end-to-end V100 profiling
4. (done) Uncertainty-diversity Round 002–003
5. (done) Trust-level Round 002–003
6. (done) Full 4-strategy comparison
7. (started) Small real DFT/AIMD dataset migration — rMD17 ethanol: uncertainty branch Round 0–3 closed loop done, summary pending
