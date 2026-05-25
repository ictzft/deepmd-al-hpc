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
| 9 | Random multi-seed mean ± std | Round 001/002/003 | done |
| 10 | Uncertainty vs random comparison table | 16 rows, all rounds | done |
| 11 | Comparison learning curve figures | 4 SVG | done |
| 12 | Uncertainty-diversity selector | prototype, Round 001 | done |
| 13 | Trust-level (DP-GEN-style) selector | prototype, Round 001 | done |
| 14 | V100 training wall-time profiling | 36 train.log | done |
| 15 | V100 GPU utilization sample | representative | done |

---

## 3. Available Data and Results

### 3.1 Uncertainty branch (al_rounds_summary.csv)

| Round | Train | Candidate | Force RMSE | force_dev_max (selected top-K) |
|---:|---:|---:|---:|---:|
| 0 | 200 | 50 | 0.182 | 0.441 |
| 1 | 210 | 40 | 0.162 | 0.269 |
| 2 | 220 | 30 | 0.194 | 0.187 |
| 3 | 230 | 20 | 0.174 | 0.170 |

### 3.2 Random baseline (multi-seed mean Force RMSE)

| Round | Mean Force RMSE / eV/Å | Cross-seed std |
|---:|---:|---:|
| 001 | 0.211 | 0.055 |
| 002 | 0.196 | 0.016 |
| 003 | 0.189 | 0.048 |

### 3.3 Key result files

- `experiments/al_rounds_summary.csv` — uncertainty Round 0–3
- `experiments/baselines/random_round001_baseline_summary.csv`
- `experiments/baselines/random_round002_baseline_summary.csv`
- `experiments/baselines/random_round003_baseline_summary.csv`
- `experiments/baselines/random_vs_uncertainty_summary.csv` — 16 rows, all methods/rounds
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
- Training wall time: 36 models, mean=10.9s (1000 steps), 8.7ms/batch
- 2×V100 parallel speedup: 1.97× (near-linear)
- GPU utilization sample: SM 23%, memory 5407 MiB/16384 MiB (33%)
- Environment: Tesla V100-SXM2-16GB, DeepMD-kit v3.1.4.dev81

**Not done:**
- Systematic prediction time measurement
- Dataset update time measurement
- End-to-end round wall-clock time
- GPU utilization curves across full rounds
- Power monitoring

---

## 7. Remaining V100 Tasks

1. Systematic end-to-end profiling (prediction, I/O, dataset update)
2. GPU utilization/memory curves for a full training round
3. Uncertainty-diversity multi-round (Round 002–003)
4. Trust-level multi-round (Round 002–003)
5. Full 4-strategy comparison (uncertainty / random / diversity / trust-level)

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
4. Uncertainty-diversity Round 002–003
5. Trust-level Round 002–003
6. Full 4-strategy comparison
7. Small real DFT/AIMD dataset migration
