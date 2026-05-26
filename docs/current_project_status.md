# Current Project Status

Date: 2026-05-26 | Host: shared-v100 (2×Tesla V100-16GB) | Branch: main

---

## 1. One-sentence Summary

toy H2: 四策略 multi-seed multi-round 已完成, V100 profiling baseline 已完成。rMD17 ethanol: uncertainty branch Round 0–3 已完成, random baseline (3 seeds × 3 rounds) 已完成, independent test evaluation 已完成, diversity/trust_level baselines 待完成。H100 scaling 未开始。

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
| 19 | rMD17 ethanol uncertainty branch | Round 0–3 training + prediction + summary + independent test done | done |
| 20 | rMD17 ethanol independent test | 10000-frame test set, Force RMSE 0.344→0.327 eV/Å (monotonic) | done |
| 21 | rMD17 ethanol random baseline | 3 seeds × 3 rounds (36 models), uncertainty clearly better | done |
| 22 | rMD17 ethanol MD stability | NVE 10K stable (drift ~0.035 eV/ps), 100K+ dissociates | done |

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

**toy H2:**
- `experiments/al_rounds_summary.csv` — uncertainty Round 0–3
- `experiments/baselines/random_round001_baseline_summary.csv`
- `experiments/baselines/random_round002_baseline_summary.csv`
- `experiments/baselines/random_round003_baseline_summary.csv`
- `experiments/baselines/aligned_comparison.csv` — 统一 remaining-candidate-pool 指标的四策略对比（authoritative）

**rMD17 ethanol:**
- `experiments/rmd17_ethanol_summary/al_rounds_summary.csv`
- `experiments/rmd17_ethanol_summary/independent_test_all_summary.csv`
- `experiments/rmd17_ethanol_summary/random_baseline_round_summary.csv`
- `experiments/rmd17_ethanol_summary/profiling_unified.csv`
- `experiments/rmd17_ethanol_summary/md_stability/md_summary.json`

---

## 4. Available Scripts

| Category | Key Scripts |
|---|---|
| Data | `make_toy_h2_deepmd.py`, `merge_selected_frames.py`, `make_remaining_candidate.py`, `convert_rmd17_to_deepmd.py`, `split_rmd17_to_deepmd.py` |
| Training | `train_single_model.sh`, `train_committee_models.sh`, `train_round_committee_models.sh` |
| Inference | `predict_committee_models.py` |
| Selection | `select_from_predictions.py`, `select_by_strategy.py` |
| Analysis | `summarize_al_rounds.py`, `summarize_rmd17_random_baseline.py`, `plot_rmd17_ethanol.py`, `plot_rmd17_final_comparison.py` |
| Evaluation | `quick_md_test.py`, `run_md_stability.py`, `test_rmd17_all_independent.sh` |
| Profiling | `record_round_profiling.sh` |
| Runner | `run_random_baseline_round.sh`, `run_rmd17_random_baseline.sh` |

---

## 5. Available Figures

**toy H2:**
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

**rMD17 ethanol:**
```
experiments/rmd17_ethanol_summary/
  rmd17_ethanol_force_model_deviation_rounds.svg
  rmd17_ethanol_validation_rmse_rounds.svg
  rmd17_ethanol_dataset_size_rounds.svg
  rmd17_ethanol_independent_test_force_rmse.svg
  rmd17_ethanol_independent_test_energy_rmse.svg
  rmd17_ethanol_final_comparison.svg
  rmd17_ethanol_valid_vs_test.svg
```

---

## 6. V100 Profiling Progress

**Done (toy H2):**
- Training wall time: 132 models, mean=11.0s ± 0.5s, 8.7ms/batch (1000 steps)
- 2×V100 parallel speedup: 1.97× (near-linear), ~22s/round
- Estimated end-to-end: ~32s/round (train + predict + I/O)

**Done (rMD17 ethanol):**
- Training wall time: 52 models, uncertainty mean=50.4s, random mean=56.7s (2000 steps, 1000-4000 frames)
- Prediction: 57k-60k frames × 4 models = ~176s/round
- Per-round end-to-end: uncertainty ~5 min, random (3 seeds) ~10 min
- Full uncertainty branch (4 rounds): ~20 min; random baseline (3 seeds × 3 rounds): ~29 min
- Unified per-stage CSV: `experiments/rmd17_ethanol_summary/profiling_unified.csv`
- Per-model CSV: `experiments/rmd17_ethanol_summary/profiling_all_models.csv`

**Not done:**
- Full GPU utilization curves across entire rounds

---

## 7. Remaining V100 Tasks

1. (done) Systematic end-to-end profiling (training + prediction + I/O, 52 models, unified CSV)
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
- Toy H2 prototype with reproducible active learning pipeline on 2×V100
- Uncertainty top-K selects higher-uncertainty frames than random (both toy H2 and rMD17)
- Multi-seed multi-round four-strategy comparison completed on toy H2
- 2×V100 model-level parallel training is near-linear (1.97× speedup)
- rMD17 ethanol uncertainty branch shows monotonically decreasing Force RMSE on both validation (0.374→0.354) and independent test (0.344→0.327 eV/Å)
- rMD17 ethanol random baseline (3 seeds × 3 rounds) shows uncertainty has more stable RMSE improvement than random (Round 3: 0.354 vs 0.607 eV/Å, random std = 0.385)
- NVE MD at 10K stable with drift ~0.035 eV/ps; 100K+ dissociation indicates more training data needed
- Pipeline profiling complete (52 models, all stages, unified CSV)

**Cannot claim (yet):**
- Method works on multiple real DFT/AIMD systems (only rMD17 ethanol tested)
- Uncertainty sampling significantly outperforms random on multiple realistic datasets
- Diversity/trust_level strategies validated on real data (toy H2 only)
- High-temperature MD stability (100K+ dissociation)
- H100 multi-GPU scaling results
- Full CCF-B paper-level evidence

---

## 10. Recommended Next Commit / Experiment Order

1. (done) Sync documentation
2. (now) Create local data/artifact inventory
3. Systematic end-to-end V100 profiling
4. (done) Uncertainty-diversity Round 002–003
5. (done) Trust-level Round 002–003
6. (done) Full 4-strategy comparison
7. (done) rMD17 ethanol: uncertainty + random baseline + independent test + MD + profiling done
8. rMD17 ethanol: diversity + trust_level baselines
9. GPU utilization curves
10. H100 scaling
