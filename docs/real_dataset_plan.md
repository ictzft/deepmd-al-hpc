# Real DFT/AIMD Dataset Migration Plan

---

## 1. Why toy H2 is insufficient

Toy H2 (2 atoms, 250 frames) validates that the pipeline works but cannot support claims about real material systems:

- 2-atom H2 has no structural diversity beyond H-H bond length
- Committee model variance is inflated by tiny dataset size
- Valid set doubles as candidate pool — no independent evaluation
- Results cannot generalize to multi-element or periodic systems

A real DFT/AIMD dataset is the minimum requirement for a paper submission.

---

## 2. Minimum dataset requirements

- At least one realistic material system (e.g., bulk metal, oxide, water)
- Hundreds to thousands of frames
- Multi-element (not single-species)
- Periodic boundary conditions (PBC)
- Energy, force, and virial labels

---

## 3. Recommended data splits

| Split | Purpose | Typical Size |
|---|---|---|
| `initial_train` | Train Round 0 committee models | 100–200 frames |
| `candidate_pool` | Unlabeled pool for active learning selection | 500–2000 frames |
| `validation` | Model selection and hyperparameter tuning | 100–200 frames |
| `independent_test` | Final evaluation ONLY, never used in AL | 100–200 frames |

### Critical rules

1. `independent_test` must NEVER be used for selection
2. `candidate_pool` simulates expensive DFT labeling — each selected frame is "labeled"
3. `validation` is separate from `independent_test`
4. Toy H2 currently violates rule 1 (valid = candidate_pool). This must be fixed in the real dataset pipeline.

---

## 4. How to convert existing DeePMD data

If data is already in DeePMD format (npy arrays with `set.*/` structure):

```bash
python scripts/data/prepare_real_dataset_template.py \
  --source data/real_datasets/my_system/all_data \
  --output data/real_datasets/my_system \
  --initial-train 100 --candidate-pool 1000 \
  --validation 100 --test 100 \
  --seed 0
```

This creates the directory structure and metadata, then prints the manual steps needed to split the actual data.

---

## 5. How to reuse the four-strategy pipeline

The existing scripts work with any DeePMD-format data:

1. Replace `data/toy_h2/` paths with `data/real_datasets/<name>/` paths
2. Use `initial_train` as the starting training set
3. Use `candidate_pool` as the initial candidate pool
4. Use `validation` for `dp test` evaluation
5. Use `independent_test` only in final evaluation (never feed into AL round)

The `scripts/experiments/run_toy_h2_strategy_comparison.sh` template can be adapted by changing data paths.

---

## 6. Independent test evaluation

After all AL rounds complete:
1. Train final models on full training set
2. Compute Energy RMSE, Force RMSE on `independent_test`
3. Compute model deviation statistics
4. Compare against baseline (no AL, random selection)

The `independent_test` must be used exactly once — at the end.

---

## 7. MD stability validation (future)

MD stability is an important physics-based validation that goes beyond RMSE:

1. Run LAMMPS MD with the final frozen model
2. Check energy conservation (NVE ensemble)
3. Check RDF / structural properties match DFT reference
4. Check no unphysical bond breaking

This is deferred to after the real dataset pipeline is validated.

---

## 8. Files NOT to commit to Git

```
data/real_datasets/**/*.npy
data/real_datasets/**/*.npz
data/real_datasets/**/coord.npy
data/real_datasets/**/force.npy
data/real_datasets/**/energy.npy
data/real_datasets/**/box.npy
```

The `metadata.json` and associated config files ARE suitable for Git.

---

## 9. Next steps

1. Acquire a small real DFT/AIMD dataset (e.g., from a public repository or collaborator)
2. Convert to DeePMD npy format if needed
3. Run `prepare_real_dataset_template.py` to create splits
4. Adapt the strategy comparison runner to real data paths
5. Run 4-strategy comparison on real data
6. Perform independent test evaluation
7. (Future) Run MD stability validation

---

## 10. Current Status (2026-05-28)

rMD17 ethanol dataset pipeline — uncertainty branch Round 0–3 active learning loop is nearly complete:

**Data** (C₂H₅OH, 9 atoms, 27 Cartesian force components):
| Split | Frames | Path |
|---|---|---|
| Initial train | 1000 | `data/rmd17/ethanol/train` |
| Validation | 5000 | `data/rmd17/ethanol/valid` |
| Test | 10000 | `data/rmd17/ethanol/test` |
| Initial candidate | 60000 | `data/rmd17/ethanol/candidate` |

**Active learning rounds (uncertainty branch)**:
| Round | Train frames | Candidate frames | Training | Prediction |
|---|---:|---:|---|---|
| 0 | 1000 | 60000 | done | done |
| 1 | 2000 | 59000 | done | done |
| 2 | 3000 | 58000 | done | done |
| 3 | 4000 | 57000 | done | done |

Each round selects 1000 uncertainty top-K frames from the candidate pool.

**Done:**
- Data conversion script: `scripts/data/convert_rmd17_to_deepmd.py`
- Data splitting script: `scripts/data/split_rmd17_to_deepmd.py`
- Round 0–3 committee configs (4 models × 4 rounds = 16 configs)
- Round 0–3 committee model training (16 frozen models)
- Round 0–3 committee predictions with uncertainty top-K selection
- Round 0–3 summary CSV + MD + learning curve figures
- Unified profiling CSV (52 models, all pipeline stages)

**End-to-End Pipeline Profiling (2x V100)**:
| Round | Train (s) | Pred (s) | Other (s) | Total (s) |
|---:|---:|---:|---:|---:|
| 0 | 87 | 185 | 21 | 293 |
| 1 | 104 | 182 | 21 | 307 |
| 2 | 107 | 179 | 21 | 307 |
| 3 | 106 | 176 | 21 | 303 |

- Training: 4 models / 2 GPUs parallel, mean 50.4s/model (uncertainty), 56.7s/model (random)
- Prediction: 57k–60k frames × 4 models, ~3 min per round
- Per-round total: ~5 min (uncertainty), ~10 min (random, 3 seeds)
- Full uncertainty branch (Round 0–3): ~20 min
- Full random baseline (3 seeds × 3 rounds): ~29 min
- Data: `experiments/rmd17_ethanol_summary/profiling_unified.csv`, `profiling_all_models.csv`

**Key Results (uncertainty branch)**:

| Round | Train | Candidate | Force RMSE mean | force_dev_max (selected top-1000) |
|---:|---:|---:|---:|---:|
| 0 | 1000 | 60000 | 3.739e-01 | 6.129e-01 |
| 1 | 2000 | 59000 | 3.715e-01 | 4.570e-01 |
| 2 | 3000 | 58000 | 3.644e-01 | 3.906e-01 |
| 3 | 4000 | 57000 | 3.537e-01 | 4.569e-01 |

- Force RMSE monotonically decreasing (0.374 → 0.354 eV/Å), unlike toy H2
- top-1000 force_dev_max_mean decreased Round 0→2 (0.613 → 0.391), then bounced to 0.457 in Round 3
- Energy RMSE stable at ~0.12–0.13 eV across all rounds
- Summary files: `experiments/rmd17_ethanol_summary/`

**Independent Test Results (10000 frames, never used in AL)**:
| Round | Force RMSE (test) | Force RMSE (valid) |
|---:|---:|---:|
| 0 | 0.343914 | 0.373912 |
| 1 | 0.343304 | 0.371493 |
| 2 | 0.335249 | 0.364440 |
| 3 | 0.326594 | 0.353702 |

- Test Force RMSE decreases monotonically (0.344→0.327 eV/Å), confirming genuine improvement
- Test RMSE consistently ~0.028 eV/Å lower than validation

**Random Baseline (3 seeds × 3 rounds)**:
| Round | Uncertainty F_RMSE | Random F_RMSE (mean±std) |
|---:|---:|---:|
| 1 | 0.3715 | 0.3734 ± 0.010 |
| 2 | 0.3644 | 0.3990 ± 0.031 |
| 3 | 0.3537 | 0.6067 ± 0.385 |

- Uncertainty Force RMSE monotonically decreases, random worsens significantly
- Random Round 3 shows catastrophic degradation (0.607 vs 0.354 eV/Å)

**MD Stability (NVE, 10K)**:
- All models stable at 10K with drift ~0.035 eV/ps
- At 100K+, all models dissociate immediately (Force RMSE ~0.35 eV/Å insufficient for MD)
- Uncertainty Round 3 has marginally lowest drift (-0.0338 eV/ps)

**Four-Strategy Comparison (Round 3, 3-seed mean ± std)**:
| Strategy | Force RMSE | Std |
|---|---:|---:|
| uncertainty | 0.3537 | 0.0247 |
| diversity | 0.3555 | 0.0143 |
| trust_level | 0.3616 | 0.0166 |
| random | 0.6067 | 0.6826 |

All three active strategies within 1σ, all have clearly lower mean Force RMSE than random (0.607 ± 0.683 eV/Å). However, random variance is large — strict statistical significance cannot be claimed. Consistent with toy H2.

**Top-K Labeling Budget Ablation (Round 3, 3-seed mean ± std)**:
| K | Force RMSE | Std | vs K=1000 |
|---:|---:|---:|---:|
| 250 | 0.3408 | 0.0141 | 3.6% better |
| 500 | 0.4146 | 0.1790 | one outlier seed |
| 1000 | 0.3537 | 0.0247 | baseline |
| 2000 | 0.3315 | 0.0176 | 6.3% better |

- K=250 (most selective) and K=2000 (most data) both outperform K=1000
- Larger K benefits from more training data; smaller K benefits from higher selection precision

**Committee Size Ablation (Round 1, 3-seed mean ± std)**:
| Committee | Force RMSE | Std | Train Time/round |
|---:|---:|---:|---:|
| 2 models | 0.3436 | 0.0155 | ~55s (1 batch) |
| 4 models | 0.3715 | 0.0146 | ~110s (2 batches) |
| 8 models | 0.3392 | 0.0206 | ~220s (4 batches) |

- 8-model best Force RMSE but 2× training cost vs 4-model
- 2-model competitive with 4-model at half the cost
- Diminishing returns: 4→8 models gives only 8.7% RMSE improvement for 2× cost

---

## 11. rMD17 Benzene Results (2026-05-27)

rMD17 benzene (C₆H₆, 12 atoms) is the second real molecular system validated.

**Data**:
| Split | Frames | Path |
|---|---|---|
| Initial train | 1000 | `data/rmd17/benzene/train` |
| Validation | 5000 | `data/rmd17/benzene/valid` |
| Test | 10000 | `data/rmd17/benzene/test` |
| Initial candidate | 60000 | `data/rmd17/benzene/candidate` |

**Active learning rounds (uncertainty branch)**:
| Round | Train frames | Candidate frames | Selection | Status |
|---:|---:|---:|---|---|
| 000 | 1000 | 60000 | initial | done |
| 001 | 2000 | 59000 | uncertainty top-1000 | done |
| 002 | 3000 | 58000 | uncertainty top-1000 | done |
| 003 | 4000 | 57000 | uncertainty top-1000 | done |

- 4 committee models per round, `DP_INFER_BATCH_SIZE=1800` to avoid V100 OOM
- Same pipeline and scripts as rMD17 ethanol

**Done:**
- Data conversion and splitting
- Round 000–003 committee training (4 models × 4 rounds = 16 models)
- Round 000–003 committee prediction + uncertainty top-1000 selection
- Random baseline (seed0/1/2 Round 001–003)
- Independent test evaluation

**Pending:**
- Diversity baseline (3 seeds × 3 rounds)
- Trust_level baseline (3 seeds × 3 rounds)
- MD stability (NVE 10K/100K)
- Four-strategy comparison
- Pipeline profiling

**Pending (general):**
- Additional molecular/material systems beyond rMD17
