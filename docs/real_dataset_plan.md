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

## 10. Current Status (2026-05-26)

rMD17 ethanol dataset pipeline — uncertainty branch Round 0–3 active learning loop is nearly complete:

**Data** (27 atoms/frame, C₂H₅OH):
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
| 3 | 4000 | 57000 | done | **pending** |

Each round selects 1000 uncertainty top-K frames from the candidate pool.

**Done:**
- Data conversion script: `scripts/data/convert_rmd17_to_deepmd.py`
- Data splitting script: `scripts/data/split_rmd17_to_deepmd.py`
- Round 0–3 committee configs (4 models × 4 rounds = 16 configs)
- Round 0–3 committee model training (16 frozen models)
- Round 0–2 committee predictions with uncertainty top-K selection

**Pending:**
- Round 3 committee prediction (next step)
- Round 0–3 summary and learning curve
- Independent test evaluation
- Multi-strategy comparison on real dataset
