# rMD17 Benzene Active Learning Experiments

## Dataset

- Dataset: rMD17 benzene
- Raw data: `data/raw/rmd17_benzene.npz`
- DeePMD format data: `data/rmd17/benzene/`
- Type map: `C H O`

Data split:

| Split | Frames |
|---|---:|
| Initial train | 1000 |
| Candidate | 60000 |
| Validation | 5000 |
| Test | 10000 |

> Raw `.npz`, processed `.npy`, candidate pools, predictions, and frozen models are not tracked by Git. They are stored on the training server.

## Active Learning Pipeline

Each round follows:

1. Train 4 DeePMD committee models.
2. Predict the candidate pool with all committee models.
3. Compute committee model deviation.
4. Select top-1000 frames by uncertainty.
5. Merge selected frames into the next-round training set.
6. Remove selected frames from the candidate pool.

## Completed Rounds

| Round | Train frames | Candidate frames | Selection strategy | Committee models | Status |
|---|---:|---:|---|---:|---|
| 000 | 1000 | 60000 | initial | 4 | done |
| 001 | 2000 | 59000 | uncertainty top-1000 | 4 | done |
| 002 | 3000 | 58000 | uncertainty top-1000 | 4 | done |
| 003 | 4000 | 57000 | uncertainty top-1000 | 4 | done |

## Notes

- Benzene has completed the same real rMD17 active learning loop as ethanol.
- `DP_INFER_BATCH_SIZE=1800` was used during committee prediction to avoid V100 16GB OOM.
- Large artifacts are intentionally excluded from GitHub.

## Independent Test Results

The following results were evaluated on the 10000-frame independent test split using `model_000` from each committee. These are single-model test results, not committee-averaged results.

| Round | Train frames | Test Energy RMSE (eV) | Test Energy RMSE/Natoms (eV) | Test Force RMSE (eV/Å) |
|---|---:|---:|---:|---:|
| 000 | 1000 | 3.318915e-02 | 2.765762e-03 | 1.821129e-01 |
| 001 | 2000 | 4.113166e-02 | 3.427639e-03 | 2.018259e-01 |
| 002 | 3000 | 2.739586e-02 | 2.282988e-03 | 1.610441e-01 |
| 003 | 4000 | 3.360915e-02 | 2.800763e-03 | 1.873087e-01 |

Current observation: Round 002 gives the best independent-test force RMSE among the evaluated benzene uncertainty rounds. The trend is not strictly monotonic, so future comparison should include random baselines and multi-seed evaluation.

## Random Baseline Results

The following random baseline results were evaluated on the 10000-frame independent test split using `model_000` from each 4-model committee. These are single-model test results, not committee-averaged results.

| Baseline | Round | Train frames | Test Energy RMSE (eV) | Test Force RMSE (eV/Å) |
|---|---:|---:|---:|---:|
| random seed0 | 001 | 2000 | 3.067832e-02 | 1.939204e-01 |
| random seed0 | 002 | 3000 | 4.373666e-02 | 2.978373e-01 |
| random seed0 | 003 | 4000 | 3.684600e-02 | 1.975246e-01 |

Current observation: the random seed0 baseline is non-monotonic and shows a large degradation at Round 002. More random seeds are needed before drawing a statistical conclusion.

## Random Baseline Results: seed1

The following results were evaluated on the 10000-frame independent test split using `model_000` from each 4-model committee.

| Baseline | Round | Train frames | Test Energy RMSE (eV) | Test Energy RMSE/Natoms (eV) | Test Force RMSE (eV/Å) |
|---|---:|---:|---:|---:|---:|
| random seed1 | 001 | 2000 | 3.612558e-02 | 3.010465e-03 | 2.108467e-01 |
| random seed1 | 002 | 3000 | 3.393902e-02 | 2.828252e-03 | 2.037351e-01 |
| random seed1 | 003 | 4000 | 3.601689e-02 | 3.001408e-03 | 2.092180e-01 |

Current observation: random seed1 is relatively stable across Rounds 001-003, while random seed0 showed a larger degradation at Round 002. At least one more random seed is needed for a more reliable mean/std comparison.
