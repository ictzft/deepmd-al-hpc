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
