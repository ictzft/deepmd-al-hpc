# Round 3 Committee Prediction Summary

## Dataset

- Training set: data/toy_h2/round_003_train
- Training frames: 230
- Candidate pool: data/toy_h2/round_003_candidate
- Candidate frames: 20
- Validation set: data/toy_h2/valid
- Validation frames: 50

## Committee Models

| Model | Energy RMSE (eV) | Energy RMSE/Natoms (eV) | Force RMSE (eV/Å) |
|---|---:|---:|---:|
| model_000 | 7.858708e-01 | 3.929354e-01 | 3.768867e-01 |
| model_001 | 6.801369e-01 | 3.400684e-01 | 1.493383e-01 |
| model_002 | 6.787318e-01 | 3.393659e-01 | 1.070028e-01 |
| model_003 | 1.851586e+00 | 9.257932e-01 | 6.383155e-02 |

## Prediction

- n_models: 4
- n_frames: 20
- n_atoms: 2
- top_k: 10
- prediction time: 7 s

## Top-K Selected Frames

| Rank | Frame index | force_dev_max | force_dev_mean | energy_dev |
|---:|---:|---:|---:|---:|
| 1 | 5 | 0.2426791266 | 0.2426791266 | 1.0800038310 |
| 2 | 3 | 0.1903137402 | 0.1903137402 | 1.0794521494 |
| 3 | 0 | 0.1664500300 | 0.1664500300 | 1.0793401685 |
| 4 | 19 | 0.1605249853 | 0.1605249853 | 1.0789566469 |
| 5 | 4 | 0.1604581250 | 0.1604581250 | 1.0795085887 |
| 6 | 14 | 0.1594691271 | 0.1594691271 | 1.0798178017 |
| 7 | 13 | 0.1593927813 | 0.1593927813 | 1.0787198451 |
| 8 | 18 | 0.1585816729 | 0.1585816729 | 1.0800238925 |
| 9 | 8 | 0.1535236351 | 0.1535236351 | 1.0809361418 |
| 10 | 12 | 0.1504961229 | 0.1504961229 | 1.0782071722 |

## Notes

Round 3 committee retraining and prediction have been completed on 2 × Tesla V100 GPUs.

Energy deviation is nearly constant across the selected frames, likely because one committee member shows a relatively large energy RMSE. Therefore, the active learning selection in this round should mainly be interpreted as force-deviation-based selection.
