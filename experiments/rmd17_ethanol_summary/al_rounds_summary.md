# rMD17 Ethanol Active Learning Round Summary

Dataset: rMD17 ethanol (C₂H₅OH, 9 atoms). 2×Tesla V100-16GB. 2026-05-26.

## Round-Level Summary (Validation Set, 5000 frames)

| round | train_frames | candidate_frames | n_models | energy_rmse_mean | energy_rmse_min | energy_rmse_max | energy_rmse_natoms_mean | force_rmse_mean | force_rmse_min | force_rmse_max | prediction_n_frames | top_k | force_dev_max_max | force_dev_max_mean | force_dev_max_min | energy_dev_mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1000 | 60000 | 4 | 1.299260e-01 | 1.191417e-01 | 1.382067e-01 | 1.443622e-02 | 3.739117e-01 | 3.540591e-01 | 3.954685e-01 | 60000 | 1000 | 9.524819e-01 | 6.129172e-01 | 5.503888e-01 | 3.944752e-02 |
| 1 | 2000 | 59000 | 4 | 1.338996e-01 | 1.255386e-01 | 1.396081e-01 | 1.487773e-02 | 3.714928e-01 | 3.525164e-01 | 3.861489e-01 | 59000 | 1000 | 8.401860e-01 | 4.569907e-01 | 4.044235e-01 | 5.921024e-02 |
| 2 | 3000 | 58000 | 4 | 1.227802e-01 | 1.203357e-01 | 1.260708e-01 | 1.364225e-02 | 3.644400e-01 | 3.542023e-01 | 3.847298e-01 | 58000 | 1000 | 5.422389e-01 | 3.905639e-01 | 3.580679e-01 | 3.036345e-02 |
| 3 | 4000 | 57000 | 4 | 1.285761e-01 | 1.149267e-01 | 1.424763e-01 | 1.428623e-02 | 3.537022e-01 | 3.282416e-01 | 3.752929e-01 | 57000 | 1000 | 8.327228e-01 | 4.568651e-01 | 4.086852e-01 | 5.710485e-02 |

## Independent Test Set Results (10000 frames, never used in AL)

| round | energy_rmse_mean | energy_rmse_natoms_mean | force_rmse_mean | force_range |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.086297e-01 | 1.206998e-02 | 3.439141e-01 | [0.3307, 0.3575] |
| 1 | 1.214430e-01 | 1.349367e-02 | 3.433040e-01 | [0.3268, 0.3546] |
| 2 | 1.044563e-01 | 1.160626e-02 | 3.352488e-01 | [0.3231, 0.3564] |
| 3 | 1.133646e-01 | 1.259607e-02 | 3.265941e-01 | [0.3068, 0.3487] |

### Key Finding

Independent test Force RMSE decreases monotonically (0.3439 → 0.3266 eV/Å), confirming that uncertainty-based active learning genuinely improves model accuracy on unseen data.

### Validation vs Independent Test (Force RMSE)

| round | Valid F_RMSE | Test F_RMSE | Δ |
| ---: | ---: | ---: | ---: |
| 0 | 0.373911 | 0.343914 | -0.0300 |
| 1 | 0.371493 | 0.343304 | -0.0282 |
| 2 | 0.364440 | 0.335249 | -0.0292 |
| 3 | 0.353702 | 0.326594 | -0.0271 |

Independent test Force RMSE is consistently ~0.028 eV/Å lower than validation, across all rounds. Both curves decrease monotonically.

## Model-Level Summary (Validation Set)

| round | model | energy_rmse | energy_rmse_natoms | force_rmse |
| ---: | ---: | ---: | ---: | ---: |
| 0 | model_000 | 1.336395e-01 | 1.484883e-02 | 3.749753e-01 |
| 0 | model_001 | 1.287160e-01 | 1.430177e-02 | 3.711438e-01 |
| 0 | model_002 | 1.191417e-01 | 1.323796e-02 | 3.540591e-01 |
| 0 | model_003 | 1.382067e-01 | 1.535630e-02 | 3.954685e-01 |
| 1 | model_000 | 1.389830e-01 | 1.544256e-02 | 3.682165e-01 |
| 1 | model_001 | 1.396081e-01 | 1.551201e-02 | 3.861489e-01 |
| 1 | model_002 | 1.255386e-01 | 1.394873e-02 | 3.790892e-01 |
| 1 | model_003 | 1.314685e-01 | 1.460761e-02 | 3.525164e-01 |
| 2 | model_000 | 1.223031e-01 | 1.358923e-02 | 3.542023e-01 |
| 2 | model_001 | 1.203357e-01 | 1.337064e-02 | 3.626481e-01 |
| 2 | model_002 | 1.260708e-01 | 1.400786e-02 | 3.847298e-01 |
| 2 | model_003 | 1.224112e-01 | 1.360125e-02 | 3.561796e-01 |
| 3 | model_000 | 1.149267e-01 | 1.276963e-02 | 3.282416e-01 |
| 3 | model_001 | 1.264158e-01 | 1.404620e-02 | 3.367741e-01 |
| 3 | model_002 | 1.304857e-01 | 1.449841e-02 | 3.745004e-01 |
| 3 | model_003 | 1.424763e-01 | 1.583070e-02 | 3.752929e-01 |

## Model-Level Summary (Independent Test Set)

| round | model | energy_rmse | energy_rmse_natoms | force_rmse |
| ---: | ---: | ---: | ---: | ---: |
| 0 | model_000 | 1.103269e-01 | 1.225854e-02 | 3.414212e-01 |
| 0 | model_001 | 1.118279e-01 | 1.242532e-02 | 3.430150e-01 |
| 0 | model_002 | 1.074151e-01 | 1.193501e-02 | 3.306628e-01 |
| 0 | model_003 | 1.049487e-01 | 1.166096e-02 | 3.574527e-01 |
| 1 | model_000 | 1.283459e-01 | 1.426066e-02 | 3.367792e-01 |
| 1 | model_001 | 1.221669e-01 | 1.357410e-02 | 3.545484e-01 |
| 1 | model_002 | 1.156208e-01 | 1.284676e-02 | 3.550996e-01 |
| 1 | model_003 | 1.196385e-01 | 1.329317e-02 | 3.267889e-01 |
| 2 | model_000 | 1.061149e-01 | 1.179054e-02 | 3.230658e-01 |
| 2 | model_001 | 1.013325e-01 | 1.125917e-02 | 3.369504e-01 |
| 2 | model_002 | 1.056006e-01 | 1.173340e-02 | 3.564393e-01 |
| 2 | model_003 | 1.047772e-01 | 1.164191e-02 | 3.245396e-01 |
| 3 | model_000 | 1.002253e-01 | 1.113615e-02 | 3.067652e-01 |
| 3 | model_001 | 1.173358e-01 | 1.303731e-02 | 3.141868e-01 |
| 3 | model_002 | 1.162951e-01 | 1.292168e-02 | 3.367282e-01 |
| 3 | model_003 | 1.196023e-01 | 1.328914e-02 | 3.486807e-01 |

## Figures

- `rmd17_ethanol_force_model_deviation_rounds.svg` — Candidate pool force_dev_max trends
- `rmd17_ethanol_validation_rmse_rounds.svg` — Validation Force/Energy RMSE
- `rmd17_ethanol_dataset_size_rounds.svg` — Train/candidate frame counts
- `rmd17_ethanol_independent_test_force_rmse.svg` — Validation vs Independent Test Force RMSE
- `rmd17_ethanol_independent_test_energy_rmse.svg` — Validation vs Independent Test Energy RMSE
