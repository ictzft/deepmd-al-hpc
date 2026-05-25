# Random Round003 Baseline Summary

## Multi-seed retraining metrics (Round 003)

| Seed | Train frames | Candidate frames | Energy RMSE mean / eV | Energy RMSE std / eV | Force RMSE mean / eV/Å | Force RMSE std / eV/Å |
|---|---:|---:|---:|---:|---:|---:|
| seed0 | 230 | 20 | 1.369364e+00 | 1.044380e+00 | 2.373825e-01 | 7.212040e-02 |
| seed1 | 230 | 20 | 1.707502e+00 | 1.136900e+00 | 1.417639e-01 | 5.592467e-02 |
| seed2 | 230 | 20 | 1.310121e+00 | 9.747605e-01 | 1.879388e-01 | 1.007966e-01 |
| mean | 230 | 20 | 1.462329e+00 | 2.143827e-01 | 1.890284e-01 | 4.781861e-02 |

## Candidate pool uncertainty after Round 003 retraining

| Seed | force_dev_max mean | force_dev_max max | force_dev_max min | energy_dev mean |
|---|---:|---:|---:|---:|
| seed0 | 1.578821e-01 | 2.274971e-01 | 4.174930e-02 | 1.493153e+00 |
| seed1 | 1.352771e-01 | 2.358352e-01 | 2.511937e-02 | 1.823728e+00 |
| seed2 | 1.239594e-01 | 1.967203e-01 | 7.834327e-02 | 1.492024e+00 |
| mean | 1.390395e-01 | — | — | — |

## Notes

- Random Round003 multi-seed baseline (seed0/seed1/seed2).
- Cross-seed mean and std are reported for energy_rmse, force_rmse, and candidate force_dev_max.
- This is still a toy H2 workflow validation and does not represent realistic first-principles material systems.
