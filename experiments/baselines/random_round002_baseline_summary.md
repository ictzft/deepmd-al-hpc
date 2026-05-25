# Random Round002 Baseline Summary

## Multi-seed retraining metrics (Round 002)

| Seed | Train frames | Candidate frames | Energy RMSE mean / eV | Energy RMSE std / eV | Force RMSE mean / eV/Å | Force RMSE std / eV/Å |
|---|---:|---:|---:|---:|---:|---:|
| seed0 | 220 | 30 | 1.977364e+00 | 1.330685e+00 | 2.120793e-01 | 1.907285e-02 |
| seed1 | 220 | 30 | 1.847642e+00 | 1.213226e+00 | 1.796397e-01 | 6.499082e-02 |
| seed2 | 220 | 30 | 9.067673e-01 | 1.162607e+00 | 1.968315e-01 | 5.614654e-02 |
| mean | 220 | 30 | 1.577258e+00 | 5.842730e-01 | 1.961835e-01 | 1.622950e-02 |

## Candidate pool uncertainty after Round 002 retraining

| Seed | force_dev_max mean | force_dev_max max | force_dev_max min | energy_dev mean |
|---|---:|---:|---:|---:|
| seed0 | 1.362240e-01 | 3.035045e-01 | 4.961156e-02 | 1.155043e+00 |
| seed1 | 1.219434e-01 | 2.329508e-01 | 4.664612e-02 | 1.051151e+00 |
| seed2 | 1.611978e-01 | 3.861262e-01 | 4.946092e-02 | 1.027315e+00 |
| mean | 1.397884e-01 | — | — | — |

## Notes

- Random Round002 multi-seed baseline (seed0/seed1/seed2).
- Cross-seed mean and std are reported for energy_rmse, force_rmse, and candidate force_dev_max.
- This is still a toy H2 workflow validation and does not represent realistic first-principles material systems.
