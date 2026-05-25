# Random Round001 Baseline Summary

## Multi-seed retraining metrics (Round 001)

| Seed | Train frames | Candidate frames | Energy RMSE mean / eV | Energy RMSE std / eV | Force RMSE mean / eV/Å | Force RMSE std / eV/Å |
|---|---:|---:|---:|---:|---:|---:|
| seed0 | 210 | 40 | 6.908853e-01 | 7.559906e-01 | 2.553366e-01 | 1.729852e-01 |
| seed1 | 210 | 40 | 2.488029e-01 | 4.563935e-01 | 2.288370e-01 | 3.565438e-02 |
| seed2 | 210 | 40 | 4.284121e-01 | 5.054377e-01 | 1.494923e-01 | 4.669047e-02 |
| mean | 210 | 40 | 4.560334e-01 | 2.223318e-01 | 2.112220e-01 | 5.507694e-02 |

## Candidate pool uncertainty after Round 001 retraining

| Seed | force_dev_max mean | force_dev_max max | force_dev_max min | energy_dev mean |
|---|---:|---:|---:|---:|
| seed0 | 3.554204e-01 | 1.586355e+00 | 8.666701e-02 | 6.565407e-01 |
| seed1 | 4.877953e-01 | 1.262038e+00 | 3.274831e-01 | 3.967256e-01 |
| seed2 | 3.321379e-01 | 1.117230e+00 | 1.393214e-01 | 4.462596e-01 |
| mean | 3.917845e-01 | — | — | — |

## Notes

- Current random Round001 baseline has been extended from single-seed (seed0) to multi-seed (seed0 / seed1 / seed2).
- The cross-seed mean and std are reported for energy_rmse, force_rmse, and candidate force_dev_max.
- This is still a toy H2 workflow validation and does not represent realistic first-principles material systems.
- The next step is to complete random Round002 / Round003 retraining for a full multi-round learning curve.
