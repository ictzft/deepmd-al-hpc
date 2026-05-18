# Random Seed0 Round 001 Committee Metrics

Training data: `data/toy_h2/random_seed0_round_001_train`

| Model | Energy MAE / eV | Energy RMSE / eV | Force MAE / eV/Å | Force RMSE / eV/Å |
|---|---:|---:|---:|---:|
| model_000 | 8.811964e-01 | 8.813657e-01 | 1.298119e-01 | 2.551970e-01 |
| model_001 | 8.817706e-03 | 9.782892e-03 | 1.125556e-01 | 1.827423e-01 |
| model_002 | 1.965592e-01 | 1.965928e-01 | 5.685042e-02 | 8.939603e-02 |
| model_003 | 1.674010e+00 | 1.675800e+00 | 2.892547e-01 | 4.940109e-01 |
| **Mean** | 6.901458e-01 | 6.908853e-01 | 1.471182e-01 | 2.553366e-01 |
| **Std** | 7.554965e-01 | 7.559906e-01 | 9.974156e-02 | 1.729852e-01 |

Note: This is the first retrained random-sampling baseline with seed 0. The committee variance is relatively large, so results should be compared using mean/std and committee prediction behavior rather than a single model.