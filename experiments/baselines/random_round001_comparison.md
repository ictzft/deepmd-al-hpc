# Random Round001 Comparison

Uncertainty branch vs random seed0/seed1/seed2 Round001 retraining comparison.

## Retraining metrics

| Branch | Energy RMSE mean / eV | Energy RMSE std / eV | Force RMSE mean / eV/A | Force RMSE std / eV/A |
|---|---:|---:|---:|---:|
| uncertainty | 0.729295 | 0.516356 | 0.161767 | 0.074440 |
| random_seed0 | 0.690885 | 0.755991 | 0.255337 | 0.172985 |
| random_seed1 | 0.248803 | 0.456393 | 0.228837 | 0.035654 |
| random_seed2 | 0.428412 | 0.505438 | 0.149492 | 0.046690 |
| **random mean** | **0.456033** | — | **0.211222** | — |
| **random std** | **0.222332** | — | **0.055077** | — |

Note: The uncertainty branch Energy RMSE mean (0.729) is higher than the random mean (0.456). This reflects the fact that uncertainty sampling selects high-uncertainty frames, which does not guarantee lower Energy RMSE on the validation set. The primary benefit of uncertainty sampling is in reducing candidate-pool model deviation, as shown below.

## Candidate-pool / selected uncertainty after Round001 retraining

| Branch | candidate_force_dev_max_mean | candidate_energy_dev_mean | candidate_force_dev_max_std |
|---|---:|---:|---:|
| uncertainty | 0.269333 | 0.447755 | — |
| random_seed0 | 0.838753 | 0.687085 | — |
| random_seed1 | 0.487795 | 0.396726 | — |
| random_seed2 | 0.332138 | 0.446260 | — |
| **random mean** | **0.552896** | **0.510024** | **0.259506** |

**Field definitions:**

- `candidate_force_dev_max_mean`: For random branches, this is the mean `force_dev_max` of the remaining candidate pool after Round 001 retraining (from `random_seed*_round001_prediction_summary.csv`). For the uncertainty branch, this value (0.269333) is the mean `force_dev_max` of the 10 top-K selected frames from Round 1 committee prediction (from `al_rounds_summary.csv`), **not** the remaining candidate-pool value. The remaining candidate-pool `force_dev_max_mean` for uncertainty_round001 is 0.126442 (see `random_seed0_round001_prediction_summary.csv`).
- `candidate_energy_dev_mean`: Mean `energy_dev` of the remaining candidate pool (for random) or selected top-K frames (for uncertainty), matching the same source as `candidate_force_dev_max_mean`.
- `candidate_force_dev_max_std`: Cross-seed standard deviation of `candidate_force_dev_max_mean` across seed0/seed1/seed2 (random_mean row only).

**Important:** This comparison table mixes two different metrics:
- For **random** branches: `candidate_force_dev_max_mean` = remaining candidate-pool uncertainty after Round 001 retraining
- For **uncertainty** branch: `candidate_force_dev_max_mean` = top-K selected frames' force_dev_max from Round 1 (NOT remaining candidate-pool)

For a fair comparison of remaining candidate-pool uncertainty, see `random_seed0_round001_prediction_summary.csv` which includes the uncertainty_round001 remaining candidate-pool value (force_dev_max_mean = 0.126442) alongside random_seed0.

## Notes

- The uncertainty branch selected top-K `force_dev_max_mean` (0.269) is reported here for the uncertainty row.
- The remaining candidate-pool `force_dev_max_mean` for uncertainty_round001 is 0.126442, which is lower than all three random branches (seed0: 0.355, seed1: 0.488, seed2: 0.332).
- This observation is consistent across seed0/seed1/seed2 for the remaining candidate-pool comparison.
- The random multi-seed mean and std provide a baseline for comparison.
- This is still a **toy H2 workflow validation** and does not represent realistic first-principles material systems.
- Full random Round002/Round003 retraining and realistic DFT/AIMD datasets are needed before drawing general conclusions.
- This file is a lightweight companion to `experiments/baselines/random_round001_comparison.csv`.
