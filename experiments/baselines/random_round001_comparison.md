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

## Candidate pool uncertainty after Round001 retraining

| Branch | candidate_dev_mean | candidate_energy_dev_mean | candidate_dev_std |
|---|---:|---:|---:|
| uncertainty | 0.269333 | 0.447755 | — |
| random_seed0 | 0.838753 | 0.687085 | — |
| random_seed1 | 0.487795 | 0.396726 | — |
| random_seed2 | 0.332138 | 0.446260 | — |
| **random mean** | **0.552896** | **0.510024** | **0.259506** |

## Notes

- In Round001, the uncertainty branch yields a lower remaining candidate-pool `candidate_dev_mean` (0.269) than all three random seeds (seed0: 0.839, seed1: 0.488, seed2: 0.332), with random mean 0.553.
- This observation is consistent across seed0/seed1/seed2 for this specific metric.
- The random multi-seed mean and std provide a baseline for comparison.
- This is still a **toy H2 workflow validation** and does not represent realistic first-principles material systems.
- Full random Round002/Round003 retraining and realistic DFT/AIMD datasets are needed before drawing general conclusions.
- This file is a lightweight companion to `experiments/baselines/random_round001_comparison.csv`.
