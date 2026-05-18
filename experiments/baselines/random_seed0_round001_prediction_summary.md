# Random Seed0 Round 001 Committee Prediction Summary

This file summarizes the committee prediction result after retraining with the random seed0 selected frames.

## Compared Runs

| Run | Candidate Pool | n_frames | force_dev_max mean | force_dev_max max | force_dev_max min | energy_dev mean | energy_dev max | energy_dev min |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uncertainty_round001 | data/toy_h2/round_001_candidate | 40 | 0.12644162 | 0.50833874 | 0.04264456 | 0.44821195 | 0.45067376 | 0.44584389 |
| random_seed0_round001 | data/toy_h2/random_seed0_round_001_candidate | 40 | 0.35542039 | 1.58635493 | 0.08666701 | 0.65654074 | 0.81034517 | 0.64205221 |

## Observation

After adding 10 randomly selected frames and retraining the committee models, the remaining candidate pool still shows a higher average force model deviation than the uncertainty-sampling branch.

This suggests that uncertainty sampling is more effective than this random seed0 baseline at reducing candidate-pool uncertainty in the toy H2 offline active learning setup.

Note: This is still a toy-system result and should be extended to more random seeds and realistic DFT/AIMD datasets.
