# Round 1 Committee Models

This experiment retrains four DeePMD committee models after adding the top-10 selected high-uncertainty frames from Round 0.

- Round: 1
- Training frames: 210
- Committee size: 4
- Platform: 2 x V100
- Heavy outputs such as checkpoints, frozen models, logs, and numerical arrays are excluded from Git.

The model-level test metrics are summarized in:

- `../al_model_level_summary.csv`
- `../al_rounds_summary.md`
