# Round 3 Committee Models

This experiment retrains four DeePMD committee models after adding the top-10 selected high-uncertainty frames from Round 2.

- Round: 3
- Training frames: 230
- Committee size: 4
- Platform: 2 x V100
- Recorded committee training elapsed time: 76 s
- Heavy outputs such as checkpoints, frozen models, logs, and numerical arrays are excluded from Git.

The model-level test metrics are summarized in:

- `../al_model_level_summary.csv`
- `../al_rounds_summary.md`
