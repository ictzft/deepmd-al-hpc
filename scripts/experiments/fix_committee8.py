"""Finish committee_8 seeds 1-2 (round 1 only)."""
import subprocess, sys
from pathlib import Path

ROOT = Path("/data/guyida/deepmd-al-hpc")
DOCKER = "6a8531d2d15a"
VALID = "/data/guyida/deepmd-al-hpc/data/rmd17/ethanol/valid"

for seed in [1, 2]:
    rlabel = f"committee_8_seed{seed}_round001"
    print(f"\n=== {rlabel} ===")

    # Selection
    subprocess.run(["python3", "scripts/active_learning/select_from_predictions.py",
        "--predictions", "experiments/rmd17_ethanol_round000_committee_prediction/committee_predictions.npz",
        "--strategy", "uncertainty", "--top-k", "1000", "--seed", str(seed),
        "--output", f"experiments/baselines/{rlabel}_committee_prediction/selected_uncertainty.json"],
        cwd=ROOT, check=True)

    # Merge
    subprocess.run(["python3", "scripts/data/merge_selected_frames.py",
        "--train", "data/rmd17/ethanol/train", "--candidate", "data/rmd17/ethanol/candidate",
        "--selection", f"experiments/baselines/{rlabel}_committee_prediction/selected_uncertainty.json",
        "--output", f"data/rmd17/ethanol/train_round001_committee_8_seed{seed}", "--overwrite"],
        cwd=ROOT, check=True)

    # Candidate
    subprocess.run(["python3", "scripts/data/make_remaining_candidate.py",
        "--candidate", "data/rmd17/ethanol/candidate",
        "--selection", f"experiments/baselines/{rlabel}_committee_prediction/selected_uncertainty.json",
        "--output", f"data/rmd17/ethanol/candidate_round001_committee_8_seed{seed}", "--overwrite"],
        cwd=ROOT, check=True)

    # Configs
    subprocess.run(["python3", "scripts/config/make_round_committee_configs.py",
        "--base", "configs/deepmd/rmd17_ethanol_base.json",
        "--output-dir", f"configs/deepmd/rmd17_ethanol_committee_8_seed{seed}_round001_committee",
        "--train-system", f"data/rmd17/ethanol/train_round001_committee_8_seed{seed}",
        "--valid-system", "data/rmd17/ethanol/valid",
        "--round-id", "1", "--n-models", "8",
        "--base-seed", str((seed + 1) * 1000 + 100)], cwd=ROOT, check=True)

    # Train 8 models (2 at a time on 2 GPUs)
    model_dir = ROOT / f"experiments/baselines/{rlabel}_committee_models"
    cfg_dir = f"configs/deepmd/rmd17_ethanol_committee_8_seed{seed}_round001_committee"

    for batch in range(4):
        m0, m1 = batch * 2, batch * 2 + 1
        cmd = (f"mkdir -p {model_dir}/model_00{m0} {model_dir}/model_00{m1} && "
               f"CUDA_VISIBLE_DEVICES=0 dp train {cfg_dir}/toy_h2_round001_model_00{m0}.json && "
               f"dp freeze -o {model_dir}/model_00{m0}/frozen_model.pb && "
               f"dp test -m {model_dir}/model_00{m0}/frozen_model.pb -s {VALID} -n 0 > {model_dir}/model_00{m0}/test.log 2>&1 & "
               f"CUDA_VISIBLE_DEVICES=1 dp train {cfg_dir}/toy_h2_round001_model_00{m1}.json && "
               f"dp freeze -o {model_dir}/model_00{m1}/frozen_model.pb && "
               f"dp test -m {model_dir}/model_00{m1}/frozen_model.pb -s {VALID} -n 0 > {model_dir}/model_00{m1}/test.log 2>&1 & wait")
        subprocess.run(["docker", "exec", "-w", str(ROOT), DOCKER, "bash", "-lc",
            f"export PATH=/opt/deepmd-kit/bin:$PATH; export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:$LD_LIBRARY_PATH; {cmd}"], check=True)
        print(f"  batch {batch+1}/4 done (models {m0},{m1})")

    print(f"  {rlabel} DONE")

print("\nALL DONE")
