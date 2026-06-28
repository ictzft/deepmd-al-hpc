#!/usr/bin/env python3
"""Run rMD17 benzene strategy baseline (diversity or trust-level)."""
import subprocess, sys, time
from pathlib import Path


def run(cmd, docker=False, root=None, docker_id=None):
    if docker:
        cmd = ["docker", "exec", "-w", str(root), docker_id, "bash", "-lc",
               f"export PATH=/opt/deepmd-kit/bin:$PATH; export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:$LD_LIBRARY_PATH; {' '.join(cmd)}"]
    print(f"  RUN: {' '.join(cmd)[:120]}")
    r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  STDERR: {r.stderr[-500:]}")
        raise RuntimeError(f"Failed: {r.stderr[-200:]}")
    return r.stdout


def main():
    strategy = sys.argv[1]
    if strategy not in ("diversity", "trust-level"):
        print(f"ERROR: unknown strategy '{strategy}'. Supported: diversity, trust-level")
        sys.exit(1)

    label = strategy.replace("-", "_")  # trust-level -> trust_level for file paths
    seeds = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else [0, 1, 2]

    ROOT = Path("/data/guyida/deepmd-al-hpc")
    ROUND0_PRED = ROOT / "experiments/rmd17_benzene_round000_committee_prediction/committee_predictions.npz"
    ROUND0_TMPL = ROOT / "experiments/rmd17_benzene_round000_committee_prediction/selected_uncertainty_1000.json"
    BASE_CFG = ROOT / "configs/deepmd/rmd17_benzene_base.json"
    VALID = ROOT / "data/rmd17/benzene/valid"
    INIT_TRAIN = ROOT / "data/rmd17/benzene/train"
    INIT_CAND = ROOT / "data/rmd17/benzene/candidate"
    DOCKER = "6a8531d2d15a"
    TOPK = 1000

    extra_sel = (["--top-m", "3000"] if label == "diversity" else
                 ["--low-pct", "50", "--high-pct", "90", "--candidate-ratio", "0.8"])

    for seed in seeds:
        base_seed = (seed + 1) * 1000 + 500
        pred_npz = ROUND0_PRED

        for rd in [1, 2, 3]:
            rlabel = f"rmd17_benzene_{label}_seed{seed}_round00{rd}"
            prev_rd = rd - 1
            t0 = time.time()
            print(f"\n{'='*60}\n  {rlabel.upper()}\n{'='*60}")

            # 1. Selection
            sel_dir = ROOT / f"experiments/baselines/{rlabel}_committee_prediction"
            sel_dir.mkdir(parents=True, exist_ok=True)
            sel_json = sel_dir / f"selected_{label}.json"
            run(["python3", "scripts/active_learning/select_from_predictions.py",
                 "--predictions", str(pred_npz), "--strategy", strategy,
                 "--top-k", str(TOPK), "--seed", str(seed),
                 *extra_sel, "--template-json", str(ROUND0_TMPL),
                 "--output", str(sel_json)], root=ROOT)

            # 2. Merge
            prev_train = INIT_TRAIN if rd == 1 else ROOT / f"data/rmd17/benzene/train_round00{prev_rd}_{label}_seed{seed}"
            cand_dir = INIT_CAND if rd == 1 else ROOT / f"data/rmd17/benzene/candidate_round00{prev_rd}_{label}_seed{seed}"
            train_dir = ROOT / f"data/rmd17/benzene/train_round00{rd}_{label}_seed{seed}"
            run(["python3", "scripts/data/merge_selected_frames.py",
                 "--train", str(prev_train), "--candidate", str(cand_dir),
                 "--selection", str(sel_json), "--output", str(train_dir), "--overwrite"],
                root=ROOT)

            # 3. Candidate
            new_cand = ROOT / f"data/rmd17/benzene/candidate_round00{rd}_{label}_seed{seed}"
            run(["python3", "scripts/data/make_remaining_candidate.py",
                 "--candidate", str(cand_dir), "--selection", str(sel_json),
                 "--output", str(new_cand), "--overwrite"], root=ROOT)

            # 4. Configs
            cfg_dir = ROOT / f"configs/deepmd/rmd17_benzene_{label}_seed{seed}_round00{rd}_committee"
            run(["python3", "scripts/config/make_round_committee_configs.py",
                 "--base", str(BASE_CFG), "--output-dir", str(cfg_dir),
                 "--train-system", str(train_dir), "--valid-system", str(VALID),
                 "--round-id", str(rd), "--n-models", "4",
                 "--base-seed", str(base_seed + rd * 100)], root=ROOT)

            # 5. Train (Docker)
            rel_cfg = f"configs/deepmd/rmd17_benzene_{label}_seed{seed}_round00{rd}_committee"
            rel_model = f"experiments/baselines/{rlabel}_committee_models"
            run(["bash", "scripts/train/train_round_committee_models.sh",
                 f"{rd:03d}", rel_cfg, rel_model, str(VALID)],
                docker=True, root=ROOT, docker_id=DOCKER)

            model_dir = ROOT / f"experiments/baselines/{rlabel}_committee_models"
            for m in range(4):
                if not (model_dir / f"model_00{m}/frozen_model.pb").exists():
                    raise RuntimeError(f"Missing: model_00{m}/frozen_model.pb")

            # 6. Predict (Docker)
            rel_cand = f"data/rmd17/benzene/candidate_round00{rd}_{label}_seed{seed}"
            rel_pred = f"experiments/baselines/{rlabel}_committee_prediction"
            run(["python3", "scripts/inference/predict_committee_models.py",
                 "--data", rel_cand,
                 "--models",
                 f"{rel_model}/model_000/frozen_model.pb",
                 f"{rel_model}/model_001/frozen_model.pb",
                 f"{rel_model}/model_002/frozen_model.pb",
                 f"{rel_model}/model_003/frozen_model.pb",
                 "--output", f"{rel_pred}/committee_predictions.npz",
                 "--top-k", str(TOPK)],
                docker=True, root=ROOT, docker_id=DOCKER)

            pred_npz = ROOT / f"experiments/baselines/{rlabel}_committee_prediction/committee_predictions.npz"
            print(f"  {rlabel} DONE in {time.time()-t0:.0f}s")

    print(f"\nALL DONE: {label} baseline ({len(seeds)} seeds x 3 rounds)")


if __name__ == "__main__":
    main()
