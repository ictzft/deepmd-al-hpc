#!/usr/bin/env python3
"""Run rMD17 ethanol ablation experiments: top-K or committee size."""
import subprocess, sys, time
from pathlib import Path

MODE = sys.argv[1]     # topk | committee
VALUE = sys.argv[2]    # 250|500|2000  or  2|8
SEEDS = [int(s) for s in sys.argv[3:]] if len(sys.argv) > 3 else [0, 1, 2]

ROOT = Path("/data/guyida/deepmd-al-hpc")
ROUND0_PRED = ROOT / "experiments/rmd17_ethanol_round000_committee_prediction/committee_predictions.npz"
ROUND0_TMPL = ROOT / "experiments/rmd17_ethanol_round000_committee_prediction/selected_uncertainty_1000.json"
BASE_CFG = ROOT / "configs/deepmd/rmd17_ethanol_base.json"
VALID = ROOT / "data/rmd17/ethanol/valid"
INIT_TRAIN = ROOT / "data/rmd17/ethanol/train"
INIT_CAND = ROOT / "data/rmd17/ethanol/candidate"
DOCKER = "6a8531d2d15a"

TOPK = int(VALUE) if MODE == "topk" else 1000
N_MODELS = int(VALUE) if MODE == "committee" else 4
LABEL = f"{MODE}_{VALUE}"

def run(cmd, docker=False):
    if docker:
        cmd = ["docker", "exec", "-w", str(ROOT), DOCKER, "bash", "-lc",
               f"export PATH=/opt/deepmd-kit/bin:$PATH; export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:$LD_LIBRARY_PATH; {' '.join(cmd)}"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr[-300:]}")
        raise RuntimeError("Command failed")
    return r.stdout

# Train models (handles variable N_MODELS)
def train_models(rlabel, rd, cfg_dir, model_dir):
    # Build model directories
    models = [model_dir / f"model_00{m}" for m in range(N_MODELS)]
    for md in models:
        md.mkdir(parents=True, exist_ok=True)

    # Train in batches of 2 (2 GPUs)
    for batch_start in range(0, N_MODELS, 2):
        batch_models = list(range(batch_start, min(batch_start + 2, N_MODELS)))
        if not batch_models:
            break

        # Start both GPUs
        procs = []
        for i, m in enumerate(batch_models):
            gpu = i  # GPU 0 or 1
            cfg = cfg_dir / f"toy_h2_round00{rd}_model_00{m}.json"
            model_d = model_dir / f"model_00{m}"
            cmd = (f"CUDA_VISIBLE_DEVICES={gpu} dp train {cfg} && "
                   f"dp freeze -o {model_d}/frozen_model.pb && "
                   f"dp test -m {model_d}/frozen_model.pb -s {VALID} -n 0 > {model_d}/test.log 2>&1")
            full_cmd = ["docker", "exec", "-w", str(ROOT), DOCKER, "bash", "-lc",
                        f"export PATH=/opt/deepmd-kit/bin:$PATH; export LD_LIBRARY_PATH=/opt/deepmd-kit/lib:$LD_LIBRARY_PATH; {cmd}"]
            procs.append(subprocess.Popen(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

        for p in procs:
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"Training failed for batch starting at model {batch_start}")

    # Verify
    for m in range(N_MODELS):
        pb = model_dir / f"model_00{m}/frozen_model.pb"
        if not pb.exists():
            raise RuntimeError(f"Missing frozen model: {pb}")

# Predict with variable N_MODELS
def predict(rlabel, rd, cand_dir, model_dir, pred_dir):
    model_paths = [f"{model_dir}/model_00{m}/frozen_model.pb" for m in range(N_MODELS)]
    # Use relative paths since we run inside Docker
    rel_model_paths = [f"experiments/baselines/{rlabel}_committee_models/model_00{m}/frozen_model.pb"
                       for m in range(N_MODELS)]
    rel_cand = f"data/rmd17/ethanol/candidate_round00{rd}_{LABEL}_seed{seed}"
    rel_pred = f"experiments/baselines/{rlabel}_committee_prediction"

    cmd = (f"python3 scripts/inference/predict_committee_models.py "
           f"--data {rel_cand} "
           f"--models {' '.join(rel_model_paths)} "
           f"--output {rel_pred}/committee_predictions.npz "
           f"--top-k {TOPK}")
    run(cmd.split(), docker=True)

# --- Main ---
for seed in SEEDS:
    pred_npz = ROUND0_PRED

    for rd in [1, 2, 3]:
        rlabel = f"{LABEL}_seed{seed}_round00{rd}"
        prev_rd = rd - 1
        t0 = time.time()
        print(f"\n{'='*60}\n  {rlabel}  ({MODE}={VALUE}, seed={seed}, round={rd})\n{'='*60}")

        # 1. Selection
        sel_dir = ROOT / f"experiments/baselines/{rlabel}_committee_prediction"
        sel_dir.mkdir(parents=True, exist_ok=True)
        sel_json = sel_dir / "selected_uncertainty.json"
        run(["python3", "scripts/active_learning/select_from_predictions.py",
             "--predictions", str(pred_npz), "--strategy", "uncertainty",
             "--top-k", str(TOPK), "--seed", str(seed),
             "--template-json", str(ROUND0_TMPL), "--output", str(sel_json)])

        # 2. Merge
        prev_train = INIT_TRAIN if rd == 1 else ROOT / f"data/rmd17/ethanol/train_round00{prev_rd}_{LABEL}_seed{seed}"
        cand_dir = INIT_CAND if rd == 1 else ROOT / f"data/rmd17/ethanol/candidate_round00{prev_rd}_{LABEL}_seed{seed}"
        train_dir = ROOT / f"data/rmd17/ethanol/train_round00{rd}_{LABEL}_seed{seed}"
        run(["python3", "scripts/data/merge_selected_frames.py",
             "--train", str(prev_train), "--candidate", str(cand_dir),
             "--selection", str(sel_json), "--output", str(train_dir), "--overwrite"])

        # 3. Candidate
        new_cand = ROOT / f"data/rmd17/ethanol/candidate_round00{rd}_{LABEL}_seed{seed}"
        run(["python3", "scripts/data/make_remaining_candidate.py",
             "--candidate", str(cand_dir), "--selection", str(sel_json),
             "--output", str(new_cand), "--overwrite"])

        # 4. Configs (variable N_MODELS)
        cfg_dir = ROOT / f"configs/deepmd/rmd17_ethanol_{LABEL}_seed{seed}_round00{rd}_committee"
        run(["python3", "scripts/config/make_round_committee_configs.py",
             "--base", str(BASE_CFG), "--output-dir", str(cfg_dir),
             "--train-system", str(train_dir), "--valid-system", str(VALID),
             "--round-id", str(rd), "--n-models", str(N_MODELS),
             "--base-seed", str((seed + 1) * 1000 + rd * 100)])

        # 5. Train (variable N_MODELS)
        model_dir = ROOT / f"experiments/baselines/{rlabel}_committee_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Training {N_MODELS} models...")
        train_models(rlabel, rd, cfg_dir, model_dir)

        # 6. Predict
        print(f"  Predicting with {N_MODELS} models...")
        predict(rlabel, rd, new_cand, model_dir, sel_dir)

        pred_npz = ROOT / f"experiments/baselines/{rlabel}_committee_prediction/committee_predictions.npz"
        print(f"  DONE in {time.time()-t0:.0f}s")

print(f"\nALL DONE: {LABEL} ablation complete")
