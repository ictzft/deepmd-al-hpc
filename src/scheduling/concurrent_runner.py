#!/usr/bin/env python
"""Concurrent committee training runner (model-level parallelism).

Trains N committee models on G GPUs: each model occupies one GPU; models run
in ceil(N/G) waves. This is the core infrastructure for the multi-GPU strong
scaling experiments (CCGrid 2027 roadmap Fig.3/Fig.4).

Designed to run INSIDE the deepmd-5090 container (launched via
``scripts/docker/run_in_5090.sh``), where ``dp`` is on PATH and G GPUs are
visible. GPU utilization curves are recorded separately by an external
``nvidia-smi dmon`` process (see scripts/scaling/); this script focuses on
correct concurrent scheduling and per-model wall-time accounting.

Usage (inside container, or via run_in_5090.sh):

    bash scripts/docker/run_in_5090.sh 0,1,2,3 -- python \\
        src/scheduling/concurrent_runner.py \\
        --n-models 8 --n-gpus 4 \\
        --config-template configs/deepmd/pt_smoke_toy_h2.json \\
        --train-system /data/zft/deepmd-al-hpc/data/toy_h2/train \\
        --valid-system /data/zft/deepmd-al-hpc/data/toy_h2/valid \\
        --exp-dir experiments/scaling/smoke_n8_g4 \\
        --numb-steps 100

Outputs:
    <exp-dir>/configs/model_XXX.json   per-model configs (varied seed)
    <exp-dir>/model_XXX/{train.log, lcurve.out, model.ckpt.pt}
    <exp-dir>/summary.json             per-model + per-wave + total wall-time
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def make_model_config(base: dict, model_id: int, base_seed: int,
                      train_sys: str, valid_sys: str, numb_steps: int,
                      batch_size: int, out_dir: Path) -> Path:
    """Generate a per-model config from the base template with a distinct seed.

    Deep copy the base config, bump descriptor/fitting_net/training seeds by
    model_id (so committee members differ only in initialization), and force
    the (container-internal) data paths.
    """
    cfg = json.loads(json.dumps(base))  # deep copy
    seed = base_seed + model_id
    cfg.setdefault("model", {}).setdefault("descriptor", {})["seed"] = seed
    cfg["model"].setdefault("fitting_net", {})["seed"] = seed
    cfg.setdefault("training", {})["seed"] = seed
    cfg["training"]["training_data"]["systems"] = [train_sys]
    cfg["training"]["validation_data"]["systems"] = [valid_sys]
    if numb_steps and numb_steps > 0:
        cfg["training"]["numb_steps"] = numb_steps
    if batch_size and batch_size > 0:
        cfg["training"]["training_data"]["batch_size"] = batch_size
        cfg["training"]["validation_data"]["batch_size"] = batch_size
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"model_{model_id:03d}.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def _launch_model(model_id, gpu_id, cfg_path, exp_dir, backend, tag="wave"):
    """Launch one `dp train` on a specific GPU. Returns (p, start, logf, log_path)."""
    model_dir = exp_dir / f"model_{model_id:03d}"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "train.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = ["dp", "-b", backend, "train", str(cfg_path)]
    logf = open(log_path, "w")
    p = subprocess.Popen(cmd, cwd=str(model_dir), env=env,
                         stdout=logf, stderr=subprocess.STDOUT)
    print(f"  [{tag}] model_{model_id:03d} -> GPU {gpu_id} (pid {p.pid})")
    return model_id, gpu_id, p, time.time(), str(log_path), logf


def _collect(launched, tag="wave"):
    """Wait for a list of launched procs and return result dicts."""
    results = []
    for model_id, gpu_id, p, start, log_path, logf in launched:
        rc = p.wait()
        logf.close()
        wall = time.time() - start
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        results.append({
            "model_id": model_id,
            "gpu_id": gpu_id,
            "wall_s": round(wall, 3),
            "rc": rc,
            "log": log_path,
        })
        print(f"  [{tag}] model_{model_id:03d} done: {wall:.1f}s {status}")
    return results


def run_wave(wave_jobs, exp_dir: Path, backend: str):
    """Run one wave concurrently.

    ``wave_jobs`` is a list of ``(model_id, cfg_path)`` with len <= n_gpus.
    Within a wave, model i runs on GPU i (CUDA_VISIBLE_DEVICES=i), so the
    G visible GPUs are filled one model each.
    """
    launched = [_launch_model(mid, gid, cfg, exp_dir, backend, "wave")
                for gid, (mid, cfg) in enumerate(wave_jobs)]
    return _collect(launched, "wave")


def run_all_concurrent(jobs, n_gpus: int, exp_dir: Path, backend: str):
    """Launch ALL N models concurrently; model i runs on GPU (i % n_gpus).

    When n_gpus < n_models, multiple models share each GPU. Without MPS this is
    GPU time-slicing (still serialized on the GPU); with NVIDIA MPS enabled the
    processes run concurrently on the same GPU, which is what the CCGrid MPS
    experiment compares against the wave (one-model-per-GPU) baseline.
    """
    launched = [_launch_model(mid, mid % n_gpus, cfg, exp_dir, backend, "all")
                for mid, cfg in jobs]
    return _collect(launched, "all")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-models", type=int, required=True,
                    help="committee size (number of models to train)")
    ap.add_argument("--n-gpus", type=int, required=True,
                    help="number of GPUs visible to the container (wave size)")
    ap.add_argument("--config-template", required=True,
                    help="base deepmd config json (data paths will be overridden)")
    ap.add_argument("--train-system", required=True,
                    help="container-internal path to training system")
    ap.add_argument("--valid-system", required=True,
                    help="container-internal path to validation system")
    ap.add_argument("--exp-dir", required=True, help="experiment output dir")
    ap.add_argument("--numb-steps", type=int, default=0,
                    help="override numb_steps (0 = keep template value)")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="override training/validation batch_size (0 = keep template)")
    ap.add_argument("--base-seed", type=int, default=2026)
    ap.add_argument("--mode", choices=["wave", "all"], default="wave",
                    help="wave = ceil(N/G) batches, one model per GPU (default); "
                         "all = all N models concurrent, model i on GPU i%G "
                         "(needs MPS for real GPU concurrency when N>G)")
    ap.add_argument("--backend", default="pytorch",
                    help="deepmd backend flag passed to dp (-b)")
    args = ap.parse_args()

    if not (1 <= args.n_gpus <= 64):
        sys.exit("--n-gpus must be >= 1")
    if args.n_models < 1:
        sys.exit("--n-models must be >= 1")

    base = json.loads(Path(args.config_template).read_text())
    exp_dir = Path(args.exp_dir).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = exp_dir / "configs"

    cfg_paths = [
        make_model_config(base, i, args.base_seed, args.train_system,
                          args.valid_system, args.numb_steps, args.batch_size, cfg_dir)
        for i in range(args.n_models)
    ]
    jobs = list(enumerate(cfg_paths))  # (model_id, cfg_path)

    t0 = time.time()
    all_results = []
    if args.mode == "all":
        waves = [jobs]  # placeholder so summary's len(waves) == 1
        print(f"=== concurrent_runner [all]: {args.n_models} models concurrent "
              f"on {args.n_gpus} GPU(s) (shared) ===")
        res = run_all_concurrent(jobs, args.n_gpus, exp_dir, args.backend)
        for r in res:
            r["wave"] = 1
        all_results.extend(res)
    else:
        waves = [jobs[i:i + args.n_gpus]
                 for i in range(0, args.n_models, args.n_gpus)]
        print(f"=== concurrent_runner [wave]: {args.n_models} models, "
              f"{args.n_gpus} GPUs, {len(waves)} wave(s) ===")
        for wi, wave in enumerate(waves):
            t_wave = time.time()
            print(f"--- wave {wi + 1}/{len(waves)} ({len(wave)} model(s)) ---")
            res = run_wave(wave, exp_dir, args.backend)
            for r in res:
                r["wave"] = wi + 1
            all_results.extend(res)
            print(f"    wave {wi + 1} wall = {time.time() - t_wave:.1f}s")
    t_total = time.time() - t0

    n_ok = sum(1 for r in all_results if r["rc"] == 0)
    mean_wall = sum(r["wall_s"] for r in all_results) / len(all_results)
    summary = {
        "n_models": args.n_models,
        "n_gpus": args.n_gpus,
        "n_waves": len(waves),
        "total_wall_s": round(t_total, 3),
        "throughput_models_per_s": round(args.n_models / t_total, 5),
        "mean_model_wall_s": round(mean_wall, 3),
        "n_ok": n_ok,
        "n_fail": args.n_models - n_ok,
        "models": all_results,
    }
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== DONE: total {t_total:.1f}s, {n_ok}/{args.n_models} OK, "
          f"mean {mean_wall:.1f}s/model, "
          f"throughput {summary['throughput_models_per_s']:.4f} models/s ===")
    print(f"summary -> {exp_dir / 'summary.json'}")
    if n_ok != args.n_models:
        sys.exit(1)


if __name__ == "__main__":
    main()
