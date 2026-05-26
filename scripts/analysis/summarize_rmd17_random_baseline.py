"""Summarize rMD17 ethanol random baseline results and compare with uncertainty."""

import csv, json, re
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("/data/guyida/deepmd-al-hpc")
OUT_DIR = ROOT / "experiments/rmd17_ethanol_summary"


def parse_test_log(path):
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    def last_float(pattern):
        vals = re.findall(pattern, text)
        return float(vals[-1]) if vals else None
    return {
        "energy_rmse": last_float(r"Energy RMSE\s*:\s*([0-9.eE+-]+)\s*eV"),
        "energy_rmse_natoms": last_float(r"Energy RMSE/Natoms\s*:\s*([0-9.eE+-]+)\s*eV"),
        "force_rmse": last_float(r"Force\s+RMSE\s*:\s*([0-9.eE+-]+)\s*eV/"),
    }


def parse_prediction(selected_json):
    if not selected_json.exists():
        # Try alternative filenames
        for alt in [selected_json.parent / "selected_uncertainty_1000.json",
                     selected_json.parent / "selected_topk.json"]:
            if alt.exists():
                selected_json = alt
                break
        else:
            return {}
    data = json.loads(selected_json.read_text(errors="replace"))
    selected = data.get("selected_frames", [])
    force_vals = [float(x["force_dev_max"]) for x in selected if "force_dev_max" in x]
    return {
        "n_frames": data.get("n_frames"),
        "top_k": data.get("top_k", len(selected)),
        "force_dev_max_mean": mean(force_vals) if force_vals else None,
        "energy_dev_mean": mean([float(x["energy_dev"]) for x in selected if "energy_dev" in x]) if selected else None,
    }


def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6e}"
    return str(x)


def main():
    # Collect results
    seeds = [0, 1, 2]
    rounds = [1, 2, 3]

    random_rounds = {}  # round -> {force_rmse: [...], energy_rmse: [...], ...}
    model_rows = []

    for s in seeds:
        for r in rounds:
            label = f"random_seed{s}_round00{r}"
            model_dir = ROOT / f"experiments/baselines/{label}_committee_models"
            pred_dir = ROOT / f"experiments/baselines/{label}_committee_prediction"

            if not model_dir.exists():
                print(f"SKIP seed={s} round={r} (no model dir)")
                continue

            # Parse models
            frmses, ermses = [], []
            for mp in sorted(model_dir.glob("model_*")):
                if not mp.is_dir():
                    continue
                metrics = parse_test_log(mp / "test.log")
                if metrics:
                    model_rows.append({"seed": s, "round": r, "model": mp.name, **metrics})
                    frmses.append(metrics["force_rmse"])
                    ermses.append(metrics["energy_rmse"])

            # Parse prediction
            pred = parse_prediction(pred_dir / "selected_random.json")

            if r not in random_rounds:
                random_rounds[r] = {"force": [], "energy": [], "force_dev": [], "energy_dev": []}
            if frmses:
                random_rounds[r]["force"].append(mean(frmses))
                random_rounds[r]["energy"].append(mean(ermses))
            if pred.get("force_dev_max_mean"):
                random_rounds[r]["force_dev"].append(pred["force_dev_max_mean"])
                random_rounds[r]["energy_dev"].append(pred["energy_dev_mean"])

    # Load uncertainty baseline for comparison
    uncertainty_csv = ROOT / "experiments/rmd17_ethanol_summary/al_rounds_summary.csv"
    unc_data = {}
    if uncertainty_csv.exists():
        with uncertainty_csv.open() as f:
            for row in csv.DictReader(f):
                r = int(row["round"])
                if r == 0:
                    continue  # Round 0 is same for both
                unc_data[r] = row

    # Generate round-level summary
    print("=== rMD17 Ethanol Random Baseline Summary ===\n")
    print(f"{'Round':<7} {'Strategy':<14} {'F_RMSE_mean':<15} {'F_RMSE_std':<12} {'E_RMSE_mean':<15} {'force_dev_max':<15}")
    print("-" * 90)

    round_fields = ["round", "strategy", "force_rmse_mean", "force_rmse_std",
                    "energy_rmse_mean", "force_dev_max_mean"]
    round_rows = []

    for r in [1, 2, 3]:
        rd = random_rounds.get(r, {})
        rf = rd.get("force", [])
        re = rd.get("energy", [])
        rfd = rd.get("force_dev", [])

        # Random
        if rf:
            f_mean, f_std = mean(rf), stdev(rf) if len(rf) > 1 else 0
            e_mean = mean(re) if re else 0
            fd_mean = mean(rfd) if rfd else 0
            print(f"{r:<7} {'random':<14} {f_mean:.6e}    {f_std:.4e}     {e_mean:.6e}    {fd_mean:.6e}")
            round_rows.append({"round": r, "strategy": "random",
                               "force_rmse_mean": f_mean, "force_rmse_std": f_std,
                               "energy_rmse_mean": e_mean, "force_dev_max_mean": fd_mean})

        # Uncertainty (from stored CSV)
        if r in unc_data:
            u = unc_data[r]
            uf = float(u["force_rmse_mean"])
            ue = float(u["energy_rmse_mean"])
            ufd = float(u["force_dev_max_mean"])
            print(f"{r:<7} {'uncertainty':<14} {uf:.6e}             {ue:.6e}    {ufd:.6e}")

    # Write CSV
    round_csv = OUT_DIR / "random_baseline_round_summary.csv"
    with round_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=round_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(round_rows)

    model_csv = OUT_DIR / "random_baseline_model_level.csv"
    model_fields = ["seed", "round", "model", "energy_rmse", "energy_rmse_natoms", "force_rmse"]
    with model_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=model_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(model_rows)

    print(f"\nWrote {round_csv}")
    print(f"Wrote {model_csv}")


if __name__ == "__main__":
    main()
