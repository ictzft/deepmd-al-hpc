#!/usr/bin/env python3
"""Plot uncertainty branch vs random baseline comparison.

Reads from experiments/baselines/random_vs_uncertainty_summary.csv.
Generates SVG figures suitable for paper drafts.

Output:
  experiments/figures/random_vs_uncertainty_force_rmse.svg
  experiments/figures/random_vs_uncertainty_energy_rmse.svg
  experiments/figures/random_vs_uncertainty_candidate_force_dev.svg
  experiments/figures/random_vs_uncertainty_dataset_size.svg
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "experiments" / "baselines" / "random_vs_uncertainty_summary.csv"
FIG_DIR = PROJECT_ROOT / "experiments" / "figures"


def scale(value, vmin, vmax, out_min, out_max):
    if vmax == vmin:
        return (out_min + out_max) / 2
    return out_min + (value - vmin) / (vmax - vmin) * (out_max - out_min)


def make_svg(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    x_values: list[int],
    uncertainty_vals: list[float | None],
    random_mean_vals: list[float | None],
    random_std_vals: list[float | None],
    uncertainty_label: str = "Uncertainty",
    random_label: str = "Random (mean)",
) -> None:
    """Generate a clean SVG line plot with error shading for random baseline."""
    width, height = 760, 480
    left, right, top, bottom = 95, 40, 60, 75
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_y = [v for v in uncertainty_vals + random_mean_vals if v is not None]
    if not all_y:
        print(f"  [SKIP] {title}: no data")
        return

    y_min, y_max = min(all_y), max(all_y)
    pad = max((y_max - y_min) * 0.15, 0.01)
    y_min -= pad
    y_max += pad

    x_min, x_max = 0, max(x_values) if x_values else 3
    if x_min == x_max:
        x_max = x_min + 1

    colors = {"uncertainty": "#1f77b4", "random": "#d62728"}

    def xp(x):
        return scale(x, x_min, x_max, left, left + plot_w)

    def yp(y):
        return scale(y, y_min, y_max, top + plot_h, top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-size="19" font-family="sans-serif">{title}</text>',
    ]

    # Grid lines
    for i in range(6):
        yv = y_min + (y_max - y_min) * i / 5
        yv_pos = yp(yv)
        parts.append(f'<line x1="{left}" y1="{yv_pos:.1f}" x2="{left+plot_w}" y2="{yv_pos:.1f}" stroke="#e8e8e8" stroke-width="0.5"/>')

    # Axes
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black" stroke-width="1.5"/>')

    # X ticks
    for x in x_values:
        if x < x_min or x > x_max:
            continue
        x_pos = xp(x)
        parts.append(f'<line x1="{x_pos:.1f}" y1="{top+plot_h}" x2="{x_pos:.1f}" y2="{top+plot_h+6}" stroke="black"/>')
        parts.append(f'<text x="{x_pos:.1f}" y="{top+plot_h+23}" text-anchor="middle" font-size="13" font-family="sans-serif">{x}</text>')

    # Y ticks
    for i in range(6):
        yv = y_min + (y_max - y_min) * i / 5
        yv_pos = yp(yv)
        parts.append(f'<line x1="{left-6}" y1="{yv_pos:.1f}" x2="{left}" y2="{yv_pos:.1f}" stroke="black"/>')
        parts.append(f'<text x="{left-10}" y="{yv_pos+4:.1f}" text-anchor="end" font-size="12" font-family="sans-serif">{yv:.3g}</text>')

    # Labels
    parts.append(f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" font-size="14" font-family="sans-serif">{xlabel}</text>')
    parts.append(f'<text x="22" y="{top+plot_h/2}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90 22 {top+plot_h/2})">{ylabel}</text>')

    # Random std shading (behind mean line)
    for i, (m, s, x) in enumerate(zip(random_mean_vals, random_std_vals, x_values)):
        if m is None or s is None:
            continue
        x_center = xp(x)
        y_top = yp(m + s)
        y_bot = yp(m - s)
        # Vertical error bar
        parts.append(f'<line x1="{x_center:.1f}" y1="{y_top:.1f}" x2="{x_center:.1f}" y2="{y_bot:.1f}" stroke="{colors["random"]}" stroke-width="1.8" opacity="0.4"/>')
        # Cap lines
        cap_w = 8
        parts.append(f'<line x1="{x_center-cap_w:.1f}" y1="{y_top:.1f}" x2="{x_center+cap_w:.1f}" y2="{y_top:.1f}" stroke="{colors["random"]}" stroke-width="1.8" opacity="0.4"/>')
        parts.append(f'<line x1="{x_center-cap_w:.1f}" y1="{y_bot:.1f}" x2="{x_center+cap_w:.1f}" y2="{y_bot:.1f}" stroke="{colors["random"]}" stroke-width="1.8" opacity="0.4"/>')

    # Draw lines
    for label, values, color_key in [
        (uncertainty_label, uncertainty_vals, "uncertainty"),
        (random_label, random_mean_vals, "random"),
    ]:
        color = colors[color_key]
        points = []
        for x, y in zip(x_values, values):
            if y is None:
                continue
            points.append((xp(x), yp(y)))

        if len(points) >= 2:
            poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5"/>')

        for px, py in points:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{color}"/>')

    # Legend
    lx, ly = left + plot_w - 230, top + 12
    for i, (label, color_key) in enumerate([
        (uncertainty_label, "uncertainty"),
        (random_label, "random"),
    ]):
        y_off = ly + i * 26
        c = colors[color_key]
        parts.append(f'<line x1="{lx}" y1="{y_off}" x2="{lx+28}" y2="{y_off}" stroke="{c}" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{lx+14}" cy="{y_off}" r="4" fill="{c}"/>')
        parts.append(f'<text x="{lx+36}" y="{y_off+5}" font-size="13" font-family="sans-serif">{label}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        print("Run summarize_random_vs_uncertainty.py first.")
        return

    # Read data
    rows = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Group by method + seed
    uncertainty_rows = [r for r in rows if r["method"] == "uncertainty"]
    random_mean_rows = [r for r in rows if r["method"] == "random" and r["seed"] == "mean"]
    random_seed_rows = [r for r in rows if r["method"] == "random" and r["seed"] != "mean"]

    def _f(row, key):
        v = row.get(key, "")
        return float(v) if v else None

    rounds_u = sorted(set(int(r["round"]) for r in uncertainty_rows))
    rounds_r = sorted(set(int(r["round"]) for r in random_mean_rows))
    all_rounds = sorted(set(rounds_u + rounds_r))

    # --- Plot 1: Force RMSE ---
    make_svg(
        FIG_DIR / "random_vs_uncertainty_force_rmse.svg",
        "Validation Force RMSE (uncertainty vs random)",
        "Active learning round",
        "Force RMSE / eV/A",
        all_rounds,
        [_f(next((r for r in uncertainty_rows if int(r["round"]) == rnd), {}), "force_rmse_mean") for rnd in all_rounds],
        [_f(next((r for r in random_mean_rows if int(r["round"]) == rnd), {}), "force_rmse_mean") for rnd in all_rounds],
        [_f(next((r for r in random_mean_rows if int(r["round"]) == rnd), {}), "force_rmse_std") for rnd in all_rounds],
    )

    # --- Plot 2: Energy RMSE ---
    make_svg(
        FIG_DIR / "random_vs_uncertainty_energy_rmse.svg",
        "Validation Energy RMSE (uncertainty vs random)",
        "Active learning round",
        "Energy RMSE / eV",
        all_rounds,
        [_f(next((r for r in uncertainty_rows if int(r["round"]) == rnd), {}), "energy_rmse_mean") for rnd in all_rounds],
        [_f(next((r for r in random_mean_rows if int(r["round"]) == rnd), {}), "energy_rmse_mean") for rnd in all_rounds],
        [_f(next((r for r in random_mean_rows if int(r["round"]) == rnd), {}), "energy_rmse_std") for rnd in all_rounds],
    )

    # --- Plot 3: Candidate force_dev_max ---
    make_svg(
        FIG_DIR / "random_vs_uncertainty_candidate_force_dev.svg",
        "Candidate pool force model deviation (uncertainty vs random)",
        "Active learning round",
        "force_dev_max mean",
        all_rounds,
        [_f(next((r for r in uncertainty_rows if int(r["round"]) == rnd), {}), "force_dev_max_mean") for rnd in all_rounds],
        [_f(next((r for r in random_mean_rows if int(r["round"]) == rnd), {}), "force_dev_max_mean") for rnd in all_rounds],
        [_f(next((r for r in random_mean_rows if int(r["round"]) == rnd), {}), "force_dev_max_std") for rnd in all_rounds],
        random_label="Random (mean ± std)",
    )

    # --- Plot 4: Dataset size ---
    make_svg(
        FIG_DIR / "random_vs_uncertainty_dataset_size.svg",
        "Training set size evolution",
        "Active learning round",
        "Number of frames",
        all_rounds,
        [_f(next((r for r in uncertainty_rows if int(r["round"]) == rnd), {}), "train_frames") for rnd in all_rounds],
        [_f(next((r for r in random_mean_rows if int(r["round"]) == rnd), {}), "train_frames") for rnd in all_rounds],
        [None] * len(all_rounds),
        uncertainty_label="Uncertainty train",
        random_label="Random train",
    )

    print("Saved figures:")
    for p in sorted(FIG_DIR.glob("random_vs_uncertainty_*.svg")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
