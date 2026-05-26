"""Generate learning curve plots for rMD17 ethanol active learning."""

import csv
from pathlib import Path

ROOT = Path("/data/guyida/deepmd-al-hpc")
CSV_PATH = ROOT / "experiments/rmd17_ethanol_summary/al_rounds_summary.csv"
OUT_DIR = ROOT / "experiments/rmd17_ethanol_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_rows():
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def to_float(row, key):
    value = row.get(key, "")
    return None if value == "" else float(value)


def scale(value, vmin, vmax, out_min, out_max):
    if vmax == vmin:
        return (out_min + out_max) / 2
    return out_min + (value - vmin) / (vmax - vmin) * (out_max - out_min)


def make_svg(path, title, xlabel, ylabel, x_values, series):
    width, height = 760, 480
    left, right, top, bottom = 90, 40, 60, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_y = []
    for _, values in series:
        all_y.extend([v for v in values if v is not None])

    y_min = min(all_y)
    y_max = max(all_y)
    pad = (y_max - y_min) * 0.12 if y_max > y_min else 1.0
    y_min -= pad
    y_max += pad

    x_min, x_max = min(x_values), max(x_values)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    def x_pos(x):
        return scale(x, x_min, x_max, left, left + plot_w)

    def y_pos(y):
        return scale(y, y_min, y_max, top + plot_h, top)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>')

    # axes
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black" stroke-width="1.5"/>')

    # x ticks
    for x in x_values:
        xp = x_pos(x)
        parts.append(f'<line x1="{xp}" y1="{top+plot_h}" x2="{xp}" y2="{top+plot_h+6}" stroke="black"/>')
        parts.append(f'<text x="{xp}" y="{top+plot_h+25}" text-anchor="middle" font-size="13" font-family="Arial">{int(x)}</text>')

    # y ticks
    for i in range(5):
        yv = y_min + (y_max - y_min) * i / 4
        yp = y_pos(yv)
        parts.append(f'<line x1="{left-6}" y1="{yp}" x2="{left}" y2="{yp}" stroke="black"/>')
        parts.append(f'<text x="{left-10}" y="{yp+4}" text-anchor="end" font-size="12" font-family="Arial">{yv:.3g}</text>')
        parts.append(f'<line x1="{left}" y1="{yp}" x2="{left+plot_w}" y2="{yp}" stroke="#dddddd" stroke-width="0.5"/>')

    # labels
    parts.append(f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" font-size="15" font-family="Arial">{xlabel}</text>')
    parts.append(f'<text x="22" y="{top+plot_h/2}" text-anchor="middle" font-size="15" font-family="Arial" transform="rotate(-90 22 {top+plot_h/2})">{ylabel}</text>')

    # data series
    legend_x = left + plot_w - 170
    legend_y = top + 10

    for idx, (name, values) in enumerate(series):
        color = colors[idx % len(colors)]
        points = []
        for x, y in zip(x_values, values):
            if y is None:
                continue
            points.append((x_pos(x), y_pos(y)))

        if len(points) >= 2:
            poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5"/>')

        for xp, yp in points:
            parts.append(f'<circle cx="{xp:.2f}" cy="{yp:.2f}" r="5" fill="{color}"/>')

        ly = legend_y + idx * 22
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+25}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{legend_x+32}" y="{ly+4}" font-size="13" font-family="Arial">{name}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main():
    rows = read_rows()
    rounds = [int(r["round"]) for r in rows]

    train_frames = [to_float(r, "train_frames") for r in rows]
    candidate_frames = [to_float(r, "candidate_frames") for r in rows]

    force_dev_mean = [to_float(r, "force_dev_max_mean") for r in rows]
    force_dev_max = [to_float(r, "force_dev_max_max") for r in rows]
    force_dev_min = [to_float(r, "force_dev_max_min") for r in rows]

    force_rmse_mean = [to_float(r, "force_rmse_mean") for r in rows]
    energy_rmse_mean = [to_float(r, "energy_rmse_mean") for r in rows]

    prefix = "rmd17_ethanol_"

    make_svg(
        OUT_DIR / f"{prefix}force_model_deviation_rounds.svg",
        "rMD17 Ethanol — Force model deviation over active learning rounds",
        "Active learning round",
        "Top-1000 force_dev_max",
        rounds,
        [
            ("Mean", force_dev_mean),
            ("Max", force_dev_max),
            ("Min", force_dev_min),
        ],
    )

    make_svg(
        OUT_DIR / f"{prefix}dataset_size_rounds.svg",
        "rMD17 Ethanol — Dataset size evolution",
        "Active learning round",
        "Number of frames",
        rounds,
        [
            ("Training frames", train_frames),
            ("Candidate frames", candidate_frames),
        ],
    )

    make_svg(
        OUT_DIR / f"{prefix}validation_rmse_rounds.svg",
        "rMD17 Ethanol — Validation errors over active learning rounds",
        "Active learning round",
        "Mean RMSE",
        rounds,
        [
            ("Force RMSE (eV/A)", force_rmse_mean),
            ("Energy RMSE (eV)", energy_rmse_mean),
        ],
    )

    print("Saved figures:")
    for p in sorted(OUT_DIR.glob("*.svg")):
        print(p)


if __name__ == "__main__":
    main()
