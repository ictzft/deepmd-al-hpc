"""Generate combined validation vs independent test comparison plot for rMD17 ethanol."""

import csv
from pathlib import Path

ROOT = Path("/data/guyida/deepmd-al-hpc")

VALID_CSV = ROOT / "experiments/rmd17_ethanol_summary/al_rounds_summary.csv"
TEST_CSV = ROOT / "experiments/rmd17_ethanol_summary/independent_test_round_summary.csv"
OUT_DIR = ROOT / "experiments/rmd17_ethanol_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def to_float(row, key):
    v = row.get(key, "")
    return None if v == "" else float(v)


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
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black" stroke-width="1.5"/>')

    for x in x_values:
        xp = x_pos(x)
        parts.append(f'<line x1="{xp}" y1="{top+plot_h}" x2="{xp}" y2="{top+plot_h+6}" stroke="black"/>')
        parts.append(f'<text x="{xp}" y="{top+plot_h+25}" text-anchor="middle" font-size="13" font-family="Arial">{int(x)}</text>')

    for i in range(5):
        yv = y_min + (y_max - y_min) * i / 4
        yp = y_pos(yv)
        parts.append(f'<line x1="{left-6}" y1="{yp}" x2="{left}" y2="{yp}" stroke="black"/>')
        parts.append(f'<text x="{left-10}" y="{yp+4}" text-anchor="end" font-size="12" font-family="Arial">{yv:.4f}</text>')
        parts.append(f'<line x1="{left}" y1="{yp}" x2="{left+plot_w}" y2="{yp}" stroke="#dddddd" stroke-width="0.5"/>')

    parts.append(f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" font-size="15" font-family="Arial">{xlabel}</text>')
    parts.append(f'<text x="22" y="{top+plot_h/2}" text-anchor="middle" font-size="15" font-family="Arial" transform="rotate(-90 22 {top+plot_h/2})">{ylabel}</text>')

    legend_x = left + plot_w - 220
    legend_y = top + 10
    for idx, (name, values) in enumerate(series):
        color = colors[idx % len(colors)]
        pts = [(x_pos(x), y_pos(y)) for x, y in zip(x_values, values) if y is not None]
        if len(pts) >= 2:
            poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
            parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for xp, yp in pts:
            parts.append(f'<circle cx="{xp:.2f}" cy="{yp:.2f}" r="5" fill="{color}"/>')
        ly = legend_y + idx * 22
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+25}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{legend_x+32}" y="{ly+4}" font-size="13" font-family="Arial">{name}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main():
    valid_rows = read_csv(VALID_CSV)
    test_rows = read_csv(TEST_CSV)
    rounds = list(range(4))

    valid_force = [to_float(r, "force_rmse_mean") for r in valid_rows]
    test_force = [to_float(r, "force_rmse_mean") for r in test_rows]

    valid_energy = [to_float(r, "energy_rmse_mean") for r in valid_rows]
    test_energy = [to_float(r, "energy_rmse_mean") for r in test_rows]

    make_svg(
        OUT_DIR / "rmd17_ethanol_independent_test_force_rmse.svg",
        "rMD17 Ethanol — Force RMSE: Validation vs Independent Test",
        "Active learning round",
        "Force RMSE (eV/Å)",
        rounds,
        [("Valid (5000 frames)", valid_force), ("Independent Test (10000 frames)", test_force)],
    )

    make_svg(
        OUT_DIR / "rmd17_ethanol_independent_test_energy_rmse.svg",
        "rMD17 Ethanol — Energy RMSE: Validation vs Independent Test",
        "Active learning round",
        "Energy RMSE (eV)",
        rounds,
        [("Valid (5000 frames)", valid_energy), ("Independent Test (10000 frames)", test_energy)],
    )

    print("Saved figures:")
    for p in sorted(OUT_DIR.glob("*independent_test*.svg")):
        print(p)


if __name__ == "__main__":
    main()
