#!/usr/bin/env python3
"""Render the paper-ready MOV1 spectrum and Pareto figure."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = ("legacy_full", "legacy_svd_rank30", "tgprs_rank30")
COLORS = {
    "legacy_full": "#0072B2",
    "legacy_svd_rank30": "#D55E00",
    "tgprs_rank30": "#009E73",
}
MARKERS = {
    "legacy_full": "o",
    "legacy_svd_rank30": "s",
    "tgprs_rank30": "^",
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def group_by_method(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    return grouped


def render_figure(spectrum_path, rank_path, metrics_path, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    spectrum = group_by_method(read_rows(spectrum_path))
    metrics = group_by_method(read_rows(metrics_path))
    rank_rows = {row["method"]: row for row in read_rows(rank_path)}
    missing = (set(METHOD_ORDER) - set(spectrum)) | (
        set(METHOD_ORDER) - set(metrics)
    )
    if missing:
        raise ValueError(f"Missing MOV1 methods: {sorted(missing)}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))

    spectrum_axis = axes[0]
    for method in METHOD_ORDER:
        rows = sorted(spectrum[method], key=lambda row: int(row["component"]))
        rank = rank_rows[method]
        components = [int(row["component"]) for row in rows]
        energy = [max(float(row["normalized_energy"]), 1e-10) for row in rows]
        label = (
            f"{rows[0]['method_label']} "
            f"(r={int(rank['residual_numerical_rank'])}, "
            f"sr={float(rank['residual_stable_rank']):.1f}, "
            f"rD={int(rank['edit_statistic_numerical_rank'])})"
        )
        spectrum_axis.plot(
            components,
            energy,
            color=COLORS[method],
            linewidth=1.6,
            label=label,
        )
    spectrum_axis.axvline(30, color="#666666", linestyle="--", linewidth=0.8)
    spectrum_axis.text(31.5, 2e-2, "rank 30", color="#555555", fontsize=7)
    spectrum_axis.set_yscale("log")
    spectrum_axis.set_xlim(1, 100)
    spectrum_axis.set_ylim(1e-8, 1.0)
    spectrum_axis.set_xlabel("Singular component")
    spectrum_axis.set_ylabel("Normalized spectral energy")
    spectrum_axis.set_title("(a) Residual spectral complexity", loc="left")
    spectrum_axis.grid(True, which="major", alpha=0.2, linewidth=0.5)
    spectrum_axis.legend(frameon=False, loc="upper right")

    pareto_axis = axes[1]
    for method in METHOD_ORDER:
        rows = sorted(metrics[method], key=lambda row: float(row["residual_scale"]))
        x = [float(row["retain_identity_hit_rate"]) for row in rows]
        y = [float(row["erasure_success"]) for row in rows]
        xerr = [
            [
                value - float(row["retain_identity_hit_ci_low"])
                for value, row in zip(x, rows)
            ],
            [
                float(row["retain_identity_hit_ci_high"]) - value
                for value, row in zip(x, rows)
            ],
        ]
        yerr = [
            [
                value - float(row["erasure_success_ci_low"])
                for value, row in zip(y, rows)
            ],
            [
                float(row["erasure_success_ci_high"]) - value
                for value, row in zip(y, rows)
            ],
        ]
        pareto_axis.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=4,
            linewidth=1.4,
            elinewidth=0.7,
            capsize=1.8,
            label=rows[0]["method_label"],
        )
        for x_value, y_value, row in zip(x, y, rows):
            pareto_axis.annotate(
                f"{float(row['residual_scale']):.1f}",
                (x_value, y_value),
                xytext=(3, 3),
                textcoords="offset points",
                color=COLORS[method],
                fontsize=6,
            )
    pareto_axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    pareto_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    pareto_axis.set_xlabel("Retain identity hit rate ↑")
    pareto_axis.set_ylabel("Erasure success ↑")
    pareto_axis.set_title("(b) Erasure–retention trade-off", loc="left")
    pareto_axis.grid(True, alpha=0.2, linewidth=0.5)
    pareto_axis.legend(frameon=False, loc="best")

    figure.tight_layout(w_pad=2.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "common_anchor_rank_tradeoff"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)
    return stem.with_suffix(".pdf"), stem.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectrum", type=Path, required=True)
    parser.add_argument("--rank-summary", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pdf_path, png_path = render_figure(
        args.spectrum,
        args.rank_summary,
        args.metrics,
        args.output_dir,
    )
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
