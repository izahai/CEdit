#!/usr/bin/env python3
"""Re-render the MOV2 figure from cached CSV/JSON artifacts."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_projection(path):
    targets = []
    anchor = None
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            point = (float(row["pc1"]), float(row["pc2"]))
            if row["kind"] == "anchor":
                anchor = point
            else:
                targets.append(point)
    if not targets or anchor is None:
        raise ValueError("Projection CSV must contain targets and one anchor")
    return targets, anchor


def read_spectrum(path):
    spectra = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            spectra[row["method"]].append(
                (int(row["component"]), float(row["relative_singular_value"]))
            )
    for values in spectra.values():
        values.sort()
    return spectra


def render(output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets, anchor = read_projection(output_dir / "embedding_projection.csv")
    spectra = read_spectrum(output_dir / "residual_spectrum.csv")
    with (output_dir / "summary.json").open(encoding="utf-8") as file:
        summary = json.load(file)

    rank = int(summary["tgprs_rank"])
    explained = summary["pca_explained_variance"]
    residual_summaries = summary["residual_summaries"]
    colors = {"legacy": "#0072B2", "tgprs": "#D55E00"}
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
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    geometry_axis = axes[0]
    for point in targets:
        geometry_axis.plot(
            [point[0], anchor[0]],
            [point[1], anchor[1]],
            color=colors["legacy"],
            alpha=0.10,
            linewidth=0.55,
            zorder=1,
        )
    geometry_axis.scatter(
        [point[0] for point in targets],
        [point[1] for point in targets],
        s=12,
        facecolors="none",
        edgecolors=colors["legacy"],
        linewidths=0.7,
        label="Celebrity prompts",
        zorder=2,
    )
    geometry_axis.scatter(
        [anchor[0]],
        [anchor[1]],
        marker="*",
        s=90,
        color=colors["tgprs"],
        edgecolor="black",
        linewidth=0.5,
        label='Anchor: "person"',
        zorder=4,
    )
    geometry_axis.annotate(
        '"person"',
        anchor,
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=7,
        fontweight="bold",
    )
    geometry_axis.axhline(0, color="#BBBBBB", linewidth=0.45, zorder=0)
    geometry_axis.axvline(0, color="#BBBBBB", linewidth=0.45, zorder=0)
    geometry_axis.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% variance)")
    geometry_axis.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% variance)")
    geometry_axis.set_title("One anchor, many residual directions", loc="left")
    geometry_axis.legend(frameon=False, loc="best")
    geometry_axis.text(
        0.02,
        0.03,
        "(a)",
        transform=geometry_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.85},
        zorder=6,
    )

    spectrum_axis = axes[1]
    labels = {"legacy": "Common anchor", "tgprs": "TGPRS"}
    styles = {"legacy": "-", "tgprs": "--"}
    for method in ("legacy", "tgprs"):
        values = spectra[method]
        spectrum_axis.plot(
            [item[0] for item in values],
            [max(item[1], 1e-8) for item in values],
            color=colors[method],
            linestyle=styles[method],
            linewidth=1.6,
            label=(
                f"{labels[method]} "
                f"($r={residual_summaries[method]['numerical_rank']}$)"
            ),
        )
    tolerance = residual_summaries["legacy"]["relative_rank_tolerance"]
    spectrum_axis.axhline(
        tolerance,
        color="#777777",
        linestyle=":",
        linewidth=0.8,
        label="Rank tolerance",
    )
    spectrum_axis.axvline(rank, color="#777777", linestyle="--", linewidth=0.8)
    spectrum_axis.text(
        rank + 1.5,
        1.5e-3,
        f"$k={rank}$",
        fontsize=7,
        rotation=90,
        va="bottom",
    )
    spectrum_axis.set_yscale("log")
    spectrum_axis.set_xlim(1, len(targets))
    spectrum_axis.set_ylim(1e-8, 1.5)
    spectrum_axis.set_xlabel("Singular component")
    spectrum_axis.set_ylabel(r"Relative singular value $\sigma_j/\sigma_1$")
    spectrum_axis.set_title("Rank in the original CLIP space", loc="left")
    spectrum_axis.grid(True, which="major", alpha=0.18, linewidth=0.5)
    spectrum_axis.legend(frameon=False, loc="upper right")
    spectrum_axis.text(
        0.02,
        0.03,
        "(b)",
        transform=spectrum_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.85},
        zorder=6,
    )

    figure.suptitle(
        "A shared anchor does not imply a shared residual subspace",
        fontsize=10,
        y=0.995,
    )
    figure.text(
        0.5,
        0.015,
        r"$R \rightarrow D=R^\top T/N \rightarrow \Delta W$"
        r"$\qquad \mathrm{rank}(\Delta W)\leq\mathrm{rank}(D)\leq\mathrm{rank}(R)$",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.95), w_pad=2.0)
    stem = output_dir / "common_anchor_clip_geometry"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"Overwrote {stem.with_suffix('.pdf')}")
    print(f"Overwrote {stem.with_suffix('.png')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = parser.parse_args()
    render(args.output_dir)


if __name__ == "__main__":
    main()
