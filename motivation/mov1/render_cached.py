#!/usr/bin/env python3
"""Re-render the MOV3 figure from cached CSV/JSON artifacts."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from native_panels import render_native_panel_pdfs


def read_projection(path):
    targets = []
    tgprs_endpoints = []
    anchor = None
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            point = (float(row["pc1"]), float(row["pc2"]))
            if row["kind"] == "anchor":
                anchor = point
            else:
                targets.append(point)
                tgprs_endpoints.append(
                    (float(row["tgprs_end_pc1"]), float(row["tgprs_end_pc2"]))
                )
    if not targets or anchor is None:
        raise ValueError("Projection CSV must contain targets and one anchor")
    return targets, anchor, tgprs_endpoints


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
    import numpy as np

    targets, anchor, tgprs_endpoints = read_projection(
        output_dir / "embedding_projection.csv"
    )
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
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.15))

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
        label="Targets",
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
        label='"person"',
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
    geometry_axis.set_title("Common-anchor residuals", loc="left")
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

    tgprs_axis = axes[1]
    target_xy = np.asarray(targets)
    endpoint_xy = np.asarray(tgprs_endpoints)
    displacement_xy = endpoint_xy - target_xy
    pair_count = len(target_xy)
    columns = 10
    grid_x = np.arange(pair_count) % columns
    grid_y = (pair_count - 1 - np.arange(pair_count)) // columns
    residual_lengths = np.linalg.norm(displacement_xy, axis=1)
    length_scale = max(float(np.percentile(residual_lengths, 90)), 1e-12)
    display_displacements = displacement_xy * (0.68 / length_scale)
    display_lengths = np.linalg.norm(display_displacements, axis=1)
    long_mask = display_lengths > 0.74
    display_displacements[long_mask] *= (
        0.74 / display_lengths[long_mask]
    )[:, None]
    display_endpoints = np.column_stack([grid_x, grid_y]) + display_displacements
    tgprs_axis.quiver(
        grid_x,
        grid_y,
        display_displacements[:, 0],
        display_displacements[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0030,
        headwidth=4.2,
        headlength=5.2,
        headaxislength=4.6,
        color=colors["tgprs"],
        alpha=0.65,
        zorder=1,
    )
    tgprs_axis.scatter(
        grid_x,
        grid_y,
        s=11,
        facecolors="none",
        edgecolors=colors["legacy"],
        linewidths=0.6,
        alpha=0.90,
        label="Targets",
        zorder=2,
    )
    tgprs_axis.scatter(
        display_endpoints[:, 0],
        display_endpoints[:, 1],
        s=9,
        color=colors["tgprs"],
        alpha=0.90,
        linewidths=0,
        label=r"Effective anchors $\tilde a_i$",
        zorder=3,
    )
    tgprs_axis.set_xlim(-0.75, columns - 0.25)
    tgprs_axis.set_ylim(-1.35, math.ceil(pair_count / columns) + 0.55)
    tgprs_axis.set_aspect("equal")
    tgprs_axis.set_xticks([])
    tgprs_axis.set_yticks([])
    for spine in tgprs_axis.spines.values():
        spine.set_visible(False)
    tgprs_axis.set_title(
        f"Target-specific pairs ($k={rank}$; translated)", loc="left"
    )
    tgprs_axis.legend(
        frameon=True,
        framealpha=0.96,
        edgecolor="none",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    tgprs_axis.text(
        0.015,
        0.015,
        "(b)",
        transform=tgprs_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.85},
        zorder=6,
    )

    all_x = [point[0] for point in targets] + [anchor[0]] + [
        point[0] for point in tgprs_endpoints
    ]
    all_y = [point[1] for point in targets] + [anchor[1]] + [
        point[1] for point in tgprs_endpoints
    ]
    x_padding = 0.06 * (max(all_x) - min(all_x))
    y_padding = 0.06 * (max(all_y) - min(all_y))
    geometry_axis.set_xlim(min(all_x) - x_padding, max(all_x) + x_padding)
    geometry_axis.set_ylim(min(all_y) - y_padding, max(all_y) + y_padding)

    spectrum_axis = axes[2]
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
    spectrum_axis.legend(
        frameon=True,
        framealpha=0.96,
        edgecolor="#DDDDDD",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    spectrum_axis.text(
        0.02,
        0.03,
        "(c)",
        transform=spectrum_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.85},
        zorder=6,
    )

    figure.suptitle(
        "Common-anchor residuals versus an explicit pairwise subspace",
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
    figure.tight_layout(rect=(0, 0.07, 1, 0.95), w_pad=1.5)
    stem = output_dir / "common_anchor_clip_geometry"
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)
    render_native_panel_pdfs(
        targets=targets,
        anchor=anchor,
        endpoints=tgprs_endpoints,
        explained=explained,
        spectra={
            method: [value for _, value in spectra[method]]
            for method in ("legacy", "tgprs")
        },
        ranks={
            method: residual_summaries[method]["numerical_rank"]
            for method in ("legacy", "tgprs")
        },
        rank_tolerance=residual_summaries["legacy"]["relative_rank_tolerance"],
        rank=rank,
        output_dir=output_dir,
    )
    print(f"Overwrote {stem.with_suffix('.png')}")
    for filename in (
        "common_anchor_residuals.pdf",
        "target_specific_residual_pairs.pdf",
        "residual_rank_spectrum.pdf",
    ):
        print(f"Overwrote {output_dir / filename}")


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
