"""Native one-panel PDF renderers for the MOV3 paper figures."""

import math
from pathlib import Path

import numpy as np


COLORS = {"legacy": "#0072B2", "tgprs": "#D55E00"}


def _save(figure, path, fixed_canvas=False):
    if fixed_canvas:
        figure.savefig(path, format="pdf")
    else:
        figure.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.04)


def _render_common_anchor(targets, anchor, explained, output_dir, plt):
    figure, axis = plt.subplots(figsize=(3.6, 3.6))
    for point in targets:
        axis.plot(
            [point[0], anchor[0]],
            [point[1], anchor[1]],
            color=COLORS["legacy"],
            alpha=0.11,
            linewidth=0.6,
            zorder=1,
        )
    axis.scatter(
        targets[:, 0],
        targets[:, 1],
        s=15,
        facecolors="none",
        edgecolors=COLORS["legacy"],
        linewidths=0.8,
        zorder=2,
    )
    axis.scatter(
        [anchor[0]],
        [anchor[1]],
        marker="*",
        s=105,
        color=COLORS["tgprs"],
        edgecolor="black",
        linewidth=0.55,
        zorder=4,
    )
    axis.annotate(
        '"person"',
        anchor,
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
    )
    x_values = np.append(targets[:, 0], anchor[0])
    y_values = np.append(targets[:, 1], anchor[1])
    x_pad = 0.07 * np.ptp(x_values)
    y_pad = 0.10 * np.ptp(y_values)
    axis.set_xlim(x_values.min() - x_pad, x_values.max() + x_pad)
    axis.set_ylim(y_values.min() - y_pad, y_values.max() + y_pad)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.tick_params(which="both", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    figure.subplots_adjust(left=0.035, right=0.965, bottom=0.035, top=0.965)
    _save(figure, output_dir / "common_anchor_residuals.pdf", fixed_canvas=True)
    plt.close(figure)


def _render_target_pairs(targets, endpoints, rank, output_dir, plt):
    displacement = endpoints - targets
    pair_count = len(targets)
    columns = 10
    origins = np.column_stack([
        np.arange(pair_count) % columns,
        (pair_count - 1 - np.arange(pair_count)) // columns,
    ]).astype(float)
    residual_lengths = np.linalg.norm(displacement, axis=1)
    scale = max(float(np.percentile(residual_lengths, 90)), 1e-12)
    displayed = displacement * (0.68 / scale)
    displayed_lengths = np.linalg.norm(displayed, axis=1)
    long_mask = displayed_lengths > 0.74
    displayed[long_mask] *= (0.74 / displayed_lengths[long_mask])[:, None]
    displayed_endpoints = origins + displayed

    figure, axis = plt.subplots(figsize=(3.6, 3.6))
    axis.quiver(
        origins[:, 0],
        origins[:, 1],
        displayed[:, 0],
        displayed[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0032,
        headwidth=4.2,
        headlength=5.2,
        headaxislength=4.6,
        color=COLORS["tgprs"],
        alpha=0.72,
        zorder=1,
    )
    axis.scatter(
        origins[:, 0],
        origins[:, 1],
        s=14,
        facecolors="none",
        edgecolors=COLORS["legacy"],
        linewidths=0.75,
        zorder=2,
    )
    axis.scatter(
        displayed_endpoints[:, 0],
        displayed_endpoints[:, 1],
        s=11,
        color=COLORS["tgprs"],
        linewidths=0,
        zorder=3,
    )
    rows = math.ceil(pair_count / columns)
    axis.set_xlim(-0.8, columns - 0.2)
    axis.set_ylim(-0.8, rows - 0.2)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.tick_params(which="both", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    figure.subplots_adjust(left=0.035, right=0.965, bottom=0.035, top=0.965)
    _save(
        figure,
        output_dir / "target_specific_residual_pairs.pdf",
        fixed_canvas=True,
    )
    plt.close(figure)


def _render_spectrum(spectra, ranks, rank_tolerance, rank, output_dir, plt):
    figure, axis = plt.subplots(figsize=(4.8, 3.65))
    labels = {"legacy": "Common anchor", "tgprs": "RASE"}
    styles = {"legacy": "-", "tgprs": "--"}
    for method in ("legacy", "tgprs"):
        values = np.maximum(np.asarray(spectra[method], dtype=float), 1e-8)
        axis.plot(
            np.arange(1, len(values) + 1),
            values,
            color=COLORS[method],
            linestyle=styles[method],
            linewidth=1.7,
            label=f"{labels[method]} ($r={ranks[method]}$)",
        )
    axis.axhline(
        rank_tolerance,
        color="#777777",
        linestyle=":",
        linewidth=0.9,
        label="Rank tolerance",
    )
    axis.axvline(rank, color="#777777", linestyle="--", linewidth=0.9)
    axis.text(rank + 1.5, 1.5e-3, f"$k={rank}$", fontsize=8, rotation=90, va="bottom")
    axis.set_yscale("log")
    axis.minorticks_off()
    axis.tick_params(axis="y", which="minor", left=False, right=False)
    axis.set_xlim(1, len(spectra["legacy"]))
    axis.set_ylim(1e-8, 1.5)
    axis.grid(True, which="major", alpha=0.18, linewidth=0.5)
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=3,
        columnspacing=0.9,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.72)
    _save(figure, output_dir / "residual_rank_spectrum.pdf")
    plt.close(figure)


def render_native_panel_pdfs(
    targets,
    anchor,
    endpoints,
    explained,
    spectra,
    ranks,
    rank_tolerance,
    rank,
    output_dir,
):
    """Create three independent vector PDFs, one native canvas per panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    targets = np.asarray(targets, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    endpoints = np.asarray(endpoints, dtype=float)
    explained = np.asarray(explained, dtype=float)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    _render_common_anchor(targets, anchor, explained, output_dir, plt)
    _render_target_pairs(targets, endpoints, rank, output_dir, plt)
    _render_spectrum(spectra, ranks, rank_tolerance, rank, output_dir, plt)
