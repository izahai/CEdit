#!/usr/bin/env python3
"""Create paper-ready figures from rank-analysis JSONL outputs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "legacy": "Legacy",
    "legacy_full": "Legacy full",
    "legacy_svd": "Legacy-SVD",
    "random_subspace": "Random subspace",
    "tgprs": "TGPRS",
}
METHOD_COLORS = {
    "legacy": "#222222",
    "legacy_full": "#222222",
    "legacy_svd": "#377eb8",
    "random_subspace": "#999999",
    "tgprs": "#e41a1c",
}
TGPRS_RANK_COLORS = {
    5: "#4daf4a",
    10: "#ff7f00",
    30: "#e41a1c",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="Rank-analysis output directory")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def save_figure(figure, figures_dir, stem, dpi):
    figure.tight_layout()
    figure.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(figures_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def flatten_geometry(rows):
    flattened = []
    for row in rows:
        common = {
            "record_id": row["record_id"],
            "seed": row["seed"],
            "target_count": row["target_count"],
            "method": row["method"],
            "requested_rank": row["requested_rank"],
            "realized_rank": row["realized_rank"],
        }
        for matrix_name in (
            "residual",
            "direction_normalized_residual",
            "edit_statistic",
        ):
            flattened.append({
                **common,
                "matrix": matrix_name,
                **row[matrix_name],
            })
    return pd.DataFrame(flattened)


def curve_label(method, requested_rank):
    label = METHOD_LABELS.get(method, method)
    if method == "tgprs" and pd.notna(requested_rank):
        label += f" k={int(requested_rank)}"
    return label


def curve_color(method, requested_rank):
    if method == "tgprs" and pd.notna(requested_rank):
        return TGPRS_RANK_COLORS.get(int(requested_rank), METHOD_COLORS[method])
    return METHOD_COLORS.get(method)


def plot_rank_inflation(frame, figures_dir, dpi):
    selected = frame[frame["matrix"].isin(["residual", "edit_statistic"])]
    grouped = (
        selected.groupby(
            ["method", "requested_rank", "target_count", "matrix"],
            dropna=False,
        )["effective_rank"]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for (method, rank, matrix_name), values in grouped.groupby(
        ["method", "requested_rank", "matrix"], dropna=False
    ):
        values = values.sort_values("target_count")
        color = curve_color(method, rank)
        line_style = "-" if matrix_name == "residual" else "--"
        label = curve_label(method, rank)
        label += " R" if matrix_name == "residual" else " D"
        axis.plot(
            values["target_count"],
            values["mean"],
            color=color,
            linestyle=line_style,
            marker="o",
            label=label,
        )
        axis.fill_between(
            values["target_count"],
            values["mean"] - values["std"],
            values["mean"] + values["std"],
            color=color,
            alpha=0.10,
        )
    axis.set_xlabel("Number of erased targets N")
    axis.set_ylabel("Effective rank")
    axis.set_title("Residual and edit-statistic rank inflation")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    save_figure(figure, figures_dir, "figure_a_rank_vs_targets", dpi)
    return grouped


def plot_spectral_energy(geometry_rows, spectra_path, figures_dir, dpi):
    if not spectra_path.exists() or not geometry_rows:
        return pd.DataFrame()
    maximum_count = max(row["target_count"] for row in geometry_rows)
    seed = min(row["seed"] for row in geometry_rows)
    selected = [
        row
        for row in geometry_rows
        if row["target_count"] == maximum_count and row["seed"] == seed
    ]
    records = []
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    with np.load(spectra_path) as spectra:
        for row in selected:
            key = f"{row['record_id']}__R"
            if key not in spectra:
                continue
            singular_values = spectra[key]
            energy = singular_values ** 2
            cumulative = np.cumsum(energy) / max(float(energy.sum()), 1e-30)
            label = curve_label(row["method"], row["requested_rank"])
            axis.plot(
                np.arange(1, len(cumulative) + 1),
                cumulative,
                label=label,
                color=curve_color(row["method"], row["requested_rank"]),
            )
            records.extend({
                "record_id": row["record_id"],
                "method": row["method"],
                "requested_rank": row["requested_rank"],
                "component": index + 1,
                "cumulative_energy": value,
            } for index, value in enumerate(cumulative.tolist()))
    axis.axhline(0.9, color="#555555", linestyle=":", linewidth=1)
    axis.set_xlabel("Singular component")
    axis.set_ylabel("Cumulative residual energy")
    axis.set_ylim(0.0, 1.02)
    axis.set_title(f"Residual spectral concentration at N={maximum_count}")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    save_figure(figure, figures_dir, "figure_b_singular_energy", dpi)
    return pd.DataFrame(records)


def aggregate_edits(rows):
    aggregate_rows = [row for row in rows if row.get("layer_index") == "aggregate"]
    if not aggregate_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(aggregate_rows)
    grouped = (
        frame.groupby(
            ["method", "requested_rank", "retain_scale"], dropna=False
        )[
            [
                "target_effect_mean",
                "target_rotation_deg_mean",
                "retain_leakage_mean",
                "directional_realization_cosine_mean",
                "directional_relative_error_mean",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in grouped.columns
    ]
    return grouped


def plot_pareto(grouped, figures_dir, dpi):
    if grouped.empty:
        return
    methods = ["legacy_svd", "random_subspace", "tgprs"]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    legacy = grouped[grouped["method"] == "legacy_full"]
    ranks = grouped["requested_rank"].dropna().astype(int)
    color_min = int(ranks.min()) if len(ranks) else 1
    color_max = int(ranks.max()) if len(ranks) else 1
    normalization = plt.Normalize(color_min, color_max)
    color_map = plt.cm.viridis
    for axis, method in zip(axes, methods):
        current = grouped[grouped["method"] == method]
        for rank, values in current.groupby("requested_rank"):
            values = values.sort_values("retain_scale")
            color = color_map(normalization(int(rank)))
            axis.plot(
                values["retain_leakage_mean_mean"],
                values["target_rotation_deg_mean_mean"],
                color=color,
                marker="o",
                linewidth=1.2,
            )
        if not legacy.empty:
            axis.scatter(
                legacy["retain_leakage_mean_mean"],
                legacy["target_rotation_deg_mean_mean"],
                marker="x",
                color="black",
                label="Legacy full",
            )
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlabel("Retain leakage (lower is better)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Target rotation in degrees (higher is better)")
    axes[0].legend(fontsize=8)
    color_bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=color_map),
        ax=axes,
        pad=0.02,
    )
    color_bar.set_label("Requested residual rank q")
    figure.suptitle("Rank-controlled erasure–retention trade-off")
    save_figure(figure, figures_dir, "figure_c_rank_pareto", dpi)


def plot_direction_quality(grouped, figures_dir, dpi):
    if grouped.empty:
        return
    selected = grouped[grouped["method"].isin(
        ["legacy_svd", "random_subspace", "tgprs"]
    )].copy()
    summary = (
        selected.groupby(["method", "requested_rank"])[
            "directional_realization_cosine_mean_mean"
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for method, values in summary.groupby("method"):
        values = values.sort_values("requested_rank")
        axis.errorbar(
            values["requested_rank"],
            values["mean"],
            yerr=values["std"].fillna(0.0),
            marker="o",
            capsize=3,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xlabel("Requested residual rank q")
    axis.set_ylabel("Cosine between ΔWt and Wr")
    axis.set_title("Realization of the requested edit direction")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, figures_dir, "figure_d_direction_quality", dpi)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    figures_dir = input_dir / "figures"
    data_dir = input_dir / "figure_data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    geometry_rows = read_jsonl(input_dir / "geometry_metrics.jsonl")
    edit_rows = read_jsonl(input_dir / "edit_metrics.jsonl")
    if geometry_rows:
        geometry = flatten_geometry(geometry_rows)
        geometry.to_csv(data_dir / "geometry_metrics_flat.csv", index=False)
        rank_summary = plot_rank_inflation(geometry, figures_dir, args.dpi)
        rank_summary.to_csv(data_dir / "rank_summary.csv", index=False)
        energy = plot_spectral_energy(
            geometry_rows, input_dir / "spectra.npz", figures_dir, args.dpi
        )
        if not energy.empty:
            energy.to_csv(data_dir / "spectral_energy.csv", index=False)
    if edit_rows:
        edit_summary = aggregate_edits(edit_rows)
        edit_summary.to_csv(data_dir / "edit_summary.csv", index=False)
        plot_pareto(edit_summary, figures_dir, args.dpi)
        plot_direction_quality(edit_summary, figures_dir, args.dpi)
    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()
