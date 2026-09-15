#!/usr/bin/env python3
"""Create paper-ready figures from rank-analysis JSONL outputs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


METHOD_LABELS = {
    "legacy": "Legacy",
    "legacy_full": "Legacy full",
    "legacy_svd": "Legacy-SVD",
    "random_subspace": "Random subspace",
    "tgprs": "TGPRS",
    "tgprs_positive": r"TGPRS $+P_S t$",
    "tgprs_complement": r"TGPRS $-P_{S^\perp}t$",
    "random_negative_target": r"$-P_{S_{random}}t$",
}
METHOD_COLORS = {
    "legacy": "#222222",
    "legacy_full": "#222222",
    "legacy_svd": "#377eb8",
    "random_subspace": "#999999",
    "tgprs": "#e41a1c",
    "tgprs_positive": "#984ea3",
    "tgprs_complement": "#ff7f00",
    "random_negative_target": "#4daf4a",
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


def plot_rank_inflation(frame, delta_rows, figures_dir, dpi):
    selected = frame[frame["matrix"].isin(["residual", "edit_statistic"])]
    grouped = selected.groupby(
        ["method", "requested_rank", "target_count", "matrix"], dropna=False
    )["effective_rank"].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    delta = pd.DataFrame([
        row for row in delta_rows if row.get("layer_index") == "aggregate"
    ])
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), sharex=True)
    panels = [("residual", "Residual R"), ("edit_statistic", "Edit statistic D")]
    for axis, (matrix_name, title) in zip(axes[:2], panels):
        current = grouped[grouped["matrix"] == matrix_name]
        for (method, rank), values in current.groupby(
            ["method", "requested_rank"], dropna=False
        ):
            values = values.sort_values("target_count")
            color = curve_color(method, rank)
            axis.plot(values["target_count"], values["mean"], color=color,
                      marker="o", label=curve_label(method, rank))
            axis.fill_between(values["target_count"], values["mean"] - values["std"],
                              values["mean"] + values["std"], color=color, alpha=0.10)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    if not delta.empty:
        delta_grouped = delta.groupby(
            ["method", "requested_rank", "target_count"], dropna=False
        )["delta_effective_rank_mean"].agg(["mean", "std"]).reset_index()
        delta_grouped["std"] = delta_grouped["std"].fillna(0.0)
        for (method, rank), values in delta_grouped.groupby(
            ["method", "requested_rank"], dropna=False
        ):
            values = values.sort_values("target_count")
            color = curve_color(method, rank)
            axes[2].plot(values["target_count"], values["mean"], color=color,
                         marker="o", label=curve_label(method, rank))
            axes[2].fill_between(values["target_count"], values["mean"] - values["std"],
                                 values["mean"] + values["std"], color=color, alpha=0.10)
    axes[2].set_title("Weight update delta W")
    axes[2].grid(alpha=0.25)
    for axis in axes:
        axis.set_xlabel("Number of erased targets N")
    axes[0].set_ylabel("Effective rank")
    axes[0].legend(fontsize=8, ncol=2)
    figure.suptitle("Rank propagation through SPEED's closed-form update")
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
    if "retain_profile" not in frame:
        frame["retain_profile"] = "fixed"
    frame["retain_profile"] = frame["retain_profile"].fillna("fixed")
    if "direction_variant" not in frame:
        frame["direction_variant"] = frame["method"]
    if "retain_threshold" not in frame:
        frame["retain_threshold"] = 0.1
    metric_names = [
        name for name in (
            "target_effect_mean",
            "target_rotation_deg_mean",
            "retain_leakage_mean",
            "directional_realization_cosine_mean",
            "directional_relative_error_mean",
            "edited_output_norm_ratio_mean",
            "canonical_erasure_alignment_mean",
            "anchor_distance_ratio_mean",
            "retain_low_rank_mean",
        ) if name in frame.columns
    ]
    grouped = (
        frame.groupby(
            ["method", "direction_variant", "requested_rank", "retain_profile",
             "retain_threshold", "retain_scale"], dropna=False
        )[metric_names]
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
    grouped = grouped[
        (grouped["retain_profile"] == "fixed")
        & np.isclose(grouped["retain_threshold"], 0.1)
    ]
    methods = ["legacy_svd", "random_subspace", "tgprs"]
    figure = plt.figure(figsize=(15.5, 4.6))
    grid = figure.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045])
    axes = []
    for index in range(3):
        shared_axis = axes[0] if axes else None
        axes.append(
            figure.add_subplot(
                grid[0, index], sharex=shared_axis, sharey=shared_axis
            )
        )
    color_axis = figure.add_subplot(grid[0, 3])
    legacy = grouped[grouped["method"] == "legacy_full"]
    ranks = grouped["requested_rank"].dropna().astype(int)
    color_min = int(ranks.min()) if len(ranks) else 1
    color_max = int(ranks.max()) if len(ranks) else 1
    normalization = LogNorm(max(color_min, 1), max(color_max, 1))
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
        axis.set_xscale("log")
        axis.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Target rotation in degrees (higher is better)")
    axes[0].legend(fontsize=8)
    color_bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=color_map),
        cax=color_axis,
    )
    color_bar.set_label("Requested residual rank q")
    figure.suptitle("Rank-controlled erasure–retention trade-off")
    save_figure(figure, figures_dir, "figure_c_rank_pareto", dpi)


def plot_direction_quality(grouped, figures_dir, dpi):
    if grouped.empty:
        return
    selected = grouped[
        grouped["method"].isin(["legacy_svd", "random_subspace", "tgprs"])
        & (grouped["retain_profile"] == "fixed")
        & np.isclose(grouped["retain_threshold"], 0.1)
    ].copy()
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


def plot_common_anchor(path, figures_dir, dpi):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    targets = np.asarray(data["target_coordinates"], dtype=float)
    anchor = np.asarray(data["anchor_coordinate"], dtype=float)
    cosines = np.asarray(data["residual_cosine_similarity"], dtype=float)
    names = data["target_names"]
    explained = 100.0 * sum(data["pca_explained_variance_ratio"])
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.3))
    for index, (point, name) in enumerate(zip(targets, names)):
        axes[0].annotate(
            "", xy=anchor, xytext=point,
            arrowprops={"arrowstyle": "->", "alpha": 0.65, "color": plt.cm.tab10(index % 10)},
        )
        axes[0].scatter(*point, color=plt.cm.tab10(index % 10), s=28)
        axes[0].annotate(name, point, fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].scatter(*anchor, marker="*", s=180, color="black", label=data["anchor_name"])
    axes[0].set_title(f"Targets point to one anchor (PCA-2D, {explained:.1f}% variance)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    image = axes[1].imshow(cosines, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[1].set_xticks(range(len(names)), names, rotation=75, ha="right", fontsize=7)
    axes[1].set_yticks(range(len(names)), names, fontsize=7)
    rank = data["residual"]
    axes[1].set_title(
        f"Residual cosine similarity\nrank={rank['numerical_rank']}, "
        f"effective rank={rank['effective_rank']:.2f}"
    )
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    figure.suptitle("A common anchor does not imply a common edit direction")
    save_figure(figure, figures_dir, "figure_0_common_anchor_directions", dpi)


def plot_threshold_sensitivity(grouped, figures_dir, dpi):
    if grouped.empty:
        return
    selected = grouped[
        (grouped["retain_profile"] == "fixed")
        & np.isclose(grouped["retain_scale"], 0.1)
        & (
            (grouped["method"] == "legacy_full")
            | ((grouped["method"].isin(["legacy_svd", "random_subspace", "tgprs"]))
               & np.isclose(grouped["requested_rank"], 30))
        )
    ]
    if selected.empty or selected["retain_threshold"].nunique() < 2:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True)
    for (method, rank), values in selected.groupby(
        ["method", "requested_rank"], dropna=False
    ):
        values = values.sort_values("retain_threshold")
        label = curve_label(method, rank)
        color = METHOD_COLORS.get(method)
        axes[0].plot(values["retain_threshold"], values["target_rotation_deg_mean_mean"],
                     marker="o", label=label, color=color)
        axes[1].plot(values["retain_threshold"], values["retain_leakage_mean_mean"],
                     marker="o", label=label, color=color)
    axes[0].set_ylabel("Target rotation (degrees)")
    axes[1].set_ylabel("Retain leakage")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("Retain singular-value threshold")
        axis.set_xscale("log")
        axis.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8)
    figure.suptitle("Sensitivity to the retain-low threshold at retain scale 0.1")
    save_figure(figure, figures_dir, "figure_e_threshold_sensitivity", dpi)


def plot_retain_profile_robustness(grouped, figures_dir, dpi):
    if grouped.empty or "full_speed" not in set(grouped["retain_profile"]):
        return
    selected = grouped[
        np.isclose(grouped["retain_threshold"], 0.1)
        & np.isclose(grouped["retain_scale"], 0.1)
        & ((grouped["method"] == "legacy_full") | (
            (grouped["method"] == "tgprs")
            & grouped["requested_rank"].isin([5, 10, 30])
        ))
    ].copy()
    if selected.empty:
        return
    selected["label"] = selected.apply(
        lambda row: curve_label(row["method"], row["requested_rank"]), axis=1
    )
    labels = list(dict.fromkeys(selected["label"].tolist()))
    x = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for profile_index, profile in enumerate(["fixed", "full_speed"]):
        current = selected[selected["retain_profile"] == profile].set_index("label")
        offset = (profile_index - 0.5) * width
        rotation = [current.loc[label, "target_rotation_deg_mean_mean"] for label in labels]
        leakage = [current.loc[label, "retain_leakage_mean_mean"] for label in labels]
        axes[0].bar(x + offset, rotation, width, label=profile)
        axes[1].bar(x + offset, leakage, width, label=profile)
    axes[0].set_ylabel("Target rotation (degrees)")
    axes[1].set_ylabel("Retain leakage")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.suptitle("Fixed versus full-SPEED retain construction")
    save_figure(figure, figures_dir, "figure_f_retain_profile_robustness", dpi)


def plot_direction_ablation(grouped, figures_dir, dpi):
    metrics = [
        ("target_rotation_deg_mean_mean", "Target rotation (degrees)"),
        ("canonical_erasure_alignment_mean_mean", r"Alignment with $-W P_S t$"),
        ("retain_leakage_mean_mean", "Retain leakage"),
    ]
    if grouped.empty or any(name not in grouped for name, _ in metrics):
        return
    methods = ["tgprs", "tgprs_positive", "tgprs_complement", "random_negative_target"]
    selected = grouped[
        grouped["method"].isin(methods)
        & (grouped["retain_profile"] == "fixed")
        & np.isclose(grouped["retain_threshold"], 0.1)
        & np.isclose(grouped["retain_scale"], 0.1)
    ]
    if selected.empty:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    for axis, (metric, ylabel) in zip(axes, metrics):
        for method, values in selected.groupby("method"):
            values = values.sort_values("requested_rank")
            mean = values[metric]
            std_name = metric.replace("_mean_mean", "_mean_std")
            std = values[std_name].fillna(0.0) if std_name in values else None
            axis.errorbar(values["requested_rank"], mean, yerr=std, marker="o",
                          capsize=3, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        axis.set_xlabel("Requested rank q")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log")
        axis.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8)
    figure.suptitle("Direction ablations at threshold=0.1 and retain scale=0.1")
    save_figure(figure, figures_dir, "figure_g_direction_ablation", dpi)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    figures_dir = input_dir / "figures"
    data_dir = input_dir / "figure_data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    geometry_rows = read_jsonl(input_dir / "geometry_metrics.jsonl")
    delta_rows = read_jsonl(input_dir / "delta_rank_metrics.jsonl")
    edit_rows = read_jsonl(input_dir / "edit_metrics.jsonl")
    plot_common_anchor(
        input_dir / "common_anchor_geometry.json", figures_dir, args.dpi
    )
    if geometry_rows:
        geometry = flatten_geometry(geometry_rows)
        geometry.to_csv(data_dir / "geometry_metrics_flat.csv", index=False)
        rank_summary = plot_rank_inflation(
            geometry, delta_rows, figures_dir, args.dpi
        )
        rank_summary.to_csv(data_dir / "rank_summary.csv", index=False)
        energy = plot_spectral_energy(
            geometry_rows, input_dir / "spectra.npz", figures_dir, args.dpi
        )
        if not energy.empty:
            energy.to_csv(data_dir / "spectral_energy.csv", index=False)
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(
            data_dir / "delta_rank_metrics.csv", index=False
        )
    if edit_rows:
        edit_summary = aggregate_edits(edit_rows)
        edit_summary.to_csv(data_dir / "edit_summary.csv", index=False)
        plot_pareto(edit_summary, figures_dir, args.dpi)
        plot_direction_quality(edit_summary, figures_dir, args.dpi)
        plot_threshold_sensitivity(edit_summary, figures_dir, args.dpi)
        plot_retain_profile_robustness(edit_summary, figures_dir, args.dpi)
        plot_direction_ablation(edit_summary, figures_dir, args.dpi)
    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()
