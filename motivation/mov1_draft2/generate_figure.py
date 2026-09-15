#!/usr/bin/env python3
"""Build the real-CLIP common-anchor motivation figure."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from transformers import CLIPTextModel, CLIPTokenizer

from src.residual_subspace import (
    build_target_global_pairwise_residual_subspace_residuals,
)


def load_raw_celebrity_names(path, expected_count):
    names = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            name = (row.get("concept") or "").strip()
            if row.get("type") == "erase" and name and name not in seen:
                seen.add(name)
                names.append(name)
    if len(names) != expected_count:
        raise ValueError(
            f"Expected {expected_count} unique erase concepts, found {len(names)}"
        )
    return names


@torch.no_grad()
def encode_last_subject_tokens(tokenizer, text_encoder, prompts, device, batch_size):
    batches = []
    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start:start + batch_size]
        inputs = tokenizer(
            prompt_batch,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        hidden = text_encoder(
            inputs.input_ids.to(device)
        ).last_hidden_state
        subject_indices = (inputs.attention_mask.sum(dim=1) - 2).to(device)
        batch_indices = torch.arange(hidden.shape[0], device=device)
        batches.append(hidden[batch_indices, subject_indices])
    return torch.cat(batches, dim=0)


def orient_component(component):
    pivot = component.abs().argmax()
    return component if component[pivot] >= 0 else -component


def pca_projection(targets, anchor):
    points = torch.cat([targets, anchor], dim=0).float().cpu()
    centered = points - points.mean(dim=0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    components = torch.stack([
        orient_component(vh[0]),
        orient_component(vh[1]),
    ])
    coordinates = centered @ components.T
    explained = singular_values.square()
    explained = explained[:2] / explained.sum()
    return coordinates, explained


def spectral_summary(matrix):
    matrix = matrix.reshape(matrix.shape[0], -1).float().cpu()
    singular_values = torch.linalg.svdvals(matrix)
    maximum = singular_values.max()
    relative = singular_values / maximum.clamp_min(
        torch.finfo(singular_values.dtype).eps
    )
    relative_tolerance = (
        max(matrix.shape) * torch.finfo(singular_values.dtype).eps
    )
    numerical_rank = int((relative > relative_tolerance).sum().item())
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(
        torch.finfo(energy.dtype).eps
    )
    nonzero = probabilities > 0
    effective_rank = math.exp(
        -(probabilities[nonzero] * probabilities[nonzero].log()).sum().item()
    )
    return {
        "singular_values": singular_values.tolist(),
        "relative_singular_values": relative.tolist(),
        "relative_rank_tolerance": relative_tolerance,
        "numerical_rank": numerical_rank,
        "stable_rank": (energy.sum() / maximum.square()).item(),
        "spectral_effective_rank": effective_rank,
    }


def write_projection_csv(path, names, coordinates):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["prompt", "kind", "pc1", "pc2"],
        )
        writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow({
                "prompt": name,
                "kind": "target",
                "pc1": coordinates[index, 0].item(),
                "pc2": coordinates[index, 1].item(),
            })
        writer.writerow({
            "prompt": "person",
            "kind": "anchor",
            "pc1": coordinates[-1, 0].item(),
            "pc2": coordinates[-1, 1].item(),
        })


def write_spectrum_csv(path, summaries):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["method", "component", "singular_value", "relative_singular_value"],
        )
        writer.writeheader()
        for method, summary in summaries.items():
            for component, (value, relative) in enumerate(
                zip(
                    summary["singular_values"],
                    summary["relative_singular_values"],
                ),
                start=1,
            ):
                writer.writerow({
                    "method": method,
                    "component": component,
                    "singular_value": value,
                    "relative_singular_value": relative,
                })


def render_figure(
    names,
    coordinates,
    explained,
    summaries,
    rank,
    output_dir,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    anchor_xy = coordinates[-1].numpy()
    for point in coordinates[:-1].numpy():
        geometry_axis.plot(
            [point[0], anchor_xy[0]],
            [point[1], anchor_xy[1]],
            color=colors["legacy"],
            alpha=0.10,
            linewidth=0.55,
            zorder=1,
        )
    geometry_axis.scatter(
        coordinates[:-1, 0],
        coordinates[:-1, 1],
        s=12,
        facecolors="none",
        edgecolors=colors["legacy"],
        linewidths=0.7,
        label="Celebrity prompts",
        zorder=2,
    )
    geometry_axis.scatter(
        [anchor_xy[0]],
        [anchor_xy[1]],
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
        anchor_xy,
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=7,
        fontweight="bold",
    )
    geometry_axis.axhline(0, color="#BBBBBB", linewidth=0.45, zorder=0)
    geometry_axis.axvline(0, color="#BBBBBB", linewidth=0.45, zorder=0)
    geometry_axis.set_xlabel(f"PC1 ({100 * explained[0].item():.1f}% variance)")
    geometry_axis.set_ylabel(f"PC2 ({100 * explained[1].item():.1f}% variance)")
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
    line_styles = {"legacy": "-", "tgprs": "--"}
    for method in ("legacy", "tgprs"):
        summary = summaries[method]
        values = [max(value, 1e-8) for value in summary["relative_singular_values"]]
        components = range(1, len(values) + 1)
        spectrum_axis.plot(
            components,
            values,
            color=colors[method],
            linestyle=line_styles[method],
            linewidth=1.6,
            label=f"{labels[method]} ($r={summary['numerical_rank']}$)",
        )
    tolerance = summaries["legacy"]["relative_rank_tolerance"]
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
    spectrum_axis.set_xlim(1, len(names))
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


@torch.no_grad()
def run(config, output_dir):
    target_count = int(config["target_count"])
    benchmark_path = REPO_ROOT / config["benchmark_csv"]
    names = load_raw_celebrity_names(benchmark_path, target_count)
    anchor_prompt = str(config["anchor_prompt"])
    if anchor_prompt != "person":
        raise ValueError('MOV2 fixes the shared anchor prompt to exactly "person"')

    device = torch.device("cuda")
    tokenizer = CLIPTokenizer.from_pretrained(
        config["model_id"],
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config["model_id"],
        subfolder="text_encoder",
    ).to(device)
    text_encoder.eval()
    targets = encode_last_subject_tokens(
        tokenizer,
        text_encoder,
        names,
        device,
        int(config["embedding_batch_size"]),
    )
    anchor = encode_last_subject_tokens(
        tokenizer,
        text_encoder,
        [anchor_prompt],
        device,
        1,
    )
    anchors = anchor.expand_as(targets)
    legacy = anchors - targets
    rank = int(config["residual_rank"])
    tgprs, tgprs_diagnostics = (
        build_target_global_pairwise_residual_subspace_residuals(
            targets.unsqueeze(1),
            anchors.unsqueeze(1),
            extra_anchor_embeddings=anchor.unsqueeze(1),
            rank=rank,
        )
    )
    tgprs = tgprs.squeeze(1)

    summaries = {
        "legacy": spectral_summary(legacy),
        "tgprs": spectral_summary(tgprs),
    }
    target_matrix = targets.float().cpu()
    edit_ranks = {}
    for method, residuals in (("legacy", legacy), ("tgprs", tgprs)):
        edit_statistic = residuals.float().cpu().T @ target_matrix / target_count
        edit_ranks[method] = spectral_summary(edit_statistic)["numerical_rank"]

    coordinates, explained = pca_projection(targets, anchor)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_projection_csv(
        output_dir / "embedding_projection.csv",
        names,
        coordinates,
    )
    write_spectrum_csv(output_dir / "residual_spectrum.csv", summaries)
    metadata = {
        "model_id": config["model_id"],
        "target_prompts": "raw celebrity names from concept column",
        "anchor_prompt": anchor_prompt,
        "embedding": "last non-special subject-token hidden state",
        "clip_dimension": int(targets.shape[1]),
        "target_count": target_count,
        "tgprs_rank": rank,
        "tgprs_extra_anchor_prompts": [anchor_prompt],
        "pca_explained_variance": explained.tolist(),
        "residual_summaries": summaries,
        "edit_statistic_numerical_rank": edit_ranks,
        "tgprs_diagnostics": tgprs_diagnostics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    render_figure(
        names,
        coordinates,
        explained,
        summaries,
        rank,
        output_dir,
    )
    print(
        "Residual ranks:",
        {method: summary["numerical_rank"] for method, summary in summaries.items()},
    )
    print("Edit-statistic ranks:", edit_ranks)
    print(f"Saved MOV2 outputs to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
