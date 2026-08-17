#!/usr/bin/env python3
"""Measure existing TGPRS residuals before and after every UNet to_v weight."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.residual_subspace import (
    build_target_global_pairwise_residual_subspace_residuals,
)


DETAIL_FIELDS = [
    "target_index",
    "target_concept",
    "layer_index",
    "layer_name",
    "embedding_cosine",
    "embedding_angle_degrees",
    "target_embedding_norm",
    "shifted_embedding_norm",
    "residual_norm",
    "relative_embedding_residual_norm",
    "output_cosine",
    "output_angle_degrees",
    "target_output_norm",
    "shifted_output_norm",
    "output_residual_norm",
    "relative_output_residual_norm",
]

SUMMARY_METRICS = [
    "embedding_cosine",
    "embedding_angle_degrees",
    "residual_norm",
    "relative_embedding_residual_norm",
    "output_cosine",
    "output_angle_degrees",
    "output_residual_norm",
    "relative_output_residual_norm",
]


def cosine_and_angle(left, right, eps=1e-8):
    """Return row-wise cosine and angle in degrees using float32 arithmetic."""
    left = left.float()
    right = right.float()
    numerator = (left * right).sum(dim=-1)
    denominator = (
        torch.linalg.vector_norm(left, dim=-1)
        * torch.linalg.vector_norm(right, dim=-1)
    ).clamp_min(eps)
    cosine = (numerator / denominator).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    return cosine, angle


def build_detail_rows(
    target_concepts,
    target_embeddings,
    residuals,
    layer_weights,
    eps=1e-8,
):
    """Calculate embedding- and output-space metrics for target-layer pairs."""
    target_vectors = target_embeddings.reshape(len(target_concepts), -1).float()
    residual_vectors = residuals.reshape(len(target_concepts), -1).float()
    shifted_vectors = target_vectors + residual_vectors

    embedding_cosine, embedding_angle = cosine_and_angle(
        target_vectors, shifted_vectors, eps=eps
    )
    target_norm = torch.linalg.vector_norm(target_vectors, dim=1)
    shifted_norm = torch.linalg.vector_norm(shifted_vectors, dim=1)
    residual_norm = torch.linalg.vector_norm(residual_vectors, dim=1)
    relative_residual_norm = residual_norm / target_norm.clamp_min(eps)

    rows = []
    for layer_index, (layer_name, layer_weight) in enumerate(layer_weights):
        weight = layer_weight.detach().float()
        if weight.ndim != 2 or weight.shape[1] != target_vectors.shape[1]:
            raise ValueError(
                f"Layer {layer_name} has incompatible shape {tuple(weight.shape)}; "
                f"expected (*, {target_vectors.shape[1]})"
            )
        target_output = target_vectors @ weight.T
        shifted_output = shifted_vectors @ weight.T
        output_residual = residual_vectors @ weight.T
        output_cosine, output_angle = cosine_and_angle(
            target_output, shifted_output, eps=eps
        )
        target_output_norm = torch.linalg.vector_norm(target_output, dim=1)
        shifted_output_norm = torch.linalg.vector_norm(shifted_output, dim=1)
        output_residual_norm = torch.linalg.vector_norm(output_residual, dim=1)
        relative_output_residual_norm = (
            output_residual_norm / target_output_norm.clamp_min(eps)
        )

        for target_index, target_concept in enumerate(target_concepts):
            rows.append({
                "target_index": target_index,
                "target_concept": target_concept,
                "layer_index": layer_index,
                "layer_name": layer_name,
                "embedding_cosine": embedding_cosine[target_index].item(),
                "embedding_angle_degrees": embedding_angle[target_index].item(),
                "target_embedding_norm": target_norm[target_index].item(),
                "shifted_embedding_norm": shifted_norm[target_index].item(),
                "residual_norm": residual_norm[target_index].item(),
                "relative_embedding_residual_norm": relative_residual_norm[
                    target_index
                ].item(),
                "output_cosine": output_cosine[target_index].item(),
                "output_angle_degrees": output_angle[target_index].item(),
                "target_output_norm": target_output_norm[target_index].item(),
                "shifted_output_norm": shifted_output_norm[target_index].item(),
                "output_residual_norm": output_residual_norm[target_index].item(),
                "relative_output_residual_norm": relative_output_residual_norm[
                    target_index
                ].item(),
            })
    return rows


def summarize_rows(rows, group_field=None):
    """Summarize numeric metrics overall or by a detail-row field."""
    grouped = defaultdict(list)
    if group_field is None:
        grouped["overall"] = rows
    else:
        for row in rows:
            grouped[row[group_field]].append(row)

    summaries = []
    for group, group_rows in grouped.items():
        summary = {"group": group, "count": len(group_rows)}
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in group_rows]
            summary[f"{metric}_mean"] = fmean(values)
            summary[f"{metric}_min"] = min(values)
            summary[f"{metric}_max"] = max(values)
            summary[f"{metric}_std"] = pstdev(values)
        summaries.append(summary)
    return summaries


def write_csv(path, rows, fieldnames=None):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_target_concepts(experiment, repo_root):
    configured = experiment.get("target_concepts")
    if configured:
        concepts = [str(value).strip() for value in configured]
    else:
        raw_path = experiment.get("target_csv")
        if not raw_path:
            raise ValueError("experiment requires target_concepts or target_csv")
        path = Path(os.path.expandvars(str(raw_path)))
        if not path.is_absolute():
            path = repo_root / path
        concepts = []
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames or "concept" not in reader.fieldnames:
                raise ValueError(f"Target CSV lacks a concept column: {path}")
            for row in reader:
                concept = (row.get("concept") or "").strip()
                if row.get("type") == "erase" and concept and concept not in concepts:
                    concepts.append(concept)
    max_targets = int(experiment.get("max_targets", len(concepts)))
    concepts = concepts[:max_targets]
    if not concepts or any(not concept for concept in concepts):
        raise ValueError("At least one non-empty target concept is required")
    return concepts


@torch.no_grad()
def encode_last_subject_embeddings(text_encoder, tokenizer, concepts, device, batch_size):
    embeddings = []
    for start in range(0, len(concepts), batch_size):
        batch = concepts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        hidden_states = text_encoder(
            inputs.input_ids.to(device)
        ).last_hidden_state
        subject_indices = (inputs.attention_mask.sum(1) - 2).to(device)
        batch_indices = torch.arange(len(batch), device=device)
        embeddings.append(hidden_states[batch_indices, subject_indices].unsqueeze(1))
    return torch.cat(embeddings, dim=0)


def resolve_dtype(name):
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtypes[name]
    except KeyError as error:
        raise ValueError(f"Unsupported model dtype: {name}") from error


def make_heatmap(
    rows,
    target_concepts,
    layer_names,
    metric,
    title,
    colorbar_label,
    output_path,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = torch.empty((len(target_concepts), len(layer_names)))
    for row in rows:
        values[row["target_index"], row["layer_index"]] = row[metric]
    figure_width = max(10, 0.55 * len(layer_names))
    figure, axis = plt.subplots(figsize=(figure_width, 0.7 * len(target_concepts) + 2))
    image = axis.imshow(values.numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(layer_names)))
    axis.set_xticklabels(range(len(layer_names)), rotation=0)
    axis.set_xlabel("to_v layer index (see layer_summary.csv)")
    axis.set_yticks(range(len(target_concepts)))
    axis.set_yticklabels(target_concepts)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def make_scatter(rows, x_metric, y_metric, x_label, y_label, title, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6))
    points = axis.scatter(
        [row[x_metric] for row in rows],
        [row[y_metric] for row in rows],
        c=[row["layer_index"] for row in rows],
        cmap="viridis",
        alpha=0.8,
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    figure.colorbar(points, ax=axis, label="to_v layer index")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def serializable(value):
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def run(config_path, output_override=None):
    repo_root = REPO_ROOT
    os.environ.setdefault("REPO_ROOT", str(repo_root))
    os.environ.setdefault("WORKSPACE_DIR", str(repo_root.parent))
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or not isinstance(config.get("experiment"), dict):
        raise ValueError("Workflow config must contain an experiment mapping")

    experiment = config["experiment"]
    target_concepts = load_target_concepts(experiment, repo_root)
    residual_rank = int(experiment["residual_rank"])
    anchor_concept = str(experiment.get("anchor_concept", "person"))
    extra_anchor_concepts = experiment.get("extra_anchor_concepts", ["", "person"])
    if extra_anchor_concepts != ["", "person"]:
        raise ValueError("TGPRS requires extra_anchor_concepts to be ['', 'person']")
    batch_size = int(experiment.get("batch_size", 16))
    eps = float(experiment.get("eps", 1e-8))
    checkpoint = str(experiment["sd_ckpt"])
    dtype = resolve_dtype(str(experiment.get("model_dtype", "float32")))
    device = torch.device(str(experiment.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    raw_output = output_override or config.get("paths", {}).get(
        "output_root", "${WORKSPACE_DIR}/cedit_toys_cosine_output"
    )
    output_dir = Path(os.path.expandvars(str(raw_output))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    print(f"Loading tokenizer and text encoder: {checkpoint}")
    tokenizer = CLIPTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        checkpoint,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).to(device)
    text_encoder.eval()
    target_embeddings = encode_last_subject_embeddings(
        text_encoder, tokenizer, target_concepts, device, batch_size
    )
    anchor_embeddings = encode_last_subject_embeddings(
        text_encoder,
        tokenizer,
        [anchor_concept] * len(target_concepts),
        device,
        batch_size,
    )
    extra_anchor_embeddings = encode_last_subject_embeddings(
        text_encoder,
        tokenizer,
        extra_anchor_concepts,
        device,
        batch_size,
    )
    residuals, diagnostics = build_target_global_pairwise_residual_subspace_residuals(
        target_embeddings,
        anchor_embeddings,
        extra_anchor_embeddings,
        rank=residual_rank,
        eps=eps,
    )
    target_embeddings = target_embeddings.cpu()
    residuals = residuals.cpu()
    del text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Loading original UNet weights: {checkpoint}")
    unet = UNet2DConditionModel.from_pretrained(
        checkpoint,
        subfolder="unet",
        torch_dtype=dtype,
    )
    layer_weights = [
        (name, value.cpu())
        for name, value in unet.state_dict().items()
        if name.endswith("attn2.to_v.weight")
    ]
    if not layer_weights:
        raise RuntimeError("No attn2.to_v weights were found in the UNet")

    rows = build_detail_rows(
        target_concepts,
        target_embeddings,
        residuals,
        layer_weights,
        eps=eps,
    )
    layer_summaries = [
        {
            "layer_index": layer_index,
            "layer_name": summary.pop("group"),
            **summary,
        }
        for layer_index, summary in enumerate(summarize_rows(rows, "layer_name"))
    ]
    target_summaries = [
        {
            "target_index": target_index,
            "target_concept": summary.pop("group"),
            **summary,
        }
        for target_index, summary in enumerate(
            summarize_rows(rows, "target_concept")
        )
    ]
    overall = summarize_rows(rows)[0]

    write_csv(output_dir / "target_layer_metrics.csv", rows, DETAIL_FIELDS)
    write_csv(output_dir / "layer_summary.csv", layer_summaries)
    write_csv(output_dir / "target_summary.csv", target_summaries)
    make_heatmap(
        rows,
        target_concepts,
        [name for name, _ in layer_weights],
        "output_cosine",
        "Existing TGPRS output cosine by target and original to_v weight",
        "cosine similarity",
        output_dir / "output_cosine_heatmap.png",
    )
    make_heatmap(
        rows,
        target_concepts,
        [name for name, _ in layer_weights],
        "output_angle_degrees",
        "Existing TGPRS output angle by target and original to_v weight",
        "angle (degrees)",
        output_dir / "output_angle_heatmap.png",
    )
    make_scatter(
        rows,
        "residual_norm",
        "output_angle_degrees",
        "embedding residual norm",
        "output angle (degrees)",
        "TGPRS residual norm versus output angle",
        output_dir / "residual_norm_vs_output_angle.png",
    )
    make_scatter(
        rows,
        "embedding_angle_degrees",
        "output_angle_degrees",
        "embedding angle (degrees)",
        "output angle (degrees)",
        "TGPRS embedding angle versus output angle",
        output_dir / "embedding_angle_vs_output_angle.png",
    )
    metadata = {
        "checkpoint": checkpoint,
        "target_concepts": target_concepts,
        "anchor_concept": anchor_concept,
        "extra_anchor_concepts": extra_anchor_concepts,
        "requested_residual_rank": residual_rank,
        "layer_count": len(layer_weights),
        "target_layer_pair_count": len(rows),
        "tgprs_diagnostics": serializable(diagnostics),
        "overall_summary": overall,
    }
    with (output_dir / "analysis.json").open("w", encoding="utf-8") as json_file:
        json.dump(metadata, json_file, indent=2)

    print(
        "TGPRS output analysis complete | "
        f"targets={len(target_concepts)} | layers={len(layer_weights)} | "
        f"mean output cosine={overall['output_cosine_mean']:.6f} | "
        f"mean output angle={overall['output_angle_degrees_mean']:.3f} deg | "
        f"range={overall['output_angle_degrees_min']:.3f}-"
        f"{overall['output_angle_degrees_max']:.3f} deg"
    )
    print(f"Artifacts: {output_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
