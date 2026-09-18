"""Shared input loading, prompt caching, and diffusion state helpers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import os
import random
import re
from collections.abc import Sequence
from pathlib import Path

import torch

from src.closed_form_anchor_training import sample_prefix_diffusion_state


def load_prompt_csv(path: str | os.PathLike[str]) -> list[str]:
    """Load and validate prompts from a CSV containing a 'prompt' column."""
    prompts: list[str] = []
    with open(path, newline="", encoding="utf-8") as prompt_file:
        reader = csv.DictReader(prompt_file)
        if not reader.fieldnames or "prompt" not in reader.fieldnames:
            raise ValueError(f"Prompt CSV must contain a 'prompt' column: {path}")
        for row_number, row in enumerate(reader, start=2):
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                raise ValueError(f"Prompt CSV contains a blank prompt at row {row_number}: {path}")
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"Prompt CSV contains no prompts: {path}")
    return prompts


def resolve_prompts(path: str | None, fallback: Sequence[str]) -> list[str]:
    """Resolve prompt list from path if provided, else fallback to sequence."""
    return load_prompt_csv(path) if path is not None else list(fallback)


def load_retain_texts(
    path: str | os.PathLike[str],
    heads: str,
    targets: Sequence[str],
) -> list[str]:
    """Load retain concept strings from CSV, excluding target concepts."""
    if not str(path).endswith(".csv"):
        raise ValueError("retain_path must point to a CSV file")
    requested_heads = [head.strip() for head in heads.split(",") if head.strip()]
    if not requested_heads:
        raise ValueError("heads must contain at least one CSV column")
    values: list[str] = []
    seen = set()
    with open(path, newline="", encoding="utf-8") as retain_file:
        reader = csv.DictReader(retain_file)
        missing = [head for head in requested_heads if head not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("Retain CSV is missing column(s): " + ", ".join(missing))
        for row_number, row in enumerate(reader, start=2):
            for head in requested_heads:
                text = (row.get(head) or "").strip()
                if not text:
                    raise ValueError(
                        f"Retain CSV contains a blank {head!r} value at row {row_number}"
                    )
                if text not in seen:
                    seen.add(text)
                    values.append(text)

    filtered = [
        text
        for text in values
        if not any(
            re.search(r"\b" + re.escape(target.lower()) + r"\b", text.lower())
            for target in targets
        )
    ]
    if not filtered:
        raise ValueError("Retain set is empty after target exclusion")
    return filtered


def load_candidate_concepts(
    path: str | os.PathLike[str],
) -> tuple[list[str], str]:
    """Load candidate bank from CSV or text file in order, returning concepts and sha256 hash.

    Rejects empty banks, blank concepts, and duplicates.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Candidate concepts file not found: {path}")

    file_hash = sha256_file(path_obj)
    candidates: list[str] = []
    seen = set()

    if str(path_obj).endswith(".csv"):
        with open(path_obj, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            col = None
            for candidate_col in ("concept", "prompt"):
                if reader.fieldnames and candidate_col in reader.fieldnames:
                    col = candidate_col
                    break
            if col is None:
                raise ValueError(
                    f"Candidate CSV must contain a 'concept' or 'prompt' column: {path}"
                )
            for row_num, row in enumerate(reader, start=2):
                val = (row.get(col) or "").strip()
                if not val:
                    raise ValueError(
                        f"Candidate CSV contains a blank concept at row {row_num}: {path}"
                    )
                if val in seen:
                    raise ValueError(
                        f"Candidate bank contains duplicate concept {val!r} at row {row_num}: {path}"
                    )
                seen.add(val)
                candidates.append(val)
    else:
        with open(path_obj, encoding="utf-8") as txt_file:
            for line_num, line in enumerate(txt_file, start=1):
                val = line.strip()
                if not val:
                    raise ValueError(
                        f"Candidate file contains a blank line at line {line_num}: {path}"
                    )
                if val in seen:
                    raise ValueError(
                        f"Candidate bank contains duplicate concept {val!r} at line {line_num}: {path}"
                    )
                seen.add(val)
                candidates.append(val)

    if not candidates:
        raise ValueError(f"Candidate bank is empty: {path}")

    return candidates, file_hash


@torch.no_grad()
def encode_prompts(pipe, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
    """Encode full prompt hidden states using the text encoder."""
    tokens = pipe.tokenizer(
        list(prompts),
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return pipe.text_encoder(tokens.input_ids.to(device)).last_hidden_state


@torch.no_grad()
def encode_last_subject_embeddings(
    pipe,
    prompts: Sequence[str],
    device: torch.device,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Extract contextual subject embedding for each prompt (shape [N, 1, d])."""
    embeddings = []
    for start in range(0, len(prompts), chunk_size):
        values = list(prompts[start : start + chunk_size])
        tokens = pipe.tokenizer(
            values,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        hidden = pipe.text_encoder(tokens.input_ids.to(device)).last_hidden_state
        subject_indices = (tokens.attention_mask.sum(1) - 2).to(device)
        batch_indices = torch.arange(hidden.shape[0], device=device)
        embeddings.append(hidden[batch_indices, subject_indices].unsqueeze(1))
    return torch.cat(embeddings)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)


def build_state(
    pipe,
    hidden_cache: dict[str, torch.Tensor],
    prompts: Sequence[str],
    null_hidden: torch.Tensor,
    args: argparse.Namespace,
    *,
    seed: int,
    prefix_index: int,
):
    """Build a diffusion prefix state from cached prompt hidden states."""
    hidden = torch.cat([hidden_cache[prompt] for prompt in prompts])
    num_steps = (
        getattr(args, "num_train_inference_steps", None)
        or getattr(args, "num_inference_steps", 50)
    )
    return sample_prefix_diffusion_state(
        pipe,
        hidden,
        null_hidden,
        prompt=" | ".join(prompts),
        seed=seed,
        prefix_index=prefix_index,
        num_inference_steps=num_steps,
        guidance_scale=args.guidance_scale,
        resolution=args.resolution,
        generator=_make_generator(pipe.unet.device, seed),
    )


def build_validation_bank(
    pipe,
    hidden_cache: dict[str, torch.Tensor],
    prompts: Sequence[str],
    null_hidden: torch.Tensor,
    args: argparse.Namespace,
    *,
    seed_offset: int = 0,
) -> list:
    """Construct deterministic held-out validation diffusion states."""
    states = []
    rng = random.Random(args.validation_seed + seed_offset)
    num_steps = (
        getattr(args, "num_train_inference_steps", None)
        or getattr(args, "num_inference_steps", 50)
    )
    for index in range(args.validation_samples):
        prompt = prompts[index % len(prompts)]
        seed = args.validation_seed + seed_offset + index
        prefix_index = rng.randrange(num_steps)
        states.append(
            build_state(
                pipe,
                hidden_cache,
                [prompt],
                null_hidden,
                args,
                seed=seed,
                prefix_index=prefix_index,
            )
        )
    return states


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Compute sha256 hex digest for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_package_versions() -> dict[str, str]:
    """Retrieve runtime package versions for audit reproducibility."""
    versions = {"torch": torch.__version__, "cuda": str(torch.version.cuda)}
    for package in ("diffusers", "transformers", "safetensors", "kmeans-pytorch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def cpu_tensor_dict(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Clone and move a dictionary of tensors to CPU contiguous memory."""
    return {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in values.items()
    }

