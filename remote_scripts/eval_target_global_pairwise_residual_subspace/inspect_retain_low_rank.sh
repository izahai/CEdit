#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_workflow_config
cd -- "${REPO_ROOT}"

TRAIN_CONFIG="${WORKFLOW_DIR}/train_config.yaml"
require_file "${TRAIN_CONFIG}"
require_file "${REPO_ROOT}/train_erase_null.py"

if ! "${PYTHON_BIN}" -c 'import torch, transformers, yaml' >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m pip install \
        'transformers==4.48.0' accelerate PyYAML
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" USE_TF=0 TRANSFORMERS_NO_TF=1 \
"${PYTHON_BIN}" - \
    "${TRAIN_CONFIG}" "${BENCHMARK_NAMES_RAW}" "${REPO_ROOT}" <<'PY'
import csv
import os
import re
import sys
from pathlib import Path

import torch
import yaml
from transformers import CLIPTextModel, CLIPTokenizer


def ordered_unique(values):
    return list(dict.fromkeys(values))


def load_benchmark(path, head):
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    targets = ordered_unique(
        (row.get("concept") or "").strip()
        for row in rows
        if row.get("type") == "erase" and (row.get("concept") or "").strip()
    )
    retain_texts = ordered_unique(
        (row.get(head) or "").strip()
        for row in rows
        if (row.get(head) or "").strip()
    )
    retain_texts = [
        text
        for text in retain_texts
        if not any(
            re.search(
                r"\b" + re.escape(target.lower()) + r"\b",
                text.lower(),
            )
            for target in targets
        )
    ]
    if not retain_texts:
        raise ValueError(f"No retain texts remain after filtering {path}")
    return targets, retain_texts


@torch.no_grad()
def encode_last_subject_embeddings(
    texts,
    tokenizer,
    text_encoder,
    device,
    chunk_size=128,
):
    embeddings = []
    for start in range(0, len(texts), chunk_size):
        inputs = tokenizer(
            texts[start:start + chunk_size],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        hidden_states = text_encoder(
            inputs.input_ids.to(device)
        ).last_hidden_state
        subject_indices = (inputs.attention_mask.sum(1) - 2).to(device)
        batch_indices = torch.arange(hidden_states.shape[0], device=device)
        embeddings.append(hidden_states[batch_indices, subject_indices])
    return torch.cat(embeddings, dim=0)


config_path, benchmarks_raw, repo_root = sys.argv[1:]
repo_root = Path(repo_root)
with open(config_path, encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

threshold = float(os.environ.get("THRESHOLD", config["threshold"]))
model_id = os.environ.get("SD_CKPT", config["sd_ckpt"])
head = config["heads"].strip()
device = torch.device("cuda")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")

print(f"Loading tokenizer and text encoder: {model_id}")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
).to(device)
text_encoder.eval()

print(f"Retain-low threshold: {threshold:g}")
for benchmark_name in benchmarks_raw.split():
    benchmark_path = repo_root / "data" / f"{benchmark_name}.csv"
    targets, retain_texts = load_benchmark(benchmark_path, head)
    embeddings = encode_last_subject_embeddings(
        retain_texts,
        tokenizer,
        text_encoder,
        device,
    ).float()
    second_moment = embeddings.T @ embeddings / embeddings.shape[0]
    _, singular_values, _ = torch.svd(second_moment)
    retain_low_rank = int((singular_values < threshold).sum().item())
    print(
        f"{benchmark_name}: retain_texts={len(retain_texts)} | "
        f"erase_targets={len(targets)} | threshold={threshold:g} | "
        f"retain_low_rank={retain_low_rank}/{singular_values.numel()}"
    )
PY

printf 'Retain-low rank inspection complete. No training or evaluation was run.\n'
