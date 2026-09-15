#!/usr/bin/env python3
"""Load the MOV1 workflow config and emit shell-safe environment values."""

import argparse
import os
import shlex
from pathlib import Path

import yaml


METHODS = ("legacy_full", "legacy_svd_rank30", "tgprs_rank30")


def required(config, section, key):
    try:
        return config[section][key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing workflow config value: {section}.{key}") from error


def _string_list(config, section, key):
    values = required(config, section, key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{section}.{key} must be a non-empty list")
    return " ".join(str(value) for value in values)


def _override(name, default):
    return os.environ.get(name, default)


def build_environment(config, workflow_dir):
    methods = required(config, "experiment", "methods")
    if tuple(methods) != METHODS:
        raise ValueError(f"experiment.methods must be exactly {list(METHODS)!r}")
    workspace_dir = os.environ["WORKSPACE_DIR"]
    repo_root = os.environ["REPO_ROOT"]
    values = {
        "PYTHON_BIN": _override(
            "PYTHON_BIN", required(config, "runtime", "python_bin")
        ),
        "GPU_ID": _override("GPU_ID", required(config, "runtime", "gpu_id")),
        "CE_EVAL_REPOSITORY": _override(
            "CE_EVAL_REPOSITORY",
            required(config, "repositories", "ce_eval_repository"),
        ),
        "CE_EVAL_BRANCH": _override(
            "CE_EVAL_BRANCH", required(config, "repositories", "ce_eval_branch")
        ),
        "SD_CKPT": _override(
            "SD_CKPT", required(config, "experiment", "sd_ckpt")
        ),
        "BENCHMARK_NAME": _override(
            "BENCHMARK_NAME", required(config, "experiment", "benchmark_name")
        ),
        "ANCHOR_CONCEPT": _override(
            "ANCHOR_CONCEPT", required(config, "experiment", "anchor_concept")
        ),
        "METHODS_RAW": _override(
            "METHODS_RAW", _string_list(config, "experiment", "methods")
        ),
        "RESIDUAL_SCALES_RAW": _override(
            "RESIDUAL_SCALES_RAW",
            _string_list(config, "experiment", "residual_scales"),
        ),
        "RESIDUAL_RANK": _override(
            "RESIDUAL_RANK", required(config, "experiment", "residual_rank")
        ),
        "BATCH_SIZE": _override(
            "BATCH_SIZE", required(config, "experiment", "batch_size")
        ),
        "EMBEDDING_BATCH_SIZE": _override(
            "EMBEDDING_BATCH_SIZE",
            required(config, "experiment", "embedding_batch_size"),
        ),
        "INFERENCE_TIMESTEPS": _override(
            "INFERENCE_TIMESTEPS",
            required(config, "experiment", "inference_timesteps"),
        ),
        "EXPECTED_IMAGES_PER_SPLIT": _override(
            "EXPECTED_IMAGES_PER_SPLIT",
            required(config, "experiment", "expected_images_per_split"),
        ),
        "GCD_USE_CUDA": _override(
            "GCD_USE_CUDA", required(config, "experiment", "gcd_use_cuda")
        ),
        "GCD_NUM_WORKERS": _override(
            "GCD_NUM_WORKERS", required(config, "experiment", "gcd_num_workers")
        ),
        "GCD_BATCH_SIZE": _override(
            "GCD_BATCH_SIZE", required(config, "experiment", "gcd_batch_size")
        ),
        "GCD_PREFETCH_FACTOR": _override(
            "GCD_PREFETCH_FACTOR",
            required(config, "experiment", "gcd_prefetch_factor"),
        ),
    }
    os.environ.setdefault("WORKSPACE_DIR", workspace_dir)
    os.environ.setdefault("REPO_ROOT", repo_root)
    ce_eval_root = os.path.expandvars(required(config, "paths", "ce_eval_root"))
    output_root = os.path.expandvars(required(config, "paths", "output_root"))
    values.update({
        "WORKFLOW_DIR": workflow_dir,
        "CE_EVAL_ROOT": _override("CE_EVAL_ROOT", ce_eval_root),
        "OUTPUT_ROOT": _override("OUTPUT_ROOT", output_root),
    })
    values["CHECKPOINT_ROOT"] = os.path.join(values["OUTPUT_ROOT"], "checkpoints")
    values["IMAGE_ROOT"] = os.path.join(values["OUTPUT_ROOT"], "images")
    values["GCD_OUTPUT_DIR"] = os.path.join(values["OUTPUT_ROOT"], "gcd")
    values["ANALYSIS_DIR"] = os.path.join(values["OUTPUT_ROOT"], "analysis")
    values["SUMMARY_DIR"] = os.path.join(values["OUTPUT_ROOT"], "summary")
    values["FIGURE_DIR"] = os.path.join(values["OUTPUT_ROOT"], "figures")
    values["LOG_DIR"] = os.path.join(values["OUTPUT_ROOT"], "logs")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("command", choices=["export"])
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        parser.error("Workflow YAML must contain a mapping")
    try:
        values = build_environment(config, str(args.config.parent.resolve()))
    except ValueError as error:
        parser.error(str(error))
    for key, value in values.items():
        if isinstance(value, bool):
            value = str(value).lower()
        print(f"{key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()

