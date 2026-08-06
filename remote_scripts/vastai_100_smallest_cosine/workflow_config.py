#!/usr/bin/env python3
"""Load the Vast workflow YAML and print shell-safe environment assignments."""

import argparse
import os
import shlex
from pathlib import Path

import yaml


def required(config, section, key):
    try:
        return config[section][key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing workflow config value: {section}.{key}") from error


def get_value(name, default):
    return os.environ.get(name, default)


def stringify(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def build_environment(config, workflow_dir):
    workspace_dir = os.environ["WORKSPACE_DIR"]
    repo_root = os.environ["REPO_ROOT"]
    values = {
        "PYTHON_BIN": get_value("PYTHON_BIN", required(config, "runtime", "python_bin")),
        "GPU_ID": get_value("GPU_ID", required(config, "runtime", "gpu_id")),
        "CEDIT_BRANCH": get_value("CEDIT_BRANCH", required(config, "repositories", "cedit_branch")),
        "CEDIT_REPOSITORY": get_value("CEDIT_REPOSITORY", required(config, "repositories", "cedit_repository")),
        "CE_EVAL_REPOSITORY": get_value("CE_EVAL_REPOSITORY", required(config, "repositories", "ce_eval_repository")),
        "CE_EVAL_BRANCH": get_value("CE_EVAL_BRANCH", required(config, "repositories", "ce_eval_branch")),
        "SD_CKPT": get_value("SD_CKPT", required(config, "experiment", "sd_ckpt")),
        "BENCHMARK_NAME": get_value("BENCHMARK_NAME", required(config, "experiment", "benchmark_name")),
        "ANCHOR_MODE": get_value("ANCHOR_MODE", required(config, "experiment", "anchor_mode")),
        "BATCH_SIZE": get_value("BATCH_SIZE", required(config, "experiment", "batch_size")),
        "EXPECTED_IMAGE_COUNT": get_value("EXPECTED_IMAGE_COUNT", required(config, "experiment", "expected_image_count")),
        "GCD_USE_CUDA": get_value("GCD_USE_CUDA", required(config, "experiment", "gcd_use_cuda")),
    }
    ce_eval_root = os.path.expandvars(required(config, "paths", "ce_eval_root"))
    output_root = os.path.expandvars(required(config, "paths", "output_root"))
    values.update({
        "WORKFLOW_DIR": workflow_dir,
        "CE_EVAL_ROOT": get_value("CE_EVAL_ROOT", ce_eval_root),
        "OUTPUT_ROOT": get_value("OUTPUT_ROOT", output_root),
        "TRAIN_CONFIG": get_value("TRAIN_CONFIG", str(Path(workflow_dir) / "train.yaml")),
    })
    values["CHECKPOINT_DIR"] = get_value(
        "CHECKPOINT_DIR",
        os.path.join(values["OUTPUT_ROOT"], "checkpoints", str(values["ANCHOR_MODE"])),
    )
    values["CHECKPOINT_PATH"] = get_value(
        "CHECKPOINT_PATH", os.path.join(values["CHECKPOINT_DIR"], "weight.pt")
    )
    values["IMAGE_ROOT"] = get_value(
        "IMAGE_ROOT", os.path.join(values["OUTPUT_ROOT"], "images")
    )
    values["GCD_OUTPUT_DIR"] = get_value(
        "GCD_OUTPUT_DIR", os.path.join(values["OUTPUT_ROOT"], "gcd")
    )
    values["BENCHMARK_CSV"] = get_value(
        "BENCHMARK_CSV",
        os.path.join(repo_root, "data", f"{values['BENCHMARK_NAME']}.csv"),
    )
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
        print(f"{key}={shlex.quote(stringify(value))}")


if __name__ == "__main__":
    main()
