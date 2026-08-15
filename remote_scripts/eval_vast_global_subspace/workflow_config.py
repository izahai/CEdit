#!/usr/bin/env python3
"""Load workflow.yaml and print shell-safe environment assignments."""

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


def string_list(config, section, key):
    values = required(config, section, key)
    if not isinstance(values, list) or not values or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ValueError(f"{section}.{key} must be a non-empty list of strings")
    if any(any(character.isspace() for character in value) for value in values):
        raise ValueError(f"{section}.{key} values cannot contain whitespace")
    return " ".join(values)


def build_environment(config, workflow_dir):
    workspace_dir = os.environ["WORKSPACE_DIR"]
    values = {
        "PYTHON_BIN": get_value(
            "PYTHON_BIN", required(config, "runtime", "python_bin")
        ),
        "GPU_ID": get_value("GPU_ID", required(config, "runtime", "gpu_id")),
        "CEDIT_BRANCH": get_value(
            "CEDIT_BRANCH", required(config, "repositories", "cedit_branch")
        ),
        "CEDIT_REPOSITORY": get_value(
            "CEDIT_REPOSITORY",
            required(config, "repositories", "cedit_repository"),
        ),
        "CE_EVAL_REPOSITORY": get_value(
            "CE_EVAL_REPOSITORY",
            required(config, "repositories", "ce_eval_repository"),
        ),
        "CE_EVAL_BRANCH": get_value(
            "CE_EVAL_BRANCH", required(config, "repositories", "ce_eval_branch")
        ),
        "SD_CKPT": get_value(
            "SD_CKPT", required(config, "experiment", "sd_ckpt")
        ),
        "ANCHOR_MODE": get_value(
            "ANCHOR_MODE", required(config, "experiment", "anchor_mode")
        ),
        "BENCHMARK_NAMES_RAW": get_value(
            "BENCHMARK_NAMES_RAW",
            string_list(config, "experiment", "benchmark_names"),
        ),
        "BATCH_SIZE": get_value(
            "BATCH_SIZE", required(config, "experiment", "batch_size")
        ),
        "EXPECTED_IMAGES_PER_SPLIT": get_value(
            "EXPECTED_IMAGES_PER_SPLIT",
            required(config, "experiment", "expected_images_per_split"),
        ),
        "GCD_USE_CUDA": get_value(
            "GCD_USE_CUDA", required(config, "experiment", "gcd_use_cuda")
        ),
    }
    ce_eval_root = os.path.expandvars(required(config, "paths", "ce_eval_root"))
    output_root = os.path.expandvars(required(config, "paths", "output_root"))
    values.update(
        {
            "WORKFLOW_DIR": workflow_dir,
            "CE_EVAL_ROOT": get_value("CE_EVAL_ROOT", ce_eval_root),
            "OUTPUT_ROOT": get_value("OUTPUT_ROOT", output_root),
        }
    )
    values["IMAGE_ROOT"] = get_value(
        "IMAGE_ROOT", os.path.join(values["OUTPUT_ROOT"], "images")
    )
    values["GCD_OUTPUT_DIR"] = get_value(
        "GCD_OUTPUT_DIR", os.path.join(values["OUTPUT_ROOT"], "gcd")
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
