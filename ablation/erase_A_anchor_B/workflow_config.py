#!/usr/bin/env python3
"""Validate the A-to-B probe configuration and export shell variables."""

import argparse
import os
import shlex
from pathlib import Path

import yaml


def expand(value, environment):
    for key, replacement in environment.items():
        value = value.replace("${" + key + "}", replacement)
    return os.path.expandvars(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("command", choices=("export",))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    runtime = config["runtime"]
    repositories = config["repositories"]
    experiment = config["experiment"]
    paths = config["paths"]
    repo_root = str(Path(__file__).resolve().parents[2])
    workspace_dir = str(Path(repo_root).parent)
    environment = {"REPO_ROOT": repo_root, "WORKSPACE_DIR": workspace_dir}

    target_a = str(experiment["target_a"]).strip()
    anchor_b = str(experiment["anchor_b"]).strip()
    if not target_a or not anchor_b or target_a == anchor_b:
        raise SystemExit("target_a and anchor_b must be non-empty and different")
    templates = experiment["prompt_templates"]
    if not templates or any(template.count("{}") != 1 for template in templates):
        raise SystemExit("Every prompt template must contain exactly one {} slot")
    samples = int(experiment["samples_per_template"])
    batch_size = int(experiment["sample_batch_size"])
    if samples <= 0 or batch_size <= 0 or samples % batch_size != 0:
        raise SystemExit("samples_per_template must be positive and divisible by batch size")

    values = {
        "PYTHON_BIN": str(runtime["python_bin"]),
        "GPU_ID": str(runtime["gpu_id"]),
        "CE_EVAL_REPOSITORY": str(repositories["ce_eval_repository"]),
        "CE_EVAL_BRANCH": str(repositories["ce_eval_branch"]),
        "TARGET_A": target_a,
        "ANCHOR_B": anchor_b,
        "SD_CKPT": str(experiment["sd_ckpt"]),
        "SEED": str(experiment["seed"]),
        "DIFFUSION_STEPS": str(experiment["diffusion_steps"]),
        "GUIDANCE_SCALE": str(experiment["guidance_scale"]),
        "PROMPT_TEMPLATES": ";".join(str(item) for item in templates),
        "TEMPLATE_COUNT": str(len(templates)),
        "SAMPLES_PER_TEMPLATE": str(samples),
        "SAMPLE_BATCH_SIZE": str(batch_size),
        "EXPECTED_IMAGES_PER_IDENTITY": str(len(templates) * samples),
        "GCD_USE_CUDA": str(experiment["gcd_use_cuda"]).lower(),
        "GCD_NUM_WORKERS": str(experiment["gcd_num_workers"]),
        "GCD_BATCH_SIZE": str(experiment["gcd_batch_size"]),
        "GCD_PREFETCH_FACTOR": str(experiment["gcd_prefetch_factor"]),
        "BENCHMARK_CSV": expand(str(paths["benchmark_csv"]), environment),
        "CE_EVAL_ROOT": expand(str(paths["ce_eval_root"]), environment),
        "OUTPUT_ROOT": expand(str(paths["output_root"]), environment),
    }
    values.update({
        "CHECKPOINT_DIR": values["OUTPUT_ROOT"] + "/checkpoint",
        "CHECKPOINT_PATH": values["OUTPUT_ROOT"] + "/checkpoint/weight.pt",
        "RETAIN_CSV": values["OUTPUT_ROOT"] + "/config/retain_concepts.csv",
        "IMAGE_ROOT": values["OUTPUT_ROOT"] + "/images",
        "GCD_ROOT": values["OUTPUT_ROOT"] + "/gcd",
        "SUMMARY_ROOT": values["OUTPUT_ROOT"] + "/summary",
        "FIGURE_ROOT": values["OUTPUT_ROOT"] + "/figures",
    })
    for key, value in values.items():
        print(f"export {key}={shlex.quote(value)}")


if __name__ == "__main__":
    main()

