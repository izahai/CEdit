#!/usr/bin/env python3
"""Validate eval_few workflow YAML and expose it to shell stages."""

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path

import yaml


METHOD_ORIGINAL = "original"
METHOD_LEGACY = "legacy"
METHOD_TGPRS = "target_global_pairwise_residual_subspace"
EXPECTED_METHODS = [METHOD_ORIGINAL, METHOD_LEGACY, METHOD_TGPRS]
EXPECTED_EDITED_METHODS = [METHOD_LEGACY, METHOD_TGPRS]
FID_FEATURE_LAYERS = {"64", "192", "768", "2048"}
DEFAULT_SUBSPACE_ANCHOR_CONCEPTS = ["", "person"]
INTERNAL_TGPRS_CONFIG = "_resolved_tgprs_train_config"
FIELD_SEPARATOR = "\x1f"


def required(config, section, key):
    try:
        return config[section][key]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Missing workflow config value: {section}.{key}"
        ) from error


def get_value(name, default):
    return os.environ.get(name, default)


def stringify(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{label} must be a positive number")
    return value


def _string_list(value, label, allow_empty_items=False):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must contain only strings")
    if not allow_empty_items and any(not item.strip() for item in value):
        raise ValueError(f"{label} cannot contain blank strings")
    return value


def _subspace_anchor_list(value, label):
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must contain only strings")
    if any("\n" in item or "\r" in item for item in value):
        raise ValueError(f"{label} cannot contain newline characters")
    return list(value)


def _load_tgprs_train_config(workflow_dir):
    path = (
        Path(workflow_dir)
        / "train_config_target_global_pairwise_residual_subspace.yaml"
    )
    if path.is_file():
        with path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        if not isinstance(config, dict):
            raise ValueError(f"TGPRS training config must be a mapping: {path}")
    else:
        config = {}
    residual_rank = config.get("residual_rank", 30)
    _positive_integer(residual_rank, "TGPRS residual_rank")
    anchors = _subspace_anchor_list(
        config.get(
            "subspace_anchor_concepts",
            DEFAULT_SUBSPACE_ANCHOR_CONCEPTS,
        ),
        "TGPRS subspace_anchor_concepts",
    )
    return {
        "residual_rank": residual_rank,
        "subspace_anchor_concepts": anchors,
    }


def _default_templates():
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.template import template_dict

    return template_dict


def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Workflow YAML must contain a mapping")
    config[INTERNAL_TGPRS_CONFIG] = _load_tgprs_train_config(path.parent)
    validate_config(config)
    return config


def domain_specs(config):
    templates_by_domain = _default_templates()
    domains = config.get("domains")
    if not isinstance(domains, dict) or not domains:
        raise ValueError("domains must be a non-empty mapping")

    specs = []
    for domain, raw_spec in domains.items():
        if domain not in templates_by_domain:
            raise ValueError(f"Unknown prompt-template domain: {domain}")
        if not isinstance(raw_spec, dict):
            raise ValueError(f"domains.{domain} must be a mapping")
        contents = _string_list(
            raw_spec.get("contents"), f"domains.{domain}.contents"
        )
        if len(set(contents)) != len(contents):
            raise ValueError(f"domains.{domain}.contents contains duplicates")
        prompts = raw_spec.get("prompt_templates", templates_by_domain[domain])
        prompts = _string_list(
            prompts, f"domains.{domain}.prompt_templates"
        )
        for template in prompts:
            if template.count("{}") != 1:
                raise ValueError(
                    f"Prompt template must contain one {{}} placeholder: {template}"
                )
        specs.append({
            "name": domain,
            "contents": list(contents),
            "prompt_templates": list(prompts),
        })
    return specs


def task_specs(config):
    domains = {spec["name"]: spec for spec in domain_specs(config)}
    raw_tasks = config.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("tasks must be a non-empty list")

    seen_ids = set()
    tasks = []
    for index, raw_task in enumerate(raw_tasks):
        label = f"tasks[{index}]"
        if not isinstance(raw_task, dict):
            raise ValueError(f"{label} must be a mapping")
        task_id = raw_task.get("id")
        if not isinstance(task_id, str) or not re.fullmatch(
            r"[a-z0-9]+(?:_[a-z0-9]+)*", task_id or ""
        ):
            raise ValueError(f"{label}.id must be a lowercase underscore name")
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task id: {task_id}")
        seen_ids.add(task_id)

        erase_type = raw_task.get("erase_type")
        if erase_type not in domains:
            raise ValueError(f"{label}.erase_type is not a configured domain")
        targets = _string_list(
            raw_task.get("target_concepts"), f"{label}.target_concepts"
        )
        if len(set(targets)) != len(targets):
            raise ValueError(f"{label}.target_concepts contains duplicates")
        anchor = raw_task.get("anchor_concept")
        if not isinstance(anchor, str):
            raise ValueError(f"{label}.anchor_concept must be a string")
        contents = domains[erase_type]["contents"]
        missing_targets = [target for target in targets if target not in contents]
        if missing_targets:
            raise ValueError(
                f"{label} targets are absent from {erase_type} contents: "
                + ", ".join(missing_targets)
            )

        target_count = len(targets)
        tgprs_config = config.get(INTERNAL_TGPRS_CONFIG, {
            "residual_rank": 30,
            "subspace_anchor_concepts": DEFAULT_SUBSPACE_ANCHOR_CONCEPTS,
        })
        has_subspace_anchor_override = "subspace_anchor_concepts" in raw_task
        if has_subspace_anchor_override:
            subspace_anchor_concepts = _subspace_anchor_list(
                raw_task["subspace_anchor_concepts"],
                f"{label}.subspace_anchor_concepts",
            )
        else:
            subspace_anchor_concepts = list(
                tgprs_config["subspace_anchor_concepts"]
            )
        subspace_anchor_count = len(subspace_anchor_concepts)
        residuals_per_target = target_count - 1 + subspace_anchor_count
        if residuals_per_target == 0:
            raise ValueError(
                f"{label} produces no TGPRS residual vectors; configure at "
                "least two targets or one subspace anchor"
            )
        configured_residual_rank = tgprs_config["residual_rank"]
        max_residual_rank = target_count + subspace_anchor_count - 1
        tasks.append({
            "id": task_id,
            "erase_type": erase_type,
            "target_concepts": list(targets),
            "anchor_concept": anchor,
            "target_count": target_count,
            "configured_residual_rank": configured_residual_rank,
            "applied_residual_rank": min(
                configured_residual_rank,
                max_residual_rank,
            ),
            "target_global_residual_count": (
                target_count * residuals_per_target
            ),
            "subspace_anchor_concepts": subspace_anchor_concepts,
            "subspace_anchor_count": subspace_anchor_count,
            "has_subspace_anchor_override": has_subspace_anchor_override,
            "contents": list(contents),
            "prompt_templates": list(domains[erase_type]["prompt_templates"]),
        })
    return tasks


def validate_config(config):
    methods = _string_list(
        required(config, "experiment", "methods"), "experiment.methods"
    )
    edited_methods = _string_list(
        required(config, "experiment", "edited_methods"),
        "experiment.edited_methods",
    )
    if methods != EXPECTED_METHODS:
        raise ValueError(f"experiment.methods must equal {EXPECTED_METHODS}")
    if edited_methods != EXPECTED_EDITED_METHODS:
        raise ValueError(
            f"experiment.edited_methods must equal {EXPECTED_EDITED_METHODS}"
        )
    _positive_integer(
        required(config, "experiment", "inference_timesteps"),
        "experiment.inference_timesteps",
    )
    _positive_number(
        required(config, "experiment", "guidance_scale"),
        "experiment.guidance_scale",
    )
    for key in (
        "num_samples_per_prompt",
        "batch_size",
        "mscoco_num_prompts",
        "mscoco_batch_size",
        "clip_batch_size",
        "fid_batch_size",
    ):
        _positive_integer(
            required(config, "experiment", key), f"experiment.{key}"
        )
    fid_feature_layer = str(
        required(config, "experiment", "fid_feature_layer")
    )
    if fid_feature_layer not in FID_FEATURE_LAYERS:
        raise ValueError(
            "experiment.fid_feature_layer must be one of "
            + ", ".join(sorted(FID_FEATURE_LAYERS, key=int))
        )
    if (
        required(config, "experiment", "num_samples_per_prompt")
        % required(config, "experiment", "batch_size")
    ):
        raise ValueError("num_samples_per_prompt must be divisible by batch_size")
    if (
        required(config, "experiment", "mscoco_num_prompts")
        % required(config, "experiment", "mscoco_batch_size")
    ):
        raise ValueError("mscoco_num_prompts must be divisible by mscoco_batch_size")
    domain_specs(config)
    task_specs(config)


def build_environment(config, workflow_dir):
    workspace_dir = os.environ["WORKSPACE_DIR"]
    experiment = config["experiment"]
    values = {
        "PYTHON_BIN": get_value(
            "PYTHON_BIN", required(config, "runtime", "python_bin")
        ),
        "GPU_ID": get_value("GPU_ID", required(config, "runtime", "gpu_id")),
        "SD_CKPT": get_value("SD_CKPT", experiment["sd_ckpt"]),
        "METHODS_RAW": get_value("METHODS_RAW", " ".join(experiment["methods"])),
        "EDITED_METHODS_RAW": get_value(
            "EDITED_METHODS_RAW", " ".join(experiment["edited_methods"])
        ),
        "SEED": get_value("SEED", experiment["seed"]),
        "INFERENCE_TIMESTEPS": get_value(
            "INFERENCE_TIMESTEPS", experiment["inference_timesteps"]
        ),
        "GUIDANCE_SCALE": get_value(
            "GUIDANCE_SCALE", experiment["guidance_scale"]
        ),
        "NUM_SAMPLES_PER_PROMPT": get_value(
            "NUM_SAMPLES_PER_PROMPT", experiment["num_samples_per_prompt"]
        ),
        "BATCH_SIZE": get_value("BATCH_SIZE", experiment["batch_size"]),
        "MSCOCO_NUM_PROMPTS": get_value(
            "MSCOCO_NUM_PROMPTS", experiment["mscoco_num_prompts"]
        ),
        "MSCOCO_BATCH_SIZE": get_value(
            "MSCOCO_BATCH_SIZE", experiment["mscoco_batch_size"]
        ),
        "CLIP_BATCH_SIZE": get_value(
            "CLIP_BATCH_SIZE", experiment["clip_batch_size"]
        ),
        "FID_BATCH_SIZE": get_value(
            "FID_BATCH_SIZE", experiment["fid_batch_size"]
        ),
        "FID_FEATURE_LAYER": get_value(
            "FID_FEATURE_LAYER", experiment["fid_feature_layer"]
        ),
        "CLIP_MODEL": get_value("CLIP_MODEL", experiment["clip_model"]),
    }
    output_root = os.path.expandvars(required(config, "paths", "output_root"))
    values.update({
        "WORKFLOW_DIR": workflow_dir,
        "OUTPUT_ROOT": get_value("OUTPUT_ROOT", output_root),
    })
    values["CHECKPOINT_ROOT"] = get_value(
        "CHECKPOINT_ROOT", os.path.join(values["OUTPUT_ROOT"], "checkpoints")
    )
    values["IMAGE_ROOT"] = get_value(
        "IMAGE_ROOT", os.path.join(values["OUTPUT_ROOT"], "images")
    )
    values["MSCOCO_IMAGE_ROOT"] = get_value(
        "MSCOCO_IMAGE_ROOT", os.path.join(values["OUTPUT_ROOT"], "mscoco")
    )
    values["METRICS_DIR"] = get_value(
        "METRICS_DIR", os.path.join(values["OUTPUT_ROOT"], "metrics")
    )
    values["LOG_ROOT"] = get_value(
        "LOG_ROOT", os.path.join(values["OUTPUT_ROOT"], "logs")
    )
    values["FID_CACHE_ROOT"] = get_value(
        "FID_CACHE_ROOT", os.path.join(values["METRICS_DIR"], "fid_cache")
    )
    return values


def config_fingerprint(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def expected_image_counts(config):
    experiment = config["experiment"]
    samples = experiment["num_samples_per_prompt"]
    coco_count = experiment["mscoco_num_prompts"]
    domains = {spec["name"]: spec for spec in domain_specs(config)}
    tasks = task_specs(config)
    edited_method_count = len(experiment["edited_methods"])
    original_few = sum(
        len(spec["contents"]) * len(spec["prompt_templates"]) * samples
        for spec in domains.values()
    )
    edited_few = sum(
        len(task["contents"]) * len(task["prompt_templates"]) * samples
        for task in tasks
    ) * edited_method_count
    original_coco = coco_count
    edited_coco = len(tasks) * edited_method_count * coco_count
    return {
        "original_few": original_few,
        "edited_few": edited_few,
        "original_coco": original_coco,
        "edited_coco": edited_coco,
        "total": original_few + edited_few + original_coco + edited_coco,
    }


def _print_rows(rows):
    for row in rows:
        print(FIELD_SEPARATOR.join(stringify(value) for value in row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-id")
    parser.add_argument(
        "command",
        choices=(
            "export",
            "domains",
            "tasks",
            "task-anchors",
            "counts",
            "validate",
        ),
    )
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "export":
            values = build_environment(config, str(args.config.parent.resolve()))
            for key, value in values.items():
                print(f"{key}={shlex.quote(stringify(value))}")
        elif args.command == "domains":
            samples = config["experiment"]["num_samples_per_prompt"]
            _print_rows([
                (
                    spec["name"],
                    ", ".join(spec["contents"]),
                    ";".join(spec["prompt_templates"]),
                    len(spec["prompt_templates"]),
                    len(spec["prompt_templates"]) * samples,
                )
                for spec in domain_specs(config)
            ])
        elif args.command == "tasks":
            samples = config["experiment"]["num_samples_per_prompt"]
            _print_rows([
                (
                    task["id"],
                    task["erase_type"],
                    ", ".join(task["target_concepts"]),
                    task["anchor_concept"],
                    task["target_count"],
                    task["applied_residual_rank"],
                    ", ".join(task["contents"]),
                    ";".join(task["prompt_templates"]),
                    len(task["prompt_templates"]),
                    len(task["prompt_templates"]) * samples,
                )
                for task in task_specs(config)
            ])
        elif args.command == "task-anchors":
            if not args.task_id:
                raise ValueError("--task-id is required for task-anchors")
            matches = [
                task for task in task_specs(config) if task["id"] == args.task_id
            ]
            if not matches:
                raise ValueError(f"Unknown task id: {args.task_id}")
            task = matches[0]
            print("1" if task["has_subspace_anchor_override"] else "0")
            if task["has_subspace_anchor_override"]:
                print(len(task["subspace_anchor_concepts"]))
                for anchor in task["subspace_anchor_concepts"]:
                    print(anchor)
        elif args.command == "counts":
            print(json.dumps(expected_image_counts(config), sort_keys=True))
        else:
            print(f"Valid workflow: {args.config}")
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
