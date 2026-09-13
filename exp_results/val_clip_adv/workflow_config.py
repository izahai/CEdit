#!/usr/bin/env python3
"""Validate and expose the Van Gogh learned-anchor evaluation config."""

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METHOD_ORIGINAL = "original"
METHOD_LEGACY = "legacy"
METHOD_TGPRS = "target_global_pairwise_residual_subspace"
METHOD_LEARNED = "clip_guided_learned_anchor"
EXPECTED_METHODS = [
    METHOD_ORIGINAL,
    METHOD_LEGACY,
    METHOD_TGPRS,
    METHOD_LEARNED,
]
EXPECTED_EDITED_METHODS = [METHOD_LEGACY, METHOD_TGPRS, METHOD_LEARNED]
TRAIN_CONFIG_FILENAMES = {
    METHOD_LEGACY: "train_config_legacy.yaml",
    METHOD_TGPRS: "train_config_target_global_pairwise_residual_subspace.yaml",
    METHOD_LEARNED: "train_config_learned_anchor.yaml",
}
FID_FEATURE_LAYERS = {64, 192, 768, 2048}
PROFILE_SETTINGS = {
    "full": {
        "num_samples_per_prompt": 10,
        "mscoco_num_prompts": 1000,
        "minimum_free_disk_gib": 100,
    },
    "smoke": {
        "num_samples_per_prompt": 1,
        "mscoco_num_prompts": 50,
        "minimum_free_disk_gib": 20,
    },
}


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_list(value, label):
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise ValueError(f"{label} must be a non-empty string list")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} contains duplicates")
    return value


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{label} must be a positive number")
    return value


def load_yaml(path):
    path = Path(path)
    with path.open(encoding="utf-8") as config_file:
        return _mapping(yaml.safe_load(config_file), str(path))


def load_train_configs(workflow_dir):
    workflow_dir = Path(workflow_dir)
    return {
        method: load_yaml(workflow_dir / filename)
        for method, filename in TRAIN_CONFIG_FILENAMES.items()
    }


def validate_train_configs(workflow_dir):
    configs = load_train_configs(workflow_dir)
    legacy = configs[METHOD_LEGACY]
    tgprs = configs[METHOD_TGPRS]
    learned = configs[METHOD_LEARNED]

    common_speed = {
        "anchor_mode": "legacy",
        "params": "V",
        "aug_num": 10,
        "threshold": 0.1,
        "retain_scale": 1.0,
        "lamb": 0.0,
        "disable_filter": False,
    }
    for method, config in [(METHOD_LEGACY, legacy), (METHOD_LEARNED, learned)]:
        for key, expected in common_speed.items():
            if config.get(key) != expected:
                raise ValueError(
                    f"{method} requires {key}={expected!r}, "
                    f"found {config.get(key)!r}"
                )
    if legacy.get("anchor_source") != "text":
        raise ValueError("legacy must use anchor_source=text")
    if learned.get("anchor_source") != "learned":
        raise ValueError("clip_guided_learned_anchor must use anchor_source=learned")
    if learned.get("erase_style", False):
        raise ValueError("learned anchors do not support erase_style")

    tgprs_expected = {
        "anchor_source": "text",
        "anchor_mode": METHOD_TGPRS,
        "erase_style": False,
        "residual_rank": 30,
        "residual_scale": 1.0,
        "params": "V",
        "aug_num": 0,
        "threshold": 1.0,
        "retain_scale": 1.0,
        "lamb": 0.0,
        "disable_filter": False,
    }
    for key, expected in tgprs_expected.items():
        if tgprs.get(key) != expected:
            raise ValueError(
                f"TGPRS requires {key}={expected!r}, found {tgprs.get(key)!r}"
            )
    anchors = tgprs.get("subspace_anchor_concepts")
    _string_list(anchors, "TGPRS subspace_anchor_concepts")
    if len(anchors) != 100:
        raise ValueError("TGPRS must use the current 100 neutral style anchors")
    if any("van gogh" in anchor.casefold() for anchor in anchors):
        raise ValueError("TGPRS anchors must not contain the target artist")
    return configs


def validate_learn_anchor_config(workflow_dir):
    config = load_yaml(Path(workflow_dir) / "learn_anchor_config.yaml")
    expected = {
        "sd_ckpt": "CompVis/stable-diffusion-v1-4",
        "clip_model": "openai/clip-vit-large-patch14",
        "target_concepts": ["Van Gogh"],
        "num_prefix_tokens": 4,
        "num_reference_images": 4,
        "num_validation_images": 1,
        "iterations": 1000,
        "validation_samples": 16,
        "validation_interval": 50,
        "device": "cuda",
        "dtype": "float16",
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(
                f"Anchor search requires {key}={expected_value!r}, "
                f"found {config.get(key)!r}"
            )
    return config


def validate_config(config, workflow_dir):
    runtime = _mapping(config.get("runtime"), "runtime")
    experiment = _mapping(config.get("experiment"), "experiment")
    task = _mapping(config.get("task"), "task")
    paths = _mapping(config.get("paths"), "paths")

    _string(runtime.get("python_bin"), "runtime.python_bin")
    if isinstance(runtime.get("gpu_id"), bool) or not isinstance(
        runtime.get("gpu_id"), int
    ):
        raise ValueError("runtime.gpu_id must be an integer")
    if experiment.get("methods") != EXPECTED_METHODS:
        raise ValueError(f"experiment.methods must equal {EXPECTED_METHODS}")
    if experiment.get("edited_methods") != EXPECTED_EDITED_METHODS:
        raise ValueError(
            f"experiment.edited_methods must equal {EXPECTED_EDITED_METHODS}"
        )
    profile = experiment.get("profile", "full")
    if profile not in PROFILE_SETTINGS:
        raise ValueError(
            f"experiment.profile must be one of {sorted(PROFILE_SETTINGS)}"
        )
    _string(experiment.get("sd_ckpt"), "experiment.sd_ckpt")
    _string(experiment.get("clip_model"), "experiment.clip_model")
    if isinstance(experiment.get("seed"), bool) or not isinstance(
        experiment.get("seed"), int
    ):
        raise ValueError("experiment.seed must be an integer")
    for key in (
        "inference_timesteps",
        "num_samples_per_prompt",
        "batch_size",
        "mscoco_num_prompts",
        "mscoco_batch_size",
        "clip_batch_size",
        "fid_batch_size",
    ):
        _positive_integer(experiment.get(key), f"experiment.{key}")
    _positive_number(experiment.get("guidance_scale"), "experiment.guidance_scale")
    if experiment.get("fid_feature_layer") not in FID_FEATURE_LAYERS:
        raise ValueError("experiment.fid_feature_layer is unsupported")
    if experiment["num_samples_per_prompt"] % experiment["batch_size"]:
        raise ValueError("num_samples_per_prompt must be divisible by batch_size")
    if experiment["mscoco_num_prompts"] % experiment["mscoco_batch_size"]:
        raise ValueError("mscoco_num_prompts must be divisible by mscoco_batch_size")
    fixed_experiment = {
        "sd_ckpt": "CompVis/stable-diffusion-v1-4",
        "seed": 0,
        "inference_timesteps": 20,
        "guidance_scale": 7.5,
        "fid_feature_layer": 2048,
        "clip_model": "openai/clip-vit-large-patch14",
        "num_samples_per_prompt": PROFILE_SETTINGS[profile][
            "num_samples_per_prompt"
        ],
        "mscoco_num_prompts": PROFILE_SETTINGS[profile]["mscoco_num_prompts"],
    }
    for key, expected_value in fixed_experiment.items():
        if experiment.get(key) != expected_value:
            raise ValueError(
                f"experiment.{key} must be {expected_value!r}, "
                f"found {experiment.get(key)!r}"
            )
    from src.template import template_dict

    if len(template_dict["style"]) != 30:
        raise ValueError("The full workflow requires exactly 30 style templates")

    expected_task = {
        "id": "van_gogh",
        "erase_type": "style",
        "target_concept": "Van Gogh",
        "anchor_concept": "art",
    }
    for key, expected_value in expected_task.items():
        if task.get(key) != expected_value:
            raise ValueError(
                f"task.{key} must be {expected_value!r}, found {task.get(key)!r}"
            )
    contents = _string_list(task.get("contents"), "task.contents")
    if contents != ["Van Gogh", "Picasso", "Monet", "Paul Gauguin", "Caravaggio"]:
        raise ValueError("task.contents must use the fixed five-artist evaluation set")
    _string(paths.get("output_root"), "paths.output_root")

    validate_train_configs(workflow_dir)
    validate_learn_anchor_config(workflow_dir)


def load_config(path):
    path = Path(path)
    config = load_yaml(path)
    validate_config(config, path.parent)
    return config


def expected_image_counts(config):
    experiment = config["experiment"]
    contents = config["task"]["contents"]
    from src.template import template_dict

    few_per_method = (
        len(contents)
        * len(template_dict["style"])
        * experiment["num_samples_per_prompt"]
    )
    edited_count = len(experiment["edited_methods"])
    coco = experiment["mscoco_num_prompts"]
    return {
        "original_few": few_per_method,
        "edited_few": few_per_method * edited_count,
        "original_coco": coco,
        "edited_coco": coco * edited_count,
        "total": few_per_method * (edited_count + 1) + coco * (edited_count + 1),
    }


def config_fingerprint(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_environment(config, workflow_dir):
    experiment = config["experiment"]
    task = config["task"]
    workspace_dir = os.environ["WORKSPACE_DIR"]
    output_default = os.path.expandvars(config["paths"]["output_root"])
    from src.template import template_dict

    values = {
        "WORKFLOW_DIR": str(Path(workflow_dir).resolve()),
        "PYTHON_BIN": os.environ.get("PYTHON_BIN", config["runtime"]["python_bin"]),
        "GPU_ID": os.environ.get("GPU_ID", config["runtime"]["gpu_id"]),
        "SD_CKPT": os.environ.get("SD_CKPT", experiment["sd_ckpt"]),
        "CLIP_MODEL": os.environ.get("CLIP_MODEL", experiment["clip_model"]),
        "SEED": os.environ.get("SEED", experiment["seed"]),
        "INFERENCE_TIMESTEPS": os.environ.get(
            "INFERENCE_TIMESTEPS", experiment["inference_timesteps"]
        ),
        "GUIDANCE_SCALE": os.environ.get(
            "GUIDANCE_SCALE", experiment["guidance_scale"]
        ),
        "NUM_SAMPLES_PER_PROMPT": os.environ.get(
            "NUM_SAMPLES_PER_PROMPT", experiment["num_samples_per_prompt"]
        ),
        "BATCH_SIZE": os.environ.get("BATCH_SIZE", experiment["batch_size"]),
        "MSCOCO_NUM_PROMPTS": os.environ.get(
            "MSCOCO_NUM_PROMPTS", experiment["mscoco_num_prompts"]
        ),
        "MSCOCO_BATCH_SIZE": os.environ.get(
            "MSCOCO_BATCH_SIZE", experiment["mscoco_batch_size"]
        ),
        "CLIP_BATCH_SIZE": os.environ.get(
            "CLIP_BATCH_SIZE", experiment["clip_batch_size"]
        ),
        "FID_BATCH_SIZE": os.environ.get(
            "FID_BATCH_SIZE", experiment["fid_batch_size"]
        ),
        "FID_FEATURE_LAYER": os.environ.get(
            "FID_FEATURE_LAYER", experiment["fid_feature_layer"]
        ),
        "WORKFLOW_PROFILE": experiment.get("profile", "full"),
        "MINIMUM_FREE_DISK_GIB": PROFILE_SETTINGS[
            experiment.get("profile", "full")
        ]["minimum_free_disk_gib"],
        "METHODS_RAW": " ".join(experiment["methods"]),
        "EDITED_METHODS_RAW": " ".join(experiment["edited_methods"]),
        "TASK_ID": task["id"],
        "ERASE_TYPE": task["erase_type"],
        "TARGET_CONCEPT": task["target_concept"],
        "ANCHOR_CONCEPT": task["anchor_concept"],
        "CONTENTS": ", ".join(task["contents"]),
        "STYLE_TEMPLATE_COUNT": len(template_dict["style"]),
        "OUTPUT_ROOT": os.environ.get("OUTPUT_ROOT", output_default),
        "WORKSPACE_DIR": workspace_dir,
    }
    values["CHECKPOINT_ROOT"] = os.environ.get(
        "CHECKPOINT_ROOT", os.path.join(values["OUTPUT_ROOT"], "checkpoints")
    )
    values["IMAGE_ROOT"] = os.environ.get(
        "IMAGE_ROOT", os.path.join(values["OUTPUT_ROOT"], "images")
    )
    values["MSCOCO_IMAGE_ROOT"] = os.environ.get(
        "MSCOCO_IMAGE_ROOT", os.path.join(values["OUTPUT_ROOT"], "mscoco")
    )
    values["METRICS_DIR"] = os.environ.get(
        "METRICS_DIR", os.path.join(values["OUTPUT_ROOT"], "metrics")
    )
    values["LOG_ROOT"] = os.environ.get(
        "LOG_ROOT", os.path.join(values["OUTPUT_ROOT"], "logs")
    )
    values["FID_CACHE_ROOT"] = os.environ.get(
        "FID_CACHE_ROOT", os.path.join(values["METRICS_DIR"], "fid_cache")
    )
    values["LEARNED_ANCHOR_ROOT"] = os.environ.get(
        "LEARNED_ANCHOR_ROOT",
        os.path.join(values["OUTPUT_ROOT"], "learned_anchors", task["id"]),
    )
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("command", choices=("export", "counts", "validate"))
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "export":
            values = build_environment(config, args.config.parent)
            for key, value in values.items():
                print(f"{key}={shlex.quote(str(value))}")
        elif args.command == "counts":
            print(json.dumps(expected_image_counts(config), sort_keys=True))
        else:
            print(f"Valid workflow: {args.config}")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
