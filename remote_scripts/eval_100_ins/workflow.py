#!/usr/bin/env python3
"""Prepare, train, sample and score the paired 100-instance comparison."""

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
METHODS = ("original", "legacy", "target_global_pairwise_residual_subspace")
SPLITS = ("erase", "retain")
METHOD_LABELS = {"original": "Original SD v1.4", "legacy": "SPEED",
                 "target_global_pairwise_residual_subspace": "Ours"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def instance_splits(source):
    rows = read_csv(source)
    if len(rows) != 1352 or set(rows[0]) != {"id", "concept"}:
        raise ValueError("instance.csv must contain exactly 1,352 id,concept rows")
    ids = [int(row["id"]) for row in rows]
    names = [row["concept"].strip() for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(range(1, 1353)) or any(not name for name in names):
        raise ValueError("Instance source IDs or names are invalid")
    by_id = dict(zip(ids, names))
    normalized = [{"id": str(index), "concept": by_id[index]} for index in range(1, 1353)]
    erase = normalized[:100]
    retain_eval = normalized[100:200]
    target_names = {row["concept"].casefold() for row in erase}
    if len(target_names) != 100 or len({row["concept"].casefold() for row in retain_eval}) != 100:
        raise ValueError("Erase and evaluation instance names must each be unique")
    if target_names & {row["concept"].casefold() for row in retain_eval}:
        raise ValueError("Erase and evaluation instance names overlap")
    retain_train = []
    seen = set()
    for row in normalized:
        key = row["concept"].casefold()
        if key not in target_names and key not in seen:
            retain_train.append(row)
            seen.add(key)
    if (len(erase), len(retain_eval), len(retain_train)) != (100, 100, 1234):
        raise ValueError("Invalid instance split counts")
    return erase, retain_eval, retain_train


def validate_committed_splits():
    expected = instance_splits(ROOT / "data/instance.csv")
    names = ("instance_100_erase.csv", "instance_100_retain_eval.csv", "instance_100_retain_train.csv")
    for filename, rows in zip(names, expected):
        path = ROOT / "data" / filename
        if read_csv(path) != rows:
            raise ValueError(f"Derived split differs from data/instance.csv: {path}")
    return expected


def load_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    exp = config["experiment"]
    references = config["training"]
    if set(references) != set(METHODS[1:]):
        raise ValueError("training must name the legacy and TGPRS YAML files")
    training = {}
    for method, filename in references.items():
        if not isinstance(filename, str) or not filename.endswith(".yaml"):
            raise ValueError(f"Invalid training YAML path for {method}")
        train_path = path.parent / filename
        train = yaml.safe_load(train_path.read_text(encoding="utf-8"))
        if not isinstance(train, dict) or train.get("anchor_mode") != method:
            raise ValueError(f"Training YAML anchor_mode does not match {method}")
        training[method] = train
    config["_training_configs"] = training
    if exp["instance_templates"] != ["An image of {concept}.", "An illustration of {concept}."]:
        raise ValueError("The instance evaluation requires the two fixed prompts")
    if len(set(exp["sample_seeds"])) != len(exp["sample_seeds"]) or any(type(s) is not int or s < 0 for s in exp["sample_seeds"]):
        raise ValueError("Sample seeds must be distinct nonnegative integers")
    if type(exp["sample_variants_per_seed"]) is not int or exp["sample_variants_per_seed"] < 1:
        raise ValueError("sample_variants_per_seed must be positive")
    shared = dict(sd_ckpt=exp["sd_ckpt"], seed=0, baseline="SPEED", params="V",
                  anchor_concepts="", erase_style=False,
                  threshold=0.1, retain_scale=1.0, heads="concept")
    for method in METHODS[1:]:
        expected = dict(shared, anchor_mode=method, aug_num=10 if method == METHODS[1] else 0)
        if method == METHODS[2]:
            expected.update(residual_rank=30, subspace_anchor_concepts=[""])
        if training[method] != expected:
            raise ValueError(f"Training settings differ from the comparison specification: {method}")
    if exp["seed"] != 0:
        raise ValueError("Comparison seed must be zero")
    if exp["mscoco_prompts"] < 1 or exp["mscoco_prompts"] > 30100:
        raise ValueError("Invalid MS-COCO prompt count")
    if exp["sample_seeds"] != [0] or exp["sample_variants_per_seed"] != 1:
        raise ValueError("Instance prompts must use one paired image with seed 0")
    if exp["inference_steps"] != 20 or exp["guidance_scale"] != 7.5:
        raise ValueError("Instance inference must use 20 DPM-Solver steps and CFG 7.5")
    return config


def selected_splits(config):
    erase, retain, _ = validate_committed_splits()
    exp = config["experiment"]
    return {"erase": erase[:exp.get("smoke_erase_instances", 100)],
            "retain": retain[:exp.get("smoke_retain_instances", 100)]}


def build_manifest(config, splits, coco_rows):
    exp = config["experiment"]
    rows = []
    for group, instances in splits.items():
        group_start = len(rows)
        for concept in instances:
            for template_index, template in enumerate(exp["instance_templates"]):
                for seed in exp["sample_seeds"]:
                    for variant in range(exp["sample_variants_per_seed"]):
                        rows.append(dict(group=group, instance_id=int(concept["id"]), concept=concept["concept"],
                                         prompt=template.format(concept=concept["concept"]), seed=seed,
                                         variant=variant,
                                         filename=f"{int(concept['id']):04d}_{template_index}_{seed}_{variant}.png"))
        expected = len(instances) * len(exp["instance_templates"]) * len(exp["sample_seeds"]) * exp["sample_variants_per_seed"]
        if len(rows) - group_start != expected:
            raise ValueError(f"Incorrect {group} instance image count")
        if len(instances) == 100 and expected != 200:
            raise ValueError(f"Full {group} evaluation must contain 200 images, found {expected}")
    seen = set()
    for row in coco_rows[:exp["mscoco_prompts"]]:
        filename = f"COCO_val2014_{int(row['image_id']):012d}.png"
        if filename in seen:
            raise ValueError("Duplicate MS-COCO image ID")
        seen.add(filename)
        rows.append(dict(group="coco", instance_id="", concept="", prompt=row["text"],
                         seed=exp["seed"], variant=0, filename=filename))
    if len(seen) != exp["mscoco_prompts"]:
        raise ValueError("Not enough MS-COCO prompts")
    return rows


def training_config(config, method, targets):
    resolved = dict(config["_training_configs"][method])
    resolved.update(target_concepts=[row["concept"] for row in targets],
                    retain_path=str(ROOT / "data/instance_100_retain_train.csv"),
                    save_path="", file_name="edit")
    if method == METHODS[2]:
        resolved["residual_rank"] = min(resolved["residual_rank"], len(targets))
    return resolved


def run_id(config):
    inputs = {str(path.relative_to(ROOT)): file_hash(path) for path in
              [ROOT / "data/instance.csv", ROOT / "data/instance_100_erase.csv",
               ROOT / "data/instance_100_retain_eval.csv", ROOT / "data/instance_100_retain_train.csv",
               ROOT / "data/mscoco.csv", ROOT / "train_erase_null.py",
               ROOT / "src/residual_subspace.py", ROOT / "src/edit_checkpoint.py", Path(__file__)]}
    return digest({"config": config, "inputs": inputs})[:16]


def sampling_identity(output, config, method):
    exp = config["experiment"]
    checkpoint = output / "checkpoints" / method / "edit.pt"
    return digest({"manifest": file_hash(output / "manifest.csv"), "steps": exp["inference_steps"],
                   "cfg": exp["guidance_scale"], "sd": exp["sd_ckpt"], "method": method,
                   "checkpoint": file_hash(checkpoint) if method != "original" else "original"})


def valid_png(path):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    from PIL import Image
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def validate_image_set(root, rows, group):
    expected = {r["filename"] for r in rows if r["group"] == group}
    actual = {p.name for p in (root / group).glob("*.png")}
    if expected != actual:
        raise ValueError(f"Incomplete or unexpected images in {root / group}: {len(actual)} vs {len(expected)}")
    if any(not valid_png(root / group / filename) for filename in expected):
        raise ValueError(f"A generated image is corrupt in {root / group}")
    return expected


def context(args):
    config = load_config(args.config)
    splits = selected_splits(config)
    output = Path(args.output_root).expanduser().resolve() / "runs" / run_id(config)
    return config, splits, output


def prepare(args):
    config, splits, output = context(args)
    rows = build_manifest(config, splits, read_csv(ROOT / "data/mscoco.csv"))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_workflow.json", {"config": config, "run_id": output.name,
                                                     "source_instance_sha256": file_hash(ROOT / "data/instance.csv")})
    write_csv(output / "manifest.csv", rows,
              ["group", "instance_id", "concept", "prompt", "seed", "variant", "filename"])
    print(f"Run: {output}; manifest rows: {len(rows)}; images across three models: {len(rows) * 3}")


def train(args):
    config, splits, output = context(args)
    output.mkdir(parents=True, exist_ok=True)
    python_bin = args.python_bin or config["runtime"]["python_bin"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu_id if args.gpu_id is not None else config["runtime"]["gpu_id"]),
               USE_TF="0", TRANSFORMERS_NO_TF="1")
    for method in METHODS[1:]:
        directory = output / "checkpoints" / method
        directory.mkdir(parents=True, exist_ok=True)
        resolved = training_config(config, method, splits["erase"])
        resolved["save_path"] = str(directory)
        identity = digest({"config": resolved, "retain_sha256": file_hash(ROOT / "data/instance_100_retain_train.csv")})
        checkpoint, stamp = directory / "edit.pt", directory / "completed.json"
        if checkpoint.is_file() and stamp.is_file() and json.loads(stamp.read_text()).get("identity") == identity and file_hash(checkpoint) == json.loads(stamp.read_text()).get("sha256"):
            print(f"Using completed {method} checkpoint")
            continue
        config_path = directory / "resolved_train.yaml"
        config_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        print(f"Training {method}; checkpoint: {checkpoint}", flush=True)
        result = subprocess.run([python_bin, "-u", str(ROOT / "train_erase_null.py"), "--config", str(config_path)],
                                cwd=ROOT, env=env)
        if result.returncode or not checkpoint.is_file():
            raise RuntimeError(f"Training {method} failed; see {args.output_root / 'workflow.log'}")
        write_json(stamp, {"identity": identity, "sha256": file_hash(checkpoint)})


def sample(args):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    from src.edit_checkpoint import apply_edit_checkpoint
    from tqdm import tqdm

    config, _, output = context(args)
    exp = config["experiment"]
    rows = read_csv(output / "manifest.csv")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sampling")
    for method in METHODS:
        checkpoint = output / "checkpoints" / method / "edit.pt"
        if method != "original" and not (checkpoint.is_file() and (checkpoint.parent / "completed.json").is_file()):
            raise FileNotFoundError(f"Completed checkpoint missing: {checkpoint}")
        model_identity = sampling_identity(output, config, method)
        image_root = output / "images" / method
        stamp = image_root / "manifest.json"
        if stamp.is_file() and json.loads(stamp.read_text()).get("identity") != model_identity:
            raise ValueError(f"Stale image directory: {image_root}; use a new output root")
        image_root.mkdir(parents=True, exist_ok=True)
        write_json(stamp, {"identity": model_identity})
        pipe = StableDiffusionPipeline.from_pretrained(exp["sd_ckpt"], safety_checker=None, torch_dtype=torch.float16).to("cuda")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if method != "original":
            apply_edit_checkpoint(pipe.unet, checkpoint, device="cpu")
        pipe.set_progress_bar_config(disable=True)
        for row in tqdm(rows, desc=f"Sampling {method}", unit="image", file=sys.stdout,
                        dynamic_ncols=True, mininterval=2):
            path = image_root / row["group"] / row["filename"]
            if valid_png(path):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            generator = torch.Generator(device="cpu").manual_seed(int(row["seed"]))
            for _ in range(int(row["variant"]) + 1):
                latents = torch.randn((1, pipe.unet.config.in_channels, 64, 64), generator=generator,
                                      dtype=torch.float32)
            latents = latents.to(device="cuda", dtype=pipe.unet.dtype)
            image = pipe(row["prompt"], latents=latents, num_inference_steps=exp["inference_steps"],
                         guidance_scale=exp["guidance_scale"]).images[0]
            temporary = path.with_suffix(".tmp.png")
            image.save(temporary)
            temporary.replace(path)
        del pipe
        torch.cuda.empty_cache()


def bootstrap_ci(values, replicates, seed=0):
    if not values:
        raise ValueError("Cannot bootstrap empty values")
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(replicates))
    return draws[int(.025 * (replicates - 1))], draws[int(.975 * (replicates - 1))]


def aggregate_scores(instance_scores, replicates):
    """Macro average instance scores; bootstrap resamples instances."""
    result = []
    for method in METHODS:
        target = [r["clip_score"] for r in instance_scores if r["method"] == method and r["group"] == "erase"]
        retained = [r["clip_score"] for r in instance_scores if r["method"] == method and r["group"] == "retain"]
        if not target or not retained:
            raise ValueError("Every model needs erase and retain instance scores")
        e_ci = bootstrap_ci(target, replicates)
        s_ci = bootstrap_ci(retained, replicates)
        e, s = sum(target) / len(target), sum(retained) / len(retained)
        result.append(dict(method=method, model=METHOD_LABELS[method], erase_clip=e,
                           erase_ci_low=e_ci[0], erase_ci_high=e_ci[1], retain_clip=s,
                           retain_ci_low=s_ci[0], retain_ci_high=s_ci[1]))
    original = result[0]
    for row in result:
        for key in ("erase_clip", "retain_clip"):
            row[f"delta_{key}_vs_original"] = row[key] - original[key]
    return result


def evaluate(args):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    config, _, output = context(args)
    exp = config["experiment"]
    rows = read_csv(output / "manifest.csv")
    for method in METHODS:
        root = output / "images" / method
        stamp = root / "manifest.json"
        if not stamp.is_file() or json.loads(stamp.read_text()).get("identity") != sampling_identity(output, config, method):
            raise ValueError(f"Missing or stale sampling manifest: {root}")
        for group in (*SPLITS, "coco"):
            validate_image_set(root, rows, group)
    image_manifest = digest({m: [(str(p.relative_to(output)), p.stat().st_size, p.stat().st_mtime_ns)
                                  for p in sorted((output / "images" / m).rglob("*.png"))] for m in METHODS})
    metrics_dir = output / "metrics"
    done = metrics_dir / "completed.json"
    identity = digest([image_manifest, config])
    if done.is_file() and json.loads(done.read_text()).get("identity") == identity:
        print(f"Using completed metrics in {metrics_dir}")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(exp["clip_model"]).to(device).eval()
    processor = CLIPProcessor.from_pretrained(exp["clip_model"])
    sample_scores = []
    for method in METHODS:
        for start in range(0, len(rows), exp["clip_batch_size"]):
            batch = rows[start:start+exp["clip_batch_size"]]
            images = []
            for row in batch:
                with Image.open(output / "images" / method / row["group"] / row["filename"]) as image:
                    images.append(image.convert("RGB").copy())
            image_inputs = processor(images=images, return_tensors="pt")
            text_inputs = processor(text=[r["prompt"] for r in batch], return_tensors="pt",
                                    padding=True, truncation=True, max_length=77)
            with torch.inference_mode():
                a = model.get_image_features(**{k: v.to(device) for k, v in image_inputs.items()})
                b = model.get_text_features(**{k: v.to(device) for k, v in text_inputs.items()})
                a = a / a.norm(dim=-1, keepdim=True)
                b = b / b.norm(dim=-1, keepdim=True)
                scores = (a*b).sum(dim=-1).cpu().tolist()
            sample_scores.extend(dict(method=method, model=METHOD_LABELS[method],
                                      group=row["group"], instance_id=row["instance_id"],
                                      concept=row["concept"], filename=row["filename"], score=100*score)
                                 for row, score in zip(batch, scores))
    write_csv(metrics_dir / "image_clip.csv", sample_scores,
              ["method", "model", "group", "instance_id", "concept", "filename", "score"])
    instance_rows = []
    for method in METHODS:
        for group in SPLITS:
            for instance_id in dict.fromkeys(r["instance_id"] for r in rows if r["group"] == group):
                values = [r["score"] for r in sample_scores if r["method"] == method and r["group"] == group and r["instance_id"] == instance_id]
                concept = next(r["concept"] for r in rows if r["instance_id"] == instance_id and r["group"] == group)
                instance_rows.append(dict(method=method, model=METHOD_LABELS[method], group=group,
                                          instance_id=instance_id, concept=concept, n_images=len(values),
                                          clip_score=sum(values)/len(values)))
    original = {(r["group"], r["instance_id"]): r["clip_score"] for r in instance_rows if r["method"] == "original"}
    for row in instance_rows:
        row["delta_vs_original"] = row["clip_score"] - original[row["group"], row["instance_id"]]
    write_csv(metrics_dir / "per_instance.csv", instance_rows,
              ["method", "model", "group", "instance_id", "concept", "n_images", "clip_score", "delta_vs_original"])
    aggregate = aggregate_scores(instance_rows, exp["bootstrap_replicates"])
    for row in aggregate:
        method = row["method"]
        coco = [r["score"] for r in sample_scores if r["method"] == method and r["group"] == "coco"]
        row["n_erase_instances"] = len({r["instance_id"] for r in rows if r["group"] == "erase"})
        row["n_retain_instances"] = len({r["instance_id"] for r in rows if r["group"] == "retain"})
        row["n_coco_images"] = len(coco)
        row["mscoco_clip"] = sum(coco)/len(coco)
    original_coco = aggregate[0]["mscoco_clip"]
    for row in aggregate:
        row["delta_mscoco_clip_vs_original"] = row["mscoco_clip"]-original_coco
    write_csv(metrics_dir / "comparison_clip.csv", aggregate, list(aggregate[0]))
    write_json(done, {"identity": identity, "image_manifest": image_manifest})
    print(f"Metrics: {metrics_dir / 'comparison_clip.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "sample", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-bin")
    parser.add_argument("--gpu-id", type=int)
    args = parser.parse_args()
    {"prepare": prepare, "train": train, "sample": sample, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
