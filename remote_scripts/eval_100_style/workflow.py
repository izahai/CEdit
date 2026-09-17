#!/usr/bin/env python3
"""Prepare, train, sample and score the paired 100-style comparison."""

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


def style_splits(source):
    rows = read_csv(source)
    if len(rows) != 1734 or set(rows[0]) != {"id", "concept"}:
        raise ValueError("style.csv must contain exactly 1,734 id,concept rows")
    ids = [int(row["id"]) for row in rows]
    names = [row["concept"].strip() for row in rows]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("Style IDs and nonempty names must each be unique")
    by_id = dict(zip(ids, rows))
    if set(ids) != set(range(1, 1735)) or by_id[437]["concept"] != "Caravaggio" or by_id[1266]["concept"] != "Monet":
        raise ValueError("Unexpected source style IDs or artist mapping")
    target_ids = (set(range(401, 501)) - {437}) | {1266}
    erase = [by_id[index] for index in sorted(target_ids)]
    retain_eval = [by_id[index] for index in range(1, 101)]
    retain_train = [row for row in rows if int(row["id"]) not in target_ids]
    if (len(erase), len(retain_eval), len(retain_train)) != (100, 100, 1634):
        raise ValueError("Invalid style split counts")
    if {row["concept"] for row in erase} & {row["concept"] for row in retain_eval}:
        raise ValueError("Erase and evaluation artist names overlap")
    return erase, retain_eval, retain_train


def validate_committed_splits():
    expected = style_splits(ROOT / "data/style.csv")
    names = ("style_100_erase.csv", "style_100_retain_eval.csv", "style_100_retain_train.csv")
    for filename, rows in zip(names, expected):
        path = ROOT / "data" / filename
        if read_csv(path) != rows:
            raise ValueError(f"Derived split differs from data/style.csv: {path}")
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
    if len(exp["style_templates"]) != 5 or any(t.count("{artist}") != 1 for t in exp["style_templates"]):
        raise ValueError("Exactly five artist templates are required")
    prompt_count = exp["style_prompts_per_artist"]
    if type(prompt_count) is not int or not 1 <= prompt_count <= len(exp["style_templates"]):
        raise ValueError("style_prompts_per_artist must be between one and five")
    if len(set(exp["sample_seeds"])) != len(exp["sample_seeds"]) or any(type(s) is not int or s < 0 for s in exp["sample_seeds"]):
        raise ValueError("Sample seeds must be distinct nonnegative integers")
    if type(exp["sample_variants_per_seed"]) is not int or exp["sample_variants_per_seed"] < 1:
        raise ValueError("sample_variants_per_seed must be positive")
    shared = dict(sd_ckpt=exp["sd_ckpt"], seed=0, baseline="SPEED", params="V",
                  anchor_concepts="art", erase_style=False,
                  threshold=0.1, retain_scale=1.0, heads="concept")
    for method in METHODS[1:]:
        expected = dict(shared, anchor_mode=method)
        if method == METHODS[2]:
            expected.update(residual_rank=30, subspace_anchor_concepts=["art"])
        aug_num = training[method].get("aug_num")
        if type(aug_num) is not int or aug_num < 0 or {k: v for k, v in training[method].items() if k != "aug_num"} != expected:
            raise ValueError(f"Training settings differ from the comparison specification: {method}")
    if exp["seed"] != 0:
        raise ValueError("Comparison seed must be zero")
    if exp["mscoco_prompts"] < 1 or exp["mscoco_prompts"] > 30100:
        raise ValueError("Invalid MS-COCO prompt count")
    if exp["fid_feature_layer"] not in (64, 192, 768, 2048):
        raise ValueError("Invalid FID feature layer")
    return config


def selected_splits(config):
    erase, retain, _ = validate_committed_splits()
    exp = config["experiment"]
    return {"erase": erase[:exp.get("smoke_erase_artists", 100)],
            "retain": retain[:exp.get("smoke_retain_artists", 100)]}


def build_manifest(config, splits, coco_rows):
    exp = config["experiment"]
    rows = []
    for group, artists in splits.items():
        group_start = len(rows)
        for artist_index, artist in enumerate(artists):
            prompt_count = exp["style_prompts_per_artist"]
            template_indices = [0] + [
                1 + (artist_index * (prompt_count - 1) + offset) % (len(exp["style_templates"]) - 1)
                for offset in range(prompt_count - 1)
            ]
            for template_index in template_indices:
                template = exp["style_templates"][template_index]
                for seed in exp["sample_seeds"]:
                    for variant in range(exp["sample_variants_per_seed"]):
                        rows.append(dict(group=group, artist_id=int(artist["id"]), artist=artist["concept"],
                                         prompt=template.format(artist=artist["concept"]), seed=seed,
                                         variant=variant,
                                         filename=f"{int(artist['id']):04d}_{template_index}_{seed}_{variant}.png"))
        expected = len(artists) * exp["style_prompts_per_artist"] * len(exp["sample_seeds"]) * exp["sample_variants_per_seed"]
        if len(rows) - group_start != expected:
            raise ValueError(f"Incorrect {group} style image count")
        if len(artists) == 100 and expected != 200:
            raise ValueError(f"Full {group} evaluation must contain 200 images, found {expected}")
    seen = set()
    for row in coco_rows[:exp["mscoco_prompts"]]:
        filename = f"COCO_val2014_{int(row['image_id']):012d}.png"
        if filename in seen:
            raise ValueError("Duplicate MS-COCO image ID")
        seen.add(filename)
        rows.append(dict(group="coco", artist_id="", artist="", prompt=row["text"],
                         seed=exp["seed"], variant=0, filename=filename))
    if len(seen) != exp["mscoco_prompts"]:
        raise ValueError("Not enough MS-COCO prompts")
    return rows


def training_config(config, method, targets):
    resolved = dict(config["_training_configs"][method])
    resolved.update(target_concepts=[row["concept"] for row in targets],
                    retain_path=str(ROOT / "data/style_100_retain_train.csv"),
                    save_path="", file_name="edit")
    if method == METHODS[2]:
        resolved["residual_rank"] = min(resolved["residual_rank"], len(targets))
    return resolved


def run_id(config):
    inputs = {str(path.relative_to(ROOT)): file_hash(path) for path in
              [ROOT / "data/style.csv", ROOT / "data/mscoco.csv", ROOT / "train_erase_null.py",
               ROOT / "src/residual_subspace.py", ROOT / "src/template.py", Path(__file__)]}
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
                                                     "source_style_sha256": file_hash(ROOT / "data/style.csv")})
    write_csv(output / "manifest.csv", rows,
              ["group", "artist_id", "artist", "prompt", "seed", "variant", "filename"])
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
        identity = digest({"config": resolved, "retain_sha256": file_hash(ROOT / "data/style_100_retain_train.csv")})
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


def aggregate_scores(style_scores, replicates):
    """Macro average artist scores; bootstrap resamples artists, not images."""
    result = []
    for method in METHODS:
        target = [r["score"] for r in style_scores if r["model"] == method and r["group"] == "erase"]
        retained = [r["score"] for r in style_scores if r["model"] == method and r["group"] == "retain"]
        if not target or not retained:
            raise ValueError("Every model needs erase and retain artist scores")
        rng = random.Random(0)
        h_draws = sorted(sum(rng.choices(retained, k=len(retained))) / len(retained) -
                         sum(rng.choices(target, k=len(target))) / len(target) for _ in range(replicates))
        e_ci = bootstrap_ci(target, replicates)
        s_ci = bootstrap_ci(retained, replicates)
        e, s = sum(target) / len(target), sum(retained) / len(retained)
        result.append(dict(model=method, clip_e=e, clip_e_ci_low=e_ci[0], clip_e_ci_high=e_ci[1],
                           clip_s=s, clip_s_ci_low=s_ci[0], clip_s_ci_high=s_ci[1],
                           h_a=s-e, h_a_ci_low=h_draws[int(.025*(replicates-1))],
                           h_a_ci_high=h_draws[int(.975*(replicates-1))]))
    original = result[0]
    for row in result:
        for key in ("clip_e", "clip_s", "h_a"):
            row[f"delta_{key}_vs_original"] = row[key] - original[key]
    return result


def evaluate(args):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    import torch_fidelity

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
            sample_scores.extend(dict(model=method, group=row["group"], artist_id=row["artist_id"],
                                      artist=row["artist"], filename=row["filename"], score=100*score)
                                 for row, score in zip(batch, scores))
    write_csv(metrics_dir / "image_clip.csv", sample_scores,
              ["model", "group", "artist_id", "artist", "filename", "score"])
    artist_rows = []
    for method in METHODS:
        for group in SPLITS:
            for artist_id in dict.fromkeys(r["artist_id"] for r in rows if r["group"] == group):
                values = [r["score"] for r in sample_scores if r["model"] == method and r["group"] == group and r["artist_id"] == artist_id]
                ci = bootstrap_ci(values, exp["bootstrap_replicates"])
                artist = next(r["artist"] for r in rows if r["artist_id"] == artist_id and r["group"] == group)
                artist_rows.append(dict(model=method, group=group, artist_id=artist_id, artist=artist,
                                        n_images=len(values), score=sum(values)/len(values), ci_low=ci[0], ci_high=ci[1]))
    original = {(r["group"], r["artist_id"]): r["score"] for r in artist_rows if r["model"] == "original"}
    for row in artist_rows:
        row["delta_vs_original"] = row["score"] - original[row["group"], row["artist_id"]]
    write_csv(metrics_dir / "per_artist.csv", artist_rows,
              ["model", "group", "artist_id", "artist", "n_images", "score", "ci_low", "ci_high", "delta_vs_original"])
    aggregate = aggregate_scores(artist_rows, exp["bootstrap_replicates"])
    for row in aggregate:
        method = row["model"]
        coco = [r["score"] for r in sample_scores if r["model"] == method and r["group"] == "coco"]
        row["n_erase_artists"] = len({r["artist_id"] for r in rows if r["group"] == "erase"})
        row["n_retain_artists"] = len({r["artist_id"] for r in rows if r["group"] == "retain"})
        row["n_coco_images"] = len(coco)
        row["mscoco_clip"] = sum(coco)/len(coco)
        row["fid_feature_layer"] = exp["fid_feature_layer"]
        row["fid_label"] = "FID-1K" if exp["mscoco_prompts"] == 1000 and exp["fid_feature_layer"] == 2048 else f"FID-{exp['mscoco_prompts']}"
        if method == "original":
            row["fid_vs_original"] = 0.0
        else:
            previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            try:
                metrics = torch_fidelity.calculate_metrics(
                    input1=str(output / "images" / method / "coco"),
                    input2=str(output / "images" / "original" / "coco"),
                    cuda=torch.cuda.is_available(), batch_size=exp["fid_batch_size"],
                    feature_layer_fid=exp["fid_feature_layer"], fid=True,
                    isc=False, kid=False, prc=False, verbose=False)
            finally:
                if previous is None:
                    os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
                else:
                    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous
            row["fid_vs_original"] = float(metrics["frechet_inception_distance"])
    original_coco = aggregate[0]["mscoco_clip"]
    for row in aggregate:
        row["delta_mscoco_clip_vs_original"] = row["mscoco_clip"]-original_coco
    write_csv(metrics_dir / "comparison.csv", aggregate, list(aggregate[0]))
    write_json(done, {"identity": identity, "image_manifest": image_manifest})
    print(f"Metrics: {metrics_dir / 'comparison.csv'}")


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
