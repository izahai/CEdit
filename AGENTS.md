# Repository Guidelines

## Project Structure & Module Organization

This repository implements SPEED concept erasure for Stable Diffusion v1.4. Top-level entry points are `train_erase_null.py` (model editing), `sample.py` (few-concept sampling), and `sample2.py` (benchmark and multi-concept sampling). Shared Python code lives in `src/`: `utils.py` contains common helpers, `template.py` owns prompt templates, and `*_cal.py` files calculate metrics. Keep benchmark CSV inputs under `data/`; place generated checkpoints and images under `logs/` rather than committing them. Shell workflows live in `scripts/`, while static figures belong in `assets/`.

## Setup, Build, and Development Commands

Use Python 3.10 with CUDA-capable PyTorch:

```bash
conda create -n SPEED -y python=3.10 && conda activate SPEED
pip install torch==2.3.0 torchvision==0.18.0
pip install -r requirements.txt
```

Run a focused edit with `CUDA_VISIBLE_DEVICES=0 python train_erase_null.py --target_concepts "Snoopy" --anchor_concepts "" --retain_path data/instance.csv --heads concept`. Generate outputs with `python sample.py ...`, following the README arguments. `bash scripts/eval_few.sh`, `bash scripts/eval_multi.sh`, and `bash scripts/eval_nudity.sh` execute GPU-heavy end-to-end evaluations; review their GPU arrays and output paths before running. `bash data/pretrain/pretrain_sample.sh` creates baseline samples.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for variables and functions, and concise comments around algorithm stages. Keep command-line arguments in `argparse` and preserve established names such as `target_concepts`, `retain_path`, and `save_root`. Add shared logic to `src/` instead of duplicating it across entry points. Use lowercase, underscore-separated data and experiment identifiers (for example, `10_celebrity`); quote shell variables and keep GPU selection explicit via `CUDA_VISIBLE_DEVICES`.

## Testing & Evaluation

There is no unit-test suite or configured formatter/linter. For code changes, first run an applicable small command (one concept and one GPU), verify that the edited weight file is written, then sample and inspect the expected `logs/.../edit` output. Use the relevant evaluation script for full regression checks; these require model downloads, CUDA, and may take substantial time.

## Commit & Pull Request Guidelines

This checkout has no accessible Git metadata, so no repository-specific commit convention can be inferred. Use concise imperative subjects, such as `Add retain-set validation`, and keep commits focused. Pull requests should state the experiment or behavior changed, list commands and hardware used for validation, link related issues when available, and include representative output paths or before/after images for sampling changes. Do not commit model weights, generated images, credentials, or other large artifacts.
