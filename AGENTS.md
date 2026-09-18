# Repository Guidelines

## Project Structure

This repository implements SPEED concept erasure for Stable Diffusion v1.4. Main entry points are `train_erase_null.py` for model editing, `sample.py` for few-concept sampling, and `sample2.py` for benchmark and multi-concept sampling. Put reusable Python logic in `src/`; prompt templates live in `src/template.py`, while `*_cal.py` modules calculate metrics. Benchmark CSVs belong in `data/`, tests in `tests/`, local workflows in `scripts/`, and reproducible remote pipelines in `remote_scripts/`. Store generated checkpoints and images under `logs/` or workflow output directories; do not commit them.

Run unit tests with:

```bash
python -m unittest discover -s tests
```

On the remote CUDA server, a focused edit can use `CUDA_VISIBLE_DEVICES=0 python train_erase_null.py --target_concepts "Snoopy" --anchor_concepts "" --retain_path data/instance.csv --heads concept`. Use `sample.py` or `sample2.py` to inspect the result there. Full evaluations are available through `scripts/eval_few.sh`, `scripts/eval_multi.sh`, and `scripts/eval_nudity.sh`; check GPU arrays and output paths before running them.

## Coding Style

Use four-space indentation, `snake_case` for Python names, and concise comments around algorithm stages. Keep CLI options in `argparse` and preserve established terms such as `target_concepts`, `anchor_mode`, `retain_path`, and `save_root`. Add shared functionality to `src/` instead of duplicating entry-point code. Use lowercase underscore-separated experiment names, for example `100_celebrity`. Quote shell variables and make GPU selection explicit.

## Testing and Evaluation

Name tests `test_*.py` and use `unittest`. Add focused numerical tests for residual construction and configuration tests for new modes. Run the smallest relevant tests first, then full discovery.

The local environment is a MacBook. Local tests must be CPU-only and must not download pretrained model weights or run GPU workflows. Use synthetic tensors, small fake models, and mocks for local validation. Heavy tests, model-backed training, sampling, and evaluations run only on a remote CUDA server. If a task needs those checks, ask the user for the SSH command to that server before attempting them. Report clearly when validation was limited to local CPU tests.
