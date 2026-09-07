# Vast AI: Few-Concept Artistic Style TGPRS

This self-contained workflow reproduces the artistic-style portion of
`scripts/eval_few.sh` and evaluates two model states:

- original Stable Diffusion v1.4;
- `target_global_pairwise_residual_subspace` (TGPRS) with `aug_num=0`.

Original images are retained only as the reference distribution. No Legacy or
SPEED checkpoint is trained, sampled, or evaluated by this workflow.

## Evaluation matrix

The workflow trains separate checkpoints that erase `Van Gogh`, `Picasso`, or
`Monet` into the `art` anchor. Each checkpoint is sampled on Van Gogh, Picasso,
Monet, Paul Gauguin, and Caravaggio using all 30 style templates from
`src/template.py`. Preservation is also evaluated on the first 1,000 MS-COCO
prompts.

TGPRS uses the 30 style templates formatted with `art` as artist-neutral
subspace anchors. None of the five evaluated artist names occurs in that list.
The configured and feasible residual rank is 30 for every single-target task.
Training uses threshold `0.3`, residual scale `1.0`, and retain scale `0.5`.

Sampling uses seed 0, DPM-Solver, 20 denoising steps, CFG 7.5, and the same
latent sequence for original and edited images. The full profile trains three
edited checkpoints and produces 10,000 PNG files. Use an instance with at least
100 GB of disk. The smoke profile trains one checkpoint and produces 60 PNG
files; it uses the 64-dimensional Inception feature layer for its small FID
comparison.

## Upload and run

From the local repository, set the remote host and SSH port:

```bash
export VAST_HOST='114.34.26.236'
export VAST_PORT='41908'

rsync -az --progress \
  --exclude='.git/' \
  --exclude='logs/' \
  --exclude='images/' \
  --exclude='mscoco/' \
  --exclude='*.png' \
  --exclude='*.jpg' \
  --exclude='*.jpeg' \
  --exclude='*.webp' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  -e "ssh -p ${VAST_PORT}" \
  ./ "root@${VAST_HOST}:/workspace/CEdit/"
```

Run the smoke profile first:

```bash
ssh -p "${VAST_PORT}" -L 8080:localhost:8080 "root@${VAST_HOST}"
cd /workspace/CEdit
export WORKFLOW_CONFIG=/workspace/CEdit/remote_scripts/eval_few/eval_few_style_1_tgprs/workflow_smoke.yaml
bash remote_scripts/eval_few/eval_few_style_1_tgprs/run_all.sh
```

Run the full workflow by removing the override:

```bash
cd /workspace/CEdit
unset WORKFLOW_CONFIG
bash remote_scripts/eval_few/eval_few_style_1_tgprs/run_all.sh
```

The workflow is resumable. Force individual stages when necessary:

```bash
FORCE_RETRAIN=1 bash remote_scripts/eval_few/eval_few_style_1_tgprs/03_train.sh
FORCE_RESAMPLE=1 bash remote_scripts/eval_few/eval_few_style_1_tgprs/04_generate_edits.sh
FORCE_RESAMPLE=1 bash remote_scripts/eval_few/eval_few_style_1_tgprs/05_generate_mscoco.sh
FORCE_EVAL=1 bash remote_scripts/eval_few/eval_few_style_1_tgprs/06_evaluate.sh
```

`WORKFLOW_CONFIG`, `OUTPUT_ROOT`, `PYTHON_BIN`, `GPU_ID`, `FID_FEATURE_LAYER`,
and the sampling or metric batch-size variables can be overridden without
editing YAML.

When launched inside tmux, pane output is appended to
`${WORKSPACE_DIR}/tmux-log.log`. Do not run `tail -f` on that file inside the
logged pane because it would feed the log back into itself.

## Outputs

The default output root is `/workspace/cedit_eval_few_style_1_tgprs/`; the smoke
root is `/workspace/cedit_eval_few_style_1_tgprs_smoke/`:

```text
<output_root>/
├── checkpoints/target_global_pairwise_residual_subspace/<task>/weight.pt
├── images/original/style/shared/<content>/original/*.png
├── images/target_global_pairwise_residual_subspace/style/<task>/<content>/edit/*.png
├── mscoco/original/coco/original/*.png
├── mscoco/target_global_pairwise_residual_subspace/<task>/coco/edit/*.png
├── logs/train/*.log
└── metrics/
    ├── detailed_metrics.csv
    ├── summary.csv
    ├── comparison.csv
    └── fid_cache/*.pt
```

`detailed_metrics.csv` contains per-content CLIP score and FID and records the
resolved subspace-anchor list and rank. `summary.csv` reports mean target CLIP,
mean non-target FID, and MS-COCO CLIP/FID for each task and model.
`comparison.csv` places the original reference and TGPRS side by side without
requiring Legacy results.

Metric evaluation validates exact expected filenames, checkpoints rows
atomically, and caches original-image FID statistics with image-manifest
fingerprints so interrupted evaluation can resume safely.
