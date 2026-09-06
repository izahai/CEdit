# Vast AI: Few-Concept Artistic Style Legacy vs. TGPRS

This self-contained workflow reproduces the artistic-style portion of
`scripts/eval_few.sh` and compares three model states:

- original Stable Diffusion v1.4;
- legacy SPEED with `aug_num=10`; and
- `target_global_pairwise_residual_subspace` (TGPRS) with `aug_num=0`.

Every metric row records the effective training configuration, including the
intentional augmentation difference between legacy SPEED and TGPRS.

## Evaluation matrix

The workflow trains separate checkpoints that erase `Van Gogh`, `Picasso`,
or `Monet` into the `art` anchor. Each checkpoint is sampled on Van Gogh,
Picasso, Monet, Paul Gauguin, and Caravaggio using all 30 style templates from
`src/template.py`. It also evaluates preservation on the first 100 MS-COCO
prompts.

TGPRS uses 30 artist-neutral prompts derived by formatting the style templates
with `art`. None of the five evaluated artist names occur in the TGPRS
subspace-anchor list. Its configured and feasible residual rank is 30 for each
single-target task.

Sampling uses seed 0, DPM-Solver, 20 denoising steps, CFG 7.5, and the same
latent sequence for original and edited images. The full profile trains six
edited checkpoints and produces 1,750 PNG files. The smoke profile trains two
edited checkpoints and produces 90 PNG files.

## Upload and run

From the local repository, set the remote host and port:

```bash
export VAST_HOST='180.189.55.43'
export VAST_PORT='57595'

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
export WORKFLOW_CONFIG=/workspace/CEdit/remote_scripts/eval_few/eval_few_style_1/workflow_smoke.yaml
bash remote_scripts/eval_few/eval_few_style_1/run_all.sh
```

Run the full workflow by omitting the override:

```bash
cd /workspace/CEdit
unset WORKFLOW_CONFIG
bash remote_scripts/eval_few/eval_few_style_1/run_all.sh
```

The workflow is resumable. Force individual stages when necessary:

```bash
FORCE_RETRAIN=1 bash remote_scripts/eval_few/eval_few_style_1/03_train.sh
FORCE_RESAMPLE=1 bash remote_scripts/eval_few/eval_few_style_1/04_generate_edits.sh
FORCE_RESAMPLE=1 bash remote_scripts/eval_few/eval_few_style_1/05_generate_mscoco.sh
FORCE_EVAL=1 bash remote_scripts/eval_few/eval_few_style_1/06_evaluate.sh
```

`WORKFLOW_CONFIG`, `OUTPUT_ROOT`, `PYTHON_BIN`, `GPU_ID`,
`FID_FEATURE_LAYER`, and the sampling or metric batch-size variables can be
overridden without editing YAML.

When launched inside tmux, the complete pane output is appended to
`${WORKSPACE_DIR}/tmux-log.log`. Do not run `tail -f` for that file inside
the logged pane because it would feed the log back into itself.

## Outputs

The full output root is `/workspace/cedit_eval_few_style_1/`; the smoke output
root is `/workspace/cedit_eval_few_style_1_smoke/`:

```text
<output_root>/
├── checkpoints/{legacy,target_global_pairwise_residual_subspace}/<task>/weight.pt
├── images/original/style/shared/<content>/original/*.png
├── images/<method>/style/<task>/<content>/edit/*.png
├── mscoco/original/coco/original/*.png
├── mscoco/<method>/<task>/coco/edit/*.png
├── logs/train/*.log
└── metrics/
    ├── detailed_metrics.csv
    ├── summary.csv
    ├── comparison.csv
    └── fid_cache/*.pt
```

`detailed_metrics.csv` contains per-content CLIP score and FID.
`summary.csv` reports mean target CLIP score, mean non-target FID, and
MS-COCO CLIP/FID for every task and model. `comparison.csv` places legacy
and TGPRS side by side; positive improvement values always favor TGPRS.

Evaluation validates the exact expected filenames, checkpoints metric rows
atomically, and caches original-image FID statistics with image-manifest
fingerprints so interrupted runs can resume safely.
