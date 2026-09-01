# Vast AI: Few-Concept Instance TGPRS

This self-contained workflow reproduces the instance portion of
`scripts/eval_few.sh` and evaluates two model states:

- original Stable Diffusion v1.4;
- `target_global_pairwise_residual_subspace` (TGPRS) with `aug_num=0`.

Original images are retained only as the reference distribution. No Legacy or
SPEED checkpoint is trained, sampled, or evaluated by this workflow.

## Evaluation matrix

The checkpoints erase `Snoopy`, `Snoopy + Mickey`, and
`Snoopy + Mickey + Spongebob` into the null prompt. Every checkpoint is sampled
on all five configured instance contents and on the first 100 MS-COCO prompts.

The current TGPRS training config supplies 101 extra subspace anchors,
including the null prompt: 100 generic style-like prompts plus `""`. TGPRS
uses the configured nominal rank 30 for these tasks, subject to the measured
effective rank of the target-global span. Sampling uses
seed 0, DPM-Solver, 20 denoising steps, CFG 7.5, and the same latent sequence for
original and edited images.

An individual task can replace the TGPRS subspace-anchor list without changing
other tasks:

```yaml
tasks:
  - id: snoopy
    erase_type: instance
    target_concepts: [Snoopy]
    anchor_concept: ""
    subspace_anchor_concepts: ["", "person"]
```

Use `subspace_anchor_concepts: []` for target-pair residuals only. That requires
at least two targets; a one-target task with no subspace anchors is invalid.

The full workflow trains 3 edited checkpoints and produces 2,000 PNG files.
Use an instance with at least 100 GB of disk. The included smoke configuration
produces one edited checkpoint and 60 PNG files. The full profile reports
standard FID-2048; the smoke profile uses the 64-dimensional Inception feature
layer so its tiny 10-image comparisons finish quickly on a low-vCPU server.

## Upload and validate on the provided server

From the local repository:

```bash
export VAST_HOST='180.189.55.43'
export VAST_PORT='19478'

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

Generated image directories and common raster image files are excluded from
the upload.

The supplied server has only 32 GB of non-persistent disk, so run the smoke
workflow there:

```bash
ssh -p "${VAST_PORT}" -L 8080:localhost:8080 "root@${VAST_HOST}"
cd /workspace/CEdit
export WORKFLOW_CONFIG=/workspace/CEdit/remote_scripts/eval_few/eval_few_ins_1_tgprs/workflow_smoke.yaml
bash remote_scripts/eval_few/eval_few_ins_1_tgprs/run_all.sh
```

On a suitably sized instance, omit `WORKFLOW_CONFIG` to run the full workflow:

```bash
cd /workspace/CEdit
bash remote_scripts/eval_few/eval_few_ins_1_tgprs/run_all.sh
```

The workflow is resumable. Force individual work when necessary:

```bash
FORCE_RETRAIN=1 bash remote_scripts/eval_few/eval_few_ins_1_tgprs/03_train.sh
FORCE_RESAMPLE=1 bash remote_scripts/eval_few/eval_few_ins_1_tgprs/04_generate_edits.sh
FORCE_RESAMPLE=1 bash remote_scripts/eval_few/eval_few_ins_1_tgprs/05_generate_mscoco.sh
FORCE_EVAL=1 bash remote_scripts/eval_few/eval_few_ins_1_tgprs/06_evaluate.sh
```

`WORKFLOW_CONFIG`, `OUTPUT_ROOT`, `PYTHON_BIN`, `GPU_ID`, `FID_FEATURE_LAYER`,
and the sampling or metric batch-size variables can be overridden without
editing YAML.

## Outputs

The default output root is `/workspace/cedit_eval_few_ins_1_tgprs/`; the smoke
root is `/workspace/cedit_eval_few_ins_1_tgprs_smoke/`:

```text
<output_root>/
├── checkpoints/target_global_pairwise_residual_subspace/<task>/weight.pt
├── images/original/<domain>/shared/<content>/original/*.png
├── images/<method>/<domain>/<task>/<content>/edit/*.png
├── mscoco/original/coco/original/*.png
├── mscoco/<method>/<task>/coco/edit/*.png
├── logs/train/*.log
└── metrics/
    ├── detailed_metrics.csv
    ├── summary.csv
    ├── comparison.csv
    └── fid_cache/*.pt
```

`detailed_metrics.csv` contains per-content CLIP score and FID. It also records
the resolved subspace-anchor list and count. `summary.csv` reports mean target
CLIP score, mean non-target FID, and MS-COCO CLIP/FID for each task and model.
`comparison.csv` places the original reference and TGPRS side by side without
requiring Legacy results.

Metric evaluation validates the exact expected filenames. It checkpoints rows
atomically and caches original-image FID statistics using an image-manifest
fingerprint, so interrupted evaluation can resume safely. A scoped PyTorch 2.6
compatibility guard permits Torch-Fidelity 0.3 to reload only those locally
generated NumPy-statistics cache files.
