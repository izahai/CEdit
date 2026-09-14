# Vast AI: Few-Concept Artistic Style Legacy vs. CFB

This self-contained workflow benchmarks three model states:

- original Stable Diffusion v1.4;
- legacy SPEED with `aug_num=10`; and
- Closed-Form Backprop (`cfb`) with `aug_num=0`.

## Evaluation matrix

The workflow trains separate legacy and CFB checkpoints that erase `Van Gogh`,
`Picasso`, or `Monet` into the `art` anchor. Each checkpoint is sampled on Van
Gogh, Picasso, Monet, Paul Gauguin, and Caravaggio using all 30 style templates
from `src/template.py`, with one image per prompt. Preservation is evaluated on
the first 100 MS-COCO prompts.

Sampling uses seed 0, DPM-Solver, 20 denoising steps, CFG 7.5, and the same
latent sequence for original and edited images. The full profile produces 1,750
PNG files; the smoke profile produces 90.

CFB uses style-aware target embeddings and the fixed benchmark configuration
in `train_config_cfb.yaml`: retain projection rank 150, 200 anchor steps,
`use_k2=false`, and one final validation sample.

## Upload and run

From the local repository, configure the Vast AI host and upload the code:

```bash
export VAST_HOST='182.224.239.168'
export VAST_PORT='40237'

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
export WORKFLOW_CONFIG=/workspace/CEdit/remote_scripts/eval_few/eval_few_style_cfb/workflow_smoke.yaml
bash remote_scripts/eval_few/eval_few_style_cfb/run_all.sh
```

Run the full benchmark:

```bash
cd /workspace/CEdit
unset WORKFLOW_CONFIG
bash remote_scripts/eval_few/eval_few_style_cfb/run_all.sh
```

The workflow is resumable. Force individual stages when necessary:

```bash
FORCE_RETRAIN=1 bash remote_scripts/eval_few/eval_few_style_cfb/03_train.sh
FORCE_RESAMPLE=1 bash remote_scripts/eval_few/eval_few_style_cfb/04_generate_edits.sh
FORCE_RESAMPLE=1 bash remote_scripts/eval_few/eval_few_style_cfb/05_generate_mscoco.sh
FORCE_EVAL=1 bash remote_scripts/eval_few/eval_few_style_cfb/06_evaluate.sh
```

`WORKFLOW_CONFIG`, `OUTPUT_ROOT`, `PYTHON_BIN`, `GPU_ID`,
`FID_FEATURE_LAYER`, and sampling or metric batch sizes can be overridden
without editing YAML.

## Outputs

The full output root is `/workspace/cedit_eval_few_style_cfb/`; the smoke root
is `/workspace/cedit_eval_few_style_cfb_smoke/`:

```text
<output_root>/
├── checkpoints/legacy/<task>/weight.pt
├── checkpoints/cfb/<task>/weight.safetensors
├── images/original/style/shared/<content>/original/*.png
├── images/{legacy,cfb}/style/<task>/<content>/edit/*.png
├── mscoco/original/coco/original/*.png
├── mscoco/{legacy,cfb}/<task>/coco/edit/*.png
├── logs/train/*.log
└── metrics/
    ├── detailed_metrics.csv
    ├── summary.csv
    ├── comparison.csv
    └── fid_cache/*.pt
```

`detailed_metrics.csv` contains per-content CLIP score and FID. `summary.csv`
reports target erasure and non-target/MS-COCO preservation for each model.
`comparison.csv` places legacy and CFB side by side; positive improvement
values always favor CFB.

Evaluation validates exact filenames, checkpoints metric rows atomically, and
caches original-image FID statistics so interrupted runs can resume safely.
