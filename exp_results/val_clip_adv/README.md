# Colab workflow: Van Gogh learned-anchor evaluation

This full evaluation compares four model states on erasing the `Van Gogh`
style:

1. original Stable Diffusion v1.4;
2. legacy SPEED with the text anchor `art`;
3. `target_global_pairwise_residual_subspace` (TGPRS); and
4. SPEED with the CLIP-guided learned anchor.

It uses all 30 style templates, 10 images per template, five artist contents,
and the first 1,000 MS-COCO prompts. It generates 10,000 evaluation PNG files
and requires a CUDA runtime. Reserve at least 100 GiB of free storage. This
workflow also includes a reduced smoke profile for end-to-end infrastructure
validation.

For the smoke run, use all 30 templates but generate one image per template and
only the first 50 MS-COCO prompts. Across the four methods this produces 800
evaluation PNGs. It is useful for verifying the pipeline, but its FID values are
too sample-limited for publication-quality conclusions.

## Metrics

The evaluator reports:

- target CLIP score (lower is stronger Van Gogh erasure);
- target FID against matched original images (descriptive only);
- mean CLIP and FID over Picasso, Monet, Paul Gauguin, and Caravaggio;
- MS-COCO CLIP score (higher is better preservation); and
- MS-COCO FID (lower is better preservation).

CLIP uses `openai/clip-vit-large-patch14`. FID uses the 2048-dimensional
Inception feature layer. Every method uses seed 0, DPM-Solver, 20 denoising
steps, and CFG 7.5.

## 1. Start from an empty Colab runtime

Select a GPU runtime, open its terminal, and clone the revision containing this
workflow. The current feature branch must be pushed before it can be cloned;
after merge, use `main` instead.

```bash
export CEDIT_REPO_URL="https://github.com/izahai/CEdit.git"
export CEDIT_GIT_REF="codex/CEdit-clip-learnt-anchor"

cd /content
git clone --branch "${CEDIT_GIT_REF}" --single-branch \
    "${CEDIT_REPO_URL}" CEdit
cd /content/CEdit
git rev-parse --show-toplevel
git log -1 --oneline
```

If the repository is private, authenticate Git before cloning. Do not place a
personal token in this repository or in the workflow YAML.

Stable Diffusion v1.4 may require Hugging Face authentication for your account.
If model loading returns an authorization error:

```bash
python3 -m pip install --upgrade huggingface_hub
read -rsp "Hugging Face token: " HF_TOKEN && echo
export HF_TOKEN
hf auth login --token "${HF_TOKEN}"
unset HF_TOKEN
```

## 2. Choose persistent output storage

By default, results are written to `/content/cedit_val_clip_adv`, which is lost
when the Colab runtime is recycled. To persist a long run, mount Google Drive
from a notebook cell first:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then set the output directory in the terminal:

```bash
export OUTPUT_ROOT="/content/drive/MyDrive/cedit_val_clip_adv"
```

For faster generation, keep `OUTPUT_ROOT` under `/content` and archive the
result to Drive at the end. The workflow validates free disk space and stops
below 100 GiB unless `ALLOW_LOW_DISK=1` is explicitly set.

## 3. Validate and run in tmux

Set the Python interpreter and run the non-GPU validation first:

```bash
cd /content/CEdit
export PYTHON_BIN="$(command -v python3)"
bash exp_results/val_clip_adv/00_validate.sh
```

Install `tmux` so a disconnected terminal does not terminate the shell process:

```bash
sudo apt-get update
sudo apt-get install -y tmux
tmux new -s val_clip_adv
```

Inside tmux, restore the variables and start the full workflow:

```bash
cd /content/CEdit
export PYTHON_BIN="$(command -v python3)"
# Re-run this line only when using mounted Drive:
# export OUTPUT_ROOT="/content/drive/MyDrive/cedit_val_clip_adv"

mkdir -p "${OUTPUT_ROOT:-/content/cedit_val_clip_adv}/logs"
bash exp_results/val_clip_adv/run_all.sh 2>&1 | \
    tee "${OUTPUT_ROOT:-/content/cedit_val_clip_adv}/logs/run_all.log"
```

To run the 800-image smoke profile instead, set its config before invoking the
same staged workflow. Use a distinct output root so smoke artifacts cannot be
mistaken for the full evaluation:

```bash
export WORKFLOW_CONFIG="$(pwd)/exp_results/val_clip_adv/workflow_smoke.yaml"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/content/cedit_val_clip_adv_smoke}"
bash exp_results/val_clip_adv/run_all.sh 2>&1 | \
    tee "${OUTPUT_ROOT}/logs/run_all.log"
```

Detach with `Ctrl-b`, then `d`. Reattach later with:

```bash
tmux attach -t val_clip_adv
```

Colab can still recycle the entire runtime; tmux protects only against a closed
terminal. Persistent output storage is required to resume after recycling.

## Stage order and resuming

`run_all.sh` executes:

```text
00_validate.sh
01_setup_environment.sh
02_learn_anchor.sh
03_generate_original.sh
04_train.sh
05_generate_edits.sh
06_generate_mscoco.sh
07_evaluate.sh
```

Completed anchor artifacts, checkpoints, image sets, and metric rows are reused.
It is safe to invoke `run_all.sh` again after interruption. Individual stages
can also be run directly, for example:

```bash
bash exp_results/val_clip_adv/02_learn_anchor.sh
bash exp_results/val_clip_adv/04_train.sh
bash exp_results/val_clip_adv/07_evaluate.sh
```

Force controls are opt-in:

```bash
FORCE_RELEARN=1 bash exp_results/val_clip_adv/02_learn_anchor.sh
FORCE_RESAMPLE=1 bash exp_results/val_clip_adv/03_generate_original.sh
FORCE_RETRAIN=1 bash exp_results/val_clip_adv/04_train.sh
FORCE_RESAMPLE=1 bash exp_results/val_clip_adv/05_generate_edits.sh
FORCE_RESAMPLE=1 bash exp_results/val_clip_adv/06_generate_mscoco.sh
FORCE_EVAL=1 bash exp_results/val_clip_adv/07_evaluate.sh
```

`FORCE_RELEARN=1` moves the existing anchor directory to a timestamped backup.
`FORCE_RESAMPLE=1` removes only PNG files inside the configured output root.
`FORCE_RETRAIN=1` replaces the three `weight.pt` checkpoint files.
Downstream edited images are automatically refreshed when their checkpoint is
newer, so a relearned anchor cannot silently reuse generations from an older
learned-anchor checkpoint.

Useful runtime overrides include `GPU_ID`, `PYTHON_BIN`, `OUTPUT_ROOT`,
`BATCH_SIZE`, `MSCOCO_BATCH_SIZE`, `CLIP_BATCH_SIZE`, and `FID_BATCH_SIZE`.
Reducing a batch size changes memory use but not the configured sample count.

## Method configurations

- SPEED: `anchor_source=text`, `anchor=art`, legacy residual, `params=V`,
  `aug_num=10`, threshold `0.1`, retain scale `1.0`.
- TGPRS: text anchor `art`, 100 artist-neutral subspace anchors, rank `30`,
  `erase_style=false`, `aug_num=0`, threshold `1.0`, retain scale `1.0`.
- Learned anchor: four categorical prefix tokens, four training reference
  images, one validation image, 1,000 optimization iterations, and 16 fixed
  validation noise/timestep samples. Its SPEED edit matches the legacy SPEED
  hyperparameters; only the anchor source changes.

## Outputs

```text
<output_root>/
├── learned_anchors/van_gogh/
│   ├── manifest.json
│   ├── embeddings.safetensors
│   ├── metrics.jsonl
│   └── previews/
├── checkpoints/
│   ├── legacy/van_gogh/weight.pt
│   ├── target_global_pairwise_residual_subspace/van_gogh/weight.pt
│   └── clip_guided_learned_anchor/van_gogh/weight.pt
├── images/{original,legacy,target_global_pairwise_residual_subspace,clip_guided_learned_anchor}/
├── mscoco/{original,legacy,target_global_pairwise_residual_subspace,clip_guided_learned_anchor}/
├── logs/
└── metrics/
    ├── detailed_metrics.csv
    ├── summary.csv
    ├── comparison.csv
    ├── report.md
    └── fid_cache/
```

Inspect progress and final results with:

```bash
tail -n 100 "${OUTPUT_ROOT:-/content/cedit_val_clip_adv}/logs/run_all.log"
cat "${OUTPUT_ROOT:-/content/cedit_val_clip_adv}/metrics/report.md"
column -s, -t < "${OUTPUT_ROOT:-/content/cedit_val_clip_adv}/metrics/comparison.csv"
```

Positive `*_improvement` columns in `comparison.csv` favor TGPRS or the learned
anchor over SPEED. `target_fid_delta` and `non_target_clip_score_delta` are raw
deltas and deliberately have no assumed direction.

## Archive ephemeral results

When output was kept under `/content`, create one archive before ending the
runtime:

```bash
export OUTPUT_ROOT="${OUTPUT_ROOT:-/content/cedit_val_clip_adv}"
tar -C "$(dirname -- "${OUTPUT_ROOT}")" -czf \
    /content/val_clip_adv_results.tar.gz "$(basename -- "${OUTPUT_ROOT}")"
ls -lh /content/val_clip_adv_results.tar.gz
```

Copy that archive to mounted Drive or download it before disconnecting.
