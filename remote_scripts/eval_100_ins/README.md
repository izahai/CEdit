# 100-instance SPEED evaluation

This workflow compares original SD v1.4, SPEED, and Ours after a single 100-instance edit for each method. Erasure uses IDs 1–100 in `data/instance.csv`; retention evaluation uses IDs 101–200. The training retain CSV has 1,234 distinct non-target concepts after trimming names and keeping the first ID for each case-insensitive name. The source CSV is unchanged.

The full run uses two prompts for every instance: `An image of {concept}.` and `An illustration of {concept}.` With seed 0, this yields 200 erasure and 200 retention images per model. The first 100 MS-COCO prompts add 100 images per model: **1,500 images total**. Original, SPEED, and Ours use identical prompts and initial latents. Sampling uses DPM-Solver, 20 steps, and CFG 7.5.

SPEED maps targets to the blank anchor with `aug_num: 10`. Ours uses the target-global pairwise residual subspace with rank 30, a blank extra anchor, and `aug_num: 0`. Both use `params: V`, `threshold: 0.1`, and `retain_scale: 1.0`. These augmentation settings differ, so their comparison includes that difference.

## Run on a CUDA-ready Vast PyTorch instance

Upload only the workflow inputs, avoiding generated images and caches. Set `VAST_HOST` to the SSH destination and `VAST_PORT` to its SSH port on your local machine:

```bash
rsync -az --relative --exclude='__pycache__/' --exclude='*.pyc' \
  -e "ssh -p ${VAST_PORT}" requirements.txt train_erase_null.py src/ \
  data/instance.csv data/instance_100_erase.csv \
  data/instance_100_retain_eval.csv data/instance_100_retain_train.csv \
  data/mscoco.csv remote_scripts/eval_100_ins/ \
  "${VAST_HOST}:/workspace/CEdit/"
```

On the instance, `/venv/main/bin/python` is the default; override `PYTHON_BIN`, `GPU_ID`, or `OUTPUT_ROOT` as needed:

```bash
cd /workspace/CEdit
PYTHON_BIN=/venv/main/bin/python GPU_ID=0 OUTPUT_ROOT=/workspace/eval_100_ins \
  WORKFLOW_CONFIG=remote_scripts/eval_100_ins/workflow_smoke.yaml \
  bash remote_scripts/eval_100_ins/run_all.sh
PYTHON_BIN=/venv/main/bin/python GPU_ID=0 OUTPUT_ROOT=/workspace/eval_100_ins \
  bash remote_scripts/eval_100_ins/run_all.sh
```

The smoke run trains both methods on three targets and evaluates three retention instances and eight COCO prompts. It keeps the full training retain set and caps Ours' rank at three. The setup stage installs `requirements.txt` and checks CUDA; training, sampling, and evaluation download SD and CLIP through their normal libraries on first use.

Resume an interrupted full run using the same configuration and output root, for example:

```bash
START_STAGE=sample OUTPUT_ROOT=/workspace/eval_100_ins \
  bash remote_scripts/eval_100_ins/run_all.sh
```

Each run is isolated under `OUTPUT_ROOT/runs/<fingerprint>/`. Changed inputs create a new run; checkpoints and images are reused only when their configuration and manifests match. Sampling progress appears in the terminal, and all stage output is appended to `OUTPUT_ROOT/workflow.log`.

Download results from your local machine:

```bash
rsync -az -e "ssh -p ${VAST_PORT}" \
  "${VAST_HOST}:/workspace/eval_100_ins/" ./eval_100_ins_results/
```

`metrics/image_clip.csv` contains image–prompt CLIP scores scaled by 100. `metrics/per_instance.csv` contains each instance's mean and change from original. `metrics/comparison_clip.csv` contains macro-averaged erasure and retention CLIP, COCO CLIP, and 95% bootstrap intervals from resampling instances. The report labels are **Original SD v1.4**, **SPEED**, and **Ours**. No FID is computed.
