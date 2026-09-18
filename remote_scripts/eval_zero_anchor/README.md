# Van Gogh Erasure Evaluation: Art Anchor vs Zero Anchor

This directory implements a comprehensive evaluation suite comparing:
1. **Original Stable Diffusion v1.4** (unmodified base model)
2. **SPEED with "art" anchor** (`anchor_concepts: "art"`)
3. **SPEED with "<zero>" anchor** (`anchor_concepts: "<zero>"`)

---

## Directory Structure

```text
remote_scripts/eval_zero_anchor/
├── README.md                      # Comprehensive documentation
├── run_100.sh                     # 100-image benchmark runner (20 steps, end-to-end metrics)
├── preview_concat.py              # Proxy to preview_concat generator
├── eval_zero_anchor_img/          # Image generation pipeline & configs
│   ├── README.md                  # Detailed 10-image workflow documentation
│   ├── preview_concat.py          # Core generator stitching [Original | Art | Zero]
│   ├── run_all.sh                 # 10-image workflow runner (train + sample)
│   ├── train_art.yaml             # SPEED config (art anchor)
│   └── train_zero.yaml            # SPEED config (<zero> anchor)
├── eval_metrics/                  # Fast CLIP evaluation suite
│   ├── README.md                  # Metrics formulation and interpretation
│   ├── compute_quick_metrics.py   # Directly crops 3-way images and computes CLIP scores
│   └── run_eval_metrics.sh        # Bash runner for metrics computation
└── outputs/                       # (Not committed) Generated checkpoints and images
    ├── checkpoints/               # Trained weights (art/weight.pt, zero/weight.pt)
    ├── concat_images/             # 10-image comparisons (512x1536)
    ├── concat_images_100/         # 100-image comparisons (512x1536)
    └── eval_metrics_100/          # Summary table, JSON, and 200-row CSV
```

---

## 1. Fast 100-Image Benchmark Evaluation (`run_100.sh`)

Evaluates **100 Van Gogh target images** and **100 Retain artist images** (200 three-way comparisons, 600 model inferences total) using **20 diffusion timesteps**.

### Prompt & Seed Strategy
* **5 Canonical Templates**:
  1. `"An image in the style of {artist}."`
  2. `"A portrait in the style of {artist}."`
  3. `"A landscape in the style of {artist}."`
  4. `"A city street in the style of {artist}."`
  5. `"A still life in the style of {artist}."`
* **Target (Van Gogh)**: 5 templates cycled $\times$ 20 distinct seeds = 100 images.
* **Retain**: 100 unique artists from `data/style_100_retain_eval.csv` paired cyclically across the 5 templates with matching seeds.
* **Diffusion Steps**: 20 timesteps (`DPMSolverMultistepScheduler`).
* **Checkpoints**: Reuses the already-trained `weight.pt` checkpoints (zero retraining).

### Run on Remote Server

```bash
cd /workspace/ReSam
source /venv/main/bin/activate
GPU_ID=0 bash remote_scripts/eval_zero_anchor/run_100.sh
```

This single command:
1. Checks that Art & Zero checkpoints exist.
2. Generates the 200 concatenated images into `outputs/concat_images_100/`.
3. Evaluates CLIP metrics and prints the formatted summary comparison table.
4. Saves `summary_metrics.json`, `summary_table.txt`, and `per_image_metrics.csv`.

### Download 100-Image Results to Laptop

Stream download all 200 comparison images in memory directly to your local laptop:

```bash
OUT_DIR="remote_scripts/eval_zero_anchor/outputs/results_100_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUT_DIR}" && \
ssh -p 26546 root@202.122.49.242 \
  "tar -C /workspace/ReSam/remote_scripts/eval_zero_anchor/outputs/concat_images_100 \
   --exclude='*.safetensors' --exclude='*.pt' --exclude='*.ckpt' --exclude='*.bin' \
   -cf - ." | tar -C "${OUT_DIR}" -xf -
```

---

## 2. 10-Image Preview Workflow (`eval_zero_anchor_img/run_all.sh`)

For a quick 10-image visual check (30 timesteps):

```bash
cd /workspace/ReSam
source /venv/main/bin/activate
GPU_ID=0 bash remote_scripts/eval_zero_anchor/eval_zero_anchor_img/run_all.sh
```

---

## 3. Metrics Evaluation Standalone (`eval_metrics/`)

To re-run metrics locally on your laptop without GPU:

```bash
python3 remote_scripts/eval_zero_anchor/eval_metrics/compute_quick_metrics.py
```
*(Automatically detects downloaded images and appropriate prompt mode)*.

---

## 4. Code Synchronization (Local Laptop -> Remote Server)

To upload newly updated scripts to the Vast.ai server:

```bash
rsync -avz --progress -e 'ssh -p 26546' \
  --exclude='.git' \
  --exclude='outputs' \
  --exclude='logs' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  ./remote_scripts/eval_zero_anchor/ \
  root@202.122.49.242:/workspace/ReSam/remote_scripts/eval_zero_anchor/
```

