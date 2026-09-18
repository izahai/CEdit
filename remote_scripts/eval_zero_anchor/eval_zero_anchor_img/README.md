# Van Gogh Erasure: Normal Anchor ("art") vs Zero Anchor ("<zero>") vs Original SD 1.4

This directory provides an end-to-end evaluation pipeline comparing:
1. **Original Stable Diffusion v1.4** (unmodified baseline)
2. **SPEED with "art" anchor** (`anchor_concepts: "art"`)
3. **SPEED with zero anchor** (`anchor_concepts: "<zero>"`)

The evaluation produces **20 three-way horizontally concatenated images** of size `512 × 1536`:
- **10 Van Gogh prompts**: generated using the first 10 generic templates from `src/template.py` (`painting_templates[:10]`).
- **10 Random artist prompts**: 10 artists sampled randomly from `data/style.csv` with a fixed seed (`seed=42`), evaluated using the same 10 templates.

Each concatenated image stitches `[Original | Art Anchor | Zero Anchor]` side-by-side using the exact same latent noise seed across all 3 models. No single images are saved, and no metric calculations are performed.

---

## Directory Structure

```text
remote_scripts/eval_zero_anchor/
├── README.md               # Documentation and execution guide
├── train_art.yaml          # SPEED training config for anchor "art"
├── train_zero.yaml         # SPEED training config for anchor "<zero>"
├── preview_concat.py       # Inference and horizontal 3-image concatenation script
├── run_all.sh              # Orchestrating bash script for remote execution
└── outputs/                # Generated artifacts (created at runtime)
    ├── checkpoints/
    │   ├── art/weight.pt
    │   └── zero/weight.pt
    └── concat_images/
        ├── van_gogh/       # 10 concatenated images [512x1536]
        └── random_artists/ # 10 concatenated images [512x1536]
```

---

## 1. Remote Server Access & Code Upload (Vast.ai)

### SSH Connection

```bash
ssh -p 26546 root@202.122.49.242 -L 8080:localhost:8080
```

### Sync Code to Server (`rsync`)

From your local machine repository root (`/Users/hainguyen/Repo/2026/ConceptErasure/Working/ReSam`), upload the codebase to `/workspace/ReSam/` while excluding large weights, local git files, logs, and caches (per `docs/vast_ai_AGENTS.md`):

```bash
rsync -avz --progress -e 'ssh -p 26546' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'logs/' \
  --exclude 'outputs/' \
  --exclude 'result*/' \
  --exclude '*.png' \
  --exclude '*.jpg' \
  --exclude '*.jpeg' \
  --exclude '*.safetensors' \
  --exclude '*.pt' \
  --exclude '*.bin' \
  --exclude '.pytest_cache/' \
  --exclude 'tmp/' \
  ./ root@202.122.49.242:/workspace/ReSam/
```

> **Tip:** Add `--dry-run` (or `-n`) to preview the transferred files before copying.

---

## 2. Remote Execution

On the remote server:

```bash
# 1. Navigate to the repository
cd /workspace/ReSam

# 2. Activate Python environment (as documented in docs/vast_ai_AGENTS.md)
source /venv/main/bin/activate

# 3. Run the end-to-end pipeline (auto-installs requirements via uv, trains, and generates images)
GPU_ID=0 bash remote_scripts/eval_zero_anchor/run_all.sh
```

Customization options:
```bash
# Example: Use GPU 1, 50 inference steps, skip dependency install, force retrain
GPU_ID=1 STEPS=50 INSTALL_DEPS=0 FORCE_RETRAIN=1 bash remote_scripts/eval_zero_anchor/run_all.sh
```

---

## 3. Download Generated 3-Way Comparisons Back to Local

### Option A: In-Memory Tar Stream over SSH (Fastest & Direct to Timestamped Folder)

Streams the 20 concatenated images directly from the remote server into a timestamped local folder over a single TCP connection. Excludes all weight and checkpoint files to guarantee only images are transferred, without writing temporary archive files on either machine:

```bash
OUT_DIR="remote_scripts/eval_zero_anchor/outputs/results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUT_DIR}" && \
ssh -p 26546 root@202.122.49.242 \
  "tar -C /workspace/ReSam/remote_scripts/eval_zero_anchor/outputs/concat_images \
   --exclude='*.safetensors' --exclude='*.pt' --exclude='*.ckpt' --exclude='*.bin' \
   -cf - ." | tar -C "${OUT_DIR}" -xf -
```

> **Why this is fast:** Streams directly in memory over SSH without per-file handshake overhead, extracting `van_gogh/` and `random_artists/` subfolders instantly into `${OUT_DIR}`.

### Option B: Rsync (Mirrored Directory)

If you prefer `rsync` to mirror directly into `remote_scripts/eval_zero_anchor/outputs/concat_images/`:

```bash
rsync -avz --progress -e 'ssh -p 26546' \
  --exclude='*.safetensors' \
  --exclude='*.pt' \
  --exclude='*.ckpt' \
  --exclude='*.bin' \
  root@202.122.49.242:/workspace/ReSam/remote_scripts/eval_zero_anchor/outputs/concat_images/ \
  ./remote_scripts/eval_zero_anchor/outputs/concat_images/
```

---

## 4. Individual Step Execution (Manual)

If you prefer to execute each stage manually on the remote server:

### Step 1: Train SPEED with "art" Anchor
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_erase_null.py \
    --config remote_scripts/eval_zero_anchor/train_art.yaml
```

### Step 2: Train SPEED with "<zero>" Anchor
```bash
CUDA_VISIBLE_DEVICES=0 python3 train_erase_null.py \
    --config remote_scripts/eval_zero_anchor/train_zero.yaml
```

### Step 3: Generate 3-Way Concatenated Images
```bash
CUDA_VISIBLE_DEVICES=0 python3 remote_scripts/eval_zero_anchor/preview_concat.py \
    --sd_ckpt "CompVis/stable-diffusion-v1-4" \
    --art_ckpt "remote_scripts/eval_zero_anchor/outputs/checkpoints/art/weight.pt" \
    --zero_ckpt "remote_scripts/eval_zero_anchor/outputs/checkpoints/zero/weight.pt" \
    --style_csv "data/style.csv" \
    --output_dir "remote_scripts/eval_zero_anchor/outputs/concat_images" \
    --steps 30 \
    --seed 42 \
    --device "cuda"
```
