# ReSAM: Van Gogh Style Erasure Workflow

This folder contains the complete reproducible workflow to erase **Van Gogh** style using **ReSAM** (Retain-Guided Sparse Anchor Mixture) with Stable Diffusion v1.4, using `data/style.csv` for both retain preservation and the candidate concept bank.

---

## 1. Files in this Directory

- **[`train_config.yaml`](file:///Users/hainguyen/Repo/2026/ConceptErasure/Working/ReSam/remote_scripts/eval_ReSAM/erase_van_gogh/train_config.yaml)**: YAML training configuration for ReSAM.
  - Target: `Van Gogh`
  - Candidate Concepts: `data/style.csv` (~1,700 artists; `"Van Gogh"` is automatically filtered out)
  - Retain Concepts: `data/style.csv`
  - Sparsity: $k = 5$ active candidates
  - Steps: 200 ($lr = 0.05$)
  - Objective: Retain MSE loss + Null-prompt preservation loss
  - Output: `logs/resam/van_gogh/weight.safetensors`
- **[`preview_samples.py`](file:///Users/hainguyen/Repo/2026/ConceptErasure/Working/ReSam/remote_scripts/eval_ReSAM/erase_van_gogh/preview_samples.py)**: Qualitative image generator.
  - Generates 10 Van Gogh comparison images across 10 diverse artistic prompts.
  - Generates 10 retain comparison images across 10 randomly sampled artists from `data/style.csv` (seeded for reproducibility).
  - Saves individual `original`, `edit`, and side-by-side combined (`original | edit`) images.
- **[`run_erase_van_gogh.sh`](file:///Users/hainguyen/Repo/2026/ConceptErasure/Working/ReSam/remote_scripts/eval_ReSAM/erase_van_gogh/run_erase_van_gogh.sh)**: Master bash runner that executes training followed by preview sampling.

---

## 2. Usage

### Run on Local/Remote GPU
```bash
# Using default GPU 0 and python3:
bash remote_scripts/eval_ReSAM/erase_van_gogh/run_erase_van_gogh.sh

# Or specifying explicit GPU and Python virtual environment:
GPU_ID=1 PYTHON_BIN=/venv/main/bin/python bash remote_scripts/eval_ReSAM/erase_van_gogh/run_erase_van_gogh.sh
```

---

## 3. Output Structure

All outputs are saved to `logs/resam/van_gogh/`:

```
logs/resam/van_gogh/
├── weight.safetensors          # Exported partial U-Net weights
├── resam_optimization.pt       # Training telemetry, final scores, active candidates
├── metrics.jsonl               # Per-step loss, gradient norms, active weights
└── preview_samples/
    ├── preview_manifest.json   # Full manifest of prompts, seeds, and image filenames
    ├── van_gogh/
    │   ├── original/           # Unedited SD 1.4 images
    │   ├── edit/               # ReSAM edited images
    │   └── combined/           # Side-by-side [ Original | Edited ] JPGs
    └── retain/
        ├── original/           # Unedited SD 1.4 retain artist images
        ├── edit/               # ReSAM preserved retain artist images
        └── combined/           # Side-by-side [ Original | Preserved ] JPGs
```

