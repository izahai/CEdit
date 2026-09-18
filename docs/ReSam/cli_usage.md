# ReSAM: CLI Usage and YAML Configuration

This document defines the minimal, train-only YAML configuration and command-line interface (CLI) for **ReSAM** (Retain-Guided Sparse Anchor Mixture), based on [`ReSAM-idea.md`](file:///Users/hainguyen/Repo/2026/ConceptErasure/Working/ReSam/docs/ReSam/ReSAM-idea.md).

---

## 1. Minimal Training YAML Configuration

Save as `configs/resam_train.yaml`:

```yaml
# Model & Concepts
sd_ckpt: CompVis/stable-diffusion-v1-4
target_concepts:
  - Snoopy
candidate_concepts: data/candidates/snoopy.csv
retain_path: data/instance.csv
heads: concept

# ReSAM Sparse Mixture
resam_k: 2
resam_temperature: 1.0

# Loss Configuration
use_null_retain_loss: true   # Always include null-prompt preservation alongside retain prompts

# Training Hyperparameters
steps: 200
lr: 0.05
num_train_inference_steps: 50

# Output
save_path: logs/resam/snoopy_k2
file_name: weight
```

> **Note on Defaults:** Backend geometry options (`baseline: SPEED`, `params: V`, `retain_projection_rank: 1`, `seed: 0`, and straight-through estimator enabled) default automatically in code and are omitted from the minimal train config.

---

## 2. Candidate Concepts File Format

The `candidate_concepts` setting points to a CSV file following repository conventions:

**`data/candidates/snoopy.csv`**
```csv
id,concept
1,dog
2,cartoon dog
3,beagle
4,white puppy
5,comic hound
```

*(Plain `.txt` files with one concept per line are also accepted).*

---

## 3. CLI Usage

### A. Run with YAML Config
```bash
CUDA_VISIBLE_DEVICES=0 python train_resam.py \
    --config configs/resam_train.yaml
```

### B. Run with CLI Overrides (e.g. Ablating $k$ or Testing Another Bank)
```bash
CUDA_VISIBLE_DEVICES=0 python train_resam.py \
    --config configs/resam_train.yaml \
    --candidate_concepts data/candidates/snoopy_expanded.csv \
    --resam_k 3 \
    --save_path logs/resam/snoopy_k3
```

### C. Pure CLI One-Liner (No YAML File)
```bash
CUDA_VISIBLE_DEVICES=0 python train_resam.py \
    --target_concepts "Snoopy" \
    --candidate_concepts data/candidates/snoopy.csv \
    --retain_path data/instance.csv \
    --heads concept \
    --resam_k 2 \
    --resam_temperature 1.0 \
    --use_null_retain_loss \
    --steps 200 \
    --lr 0.05 \
    --save_path logs/resam/snoopy_k2
```

---

## 4. Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `sd_ckpt` | `str` | `CompVis/stable-diffusion-v1-4` | Base Stable Diffusion checkpoint. |
| `target_concepts` | `str` / `list` | *Required* | Concept(s) to erase (single target). |
| `candidate_concepts` | `str` (path) | *Required* | Path to CSV or text file containing candidate concept bank. |
| `retain_path` | `str` (path) | *Required* | CSV dataset used for retain preservation (e.g., `data/instance.csv`). |
| `heads` | `str` | `"concept"` | Column name in `retain_path` from which to extract retain concepts. |
| `resam_k` | `int` | `2` | Number of top candidates to activate in forward pass ($k$). |
| `resam_temperature` | `float` | `1.0` | Softmax temperature ($\tau$) controlling mixture sharpness. |
| `resam_ste` | `bool` | `True` | Straight-through estimator for top-$k$ gradient flow. |
| `use_null_retain_loss` | `bool` | `False` | Always include unconditional/null-prompt preservation in the loss: $\mathcal{L} = 0.5 \cdot (\mathcal{L}_{\text{retain}} + \mathcal{L}_{\text{null}})$. |
| `steps` | `int` | `200` | Number of score optimization steps. |
| `lr` | `float` | `0.05` | Learning rate for candidate score logits $\mathbf{s}$. |
| `num_train_inference_steps` | `int` | `50` | Number of discrete inference steps in the training diffusion trajectory schedule (alias: `train_inference_steps`). |
| `save_path` | `str` | *Required* | Destination directory for trained weights and logs. |
| `file_name` | `str` | `"weight"` | Base filename for the exported checkpoint (`.safetensors`). |
