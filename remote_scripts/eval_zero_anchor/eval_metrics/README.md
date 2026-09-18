# Quick CLIP Evaluation Suite: Art Anchor vs Zero Anchor

This directory provides a fast (~5s on GPU, ~15s on CPU) evaluation suite to quantitatively compare **SPEED with Art Anchor** against **SPEED with Zero Anchor** directly from the generated 3-way concatenated images (`outputs/concat_images/`).

---

## Metrics Evaluated

1. **Target Concept Erasure (↓ Lower is better)**:
   - **Concept CLIP Score**: Cosine similarity between image and text `"Van Gogh"`. Measures remaining style signal.
   - **Prompt CLIP Score**: Cosine similarity between image and full prompt text.
   - **Relative Erasure Rate (%)**: Percentage reduction in target concept similarity relative to Original SD 1.4.

2. **Retain Specificity & Preservation (↑ Higher is better)**:
   - **Artist CLIP Score**: Cosine similarity between image and the retain artist name (e.g. `"Monet"`).
   - **Prompt CLIP Score**: Cosine similarity between image and full retain prompt text.
   - **Preservation Ratio (%)**: Retention percentage relative to Original SD 1.4.

3. **Image Fidelity to Original (↑ Higher is better)**:
   - **Pairwise Image-to-Image CLIP Similarity**: Cosine similarity between edited image features and original base model features on the retain artist set ($\text{sim}(\text{Image}_{\text{edit}}, \text{Image}_{\text{orig}})$).
   - Measures how faithfully the model preserves original layout, composition, and visual quality when the target concept is absent.

---

## Execution

### Run Remotely (on CUDA GPU)

```bash
# Using the bash runner:
GPU_ID=0 bash remote_scripts/eval_zero_anchor/eval_metrics/run_eval_metrics.sh

# Or directly with python:
python3 remote_scripts/eval_zero_anchor/eval_metrics/compute_quick_metrics.py \
    --image_dir remote_scripts/eval_zero_anchor/outputs/concat_images \
    --output_dir remote_scripts/eval_zero_anchor/outputs/eval_metrics \
    --device cuda
```

### Run Locally (on Laptop CPU / MPS)

If you have downloaded the concatenated images to your laptop using the tar stream command in `eval_zero_anchor/README.md`:

```bash
python3 remote_scripts/eval_zero_anchor/eval_metrics/compute_quick_metrics.py \
    --image_dir remote_scripts/eval_zero_anchor/outputs/concat_images \
    --output_dir remote_scripts/eval_zero_anchor/outputs/eval_metrics \
    --device cpu
```

---

## Output Artifacts

All outputs are saved to `remote_scripts/eval_zero_anchor/outputs/eval_metrics/`:
* `summary_table.txt`: Clean formatted ASCII table showing side-by-side comparison and delta for each metric.
* `summary_metrics.json`: Full machine-readable aggregate dictionary with means, deltas, and percentage improvements.
* `per_image_metrics.csv`: Granular row-by-row scores for each of the 20 prompts (target and retain).

