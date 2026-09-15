# MOV2: Real-CLIP common-anchor visualization

MOV2 is a motivation-only visualization. It does not train an edited model,
sample diffusion images, or run GCD evaluation.

It encodes the 100 raw celebrity names from `data/100_celebrity.csv` and the
single raw anchor prompt `person` using the Stable Diffusion v1.4 CLIP text
encoder. The embedding for each prompt is the final non-special subject token,
matching the concept embedding seam used by `train_erase_null.py`.

The figure contains:

- a two-dimensional PCA visualization of the target embeddings and their
  common-anchor residual arrows;
- the complete residual singular-value spectra for legacy common-anchor
  residuals and the rank-30 TGPRS residuals.

PCA is used only for visualization. All reported residual and edit-statistic
ranks are calculated in the original 768-dimensional CLIP space.

## Run on Vast AI

```bash
cd /workspace/CEdit
PYTHON_BIN=/venv/main/bin/python bash motivation/mov2/run.sh
```

The model is downloaded automatically on first use. Expected outputs:

```text
motivation/mov2/outputs/
├── common_anchor_clip_geometry.pdf
├── common_anchor_clip_geometry.png
├── embedding_projection.csv
├── residual_spectrum.csv
└── summary.json
```

The PDF is vector output and the PNG is rendered at 600 DPI.

## Re-render locally from cached embeddings

This does not reload CLIP or require a GPU:

```bash
uv run --with matplotlib python motivation/mov2/render_cached.py
```
