# TGPRS Style Retain-Rank Threshold Sweep

Retain-low ranks for the target-global pairwise residual subspace (TGPRS)
configuration used by `remote_scripts/eval_few/eval_few_style_1`. Each value is
the retain-low rank out of 768.

The diagnostic used Stable Diffusion v1.4's CLIP text encoder and reproduced the
training path: it removed the target artist from the unique concepts in
`data/style.csv`, encoded the remaining 1,733 retain concepts using their final
subject-token embeddings, and counted the singular values below each threshold
in the resulting uncentered second-moment matrix.

| Threshold | Van Gogh | Picasso | Monet |
|---:|---:|---:|---:|
| 0.10 | 140 | 140 | 140 |
| 0.15 | 217 | 217 | 217 |
| 0.20 | 277 | 277 | 277 |
| 0.25 | 326 | 326 | 326 |
| 0.30 | 368 | 369 | 368 |
| 0.35 | 402 | 402 | 402 |
| 0.40 | 432 | 432 | 432 |
| 0.45 | 456 | 456 | 456 |
| 0.50 | 480 | 480 | 480 |

The configured threshold is `0.3`. Its ranks match the completed TGPRS training
logs for all three tasks. These ranks depend on the retain embeddings and the
threshold; TGPRS residual rank and the subspace-anchor list do not enter this
calculation.
