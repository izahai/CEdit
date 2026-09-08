# Few-style erasure evaluation: updated TGPRS result

This report compares the new TGPRS run from server port `41908` with the SPEED
and earlier TGPRS results preserved in [`../full_scale/`](../full_scale/). The
aggregate values come from [`summary.csv`](summary.csv) and
[`../full_scale/0.3_summary_57595.csv`](../full_scale/0.3_summary_57595.csv).
Per-content values and run fingerprints are available in
[`detailed_metrics.csv`](detailed_metrics.csv) and
[`../full_scale/0.3_details_57595.csv`](../full_scale/0.3_details_57595.csv).

> SPEED uses threshold `0.1`, `aug_num=10`, retain scale `1.0`, and anchor
> `art`. Both TGPRS runs use threshold `0.3`, `aug_num=0`, anchor `art`, and 30
> artist-neutral subspace anchors with configured and applied rank 30. The old
> TGPRS run uses `retain_scale=0.5`; the new run uses `retain_scale=1.0`.
> Sampling uses seed 0, 20 denoising steps, CFG 7.5, 300 images per artist
> metric, and 1,000 images per MS-COCO metric.

Lower target CLIP score and a more negative change from the corresponding
unedited reference indicate stronger erasure. Lower non-target and MS-COCO FID
and higher MS-COCO CLIP score indicate better preservation. Target FID measures
distribution shift from the unedited target images, but is not ranked because a
larger shift does not by itself prove successful concept removal. Bold values
are the best result in each directional column among the three edited models.

## Executive summary

Increasing TGPRS retain scale from `0.5` to `1.0` shifts the method strongly
toward preservation:

- Mean non-target FID falls to **24.64**, compared with 30.81 for SPEED and
  33.27 for TGPRS at retain scale 0.5. This is a 20.0% improvement over SPEED
  and a 25.9% improvement over the earlier TGPRS run.
- Mean MS-COCO FID falls to **17.65**, compared with 20.49 for SPEED and 22.14
  for TGPRS at retain scale 0.5. This is a 13.9% improvement over SPEED and a
  20.3% improvement over the earlier TGPRS run.
- Mean MS-COCO CLIP score rises to **26.523**, slightly above 26.475 for SPEED
  and 26.504 for the earlier TGPRS run.

The preservation gain comes with substantially weaker erasure. The mean target
CLIP reduction from each run's own unedited reference is only **0.46 points**
for the new TGPRS result, versus 2.63 for SPEED and 2.20 for TGPRS at retain
scale 0.5. For Van Gogh, the new target CLIP score increases by 0.38 rather than
decreasing, so that task has no CLIP-score evidence of successful erasure.

## Cross-task average

| Method | Target CS ↓ | Δ target CS vs original ↓ | Target FID | Non-target FID ↓ | MS-COCO CS ↑ | MS-COCO FID ↓ |
|---|---:|---:|---:|---:|---:|---:|
| SPEED | **25.89** | **-2.63** | 127.15 | 30.81 | 26.475 | 20.49 |
| TGPRS, retain 0.5 | 26.33 | -2.20 | 133.71 | 33.27 | 26.504 | 22.14 |
| TGPRS, retain 1.0 (new) | 28.06 | -0.46 | 93.69 | **24.64** | **26.523** | **17.65** |

## Reference consistency across runs

The full-scale and new evaluations have different run fingerprints and image
manifest fingerprints. Their aggregate unedited CLIP scores are nevertheless
very close. Using each edited model's own unedited reference avoids attributing
this small cross-run drift to erasure.

| Task target | Old original target CS | New original target CS | Difference | Old mean non-target CS | New mean non-target CS | Difference |
|---|---:|---:|---:|---:|---:|---:|
| Van Gogh | 28.715572 | 28.715358 | -0.000214 | 28.236138 | 28.236075 | -0.000063 |
| Picasso | 27.957051 | 27.957051 | 0.000000 | 28.425768 | 28.425652 | -0.000116 |
| Monet | 28.906181 | 28.906181 | +0.000000 | 28.188486 | 28.188370 | -0.000116 |

The new unedited MS-COCO CLIP score is 26.527796, only 0.000601 above the old
reference value of 26.527196.

## Erase Van Gogh

| Method | Target CS ↓ | Δ target CS vs original ↓ | Target FID | Non-target FID ↓ | MS-COCO CS ↑ | MS-COCO FID ↓ |
|---|---:|---:|---:|---:|---:|---:|
| SPEED | **26.42** | **-2.29** | 130.23 | 30.20 | 26.529 | 20.79 |
| TGPRS, retain 0.5 | 26.64 | -2.08 | 149.29 | 36.51 | 26.542 | 21.88 |
| TGPRS, retain 1.0 (new) | 29.09 | +0.38 | 98.56 | **27.23** | **26.563** | **17.52** |

The new TGPRS setting gives the best preservation on all three preservation
columns. Relative to SPEED, it lowers mean non-target FID by 2.97 and MS-COCO
FID by 3.27 while raising MS-COCO CLIP by 0.034. However, its target CLIP score
is 2.67 points higher than SPEED and 0.38 above its own original reference.
This is the clearest case where retain scale 1.0 over-preserves the target.

## Erase Picasso

| Method | Target CS ↓ | Δ target CS vs original ↓ | Target FID | Non-target FID ↓ | MS-COCO CS ↑ | MS-COCO FID ↓ |
|---|---:|---:|---:|---:|---:|---:|
| SPEED | **26.26** | **-1.70** | 116.49 | 25.44 | 26.471 | 19.79 |
| TGPRS, retain 0.5 | 26.99 | -0.96 | 113.92 | 25.03 | **26.491** | 21.72 |
| TGPRS, retain 1.0 (new) | 27.44 | -0.52 | 90.62 | **18.02** | 26.486 | **17.09** |

The new TGPRS setting still lowers target CLIP relative to its original, but
the reduction is only 0.52 points, compared with 1.70 for SPEED. Preservation
improves substantially: non-target FID is 7.42 lower and MS-COCO FID is 2.69
lower than SPEED. The earlier TGPRS run retains the highest MS-COCO CLIP score
by a small 0.005 margin over the new run.

## Erase Monet

| Method | Target CS ↓ | Δ target CS vs original ↓ | Target FID | Non-target FID ↓ | MS-COCO CS ↑ | MS-COCO FID ↓ |
|---|---:|---:|---:|---:|---:|---:|
| SPEED | **24.99** | **-3.91** | 134.73 | 36.78 | 26.425 | 20.89 |
| TGPRS, retain 0.5 | 25.34 | -3.56 | 137.92 | 38.27 | 26.480 | 22.82 |
| TGPRS, retain 1.0 (new) | 27.66 | -1.24 | 91.89 | **28.68** | **26.520** | **18.34** |

The new TGPRS run preserves non-target and MS-COCO distributions best, reducing
non-target FID by 8.10 and MS-COCO FID by 2.56 relative to SPEED. It still
reduces target CLIP by 1.24 points, but SPEED produces a much larger 3.91-point
reduction.

## Effect of increasing TGPRS retain scale

The comparison between the two TGPRS runs isolates the main configuration
change, although it remains a cross-run rather than byte-identical paired
experiment.

| Task | Target CS change, 1.0 − 0.5 | Non-target FID change | MS-COCO CS change | MS-COCO FID change |
|---|---:|---:|---:|---:|
| Van Gogh | +2.45 | -9.28 | +0.022 | -4.36 |
| Picasso | +0.45 | -7.01 | -0.005 | -4.63 |
| Monet | +2.32 | -9.59 | +0.039 | -4.48 |

Positive target CLIP changes indicate weaker erasure; negative FID changes
indicate better preservation. Retain scale 1.0 improves both preservation FID
measures in every task, but raises target CLIP in every task. The effect is not
a free improvement: it moves the operating point along the erasure-preservation
tradeoff.

## Conclusion

The new TGPRS setting is the strongest preservation configuration in these
results, winning mean non-target FID and MS-COCO FID for all three erase tasks.
It is not the strongest erasure configuration. SPEED has the lowest target CLIP
score in every task, and TGPRS at retain scale 0.5 also erases more strongly than
the new retain scale 1.0 setting.

For a preservation-first deployment, retain scale 1.0 is preferable. For the
current concept-erasure objective, it is too conservative for Van Gogh and may
also be too conservative for Picasso. A retain-scale sweep between 0.5 and 1.0,
evaluated with multiple seeds, is the appropriate next experiment to find a
better Pareto point.

## Limitations and reproducibility notes

- All results use one seed, so the tables do not estimate sampling variance or
  statistical significance.
- SPEED and TGPRS at retain scale 0.5 come from server port `57595`; the new
  TGPRS result comes from server port `41908`. Comparisons therefore use each
  run's own original reference.
- The new result was resumed after a FID performance fix. Its first 19 detailed
  rows use SciPy's CPU matrix square root and its remaining 17 rows use the
  numerically equivalent Torch CUDA calculation. A real replay differed by
  only `5.7e-05` FID, but a forced single-backend rerun is required for bitwise
  uniformity.
- The visual package in [`../samples_57595/`](../samples_57595/) contains SPEED
  and TGPRS retain-scale-0.5 images only. It should not be used as visual
  evidence for the new retain-scale-1.0 result.
