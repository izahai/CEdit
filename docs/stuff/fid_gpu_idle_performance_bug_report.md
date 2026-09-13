# FID GPU-Idle Performance Bug Report

## Summary

The full artistic-style TGPRS evaluation appeared to fall back to CPU even
though it was launched with `--device cuda` on an NVIDIA GeForce RTX 5090. GPU
memory remained allocated, but GPU utilization stayed at 0% for long periods
while the evaluator consumed nearly all of its CPU quota. Each 2048-dimensional
FID calculation took approximately seven minutes.

The job had not silently moved all metric computation to CPU. CLIP scoring and
Inception feature extraction were using CUDA. The bottleneck was the final FID
statistic in `torch-fidelity 0.3.0`, which copied features to NumPy and used
SciPy's CPU-only matrix square root.

The affected workflow was stopped after 19 of 36 metric rows had been persisted.
Its evaluator was changed to calculate the FID covariance term with float64
Torch linear algebra on CUDA, then the workflow resumed from its cached rows and
completed successfully.

Status: resolved for both artistic-style evaluators:

- `remote_scripts/eval_few/eval_few_style_1_tgprs/evaluate_clip_fid.py`;
- `remote_scripts/eval_few/eval_few_style_1/evaluate_clip_fid.py`.

## Impact

The workflow evaluates three erasure tasks (`van_gogh`, `picasso`, and `monet`),
two model states, five artistic-style content sets, and one MS-COCO set. This
produces 36 detailed metric rows, including 18 edited-image FID calculations at
the standard 2048-dimensional Inception feature layer.

Observed effects before the fix:

- GPU utilization remained at 0% during the dominant part of each FID row.
- The evaluator retained approximately 3.4 GiB of GPU memory, which made the
  process look CUDA-active even when no kernels were running.
- CPU utilization was approximately 721% against a cgroup quota of 7.68 cores.
- The process had 179 threads, with dozens runnable during the SciPy operation.
- One FID row took approximately 7 minutes and 7 seconds.
- The first six edited FID rows took approximately 44 minutes.

The SSH tunnel on local port 8080 was unrelated to evaluation performance.

## Environment

The incident was reproduced on the Vast.ai instance reached through
`114.34.26.236:41908` on 2026-09-06 UTC.

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA build | 12.8 |
| CUDA available | Yes; one visible RTX 5090 |
| torch-fidelity | 0.3.0 |
| SciPy | 1.17.1 |
| CPU quota | 7.68 cores |
| FID feature layer | 2048 |
| FID batch size | 32 |

## Detection and Reproduction

The live evaluator was sampled 30 times at 0.5-second intervals. The diagnostic
was considered red when the evaluator had a CUDA allocation but no observed GPU
work throughout the sampling window.

```text
pid=4678 gpu_max_pct=0 gpu_avg_pct=0.0 gpu_memory_mib=3440 cpu_pct=721
RED: no observed CUDA work in this stage
```

This reproduced the reported symptom while also showing that a CUDA context was
still present. Repeated one-shot `nvidia-smi` and `dmon` checks showed the same
idle-GPU, saturated-CPU state.

The evaluator itself remained runnable (`STAT=Rl+`) and continued producing rows,
so this was a performance bottleneck rather than a deadlock. The repeated
Hugging Face tokenizer messages were fork-safety warnings and were not the cause.

## Root Cause

The workflow correctly passed `cuda=True` to `torch_fidelity.calculate_metrics`.
In `torch-fidelity 0.3.0`, that flag places the Inception feature extractor and
input batches on CUDA. It does not place the final FID statistic on CUDA.

The installed implementation performed these steps:

1. Extract Inception features with Torch.
2. Copy the feature tensors to CPU and convert them with `features.numpy()`.
3. Calculate means and covariance matrices with NumPy.
4. Evaluate `scipy.linalg.sqrtm(sigma1.dot(sigma2))` on CPU.

At feature layer 2048, the final step operates on a 2048 by 2048 covariance
product and has cubic computational cost. SciPy saturated the container's CPU
quota while the GPU waited. Thread oversubscription further increased overhead:
the process exposed many more runnable threads than its 7.68-core quota could
execute concurrently.

The confirmed hypothesis was therefore: CUDA feature extraction was working,
but the dominant FID covariance matrix operation was CPU-only by design in the
third-party implementation.

## Fix

The evaluator now temporarily replaces `torch-fidelity`'s final
`fid_statistics_to_metric` callback when `use_cuda` is true. Feature extraction,
reference caching, and all existing input validation remain unchanged.

The new `frechet_distance_from_statistics` function:

1. Transfers both means and covariance matrices to the selected Torch device.
2. Uses float64 to preserve compatibility with SciPy's numerical precision.
3. Symmetrizes both covariance matrices.
4. Eigendecomposes the first covariance matrix.
5. Constructs the symmetric equivalent of the covariance product.
6. Uses `torch.linalg.eigvalsh` to calculate the covariance square-root trace.
7. Clamps small negative eigenvalues caused by floating-point error.
8. Rejects a non-finite final FID value.

The original `torch-fidelity` callback is restored in a `finally` block. CPU
evaluation still uses the unmodified SciPy implementation.

Relevant implementation:

- `remote_scripts/eval_few/eval_few_style_1_tgprs/evaluate_clip_fid.py`
- `remote_scripts/eval_few/eval_few_style_1/evaluate_clip_fid.py`
- `tests/test_eval_few_style_tgprs_workflow.py`
- `tests/test_eval_few_style_workflow.py`

## Validation

### Numerical equivalence

A deterministic unit test compares the Torch formulation with
`scipy.linalg.sqrtm` on seeded covariance statistics. The implementations agree
to nine decimal places on that fixture.

A real completed production case was replayed through the new path:

| Implementation | FID |
|---|---:|
| Existing SciPy result | 98.556043607003 |
| New Torch CUDA result | 98.556100727093 |
| Absolute difference | 0.000057120090 |

The absolute difference was approximately 0.000058% of the FID value and is
below the precision material to the generated reports.

### Performance

The 2048-dimensional covariance calculation was benchmarked using two real
cached reference statistics:

| Measurement | Before | After |
|---|---:|---:|
| Covariance/FID row bottleneck | about 7 minutes | 0.43 seconds |
| Real FID replay, including feature extraction | about 7 minutes | 5.96 seconds |

After restart, a 60-sample live utilization check produced:

```text
pid=8839 gpu_max_pct=61 gpu_avg_pct=7.4 gpu_memory_mib=3508
GREEN: CUDA work observed
```

Average utilization remains modest because image decoding, preprocessing, cache
access, and orchestration still run on CPU, and the CUDA linear-algebra stage is
short and bursty. The important change is that the former multi-minute matrix
operation now runs on the GPU.

### Tests

Two focused regression tests pass:

- Torch FID matches the SciPy reference calculation.
- The CUDA FID path routes final statistics through Torch and restores the
  original third-party callback afterward.

The complete test discovery ran 92 tests. It reported five existing
configuration failures and one existing missing-file error unrelated to this
change. One of those failures is a TGPRS workflow mismatch where the YAML uses
`retain_scale: 1.0` while its test expects `0.5`.

### Workflow completion

The original evaluator was terminated only after validating its exact PID and
command line. `detailed_metrics.csv` had 19 completed data rows at that point.
Because rows are written atomically and keyed by run and image fingerprints, the
new evaluator skipped those rows and resumed at row 20.

The resumed run advanced from 19 to all 36 rows in under three minutes and wrote:

- `detailed_metrics.csv` with 36 data rows;
- `summary.csv` with six data rows;
- `comparison.csv` with three data rows.

All final CSV metric values were checked for `NaN` and infinity. The result files
were checksum-verified after download to
`exp_results/few_style/tgprs_41908_metrics/`.

## Reproducibility Caveat

The completed `detailed_metrics.csv` contains mixed execution backends:

- the first 19 rows were calculated before the change;
- the remaining 17 rows were calculated after the CUDA change.

Both paths implement the same FID equation, and the production replay measured
an absolute difference of only `5.7e-05`. This is suitable for the current
analysis. If bitwise-uniform metric generation is required, rerun evaluation
with `FORCE_EVAL=1` after adopting one backend for every row.

## Remaining Risks and Follow-up Work

The immediate fix is intentionally local to the running artistic-style TGPRS
workflow. The repository contains other direct `torch-fidelity` call sites that
can exhibit the same CPU bottleneck:

- `src/clip_score_cal.py`
- `remote_scripts/eval_few/eval_few_ins_1/evaluate_clip_fid.py`
- `remote_scripts/eval_few/eval_few_ins_1_tgprs/evaluate_clip_fid.py`
- `remote_scripts/eval_paper_comparison_clip_fid/evaluate_mscoco_clip_fid.py`

Recommended follow-up:

1. Move the Torch FID statistic implementation into a shared module under
   `src/` and migrate every evaluator to it.
2. Record the FID statistic backend and implementation version in the runtime
   fingerprint and output metadata.
3. Add an optional cache key for edited-image statistics so interrupted rows do
   not repeat Inception feature extraction.
4. Replace the temporary third-party callback override with a shared explicit
   metric API before running metric calculations concurrently in one process.
5. Add a GPU integration benchmark with a generous numerical tolerance; keep the
   deterministic CPU equivalence test for ordinary unit-test environments.
6. Set `TOKENIZERS_PARALLELISM=false` before worker creation to remove misleading
   warning noise from evaluation logs.

The architectural lesson is that GPU selection should apply to the complete
metric pipeline, not only feature extraction. Device ownership and the metric
backend should be explicit at a shared interface rather than repeated across
workflow-specific evaluators.
