# 100-style SPEED comparison

This workflow compares one legacy SPEED edit with one target-global pairwise residual subspace edit of the same 100 styles against shared SD v1.4 images. The current full profile produces 200 erased-style, 200 retained-style, and 100 MS-COCO images per model: 1,500 images total. Each artist has two prompts with seed 0: the generic prompt plus one of portrait, landscape, city street, or still life, rotated evenly across artists. Monet (local ID 1266) is the artist called **Claude Monet** in the [MACE style benchmark](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Lu_MACE_Mass_Concept_CVPR_2024_supplemental.pdf). Caravaggio (ID 437) is excluded from the erase group. The 100 evaluation retain artists are included in the 1,634-style training retain set.

The smoke profile edits three erase artists with both methods, samples three erase and three retain artists, and evaluates eight COCO prompts. It retains the full training retain CSV. Its TGPRS rank is capped at three because only three targets are present. Its FID uses 64 features; the current full profile reports 2048-feature FID-100.

Training settings are in separate [legacy](train_config_legacy.yaml) and [TGPRS](train_config_target_global_pairwise_residual_subspace.yaml) YAML files. Both workflow profiles reference these files. The run saves a resolved training YAML beside each checkpoint, including the selected targets and retain CSV.
The current files use `aug_num: 10` for legacy and `aug_num: 0` for TGPRS, so the comparison reflects both the residual method and augmentation setting.

## Fresh Vast PyTorch instance

Upload the repository from your local computer, then run these commands on the instance. The instance needs a CUDA-ready PyTorch image with `/venv/main/bin/python`, enough GPU memory for SD v1.4 and CLIP, and disk space for 1,500 PNGs.

```bash
# Local machine: set VAST_HOST to the instance SSH address, including user and port as needed.
rsync -az --exclude logs --exclude .git ./ "$VAST_HOST:/workspace/CEdit/"

# Vast instance
cd /workspace/CEdit
PYTHON_BIN=/venv/main/bin/python GPU_ID=0 OUTPUT_ROOT=/workspace/eval_100_style \
  WORKFLOW_CONFIG=remote_scripts/eval_100_style/workflow_smoke.yaml \
  bash remote_scripts/eval_100_style/run_all.sh
PYTHON_BIN=/venv/main/bin/python GPU_ID=0 OUTPUT_ROOT=/workspace/eval_100_style \
  bash remote_scripts/eval_100_style/run_all.sh
```

`setup` checks CUDA and installs `requirements.txt` plus SciPy. Stable Diffusion and CLIP download through Diffusers and Transformers on first use. If a model is gated, authenticate with Hugging Face in the instance before running.

Resume after interruption with the same config and output root. Completed checkpoints require matching configuration and checkpoint hash. Sampling checks the manifest identity and each expected filename; a partial image is generated again. The workflow hashes the config, source CSVs, and implementation into a run directory under `OUTPUT_ROOT/runs/`, so changed inputs create a fresh run instead of silently reusing old results. Set `START_STAGE=train`, `sample`, or `evaluate` to skip preceding stages; `prepare` must have run once for that run ID. All stage output appears in the terminal and is appended to `OUTPUT_ROOT/workflow.log`; sampling displays a live progress bar per model.

```bash
START_STAGE=sample OUTPUT_ROOT=/workspace/eval_100_style \
  bash remote_scripts/eval_100_style/run_all.sh

# Local machine: download every run, including checkpoints, logs, images, and metrics.
rsync -az "$VAST_HOST:/workspace/eval_100_style/" ./eval_100_style_results/
```

Each run contains `manifest.csv`, resolved settings, checkpoint configs and logs, `images/{original,legacy,target_global_pairwise_residual_subspace}/`, and `metrics/{image_clip,per_artist,comparison}.csv`. CLIP scores are image–prompt cosine similarities scaled by 100. CLIP-e and CLIP-s macro-average erased and retained artists; Hₐ is CLIP-s minus CLIP-e. Artist CIs resample images within an artist; aggregate CIs resample artists. COCO CLIP is the image mean. FID compares each edited COCO set to the shared original set. The current full run has 100 COCO prompts and 2048-feature FID; smoke results are validation only.
