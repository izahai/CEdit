# Vast AI: 100 Celebrity Smallest-Cosine Subspace

This folder is the SSH-friendly version of
`notebooks/CEdit_CE_Eval_VastAI_100_Smallest_Cosine_Subspace.ipynb`. It runs only
`smallest_cosine_subspace` with `RESIDUAL_TOP_K=10`; it does not train or compare
against `legacy`.

Configuration is split by responsibility: `train.yaml` contains model-editing
parameters and concepts; `workflow.yaml` contains runtime, repository, output,
sampling, and evaluation settings. The numbered shell scripts contain only their
individual workflow steps.

Bootstrap the CEdit checkout once, then run every command from its root. The
first numbered script updates that checkout and clones CE-Eval as a sibling:

```bash
cd /workspace
git clone https://github.com/izahai/CEdit.git CEdit
cd CEdit
```

Then run the workflow:

```bash
bash remote_scripts/vastai_100_smallest_cosine/00_clone_repositories.sh
bash remote_scripts/vastai_100_smallest_cosine/01_setup_environment.sh
bash remote_scripts/vastai_100_smallest_cosine/02_train.sh
bash remote_scripts/vastai_100_smallest_cosine/03_infer.sh
bash remote_scripts/vastai_100_smallest_cosine/05_setup_ce_eval.sh
bash remote_scripts/vastai_100_smallest_cosine/06_eval.sh
bash remote_scripts/vastai_100_smallest_cosine/08_summarize_results.sh
bash remote_scripts/vastai_100_smallest_cosine/09_bundle_results.sh
```

The scripts are individually resumable. Train, sampling, and evaluation skip a
completed artifact by default. Set one of these environment variables to rerun:

```bash
FORCE_RETRAIN=1 bash remote_scripts/vastai_100_smallest_cosine/02_train.sh
FORCE_RESAMPLE=1 bash remote_scripts/vastai_100_smallest_cosine/03_infer.sh
FORCE_EVAL=1 bash remote_scripts/vastai_100_smallest_cosine/06_eval.sh
```

The training parameters and all 100 target concepts live in `train.yaml`. Edit
that file to change the method or hyperparameters. Edit `workflow.yaml` for GPU,
paths, CE-Eval, sampling, or repository settings. The train script invokes
`train_erase_null.py` with only the train YAML config.

Useful runtime overrides:

```bash
GPU_ID=0 bash remote_scripts/vastai_100_smallest_cosine/02_train.sh

TRAIN_CONFIG=/path/to/another-train.yaml \
  bash remote_scripts/vastai_100_smallest_cosine/02_train.sh

WORKFLOW_CONFIG=/path/to/another-workflow.yaml \
  bash remote_scripts/vastai_100_smallest_cosine/03_infer.sh
```

The `vastai/pytorch` image keeps PyTorch in `/venv/main/bin/python`, which is
the default interpreter for these scripts. Override it only if you intentionally
use a different Docker image:

```bash
PYTHON_BIN=/some/other/environment/bin/python \
  bash remote_scripts/vastai_100_smallest_cosine/01_setup_environment.sh
```

`OUTPUT_ROOT` defaults to a sibling of the CEdit checkout. `02_train.sh` exports
the derived `CHECKPOINT_DIR`, and `train.yaml` uses `${CHECKPOINT_DIR}` for its
`save_path`, so overriding `OUTPUT_ROOT` continues to work. Set `HF_TOKEN` in
the SSH session before training if Hugging Face gated-model access requires it.
