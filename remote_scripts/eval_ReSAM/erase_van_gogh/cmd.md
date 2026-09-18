# Useful Commands for Van Gogh Style Erasure Workflow

## 1. Sync Local Code to Remote Server (Uploading Code Only)

Uploads changes from your local repository to the remote Vast.ai server while excluding large artifacts (weights, checkpoints, images, caches, logs).

> **Note:** Run this command from the repository root (`/Users/hainguyen/Repo/2026/ConceptErasure/Working/ReSam`).

```bash
rsync -avz --progress -e 'ssh -p 42312' \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'logs/' \
  --exclude 'result*/' \
  --exclude '*.png' \
  --exclude '*.jpg' \
  --exclude '*.jpeg' \
  --exclude '*.safetensors' \
  --exclude '*.pt' \
  --exclude '*.bin' \
  --exclude '.pytest_cache/' \
  ./ root@202.122.49.242:/workspace/ReSam/
```

*(Tip: Add `-n` or `--dry-run` to preview transferred files without writing anything).*

---

## 2. Stream Download Preview Images (Excluding Model Weights)

Downloads generated preview images directly from the remote Vast.ai server into a local target folder using an in-memory `tar` stream over SSH. Excludes all weight and checkpoint formats (`*.safetensors`, `*.pt`, `*.ckpt`, `*.bin`).

```bash
FOLDER="result_k5_50s_100r"  
OUT_DIR="remote_scripts/eval_ReSAM/erase_van_gogh/${FOLDER}"

mkdir -p "${OUT_DIR}" && \
ssh -p 42312 root@202.122.49.242 "tar -C /workspace/ReSam/logs/resam/van_gogh/preview_samples --exclude='*.safetensors' --exclude='*.pt' --exclude='*.ckpt' --exclude='*.bin' -cf - ." | \
tar -C "${OUT_DIR}" -xf -
```

### Why use this:
- **Fastest transfer:** Bundles files on-the-fly and streams them over a single TCP connection, eliminating per-file SSH handshake latency.
- **Weight safety:** The `--exclude` flags ensure no multi-hundred-megabyte `.safetensors` or `.pt` weight checkpoints are downloaded.
- **Zero temporary disk space:** Does not generate temporary `.zip` or `.tar` archive files on either the remote server or local machine.

---

## 3. Alternative: Rsync Image-Only Download Filter

If you prefer `rsync` with explicit file inclusion rules (only `.png`, `.jpg`, `.jpeg`, and `.json` manifests):

```bash
FOLDER="result_k1_50s"  # <-- Change destination folder name here
OUT_DIR="remote_scripts/eval_ReSAM/erase_van_gogh/${FOLDER}"

mkdir -p "${OUT_DIR}" && \
rsync -avz --progress -e 'ssh -p 42312' \
  --include='*/' \
  --include='*.png' \
  --include='*.jpg' \
  --include='*.jpeg' \
  --include='*.json' \
  --exclude='*' \
  root@202.122.49.242:/workspace/ReSam/logs/resam/van_gogh/preview_samples/ \
  "${OUT_DIR}/"
```
