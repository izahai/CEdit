# Vast AI commands

## SSH vào server và tạo tmux session mới

```bash
ssh -p 38431 -t root@182.224.239.168 \
  'tmux new-session -s cedit-negative-subspace'
```

Nếu session đã tồn tại, attach lại:

```bash
ssh -p 38431 -t root@182.224.239.168 \
  'tmux attach-session -t cedit-negative-subspace'
```

Hoặc dùng một lệnh tự động: attach nếu có, tạo mới nếu chưa có:

```bash
ssh -p 38410 -t root@182.224.239.168 \
  'tmux new-session -A -s view'
```

## Chuẩn bị thư mục làm việc

```bash
cd /workspace/CEdit
```

## Chạy workflow

```bash
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/00_clone_repositories.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/01_setup_environment.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/02_train.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/03_infer.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/04_setup_ce_eval.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/05_eval.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/06_summarize_results.sh
bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/07_bundle_results.sh
```

## Chạy training trong tmux và ghi log

Lệnh dưới đây tạo session `cedit-train`, chạy training và lưu cả stdout lẫn
stderr vào `train.log` trên server:

```bash
ssh -p 38410 -t root@182.224.239.168 \
  'tmux has-session -t cedit-train 2>/dev/null || \
   tmux new-session -d -s cedit-train \
   "cd /workspace/CEdit && bash remote_scripts/vastai_100_negative_target_normalized_residual_subspace/02_train.sh 2>&1 | tee -a remote_scripts/vastai_100_negative_target_normalized_residual_subspace/train.log"; \
   tmux attach-session -t cedit-train'
```

Xem log trực tiếp từ máy local:

```bash
ssh -p 38410 root@182.224.239.168 \
  'tail -f /workspace/CEdit/remote_scripts/vastai_100_negative_target_normalized_residual_subspace/train.log'
```

Nếu training đã chạy sẵn trong tmux, bật ghi log cho pane hiện tại bằng lệnh
này, không tạo session mới:

```bash
ssh -p 38410 root@182.224.239.168 \
  "tmux pipe-pane -t ssh_tmux:0.0 -o 'cat >> /workspace/train.log'"
```

Lệnh này chỉ ghi output phát sinh sau thời điểm bật pipe-pane; log cũ trong
terminal không được khôi phục.

Detach tmux mà không dừng process bằng `Ctrl-b`, sau đó nhấn `d`.

## Kiểm tra session và xem log

```bash
tmux ls
tmux attach -t cedit-negative-subspace
```

## Đồng bộ code từ máy local lên server

Chạy lệnh sau trên máy local, tại bất kỳ thư mục nào:

```bash
rsync -avz --progress \
  --exclude='.git/' \
  --exclude='CE-Eval/' \
  --exclude='__pycache__/' \
  --exclude='logs/' \
  --exclude='*.pt' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  -e "ssh -p 38431" \
  /Users/hainguyen/Repo/2026/ConceptErasure/Working/CEdit/ \
  root@182.224.239.168:/workspace/CEdit/
```

## GPU SM monitoring daemon

Start a background daemon that samples GPU utilization every second. The
`sm` column reports Streaming Multiprocessor utilization:

```bash
ssh -p 38410 root@182.224.239.168 \
  'nohup nvidia-smi dmon -i 0 -s u -d 1 -o T \
   >> /workspace/gpu_sm.log 2>&1 & echo $! > /workspace/gpu_sm.pid'
```

Follow the monitor log:

```bash
ssh -p 38410 root@182.224.239.168 \
  'tail -f /workspace/gpu_sm.log'
```

Stop the daemon:

```bash
ssh -p 38410 root@182.224.239.168 \
  'kill "$(cat /workspace/gpu_sm.pid)" 2>/dev/null || true'
```

## Download training log

Run this command from the local current directory:

```bash
scp -P 38410 \
  root@182.224.239.168:/workspace/train.log \
  .
```

Download the evaluation summary CSV to the current local directory:

```bash
scp -P 38410 \
  root@182.224.239.168:/workspace/cedit_ce_eval_outputs_100_celebrity_negative_target_normalized_residual_subspace/gcd/summary.csv \
  .
```

## Count generated files and images

Count all files under the workflow output directory:

```bash
ssh -p 38410 root@182.224.239.168 \
  'find /workspace/cedit_ce_eval_outputs_100_celebrity_negative_target_normalized_residual_subspace -type f | wc -l'
```

Count all generated PNG images:

```bash
ssh -p 38410 root@182.224.239.168 \
  'find /workspace/cedit_ce_eval_outputs_100_celebrity_negative_target_normalized_residual_subspace -type f -iname "*.png" | wc -l'
```

Show image counts for every rank and split:

```bash
ssh -p 38410 root@182.224.239.168 'for k in 10 20 30 40 50 60 70 80 90 100; do
  root=/workspace/cedit_ce_eval_outputs_100_celebrity_negative_target_normalized_residual_subspace/images/negative_target_normalized_residual_subspace/k_${k}/100_celebrity
  erase=$(find "$root/erase/edit" -maxdepth 1 -type f -iname "*.png" 2>/dev/null | wc -l)
  retain=$(find "$root/retain/edit" -maxdepth 1 -type f -iname "*.png" 2>/dev/null | wc -l)
  printf "k=%s erase=%s retain=%s total=%s\\n" "$k" "$erase" "$retain" "$((erase + retain))"
done'
```
