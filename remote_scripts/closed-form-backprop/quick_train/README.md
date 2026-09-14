# Chạy closed-form backprop trên server

Các lệnh bên dưới giả định server dùng user `root`, code nằm tại
`/workspace/CEdit`, và Python nằm tại `/venv/main/bin/python`.

## 1. Upload code

Chạy từ thư mục gốc `CEdit` trên máy local:

```bash
export VAST_HOST='182.224.239.168'
export VAST_PORT='54685'
export REMOTE_DIR='/workspace/CEdit'

rsync -az --progress \
  --exclude '.git/' \
  --exclude 'logs/' \
  --exclude 'exp_results/' \
  --exclude '__pycache__/' \
  -e "ssh -p ${VAST_PORT}" \
  ./ "root@${VAST_HOST}:${REMOTE_DIR}/"
```

## 2. Chạy preview

Kết nối vào server và mở `tmux`:

```bash
ssh -p "${VAST_PORT}" -t "root@${VAST_HOST}" \
  "cd '${REMOTE_DIR}' && tmux new-session -A -s closed-form"
```

Trong `tmux`, chạy:

```bash
bash remote_scripts/closed-form-backprop/quick_train/run_snoopy_preview.sh
```

Script tự cài package, sau đó đọc toàn bộ thông số train từ
`remote_scripts/closed-form-backprop/quick_train/config.yaml`. File `.sh` không override config. Có thể
truyền `GPU_ID`, `PYTHON_BIN` hoặc `CONFIG_PATH` nếu cần đổi môi trường/file YAML.

Nhấn `Ctrl-b`, rồi `d` để thoát `tmux` mà không dừng job. Kết nối lại bằng
lệnh SSH ở trên.

## 3. Download ảnh

Chạy trên máy local sau khi job hoàn tất:

```bash
mkdir -p exp_results/closed_form_backprop/train_preview_snoopy

rsync -az --progress \
  -e "ssh -p ${VAST_PORT}" \
  "root@${VAST_HOST}:${REMOTE_DIR}/logs/closed_form_backprop/snoopy_preview/samples/" \
  exp_results/closed_form_backprop/train_preview_snoopy3/
```

Ảnh so sánh `original/edit` nằm trong các thư mục `combine/` vừa tải về.
