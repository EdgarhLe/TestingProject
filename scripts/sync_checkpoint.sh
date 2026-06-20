#!/bin/bash
# scripts/sync_checkpoint.sh
# Đồng bộ checkpoint mới nhất từ Máy 1 → Máy 2
# Chạy trên Máy 1 sau mỗi curriculum transition (cuối Tuần 5, 6, 7, 8)
#
# Cách dùng: bash scripts/sync_checkpoint.sh

set -e

# Load biến môi trường
source .env 2>/dev/null || true

CHECKPOINT_DIR="${CHECKPOINT_DIR:-training/checkpoints}"
MACHINE2_HOST="${MACHINE2_HOST}"
REMOTE_PATH="${MACHINE2_HOST}:/path/to/checkpoints"

if [ -z "$MACHINE2_HOST" ]; then
  echo "LỖI: MACHINE2_HOST chưa được thiết lập trong .env"
  exit 1
fi

echo "Bắt đầu đồng bộ checkpoint..."
echo "Nguồn:     $CHECKPOINT_DIR"
echo "Đích:      $REMOTE_PATH"
echo ""

rsync -avz --progress \
  --exclude="*.tmp" \
  "$CHECKPOINT_DIR/" \
  "$REMOTE_PATH/"

echo ""
echo "Hoàn thành. Checkpoint đã được đồng bộ sang Máy 2."
