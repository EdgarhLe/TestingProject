#!/bin/bash
set -e

if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo "⚠️  Không tìm thấy .env — dùng giá trị mặc định."
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-training/checkpoints/deploy}"
MACHINE2_HOST="${MACHINE2_HOST}"
REMOTE_DIR="${REMOTE_DIR:-/c/Users/username/checkpoints}"

if [ -z "$MACHINE2_HOST" ]; then
  echo "❌ LỖI: MACHINE2_HOST chưa được thiết lập trong .env"
  exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "❌ LỖI: Không tìm thấy thư mục checkpoint local: $CHECKPOINT_DIR"
  exit 1
fi

REMOTE_URI="${MACHINE2_HOST}:${REMOTE_DIR}"

echo "🚀 Bắt đầu đồng bộ checkpoint..."
echo "   Nguồn:  $CHECKPOINT_DIR"
echo "   Đích:   $REMOTE_URI"
echo ""

rsync -avz --progress \
  --exclude="*.tmp" \
  --no-perms --no-owner --no-group \
  "$CHECKPOINT_DIR/" \
  "$REMOTE_URI/"

echo ""
echo "✅ Hoàn thành. Checkpoint đã được đồng bộ sang Máy 2."