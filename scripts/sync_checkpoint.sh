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

# --- Timeout settings (configurable via .env or defaults) ---
# SSH connection timeout in seconds (default: 10)
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-10}"
# Maximum total runtime for rsync in seconds (default: 600 = 10 minutes)
RSYNC_MAX_TIME="${RSYNC_MAX_TIME:-600}"
# ------------------------------------------------------------

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
echo "   Nguồn:       $CHECKPOINT_DIR"
echo "   Đích:        $REMOTE_URI"
echo "   Timeout SSH: ${SSH_CONNECT_TIMEOUT}s"
echo "   Timeout rsync: ${RSYNC_MAX_TIME}s"
echo ""

# Build SSH options: set connection timeout, keep‑alive, and max retries
SSH_OPTS="-o ConnectTimeout=${SSH_CONNECT_TIMEOUT} -o ServerAliveInterval=5 -o ServerAliveCountMax=2"

# Run rsync with:
#   -e "ssh ..."     → SSH connection timeout
#   timeout          → overall kill‑switch for slow/frozen transfers
timeout "${RSYNC_MAX_TIME}" rsync -avz --progress \
  -e "ssh $SSH_OPTS" \
  --exclude="*.tmp" \
  --no-perms --no-owner --no-group \
  "$CHECKPOINT_DIR/" \
  "$REMOTE_URI/"

echo ""
echo "✅ Hoàn thành. Checkpoint đã được đồng bộ sang Máy 2."