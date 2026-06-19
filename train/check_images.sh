#!/usr/bin/env bash
# check_images.sh - Parallel scan for corrupt/oversized training images on Lambda.
#
# Check-only (no deletes). Use before or after pull_data.sh to diagnose crashes
# during latent caching.
#
#   bash train/check_images.sh              # pilot dir (default)
#   MODE=full bash train/check_images.sh    # full dataset
#   MODE=full bash train/check_images.sh --fail   # exit 1 if any bad images
#
# Uses sd-venv Python when present (Pillow from sd-scripts deps).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${MODE:-pilot}"

if [ "$MODE" = "pilot" ]; then
    TRAIN_DATA="${TRAIN_DATA:-/home/ubuntu/train_data_pilot}"
elif [ "$MODE" = "full" ]; then
    TRAIN_DATA="${TRAIN_DATA:-/home/ubuntu/train_data}"
else
    echo "ERROR: MODE must be 'pilot' or 'full' (got '$MODE')" >&2
    exit 1
fi

PYTHON="${PYTHON:-python3}"
if [ -x "${VENV:-$HOME/sd-venv}/bin/python" ]; then
    PYTHON="${VENV:-$HOME/sd-venv}/bin/python"
fi

MAX_LONG_SIDE="${MAX_LONG_SIDE:-2048}"
WORKERS="${WORKERS:-$(nproc)}"

if [ ! -d "$TRAIN_DATA" ]; then
    echo "ERROR: TRAIN_DATA not found: $TRAIN_DATA — run pull_data.sh first" >&2
    exit 1
fi

echo "==> Checking images in $TRAIN_DATA (MODE=$MODE, workers=$WORKERS)"
exec "$PYTHON" "$REPO_DIR/train/check_images.py" "$TRAIN_DATA" \
    --max-long-side "$MAX_LONG_SIDE" \
    --workers "$WORKERS" \
    "$@"
