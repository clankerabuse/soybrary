#!/usr/bin/env bash
# pull_data.sh - Phase 4: download the packaged dataset from R2 and extract the
# tar shards into a single flat directory that dataset.toml's image_dir expects.
#
#   bash train/pull_data.sh
#
# Requires:
#   - .env with R2 creds (R2_ACCOUNT_ID/R2_ENDPOINT, R2_ACCESS_KEY_ID,
#     R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME)
#   - r2_sync.py on the instance (copy the repo, or at least r2_sync.py + .env)
set -euo pipefail

# MODE picks pilot vs full defaults; must match how the data was uploaded and
# the image_dir baked into the packaged dataset.toml.
#   MODE=pilot bash train/pull_data.sh   # default
#   MODE=full  bash train/pull_data.sh
MODE="${MODE:-pilot}"
if [ "$MODE" = "pilot" ]; then
  R2_PREFIX="${R2_PREFIX:-datasets/soyjak-sdxl-pilot10k}"
  TRAIN_DATA="${TRAIN_DATA:-/home/ubuntu/train_data_pilot}"
elif [ "$MODE" = "full" ]; then
  R2_PREFIX="${R2_PREFIX:-datasets/soyjak-sdxl}"
  TRAIN_DATA="${TRAIN_DATA:-/home/ubuntu/train_data}"
else
  echo "ERROR: MODE must be 'pilot' or 'full' (got '$MODE')" >&2
  exit 1
fi
PKG_DIR="${PKG_DIR:-$HOME/pkg}"          # where shards + toml land
R2_SYNC="${R2_SYNC:-r2_sync.py}"

echo "==> Downloading r2://$R2_PREFIX -> $PKG_DIR"
python "$R2_SYNC" download --prefix "$R2_PREFIX" --dest "$PKG_DIR"

echo "==> Extracting shards into $TRAIN_DATA"
mkdir -p "$TRAIN_DATA"
shopt -s nullglob
shards=("$PKG_DIR"/shards/*.tar)
if [ ${#shards[@]} -eq 0 ]; then
  echo "ERROR: no shard tars found in $PKG_DIR/shards" >&2
  exit 1
fi
for tarball in "${shards[@]}"; do
  echo "  extracting $(basename "$tarball")"
  tar -xf "$tarball" -C "$TRAIN_DATA"
done

img_count=$(find "$TRAIN_DATA" -maxdepth 1 -type f \
  \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l)
txt_count=$(find "$TRAIN_DATA" -maxdepth 1 -type f -iname '*.txt' | wc -l)

echo "==> Extracted: $img_count images, $txt_count caption files into $TRAIN_DATA"
if [ "$img_count" -ne "$txt_count" ]; then
  echo "WARNING: image/caption count mismatch — kohya errors if any image lacks a .txt" >&2
fi

echo "==> dataset.toml is at: $PKG_DIR/dataset.toml"
