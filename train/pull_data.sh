#!/usr/bin/env bash
# pull_data.sh - Pull the dataset archive from R2 and extract it on the Lambda
# instance.  Captions are already baked into the tar shards by package_dataset.py,
# so no separate metadata download or gen_captions step is needed.
#
# MODE=pilot (default): pulls the ~10K pilot shards.
# MODE=full:            pulls all ~124K full-run shards.
#
#   bash train/pull_data.sh              # pilot (default)
#   MODE=full bash train/pull_data.sh    # full run
#
# Requires: .env with R2 creds (or R2_* env vars already exported),
#           r2_sync.py available, boto3 installed (setup_lambda.sh handles this).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${MODE:-pilot}"

if [ "$MODE" = "pilot" ]; then
    TRAIN_DATA="${TRAIN_DATA:-/home/ubuntu/train_data_pilot}"
    R2_PREFIX="datasets/soyjak-sdxl-pilot"
elif [ "$MODE" = "full" ]; then
    TRAIN_DATA="${TRAIN_DATA:-/home/ubuntu/train_data}"
    R2_PREFIX="datasets/soyjak-sdxl-full"
else
    echo "ERROR: MODE must be 'pilot' or 'full' (got '$MODE')" >&2
    exit 1
fi

R2_SYNC="python $REPO_DIR/r2_sync.py"

mkdir -p "$TRAIN_DATA"

# --- 1. Download + extract all shards from R2 --------------------------------
# Each shard is a flat tar of {id}.{ext} images + {id}.txt caption sidecars.
# download-archive streams each shard, verifies sha256, extracts, then deletes
# the shard file — peak extra disk = one shard at a time.
echo "==> Downloading + extracting archive ($MODE) from R2: $R2_PREFIX"
$R2_SYNC download-archive \
    --mode "$MODE" \
    --dest "$TRAIN_DATA" \
    --r2-prefix "$R2_PREFIX"

# --- 2. Sanity check ---------------------------------------------------------
img_count=$(find "$TRAIN_DATA" -maxdepth 1 -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l)
txt_count=$(find "$TRAIN_DATA" -maxdepth 1 -type f -iname '*.txt' | wc -l)
echo "==> $TRAIN_DATA: $img_count images, $txt_count captions"
if [ "$img_count" -ne "$txt_count" ]; then
    echo "WARNING: image/caption count mismatch — re-run package_dataset.py to rebuild shards" >&2
fi

# --- 3. Drop corrupt / oversized images ------------------------------------
MAX_LONG_SIDE="${MAX_LONG_SIDE:-2048}"
echo "==> Pruning bad images (corrupt + longest side > ${MAX_LONG_SIDE}px)"
python "$REPO_DIR/train/prune_bad_images.py" "$TRAIN_DATA" --max-long-side "$MAX_LONG_SIDE"

echo ""
echo "==> Data ready at $TRAIN_DATA"
echo "    Next: bash train/train_lora.sh"
