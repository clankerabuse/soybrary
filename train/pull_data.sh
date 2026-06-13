#!/usr/bin/env bash
# pull_data.sh - Pull images + metadata from R2 directly onto the Lambda instance,
# then generate caption .txt sidecars with gen_captions.py.
#
# The images and metadata are already on R2 under images/ and metadata/ —
# no repackaging or separate upload is needed.
#
# MODE=pilot (default): pulls only the ~10k images listed in the manifest.
# MODE=full:            pulls all ~124k images listed in the full manifest.
#
#   bash train/pull_data.sh              # pilot (default)
#   MODE=full bash train/pull_data.sh    # full run
#
# Requires: .env with R2 creds, r2_sync.py + gen_captions.py available,
#           boto3 + python-dotenv installed (setup_lambda.sh handles this).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${MODE:-pilot}"

if [ "$MODE" = "pilot" ]; then
    MANIFEST_KEY="manifests/dataset_pilot10k.jsonl"
    TRAIN_DATA="${TRAIN_DATA:-/home/ubuntu/train_data_pilot}"
elif [ "$MODE" = "full" ]; then
    MANIFEST_KEY="manifests/dataset.jsonl"
    TRAIN_DATA="${TRAIN_DATA:-/home/ubuntu/train_data}"
else
    echo "ERROR: MODE must be 'pilot' or 'full' (got '$MODE')" >&2
    exit 1
fi

METADATA_DIR="${METADATA_DIR:-/home/ubuntu/metadata}"
MANIFESTS_DIR="${MANIFESTS_DIR:-/home/ubuntu/manifests}"
R2_SYNC="python $REPO_DIR/r2_sync.py"
GEN_CAPTIONS="python $REPO_DIR/train/gen_captions.py"

mkdir -p "$TRAIN_DATA" "$METADATA_DIR" "$MANIFESTS_DIR"

# --- 1. Fetch the manifest from R2 ------------------------------------------
MANIFEST_LOCAL="$MANIFESTS_DIR/$(basename "$MANIFEST_KEY")"
echo "==> Fetching manifest ($MODE): r2://$MANIFEST_KEY"
$R2_SYNC download-file --key "$MANIFEST_KEY" --dest "$MANIFEST_LOCAL"

# --- 2. Download images + metadata listed in the manifest -------------------
echo "==> Downloading images + metadata ($MODE) from R2"
$R2_SYNC download-manifest \
    --manifest "$MANIFEST_LOCAL" \
    --image-dir "$TRAIN_DATA" \
    --metadata-dir "$METADATA_DIR"

# --- 3. Generate caption .txt sidecars in-place -----------------------------
echo "==> Generating caption sidecars"
$GEN_CAPTIONS \
    --image-dir "$TRAIN_DATA" \
    --metadata-dir "$METADATA_DIR" \
    --manifest "$MANIFEST_LOCAL"

# --- Sanity check -----------------------------------------------------------
img_count=$(find "$TRAIN_DATA" -maxdepth 1 -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | wc -l)
txt_count=$(find "$TRAIN_DATA" -maxdepth 1 -type f -iname '*.txt' | wc -l)
echo "==> $TRAIN_DATA: $img_count images, $txt_count captions"
if [ "$img_count" -ne "$txt_count" ]; then
    echo "WARNING: image/caption count mismatch — check gen_captions output above" >&2
fi

echo ""
echo "==> Data ready at $TRAIN_DATA"
echo "    Next: bash train/train_lora.sh"
