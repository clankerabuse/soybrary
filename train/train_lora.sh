#!/usr/bin/env bash
# train_lora.sh - Phase 4: launch SDXL LoRA training with kohya sd-scripts.
#
#   source ~/sd-venv/bin/activate
#   bash train/train_lora.sh
#
# Expects (from setup_lambda.sh / pull_data.sh):
#   - venv active with torch 2.6.0 + sd-scripts v0.10.6
#   - dataset extracted to image_dir referenced by dataset.toml
#   - SDXL base checkpoint downloaded (see DOWNLOAD note below)
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME}"
SDSCRIPTS_DIR="${SDSCRIPTS_DIR:-$WORKDIR/sd-scripts}"
# MODE selects the pilot (~10k) or full (~124k) run. Override CONFIG/DATASET_CONFIG
# directly for anything custom.
#   MODE=pilot bash train/train_lora.sh   # small first run (default)
#   MODE=full  bash train/train_lora.sh   # full dataset
MODE="${MODE:-pilot}"
if [ "$MODE" = "pilot" ]; then
  CONFIG="${CONFIG:-$WORKDIR/soybrary/train/config_pilot.toml}"
  DATASET_CONFIG="${DATASET_CONFIG:-$WORKDIR/pkg/dataset.toml}"
elif [ "$MODE" = "full" ]; then
  CONFIG="${CONFIG:-$WORKDIR/soybrary/train/config.toml}"
  DATASET_CONFIG="${DATASET_CONFIG:-$WORKDIR/pkg/dataset.toml}"
else
  echo "ERROR: MODE must be 'pilot' or 'full' (got '$MODE')" >&2
  exit 1
fi
MODEL="/home/ubuntu/models/sd_xl_base_1.0_fixvae_fp16.safetensors"

mkdir -p /home/ubuntu/models /home/ubuntu/out /home/ubuntu/logs

# --- Download SDXL base (fixed-VAE fp16) if missing --------------------------
if [ ! -f "$MODEL" ]; then
  echo "==> Downloading SDXL base (fixed-VAE fp16)"
  python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os
p = hf_hub_download(
    repo_id="bdsqlsz/stable-diffusion-xl-base-1.0_fixvae_fp16",
    filename="sd_xl_base_1.0_fixvae_fp16.safetensors",
)
os.makedirs("/home/ubuntu/models", exist_ok=True)
shutil.copy(p, "/home/ubuntu/models/sd_xl_base_1.0_fixvae_fp16.safetensors")
print("downloaded base model")
PY
fi

[ -f "$DATASET_CONFIG" ] || { echo "ERROR: dataset config not found: $DATASET_CONFIG" >&2; exit 1; }
[ -f "$CONFIG" ]         || { echo "ERROR: train config not found: $CONFIG" >&2; exit 1; }

cd "$SDSCRIPTS_DIR"
echo "==> Launching SDXL LoRA training (MODE=$MODE)"
echo "    train config:   $CONFIG"
echo "    dataset config: $DATASET_CONFIG"

accelerate launch --num_cpu_threads_per_process 8 \
  sdxl_train_network.py \
  --config_file "$CONFIG" \
  --dataset_config "$DATASET_CONFIG"

echo "==> Training finished. LoRA written under /home/ubuntu/out"
echo "    Push it to R2 with: bash train/push_model.sh"
