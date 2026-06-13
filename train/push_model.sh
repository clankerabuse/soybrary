#!/usr/bin/env bash
# push_model.sh - Phase 4: upload the trained LoRA to R2 (default) and,
# optionally, to Hugging Face.
#
#   bash train/push_model.sh                 # -> R2 only (default)
#   PUSH_HF=1 bash train/push_model.sh       # -> R2 + Hugging Face
#
# The Lambda instance disk is wiped on shutdown, so this MUST run before you
# terminate the instance.
set -euo pipefail

OUT_DIR="${OUT_DIR:-/home/ubuntu/out}"
R2_SYNC="${R2_SYNC:-r2_sync.py}"
# Pilot and full models go to distinct prefixes so they don't overwrite.
MODE="${MODE:-pilot}"
if [ "$MODE" = "pilot" ]; then
  R2_MODEL_PREFIX="${R2_MODEL_PREFIX:-models/soyjak-lora-sdxl-pilot}"
else
  R2_MODEL_PREFIX="${R2_MODEL_PREFIX:-models/soyjak-lora-sdxl}"
fi
PUSH_HF="${PUSH_HF:-0}"

shopt -s nullglob
models=("$OUT_DIR"/*.safetensors)
if [ ${#models[@]} -eq 0 ]; then
  echo "ERROR: no .safetensors found in $OUT_DIR" >&2
  exit 1
fi

# --- R2 (default) -----------------------------------------------------------
echo "==> Uploading ${#models[@]} file(s) to r2://$R2_MODEL_PREFIX/"
for m in "${models[@]}"; do
  base="$(basename "$m")"
  python "$R2_SYNC" upload-file --src "$m" --key "$R2_MODEL_PREFIX/$base"
done
echo "==> R2 upload complete."

# --- Hugging Face (optional) ------------------------------------------------
if [ "$PUSH_HF" = "1" ]; then
  : "${HF_TOKEN:?set HF_TOKEN in env/.env to push to Hugging Face}"
  : "${HF_REPO_ID:?set HF_REPO_ID (e.g. your_username/soyjak-lora-sdxl)}"
  echo "==> Uploading to Hugging Face repo $HF_REPO_ID"
  for m in "${models[@]}"; do
    python - "$m" <<'PY'
import os, sys
from huggingface_hub import HfApi
path = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["HF_REPO_ID"]
api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj=path, path_in_repo=os.path.basename(path),
                repo_id=repo, repo_type="model")
print("uploaded", path, "->", repo)
PY
  done
  echo "==> Hugging Face upload complete."
else
  echo "==> Skipping Hugging Face (set PUSH_HF=1 to enable)."
fi
