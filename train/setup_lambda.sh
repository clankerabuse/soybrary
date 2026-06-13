#!/usr/bin/env bash
# setup_lambda.sh - Phase 4: prepare a Lambda Labs A100 instance for SDXL LoRA
# training with kohya-ss/sd-scripts.
#
# Run this ONCE on a fresh Lambda Labs instance (Ubuntu, Lambda Stack preinstalled).
# It deliberately creates an isolated venv so we control torch/sd-scripts versions
# instead of relying on whatever Lambda Stack ships.
#
#   bash train/setup_lambda.sh
#
# Pinned, verified versions (see train/requirements-lock.txt):
#   Python 3.10, torch 2.6.0 + cu124, sd-scripts tag v0.10.6
set -euo pipefail

SDSCRIPTS_TAG="v0.10.6"          # stable release BEFORE the v0.11.0 refactor
WORKDIR="${WORKDIR:-$HOME}"
VENV="${VENV:-$WORKDIR/sd-venv}"
SDSCRIPTS_DIR="$WORKDIR/sd-scripts"

echo "==> Working dir: $WORKDIR"
cd "$WORKDIR"

# --- System packages --------------------------------------------------------
# OpenCV imports cv2 even for headless training; libGL.so.1 is not present on
# some minimal Lambda/Ubuntu images unless libgl1 is installed.
echo "==> Installing system runtime packages"
sudo apt-get update -y
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev libgl1 libglib2.0-0

# --- Python 3.10 venv -------------------------------------------------------
if ! command -v python3.10 >/dev/null 2>&1; then
  echo "ERROR: python3.10 was not installed successfully" >&2
  exit 1
fi

echo "==> Creating venv at $VENV"
python3.10 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

# --- PyTorch (CUDA 12.4) ----------------------------------------------------
# NOTE: For RTX 50-series use torch 2.8.0 + cu128 instead. A100 uses cu124.
echo "==> Installing PyTorch 2.6.0 (cu124)"
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# --- sd-scripts @ pinned tag ------------------------------------------------
if [ ! -d "$SDSCRIPTS_DIR" ]; then
  echo "==> Cloning sd-scripts @ $SDSCRIPTS_TAG"
  git clone https://github.com/kohya-ss/sd-scripts.git "$SDSCRIPTS_DIR"
fi
cd "$SDSCRIPTS_DIR"
git fetch --tags
git checkout "$SDSCRIPTS_TAG"
echo "==> sd-scripts at: $(git describe --tags)"
pip install --use-pep517 -r requirements.txt
cd "$WORKDIR"

# --- our helper deps (R2 sync etc.) -----------------------------------------
echo "==> Installing R2 sync deps"
pip install "boto3>=1.34.0,<1.36.0" "botocore>=1.34.0,<1.36.0" python-dotenv huggingface_hub

# --- accelerate config (non-interactive, single A100, bf16) -----------------
echo "==> Writing accelerate config (single GPU, bf16)"
mkdir -p "$HOME/.cache/huggingface/accelerate"
cat > "$HOME/.cache/huggingface/accelerate/default_config.yaml" <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML

cat <<EOF

==> Setup complete.
    venv:        $VENV
    sd-scripts:  $SDSCRIPTS_DIR ($SDSCRIPTS_TAG)

Next steps:
  source "$VENV/bin/activate"
  # put R2 creds in .env (copy from .env.example), then:
  bash train/pull_data.sh
  bash train/train_lora.sh
EOF
