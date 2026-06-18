#!/usr/bin/env bash
# setup_lambda.sh - Phase 4: prepare a Lambda Labs GPU instance for SDXL LoRA
# training with kohya-ss/sd-scripts.
#
# Run this ONCE on a fresh Lambda Labs instance (Ubuntu, Lambda Stack preinstalled).
# It deliberately creates an isolated venv so we control torch/sd-scripts versions
# instead of relying on whatever Lambda Stack ships.
#
#   bash train/setup_lambda.sh
#
# Multi-GPU: the generated accelerate config auto-detects the number of visible
# GPUs (nvidia-smi) and configures DDP (MULTI_GPU) when >1 is present. On a
# 2× H100 box this trains both GPUs out of the box; on a single GPU it falls
# back to distributed_type NO. Override with NUM_GPUS=N if needed.
#
# Pinned, verified versions (see train/requirements-lock.txt):
#   Python 3.10, torch 2.6.0 + cu124, sd-scripts tag v0.10.6
#   (cu124 supports Hopper/H100 — no change needed vs A100.)
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
# python3.10-dev + build-essential are REQUIRED: once CUDA is active, Triton
# JIT-compiles a cuda_utils helper and needs Python.h + gcc, otherwise the
# training launch dies with "fatal error: Python.h: No such file or directory".
echo "==> Installing system runtime packages"
sudo apt-get update -y
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev build-essential \
    libgl1 libglib2.0-0

# --- NVIDIA driver ----------------------------------------------------------
# Lambda Stack images normally ship the driver, but some hosts come up WITHOUT
# it (no /dev/nvidia*, no nvidia-smi, no kernel module). In that state accelerate
# silently falls back to CPU and "training" runs ~100x slower with no error.
# This block installs a CUDA 12.4-capable driver (>=550) if one is missing.
echo "==> Checking NVIDIA driver"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  if lspci 2>/dev/null | grep -iq nvidia; then
    echo "==> NVIDIA GPU present but driver not loaded — installing nvidia-driver-550-server"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-550-server
    sudo modprobe nvidia || true
    sudo modprobe nvidia_uvm || true
    nvidia-smi || { echo "ERROR: driver installed but nvidia-smi still fails — a reboot may be required" >&2; exit 1; }
  else
    echo "ERROR: no NVIDIA GPU found on the PCI bus (lspci). This instance has no GPU." >&2
    exit 1
  fi
fi
echo "==> nvidia-smi OK:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# --- Multi-GPU fabric sanity (2× H100 SXM on Lambda Stack breaks here) ------
NUM_GPUS_CHECK="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
if [ "$NUM_GPUS_CHECK" -gt 1 ]; then
  if nvidia-smi -q 2>/dev/null | grep -A2 "Fabric" | grep -q "In Progress"; then
    echo "ERROR: NVLink fabric stuck at 'In Progress' — CUDA init will fail (error 802)." >&2
    echo "       Terminate and relaunch on plain Ubuntu 22.04 (NOT Lambda Stack)." >&2
    exit 1
  fi
  echo "==> Multi-GPU fabric state OK ($NUM_GPUS_CHECK GPUs)"
elif [ "$NUM_GPUS_CHECK" -eq 1 ]; then
  echo "WARNING: only 1 GPU visible — this project is tuned for 2× H100 (effective batch 16)." >&2
  echo "         Full runs on 1× GPU work but take ~4-6× longer. Prefer 2× H100 SXM 80GB." >&2
fi

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

# --- accelerate config (non-interactive, auto multi-GPU, bf16) --------------
# Detect visible GPUs so a 2× H100 box trains with DDP automatically. Override
# with NUM_GPUS=N (e.g. NUM_GPUS=1 to force single-GPU on a multi-GPU host).
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}"
[ "$NUM_GPUS" -ge 1 ] 2>/dev/null || NUM_GPUS=1
if [ "$NUM_GPUS" -gt 1 ]; then
  DISTRIBUTED_TYPE="MULTI_GPU"
else
  DISTRIBUTED_TYPE="'NO'"
fi
echo "==> Writing accelerate config ($NUM_GPUS GPU(s), distributed_type=$DISTRIBUTED_TYPE, bf16)"
mkdir -p "$HOME/.cache/huggingface/accelerate"
cat > "$HOME/.cache/huggingface/accelerate/default_config.yaml" <<YAML
compute_environment: LOCAL_MACHINE
distributed_type: $DISTRIBUTED_TYPE
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML

# --- Fail-fast CUDA assertion -----------------------------------------------
# The single most expensive failure mode is a silent CPU fallback (a missing
# driver cost a full ~9.5h pilot run once). Refuse to declare success unless
# torch can actually see the GPU.
echo "==> Verifying torch can see the GPU"
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("FATAL: torch.cuda.is_available() is False — training would run on CPU. "
             "Check the NVIDIA driver (nvidia-smi) before proceeding.")
print(f"CUDA OK: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
PY

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
