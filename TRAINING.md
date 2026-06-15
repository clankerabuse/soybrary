# Soybrary SDXL LoRA Training Pipeline

## What this project is

**Soybrary** is a local scraper + web UI for soybooru.com. It has scraped ~167K posts into:
- `data/images/` — 156,570 image files (`{id}.{ext}`)
- `data/metadata/` — 167,071 JSON files (`{id}.json`) with tags, variants, subvariants
- `data/soybooru.db` — SQLite index of all posts
- These are also mirrored on **Cloudflare R2** (`soyjak-training` bucket) under `images/` and `metadata/`

The goal is to train an **SDXL LoRA** on the ~124K usable static images so it can generate soyjak variants on demand.

## Key design decisions (already made, don't revisit)

- **No fixed trigger word.** Variant names (`chudjak`, `cobson`, `gapejak`, etc.) are the differentiators. Caption = `variants, subvariants, tags` joined, variants first. `keep_tokens=1` pins the lead variant.
- **sd-scripts pinned to v0.10.6** (stable release before the v0.11.0 refactor that dropped June 12 2026). Do not upgrade.
- **PyTorch 2.6.0 + CUDA 12.4** in an isolated venv (`~/sd-venv`).
- **boto3/botocore pinned `<1.36.0`** to avoid Cloudflare R2 CRC32 checksum breakage introduced in 1.36.
- **DreamBooth-style dataset** (image + `.txt` sidecar per image, flat directory). NOT the kohya fine-tuning `metadata_file` style — mixing these causes `voluptuous.error.MultipleInvalid`.
- **SDXL base model:** `bdsqlsz/stable-diffusion-xl-base-1.0_fixvae_fp16` (fixed VAE). Do NOT set `vae = ""` in config — sd-scripts treats that as an empty HF repo id and crashes.
- **Training target:** Lambda Labs GPU cloud. Images pulled directly from R2 on the instance — no local repackaging needed.

## Repository structure (all on branch `test/sdxl-lora-pipeline`)

```
build_dataset.py        # Phase 1: filter DB, build captions, emit JSONL manifest
package_dataset.py      # Phase 2: (optional) tar shards — NOT needed, R2 already has images
r2_sync.py              # R2 upload/download helper (boto3, checksum workaround)
requirements.txt        # Updated with boto3/botocore/python-dotenv pins
.env.example            # Template for R2 + Lambda + HF credentials
.env                    # Your actual credentials (gitignored)

train/
  setup_lambda.sh       # Run once on Lambda: installs Python 3.10 venv, torch, sd-scripts
  pull_data.sh          # Fetches manifest from R2, downloads images+metadata, gen captions
  gen_captions.py       # Reads metadata/{id}.json, writes {id}.txt caption sidecars
  train_lora.sh         # Generates dataset.toml, downloads base model, launches training
  push_model.sh         # Uploads trained LoRA .safetensors to R2 (must run before shutdown)
  config_pilot.toml     # Pilot run config: 7500 steps (~3 epochs over 10K images)
  config.toml           # Full run config: 30000 steps (~1 epoch over 124K images)
  config_pilot.toml     # Pilot: max_train_steps=7500, output_name=soyjak-lora-sdxl-pilot
  sample_prompts.txt    # Sample prompts for mid-training previews (tests variant separation)
  requirements-lock.txt # Pinned versions with rationale

data/manifests/         # gitignored (lives under data/ which is gitignored)
  dataset.jsonl         # Full manifest: 123,778 images (also on R2: manifests/dataset.jsonl)
  dataset.stats.json    # Stats for full manifest
  dataset_pilot10k.jsonl # Pilot manifest: 10,000 stratified images (on R2: manifests/dataset_pilot10k.jsonl)
```

## Dataset stats

- **Total DB posts:** 242,837
- **Completed static images (PNG/JPEG/WebP):** 149,614
- **After short-side ≥512px filter:** 123,778 images (~80GB)
- **Pilot subset:** 10,000 images, stratified across 3,047 distinct variants (~10 per variant, seed=42)
- **Top variants:** chudjak (22K), gapejak (16K), markiplier_soyjak (15K), soyak (14K), cobson (14K)

## R2 bucket structure (`soyjak-training`)

```
images/{id}.{ext}                       # all scraped images (~156K files)
metadata/{id}.json                      # all metadata JSONs (~167K files)
manifests/dataset.jsonl                 # full 124K manifest
manifests/dataset_pilot10k.jsonl        # pilot 10K manifest
models/soyjak-lora-sdxl-pilot/          # pilot LoRA output (after push_model.sh)
models/soyjak-lora-sdxl/                # full run LoRA output (after push_model.sh)
```

## MODE system

All train/ scripts accept `MODE=pilot` (default) or `MODE=full`:

| | Pilot | Full |
|---|---|---|
| Manifest | `dataset_pilot10k.jsonl` | `dataset.jsonl` |
| Images | 10,000 | 123,778 |
| image_dir on Lambda | `/home/ubuntu/train_data_pilot` | `/home/ubuntu/train_data` |
| Train config | `config_pilot.toml` | `config.toml` |
| Steps | 7,500 (~3 epochs) | 30,000 (~1 epoch) |
| R2 model prefix | `models/soyjak-lora-sdxl-pilot` | `models/soyjak-lora-sdxl` |
| Expected time (A100) | ~9-10 hrs total | ~15-20 hrs total |
| Expected time (2×H100) | ~2-3 hrs total | ~3-4 hrs total |

## Current state (as of session 2 completion)

**Pilot run COMPLETE.** Started June 13 2026 ~14:24 UTC, finished ~17:20 UTC (~3 hours total).

- Branch: `test/sdxl-lora-pipeline` (4 commits ahead of main, including GPU hardening fixes)
- Instance: Lambda Labs A100-SXM4-40GB at `<instance-ip>` (now terminated after `push_model.sh`)
- SSH key: `~/.ssh/lambda-training.pem` (instance no longer running)

**Pilot training summary:**
- Dataset: 9,994 images (10K pilot stratified subset), 2,576 batches/epoch, 3 epochs → 7,500 steps
- **Latent caching:** ~25 min (GPU-bound at ~2.8 it/s after resuming from CPU-cached state)
- **Training steps:** ~2.9 hrs (~1.42 s/it average, A100 at full capacity)
- **Sample checkpoints:** generated at steps 1500, 3000, 4500, 6000, 7500
  - Samples available locally: `~/Downloads/soyjak-samples/sample/soyjak-lora-sdxl-pilot_*.png`
  - Samples show distinct variant rendering (chudjak, cobson, gapejak, etc. visually separating)

**LoRA checkpoints on R2** (`soyjak-training` bucket):
- `models/soyjak-lora-sdxl-pilot-step00001500.safetensors` (229 MB)
- `models/soyjak-lora-sdxl-pilot-step00003000.safetensors` (229 MB)
- `models/soyjak-lora-sdxl-pilot-step00004500.safetensors` (229 MB)
- `models/soyjak-lora-sdxl-pilot-step00006000.safetensors` (229 MB)
- `models/soyjak-lora-sdxl-pilot-step00007500.safetensors` (229 MB) — **use this for generation**
- `models/soyjak-lora-sdxl-pilot.safetensors` (229 MB) — symlink to final (step 7500)

All pushed to R2 at 2026-06-13 12:15:59–12:16:20 CDT (note: step timestamps differ from wall clock due to when checkpoint logic triggered).

### Hardening fixes applied (commit 9312a4e)

The first launch silently ran on CPU for ~9.5 hours due to a missing NVIDIA driver. Fixed:

- `setup_lambda.sh` now:
  - Installs `nvidia-driver-550-server` if GPU is present but driver is missing
  - Installs `python3.10-dev` + `build-essential` (required for Triton CUDA JIT)
  - Asserts `torch.cuda.is_available()` at end; fails if False
- `train_lora.sh` now asserts CUDA before launch (prevents silent CPU fallback)
- Both scripts will now error loudly instead of silently training on CPU

## Next steps / for next session

**To test the pilot LoRA locally:**
```bash
# Download final model from R2:
.venv/bin/python r2_sync.py download --prefix models/soyjak-lora-sdxl-pilot/step00007500 --dest ./pilot_lora

# Use in ComfyUI / A1111 with prompts like:
# - "chudjak, open_mouth, glasses"
# - "cobson, smug"
# - "gapejak, wholesome_soyjak, stubble"
```

**Decision tree:**
- **If pilot quality is good** → run full dataset: `MODE=full` on 2× H100 SXM (~$6.58/hr, ~3-4 hrs, ~30K steps)
- **If pilot quality is bad** → check sample images (`~/Downloads/soyjak-samples/sample/`), tweak learning rate or steps in `config_pilot.toml`, re-run pilot
- **For faster iteration** → use 2× or 4× H100 (reduces ~9 hrs caching+training to ~2-3 hrs)

## Next steps after pilot

- If output quality is good → run full dataset with `MODE=full` on **2× H100 SXM** (~$6.58/hr, ~3-4 hrs)
- If output is bad → diagnose (check sample images generated mid-training at `/home/ubuntu/out/sample/`), adjust learning rate or steps in `config_pilot.toml`, re-run pilot
- For iteration → 2× or 4× H100 recommended (caching + training drops from 10 hrs to ~2-3 hrs)

## Full workflow (next time, from scratch)

```bash
# Local — already done, don't redo unless DB updated:
python build_dataset.py                          # regenerate manifest if new images scraped
# python build_dataset.py --limit 10000 --out data/manifests/dataset_pilot10k.jsonl

# Local — upload manifests if regenerated:
.venv/bin/python r2_sync.py upload-file --src data/manifests/dataset_pilot10k.jsonl --key manifests/dataset_pilot10k.jsonl
.venv/bin/python r2_sync.py upload-file --src data/manifests/dataset.jsonl --key manifests/dataset.jsonl

# Copy scripts to Lambda (minimal — just what's needed):
scp -i ~/.ssh/lambda-training.pem /path/to/soybrary/r2_sync.py ubuntu@<ip>:~/soybrary/
scp -i ~/.ssh/lambda-training.pem -r /path/to/soybrary/train ubuntu@<ip>:~/soybrary/

# On Lambda:
export R2_ACCOUNT_ID="..."
export R2_ACCESS_KEY_ID="..."
export R2_SECRET_ACCESS_KEY="..."
export R2_BUCKET_NAME="soyjak-training"
export R2_ENDPOINT="https://<account_id>.r2.cloudflarestorage.com"

cd ~/soybrary
bash train/setup_lambda.sh
source ~/sd-venv/bin/activate

TERM=xterm-256color tmux new -s training
MODE=pilot bash train/pull_data.sh      # fetches manifest, downloads 10K images+metadata, gen captions
MODE=pilot bash train/train_lora.sh     # downloads SDXL base, generates dataset.toml, trains

# Before terminating:
MODE=pilot bash train/push_model.sh
```

## Known issues fixed in this session

1. **`vae = ""`** in config causes crash — sd-scripts treats it as an empty HF repo. Key must be omitted entirely. Fixed in `config.toml` and `config_pilot.toml`.
2. **`libGL.so.1` missing** on Lambda Ubuntu — `cv2` fails to import. Fixed by adding `libgl1 libglib2.0-0` to `setup_lambda.sh`.
3. **boto3 ≥1.36 CRC32 checksum** breaks R2 uploads. Fixed by pinning `<1.36.0` and conditional `request_checksum_calculation` workaround in `r2_sync.py`.
4. **sd-scripts v0.11.0** released same day (June 12 2026) — major refactor, untested. Pinned to v0.10.6.
5. **DB `extension` column unreliable** (says `jpg` for `.png` files). Always trust filesystem extension, not DB.
6. **kohya `metadata_file`** style is incompatible with raw scraper JSON format. Use DreamBooth style (`.txt` sidecars) only.
7. **Silent CPU fallback** — a Lambda host can boot WITHOUT the NVIDIA driver. accelerate then prints `accelerator device: cpu` and trains ~100x slower with no hard error. Diagnose with `nvidia-smi` / `python -c "import torch; print(torch.cuda.is_available())"`. Fix: `sudo apt-get install -y nvidia-driver-550-server && sudo modprobe nvidia nvidia_uvm`. `setup_lambda.sh` now installs the driver if missing and both scripts assert CUDA before proceeding.
8. **`Python.h` missing / Triton compile fail** — once CUDA is active, Triton JIT-compiles `cuda_utils` and needs `python3.10-dev` + `build-essential`. Without them the launch dies with `fatal error: Python.h: No such file or directory`. Now installed by `setup_lambda.sh`.
