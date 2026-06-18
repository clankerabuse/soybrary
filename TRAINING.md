# Soybrary SDXL LoRA Training Pipeline

## What this project is

**Soybrary** is a local scraper + web UI for soybooru.com. It has scraped ~167K posts into:
- `data/images/` — 156,570 image files (`{id}.{ext}`)
- `data/metadata/` — 167,071 JSON files (`{id}.json`) with tags, variants, subvariants
- `data/soybooru.db` — SQLite index of all posts
- These are also mirrored on **Cloudflare R2** (`soyjak-training` bucket) under `images/` and `metadata/` (legacy individual files), and packaged as tar shards under `datasets/soyjak-sdxl-{full,pilot}/` for fast Lambda pulls

The goal is to train an **SDXL LoRA** on the ~124K usable static images so it can generate soyjak variants on demand.

## Key design decisions (already made, don't revisit)

- **No fixed trigger word.** Variant names (`chudjak`, `cobson`, `gapejak`, etc.) are the differentiators. Caption = `variants, subvariants, tags` joined, variants first. `keep_tokens=1` pins the lead variant.
- **sd-scripts pinned to v0.10.6** (stable release before the v0.11.0 refactor that dropped June 12 2026). Do not upgrade.
- **PyTorch 2.6.0 + CUDA 12.4** in an isolated venv (`~/sd-venv`).
- **boto3/botocore pinned `<1.36.0`** to avoid Cloudflare R2 CRC32 checksum breakage introduced in 1.36.
- **DreamBooth-style dataset** (image + `.txt` sidecar per image, flat directory). NOT the kohya fine-tuning `metadata_file` style — mixing these causes `voluptuous.error.MultipleInvalid`.
- **SDXL base model:** `bdsqlsz/stable-diffusion-xl-base-1.0_fixvae_fp16` (fixed VAE). Do NOT set `vae = ""` in config — sd-scripts treats that as an empty HF repo id and crashes.
- **Training target:** Lambda Labs **2× H100 80GB SXM** on **plain Ubuntu 22.04** (NOT Lambda Stack). Do not use 1× A100 for full runs — it works but latent caching alone can take ~8 h and training another ~12–24 h vs ~4–5 h total on 2× H100. Lambda Stack has a driver/FM pairing bug on 2× H100 SXM that leaves `Fabric State: In Progress` and blocks CUDA init (error 802). Plain Ubuntu installs a clean driver and the fabric reaches `Completed` without issue.
- **Multi-GPU (2× H100):** `setup_lambda.sh` auto-detects GPU count and writes an accelerate DDP config (`distributed_type: MULTI_GPU`, `num_processes: 2`) when both GPUs are visible. `train_lora.sh` passes `--num_processes` explicitly. Per-GPU batch is `BATCH_SIZE` (default 8) and **must match `train_batch_size`** in the config; effective batch = `BATCH_SIZE × NUM_GPUS` (8 × 2 = 16). LR was scaled `1e-4 → 1.5e-4` (moderate sqrt-style bump) for the 4× larger effective batch. Override with `NUM_GPUS=` / `BATCH_SIZE=` env vars only when debugging.
- **SSH key:** `~/.ssh/soyjak-training-new.pem` (the `soyjak-training-new` Lambda key pair). Old key was `lambda-training.pem` (retired after pilot).

## Repository structure

```
build_dataset.py        # Phase 1: filter DB, build captions, emit JSONL manifest
validate_images.py      # Pre-upload scan: strict image validation + quarantine
image_validate.py       # Shared validation logic (local + Lambda prune)
package_dataset.py      # Phase 2: tar shards (images + baked-in .txt captions) → upload to R2
r2_sync.py              # R2 upload/download helper (boto3, checksum workaround)
requirements.txt        # Updated with boto3/botocore/python-dotenv pins
.env.example            # Template for R2 + Lambda + HF credentials
.env                    # Your actual credentials (gitignored)

train/
  setup_lambda.sh       # Run once on Lambda: installs Python 3.10 venv, torch, sd-scripts
  pull_data.sh          # Downloads tar shards from R2 and extracts them (captions baked in)
  gen_captions.py       # Legacy: only needed if rebuilding captions outside the shard pipeline
  train_lora.sh         # Generates dataset.toml, downloads base model, launches training
  push_model.sh         # Uploads trained LoRA .safetensors to R2 (must run before shutdown)
  prune_bad_images.py   # Drop corrupt images (auto-run by pull_data.sh + train_lora.sh)
  config_pilot.toml     # Pilot run config: 2000 steps (~3 epochs over 10K images at eff. batch 16)
  config.toml           # Full run config: 12000 steps (~1.5 epochs over 124K images at eff. batch 16)
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
images/{id}.{ext}                       # all scraped images (~156K files, legacy individual files)
metadata/{id}.json                      # all metadata JSONs (~167K files, legacy)
manifests/dataset.jsonl                 # full 124K manifest
manifests/dataset_pilot10k.jsonl        # pilot 10K manifest

datasets/soyjak-sdxl-full/             # packaged full dataset (from package_dataset.py)
  shard_manifest.json                  #   shard list + sha256 + counts
  shards/shard_0000.tar                #   flat tar: {id}.{ext} + {id}.txt per image
  shards/shard_0001.tar
  ...

datasets/soyjak-sdxl-pilot/            # packaged pilot dataset
  shard_manifest.json
  shards/shard_0000.tar
  ...

models/soyjak-lora-sdxl-pilot/         # pilot LoRA output (after push_model.sh)
models/soyjak-lora-sdxl/               # full run LoRA output (after push_model.sh)
```

## MODE system

All train/ scripts accept `MODE=pilot` (default) or `MODE=full`:

| | Pilot | Full |
|---|---|---|
| Manifest | `dataset_pilot10k.jsonl` | `dataset.jsonl` |
| Images | 10,000 | 123,778 |
| image_dir on Lambda | `/home/ubuntu/train_data_pilot` | `/home/ubuntu/train_data` |
| Train config | `config_pilot.toml` | `config.toml` |
| Steps (2×H100, eff. batch 16) | 2,000 (~3 epochs) | 12,000 (~1.5 epochs) |
| R2 model prefix | `models/soyjak-lora-sdxl-pilot` | `models/soyjak-lora-sdxl` |
| Expected time (2×H100) | ~30-60 min | ~4-5 hrs total |
| Expected time (1×A100, avoid) | ~2-3 hrs | ~20-30+ hrs total |

### Timing breakdown (2× H100, full run)

| Phase | ~Duration | Notes |
|---|---|---|
| Data pull | 30–60 min | Network-bound; same on any GPU |
| Latent cache | 2–4 hrs | Single-GPU VAE encode + disk write; ~124k images |
| Training (12k steps) | 2–3 hrs | DDP across both H100s, effective batch 16 |
| **Total** | **~4–5 hrs** | After data pull completes |

## Instance selection

**Always use: Lambda Labs → 2× H100 80GB SXM → plain Ubuntu 22.04 (NOT Lambda Stack)**

Do **not** use Lambda Stack 22.04 on 2× H100 SXM — driver/Fabric Manager version mismatch leaves
both GPUs stuck at `Fabric State: In Progress`, blocking CUDA init with error 802. Not fixable by
reboot. Plain Ubuntu 22.04 + `setup_lambda.sh` installs a clean driver; fabric reaches
`Completed: Success`.

**Launch checklist**

1. Instance type: **gpu_2x_h100_sxm5** (or equivalent 2× H100 SXM 80GB)
2. Image: **Ubuntu 22.04** (plain — not Lambda Stack)
3. SSH key: `soyjak-training-new`
4. Disk: ≥ 200 GB (images ~80 GB + latent cache ~40 GB + checkpoints + headroom)

After launching, sanity-check **before** running setup:

```bash
nvidia-smi -L   # expect: GPU 0 + GPU 1: NVIDIA H100 80GB HBM3
nvidia-smi -q | grep -A2 "Fabric$"
# Both GPUs must show: State: Completed / Status: Success
# If "In Progress" → terminate and relaunch (Lambda Stack host or bad image)
```

## Training config key values (2× H100, full run)

| Setting | Value | Notes |
|---|---|---|
| `train_batch_size` | 8 | per GPU; effective batch = 16 |
| `max_train_steps` | 12,000 | ~1.5 epochs over 123,778 images |
| `learning_rate` / `unet_lr` | 1.5e-4 | scaled up from 1e-4 for 4× larger effective batch |
| `text_encoder_lr` | 7.5e-5 | scaled proportionally |
| `lr_warmup_steps` | 240 | ~2% of max_train_steps |
| `save_every_n_steps` | 1,000 | 12 checkpoints total |
| `sample_every_n_steps` | 1,000 | sample images at each checkpoint |

## Session history

### Session 1 — Pilot run (June 13 2026)
- Instance: Lambda Labs A100-SXM4-40GB at `<instance-ip>` (terminated)
- SSH key: `~/.ssh/lambda-training.pem` (retired)
- Result: **COMPLETE.** 9,994 images, 7,500 steps (~3 epochs), ~3 hrs total
- Samples: `~/Downloads/soyjak-samples/sample/soyjak-lora-sdxl-pilot_*.png`
- Checkpoints on R2: `models/soyjak-lora-sdxl-pilot-step*.safetensors` (229 MB each)

### Session 2 — Full run setup (June 15 2026)
- SSH key: `~/.ssh/soyjak-training-new.pem` (key pair: `soyjak-training-new`)
- Lambda API key: `...` (in `.env` as `LAMBDA_API_KEY`)
- Discovered: Lambda Stack 22.04 broken on 2× H100 SXM (Fabric State stuck In Progress)
- Fix: plain Ubuntu 22.04 + manual driver install → fabric reaches Completed
- Instance: `<instance-ip>` — full run in progress

### Session 3 — Full run on 1× A100 (June 17 2026, abandoned)
- Instance: Lambda Labs 1× A100 at `<instance-ip>` — switched away after ~8 h latent caching
- Issue: 1× GPU ~4–6× slower than 2× H100 for full run; corrupt JPEG in shards crashed first train attempt
- Fix: `prune_bad_images.py` added; project re-centered on 2× H100 SXM

## Known issues and fixes

1. **`vae = ""`** in config causes crash — sd-scripts treats it as an empty HF repo. Key must be omitted entirely.
2. **`libGL.so.1` missing** on Lambda Ubuntu — `cv2` fails to import. Fixed by adding `libgl1 libglib2.0-0` to `setup_lambda.sh`.
3. **boto3 ≥1.36 CRC32 checksum** breaks R2 uploads. Pinned `<1.36.0` in requirements.
4. **sd-scripts v0.11.0** released June 12 2026 — major refactor, untested. Pinned to v0.10.6.
5. **DB `extension` column unreliable** (says `jpg` for `.png` files). Always trust filesystem extension.
6. **kohya `metadata_file`** style incompatible with raw scraper JSON. Use DreamBooth style (`.txt` sidecars) only.
7. **Silent CPU fallback** — Lambda host can boot without NVIDIA driver. Fixed: `setup_lambda.sh` installs driver if missing and both scripts assert `torch.cuda.is_available()` before proceeding.
8. **`Python.h` missing / Triton compile fail** — Triton JIT needs `python3.10-dev` + `build-essential`. Now installed by `setup_lambda.sh`.
9. **Lambda Stack 2× H100 SXM — Fabric State stuck "In Progress" / CUDA error 802** — Driver/FM version mismatch in Lambda Stack image. FM reports "Pre-NVL5 / Nothing to do" and exits; GPUs never leave In Progress. Not fixable by reboot or FM config. Fix: use plain Ubuntu 22.04 instead.
10. **Corrupt/truncated images crash training** — sd-scripts dies on bad files during latent caching. Validation now runs the full training load path (EXIF transpose, RGB convert, bucket downscale, pixel read, re-encode), not just Pillow verify(). Use `validate_images.py` locally before packaging; `build_dataset.py --validate-images` excludes bad files from the manifest; `package_dataset.py` re-checks by default; `prune_bad_images.py` is a last-resort safety net on Lambda. Override max size with `MAX_LONG_SIDE=0`. Rebuild manifests after quarantining bad files.

---

## Command cheat-sheet

### Local — one-time / when manifest or images change
```bash
cd /path/to/soybrary

# Scan all local images before packaging (recommended after pulling data):
.venv/bin/python validate_images.py --manifest data/manifests/dataset.jsonl --quarantine

# Or scan every file under data/images/:
.venv/bin/python validate_images.py --quarantine

# Regenerate manifest (only if new images scraped since last run):
.venv/bin/python build_dataset.py --validate-images

# Re-upload manifest to R2:
.venv/bin/python r2_sync.py upload-file --src data/manifests/dataset.jsonl --key manifests/dataset.jsonl

# Package images+captions into tar shards and upload to R2 (full run, ~80 GB):
.venv/bin/python package_dataset.py --mode full

# Package pilot subset only (~10 GB):
.venv/bin/python package_dataset.py --mode pilot

# Package only (no upload), e.g. to inspect shards first:
.venv/bin/python package_dataset.py --mode full --no-upload

# Upload already-packaged shards (skip re-packaging):
.venv/bin/python package_dataset.py --mode full --upload-only
```

### Local — copy scripts to a new instance

Fish shell (use `set`, not `IP=`):

```fish
set IP <instance-ip>
set KEY ~/.ssh/soyjak-training-new.pem

ssh -i $KEY ubuntu@$IP 'mkdir -p ~/soybrary'
scp -i $KEY /path/to/soybrary/r2_sync.py ubuntu@$IP:~/soybrary/
scp -i $KEY /path/to/soybrary/image_validate.py ubuntu@$IP:~/soybrary/
scp -i $KEY -r /path/to/soybrary/train ubuntu@$IP:~/soybrary/
```

Bash:

```bash
IP=<instance-ip>
KEY=~/.ssh/soyjak-training-new.pem

ssh -i $KEY ubuntu@$IP 'mkdir -p ~/soybrary'
scp -i $KEY /path/to/soybrary/r2_sync.py ubuntu@$IP:~/soybrary/
scp -i $KEY /path/to/soybrary/image_validate.py ubuntu@$IP:~/soybrary/
scp -i $KEY -r /path/to/soybrary/train ubuntu@$IP:~/soybrary/
```

### On the instance — sanity check first (before anything else)

Plain Ubuntu may ship without the NVIDIA driver — `setup_lambda.sh` installs it if missing (reboot may be required).

```bash
nvidia-smi -L   # expect: GPU 0 + GPU 1: NVIDIA H100 80GB HBM3
nvidia-smi -q | grep -A2 "Fabric$"
# Both GPUs must show: State: Completed / Status: Success
# If "In Progress" → terminate and relaunch (Lambda Stack host or bad image)
```

### On the instance — setup (once per fresh instance)
```bash
export R2_ACCOUNT_ID="..."
export R2_ACCESS_KEY_ID="..."
export R2_SECRET_ACCESS_KEY="..."
export R2_BUCKET_NAME="soyjak-training"
export R2_ENDPOINT="https://....r2.cloudflarestorage.com"

cd ~/soybrary
bash train/setup_lambda.sh
# Expected: "2 GPU(s), distributed_type=MULTI_GPU, bf16" + "CUDA OK: NVIDIA H100 80GB HBM3"

source ~/sd-venv/bin/activate
```

### On the instance — pull data + train (in tmux)

Kitty terminal: override TERM so tmux behaves (`xterm-kitty` breaks keybindings/colors).

```bash
TERM=xterm-256color tmux new -s training
```

Inside tmux — re-export R2 vars, then:

```bash
export R2_ACCOUNT_ID="..."
export R2_ACCESS_KEY_ID="..."
export R2_SECRET_ACCESS_KEY="..."
export R2_BUCKET_NAME="soyjak-training"
export R2_ENDPOINT="https://<account-id>.r2.cloudflarestorage.com"

cd ~/soybrary
source ~/sd-venv/bin/activate

# pull_data.sh downloads shards, extracts, and prunes corrupt images automatically
MODE=full bash train/pull_data.sh      # ~30-60 min
MODE=full bash train/train_lora.sh     # ~4-5 hrs on 2× H100 (latent cache + training)

# Detach: Ctrl-b d   |   Reattach: TERM=xterm-256color tmux attach -t training
```

### On the instance — push model before terminating
```bash
MODE=full bash train/push_model.sh
# Uploads all .safetensors under /home/ubuntu/out/ to r2://soyjak-training/models/soyjak-lora-sdxl/
```

### Local — download the trained LoRA
```bash
cd /path/to/soybrary
.venv/bin/python r2_sync.py download --prefix models/soyjak-lora-sdxl --dest ./full_lora
```
