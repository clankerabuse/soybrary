# Soybrary SDXL LoRA Training Pipeline

## What this project is

**Soybrary** is a local scraper + web UI for soybooru.com. It has scraped ~167K posts into:
- `data/images/` — 136,950 image files after quarantine (`{id}.{ext}`; was 156,394 on disk before June 2026 validation)
- `data/metadata/` — 167,071 JSON files (`{id}.json`) with tags, variants, subvariants
- `data/soybooru.db` — SQLite index of all posts
- These are also mirrored on **Cloudflare R2** (`soyjak-training` bucket) under `images/` and `metadata/` (legacy individual files), and packaged as tar shards under `datasets/soyjak-sdxl-{full,pilot}/` for fast Lambda pulls

The goal is to train an **SDXL LoRA** on the **~105K** strictly validated static images so it can generate soyjak variants on demand.

## Key design decisions (already made, don't revisit)

- **No fixed trigger word.** Variant names (`chudjak`, `cobson`, `gapejak`, etc.) are the differentiators. Caption = `variants, subvariants, tags` joined, variants first. `keep_tokens=1` pins the lead variant.
- **sd-scripts pinned to v0.10.6** (stable release before the v0.11.0 refactor that dropped June 12 2026). Do not upgrade.
- **PyTorch 2.6.0 + CUDA 12.4** in an isolated venv (`~/sd-venv`).
- **boto3/botocore pinned `<1.36.0`** to avoid Cloudflare R2 CRC32 checksum breakage introduced in 1.36.
- **DreamBooth-style dataset** (image + `.txt` sidecar per image, flat directory). NOT the kohya fine-tuning `metadata_file` style — mixing these causes `voluptuous.error.MultipleInvalid`.
- **SDXL base model:** `bdsqlsz/stable-diffusion-xl-base-1.0_fixvae_fp16` (fixed VAE). Do NOT set `vae = ""` in config — sd-scripts treats that as an empty HF repo id and crashes.
- **Training target:** Lambda Labs **1× A100 40GB SXM** on **plain Ubuntu 22.04**. Slower than 2× H100 but far more reliable — multi-GPU H100 instances kept hitting fabric/CUDA init failures and burning balance on restarts. Expect ~8 h latent caching + ~12–24 h training for the full run (~20–30 h total). Run in tmux and detach; cost is predictable even if wall clock is long.
- **Single-GPU defaults:** `NUM_GPUS=1`, `BATCH_SIZE=4`, `GRAD_ACCUM_STEPS=4` → effective batch 16 (same training dynamics as the old 2× H100 recipe). LR stays at `1.5e-4`. Override env vars only when debugging.
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
  check_images.py       # Parallel bad-image scan (check-only; auto-run before training)
  check_images.sh       # Wrapper: MODE=pilot|full, uses sd-venv Python
  config_pilot.toml     # Pilot run config: 2000 steps (~3 epochs over 10K images at eff. batch 16)
  config.toml           # Full run config: 12000 steps (~1.8 epochs over 105K images at eff. batch 16)
  sample_prompts.txt    # Sample prompts for mid-training previews (tests variant separation)
  requirements-lock.txt # Pinned versions with rationale

data/manifests/         # gitignored (lives under data/ which is gitignored)
  dataset.jsonl         # Full manifest: 105,495 images (also on R2: manifests/dataset.jsonl)
  dataset.stats.json    # Stats for full manifest
  dataset_pilot10k.jsonl # Pilot manifest: 10,000 stratified images (on R2: manifests/dataset_pilot10k.jsonl)
  bad_images.json       # Report from validate_images.py (quarantined files + reasons)

data/quarantine/images/ # Bad local files moved aside by validate_images.py --quarantine
```

## Dataset stats

- **Total DB posts:** 242,837
- **Completed static images (PNG/JPEG/WebP):** 149,614
- **After strict validation + filters:** **105,495 images** in `dataset.jsonl` (packaged as **9 shards, 38.2 GB** on R2)
- **Pilot subset:** 10,000 images, stratified across 3,047 distinct variants (~10 per variant, seed=42)
- **Top variants (current manifest):** soyak (11K), chudjak (9.6K), gapejak (8.6K), bernd (8.1K), markiplier_soyjak (7.7K)

### Validation funnel (June 2026 full rebuild)

| Stage | Count | Notes |
|---|---|---|
| Files on disk (pre-scan) | 156,394 | Flat `data/images/` |
| Quarantined by `validate_images.py` | 19,620 | Moved to `data/quarantine/images/` |
| Remaining on disk | 136,950 | |
| DB candidates (`completed` static) | 149,614 | |
| **Kept in manifest** | **105,495** | `build_dataset.py --validate-images` |

**Why files were excluded:**

| Reason | Count | What it means |
|---|---|---|
| `below_min_resolution` | 24,447 | Short side &lt; 512px (thumbnails / reaction pics) |
| `missing_image_file` | 19,581 | Quarantined or never downloaded |
| `too_large` (quarantine) | 13,043 | Long side &gt; 2048px |
| `bad_magic` (quarantine) | 6,576 | Wrong extension — mostly `.jpg` files that are actually PNG |
| `corrupt` | 1 | Truncated JPEG (`214663.jpg`) — the latent-cache crash type |
| `corrupt_image` (manifest build) | 46 | Failed strict re-check during manifest build |
| `missing_dimensions` | 40 | Metadata JSON missing width/height |
| `above_max_resolution` | 5 | Metadata says &gt;2048px but file wasn't quarantined |

Validation is stricter than Pillow `verify()` alone: `image_validate.py` runs EXIF transpose, RGB convert, bucket downscale, full pixel read, and PNG re-encode — the same load path sd-scripts uses before VAE encode.

## R2 bucket structure (`soyjak-training`)

```
images/{id}.{ext}                       # all scraped images (~156K files, legacy individual files)
metadata/{id}.json                      # all metadata JSONs (~167K files, legacy)
manifests/dataset.jsonl                 # full 105K manifest
manifests/dataset_pilot10k.jsonl        # pilot 10K manifest

datasets/soyjak-sdxl-full/             # packaged full dataset (from package_dataset.py)
  shard_manifest.json                  #   shard list + sha256 + counts (9 shards, 38.2 GB)
  shards/shard_0000.tar                #   flat tar: {id}.{ext} + {id}.txt per image
  shards/shard_0001.tar
  ...
  shards/shard_0008.tar

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
| Images | 10,000 | 105,495 |
| image_dir on Lambda | `/home/ubuntu/train_data_pilot` | `/home/ubuntu/train_data` |
| Train config | `config_pilot.toml` | `config.toml` |
| Steps (1×A100, eff. batch 16) | 2,000 (~3 epochs) | 12,000 (~1.8 epochs) |
| R2 model prefix | `models/soyjak-lora-sdxl-pilot` | `models/soyjak-lora-sdxl` |
| Expected time (1×A100) | ~1-2 hrs | ~20-30 hrs total |

### Timing breakdown (1× A100, full run)

| Phase | ~Duration | Notes |
|---|---|---|
| Data pull | 30–60 min | Network-bound; same on any GPU |
| Bad-image check + prune | 10–30 min | Parallel scan + delete; run `check_images.sh` anytime |
| Latent cache | 5–8 hrs | Single-GPU VAE encode + disk write; ~105k images |
| Training (12k steps) | 12–18 hrs | batch 4 × grad accum 4, effective batch 16 |
| **Total** | **~20–30 hrs** | Slow but stable — run in tmux, detach, come back |

## Instance selection

**Use: Lambda Labs → 1× A100 40GB SXM → plain Ubuntu 22.04**

Avoid 2× H100 SXM for now — Lambda Stack has a driver/Fabric Manager bug that leaves
`Fabric State: In Progress` and blocks CUDA init (error 802), and even plain Ubuntu
multi-GPU hosts have been unreliable enough to waste balance on restarts.

**Launch checklist**

1. Instance type: **gpu_1x_a100_sxm4** (or equivalent 1× A100 40GB)
2. Image: **Ubuntu 22.04** (plain — not Lambda Stack)
3. SSH key: `soyjak-training-new`
4. Disk: ≥ 150 GB (images ~38 GB + latent cache ~35 GB + checkpoints + headroom)

After launching, sanity-check **before** running setup:

```bash
nvidia-smi -L   # expect: GPU 0: NVIDIA A100-SXM4-40GB
```

## Training config key values (1× A100, full run)

| Setting | Value | Notes |
|---|---|---|
| `train_batch_size` | 4 | per step; fits 40 GB VRAM with grad checkpoint |
| `gradient_accumulation_steps` | 4 | effective batch = 16 |
| `max_train_steps` | 12,000 | ~1.8 epochs over 105,495 images |
| `learning_rate` / `unet_lr` | 1.5e-4 | unchanged from 2× H100 recipe (same eff. batch) |
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
- Instance: Lambda Labs 1× A100 at `<instance-ip>` — switched to 2× H100 after slow latent cache
- Issue: corrupt JPEG in shards crashed first train attempt; 2× H100 path then kept failing on fabric/CUDA
- Fix: `prune_bad_images.py` + `check_images.py` added

### Session 4 — Back to 1× A100 (June 17 2026)
- Decision: re-center project on 1× A100 40GB — slower but avoids multi-GPU instability burning balance
- Config: batch 4 + grad accum 4 (effective batch 16 preserved), `NUM_GPUS=1` default

### Session 5 — Full dataset rebuild + R2 re-upload (June 19–20 2026)
- Emptied R2 bucket and rebuilt the full dataset from scratch with strict validation
- `validate_images.py --quarantine` on all 156,394 local files → 19,620 quarantined, 136,950 remain
- `build_dataset.py --validate-images` → **105,495** images in manifest (was 123,778 before strict pipeline)
- `package_dataset.py --mode full` → 9 shards, 38.21 GB, **0 failed validation** during packaging
- Uploaded to `datasets/soyjak-sdxl-full/` + `manifests/dataset.jsonl`
- Cleaned up 7 stale shards (`shard_0009`–`shard_0015`) left over from an earlier packaging run in `data/package/full/shards/` — `upload_to_r2` uploads every `*.tar` in that dir, but `pull_data.sh` only downloads shards listed in `shard_manifest.json`
- **Next:** launch fresh 1× A100 instance → `pull_data.sh` → `train_lora.sh`

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
10. **Corrupt/truncated images crash training** — sd-scripts dies on bad files during latent caching. Validation now runs the full training load path (EXIF transpose, RGB convert, bucket downscale, pixel read, re-encode), not just Pillow `verify()`. Use `validate_images.py` locally before packaging; `build_dataset.py --validate-images` excludes bad files from the manifest; `package_dataset.py` re-checks by default; `prune_bad_images.py` is a last-resort safety net on Lambda. `check_images.py` runs a parallel pre-flight scan and a post-prune `--fail` verify before training starts. Override max size with `MAX_LONG_SIDE=0` or `CHECK_IMAGES=0`. Rebuild manifests after quarantining bad files.
11. **Stale shards uploaded to R2** — `package_dataset.py` writes new shards into `data/package/full/shards/` but does not delete old `shard_*.tar` files. `upload_to_r2` then uploads every `*.tar` in that directory. `pull_data.sh` is safe (it reads `shard_manifest.json`), but stale shards waste R2 storage. **Fix:** before re-packaging, `rm data/package/full/shards/shard_*.tar` (or delete orphans manually). To remove stale objects from R2 after the fact, delete keys under `datasets/soyjak-sdxl-full/shards/` that are not listed in `shard_manifest.json`.

---

## Command cheat-sheet

### Local — one-time / when manifest or images change
```bash
cd /path/to/soybrary

# Recommended full rebuild flow (June 2026):

# 1. Scan ALL local images, quarantine bad ones (~20 min for 156K images):
.venv/bin/python validate_images.py --quarantine --fail-on-bad
# Report: data/manifests/bad_images.json

# 2. Rebuild manifest with strict per-file validation (~2.5 hrs):
.venv/bin/python build_dataset.py --validate-images

# 3. Upload manifest to R2:
.venv/bin/python r2_sync.py upload-file --src data/manifests/dataset.jsonl --key manifests/dataset.jsonl

# 4. Clear old shards before re-packaging (avoids uploading stale shard_*.tar to R2):
rm -f data/package/full/shards/shard_*.tar

# 5. Package + upload (full run, ~38 GB / 9 shards; ~2.5 hrs pack + ~30 min upload):
.venv/bin/python package_dataset.py --mode full

# Package pilot subset only (~10 GB):
.venv/bin/python package_dataset.py --mode pilot

# Package only (no upload), e.g. to inspect shards first:
.venv/bin/python package_dataset.py --mode full --no-upload

# Upload already-packaged shards (skip re-packaging):
.venv/bin/python package_dataset.py --mode full --upload-only

# Optional: scan only manifest-listed files (faster if you trust the dir):
.venv/bin/python validate_images.py --manifest data/manifests/dataset.jsonl --quarantine
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
nvidia-smi -L   # expect: GPU 0: NVIDIA A100-SXM4-40GB
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
# Expected: "1 GPU(s), distributed_type='NO', bf16" + "CUDA OK: NVIDIA A100-SXM4-40GB"

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

# Optional: parallel bad-image scan without starting training
MODE=full bash train/check_images.sh

MODE=full bash train/train_lora.sh     # ~20-30 hrs on 1× A100 (latent cache + training)

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
