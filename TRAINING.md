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

## Current state (as of this session)

- Branch: `test/sdxl-lora-pipeline` (3 commits ahead of main)
- Pilot is **actively training** on a Lambda Labs A100 instance at `<instance-ip>`
- SSH key: `~/.ssh/lambda-training.pem`
- As of last check: still in **latent caching phase** (`46/9994`), estimated ~7-8 hrs for caching, then ~1-2 hrs training
- tmux session name: `training`
- To reconnect and check progress:
  ```bash
  ssh -i ~/.ssh/lambda-training.pem ubuntu@<instance-ip>
  tmux attach -t training
  ```

## When training finishes (TODO)

```bash
# On Lambda instance:
MODE=pilot bash train/push_model.sh     # push LoRA to R2 BEFORE terminating

# Then terminate the instance in Lambda console.

# Locally, download the LoRA to test:
.venv/bin/python r2_sync.py download --prefix models/soyjak-lora-sdxl-pilot --dest ./pilot_lora
```

Test in ComfyUI / A1111 with prompts like `chudjak, open_mouth, glasses` or `cobson, smug`.

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
