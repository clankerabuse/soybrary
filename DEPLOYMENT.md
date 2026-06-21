# Soybrary SDXL LoRA — Deployment (R2 + Hugging Face)

Companion to [TRAINING.md](TRAINING.md). Covers publishing trained LoRA weights and running inference via Hugging Face.

## What gets deployed

After training on Lambda, artifacts live in `/home/ubuntu/out/` as kohya **`.safetensors`** files (~231 MB each). The canonical backup is **Cloudflare R2**; **Hugging Face** is for distribution and the browser demo.

| Artifact | R2 prefix | HF repo (current) |
|---|---|---|
| Full run final | `models/soyjak-lora-sdxl/soyjak-lora-sdxl.safetensors` | [ChineseWhiteGuy/soy_diffusion](https://huggingface.co/ChineseWhiteGuy/soy_diffusion) |
| Full run checkpoints | `models/soyjak-lora-sdxl/soyjak-lora-sdxl-step*.safetensors` | (R2 only unless you upload them) |
| Pilot run | `models/soyjak-lora-sdxl-pilot/` | separate repo if you create one |

Published HF weight name: **`soy_diffusion.safetensors`** (renamed from `soyjak-lora-sdxl.safetensors` for the public repo).

## Key design decisions (already made, don't revisit)

- **R2 is the source of truth.** HF is a mirror + demo. Always push to R2 from Lambda before terminating the instance (`train/push_model.sh`).
- **Training base ≠ Space inference base.** Training uses the single-file fixed-VAE checkpoint (`bdsqlsz/stable-diffusion-xl-base-1.0_fixvae_fp16`). The HF Space uses `stabilityai/stable-diffusion-xl-base-1.0` + `madebyollin/sdxl-vae-fp16-fix` because diffusers loads that reliably; visually equivalent for demos.
- **LoRA format is kohya sd-scripts** (`lora_unet_*`, `lora_te1_*`, `lora_te2_*` with `.lora_down`/`.lora_up`). Not a native diffusers/PEFT export.
- **HF Space loads UNet LoRA only** via `lora_state_dict` + `load_lora_into_unet`. Full TE+UNet works in Forge/A1111/Comfy with the raw `.safetensors`; diffusers' PEFT text-encoder path breaks on kohya TE keys.
- **LoRA is bundled inside the Space repo** (`hf_space/soy_diffusion.safetensors`) so the demo works even when the model repo is **private**. Re-upload the weight when you publish a new checkpoint.
- **Model repo can stay private.** Space does not need to pull from it if the weight is bundled. Alternatively: make the model public, or add `HF_TOKEN` as a Space secret.

## Repository structure (deployment-related)

```
.env.example            # HF_TOKEN, HF_REPO_ID, HF_MODEL_NAME, R2_* …
.env                    # gitignored

train/push_model.sh     # R2 upload (default) + optional HF upload (PUSH_HF=1)

hf_space/               # Hugging Face Space source (pushed to *-demo repo)
  app.py                # Gradio SDXL + kohya LoRA loader
  requirements.txt      # pinned Gradio / pydantic / peft
  README.md             # Space config (YAML front matter — quote python_version!)
  soy_diffusion.safetensors   # bundled LoRA (gitignored locally via hf_space/*.safetensors)

hf_upload/              # scratch dir for HF uploads (gitignored)
```

## Hugging Face assets

| Resource | URL | Purpose |
|---|---|---|
| Model repo | https://huggingface.co/ChineseWhiteGuy/soy_diffusion | Weight + README model card |
| Demo Space | https://huggingface.co/spaces/ChineseWhiteGuy/soy_diffusion-demo | Browser inference (Gradio) |

### `.env` variables

```bash
HF_TOKEN=hf_...                              # Write token (upload + private repo access)
HF_REPO_ID=ChineseWhiteGuy/soy_diffusion     # Model repo
HF_MODEL_NAME=soy_diffusion                  # Display name / local filename convention
```

Space secrets (optional): add **`HF_TOKEN`** under Space Settings → Repository secrets for faster Hub downloads of SDXL base weights.

## Model repo — first-time setup

1. Create a **Model** on huggingface.co (empty repo is fine).
2. Add credentials to `.env` (see `.env.example`).
3. Upload from R2 or Lambda (below).
4. Add a **README.md** model card (usage, base model, prompt style). The repo should explain booru-style tags and variant names as triggers.
5. Set visibility: **Private** while iterating, **Public** when ready.

### Upload from Lambda (after training)

```bash
# On the instance — R2 only (default)
MODE=full bash train/push_model.sh

# R2 + Hugging Face
PUSH_HF=1 HF_REPO_ID=ChineseWhiteGuy/soy_diffusion bash train/push_model.sh
```

### Upload from local machine (R2 → HF)

```bash
cd /path/to/soybrary
source .venv/bin/activate   # needs huggingface_hub

# Pull final checkpoint from R2
.venv/bin/python r2_sync.py download-file \
  --key models/soyjak-lora-sdxl/soyjak-lora-sdxl.safetensors \
  --dest ./hf_upload/soy_diffusion.safetensors

# Push to model repo
set -a && source .env && set +a
.venv/bin/python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    "hf_upload/soy_diffusion.safetensors",
    "soy_diffusion.safetensors",
    os.environ["HF_REPO_ID"],
    repo_type="model",
)
PY
```

Only upload the **final** checkpoint for public release unless you explicitly want step checkpoints on HF (~231 MB each).

## Demo Space — setup and updates

The Space is a separate repo (`ChineseWhiteGuy/soy_diffusion-demo`), not the model repo. A model page does **not** run inference by itself — you need a Space (or local UI).

### Hardware

SDXL requires a GPU. In Space **Settings → Hardware**, pick e.g. **T4 small** (paid, ~cents/hr). CPU tier will not run inference.

Wait until status is **Running** (not Building/Starting) before clicking Generate. First run downloads ~7 GB of SDXL components; later runs reuse cache on the same machine.

### Push Space changes from local

```bash
cd /path/to/soybrary
set -a && source .env && set +a

# App + config only
.venv/bin/python - <<'PY'
from huggingface_hub import upload_folder
upload_folder(
    "hf_space",
    "ChineseWhiteGuy/soy_diffusion-demo",
    repo_type="space",
    commit_message="Update Space",
)
PY
```

When you train a new LoRA, copy the new weight into the Space bundle and push:

```bash
cp hf_upload/soy_diffusion.safetensors hf_space/soy_diffusion.safetensors
# then upload_folder or upload_file for soy_diffusion.safetensors + app.py
```

### Space dependency pins (do not casually upgrade)

| Package | Pin | Why |
|---|---|---|
| `gradio` | `5.12.0` | Matches `sdk_version` in Space README |
| `pydantic` | `2.10.6` | Gradio 5.12 + pydantic 2.11+ crashes API schema (`TypeError: bool is not iterable`) |
| `peft` | `>=0.11.0` | Required by diffusers `load_lora_weights` / `load_lora_into_unet` |
| Python | `"3.10"` in README | **Must be quoted** — `3.10` unquoted parses as float `3.1` and HF tries to build Python 3.1 |

Space README front matter example:

```yaml
---
title: soy_diffusion
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
python_version: "3.10"
---
```

## Inference — how to run the LoRA

### Hugging Face Space (easiest)

1. Open https://huggingface.co/spaces/ChineseWhiteGuy/soy_diffusion-demo
2. Ensure GPU hardware is enabled and status is **Running**
3. Prompt with booru-style tags, e.g. `feraljak, screaming, snail, pink_hair, 4chan`
4. LoRA strength ~0.7–1.0, 28 steps, CFG 7, 1024×1024

### Automatic1111 / Forge (full kohya LoRA — UNet + TE)

1. Base: `sd_xl_base_1.0_fixvae_fp16.safetensors`
2. LoRA: `soy_diffusion.safetensors` in `models/Lora/`
3. Prompt: `<lora:soy_diffusion:0.85>` plus tags

### ComfyUI

Load fixed-VAE SDXL checkpoint + LoRA node with `soy_diffusion.safetensors`.

### Python (diffusers — UNet LoRA path only)

Same approach as `hf_space/app.py`: `lora_state_dict` with `unet_config`, then `load_lora_into_unet`. Do not use `from_single_file` on the bdsqlsz training checkpoint in recent diffusers/transformers stacks without version pinning.

### Download weight from HF

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="ChineseWhiteGuy/soy_diffusion",
    filename="soy_diffusion.safetensors",
    token="hf_...",  # required if repo is private
)
```

## Model card notes (for README on HF)

Include at minimum:

| Field | Value |
|---|---|
| Base model (training) | `bdsqlsz/stable-diffusion-xl-base-1.0_fixvae_fp16` |
| LoRA type | kohya `networks.lora`, rank 32 / alpha 16 |
| Training | ~105k images, 12k steps, effective batch 16 |
| Prompting | No fixed trigger — variant names in captions (`feraljak`, `chudjak`, …) |
| License | Your choice (set on repo) |

## Known issues and fixes (HF / Space)

1. **`from_pretrained("bdsqlsz/...")` 404** — That repo is a single `.safetensors`, not a diffusers pipeline. Use `from_single_file` locally with pinned versions, or use stabilityai SDXL + fixed VAE in the Space.

2. **`from_single_file` → `CLIPTextModel has no attribute text_model`** — transformers/diffusers version mismatch. Space avoids this by using the standard SDXL Hub repo.

3. **`PEFT backend is required`** — Install `peft` in Space `requirements.txt`.

4. **401 on private model repo from Space** — Space has no token. Bundle LoRA in the Space repo, make model public, or set `HF_TOKEN` secret.

5. **`IndexError` in `get_peft_kwargs` on TE load** — kohya TE keys vs diffusers PEFT. Space fix: UNet-only via `load_lora_into_unet`.

6. **`Target modules {'7.1.proj_in', ...} not found`** — kohya SGM keys converted without `_maybe_map_sgm_blocks_to_diffusers`. Use `StableDiffusionXLPipeline.lora_state_dict(..., unet_config=pipe.unet.config)`, not manual `_convert_non_diffusers_lora_to_diffusers` alone.

7. **`not enough values to unpack (expected 3, got 2)`** — Older diffusers returns `(state_dict, network_alphas)` without metadata. Handle both tuple lengths (see `hf_space/app.py`).

8. **Gradio "no API found"** — App still building, or pydantic/Gradio crash on startup. Wait for **Running**, hard-refresh, check Logs.

9. **`python_version: 3.10` builds Python 3.1** — YAML parses bare `3.10` as float `3.1`. Always quote: `python_version: "3.10"`.

10. **Gradio 4.44 + new `huggingface_hub`** — `ImportError: cannot import name 'HfFolder'`. Use Gradio 5.x on Python 3.10, not Gradio 4 on Python 3.13.

## Session history

### HF deploy — soy_diffusion (June 20–21 2026)

- Model repo: `ChineseWhiteGuy/soy_diffusion` (private), weight `soy_diffusion.safetensors`
- Space: `ChineseWhiteGuy/soy_diffusion-demo` (Gradio, T4)
- Source copied from R2 final checkpoint `models/soyjak-lora-sdxl/soyjak-lora-sdxl.safetensors`
- Space debugging: Gradio/pydantic/Python YAML pins, private-repo auth, kohya→diffusers LoRA loading (SGM block map + UNet-only)
- Local paths gitignored: `hf_upload/`, `hf_space/*.safetensors`

---

## Command cheat-sheet

### List models on R2

```bash
.venv/bin/python r2_sync.py list --prefix models/soyjak-lora-sdxl
```

### Download full checkpoint locally

```bash
.venv/bin/python r2_sync.py download --prefix models/soyjak-lora-sdxl --dest ./full_lora
```

### Re-publish model + refresh Space bundle

```bash
cd /path/to/soybrary
.venv/bin/python r2_sync.py download-file \
  --key models/soyjak-lora-sdxl/soyjak-lora-sdxl.safetensors \
  --dest ./hf_space/soy_diffusion.safetensors

set -a && source .env && set +a
.venv/bin/python - <<'PY'
import os
from huggingface_hub import HfApi, upload_folder
token = os.environ["HF_TOKEN"]
model = os.environ["HF_REPO_ID"]
space = "ChineseWhiteGuy/soy_diffusion-demo"
api = HfApi(token=token)
api.upload_file(
    "hf_space/soy_diffusion.safetensors", "soy_diffusion.safetensors",
    model, repo_type="model",
)
upload_folder("hf_space", space, repo_type="space", commit_message="Update LoRA + Space")
print("Done:", f"https://huggingface.co/{model}", f"https://huggingface.co/spaces/{space}")
PY
```

### Install local HF tooling

```bash
pip install "huggingface_hub>=0.23.0"
```

For local diffusers inference (experimental), you also need `torch`, `diffusers`, `peft` — a 6 GB GTX 1660 Ti is too tight for 1024 SDXL; use the Space or Forge instead.
