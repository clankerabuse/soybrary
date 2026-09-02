# Soybrary SDXL Fine-Tune — Deployment (R2 + Hugging Face)

Companion to [TRAINING.md](TRAINING.md). Covers publishing the trained **full SDXL checkpoint** and running inference via Hugging Face.

## What gets deployed

After training on Lambda, artifacts live in `/home/ubuntu/out/` as kohya **`.safetensors`** full SDXL checkpoints (**~6.5 GB each**). The canonical backup is **Cloudflare R2**; **Hugging Face** is for distribution and the browser demo.

| Artifact | R2 prefix | HF repo (current) |
|---|---|---|
| Full fine-tune final | `models/soyjak-sdxl-ft/soyjak-sdxl-ft.safetensors` | re-publish to [ChineseWhiteGuy/soy_diffusion](https://huggingface.co/ChineseWhiteGuy/soy_diffusion) after the FT run |
| Full fine-tune checkpoints | `models/soyjak-sdxl-ft/soyjak-sdxl-ft-step*.safetensors` | (R2 only unless you upload them) |
| Pilot fine-tune | `models/soyjak-sdxl-ft-pilot/` | separate repo if you create one |
| v1 LoRA (too general) | `models/soyjak-lora-sdxl/soyjak-lora-sdxl.safetensors` | previous public weight |

Published HF weight name: **`soy_diffusion.safetensors`** (renamed from `soyjak-sdxl-ft.safetensors` for the public repo). This is now a **full SDXL checkpoint**, not a LoRA.

## Key design decisions (already made, don't revisit)

- **R2 is the source of truth.** HF is a mirror + demo. Always push to R2 from Lambda before terminating the instance (`train/push_model.sh`).
- **The published weight IS the model.** Load it as the SDXL checkpoint (`from_single_file` / A1111 checkpoint / Comfy checkpoint). Do not stack it on `stabilityai/stable-diffusion-xl-base-1.0` as a LoRA.
- **Space bundles the checkpoint** (`hf_space/soy_diffusion.safetensors`, ~6.5 GB, git-LFS) so the demo works when the model repo is private. Re-upload when you publish a new checkpoint.
- **Model repo can stay private.** Space does not need to pull from it if the weight is bundled. Alternatively: make the model public, or add `HF_TOKEN` as a Space secret.

## Repository structure (deployment-related)

```
.env.example            # HF_TOKEN, HF_REPO_ID, HF_MODEL_NAME, R2_* …
.env                    # gitignored

train/push_model.sh     # R2 upload (default) + optional HF upload (PUSH_HF=1)

hf_space/               # Hugging Face Space source (pushed to *-demo repo)
  app.py                # Gradio SDXL from_single_file loader
  requirements.txt      # pinned Gradio / pydantic
  README.md             # Space config (YAML front matter — quote python_version!)
  soy_diffusion.safetensors   # bundled full SDXL checkpoint (gitignored locally via hf_space/*.safetensors)

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
4. Add a **README.md** model card (usage, base model, prompt style). Every prompt must start with `soyjak`, then a variant name (`feraljak`, `chudjak`, …) and booru tags.
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
  --key models/soyjak-sdxl-ft/soyjak-sdxl-ft.safetensors \
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

Only upload the **final** checkpoint for public release unless you explicitly want step checkpoints on HF (~6.5 GB each).

## Demo Space — setup and updates

The Space is a separate repo (`ChineseWhiteGuy/soy_diffusion-demo`), not the model repo. A model page does **not** run inference by itself — you need a Space (or local UI).

### Hardware

SDXL requires a GPU. In Space **Settings → Hardware**, pick e.g. **T4 small** (paid, ~cents/hr). CPU tier will not run inference.

Wait until status is **Running** (not Building/Starting) before clicking Generate. First run loads the bundled ~6.5 GB checkpoint.

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

When you train a new checkpoint, copy it into the Space bundle and push:

```bash
cp hf_upload/soy_diffusion.safetensors hf_space/soy_diffusion.safetensors
# then upload_folder or upload_file for soy_diffusion.safetensors + app.py
```

### Space dependency pins (do not casually upgrade)

| Package | Pin | Why |
|---|---|---|
| `gradio` | `5.12.0` | Matches `sdk_version` in Space README |
| `pydantic` | `2.10.6` | Gradio 5.12 + pydantic 2.11+ crashes API schema (`TypeError: bool is not iterable`) |
| `peft` | (optional) | No longer required; Space loads a full checkpoint, not a LoRA |
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

## Inference — how to run the fine-tune

### Hugging Face Space (easiest)

1. Open https://huggingface.co/spaces/ChineseWhiteGuy/soy_diffusion-demo
2. Ensure GPU hardware is enabled and status is **Running**
3. Prompt with `soyjak, <variant>, <tags>`, e.g. `soyjak, feraljak, screaming, snail, pink_hair, 4chan`
4. 28 steps, CFG 7, 1024×1024. Extra objects/settings are fine; a soyjak-less image is a failure.

### Automatic1111 / Forge

1. Put `soy_diffusion.safetensors` in `models/Stable-diffusion/` (it is the checkpoint, not a LoRA)
2. Select it as the SDXL model
3. Prompt: `soyjak, chudjak, ...`  — do **not** use `<lora:...>`

### ComfyUI

Load `soy_diffusion.safetensors` as a checkpoint. Do not attach a LoRA node.

### Python (diffusers)

```python
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch

pipe = StableDiffusionXLPipeline.from_single_file(
    "soy_diffusion.safetensors",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
pipe.to("cuda")
image = pipe("soyjak, cobson, red bicycle, forest", num_inference_steps=28).images[0]
```

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
| Type | kohya `sdxl_train.py` full fine-tune (UNet + TE1 + TE2), not a LoRA |
| Training | ~105k images, 12k steps, effective batch 16, Adafactor 1e-5 |
| Prompting | Start with `soyjak`, then a variant. `1boy` / `portrait` / `wojak` also resolve to soyjaks |
| License | Your choice (set on repo) |

## Known issues and fixes (HF / Space)

1. **`from_pretrained("bdsqlsz/...")` 404** — That repo is a single `.safetensors`, not a diffusers pipeline. Use `from_single_file` locally with pinned versions, or use stabilityai SDXL + fixed VAE in the Space.

2. **`from_single_file` → `CLIPTextModel has no attribute text_model`** — transformers/diffusers version mismatch. Pin the Space `requirements.txt` versions; do not mix a new transformers with an old diffusers.

3. **`PEFT backend is required`** — Obsolete for the fine-tune Space (no LoRA load). Ignore unless you revert to a LoRA demo.

4. **401 on private model repo from Space** — Space has no token. Bundle the checkpoint in the Space repo, make the model public, or set `HF_TOKEN` secret.

5. **Kohya LoRA TE load errors** — Obsolete. The Space loads a full checkpoint via `from_single_file`, not `load_lora_into_unet`.

8. **Gradio "no API found"** — App still building, or pydantic/Gradio crash on startup. Wait for **Running**, hard-refresh, check Logs.

9. **`python_version: 3.10` builds Python 3.1** — YAML parses bare `3.10` as float `3.1`. Always quote: `python_version: "3.10"`.

10. **Gradio 4.44 + new `huggingface_hub`** — `ImportError: cannot import name 'HfFolder'`. Use Gradio 5.x on Python 3.10, not Gradio 4 on Python 3.13.

## Session history

### HF deploy — soy_diffusion (June 20–21 2026)

- Model repo: `ChineseWhiteGuy/soy_diffusion` (private), weight `soy_diffusion.safetensors`
- Space: `ChineseWhiteGuy/soy_diffusion-demo` (Gradio, T4)
- Source copied from R2 final checkpoint `models/soyjak-lora-sdxl/soyjak-lora-sdxl.safetensors`
- Space debugging: Gradio/pydantic/Python YAML pins, private-repo auth, kohya→diffusers LoRA loading (SGM block map + UNet-only)

### HF deploy — full fine-tune (pending retraining)

- After the FT Lambda run, publish `models/soyjak-sdxl-ft/soyjak-sdxl-ft.safetensors` as `soy_diffusion.safetensors` (~6.5 GB, git-LFS)
- Prompt card: `soyjak, <variant>, <tags>`. Extra objects/settings are allowed; a soyjak-less image is a failure. `1boy` / `portrait` / `wojak` should also yield soyjaks.
- Space loads the checkpoint with `from_single_file` + fixed VAE. T4 or better.
- Local paths gitignored: `hf_upload/`, `hf_space/*.safetensors`

---

## Command cheat-sheet

### List models on R2

```bash
.venv/bin/python r2_sync.py list --prefix models/soyjak-sdxl-ft
```

### Download full checkpoint locally

```bash
.venv/bin/python r2_sync.py download --prefix models/soyjak-sdxl-ft --dest ./full_ft
```

### Re-publish model + refresh Space bundle

```bash
cd /path/to/soybrary
.venv/bin/python r2_sync.py download-file \
  --key models/soyjak-sdxl-ft/soyjak-sdxl-ft.safetensors \
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
upload_folder("hf_space", space, repo_type="space", commit_message="Update fine-tune + Space")
print("Done:", f"https://huggingface.co/{model}", f"https://huggingface.co/spaces/{space}")
PY
```

### Install local HF tooling

```bash
pip install "huggingface_hub>=0.23.0"
```

For local diffusers inference (experimental), you also need `torch`, `diffusers`, `peft` — a 6 GB GTX 1660 Ti is too tight for 1024 SDXL; use the Space or Forge instead.
