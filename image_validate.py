#!/usr/bin/env python3
"""
Shared image validation for Soybrary training data.

Pillow verify()+load() alone misses files that still crash kohya sd-scripts during
latent caching. This module mirrors the load path more closely:

  open -> verify -> load -> exif_transpose -> convert("RGB") -> bucket downscale
  -> read all pixels -> lossless re-encode

Used locally before packaging/upload and on Lambda via train/prune_bad_images.py.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None  # type: ignore[assignment,misc]
    ImageOps = None  # type: ignore[assignment,misc]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# Matches train_lora.sh dataset.toml bucket settings.
DEFAULT_MAX_LONG_SIDE = 2048
DEFAULT_MIN_BUCKET_RESO = 512
DEFAULT_TARGET_RESO = 1024


@dataclass(frozen=True)
class ImageCheckResult:
    ok: bool
    reason: str | None = None
    detail: str = ""


def validate_magic_bytes(data: bytes, ext: str) -> bool:
    """Return True when file header matches the declared extension."""
    ext = ext.lower().lstrip(".")
    if ext == "png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if ext in {"jpg", "jpeg"}:
        return data.startswith(b"\xff\xd8\xff")
    if ext == "gif":
        return data.startswith(b"GIF87a") or data.startswith(b"GIF89a")
    if ext == "webp":
        return data.startswith(b"RIFF") and len(data) > 12 and data[8:12] == b"WEBP"
    if ext == "bmp":
        return data.startswith(b"BM")
    return False


def _require_pillow() -> None:
    if Image is None or ImageOps is None:
        raise RuntimeError("Pillow is required for image validation")


def _simulate_training_load(
    img: Image.Image,
    *,
    max_long_side: int,
    min_bucket_reso: int,
    target_reso: int,
) -> Image.Image:
    """
    Apply the same transforms sd-scripts performs before VAE encode.

    bucket_no_upscale=True: only downscale when the long side exceeds max bucket.
    """
    img = ImageOps.exif_transpose(img)
    rgb = img.convert("RGB")
    w, h = rgb.size
    if w < 1 or h < 1:
        raise ValueError(f"zero dimensions after load: {w}x{h}")

    long_side = max(w, h)

    scale = 1.0
    if long_side > max_long_side:
        scale = max_long_side / long_side
    elif long_side > target_reso:
        # Downscale toward the training resolution bucket without upscaling.
        scale = target_reso / long_side

    if scale < 1.0:
        nw = max(min_bucket_reso, int(round(w * scale)))
        nh = max(min_bucket_reso, int(round(h * scale)))
        rgb = rgb.resize((nw, nh), Image.Resampling.LANCZOS)

    # Force a full pixel read (catches truncated decoders that passed verify()).
    rgb.load()
    _ = rgb.tobytes()

    # Lossless round-trip catches subtle encoder/decoder mismatches.
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    if not buf.getvalue():
        raise ValueError("re-encode produced empty output")
    return rgb


def check_image_bytes(
    data: bytes,
    ext: str,
    *,
    max_long_side: int = DEFAULT_MAX_LONG_SIDE,
    min_bucket_reso: int = DEFAULT_MIN_BUCKET_RESO,
    target_reso: int = DEFAULT_TARGET_RESO,
) -> ImageCheckResult:
    """Validate in-memory image bytes the way training will consume them."""
    _require_pillow()

    if not data:
        return ImageCheckResult(False, "empty_file", "0 bytes")

    ext = ext.lower().lstrip(".")
    if ext not in {"jpg", "jpeg", "png", "webp", "gif", "bmp"}:
        return ImageCheckResult(False, "unsupported_ext", ext)

    if not validate_magic_bytes(data, ext):
        return ImageCheckResult(False, "bad_magic", f".{ext}")

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            w, h = img.size
            if max_long_side > 0 and max(w, h) > max_long_side:
                return ImageCheckResult(
                    False,
                    "too_large",
                    f"long side {max(w, h)}px exceeds max {max_long_side}px",
                )
            img.load()
            _simulate_training_load(
                img,
                max_long_side=max_long_side,
                min_bucket_reso=min_bucket_reso,
                target_reso=target_reso,
            )
        return ImageCheckResult(True)
    except ValueError as exc:
        msg = str(exc)
        if "exceeds max" in msg:
            return ImageCheckResult(False, "too_large", msg)
        if "zero dimensions" in msg:
            return ImageCheckResult(False, "zero_size", msg)
        return ImageCheckResult(False, "training_load", msg)
    except Exception as exc:
        return ImageCheckResult(False, "corrupt", str(exc))


def check_image_path(
    path: Path,
    *,
    max_long_side: int = DEFAULT_MAX_LONG_SIDE,
    min_bucket_reso: int = DEFAULT_MIN_BUCKET_RESO,
    target_reso: int = DEFAULT_TARGET_RESO,
) -> ImageCheckResult:
    """Validate an on-disk image file."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return ImageCheckResult(False, "unreadable", str(exc))
    return check_image_bytes(
        data,
        path.suffix,
        max_long_side=max_long_side,
        min_bucket_reso=min_bucket_reso,
        target_reso=target_reso,
    )
