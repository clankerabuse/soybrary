#!/usr/bin/env python3
"""
package_dataset.py - Phase 2 of the Soybrary -> SDXL LoRA pipeline.

Reads a JSONL manifest (from build_dataset.py) and produces tar shards that
each contain flat {id}.{ext} images + {id}.txt caption sidecars. The shards
are written locally to data/package/<mode>/shards/ and then uploaded to R2
under datasets/soyjak-sdxl-<mode>/ in one shot.

On the Lambda instance, pull_data.sh downloads the shards and extracts them
into a single flat directory — the DreamBooth-style image_dir that sd-scripts
expects: one image next to one .txt caption file, nothing else.

Usage (local):
    # Package + upload the full dataset:
    python package_dataset.py --mode full

    # Package + upload the pilot subset:
    python package_dataset.py --mode pilot

    # Package only (no upload), or upload only (shards already on disk):
    python package_dataset.py --mode full --no-upload
    python package_dataset.py --mode full --upload-only

    # Override shard size (default 4 GB uncompressed):
    python package_dataset.py --mode full --shard-size-gb 4

R2 layout produced:
    datasets/soyjak-sdxl-full/
        shard_manifest.json
        shards/shard_0000.tar
        shards/shard_0001.tar
        ...
    datasets/soyjak-sdxl-pilot/
        shard_manifest.json
        shards/shard_0000.tar
        ...
"""

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

try:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config
except ImportError:
    sys.exit("ERROR: boto3 not installed. pip install -r requirements.txt")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_DIR = DATA_DIR / "metadata"
MANIFESTS_DIR = DATA_DIR / "manifests"
PACKAGE_DIR = DATA_DIR / "package"

TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024 * 1024,
    multipart_chunksize=64 * 1024 * 1024,
    max_concurrency=8,
    use_threads=True,
)


# ---------------------------------------------------------------------------
# Caption helpers (mirrors gen_captions.py logic so metadata is baked in)
# ---------------------------------------------------------------------------

def _dedup_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        tag = str(item).strip()
        if not tag:
            continue
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            out.append(tag)
    return out


def _build_caption_from_meta(meta):
    parts = []
    parts.extend(meta.get("variants") or [])
    parts.extend(meta.get("subvariants") or [])
    parts.extend(meta.get("tags") or [])
    return ", ".join(_dedup_preserve_order(parts))


# ---------------------------------------------------------------------------
# R2 helpers
# ---------------------------------------------------------------------------

def _get_env(name):
    val = os.environ.get(name)
    if not val:
        sys.exit(f"ERROR: missing required env var {name} (set it in .env)")
    return val


def _make_r2_client():
    endpoint = os.environ.get("R2_ENDPOINT")
    account_id = os.environ.get("R2_ACCOUNT_ID")
    if not endpoint and account_id:
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    if not endpoint:
        sys.exit("ERROR: set R2_ENDPOINT or R2_ACCOUNT_ID in .env")

    cfg_kwargs = {
        "region_name": "auto",
        "retries": {"max_attempts": 5, "mode": "standard"},
    }
    try:
        Config(request_checksum_calculation="when_required")
    except TypeError:
        pass
    else:
        cfg_kwargs["request_checksum_calculation"] = "when_required"
        cfg_kwargs["response_checksum_validation"] = "when_required"

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=_get_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_get_env("R2_SECRET_ACCESS_KEY"),
        config=Config(**cfg_kwargs),
    )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def _load_manifest(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

def package(manifest_path, out_dir, shard_size_gb, image_dir_on_instance):
    """Build tar shards; return shard_manifest dict."""
    records = _load_manifest(manifest_path)
    if not records:
        sys.exit("ERROR: manifest is empty.")
    print(f"Loaded {len(records)} records from {manifest_path}")

    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    shard_size_bytes = int(shard_size_gb * 1024 ** 3)
    shard_list = []
    shard_idx = 0
    cur_tar = None
    cur_tar_path = None
    cur_bytes = 0
    cur_count = 0
    written = 0
    missing_img = 0
    missing_meta = 0

    def _open_shard(idx):
        p = shards_dir / f"shard_{idx:04d}.tar"
        return tarfile.open(p, "w"), p

    def _close_shard(tar, path, count):
        tar.close()
        sz = path.stat().st_size
        print(f"  shard {path.name}: {count} images, {sz / 1e9:.2f} GB — hashing...", flush=True)
        digest = _sha256(path)
        shard_list.append({
            "name": path.name,
            "images": count,
            "bytes": sz,
            "sha256": digest,
        })

    cur_tar, cur_tar_path = _open_shard(shard_idx)
    t_start = time.monotonic()

    for i, rec in enumerate(records):
        img_path = IMAGES_DIR / rec["file"]
        if not img_path.exists():
            missing_img += 1
            continue

        # Build caption: prefer manifest caption field, fall back to metadata file.
        if rec.get("caption"):
            caption = rec["caption"]
        else:
            meta_path = METADATA_DIR / f"{rec['id']}.json"
            if not meta_path.exists():
                missing_meta += 1
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                caption = _build_caption_from_meta(meta)
            except Exception:
                missing_meta += 1
                continue

        if not caption:
            missing_meta += 1
            continue

        img_bytes = img_path.read_bytes()
        caption_bytes = caption.encode("utf-8")
        stem = str(rec["id"])

        # Roll to a new shard if this image would exceed target size.
        if cur_bytes and (cur_bytes + len(img_bytes)) > shard_size_bytes:
            _close_shard(cur_tar, cur_tar_path, cur_count)
            shard_idx += 1
            cur_tar, cur_tar_path = _open_shard(shard_idx)
            cur_bytes = 0
            cur_count = 0

        # image
        img_info = tarfile.TarInfo(name=f"{stem}.{rec['ext']}")
        img_info.size = len(img_bytes)
        cur_tar.addfile(img_info, io.BytesIO(img_bytes))

        # caption sidecar
        txt_info = tarfile.TarInfo(name=f"{stem}.txt")
        txt_info.size = len(caption_bytes)
        cur_tar.addfile(txt_info, io.BytesIO(caption_bytes))

        cur_bytes += len(img_bytes) + len(caption_bytes)
        cur_count += 1
        written += 1

        if (i + 1) % 10000 == 0:
            elapsed = time.monotonic() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(records) - (i + 1)) / rate / 60 if rate > 0 else 0
            print(
                f"  packaged {i+1}/{len(records)}  shard={shard_idx}  "
                f"elapsed={elapsed/60:.1f}m  eta={eta:.1f}m",
                flush=True,
            )

    if cur_count:
        _close_shard(cur_tar, cur_tar_path, cur_count)
    else:
        cur_tar.close()
        cur_tar_path.unlink(missing_ok=True)

    total_bytes = sum(s["bytes"] for s in shard_list)
    manifest_out = {
        "image_dir": image_dir_on_instance,
        "total_images": written,
        "missing_images": missing_img,
        "missing_metadata": missing_meta,
        "num_shards": len(shard_list),
        "total_bytes": total_bytes,
        "shards": shard_list,
    }
    shard_manifest_path = out_dir / "shard_manifest.json"
    shard_manifest_path.write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")

    print("\n=== PACKAGE SUMMARY ===")
    print(f"Images packaged:     {written}")
    print(f"Missing images:      {missing_img}")
    print(f"Missing metadata:    {missing_meta}")
    print(f"Shards:              {len(shard_list)}")
    print(f"Total size:          {total_bytes / 1e9:.2f} GB")
    print(f"Output dir:          {out_dir}")

    return manifest_out


# ---------------------------------------------------------------------------
# R2 upload
# ---------------------------------------------------------------------------

def upload_to_r2(out_dir, r2_prefix):
    """Upload shard_manifest.json + all shards to R2."""
    client = _make_r2_client()
    bucket = _get_env("R2_BUCKET_NAME")
    prefix = r2_prefix.strip("/")

    files_to_upload = [out_dir / "shard_manifest.json"] + sorted((out_dir / "shards").glob("*.tar"))
    print(f"\nUploading {len(files_to_upload)} file(s) -> r2://{bucket}/{prefix}/")

    t_start = time.monotonic()
    for p in files_to_upload:
        if p.name == "shard_manifest.json":
            key = f"{prefix}/shard_manifest.json"
        else:
            key = f"{prefix}/shards/{p.name}"
        size_mb = p.stat().st_size / 1e6
        print(f"  -> {key}  ({size_mb:.1f} MB)", flush=True)
        client.upload_file(str(p), bucket, key, Config=TRANSFER_CONFIG)

    elapsed = time.monotonic() - t_start
    print(f"Upload complete in {elapsed/60:.1f}m.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Package dataset into tar shards and upload to Cloudflare R2."
    )
    ap.add_argument(
        "--mode", choices=["pilot", "full"], default="full",
        help="Which manifest to use: pilot (~10K images) or full (~124K). Default: full.",
    )
    ap.add_argument(
        "--manifest", type=Path, default=None,
        help="Override manifest path (default: data/manifests/dataset[_pilot10k].jsonl).",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Override local output dir (default: data/package/<mode>).",
    )
    ap.add_argument(
        "--shard-size-gb", type=float, default=4.0,
        help="Max uncompressed bytes per shard (default: 4 GB).",
    )
    ap.add_argument(
        "--r2-prefix", type=str, default=None,
        help="R2 key prefix (default: datasets/soyjak-sdxl-<mode>).",
    )
    ap.add_argument(
        "--no-upload", action="store_true",
        help="Package only; do not upload to R2.",
    )
    ap.add_argument(
        "--upload-only", action="store_true",
        help="Skip packaging; upload existing shards from --out-dir.",
    )
    ap.add_argument(
        "--image-dir", type=str, default=None,
        help="image_dir path on the Lambda instance (used in shard_manifest.json). "
             "Default: /home/ubuntu/train_data_pilot or /home/ubuntu/train_data.",
    )
    args = ap.parse_args()

    # Resolve mode-specific defaults
    if args.mode == "pilot":
        default_manifest = MANIFESTS_DIR / "dataset_pilot10k.jsonl"
        default_image_dir = "/home/ubuntu/train_data_pilot"
        default_r2_prefix = "datasets/soyjak-sdxl-pilot"
    else:
        default_manifest = MANIFESTS_DIR / "dataset.jsonl"
        default_image_dir = "/home/ubuntu/train_data"
        default_r2_prefix = "datasets/soyjak-sdxl-full"

    manifest_path = args.manifest or default_manifest
    out_dir = args.out_dir or (PACKAGE_DIR / args.mode)
    r2_prefix = args.r2_prefix or default_r2_prefix
    image_dir = args.image_dir or default_image_dir

    if args.upload_only and args.no_upload:
        sys.exit("ERROR: --upload-only and --no-upload are mutually exclusive.")

    if not args.upload_only:
        if not manifest_path.exists():
            sys.exit(f"ERROR: manifest not found: {manifest_path}\nRun build_dataset.py first.")
        package(manifest_path, out_dir, args.shard_size_gb, image_dir)

    if not args.no_upload:
        upload_to_r2(out_dir, r2_prefix)


if __name__ == "__main__":
    main()
