#!/usr/bin/env python3
"""
r2_sync.py - Phase 3 of the Soybrary -> SDXL LoRA pipeline.

Upload/download files to/from Cloudflare R2 via the S3-compatible API.

Used twice in the workflow:
  1. Locally: upload the packaged dataset (shards + shard_manifest) to R2
     under datasets/soyjak-sdxl-<mode>/ (package_dataset.py does this, but
     you can also call this script directly).
  2. On Lambda: download-archive pulls all shards for a given mode and
     extracts them into a flat training directory in one shot.

R2 / boto3 notes baked in here:
  - endpoint_url = R2_ENDPOINT, region_name = "auto" (required by the SDK,
    ignored by R2).
  - Checksum workaround: starting in boto3/botocore 1.36.0 the SDK sends
    CRC32 checksum trailers by default, which R2 rejects on some operations.
    Two independent defenses:
      (a) requirements pin botocore<1.36.0, where this behavior does not
          exist (the safest fix), and
      (b) if a >=1.36 botocore is somehow installed, we set
          request_checksum_calculation/response_checksum_validation to
          "when_required". That Config key only exists on >=1.36, so we add
          it conditionally to avoid breaking the pinned (<1.36) install.
    The env vars AWS_REQUEST_CHECKSUM_CALCULATION / AWS_RESPONSE_CHECKSUM_VALIDATION
    also work as a global fallback on >=1.36.
  - Multipart transfers via TransferConfig for large shards (R2 single-PUT cap
    is 5 GB; our shards are ~4 GB but multipart is safe and resumable-friendly).

Credentials are read from .env (see .env.example):
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_ENDPOINT

Usage:
  # Upload the whole package dir to a prefix
  python r2_sync.py upload --src data/package/full --prefix datasets/soyjak-sdxl-full

  # Download all shards for a mode and extract into a flat dir (use on Lambda)
  python r2_sync.py download-archive --mode full --dest /home/ubuntu/train_data
  python r2_sync.py download-archive --mode pilot --dest /home/ubuntu/train_data_pilot

  # Download a prefix into a local dir
  python r2_sync.py download --prefix datasets/soyjak-sdxl-full --dest ./pkg

  # Upload a single file (e.g. trained LoRA)
  python r2_sync.py upload-file --src out/soyjak.safetensors --key models/soyjak-lora-sdxl/soyjak.safetensors

  # List a prefix
  python r2_sync.py list --prefix datasets/soyjak-sdxl-full
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

try:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config
except ImportError:
    sys.exit("ERROR: boto3 not installed. pip install -r train/requirements-lock.txt")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv optional; env vars may already be set in the shell.
    pass

# Multipart for anything over 64 MB, 64 MB chunks, a few parallel threads.
TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024 * 1024,
    multipart_chunksize=64 * 1024 * 1024,
    max_concurrency=8,
    use_threads=True,
)


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract tar members under dest, rejecting path traversal (tar-slip)."""
    dest = dest.resolve()
    for member in tf.getmembers():
        target = (dest / member.name).resolve()
        try:
            target.relative_to(dest)
        except ValueError:
            raise ValueError(f"unsafe tar member path: {member.name!r}") from None
    if hasattr(tarfile, "data_filter"):
        tf.extractall(path=dest, filter="data")
    else:
        for member in tf.getmembers():
            tf.extract(member, path=dest, set_attrs=False)


def get_env(name):
    val = os.environ.get(name)
    if not val:
        sys.exit(f"ERROR: missing required env var {name} (set it in .env)")
    return val


def make_client():
    endpoint = os.environ.get("R2_ENDPOINT")
    account_id = os.environ.get("R2_ACCOUNT_ID")
    if not endpoint and account_id:
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    if not endpoint:
        sys.exit("ERROR: set R2_ENDPOINT or R2_ACCOUNT_ID in .env")

    cfg_kwargs = {
        "region_name": "auto",  # required by SDK, ignored by R2
        "retries": {"max_attempts": 5, "mode": "standard"},
    }
    # The checksum Config keys only exist on botocore >= 1.36. On the pinned
    # <1.36 install they are absent (and unneeded), so add them only if the
    # running botocore actually accepts them.
    try:
        Config(request_checksum_calculation="when_required")
    except TypeError:
        pass  # botocore < 1.36: key not supported and not needed
    else:
        cfg_kwargs["request_checksum_calculation"] = "when_required"
        cfg_kwargs["response_checksum_validation"] = "when_required"
    cfg = Config(**cfg_kwargs)
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=get_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=get_env("R2_SECRET_ACCESS_KEY"),
        config=cfg,
    )


def cmd_upload(args):
    client = make_client()
    bucket = get_env("R2_BUCKET_NAME")
    src = Path(args.src)
    if not src.is_dir():
        sys.exit(f"ERROR: --src {src} is not a directory (use upload-file for a single file)")

    prefix = args.prefix.strip("/")
    files = [p for p in src.rglob("*") if p.is_file()]
    print(f"Uploading {len(files)} files from {src} -> r2://{bucket}/{prefix}/")
    for p in files:
        rel = p.relative_to(src).as_posix()
        key = f"{prefix}/{rel}"
        print(f"  -> {key} ({p.stat().st_size/1e6:.1f} MB)")
        client.upload_file(str(p), bucket, key, Config=TRANSFER_CONFIG)
    print("Upload complete.")


def cmd_upload_file(args):
    client = make_client()
    bucket = get_env("R2_BUCKET_NAME")
    src = Path(args.src)
    if not src.is_file():
        sys.exit(f"ERROR: --src {src} is not a file")
    print(f"Uploading {src} -> r2://{bucket}/{args.key} ({src.stat().st_size/1e6:.1f} MB)")
    client.upload_file(str(src), bucket, args.key, Config=TRANSFER_CONFIG)
    print("Upload complete.")


def cmd_download(args):
    client = make_client()
    bucket = get_env("R2_BUCKET_NAME")
    prefix = args.prefix.strip("/")
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):].lstrip("/")
            if not rel:
                continue
            out_path = dest / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  <- {key} ({obj['Size']/1e6:.1f} MB)")
            client.download_file(bucket, key, str(out_path), Config=TRANSFER_CONFIG)
            count += 1
    print(f"Downloaded {count} files into {dest}")


def cmd_download_file(args):
    """Download a single R2 object to a local path."""
    client = make_client()
    bucket = get_env("R2_BUCKET_NAME")
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  <- {args.key} -> {dest}")
    client.download_file(bucket, args.key, str(dest), Config=TRANSFER_CONFIG)


def cmd_download_manifest(args):
    """Download only the images (and optionally metadata) listed in a JSONL manifest."""
    import json as _json
    import time

    client = make_client()
    bucket = get_env("R2_BUCKET_NAME")
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"ERROR: manifest not found: {manifest_path}")

    records = [_json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    total = len(records)
    print(f"Manifest: {total} records")

    img_dir = Path(args.image_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = Path(args.metadata_dir) if args.metadata_dir else None
    if meta_dir:
        meta_dir.mkdir(parents=True, exist_ok=True)

    PRINT_EVERY = 500
    skipped = 0
    downloaded = 0
    t_start = time.monotonic()
    t_last = t_start

    for i, rec in enumerate(records):
        img_dest = img_dir / rec["file"]
        if not img_dest.exists():
            client.download_file(bucket, f"images/{rec['file']}", str(img_dest),
                                 Config=TRANSFER_CONFIG)
            downloaded += 1
        else:
            skipped += 1

        if meta_dir:
            meta_dest = meta_dir / f"{rec['id']}.json"
            if not meta_dest.exists():
                client.download_file(bucket, f"metadata/{rec['id']}.json", str(meta_dest),
                                     Config=TRANSFER_CONFIG)

        if (i + 1) % PRINT_EVERY == 0 or (i + 1) == total:
            now = time.monotonic()
            elapsed = now - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - (i + 1)) / rate if rate > 0 else 0
            chunk_rate = PRINT_EVERY / (now - t_last) if (now - t_last) > 0 else 0
            t_last = now
            pct = (i + 1) / total * 100
            eta_min = remaining / 60
            print(
                f"  [{i+1:>6}/{total}] {pct:5.1f}%  "
                f"downloaded={downloaded}  skipped={skipped}  "
                f"rate={chunk_rate:.0f}/s  elapsed={elapsed/60:.1f}m  eta={eta_min:.1f}m",
                flush=True,
            )

    elapsed_total = time.monotonic() - t_start
    print(f"Done: {total} records in {img_dir} "
          f"({downloaded} downloaded, {skipped} skipped) "
          f"in {elapsed_total/60:.1f}m")


def _sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def cmd_download_archive(args):
    """Download all shards for a mode from R2 and extract into a flat directory.

    This replaces the old download-manifest + gen_captions workflow.
    Each shard is a flat tar containing {id}.{ext} images and {id}.txt captions
    baked in by package_dataset.py.

    The shards are streamed: downloaded to a temp file, verified (sha256),
    extracted in place, then the temp file is deleted — so peak extra disk
    usage is only one shard at a time rather than all shards at once.
    """
    client = make_client()
    bucket = get_env("R2_BUCKET_NAME")

    mode = args.mode
    r2_prefix = args.r2_prefix or f"datasets/soyjak-sdxl-{mode}"
    r2_prefix = r2_prefix.strip("/")
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # --- 1. Fetch shard_manifest.json ---------------------------------------
    manifest_key = f"{r2_prefix}/shard_manifest.json"
    print(f"Fetching shard manifest: r2://{bucket}/{manifest_key}")
    resp = client.get_object(Bucket=bucket, Key=manifest_key)
    shard_manifest = json.loads(resp["Body"].read())
    shards = shard_manifest["shards"]
    total_bytes = shard_manifest["total_bytes"]
    print(
        f"  {len(shards)} shard(s)  |  "
        f"{shard_manifest['total_images']} images  |  "
        f"{total_bytes / 1e9:.2f} GB total"
    )

    # --- 2. Download + extract each shard -----------------------------------
    t_start = time.monotonic()
    extracted_total = 0

    for idx, shard in enumerate(shards):
        shard_key = f"{r2_prefix}/shards/{shard['name']}"
        size_mb = shard["bytes"] / 1e6
        print(
            f"\n[{idx+1}/{len(shards)}] {shard['name']}  "
            f"({size_mb:.0f} MB, {shard['images']} images)",
            flush=True,
        )

        # Download to a temp file in dest so it's on the same filesystem.
        with tempfile.NamedTemporaryFile(dir=dest, suffix=".tar", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            print(f"  downloading...", flush=True)
            t_dl = time.monotonic()
            client.download_file(bucket, shard_key, str(tmp_path), Config=TRANSFER_CONFIG)
            dl_elapsed = time.monotonic() - t_dl
            actual_size = tmp_path.stat().st_size
            speed = actual_size / dl_elapsed / 1e6 if dl_elapsed > 0 else 0
            print(f"  downloaded in {dl_elapsed:.1f}s  ({speed:.0f} MB/s)", flush=True)

            # Verify checksum
            if shard.get("sha256"):
                print(f"  verifying sha256...", flush=True)
                digest = _sha256(tmp_path)
                if digest != shard["sha256"]:
                    sys.exit(
                        f"ERROR: sha256 mismatch for {shard['name']}\n"
                        f"  expected: {shard['sha256']}\n"
                        f"  got:      {digest}"
                    )
                print(f"  checksum OK", flush=True)

            # Extract flat into dest
            print(f"  extracting into {dest}...", flush=True)
            t_ex = time.monotonic()
            with tarfile.open(tmp_path, "r") as tf:
                _safe_extract_tar(tf, dest)
            ex_elapsed = time.monotonic() - t_ex
            extracted_total += shard["images"]
            print(
                f"  extracted {shard['images']} files in {ex_elapsed:.1f}s  "
                f"({extracted_total} total so far)",
                flush=True,
            )

        finally:
            tmp_path.unlink(missing_ok=True)

    elapsed = time.monotonic() - t_start
    print(
        f"\n=== ARCHIVE EXTRACT COMPLETE ===\n"
        f"  Shards:    {len(shards)}\n"
        f"  Files:     {extracted_total}\n"
        f"  Dest:      {dest}\n"
        f"  Elapsed:   {elapsed/60:.1f}m"
    )


def cmd_list(args):
    client = make_client()
    bucket = get_env("R2_BUCKET_NAME")
    prefix = args.prefix.strip("/")
    paginator = client.get_paginator("list_objects_v2")
    total = 0
    size = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            print(f"  {obj['Key']}  ({obj['Size']/1e6:.1f} MB)")
            total += 1
            size += obj["Size"]
    print(f"{total} objects, {size/1e9:.2f} GB under r2://{bucket}/{prefix}/")


def main():
    ap = argparse.ArgumentParser(description="Cloudflare R2 sync helper.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("upload", help="Upload a directory tree under a prefix.")
    p.add_argument("--src", required=True)
    p.add_argument("--prefix", default="datasets/soyjak-sdxl")
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("upload-file", help="Upload a single file to a key.")
    p.add_argument("--src", required=True)
    p.add_argument("--key", required=True)
    p.set_defaults(func=cmd_upload_file)

    p = sub.add_parser("download", help="Download all objects under a prefix.")
    p.add_argument("--prefix", default="datasets/soyjak-sdxl")
    p.add_argument("--dest", default="./pkg")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("download-file", help="Download a single R2 object to a local path.")
    p.add_argument("--key", required=True, help="R2 object key.")
    p.add_argument("--dest", required=True, help="Local destination path.")
    p.set_defaults(func=cmd_download_file)

    p = sub.add_parser("download-manifest",
                       help="Download images (+metadata) listed in a JSONL manifest.")
    p.add_argument("--manifest", required=True, help="Path to local JSONL manifest file.")
    p.add_argument("--image-dir", required=True, help="Local dir to write images into.")
    p.add_argument("--metadata-dir", default=None,
                   help="Local dir to write metadata JSONs into (optional).")
    p.set_defaults(func=cmd_download_manifest)

    p = sub.add_parser(
        "download-archive",
        help="Download + extract dataset shards for a given mode (use on Lambda).",
    )
    p.add_argument(
        "--mode", choices=["pilot", "full"], default="full",
        help="Dataset mode: pilot (~10K) or full (~124K). Default: full.",
    )
    p.add_argument(
        "--dest", required=True,
        help="Local directory to extract images+captions into.",
    )
    p.add_argument(
        "--r2-prefix", default=None,
        help="Override R2 prefix (default: datasets/soyjak-sdxl-<mode>).",
    )
    p.set_defaults(func=cmd_download_archive)

    p = sub.add_parser("list", help="List objects under a prefix.")
    p.add_argument("--prefix", default="datasets/soyjak-sdxl")
    p.set_defaults(func=cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
