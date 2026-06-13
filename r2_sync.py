#!/usr/bin/env python3
"""
r2_sync.py - Phase 3 of the Soybrary -> SDXL LoRA pipeline.

Upload/download files to/from Cloudflare R2 via the S3-compatible API.

Used twice in the workflow:
  1. Locally: upload the packaged dataset (shards + dataset.toml + manifest)
     to R2 under datasets/soyjak-sdxl/.
  2. On Lambda: download that prefix back, and later upload the trained LoRA.

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
    is 5 GB; our shards are ~2 GB but multipart is safe and resumable-friendly).

Credentials are read from .env (see .env.example):
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_ENDPOINT

Usage:
  # Upload the whole package dir to a prefix
  python r2_sync.py upload --src data/package --prefix datasets/soyjak-sdxl

  # Download a prefix into a local dir
  python r2_sync.py download --prefix datasets/soyjak-sdxl --dest ./pkg

  # Upload a single file (e.g. trained LoRA)
  python r2_sync.py upload-file --src out/soyjak.safetensors --key models/soyjak-lora-sdxl/soyjak.safetensors

  # List a prefix
  python r2_sync.py list --prefix datasets/soyjak-sdxl
"""

import argparse
import os
import sys
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

    p = sub.add_parser("list", help="List objects under a prefix.")
    p.add_argument("--prefix", default="datasets/soyjak-sdxl")
    p.set_defaults(func=cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
