#!/usr/bin/env python3
"""
gen_captions.py - Run on the Lambda instance after images + metadata are pulled.

Reads each metadata/{id}.json and writes a matching {id}.txt caption sidecar
into the image directory. kohya sd-scripts (DreamBooth-style) picks up the
.txt file alongside the image automatically.

Caption format: dedup join of variants, subvariants, tags (variants first so
keep_tokens=1 pins the leading variant token as the differentiator).

Usage:
    python gen_captions.py --image-dir /home/ubuntu/train_data \
                           --metadata-dir /home/ubuntu/metadata

    # or with a manifest to only caption the pilot subset:
    python gen_captions.py --image-dir /home/ubuntu/train_data \
                           --metadata-dir /home/ubuntu/metadata \
                           --manifest /home/ubuntu/manifests/dataset_pilot10k.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def dedup_preserve_order(items):
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


def build_caption(meta):
    parts = []
    parts.extend(meta.get("variants") or [])
    parts.extend(meta.get("subvariants") or [])
    parts.extend(meta.get("tags") or [])
    return ", ".join(dedup_preserve_order(parts))


def main():
    ap = argparse.ArgumentParser(description="Generate .txt caption sidecars from metadata JSON.")
    ap.add_argument("--image-dir", required=True, type=Path,
                    help="Directory containing downloaded images.")
    ap.add_argument("--metadata-dir", required=True, type=Path,
                    help="Directory containing {id}.json metadata files.")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Optional JSONL manifest (from build_dataset.py). If given, only "
                         "images listed in the manifest get captions written. Use this for "
                         "the pilot subset. Omit to caption everything in image-dir.")
    args = ap.parse_args()

    if not args.image_dir.is_dir():
        sys.exit(f"ERROR: --image-dir not found: {args.image_dir}")
    if not args.metadata_dir.is_dir():
        sys.exit(f"ERROR: --metadata-dir not found: {args.metadata_dir}")

    # Build the list of post IDs to process.
    if args.manifest:
        if not args.manifest.exists():
            sys.exit(f"ERROR: --manifest not found: {args.manifest}")
        ids = []
        with open(args.manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.append(json.loads(line)["id"])
        print(f"Manifest: {len(ids)} images to caption")
    else:
        # Caption everything present in the image dir.
        ids = [int(p.stem) for p in args.image_dir.iterdir()
               if p.is_file() and p.stem.isdigit()]
        ids.sort()
        print(f"No manifest — captioning all {len(ids)} images in {args.image_dir}")

    stats = Counter()
    for i, post_id in enumerate(ids):
        if i and i % 5000 == 0:
            print(f"  {i}/{len(ids)}...", flush=True)

        meta_path = args.metadata_dir / f"{post_id}.json"
        if not meta_path.exists():
            stats["missing_metadata"] += 1
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            stats["bad_json"] += 1
            continue

        caption = build_caption(meta)
        if not caption:
            stats["empty_caption"] += 1
            continue

        # Find the image file to get its stem (extension varies).
        # We only write the .txt if the image actually exists.
        img_matches = sorted(args.image_dir.glob(f"{post_id}.*"))
        img_matches = [p for p in img_matches if not p.suffix.lower() == ".txt"]
        if not img_matches:
            stats["image_missing"] += 1
            continue

        # Write sidecar next to the image.
        txt_path = img_matches[0].with_suffix(".txt")
        txt_path.write_text(caption, encoding="utf-8")
        stats["written"] += 1

    print(f"\n=== DONE ===")
    print(f"  Written:          {stats['written']}")
    for k, v in stats.items():
        if k != "written":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
