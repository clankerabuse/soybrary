#!/usr/bin/env python3
"""
build_dataset.py - Phase 1 of the Soybrary -> SDXL LoRA pipeline.

Scans the scraped data and produces a training manifest (JSONL). It does NOT
copy any image data; packaging happens in package_dataset.py (Phase 2).

For each completed static image (PNG/JPEG/WebP) it:
  - locates the real file on disk in data/images/{id}.* (the DB `extension`
    column is unreliable, so we trust the filesystem),
  - reads data/metadata/{id}.json,
  - builds a booru-style caption: dedup(variants + subvariants + tags),
    with variants first so kohya's keep_tokens can pin them,
  - applies a minimum short-side resolution filter (default 512px),
  - applies a maximum long-side filter (default 2048px; matches kohya buckets),
  - drops images that would have an empty caption.

Output:
  data/manifests/dataset.jsonl   one JSON object per kept image
  data/manifests/stats.json      summary statistics

Usage:
  python build_dataset.py
  python build_dataset.py --min-short-side 512
  python build_dataset.py --min-short-side 0     # keep all resolutions
"""

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from image_validate import check_image_path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_DIR = DATA_DIR / "metadata"
MANIFESTS_DIR = DATA_DIR / "manifests"
DB_PATH = DATA_DIR / "soybooru.db"

# Mime types we consider trainable static images.
STATIC_IMAGE_MIMES = ("image/png", "image/jpeg", "image/webp")


def dedup_preserve_order(items):
    """Lowercase-dedup a list of tag strings while preserving first-seen order."""
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        tag = str(item).strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def build_caption(meta):
    """
    Booru-style caption with no fixed trigger word.

    Order: variants -> subvariants -> tags. Variant names are the
    differentiators (chudjak, cobson, ...) and come first so that
    keep_tokens=1 in the kohya config pins the leading variant token.
    Underscores are preserved (standard booru convention).
    """
    parts = []
    parts.extend(meta.get("variants") or [])
    parts.extend(meta.get("subvariants") or [])
    parts.extend(meta.get("tags") or [])
    tags = dedup_preserve_order(parts)
    return ", ".join(tags)


def image_is_readable(path: Path) -> bool:
    """Return False for images that would crash sd-scripts latent caching."""
    return check_image_path(path).ok


def index_images():
    """
    Scan the images directory once and map post_id -> filename.

    Globbing per-post against a 150k+ file directory is pathologically slow,
    so we list it a single time. If an id somehow has multiple files we keep
    the first by sorted name (deterministic).
    """
    index = {}
    for entry in sorted(IMAGES_DIR.iterdir()):
        if not entry.is_file():
            continue
        stem = entry.stem  # filename without extension
        if stem.isdigit():
            pid = int(stem)
            index.setdefault(pid, entry.name)
    return index


def stratified_sample(records, limit, seed):
    """
    Sample ~`limit` records while preserving variant diversity.

    Each record is bucketed by its primary variant (the leading caption token,
    i.e. the first variant/subvariant/tag). We then round-robin across buckets,
    drawing one shuffled record at a time, until we hit `limit`. This avoids a
    naive head-N slice that would over-represent low post IDs / early variants,
    and guarantees rare variants still appear in a small pilot set.
    """
    if limit <= 0 or len(records) <= limit:
        return records

    rng = random.Random(seed)
    buckets = defaultdict(list)
    for rec in records:
        primary = rec["caption"].split(",", 1)[0].strip().lower()
        buckets[primary].append(rec)

    bucket_keys = list(buckets.keys())
    for k in bucket_keys:
        rng.shuffle(buckets[k])
    rng.shuffle(bucket_keys)

    selected = []
    selected_ids = set()
    # Round-robin across variant buckets.
    while len(selected) < limit:
        progressed = False
        for k in bucket_keys:
            if buckets[k]:
                rec = buckets[k].pop()
                if rec["id"] not in selected_ids:
                    selected.append(rec)
                    selected_ids.add(rec["id"])
                    progressed = True
                    if len(selected) >= limit:
                        break
        if not progressed:
            break
    return selected


def main():
    ap = argparse.ArgumentParser(description="Build SDXL LoRA training manifest.")
    ap.add_argument(
        "--min-short-side",
        type=int,
        default=512,
        help="Minimum image short side in px. Images smaller than this are "
        "excluded (SDXL trains at 1024; tiny images upscale poorly). "
        "Set 0 to keep all. Default: 512",
    )
    ap.add_argument(
        "--max-long-side",
        type=int,
        default=2048,
        help="Maximum image long side in px. Images larger than this are excluded "
        "(matches kohya max_bucket_reso; huge sources waste latent-cache time). "
        "Set 0 to keep all. Default: 2048",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=MANIFESTS_DIR / "dataset.jsonl",
        help="Output JSONL manifest path.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, cap the manifest to roughly this many images. The subset "
        "is sampled stratified across variants (so a small pilot run keeps "
        "variant diversity instead of just the lowest post IDs). Default: 0 "
        "(no cap, full dataset).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --limit sampling (reproducible pilot sets).",
    )
    ap.add_argument(
        "--validate-images",
        action="store_true",
        help="Open each image with strict training-path validation and drop "
        "corrupt/truncated files. Slower locally but prevents sd-scripts "
        "crashes on Lambda. Recommended before package_dataset.py.",
    )
    args = ap.parse_args()

    if not DB_PATH.exists():
        sys.exit(f"ERROR: database not found at {DB_PATH}")
    if not IMAGES_DIR.is_dir():
        sys.exit(f"ERROR: images dir not found at {IMAGES_DIR}")
    if not METADATA_DIR.is_dir():
        sys.exit(f"ERROR: metadata dir not found at {METADATA_DIR}")

    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in STATIC_IMAGE_MIMES)
    rows = conn.execute(
        f"SELECT id FROM posts "
        f"WHERE status='completed' AND mime_type IN ({placeholders}) "
        f"ORDER BY id",
        STATIC_IMAGE_MIMES,
    ).fetchall()
    conn.close()

    total_candidates = len(rows)
    print(f"Candidates (completed static images in DB): {total_candidates}", flush=True)

    print("Indexing images directory (one-time scan)...", flush=True)
    image_index = index_images()
    print(f"  indexed {len(image_index)} image files", flush=True)

    stats = Counter()
    records = []

    for i, row in enumerate(rows):
        post_id = row["id"]

        if i and i % 20000 == 0:
            print(f"  processed {i}/{total_candidates} (eligible {len(records)})", flush=True)

        img_name = image_index.get(post_id)
        if img_name is None:
            stats["missing_image_file"] += 1
            continue
        img_path = IMAGES_DIR / img_name

        meta_path = METADATA_DIR / f"{post_id}.json"
        if not meta_path.exists():
            stats["missing_metadata"] += 1
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            stats["bad_metadata_json"] += 1
            continue

        width = meta.get("width") or 0
        height = meta.get("height") or 0
        if not width or not height:
            stats["missing_dimensions"] += 1
            continue

        if args.min_short_side > 0 and min(width, height) < args.min_short_side:
            stats["below_min_resolution"] += 1
            continue

        if args.max_long_side > 0 and max(width, height) > args.max_long_side:
            stats["above_max_resolution"] += 1
            continue

        if args.validate_images and not image_is_readable(img_path):
            stats["corrupt_image"] += 1
            continue

        caption = build_caption(meta)
        if not caption:
            stats["empty_caption"] += 1
            continue

        records.append({
            "id": post_id,
            "file": img_path.name,
            "ext": img_path.suffix.lstrip(".").lower(),
            "width": width,
            "height": height,
            "caption": caption,
        })

    eligible = len(records)

    # Optional stratified subsample for a smaller pilot run.
    if args.limit and args.limit > 0 and eligible > args.limit:
        records = stratified_sample(records, args.limit, args.seed)
        print(f"Sampled {len(records)} of {eligible} eligible images "
              f"(stratified by variant, seed={args.seed})", flush=True)

    # Sort by id for deterministic output ordering.
    records.sort(key=lambda r: r["id"])

    variant_counter = Counter()
    examples = records[:5]
    with open(args.out, "w", encoding="utf-8") as out_f:
        for record in records:
            primary = record["caption"].split(",", 1)[0].strip().lower()
            variant_counter[primary] += 1
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    kept = len(records)
    stats_out = {
        "total_candidates": total_candidates,
        "eligible": eligible,
        "kept": kept,
        "limit": args.limit,
        "seed": args.seed,
        "excluded": dict(stats),
        "min_short_side": args.min_short_side,
        "max_long_side": args.max_long_side,
        "top_variants": variant_counter.most_common(20),
        "manifest_path": str(args.out),
    }
    # Stats filename mirrors the manifest name so pilot and full runs coexist:
    #   dataset.jsonl          -> dataset.stats.json
    #   dataset_pilot10k.jsonl -> dataset_pilot10k.stats.json
    stats_path = args.out.with_suffix("").with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(f"Kept:       {kept}")
    print(f"Eligible:   {eligible}" + (f" (capped to {args.limit})" if args.limit else ""))
    print(f"Candidates: {total_candidates}")
    print("Excluded:")
    for reason, count in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {count}")
    print(f"\nManifest: {args.out}")
    print(f"Stats:    {stats_path}")
    print("\nExample records:")
    for ex in examples:
        cap = ex["caption"]
        cap_preview = cap if len(cap) <= 100 else cap[:100] + "..."
        print(f"  [{ex['id']}] {ex['width']}x{ex['height']} {ex['file']}")
        print(f"      caption: {cap_preview}")


if __name__ == "__main__":
    main()
