#!/usr/bin/env python3
"""
validate_images.py - Scan local Soybrary images before packaging / R2 upload.

Runs stricter checks than Pillow verify()+load() alone (RGB convert, EXIF
transpose, bucket downscale, full pixel read, re-encode) so corrupt files are
caught on your machine instead of mid-training on Lambda.

Usage:
    python validate_images.py
    python validate_images.py --manifest data/manifests/dataset.jsonl
    python validate_images.py --quarantine
    python validate_images.py --workers 8 --report data/manifests/bad_images.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from image_validate import (
    DEFAULT_MAX_LONG_SIDE,
    IMAGE_EXTS,
    ImageCheckResult,
    check_image_path,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MANIFESTS_DIR = DATA_DIR / "manifests"
QUARANTINE_DIR = DATA_DIR / "quarantine" / "images"


def _load_manifest_paths(manifest_path: Path, images_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        paths.append(images_dir / rec["file"])
    return paths


def _iter_image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _worker(args_tuple: tuple) -> tuple[str, dict]:
    path_str, max_long_side = args_tuple
    result = check_image_path(Path(path_str), max_long_side=max_long_side)
    return path_str, {
        "ok": result.ok,
        "reason": result.reason,
        "detail": result.detail,
    }


def _quarantine(path: Path, quarantine_dir: Path) -> None:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dest = quarantine_dir / path.name
    if dest.exists():
        dest.unlink()
    shutil.move(str(path), str(dest))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate local training images before packaging / R2 upload."
    )
    ap.add_argument(
        "--images-dir",
        type=Path,
        default=IMAGES_DIR,
        help=f"Directory of scraped images (default: {IMAGES_DIR})",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSONL manifest; only validate listed files.",
    )
    ap.add_argument(
        "--max-long-side",
        type=int,
        default=DEFAULT_MAX_LONG_SIDE,
        help=f"Reject images whose long side exceeds this (default: {DEFAULT_MAX_LONG_SIDE}). "
        "Set 0 to disable.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Parallel worker processes (default: half of CPU count).",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=MANIFESTS_DIR / "bad_images.json",
        help="Write JSON report of failed files.",
    )
    ap.add_argument(
        "--quarantine",
        action="store_true",
        help=f"Move failed images to {QUARANTINE_DIR} (does not touch DB).",
    )
    ap.add_argument(
        "--fail-on-bad",
        action="store_true",
        help="Exit with code 1 when any bad image is found.",
    )
    args = ap.parse_args()

    images_dir = args.images_dir
    if not images_dir.is_dir():
        sys.exit(f"ERROR: images dir not found: {images_dir}")

    if args.manifest:
        if not args.manifest.exists():
            sys.exit(f"ERROR: manifest not found: {args.manifest}")
        paths = _load_manifest_paths(args.manifest, images_dir)
        scope = f"manifest {args.manifest.name}"
    else:
        paths = _iter_image_paths(images_dir)
        scope = f"all files in {images_dir}"

    total = len(paths)
    if total == 0:
        print(f"No images to scan ({scope}).")
        return

    cap = ""
    if args.max_long_side > 0:
        cap = f", max long side {args.max_long_side}px"
    print(
        f"Scanning {total} images ({scope}) with {args.workers} workers{cap}...",
        flush=True,
    )

    bad: list[dict] = []
    by_reason: Counter[str] = Counter()
    t0 = time.monotonic()
    done = 0

    job_args = [(str(p), args.max_long_side) for p in paths]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, item) for item in job_args]
        for fut in as_completed(futures):
            path_str, payload = fut.result()
            done += 1
            if not payload["ok"]:
                entry = {
                    "file": Path(path_str).name,
                    "path": path_str,
                    "reason": payload["reason"],
                    "detail": payload["detail"],
                }
                bad.append(entry)
                by_reason[payload["reason"] or "unknown"] += 1

            if done % 5000 == 0 or done == total:
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"  [{done}/{total}] bad={len(bad)} rate={rate:.0f}/s",
                    flush=True,
                )

    bad.sort(key=lambda row: row["file"])

    if args.quarantine and bad:
        print(f"Quarantining {len(bad)} file(s) -> {QUARANTINE_DIR}", flush=True)
        for row in bad:
            src = Path(row["path"])
            if src.exists():
                _quarantine(src, QUARANTINE_DIR)

    report = {
        "scope": scope,
        "images_dir": str(images_dir),
        "manifest": str(args.manifest) if args.manifest else None,
        "max_long_side": args.max_long_side,
        "scanned": total,
        "bad": len(bad),
        "by_reason": dict(by_reason),
        "files": bad,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    elapsed = time.monotonic() - t0
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Scanned:  {total}")
    print(f"Bad:      {len(bad)}")
    if by_reason:
        print("By reason:")
        for reason, count in by_reason.most_common():
            print(f"  {reason}: {count}")
    print(f"Report:   {args.report}")
    print(f"Elapsed:  {elapsed/60:.1f}m")

    if bad:
        print("\nFirst failures:")
        for row in bad[:10]:
            detail = f" ({row['detail']})" if row["detail"] else ""
            print(f"  {row['file']}: {row['reason']}{detail}")
        if len(bad) > 10:
            print(f"  ... and {len(bad) - 10} more (see report)")

        if args.quarantine:
            print(
                "\nNext steps after quarantine:\n"
                "  1. python build_dataset.py --validate-images\n"
                "  2. python package_dataset.py --mode full"
            )
        else:
            print(
                "\nRe-run with --quarantine to move bad files aside, then rebuild "
                "the manifest and shards."
            )

        if args.fail_on_bad:
            sys.exit(1)


if __name__ == "__main__":
    main()
