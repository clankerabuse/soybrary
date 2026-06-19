#!/usr/bin/env python3
"""
Fast parallel scan for bad training images (check-only; does not delete).

Uses the same Pillow verify+load + imagesize JPEG header checks as
prune_bad_images.py / sd-scripts latent caching, so anything reported here
would crash training.

Checks:
  - empty files
  - corrupt/truncated images (full decode)
  - images whose longest side exceeds --max-long-side (default 2048)

Usage:
    python train/check_images.py /home/ubuntu/train_data
    python train/check_images.py /home/ubuntu/train_data --dry-run  # alias, default
    python train/check_images.py /home/ubuntu/train_data --fail       # exit 1 if any bad
    python train/check_images.py /home/ubuntu/train_data --workers 32
    MODE=full bash train/check_images.sh
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from image_validate import (  # noqa: E402
    DEFAULT_MAX_LONG_SIDE,
    IMAGE_EXTS,
    check_image_path,
)


def extended_check(path: Path, max_long_side: int) -> str | None:
    try:
        if path.stat().st_size == 0:
            return "empty"
    except OSError:
        return "corrupt"
    result = check_image_path(path, max_long_side=max_long_side)
    if result.ok:
        return None
    reason = result.reason or "corrupt"
    if reason == "empty_file":
        return "empty"
    if reason == "too_large":
        return "too_large"
    return "corrupt"


def _worker(args: tuple[str, int]) -> tuple[str, str | None]:
    path_str, max_long_side = args
    path = Path(path_str)
    return path.name, extended_check(path, max_long_side)


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check training images for corruption/oversize (parallel, no deletes)."
    )
    ap.add_argument("image_dir", type=Path, help="Flat DreamBooth image_dir")
    ap.add_argument(
        "--max-long-side",
        type=int,
        default=DEFAULT_MAX_LONG_SIDE,
        help="Flag images whose longest side exceeds this (default: 2048). "
        "Set 0 to skip size check.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Parallel worker processes (default: cpu_count - 1)",
    )
    ap.add_argument(
        "--fail",
        action="store_true",
        help="Exit with status 1 if any bad images are found",
    )
    ap.add_argument(
        "--list-file",
        type=Path,
        help="Write one bad filename per line to this file",
    )
    ap.add_argument(
        "--max-list",
        type=int,
        default=50,
        help="Max filenames to print per category (default: 50)",
    )
    args = ap.parse_args()

    image_dir = args.image_dir.resolve()
    if not image_dir.is_dir():
        print(f"ERROR: not a directory: {image_dir}", file=sys.stderr)
        return 2

    images = list_images(image_dir)
    cap = f", max long side {args.max_long_side}px" if args.max_long_side > 0 else ""
    print(
        f"Checking {len(images)} images in {image_dir}{cap} "
        f"({args.workers} workers)...",
        flush=True,
    )

    if not images:
        print("No images found.", flush=True)
        return 0

    bad: dict[str, list[str]] = {"empty": [], "corrupt": [], "too_large": []}
    t0 = time.perf_counter()
    done = 0

    work = [(str(p), args.max_long_side) for p in images]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, item) for item in work]
        for fut in as_completed(futures):
            name, reason = fut.result()
            done += 1
            if reason:
                bad[reason].append(name)
            if done % 10000 == 0 or done == len(images):
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"  {done}/{len(images)} ({rate:.0f} img/s)",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    total_bad = sum(len(v) for v in bad.values())
    ok = len(images) - total_bad

    print(flush=True)
    print(f"Done in {elapsed:.1f}s — OK: {ok}, bad: {total_bad}", flush=True)
    for reason, names in bad.items():
        print(f"  {reason}: {len(names)}", flush=True)
        for name in sorted(names)[: args.max_list]:
            print(f"    {name}")
        if len(names) > args.max_list:
            print(f"    ... and {len(names) - args.max_list} more")

    if args.list_file and total_bad:
        lines = sorted(
            name for names in bad.values() for name in names
        )
        args.list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWrote {len(lines)} names to {args.list_file}", flush=True)

    if total_bad:
        print("\nTo remove bad images:", flush=True)
        print(
            f"  python {Path(__file__).resolve().parent / 'prune_bad_images.py'} {image_dir} "
            f"--max-long-side {args.max_long_side}",
            flush=True,
        )
        if args.fail:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
