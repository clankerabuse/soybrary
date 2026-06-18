#!/usr/bin/env python3
"""
Remove bad images (and their .txt caption sidecars) from a DreamBooth image_dir.

Drops files that would crash kohya sd-scripts latent caching:
  - corrupt/truncated files
  - images that fail RGB convert / EXIF transpose / bucket resize
  - images whose longest side exceeds --max-long-side (default 2048)

Usage:
    python train/prune_bad_images.py /home/ubuntu/train_data
    python train/prune_bad_images.py /home/ubuntu/train_data --dry-run
    python train/prune_bad_images.py /home/ubuntu/train_data --max-long-side 0
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from image_validate import (  # noqa: E402
    DEFAULT_MAX_LONG_SIDE,
    IMAGE_EXTS,
    check_image_path,
)


def latent_cache_paths(image_dir: Path, image_path: Path) -> list[Path]:
    """Kohya disk latent cache files that may exist beside the image."""
    return [
        image_dir / f"{image_path.name}.npz",
        image_dir / f"{image_path.stem}.npz",
    ]


def remove_image(image_dir: Path, image_path: Path) -> None:
    txt = image_dir / f"{image_path.stem}.txt"
    for cache in latent_cache_paths(image_dir, image_path):
        cache.unlink(missing_ok=True)
    image_path.unlink(missing_ok=True)
    if txt.exists():
        txt.unlink()


def main():
    ap = argparse.ArgumentParser(description="Prune bad images from a training dir.")
    ap.add_argument("image_dir", type=Path, help="Flat DreamBooth image_dir")
    ap.add_argument(
        "--max-long-side",
        type=int,
        default=DEFAULT_MAX_LONG_SIDE,
        help="Drop images whose longest side exceeds this (default: 2048). "
        "Set 0 to keep all sizes.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List dropped files without deleting",
    )
    args = ap.parse_args()

    image_dir = args.image_dir
    if not image_dir.is_dir():
        sys.exit(f"ERROR: not a directory: {image_dir}")

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    cap = f", max long side {args.max_long_side}px" if args.max_long_side > 0 else ""
    print(f"Scanning {len(images)} images in {image_dir}{cap}...", flush=True)

    to_drop: dict[str, list[Path]] = {}
    for i, p in enumerate(images):
        if i and i % 20000 == 0:
            print(f"  checked {i}/{len(images)}...", flush=True)
        result = check_image_path(p, max_long_side=args.max_long_side)
        if not result.ok:
            reason = result.reason or "bad"
            to_drop.setdefault(reason, []).append(p)

    for reason, paths in sorted(to_drop.items()):
        print(f"{reason}: {len(paths)}", flush=True)
        for p in paths[:10]:
            print(f"  {p.name}")
        if len(paths) > 10:
            print(f"  ... and {len(paths) - 10} more")

    all_bad = [p for paths in to_drop.values() for p in paths]
    if args.dry_run:
        return

    for p in all_bad:
        remove_image(image_dir, p)

    remaining = sum(
        1 for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    print(
        f"Removed {len(all_bad)} image+caption pairs. {remaining} images remain.",
        flush=True,
    )


if __name__ == "__main__":
    main()
