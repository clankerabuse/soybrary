#!/usr/bin/env python3
"""
Remove bad images (and their .txt caption sidecars) from a DreamBooth image_dir.

Drops:
  - corrupt/truncated files (full Pillow decode, same as latent caching)
  - images whose longest side exceeds --max-long-side (default 2048, matches
    max_bucket_reso in train_lora.sh — huge sources waste cache time for no gain)

Usage:
    python train/prune_bad_images.py /home/ubuntu/train_data
    python train/prune_bad_images.py /home/ubuntu/train_data --dry-run
    python train/prune_bad_images.py /home/ubuntu/train_data --max-long-side 0  # disable size cap
"""
import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: Pillow not installed (should be in sd-venv from sd-scripts)")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def check_image(path: Path, max_long_side: int) -> str | None:
    """
    Return a drop reason ('corrupt', 'too_large') or None if the image is OK.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
            if max_long_side > 0 and max(w, h) > max_long_side:
                return "too_large"
            img.load()
        return None
    except Exception:
        return "corrupt"


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
        default=2048,
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

    to_drop: dict[str, list[Path]] = {"corrupt": [], "too_large": []}
    for i, p in enumerate(images):
        if i and i % 20000 == 0:
            print(f"  checked {i}/{len(images)}...", flush=True)
        reason = check_image(p, args.max_long_side)
        if reason:
            to_drop[reason].append(p)

    for reason, paths in to_drop.items():
        print(f"{reason}: {len(paths)}", flush=True)
        for p in paths[:10]:
            print(f"  {p.name}")
        if len(paths) > 10:
            print(f"  ... and {len(paths) - 10} more")

    all_bad = to_drop["corrupt"] + to_drop["too_large"]
    if args.dry_run:
        return

    for p in all_bad:
        remove_image(image_dir, p)

    remaining = sum(
        1 for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    print(f"Removed {len(all_bad)} image+caption pairs. {remaining} images remain.", flush=True)


if __name__ == "__main__":
    main()
