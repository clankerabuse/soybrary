#!/usr/bin/env python3
"""
Rewrite existing DreamBooth .txt sidecars to the training caption recipe:

  soyjak, <variant>, 1boy, portrait, wojak, <rest>

The packed R2 shards from the first full run used variant-first captions
(no shared trigger, no class hijack). Re-uploading 38 GB just to change
.txt files is wasteful; run this on the extracted image_dir before training.

Usage:
    python train/ensure_trigger_captions.py /home/ubuntu/train_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python train/ensure_trigger_captions.py` from the repo root and
# `python ensure_trigger_captions.py` from train/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from captions import TRIGGER_TOKEN, patch_caption_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Rewrite caption sidecars to the soyjak + class-hijack recipe."
    )
    ap.add_argument("image_dir", type=Path, help="DreamBooth image_dir with .txt sidecars")
    ap.add_argument("--trigger", default=TRIGGER_TOKEN, help="Style trigger token")
    args = ap.parse_args()

    if not args.image_dir.is_dir():
        sys.exit(f"ERROR: image dir not found: {args.image_dir}")

    stats = patch_caption_dir(args.image_dir, trigger=args.trigger)
    patched = stats["patched"]
    already = stats["already_prefixed"]
    print(
        f"Caption recipe (trigger '{args.trigger}' + class hijack): "
        f"patched {patched}, {already} already applied "
        f"({patched + already} total .txt files)"
    )
    if patched + already == 0:
        sys.exit(f"ERROR: no .txt captions found in {args.image_dir}")


if __name__ == "__main__":
    main()
