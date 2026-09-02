#!/usr/bin/env python3
"""
Prefix existing DreamBooth .txt sidecars with the soyjak style trigger.

The packed R2 shards from the first full run used variant-first captions
(no shared trigger). Re-uploading 38 GB just to add `soyjak, ` is wasteful;
run this on the extracted image_dir before training instead.

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
        description="Prefix caption sidecars with the soyjak style trigger."
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
        f"Trigger '{args.trigger}': patched {patched} captions, "
        f"{already} already prefixed "
        f"({patched + already} total .txt files)"
    )
    if patched + already == 0:
        sys.exit(f"ERROR: no .txt captions found in {args.image_dir}")


if __name__ == "__main__":
    main()
