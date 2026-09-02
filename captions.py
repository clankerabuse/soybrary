#!/usr/bin/env python3
"""
Shared caption construction for Soybrary SDXL LoRA training.

Every caption starts with a locked subject token (`soyjak`). Extra tags
(objects, settings, props) are allowed and expected — they decorate a soyjak
rather than replacing it. The failure mode we are training against is an
image with no soyjak in it at all (photo, landscape, unrelated character).

kohya `keep_tokens=2` pins `soyjak, <variant>` so shuffle/tag-dropout cannot
drop the subject. `caption_dropout_rate` trains some steps with an empty
caption so the unconditional prior stays on soyjaks even when the prompt is
mostly about something else.
"""

from collections import Counter
from pathlib import Path

TRIGGER_TOKEN = "soyjak"


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


def build_caption(meta, trigger=TRIGGER_TOKEN):
    """
    Booru-style caption with a locked soyjak style trigger.

    Order: trigger -> variants -> subvariants -> tags.
    Underscores are preserved (standard booru convention).
    """
    parts = []
    if trigger:
        parts.append(trigger)
    parts.extend(meta.get("variants") or [])
    parts.extend(meta.get("subvariants") or [])
    parts.extend(meta.get("tags") or [])
    return ", ".join(dedup_preserve_order(parts))


def primary_variant(caption, trigger=TRIGGER_TOKEN):
    """
    First caption token that is not the style trigger.

    Used to bucket records for stratified pilot sampling so prefixing `soyjak`
    does not collapse every image into a single bucket.
    """
    trigger_key = (trigger or "").strip().lower()
    for token in (caption or "").split(","):
        token = token.strip().lower()
        if token and token != trigger_key:
            return token
    return trigger_key or "unknown"


def ensure_trigger_prefix(caption, trigger=TRIGGER_TOKEN):
    """Idempotently prefix a caption with the style trigger."""
    caption = (caption or "").strip()
    trigger = (trigger or "").strip()
    if not trigger:
        return caption
    if not caption:
        return trigger
    first = caption.split(",", 1)[0].strip().lower()
    if first == trigger.lower():
        return caption
    return f"{trigger}, {caption}"


def patch_caption_dir(image_dir: Path, trigger: str = TRIGGER_TOKEN) -> Counter:
    """Rewrite DreamBooth .txt sidecars so each starts with the style trigger."""
    stats: Counter = Counter()
    image_dir = Path(image_dir)
    for txt_path in sorted(image_dir.glob("*.txt")):
        original = txt_path.read_text(encoding="utf-8")
        updated = ensure_trigger_prefix(original, trigger=trigger)
        stripped = original.strip()
        if updated == stripped:
            if original != updated:
                txt_path.write_text(updated, encoding="utf-8")
            stats["already_prefixed"] += 1
            continue
        txt_path.write_text(updated, encoding="utf-8")
        stats["patched"] += 1
    return stats
