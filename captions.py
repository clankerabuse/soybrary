#!/usr/bin/env python3
"""
Shared caption construction for Soybrary SDXL full fine-tuning.

Every caption is a subject-lock recipe, not a style sticker:

  soyjak, <variant>, 1boy, portrait, wojak, <subvariants>, <tags>

- `soyjak` + the lead variant are pinned with keep_tokens=2 so they cannot
  be shuffled or tag-dropped.
- `1boy` / `portrait` / `wojak` are class-hijack tokens. They appear on every
  image so SDXL's "person / portrait / wojak" concepts get overwritten by
  soyjaks. Extra object tags still train; they decorate a soyjak rather than
  replacing it.
- No regularization/class images. Caption dropout trains empty steps on
  soyjaks so the unconditional prior stays on-distribution.

The failure mode we are training against is an image with no soyjak in it
at all (photo, landscape, unrelated character).
"""

from collections import Counter
from pathlib import Path

TRIGGER_TOKEN = "soyjak"
CLASS_HIJACK_TOKENS = ("1boy", "portrait", "wojak")


def _skip_set(trigger=TRIGGER_TOKEN):
    skip = {t.lower() for t in CLASS_HIJACK_TOKENS}
    if trigger:
        skip.add(trigger.strip().lower())
    return skip


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
    Booru-style caption with a locked soyjak subject and class-hijack tokens.

    Order: trigger -> variants -> class hijack -> subvariants -> tags.
    Underscores are preserved (standard booru convention).
    """
    parts = []
    if trigger:
        parts.append(trigger)
    parts.extend(meta.get("variants") or [])
    parts.extend(CLASS_HIJACK_TOKENS)
    parts.extend(meta.get("subvariants") or [])
    parts.extend(meta.get("tags") or [])
    return ", ".join(dedup_preserve_order(parts))


def descriptive_tokens(caption, trigger=TRIGGER_TOKEN):
    """Caption tokens that are not the trigger or a class-hijack token."""
    skip = _skip_set(trigger)
    out = []
    for token in (caption or "").split(","):
        token = token.strip()
        if token and token.lower() not in skip:
            out.append(token)
    return out


def has_descriptive_content(caption, trigger=TRIGGER_TOKEN):
    """True if the caption has a variant/tag beyond the locked recipe tokens."""
    return bool(descriptive_tokens(caption, trigger=trigger))


def primary_variant(caption, trigger=TRIGGER_TOKEN):
    """
    First caption token that is not the style trigger or a class-hijack token.

    Used to bucket records for stratified pilot sampling so prefixing `soyjak`
    (and injecting 1boy/portrait/wojak) does not collapse every image into
    a single bucket.
    """
    desc = descriptive_tokens(caption, trigger=trigger)
    if desc:
        return desc[0].lower()
    trigger_key = (trigger or "").strip().lower()
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


def ensure_training_caption(caption, trigger=TRIGGER_TOKEN):
    """
    Rewrite a caption into the training recipe:

      soyjak, <variant>, 1boy, portrait, wojak, <rest>
    """
    caption = ensure_trigger_prefix(caption, trigger=trigger)
    desc = descriptive_tokens(caption, trigger=trigger)
    parts = []
    if trigger:
        parts.append(trigger)
    parts.extend(desc[:1])
    parts.extend(CLASS_HIJACK_TOKENS)
    parts.extend(desc[1:])
    return ", ".join(dedup_preserve_order(parts))


def patch_caption_dir(image_dir: Path, trigger: str = TRIGGER_TOKEN) -> Counter:
    """Rewrite DreamBooth .txt sidecars to the training caption recipe."""
    stats: Counter = Counter()
    image_dir = Path(image_dir)
    for txt_path in sorted(image_dir.glob("*.txt")):
        original = txt_path.read_text(encoding="utf-8")
        updated = ensure_training_caption(original, trigger=trigger)
        stripped = original.strip()
        if updated == stripped:
            if original != updated:
                txt_path.write_text(updated, encoding="utf-8")
            stats["already_prefixed"] += 1
            continue
        txt_path.write_text(updated, encoding="utf-8")
        stats["patched"] += 1
    return stats
