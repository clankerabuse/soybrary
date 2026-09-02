import tempfile
import unittest
from pathlib import Path

from captions import (
    TRIGGER_TOKEN,
    build_caption,
    dedup_preserve_order,
    ensure_trigger_prefix,
    patch_caption_dir,
    primary_variant,
)


class TestDedupPreserveOrder(unittest.TestCase):
    def test_drops_duplicate_case_insensitive(self):
        self.assertEqual(
            dedup_preserve_order(["Chudjak", "open_mouth", "chudjak", "  ", None]),
            ["Chudjak", "open_mouth"],
        )


class TestBuildCaption(unittest.TestCase):
    def test_prefixes_trigger_then_variant_then_tags(self):
        caption = build_caption({
            "variants": ["chudjak"],
            "subvariants": ["closed_mouth"],
            "tags": ["pink_hair", "tears"],
        })
        self.assertEqual(
            caption,
            "soyjak, chudjak, closed_mouth, pink_hair, tears",
        )

    def test_does_not_duplicate_trigger_if_variant_is_soyjak(self):
        caption = build_caption({
            "variants": ["soyjak"],
            "tags": ["looking_at_viewer"],
        })
        self.assertEqual(caption, "soyjak, looking_at_viewer")

    def test_empty_meta_is_just_the_trigger(self):
        self.assertEqual(build_caption({}), TRIGGER_TOKEN)

    def test_can_disable_trigger(self):
        caption = build_caption({"variants": ["cobson"]}, trigger="")
        self.assertEqual(caption, "cobson")


class TestPrimaryVariant(unittest.TestCase):
    def test_skips_trigger(self):
        self.assertEqual(
            primary_variant("soyjak, chudjak, open_mouth"),
            "chudjak",
        )

    def test_falls_back_to_trigger_when_only_trigger(self):
        self.assertEqual(primary_variant("soyjak"), "soyjak")


class TestEnsureTriggerPrefix(unittest.TestCase):
    def test_prefixes_legacy_captions(self):
        self.assertEqual(
            ensure_trigger_prefix("chudjak, pink_hair"),
            "soyjak, chudjak, pink_hair",
        )

    def test_idempotent_when_already_prefixed(self):
        self.assertEqual(
            ensure_trigger_prefix("soyjak, chudjak, pink_hair"),
            "soyjak, chudjak, pink_hair",
        )

    def test_empty_caption_becomes_trigger(self):
        self.assertEqual(ensure_trigger_prefix(""), "soyjak")


class TestPatchCaptionDir(unittest.TestCase):
    def test_rewrites_legacy_sidecars_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "1.txt"
            already = root / "2.txt"
            legacy.write_text("chudjak, screaming\n", encoding="utf-8")
            already.write_text("soyjak, cobson", encoding="utf-8")

            first = patch_caption_dir(root)
            second = patch_caption_dir(root)

            self.assertEqual(first["patched"], 1)
            self.assertEqual(first["already_prefixed"], 1)
            self.assertEqual(second["patched"], 0)
            self.assertEqual(second["already_prefixed"], 2)
            self.assertEqual(
                legacy.read_text(encoding="utf-8"),
                "soyjak, chudjak, screaming",
            )
            self.assertEqual(already.read_text(encoding="utf-8"), "soyjak, cobson")


class TestTrainRecipe(unittest.TestCase):
    def test_dataset_toml_pins_trigger_and_dropout(self):
        text = Path("train/train_lora.sh").read_text(encoding="utf-8")
        self.assertIn("keep_tokens = 2", text)
        self.assertIn("caption_dropout_rate = 0.08", text)
        self.assertIn("caption_tag_dropout_rate = 0.15", text)
        self.assertIn("ensure_trigger_captions.py", text)

    def test_full_config_is_v2_style_lock(self):
        text = Path("train/config.toml").read_text(encoding="utf-8")
        self.assertIn("network_dim = 64", text)
        self.assertIn("max_train_steps = 18000", text)
        self.assertIn("output_name = \"soyjak-lora-sdxl-v2\"", text)
        self.assertIn("text_encoder_lr = 1.0e-4", text)


if __name__ == "__main__":
    unittest.main()
