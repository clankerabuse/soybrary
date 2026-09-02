import tempfile
import unittest
from pathlib import Path

from captions import (
    CLASS_HIJACK_TOKENS,
    TRIGGER_TOKEN,
    build_caption,
    descriptive_tokens,
    dedup_preserve_order,
    ensure_training_caption,
    ensure_trigger_prefix,
    has_descriptive_content,
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
    def test_prefixes_trigger_variant_class_then_tags(self):
        caption = build_caption({
            "variants": ["chudjak"],
            "subvariants": ["closed_mouth"],
            "tags": ["pink_hair", "tears"],
        })
        self.assertEqual(
            caption,
            "soyjak, chudjak, 1boy, portrait, wojak, closed_mouth, pink_hair, tears",
        )

    def test_does_not_duplicate_trigger_if_variant_is_soyjak(self):
        caption = build_caption({
            "variants": ["soyjak"],
            "tags": ["looking_at_viewer"],
        })
        self.assertEqual(
            caption,
            "soyjak, 1boy, portrait, wojak, looking_at_viewer",
        )

    def test_empty_meta_is_recipe_tokens_only(self):
        caption = build_caption({})
        self.assertEqual(caption, "soyjak, 1boy, portrait, wojak")
        self.assertFalse(has_descriptive_content(caption))

    def test_can_disable_trigger(self):
        caption = build_caption({"variants": ["cobson"]}, trigger="")
        self.assertEqual(caption, "cobson, 1boy, portrait, wojak")


class TestPrimaryVariant(unittest.TestCase):
    def test_skips_trigger_and_class_tokens(self):
        self.assertEqual(
            primary_variant("soyjak, chudjak, 1boy, portrait, wojak, open_mouth"),
            "chudjak",
        )

    def test_falls_back_to_trigger_when_only_recipe_tokens(self):
        self.assertEqual(primary_variant("soyjak, 1boy, portrait, wojak"), "soyjak")


class TestEnsureTrainingCaption(unittest.TestCase):
    def test_rewrites_legacy_variant_first_captions(self):
        self.assertEqual(
            ensure_training_caption("chudjak, pink_hair"),
            "soyjak, chudjak, 1boy, portrait, wojak, pink_hair",
        )

    def test_idempotent_when_already_applied(self):
        caption = "soyjak, chudjak, 1boy, portrait, wojak, pink_hair"
        self.assertEqual(ensure_training_caption(caption), caption)

    def test_injects_class_tokens_into_trigger_only_captions(self):
        self.assertEqual(
            ensure_training_caption("soyjak, cobson"),
            "soyjak, cobson, 1boy, portrait, wojak",
        )

    def test_empty_caption_becomes_trigger(self):
        self.assertEqual(ensure_trigger_prefix(""), "soyjak")
        self.assertEqual(
            ensure_training_caption(""),
            "soyjak, 1boy, portrait, wojak",
        )

    def test_descriptive_tokens_ignore_recipe(self):
        self.assertEqual(
            descriptive_tokens("soyjak, cobson, 1boy, portrait, wojak, forest"),
            ["cobson", "forest"],
        )


class TestPatchCaptionDir(unittest.TestCase):
    def test_rewrites_legacy_sidecars_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "1.txt"
            already = root / "2.txt"
            expected = "soyjak, chudjak, 1boy, portrait, wojak, screaming"
            expected_already = "soyjak, cobson, 1boy, portrait, wojak"
            legacy.write_text("chudjak, screaming\n", encoding="utf-8")
            already.write_text(expected_already, encoding="utf-8")

            first = patch_caption_dir(root)
            second = patch_caption_dir(root)

            self.assertEqual(first["patched"], 1)
            self.assertEqual(first["already_prefixed"], 1)
            self.assertEqual(second["patched"], 0)
            self.assertEqual(second["already_prefixed"], 2)
            self.assertEqual(legacy.read_text(encoding="utf-8"), expected)
            self.assertEqual(already.read_text(encoding="utf-8"), expected_already)


class TestTrainRecipe(unittest.TestCase):
    def test_dataset_toml_pins_trigger_and_dropout(self):
        text = Path("train/train_lora.sh").read_text(encoding="utf-8")
        self.assertIn("keep_tokens = 2", text)
        self.assertIn("caption_dropout_rate = 0.15", text)
        self.assertIn("caption_tag_dropout_rate = 0.10", text)
        self.assertIn("ensure_trigger_captions.py", text)
        self.assertIn("sdxl_train.py", text)
        self.assertNotIn("sdxl_train_network.py", text)

    def test_full_config_is_domain_takeover_finetune(self):
        text = Path("train/config.toml").read_text(encoding="utf-8")
        self.assertIn("output_name = \"soyjak-sdxl-ft\"", text)
        self.assertIn("train_text_encoder = true", text)
        self.assertIn("optimizer_type = \"adafactor\"", text)
        self.assertIn("learning_rate = 1e-5", text)
        self.assertIn("max_train_steps = 12000", text)
        self.assertNotIn("network_module", text)
        self.assertNotIn("network_dim", text)

    def test_sample_prompts_include_class_hijack_and_extra_objects(self):
        text = Path("train/sample_prompts.txt").read_text(encoding="utf-8")
        self.assertIn("soyjak, cobson, red bicycle, forest", text)
        self.assertIn("1boy, portrait, crying", text)
        self.assertIn("wojak, smiling", text)
        self.assertEqual(CLASS_HIJACK_TOKENS, ("1boy", "portrait", "wojak"))
        self.assertEqual(TRIGGER_TOKEN, "soyjak")


if __name__ == "__main__":
    unittest.main()
