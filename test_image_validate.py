import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_validate import check_image_bytes, check_image_path, validate_magic_bytes


class TestValidateMagicBytes(unittest.TestCase):
    def test_png(self):
        self.assertTrue(validate_magic_bytes(b"\x89PNG\r\n\x1a\n", "png"))

    def test_jpeg(self):
        self.assertTrue(validate_magic_bytes(b"\xff\xd8\xff", "jpg"))

    def test_mismatch(self):
        self.assertFalse(validate_magic_bytes(b"not-an-image", "png"))


class TestCheckImageBytes(unittest.TestCase):
    def _make_png_bytes(self, size=(64, 64), color="red") -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", size, color=color).save(buf, format="PNG")
        return buf.getvalue()

    def test_valid_png(self):
        result = check_image_bytes(self._make_png_bytes(), "png")
        self.assertTrue(result.ok)

    def test_empty_file(self):
        result = check_image_bytes(b"", "png")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "empty_file")

    def test_bad_magic(self):
        result = check_image_bytes(b"hello", "png")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "bad_magic")

    def test_truncated_png(self):
        data = self._make_png_bytes()[:-20]
        result = check_image_bytes(data, "png")
        self.assertFalse(result.ok)

    def test_too_large(self):
        result = check_image_bytes(
            self._make_png_bytes(size=(3000, 2000)),
            "png",
            max_long_side=2048,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "too_large")

    def test_palette_mode_converts(self):
        buf = io.BytesIO()
        Image.new("P", (32, 32)).save(buf, format="PNG")
        result = check_image_bytes(buf.getvalue(), "png")
        self.assertTrue(result.ok)


class TestCheckImagePath(unittest.TestCase):
    def test_reads_file_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.png"
            Image.new("RGB", (32, 32), color="blue").save(path, format="PNG")
            self.assertTrue(check_image_path(path).ok)


if __name__ == "__main__":
    unittest.main()
