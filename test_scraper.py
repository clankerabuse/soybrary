import os
import io
import json
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

with patch("pathlib.Path.mkdir"), patch("sqlite3.connect"):
    import scraper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(data=b"fake"):
    return b"\x89PNG\r\n\x1a\n" + data

def _make_jpeg(data=b"fake"):
    return b"\xff\xd8\xff" + data

def _make_gif87a(data=b"fake"):
    return b"GIF87a" + data

def _make_gif89a(data=b"fake"):
    return b"GIF89a" + data

def _make_webp(data=b"fake"):
    return b"RIFF" + b"\x00" * 4 + b"WEBP" + data

def _make_webm(data=b"fake"):
    return b"\x1a\x45\xdf\xa3" + data

def _make_mp4(data=b"fake"):
    return b"\x00" * 4 + b"ftyp" + data


# ---------------------------------------------------------------------------
# validate_magic_bytes
# ---------------------------------------------------------------------------

class TestValidateMagicBytes(unittest.TestCase):
    # -- PNG --
    def test_png_valid(self):
        self.assertTrue(scraper.validate_magic_bytes(_make_png(), "image/png", "png"))

    def test_png_case_insensitive(self):
        self.assertTrue(scraper.validate_magic_bytes(_make_png(), "IMAGE/PNG", "PNG"))

    def test_png_invalid_data(self):
        self.assertFalse(scraper.validate_magic_bytes(b"not_png", "image/png", "png"))

    def test_png_ext_only(self):
        self.assertTrue(scraper.validate_magic_bytes(_make_png(), "application/octet-stream", "png"))

    # -- JPEG --
    def test_jpeg_valid(self):
        self.assertTrue(scraper.validate_magic_bytes(_make_jpeg(), "image/jpeg", "jpg"))

    def test_jpeg_jpg_mime(self):
        self.assertTrue(scraper.validate_magic_bytes(_make_jpeg(), "image/jpg", "jpeg"))

    def test_jpeg_invalid(self):
        self.assertFalse(scraper.validate_magic_bytes(b"bad", "image/jpeg", "jpg"))

    # -- GIF --
    def test_gif87a(self):
        self.assertTrue(scraper.validate_magic_bytes(_make_gif87a(), "image/gif", "gif"))

    def test_gif89a(self):
        self.assertTrue(scraper.validate_magic_bytes(_make_gif89a(), "image/gif", "gif"))

    def test_gif_invalid(self):
        self.assertFalse(scraper.validate_magic_bytes(b"GIF000", "image/gif", "gif"))

    # -- WEBP --
    def test_webp_valid(self):
        self.assertTrue(scraper.validate_magic_bytes(_make_webp(), "image/webp", "webp"))

    def test_webp_too_short(self):
        short = b"RIFF" + b"\x00" * 2 + b"WEBP"
        self.assertFalse(scraper.validate_magic_bytes(short, "image/webp", "webp"))

    def test_webp_wrong_tag(self):
        data = b"RIFF" + b"\x00" * 4 + b"WEBA" + b"\x00" * 10
        self.assertFalse(scraper.validate_magic_bytes(data, "image/webp", "webp"))

    # -- WEBM --
    def test_webm_valid(self):
        self.assertTrue(scraper.validate_magic_bytes(_make_webm(), "video/webm", "webm"))

    def test_webm_invalid(self):
        self.assertFalse(scraper.validate_magic_bytes(b"not_webm", "video/webm", "webm"))

    # -- MP4 --
    def test_mp4_valid(self):
        self.assertTrue(scraper.validate_magic_bytes(_make_mp4(), "video/mp4", "mp4"))

    def test_mp4_too_short(self):
        self.assertFalse(scraper.validate_magic_bytes(b"short", "video/mp4", "mp4"))

    def test_mp4_no_ftyp(self):
        self.assertFalse(scraper.validate_magic_bytes(b"\x00" * 4 + b"xxxx" + b"\x00" * 4, "video/mp4", "mp4"))

    # -- Fallback --
    def test_unknown_format_returns_false(self):
        self.assertFalse(scraper.validate_magic_bytes(b"\x00\x01\x02\x03", "application/octet-stream", "bin"))


# ---------------------------------------------------------------------------
# verify_and_sanitize_image
# ---------------------------------------------------------------------------

class TestVerifyAndSanitizeImage(unittest.TestCase):
    @patch("scraper.Image.open")
    def test_valid_image_passes(self, mock_open):
        mock_img = MagicMock()
        mock_open.return_value = mock_img

        data = _make_png()
        result = scraper.verify_and_sanitize_image(data, "image/png", "png")
        self.assertEqual(result, data)
        mock_img.verify.assert_called_once()

    def test_magic_bytes_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            scraper.verify_and_sanitize_image(b"bad", "image/png", "png")
        self.assertIn("Magic bytes signature mismatch", str(ctx.exception))

    @patch("scraper.Image.open")
    def test_pillow_verify_failure_raises(self, mock_open):
        mock_img = MagicMock()
        mock_img.verify.side_effect = Exception("corrupt")
        mock_open.return_value = mock_img

        with self.assertRaises(ValueError) as ctx:
            scraper.verify_and_sanitize_image(_make_png(), "image/png", "png")
        self.assertIn("Pillow verification failed", str(ctx.exception))

    @patch("scraper.Image.open")
    def test_sanitize_strips_metadata(self, mock_open):
        orig = scraper.config["sanitize_images"]
        scraper.config["sanitize_images"] = True
        try:
            mock_img = MagicMock()
            mock_img.format = "PNG"
            mock_open.return_value = mock_img

            def fake_save(buf, format):
                buf.write(b"clean")
            mock_img.save.side_effect = fake_save

            result = scraper.verify_and_sanitize_image(_make_png(), "image/png", "png")
            self.assertEqual(result, b"clean")
            self.assertEqual(mock_open.call_count, 2)
        finally:
            scraper.config["sanitize_images"] = orig

    def test_non_image_mime_skips_pillow(self):
        data = _make_webm()
        result = scraper.verify_and_sanitize_image(data, "video/webm", "webm")
        self.assertEqual(result, data)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig(unittest.TestCase):
    @patch("scraper.os.path.exists", return_value=False)
    def test_no_config_file_returns_defaults(self, _):
        self.assertEqual(scraper.load_config(), scraper.DEFAULT_CONFIG)

    @patch("scraper.os.path.exists", return_value=True)
    @patch("builtins.open")
    def test_merges_with_defaults(self, mock_open, _):
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.read.return_value = json.dumps({"concurrency": 10, "custom_key": 42})
        mock_open.return_value = mock_file

        config = scraper.load_config()
        self.assertEqual(config["concurrency"], 10)
        self.assertEqual(config["custom_key"], 42)
        self.assertEqual(config["data_dir"], scraper.DEFAULT_CONFIG["data_dir"])

    @patch("scraper.os.path.exists", return_value=True)
    @patch("builtins.open", side_effect=IOError("disk error"))
    def test_read_error_returns_defaults(self, _mock_open, _mock_exists):
        self.assertEqual(scraper.load_config(), scraper.DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = scraper.Database(":memory:")

    def tearDown(self):
        self.db.close()

    def test_posts_table_exists(self):
        cur = self.db.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='posts'")
        self.assertIsNotNone(cur.fetchone())

    def test_get_status_missing_post_returns_none(self):
        self.assertIsNone(self.db.get_post_status(99999))

    def test_save_and_retrieve_post(self):
        self.db.save_post(
            post_id=101,
            status="completed",
            variant="v1",
            subvariant="sv1",
            tags="a b",
            date_uploaded="2026-01-01",
            file_url="http://example.com/101.png",
            width=800, height=600,
            file_size=12345,
            image_hash="abc",
            mime_type="image/png",
            extension="png",
            uploader="alice",
            original_filename="101.png",
        )
        self.assertEqual(self.db.get_post_status(101), "completed")

        cur = self.db.conn.cursor()
        cur.execute("SELECT * FROM posts WHERE id = 101")
        row = cur.fetchone()
        self.assertEqual(row[0], 101)
        self.assertEqual(row[2], "v1")
        self.assertEqual(row[3], "sv1")
        self.assertEqual(row[4], "a b")
        self.assertEqual(row[7], 800)
        self.assertEqual(row[8], 600)
        self.assertEqual(row[9], 12345)
        self.assertEqual(row[13], "alice")
        self.assertIsNotNone(row[15])  # last_scraped
        self.assertIsNone(row[16])    # error_message

    def test_upsert_overwrites_on_conflict(self):
        self.db.save_post(200, "failed", error_message="old error")
        self.assertEqual(self.db.get_post_status(200), "failed")

        self.db.save_post(200, "completed", variant="fixed")
        self.assertEqual(self.db.get_post_status(200), "completed")

        cur = self.db.conn.cursor()
        cur.execute("SELECT variant, error_message FROM posts WHERE id = 200")
        row = cur.fetchone()
        self.assertEqual(row[0], "fixed")
        self.assertIsNone(row[1])

    def test_save_with_minimal_fields(self):
        self.db.save_post(300, "skipped")
        self.assertEqual(self.db.get_post_status(300), "skipped")

        cur = self.db.conn.cursor()
        cur.execute("SELECT variant, tags, error_message FROM posts WHERE id = 300")
        row = cur.fetchone()
        self.assertIsNone(row[0])
        self.assertIsNone(row[1])
        self.assertIsNone(row[2])

    def test_close_is_idempotent(self):
        self.db.close()
        self.db.close()  # should not raise


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------

class TestScraperClass(unittest.IsolatedAsyncioTestCase):
    async def test_get_json_success(self):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value={"status": 200, "data": {"id": 1}})

        s = scraper.Scraper(page)
        res = await s.get_json("http://example.com/api")
        self.assertEqual(res["status"], 200)
        self.assertEqual(res["data"]["id"], 1)

    async def test_get_json_not_found(self):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value={"status": 404, "data": None})

        s = scraper.Scraper(page)
        res = await s.get_json("http://example.com/api")
        self.assertEqual(res["status"], 404)
        self.assertIsNone(res["data"])

    async def test_download_file_success(self):
        import base64
        raw = b"binary_content"
        encoded = base64.b64encode(raw).decode()

        page = MagicMock()
        page.evaluate = AsyncMock(return_value={"status": 200, "data": encoded})

        s = scraper.Scraper(page)
        data, status = await s.download_file("http://example.com/file.png")
        self.assertEqual(status, 200)
        self.assertEqual(data, raw)

    async def test_download_file_not_found(self):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value={"status": 404, "data": None})

        s = scraper.Scraper(page)
        data, status = await s.download_file("http://example.com/missing")
        self.assertIsNone(data)
        self.assertEqual(status, 404)

    async def test_download_file_server_error(self):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value={"status": 500, "data": None, "error": "boom"})

        s = scraper.Scraper(page)
        data, status = await s.download_file("http://example.com/error")
        self.assertIsNone(data)
        self.assertEqual(status, 500)


# ---------------------------------------------------------------------------
# scrape_post
# ---------------------------------------------------------------------------

class TestScrapePost(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig_db = scraper.db
        self.db = scraper.Database(":memory:")
        scraper.db = self.db
        self._orig_config = scraper.config.copy()

    def tearDown(self):
        scraper.db = self._orig_db
        self.db.close()
        scraper.config = self._orig_config

    async def test_skips_already_completed(self):
        self.db.save_post(1, "completed")
        s = MagicMock()
        self.assertEqual(await scraper.scrape_post(s, 1), "skipped")

    async def test_skips_already_skipped(self):
        self.db.save_post(2, "skipped")
        s = MagicMock()
        self.assertEqual(await scraper.scrape_post(s, 2), "skipped")

    async def test_skips_already_empty(self):
        self.db.save_post(3, "empty")
        s = MagicMock()
        self.assertEqual(await scraper.scrape_post(s, 3), "skipped")

    @patch("builtins.open")
    @patch("scraper.verify_and_sanitize_image")
    async def test_completed_flow(self, mock_verify, mock_open):
        s = MagicMock()
        s.get_json = AsyncMock(return_value={
            "status": 200,
            "data": {
                "id": 100,
                "mimeType": "image/png",
                "originalFileName": "photo.png",
                "fileSize": 5000,
                "width": 640,
                "height": 480,
                "uploadedAt": "2026-06-01T12:00:00Z",
                "uploader": {"userName": "bob"},
                "tags": [
                    {"name": "landscape", "category": "general"},
                    {"name": "wide", "category": "variant"},
                    {"name": "hdr", "category": "subvariant"},
                ],
            },
        })

        png = _make_png()
        s.download_file = AsyncMock(return_value=(png, 200))
        mock_verify.return_value = png

        mf = MagicMock()
        mf.__enter__ = MagicMock(return_value=mf)
        mock_open.return_value = mf

        self.assertEqual(await scraper.scrape_post(s, 100), "completed")
        self.assertEqual(self.db.get_post_status(100), "completed")

        cur = self.db.conn.cursor()
        cur.execute("SELECT variant, subvariant, tags, uploader FROM posts WHERE id = 100")
        row = cur.fetchone()
        self.assertEqual(row[0], "wide")
        self.assertEqual(row[1], "hdr")
        self.assertEqual(row[2], "landscape wide hdr")
        self.assertEqual(row[3], "bob")

    async def test_not_found_returns_empty(self):
        s = MagicMock()
        s.get_json = AsyncMock(return_value={"status": 404, "data": None})

        self.assertEqual(await scraper.scrape_post(s, 200), "empty")
        self.assertEqual(self.db.get_post_status(200), "empty")

    async def test_data_with_null_id_returns_empty(self):
        s = MagicMock()
        s.get_json = AsyncMock(return_value={"status": 200, "data": {"id": None}})

        self.assertEqual(await scraper.scrape_post(s, 201), "empty")

    async def test_meta_server_error_returns_failed(self):
        s = MagicMock()
        s.get_json = AsyncMock(return_value={"status": 500, "data": None, "error": "boom"})

        self.assertEqual(await scraper.scrape_post(s, 300), "failed")
        self.assertEqual(self.db.get_post_status(300), "failed")

    async def test_download_failure_returns_failed(self):
        s = MagicMock()
        s.get_json = AsyncMock(return_value={"status": 200, "data": {"id": 301, "mimeType": "image/png"}})
        s.download_file = AsyncMock(return_value=(None, 404))

        self.assertEqual(await scraper.scrape_post(s, 301), "failed")

    @patch("scraper.verify_and_sanitize_image")
    async def test_verification_failure_returns_failed(self, mock_verify):
        s = MagicMock()
        s.get_json = AsyncMock(return_value={"status": 200, "data": {"id": 302, "mimeType": "image/png"}})
        s.download_file = AsyncMock(return_value=(b"bad", 200))
        mock_verify.side_effect = ValueError("bad file")

        self.assertEqual(await scraper.scrape_post(s, 302), "failed")
        self.assertEqual(self.db.get_post_status(302), "failed")

    @patch("builtins.open")
    @patch("scraper.verify_and_sanitize_image")
    async def test_extension_from_original_filename(self, mock_verify, mock_open):
        s = MagicMock()
        s.get_json = AsyncMock(return_value={
            "status": 200,
            "data": {
                "id": 400,
                "mimeType": "image/png",
                "originalFileName": "photo.webp",
            },
        })
        s.download_file = AsyncMock(return_value=(b"\x00" * 20, 200))
        mock_verify.return_value = b"\x00" * 20

        mf = MagicMock()
        mf.__enter__ = MagicMock(return_value=mf)
        mock_open.return_value = mf

        await scraper.scrape_post(s, 400)

        cur = self.db.conn.cursor()
        cur.execute("SELECT extension FROM posts WHERE id = 400")
        self.assertEqual(cur.fetchone()[0], "webp")

    @patch("builtins.open")
    @patch("scraper.verify_and_sanitize_image")
    async def test_jpeg_extension_normalized(self, mock_verify, mock_open):
        s = MagicMock()
        s.get_json = AsyncMock(return_value={
            "status": 200,
            "data": {
                "id": 401,
                "mimeType": "image/jpeg",
            },
        })
        s.download_file = AsyncMock(return_value=(b"\x00" * 20, 200))
        mock_verify.return_value = b"\x00" * 20

        mf = MagicMock()
        mf.__enter__ = MagicMock(return_value=mf)
        mock_open.return_value = mf

        await scraper.scrape_post(s, 401)

        cur = self.db.conn.cursor()
        cur.execute("SELECT extension FROM posts WHERE id = 401")
        self.assertEqual(cur.fetchone()[0], "jpg")

    @patch("scraper.verify_and_sanitize_image")
    async def test_metadata_written_to_disk(self, mock_verify):
        s = MagicMock()
        s.get_json = AsyncMock(return_value={
            "status": 200,
            "data": {
                "id": 500,
                "mimeType": "image/png",
                "originalFileName": "500.png",
                "fileSize": 999,
                "width": 100,
                "height": 200,
                "uploadedAt": "2026-01-01T00:00:00Z",
                "uploader": {"userName": "u"},
                "tags": [{"name": "t", "category": "general"}],
            },
        })
        s.download_file = AsyncMock(return_value=(b"\x00" * 10, 200))
        mock_verify.return_value = b"\x00" * 10

        written = {}
        def fake_open(path, mode="r", **kw):
            f = MagicMock()
            f.__enter__ = MagicMock(return_value=f)
            f.__exit__ = MagicMock(return_value=False)
            if "w" in mode:
                buf = []
                f.write = lambda data: buf.append(data)
                written[str(path)] = buf
            return f

        with patch("builtins.open", side_effect=fake_open):
            await scraper.scrape_post(s, 500)

        meta_key = [k for k in written if k.endswith("500.json")][0]
        meta = json.loads("".join(written[meta_key]))
        self.assertEqual(meta["postNumber"], 500)
        self.assertEqual(meta["uploader"], "u")
        self.assertEqual(meta["tags"], ["t"])

    async def test_exception_in_get_json_returns_failed(self):
        s = MagicMock()
        s.get_json = AsyncMock(side_effect=ConnectionError("network down"))

        self.assertEqual(await scraper.scrape_post(s, 600), "failed")
        self.assertEqual(self.db.get_post_status(600), "failed")


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

class TestWorker(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig_db = scraper.db
        self.db = scraper.Database(":memory:")
        scraper.db = self.db
        self._orig_config = scraper.config.copy()

    def tearDown(self):
        scraper.db = self._orig_db
        self.db.close()
        scraper.config = self._orig_config

    async def test_worker_increments_skipped(self):
        queue = asyncio.Queue()
        await queue.put(1)

        s = MagicMock()
        stats = {"completed": 0, "skipped": 0, "empty": 0, "failed": 0}

        self.db.save_post(1, "completed")

        task = asyncio.create_task(scraper.worker(queue, s, stats, 0))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        self.assertEqual(stats["skipped"], 1)

    async def test_worker_increments_failed(self):
        queue = asyncio.Queue()
        await queue.put(999)

        s = MagicMock()
        s.get_json = AsyncMock(side_effect=ConnectionError("down"))
        stats = {"completed": 0, "skipped": 0, "empty": 0, "failed": 0}

        task = asyncio.create_task(scraper.worker(queue, s, stats, 0))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        self.assertEqual(stats["failed"], 1)


if __name__ == "__main__":
    unittest.main()
