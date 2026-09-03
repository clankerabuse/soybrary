import asyncio
import io
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("SOYBRARY_DATA_DIR", tempfile.mkdtemp(prefix="soybrary-test-"))

import library  # noqa: E402
import scraper  # noqa: E402
from PIL import Image  # noqa: E402


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


def _real_image(fmt="PNG", size=(20, 15), **save_kwargs):
    buf = io.BytesIO()
    Image.new("RGB", size, "green").save(buf, fmt, **save_kwargs)
    return buf.getvalue()


def _animated_gif():
    buf = io.BytesIO()
    frames = [Image.new("RGB", (8, 8), c).convert("P")
              for c in ("red", "blue", "green")]
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], duration=50)
    return buf.getvalue()


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

    def test_missing_mime_and_ext(self):
        self.assertFalse(scraper.validate_magic_bytes(b"data", None, None))


# ---------------------------------------------------------------------------
# verify_and_sanitize_image
# ---------------------------------------------------------------------------

class TestVerifyAndSanitizeImage(unittest.TestCase):
    def setUp(self):
        self._orig = scraper.config["sanitize_images"]

    def tearDown(self):
        scraper.config["sanitize_images"] = self._orig

    def test_valid_image_passes_without_sanitization(self):
        scraper.config["sanitize_images"] = False
        data = _real_image("PNG")
        self.assertEqual(scraper.verify_and_sanitize_image(data, "image/png", "png"), data)

    def test_magic_bytes_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            scraper.verify_and_sanitize_image(b"bad", "image/png", "png")
        self.assertIn("Magic bytes signature mismatch", str(ctx.exception))

    def test_pillow_verify_failure_raises(self):
        truncated = _make_png(b"garbage that is not a png body")
        with self.assertRaises(ValueError) as ctx:
            scraper.verify_and_sanitize_image(truncated, "image/png", "png")
        self.assertIn("Pillow verification failed", str(ctx.exception))

    def test_sanitize_strips_metadata(self):
        scraper.config["sanitize_images"] = True
        exif = Image.Exif()
        exif[270] = "secret description"
        data = _real_image("JPEG", exif=exif.tobytes())
        self.assertIn(b"secret description", data)

        cleaned = scraper.verify_and_sanitize_image(data, "image/jpeg", "jpg")
        self.assertNotIn(b"secret description", cleaned)
        with Image.open(io.BytesIO(cleaned)) as img:
            self.assertEqual(img.size, (20, 15))

    def test_sanitize_keeps_jpeg_quality(self):
        scraper.config["sanitize_images"] = True
        data = _real_image("JPEG", size=(240, 180), quality=95)
        cleaned = scraper.verify_and_sanitize_image(data, "image/jpeg", "jpg")
        # A default-quality re-encode would shrink the file noticeably.
        self.assertGreater(len(cleaned), len(data) * 0.8)

    def test_sanitize_preserves_animation(self):
        scraper.config["sanitize_images"] = True
        data = _animated_gif()
        result = scraper.verify_and_sanitize_image(data, "image/gif", "gif")
        self.assertEqual(result, data)
        with Image.open(io.BytesIO(result)) as img:
            self.assertEqual(img.n_frames, 3)

    def test_animated_webp_is_not_flattened(self):
        scraper.config["sanitize_images"] = True
        buf = io.BytesIO()
        frames = [Image.new("RGB", (8, 8), c) for c in ("red", "blue", "green")]
        frames[0].save(buf, "WEBP", save_all=True, append_images=frames[1:], duration=50)
        data = buf.getvalue()

        result = scraper.verify_and_sanitize_image(data, "image/webp", "webp")
        with Image.open(io.BytesIO(result)) as img:
            self.assertEqual(getattr(img, "n_frames", 1), 3)

    def test_non_image_mime_skips_pillow(self):
        data = _make_webm()
        self.assertEqual(scraper.verify_and_sanitize_image(data, "video/webm", "webm"), data)


class TestResolveExtension(unittest.TestCase):
    def test_from_mime(self):
        self.assertEqual(scraper.resolve_extension("image/png", ""), "png")

    def test_jpeg_normalised(self):
        self.assertEqual(scraper.resolve_extension("image/jpeg", ""), "jpg")

    def test_filename_wins(self):
        self.assertEqual(scraper.resolve_extension("image/png", "photo.webp"), "webp")

    def test_unknown_mime(self):
        self.assertEqual(scraper.resolve_extension("nonsense", ""), "bin")

    def test_traversal_in_filename_is_neutralised(self):
        for name in ("evil.../../../etc/passwd", "evil.pn/g", "evil." + "x" * 50):
            self.assertEqual(scraper.resolve_extension("image/png", name), "bin", name)


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

class ScrapePostTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig_db = scraper.db
        self.db = library.Database(":memory:")
        scraper.db = self.db
        self._orig_config = dict(scraper.config)
        scraper.config["pregenerate_thumbnails"] = False
        self._written = []

    def tearDown(self):
        scraper.db = self._orig_db
        self.db.close()
        scraper.config.clear()
        scraper.config.update(self._orig_config)
        for path in self._written:
            try:
                os.unlink(path)
            except OSError:
                pass

    def track(self, post_id, ext):
        self._written.append(library.IMAGES_DIR / f"{post_id}.{ext}")
        self._written.append(library.VIDEOS_DIR / f"{post_id}.{ext}")
        self._written.append(library.METADATA_DIR / f"{post_id}.json")

    def fake_post(self, post_id, **overrides):
        data = {
            "id": post_id,
            "mimeType": "image/png",
            "originalFileName": f"{post_id}.png",
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
        }
        data.update(overrides)
        s = MagicMock()
        s.get_json = AsyncMock(return_value={"status": 200, "data": data})
        s.download_file = AsyncMock(return_value=(_real_image("PNG"), 200))
        return s


class TestScrapePost(ScrapePostTestCase):
    async def test_skips_already_completed(self):
        self.db.save_post(1, "completed")
        self.assertEqual(await scraper.scrape_post(MagicMock(), 1), "skipped")

    async def test_skips_already_skipped(self):
        self.db.save_post(2, "skipped")
        self.assertEqual(await scraper.scrape_post(MagicMock(), 2), "skipped")

    async def test_skips_already_empty(self):
        self.db.save_post(3, "empty")
        self.assertEqual(await scraper.scrape_post(MagicMock(), 3), "skipped")

    async def test_retries_failed_posts(self):
        self.db.save_post(4, "failed")
        self.track(4, "png")
        self.assertEqual(await scraper.scrape_post(self.fake_post(4), 4), "completed")

    async def test_completed_flow(self):
        self.track(100, "png")
        self.assertEqual(await scraper.scrape_post(self.fake_post(100), 100), "completed")

        row = self.db.conn.execute(
            "SELECT status, variant, subvariant, tags, uploader, extension"
            " FROM posts WHERE id = 100").fetchone()
        self.assertEqual(row, ("completed", "wide", "hdr", "landscape", "bob", "png"))
        self.assertTrue((library.IMAGES_DIR / "100.png").exists())

    async def test_video_is_written_to_the_video_directory(self):
        self.track(101, "webm")
        s = self.fake_post(101, mimeType="video/webm", originalFileName="clip.webm")
        s.download_file = AsyncMock(return_value=(_make_webm(b"\x00" * 64), 200))
        scraper.config["validate_videos"] = False

        self.assertEqual(await scraper.scrape_post(s, 101), "completed")
        self.assertTrue((library.VIDEOS_DIR / "101.webm").exists())
        self.assertFalse((library.IMAGES_DIR / "101.webm").exists())

    async def test_metadata_written_to_disk(self):
        self.track(500, "png")
        await scraper.scrape_post(self.fake_post(500), 500)

        meta = json.loads((library.METADATA_DIR / "500.json").read_text())
        self.assertEqual(meta["postNumber"], 500)
        self.assertEqual(meta["uploader"], "bob")
        self.assertEqual(meta["tags"], ["landscape"])
        self.assertEqual(meta["variants"], ["wide"])

    async def test_thumbnail_is_pregenerated(self):
        scraper.config["pregenerate_thumbnails"] = True
        self.track(600, "png")
        thumb = library.thumbnail_path(600)
        thumb.unlink(missing_ok=True)
        try:
            await scraper.scrape_post(self.fake_post(600), 600)
            self.assertTrue(thumb.exists())
        finally:
            thumb.unlink(missing_ok=True)

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
        s = self.fake_post(301)
        s.download_file = AsyncMock(return_value=(None, 404))
        self.assertEqual(await scraper.scrape_post(s, 301), "failed")

    async def test_verification_failure_returns_failed(self):
        s = self.fake_post(302)
        s.download_file = AsyncMock(return_value=(b"definitely not a png", 200))
        self.assertEqual(await scraper.scrape_post(s, 302), "failed")
        self.assertEqual(self.db.get_post_status(302), "failed")
        self.assertFalse((library.IMAGES_DIR / "302.png").exists())

    async def test_extension_from_original_filename(self):
        self.track(400, "webp")
        s = self.fake_post(400, mimeType="image/webp", originalFileName="photo.webp")
        s.download_file = AsyncMock(return_value=(_real_image("WEBP"), 200))
        await scraper.scrape_post(s, 400)
        self.assertEqual(
            self.db.conn.execute("SELECT extension FROM posts WHERE id = 400").fetchone()[0],
            "webp")

    async def test_jpeg_extension_normalized(self):
        self.track(401, "jpg")
        s = self.fake_post(401, mimeType="image/jpeg", originalFileName="")
        s.download_file = AsyncMock(return_value=(_real_image("JPEG"), 200))
        await scraper.scrape_post(s, 401)
        self.assertEqual(
            self.db.conn.execute("SELECT extension FROM posts WHERE id = 401").fetchone()[0],
            "jpg")

    async def test_hostile_filename_cannot_escape_the_media_directory(self):
        self.track(402, "bin")
        s = self.fake_post(402, originalFileName="x.../../../../tmp/pwned")
        await scraper.scrape_post(s, 402)
        self.assertEqual(
            self.db.conn.execute("SELECT extension FROM posts WHERE id = 402").fetchone()[0],
            "bin")
        self.assertTrue((library.IMAGES_DIR / "402.bin").exists())
        self.assertFalse(os.path.exists("/tmp/pwned"))

    async def test_exception_in_get_json_returns_failed(self):
        s = MagicMock()
        s.get_json = AsyncMock(side_effect=ConnectionError("network down"))
        self.assertEqual(await scraper.scrape_post(s, 600), "failed")
        self.assertEqual(self.db.get_post_status(600), "failed")

    async def test_post_is_searchable_after_scraping(self):
        self.track(700, "png")
        await scraper.scrape_post(self.fake_post(700), 700)
        rows = self.db.conn.execute(
            "SELECT rowid FROM posts_fts WHERE posts_fts MATCH 'landscape'").fetchall()
        self.assertEqual([r[0] for r in rows], [700])


# ---------------------------------------------------------------------------
# Queue planning
# ---------------------------------------------------------------------------

class TestPendingIds(unittest.TestCase):
    def setUp(self):
        self._orig_db = scraper.db
        self.db = library.Database(":memory:")
        scraper.db = self.db

    def tearDown(self):
        scraper.db = self._orig_db
        self.db.close()

    def test_skips_finished_posts(self):
        self.db.save_post(2, "completed")
        self.db.save_post(3, "empty")
        self.db.save_post(4, "failed")
        self.assertEqual(scraper.pending_ids(1, 5), [1, 4, 5])

    def test_respects_limit(self):
        self.assertEqual(scraper.pending_ids(1, 100, limit=3), [1, 2, 3])

    def test_empty_when_everything_is_done(self):
        for pid in range(1, 4):
            self.db.save_post(pid, "completed")
        self.assertEqual(scraper.pending_ids(1, 3), [])

    def test_queue_planning_is_one_query(self):
        for pid in range(1, 500):
            self.db.save_post(pid, "completed")
        with patch.object(self.db, "get_done_ids", wraps=self.db.get_done_ids) as spy:
            scraper.pending_ids(1, 500)
        self.assertEqual(spy.call_count, 1)

    def test_resume_id_prefers_database(self):
        self.db.save_post(42, "completed")
        self.assertEqual(scraper.resume_start_id(), 42)


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

class TestWorker(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig_db = scraper.db
        self.db = library.Database(":memory:")
        scraper.db = self.db

    def tearDown(self):
        scraper.db = self._orig_db
        self.db.close()

    async def _drain(self, coro_factory, queue):
        task = asyncio.create_task(coro_factory())
        await asyncio.wait_for(queue.join(), timeout=5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_worker_increments_skipped(self):
        queue = asyncio.Queue()
        await queue.put(1)
        self.db.save_post(1, "completed")
        stats = dict(scraper.STATS_TEMPLATE)

        await self._drain(lambda: scraper.worker(queue, MagicMock(), stats, 0), queue)
        self.assertEqual(stats["skipped"], 1)

    async def test_worker_increments_failed(self):
        queue = asyncio.Queue()
        await queue.put(999)
        s = MagicMock()
        s.get_json = AsyncMock(side_effect=ConnectionError("down"))
        stats = dict(scraper.STATS_TEMPLATE)

        await self._drain(lambda: scraper.worker(queue, s, stats, 0), queue)
        self.assertEqual(stats["failed"], 1)


class TestScrapeJob(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig_db = scraper.db
        self.db = library.Database(":memory:")
        scraper.db = self.db

    def tearDown(self):
        scraper.db = self._orig_db
        self.db.close()

    def _job(self):
        return scraper.ScrapeJob(start_id=1, end_id=10, main_loop=asyncio.get_event_loop())

    async def test_worker_marks_every_queue_item_done(self):
        job = self._job()
        queue = asyncio.Queue()
        for pid in (1, 2, 3):
            queue.put_nowait(pid)
            self.db.save_post(pid, "completed")

        task = asyncio.create_task(job._worker(queue, MagicMock(), 0))
        await asyncio.wait_for(queue.join(), timeout=5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self.assertEqual(job.stats["skipped"], 3)

    async def test_progress_message_without_end_id(self):
        job = scraper.ScrapeJob(start_id=1, end_id=None, main_loop=asyncio.get_event_loop())
        self.assertIn("Scraped 5: COMPLETED", job._progress_message(5, "completed", 1))

    async def test_progress_message_percentage(self):
        job = self._job()
        self.assertIn("50.00%", job._progress_message(5, "completed", 1))

    async def test_cancel_flag_is_reported(self):
        job = self._job()
        job.cancel()
        self.assertTrue(job.get_status()["cancelled"])


class FakePage:
    """Stands in for the Playwright page a scrape drives."""

    def __init__(self, posts, latest_id):
        self.posts = posts
        self.latest_id = latest_id

    async def goto(self, url, timeout=None):
        return None

    async def content(self):
        return f'<a href="/post/view/{self.latest_id}">latest</a>'

    async def evaluate(self, script, url):
        import base64
        post_id = int(url.rstrip("/file").rsplit("/", 1)[-1])
        if post_id not in self.posts:
            return {"status": 404, "data": None}
        if url.endswith("/file"):
            return {"status": 200, "data": base64.b64encode(_real_image("PNG")).decode()}
        return {"status": 200, "data": self.posts[post_id]}


def _fake_playwright(page):
    browser = MagicMock()
    browser.close = AsyncMock()
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    browser.new_context = AsyncMock(return_value=context)
    chromium = MagicMock()
    chromium.launch = AsyncMock(return_value=browser)

    driver = MagicMock()
    driver.chromium = chromium
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=driver)
    manager.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=manager), browser


class TestScrapeJobEndToEnd(unittest.IsolatedAsyncioTestCase):
    """Drive a whole scrape session with a stubbed browser."""

    def setUp(self):
        self._orig_db = scraper.db
        self.db = library.Database(":memory:")
        scraper.db = self.db
        self._orig_config = dict(scraper.config)
        scraper.config.update({"delay_ms": 0, "concurrency": 2,
                               "pregenerate_thumbnails": False})
        self.ids = [7001, 7002, 7003]
        self.posts = {
            pid: {
                "id": pid,
                "mimeType": "image/png",
                "originalFileName": f"{pid}.png",
                "width": 10, "height": 10,
                "uploadedAt": "2026-01-01T00:00:00Z",
                "uploader": {"userName": "anon"},
                "tags": [{"name": "gapejak", "category": "general"}],
            }
            for pid in self.ids[:2]  # the third id is a gap in the catalog
        }

    def tearDown(self):
        scraper.db = self._orig_db
        self.db.close()
        scraper.config.clear()
        scraper.config.update(self._orig_config)
        for pid in self.ids:
            (library.IMAGES_DIR / f"{pid}.png").unlink(missing_ok=True)
            (library.METADATA_DIR / f"{pid}.json").unlink(missing_ok=True)

    async def _run(self, **kwargs):
        page = FakePage(self.posts, latest_id=self.ids[-1])
        fake, browser = _fake_playwright(page)
        queue = asyncio.Queue()
        job = scraper.ScrapeJob(progress_queue=queue,
                                main_loop=asyncio.get_running_loop(), **kwargs)
        with patch.object(scraper, "async_playwright", fake), \
             patch.object(scraper, "ACTIVATION_WAIT_SECONDS", 0):
            await asyncio.wait_for(job.run(), timeout=30)

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        return job, events, browser

    async def test_scrapes_a_range(self):
        job, events, browser = await self._run(start_id=self.ids[0], end_id=self.ids[-1])

        self.assertEqual(job.stats, {"completed": 2, "empty": 1, "failed": 0, "skipped": 0})
        self.assertFalse(job.running)
        for pid in self.ids[:2]:
            self.assertEqual(self.db.get_post_status(pid), "completed")
            self.assertTrue((library.IMAGES_DIR / f"{pid}.png").exists())
        self.assertEqual(self.db.get_post_status(self.ids[-1]), "empty")
        browser.close.assert_awaited()

    async def test_emits_progress_events(self):
        _, events, _ = await self._run(start_id=self.ids[0], end_id=self.ids[-1])
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[-1], "complete")
        self.assertEqual(kinds.count("post_done"), 3)
        self.assertIn("status", kinds)
        self.assertNotIn("error", kinds)

    async def test_second_run_skips_finished_work(self):
        await self._run(start_id=self.ids[0], end_id=self.ids[-1])
        job, events, _ = await self._run(start_id=self.ids[0], end_id=self.ids[-1])

        self.assertEqual(sum(job.stats.values()), 0)
        self.assertEqual(job.total_queue, 0)
        self.assertTrue(any("No new posts" in e["data"].get("message", "")
                            for e in events if e["type"] == "status"))

    async def test_limit_caps_the_queue(self):
        job, _, _ = await self._run(start_id=self.ids[0], end_id=self.ids[-1], limit=1)
        self.assertEqual(job.total_queue, 1)
        self.assertEqual(sum(job.stats.values()), 1)

    async def test_detects_the_latest_id(self):
        job, _, _ = await self._run(start_id=self.ids[0])
        self.assertEqual(job.end_id, self.ids[-1])

    async def test_browser_failure_is_reported(self):
        fake, _ = _fake_playwright(FakePage({}, 1))
        fake.return_value.__aenter__ = AsyncMock(side_effect=RuntimeError("no browser"))
        queue = asyncio.Queue()
        job = scraper.ScrapeJob(start_id=1, end_id=2, progress_queue=queue,
                                main_loop=asyncio.get_running_loop())
        with patch.object(scraper, "async_playwright", fake):
            await asyncio.wait_for(job.run(), timeout=30)

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        errors = [e for e in events if e["type"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("no browser", errors[0]["data"]["message"])
        self.assertFalse(job.running)


class TestScrapeDelay(unittest.TestCase):
    def test_jitter_stays_within_bounds(self):
        for _ in range(200):
            delay = scraper.scrape_delay(1000)
            self.assertGreaterEqual(delay, 0.1)
            self.assertLessEqual(delay, 1.3)

    def test_short_delays_have_a_floor(self):
        self.assertGreaterEqual(scraper.scrape_delay(0), 0.1)


if __name__ == "__main__":
    unittest.main()
