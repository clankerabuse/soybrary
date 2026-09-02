import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SOYBRARY_DATA_DIR", tempfile.mkdtemp(prefix="soybrary-test-"))

import library  # noqa: E402
from PIL import Image  # noqa: E402


class TestLoadConfig(unittest.TestCase):
    def test_missing_file_returns_defaults(self):
        self.assertEqual(library.load_config("/nonexistent/config.json"),
                         library.DEFAULT_CONFIG)

    def test_merges_with_defaults(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"concurrency": 10, "custom_key": 42}, f)
            path = f.name
        try:
            config = library.load_config(path)
            self.assertEqual(config["concurrency"], 10)
            self.assertEqual(config["custom_key"], 42)
            self.assertEqual(config["data_dir"], library.DEFAULT_CONFIG["data_dir"])
        finally:
            os.unlink(path)

    def test_unreadable_file_returns_defaults(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{not json")
            path = f.name
        try:
            self.assertEqual(library.load_config(path), library.DEFAULT_CONFIG)
        finally:
            os.unlink(path)

    def test_defaults_are_not_mutated(self):
        config = library.load_config("/nonexistent/config.json")
        config["concurrency"] = 999
        self.assertNotEqual(library.DEFAULT_CONFIG["concurrency"], 999)


class TestSafeExtension(unittest.TestCase):
    def test_normal_extensions(self):
        self.assertEqual(library.safe_extension("PNG"), "png")
        self.assertEqual(library.safe_extension(".Jpg"), "jpg")
        self.assertEqual(library.safe_extension(" webm "), "webm")

    def test_rejects_traversal_and_separators(self):
        for bad in ("../../etc/passwd", "png/../..", "p\\ng", "png;rm", "pn g", "",
                    None, "a" * 40, "png.exe"):
            self.assertIsNone(library.safe_extension(bad), bad)

    def test_video_detection(self):
        self.assertTrue(library.is_video_extension("mp4"))
        self.assertTrue(library.is_video_extension("png", "video/quicktime"))
        self.assertFalse(library.is_video_extension("png", "image/png"))
        self.assertFalse(library.is_video_extension(None, None))


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db = library.Database(":memory:")

    def tearDown(self):
        self.db.close()

    def test_posts_table_and_index_exist(self):
        names = {r[0] for r in self.db.conn.execute(
            "SELECT name FROM sqlite_master WHERE name IN ('posts', 'idx_posts_status_id')")}
        self.assertEqual(names, {"posts", "idx_posts_status_id"})

    def test_get_status_missing_post_returns_none(self):
        self.assertIsNone(self.db.get_post_status(99999))

    def test_save_and_retrieve_post(self):
        self.db.save_post(
            post_id=101, status="completed", variant="v1", subvariant="sv1",
            tags="a b", date_uploaded="2026-01-01",
            file_url="http://example.com/101.png", width=800, height=600,
            file_size=12345, image_hash="abc", mime_type="image/png",
            extension="png", uploader="alice", original_filename="101.png",
        )
        self.assertEqual(self.db.get_post_status(101), "completed")

        row = self.db.conn.execute(
            "SELECT variant, subvariant, tags, width, height, file_size, uploader,"
            " last_scraped, error_message FROM posts WHERE id = 101").fetchone()
        self.assertEqual(row[:3], ("v1", "sv1", "a b"))
        self.assertEqual(row[3:6], (800, 600, 12345))
        self.assertEqual(row[6], "alice")
        self.assertIsNotNone(row[7])
        self.assertIsNone(row[8])

    def test_upsert_overwrites_on_conflict(self):
        self.db.save_post(200, "failed", error_message="old error")
        self.db.save_post(200, "completed", variant="fixed")
        row = self.db.conn.execute(
            "SELECT status, variant, error_message FROM posts WHERE id = 200").fetchone()
        self.assertEqual(row, ("completed", "fixed", None))

    def test_close_is_idempotent(self):
        self.db.close()
        self.db.close()

    def test_get_done_ids_only_returns_finished_work(self):
        self.db.save_post(1, "completed")
        self.db.save_post(2, "empty")
        self.db.save_post(3, "failed")
        self.db.save_post(4, "skipped")
        self.db.save_post(99, "completed")
        self.assertEqual(self.db.get_done_ids(1, 10), {1, 2, 4})

    def test_get_resume_id_ignores_failures(self):
        self.assertIsNone(self.db.get_resume_id())
        self.db.save_post(10, "completed")
        self.db.save_post(50, "failed")
        self.assertEqual(self.db.get_resume_id(), 10)
        self.db.save_post(80, "empty")
        self.assertEqual(self.db.get_resume_id(), 80)


class TestFullTextIndex(unittest.TestCase):
    def setUp(self):
        self.db = library.Database(":memory:")

    def tearDown(self):
        self.db.close()

    def _match(self, expr):
        return {r[0] for r in self.db.conn.execute(
            "SELECT rowid FROM posts_fts WHERE posts_fts MATCH ?", (expr,))}

    def test_index_is_created(self):
        self.assertTrue(self.db.has_fts)

    def test_insert_is_indexed(self):
        self.db.save_post(1, "completed", tags="gapejak smug", variant="cobson",
                          uploader="anon")
        self.assertEqual(self._match("gapejak"), {1})
        self.assertEqual(self._match("variant : cobson"), {1})
        self.assertEqual(self._match("uploader : anon"), {1})

    def test_update_replaces_indexed_terms(self):
        self.db.save_post(1, "completed", tags="gapejak")
        self.db.save_post(1, "completed", tags="chudjak")
        self.assertEqual(self._match("gapejak"), set())
        self.assertEqual(self._match("chudjak"), {1})

    def test_rebuild_indexes_preexisting_rows(self):
        path = Path(tempfile.mkdtemp(prefix="soybrary-fts-")) / "test.db"
        raw = sqlite3.connect(path)
        raw.execute("CREATE TABLE posts (id INTEGER PRIMARY KEY, status TEXT,"
                    " variant TEXT, subvariant TEXT, tags TEXT, uploader TEXT)")
        raw.execute("INSERT INTO posts (id, status, tags) VALUES (7, 'completed', 'legacy')")
        raw.commit()
        raw.close()

        db = library.Database(path)
        try:
            rows = {r[0] for r in db.conn.execute(
                "SELECT rowid FROM posts_fts WHERE posts_fts MATCH 'legacy'")}
            self.assertEqual(rows, {7})
        finally:
            db.close()


class TestMediaLookup(unittest.TestCase):
    def setUp(self):
        library.ensure_dirs()
        library._media_cache.clear()
        self.image = library.IMAGES_DIR / "5001.png"
        Image.new("RGB", (40, 30), "red").save(self.image)
        self.video = library.VIDEOS_DIR / "5002.mp4"
        self.video.write_bytes(b"\x00" * 32)

    def tearDown(self):
        library._media_cache.clear()
        for f in (self.image, self.video):
            f.unlink(missing_ok=True)

    def test_finds_image_by_extension(self):
        found = library.find_media(5001, "png", lookup_db=False)
        self.assertEqual(found, (self.image, False))

    def test_finds_video_by_extension(self):
        found = library.find_media(5002, "mp4", lookup_db=False)
        self.assertEqual(found, (self.video, True))

    def test_falls_back_to_scan_when_extension_unknown(self):
        found = library.find_media(5001, None, lookup_db=False)
        self.assertEqual(found, (self.image, False))

    def test_missing_post_returns_none(self):
        self.assertIsNone(library.find_media(999999, "png", lookup_db=False))

    def test_traversal_extension_is_rejected(self):
        outside = library.DATA_DIR / "secret.txt"
        outside.write_text("secret")
        try:
            for bad in ("../secret.txt", "../../etc/passwd", "png/../../secret.txt"):
                self.assertIsNone(library.find_media(5001, bad, lookup_db=False), bad)
        finally:
            outside.unlink()

    def test_result_is_cached(self):
        first = library.find_media(5001, "png", lookup_db=False)
        self.assertIn(5001, library._media_cache)
        self.assertEqual(library.find_media(5001), first)

    def test_media_path_for_builds_direct_path(self):
        self.assertEqual(library.media_path_for(42, "png"), library.IMAGES_DIR / "42.png")
        self.assertEqual(library.media_path_for(42, "webm"), library.VIDEOS_DIR / "42.webm")
        self.assertIsNone(library.media_path_for(42, "../x"))


class TestThumbnails(unittest.TestCase):
    def setUp(self):
        library.ensure_dirs()
        library._media_cache.clear()
        library._unthumbnailable.clear()
        self.post_id = 6001
        self.image = library.IMAGES_DIR / f"{self.post_id}.png"
        Image.new("RGB", (1200, 800), "blue").save(self.image)
        self.thumb = library.thumbnail_path(self.post_id)
        self.thumb.unlink(missing_ok=True)

    def tearDown(self):
        self.image.unlink(missing_ok=True)
        self.thumb.unlink(missing_ok=True)
        library._media_cache.clear()
        library._unthumbnailable.clear()

    def test_generates_bounded_thumbnail(self):
        result = library.ensure_thumbnail(self.post_id, "png", "image/png")
        self.assertEqual(result, self.thumb)
        with Image.open(self.thumb) as img:
            self.assertLessEqual(max(img.size), library.config["thumbnail_size"])
            self.assertEqual(img.format, "JPEG")

    def test_reuses_existing_thumbnail(self):
        library.ensure_thumbnail(self.post_id, "png", "image/png")
        mtime = self.thumb.stat().st_mtime_ns
        library.ensure_thumbnail(self.post_id, "png", "image/png")
        self.assertEqual(self.thumb.stat().st_mtime_ns, mtime)

    def test_thumbnail_from_bytes(self):
        data = self.image.read_bytes()
        self.assertEqual(library.thumbnail_from_bytes(self.post_id, data), self.thumb)
        self.assertTrue(self.thumb.exists())

    def test_undecodable_file_is_remembered(self):
        broken_id = 6002
        broken = library.IMAGES_DIR / f"{broken_id}.swf"
        broken.write_bytes(b"CWS not really flash")
        try:
            self.assertIsNone(library.ensure_thumbnail(broken_id, "swf", "application/x-shockwave-flash"))
            self.assertIn(broken_id, library._unthumbnailable)
        finally:
            broken.unlink(missing_ok=True)

    def test_missing_media_returns_none(self):
        self.assertIsNone(library.ensure_thumbnail(999999, "png", "image/png"))

    def test_no_partial_file_left_behind(self):
        library.ensure_thumbnail(self.post_id, "png", "image/png")
        leftovers = [p for p in library.THUMBNAILS_DIR.iterdir() if p.suffix != ".jpg"]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
