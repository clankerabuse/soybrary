import io
import os
import tempfile
import unittest

os.environ.setdefault("SOYBRARY_DATA_DIR", tempfile.mkdtemp(prefix="soybrary-test-"))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import library  # noqa: E402
import search  # noqa: E402
import server  # noqa: E402

POSTS = [
    # id, tags, variant, subvariant, uploader, ext, mime
    (9001, "gapejak smug", "cobson", None, "anon1", "png", "image/png"),
    (9002, "pointing text", "chudjak", "hdr", "anon2", "png", "image/png"),
    (9003, "coal", None, None, "anon1", "gif", "image/gif"),
    (9004, "clip", None, None, "anon3", "mp4", "video/mp4"),
]


def make_image(path, size=(80, 60), fmt="PNG"):
    Image.new("RGB", size, "purple").save(path, fmt)


class ServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library.ensure_dirs()
        cls.db = library.Database(library.DB_PATH)
        for pid, tags, variant, subvariant, uploader, ext, mime in POSTS:
            cls.db.save_post(pid, "completed", tags=tags, variant=variant,
                             subvariant=subvariant, uploader=uploader,
                             mime_type=mime, extension=ext, width=80, height=60)
        cls.db.save_post(9005, "failed", tags="never")

        make_image(library.IMAGES_DIR / "9001.png")
        make_image(library.IMAGES_DIR / "9002.png")
        make_image(library.IMAGES_DIR / "9003.gif", fmt="GIF")
        (library.VIDEOS_DIR / "9004.mp4").write_bytes(b"\x00" * 64)

        search.invalidate_counts()
        search.tag_index.build()
        cls.client = TestClient(server.app)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def posts(self, **params):
        r = self.client.get("/api/posts", params=params)
        self.assertEqual(r.status_code, 200)
        return r.json()


class TestCatalog(ServerTestCase):
    def test_lists_completed_posts_newest_first(self):
        data = self.posts()
        ids = [p["id"] for p in data["posts"]]
        self.assertEqual(ids[:4], [9004, 9003, 9002, 9001])
        self.assertNotIn(9005, ids)

    def test_search_by_tag(self):
        self.assertEqual([p["id"] for p in self.posts(q="gapejak")["posts"]], [9001])

    def test_search_by_variant(self):
        self.assertEqual([p["id"] for p in self.posts(q="variant:cobson")["posts"]], [9001])

    def test_search_by_uploader(self):
        self.assertEqual({p["id"] for p in self.posts(q="anon1")["posts"]}, {9001, 9003})

    def test_search_by_id(self):
        self.assertEqual([p["id"] for p in self.posts(q="9002")["posts"]], [9002])

    def test_substring_search_still_works(self):
        # Falls back to a scan because the index only matches token prefixes.
        self.assertEqual([p["id"] for p in self.posts(q="apejak")["posts"]], [9001])

    def test_unmatched_search_is_empty(self):
        data = self.posts(q="definitelynothing")
        self.assertEqual(data["posts"], [])
        self.assertEqual(data["total"], 0)

    def test_total_counts_matches_not_page(self):
        data = self.posts(limit=1)
        self.assertEqual(len(data["posts"]), 1)
        self.assertGreaterEqual(data["total"], 4)

    def test_pagination(self):
        first = self.posts(limit=2, page=1)["posts"]
        second = self.posts(limit=2, page=2)["posts"]
        self.assertEqual(len(first), 2)
        self.assertTrue(set(p["id"] for p in first).isdisjoint(p["id"] for p in second))

    def test_page_beyond_the_end_is_empty(self):
        self.assertEqual(self.posts(page=500)["posts"], [])

    def test_rejects_absurd_limits(self):
        self.assertEqual(self.client.get("/api/posts", params={"limit": 5000}).status_code, 422)
        self.assertEqual(self.client.get("/api/posts", params={"page": 0}).status_code, 422)

    def test_enrichment(self):
        by_id = {p["id"]: p for p in self.posts()["posts"]}
        self.assertEqual(by_id[9001]["image_url"], "/media/9001.png")
        self.assertEqual(by_id[9001]["thumbnail_url"], "/thumbnails/9001.jpg")
        self.assertFalse(by_id[9001]["is_video"])
        self.assertTrue(by_id[9003]["is_gif"])
        self.assertTrue(by_id[9004]["is_video"])

    def test_recent_feed(self):
        data = self.client.get("/api/recent", params={"after_id": 9002}).json()
        self.assertEqual([p["id"] for p in data["posts"]], [9003, 9004])


class TestTagSuggestions(ServerTestCase):
    def test_prefix_suggestions(self):
        tags = self.client.get("/api/tags", params={"prefix": "gap"}).json()["tags"]
        self.assertIn("gapejak", tags)

    def test_variant_prefix_suggestions(self):
        tags = self.client.get("/api/tags", params={"prefix": "variant:c"}).json()["tags"]
        self.assertIn("variant:cobson", tags)

    def test_empty_prefix_is_rejected(self):
        self.assertEqual(self.client.get("/api/tags", params={"prefix": ""}).status_code, 422)


class TestMedia(ServerTestCase):
    def test_serves_image_by_id(self):
        r = self.client.get("/media/9001")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/png")

    def test_serves_image_by_id_and_extension(self):
        r = self.client.get("/media/9001.png")
        self.assertEqual(r.status_code, 200)
        with Image.open(io.BytesIO(r.content)) as img:
            self.assertEqual(img.size, (80, 60))

    def test_media_is_cacheable(self):
        r = self.client.get("/media/9001.png")
        self.assertIn("immutable", r.headers["cache-control"])

    def test_video_route_rejects_images(self):
        self.assertEqual(self.client.get("/videos/9001.png").status_code, 404)

    def test_image_route_rejects_videos(self):
        self.assertEqual(self.client.get("/images/9004.mp4").status_code, 404)

    def test_video_route_serves_videos(self):
        self.assertEqual(self.client.get("/videos/9004.mp4").status_code, 200)

    def test_missing_media_is_404(self):
        self.assertEqual(self.client.get("/media/424242").status_code, 404)

    def test_path_traversal_is_rejected(self):
        secret = library.DATA_DIR / "secret.txt"
        secret.write_text("secret")
        try:
            for path in ("/images/9001.%2e%2e%2fsecret", "/media/9001.%2e%2e", "/images/9001.../.."):
                r = self.client.get(path)
                self.assertNotEqual(r.status_code, 200, path)
                self.assertNotIn(b"secret", r.content, path)
        finally:
            secret.unlink()

    def test_range_requests_are_supported(self):
        r = self.client.get("/media/9004.mp4", headers={"Range": "bytes=0-15"})
        self.assertEqual(r.status_code, 206)
        self.assertEqual(len(r.content), 16)


class TestThumbnails(ServerTestCase):
    def setUp(self):
        for pid, *_ in POSTS:
            library.thumbnail_path(pid).unlink(missing_ok=True)
        library._unthumbnailable.clear()

    def test_generates_on_demand(self):
        r = self.client.get("/thumbnails/9001.jpg")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/jpeg")
        with Image.open(io.BytesIO(r.content)) as img:
            self.assertLessEqual(max(img.size), library.config["thumbnail_size"])

    def test_second_request_reuses_the_file(self):
        self.client.get("/thumbnails/9001.jpg")
        mtime = library.thumbnail_path(9001).stat().st_mtime_ns
        self.client.get("/thumbnails/9001.jpg")
        self.assertEqual(library.thumbnail_path(9001).stat().st_mtime_ns, mtime)

    def test_thumbnail_is_cacheable(self):
        r = self.client.get("/thumbnails/9001.jpg")
        self.assertIn("immutable", r.headers["cache-control"])

    def test_unknown_post_is_404(self):
        self.assertEqual(self.client.get("/thumbnails/424242.jpg").status_code, 404)


class TestScrapeControl(ServerTestCase):
    def test_status_without_a_job(self):
        r = self.client.get("/api/scrape/status")
        self.assertEqual(r.status_code, 200)
        self.assertIn("running", r.json())

    def test_stop_without_a_job(self):
        self.assertEqual(self.client.post("/api/scrape/stop").json(),
                         {"message": "No active scrape to stop"})

    def test_resume_point_comes_from_the_database(self):
        self.assertEqual(server._resume_start_id(), 9004)


class TestStaticPages(ServerTestCase):
    def test_index_is_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Soybrary", r.text)

    def test_static_assets_are_served(self):
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)


if __name__ == "__main__":
    unittest.main()
