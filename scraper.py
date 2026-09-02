import asyncio
import base64
import io
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading

from PIL import Image

# Import playwright
from playwright.async_api import async_playwright

import library
from library import (
    DATA_DIR,
    DB_PATH,
    DEFAULT_CONFIG,
    IMAGES_DIR,
    METADATA_DIR,
    VIDEOS_DIR,
    Database,
    config,
    is_video_extension,
    load_config,
    safe_extension,
    thumbnail_from_bytes,
    thumbnail_from_video,
)

CONFIG_FILE = library.CONFIG_FILE

logger = logging.getLogger(__name__)

BOORU_ORIGIN = "https://soybooru.com"
FALLBACK_LATEST_ID = 245000
STATS_TEMPLATE = {"completed": 0, "skipped": 0, "empty": 0, "failed": 0}

db = Database(DB_PATH)


# Magic Bytes Signatures
def validate_magic_bytes(data: bytes, mime_type: str, ext: str) -> bool:
    mime_type = (mime_type or "").lower()
    ext = (ext or "").lower()

    if mime_type == 'image/png' or ext == 'png':
        return data.startswith(b'\x89PNG\r\n\x1a\n')
    elif mime_type in ['image/jpeg', 'image/jpg'] or ext in ['jpg', 'jpeg']:
        return data.startswith(b'\xff\xd8\xff')
    elif mime_type == 'image/gif' or ext == 'gif':
        return data.startswith(b'GIF87a') or data.startswith(b'GIF89a')
    elif mime_type == 'image/webp' or ext == 'webp':
        return data.startswith(b'RIFF') and len(data) > 12 and data[8:12] == b'WEBP'
    elif mime_type == 'video/webm' or ext == 'webm':
        return data.startswith(b'\x1a\x45\xdf\xa3')
    elif mime_type == 'video/mp4' or ext == 'mp4':
        return len(data) > 8 and data[4:8] == b'ftyp'
    # Fallback/unknown format: let it fail Pillow verification or log it
    return False


def check_ffmpeg_available():
    has_ffmpeg = shutil.which("ffmpeg") is not None
    has_ffprobe = shutil.which("ffprobe") is not None
    return has_ffmpeg, has_ffprobe


def verify_and_sanitize_video(file_data: bytes, mime_type: str, ext: str) -> bytes:
    if not validate_magic_bytes(file_data, mime_type, ext):
        raise ValueError(f"Magic bytes signature mismatch for {mime_type} / .{ext}")

    has_ffmpeg, has_ffprobe = check_ffmpeg_available()

    if not has_ffprobe:
        logger.warning("ffprobe not found. Falling back to magic-bytes-only validation for video.")
        return file_data

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(file_data)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
             "-show_entries", "format=format_name", "-of", "json", tmp_path],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            raise ValueError(f"ffprobe validation failed: {result.stderr.strip()}")

        probe_data = json.loads(result.stdout)
        streams = probe_data.get("streams", [])
        has_video_stream = any(s.get("codec_type") == "video" for s in streams)

        if not has_video_stream:
            raise ValueError("ffprobe: no video stream found in file")

        if config["sanitize_videos"] and has_ffmpeg:
            clean_path = tmp_path + ".clean"
            sanitize_result = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-c", "copy", "-map_metadata", "-1", clean_path],
                capture_output=True, text=True, timeout=60
            )

            if sanitize_result.returncode != 0:
                logger.warning("ffmpeg sanitization failed, returning original: %s",
                               sanitize_result.stderr.strip())
            else:
                with open(clean_path, "rb") as f:
                    file_data = f.read()
                os.remove(clean_path)

        return file_data
    finally:
        os.remove(tmp_path)


def verify_and_sanitize_file(file_data: bytes, mime_type: str, ext: str) -> bytes:
    if mime_type.startswith("image/") and config["validate_images"]:
        return verify_and_sanitize_image(file_data, mime_type, ext)
    elif mime_type.startswith("video/") and config["validate_videos"]:
        return verify_and_sanitize_video(file_data, mime_type, ext)
    return file_data


# Security verification and sanitization
SANITIZABLE_IMAGE_MIMES = ('image/png', 'image/jpeg', 'image/jpg', 'image/webp')
VERIFIABLE_IMAGE_MIMES = SANITIZABLE_IMAGE_MIMES + ('image/gif',)


def _resave_without_metadata(file_data: bytes) -> bytes:
    """Re-encode an image to drop EXIF/ICC payloads, preserving the pixels.

    Animated files are left untouched: a re-save would keep only the first
    frame, and animation is most of what a soyjak library is made of.
    """
    img = Image.open(io.BytesIO(file_data))
    if getattr(img, "n_frames", 1) > 1:
        return file_data

    save_kwargs = {}
    if img.format == "JPEG":
        # Re-encoding at Pillow's default quality would visibly degrade every
        # JPEG in the library; "keep" reuses the original coefficients.
        save_kwargs = {"quality": "keep", "subsampling": "keep"}
    elif img.format == "WEBP":
        save_kwargs = {"lossless": True} if img.mode in ("RGBA", "LA", "P") else {"quality": 95}

    out_buf = io.BytesIO()
    img.save(out_buf, format=img.format, **save_kwargs)
    return out_buf.getvalue()


def verify_and_sanitize_image(file_data: bytes, mime_type: str, ext: str) -> bytes:
    # 1. Magic bytes check
    if not validate_magic_bytes(file_data, mime_type, ext):
        raise ValueError(f"Magic bytes signature mismatch for {mime_type} / .{ext}")

    # 2. Image structure verification (Pillow)
    if mime_type in VERIFIABLE_IMAGE_MIMES:
        try:
            img = Image.open(io.BytesIO(file_data))
            img.verify()

            # 3. Optional sanitization (strip metadata by re-encoding)
            if config["sanitize_images"] and mime_type in SANITIZABLE_IMAGE_MIMES:
                return _resave_without_metadata(file_data)
        except Exception as e:
            raise ValueError(f"Pillow verification failed: {e}")

    return file_data


def resolve_extension(mime_type: str, original_filename: str) -> str:
    """Pick a safe on-disk extension from the reported mime type / filename."""
    ext = mime_type.split("/")[-1] if "/" in (mime_type or "") else "bin"
    if ext == "jpeg":
        ext = "jpg"
    if original_filename and "." in original_filename:
        ext = original_filename.rsplit(".", 1)[-1]
    # Remote-controlled input: never let it escape the media directories.
    return safe_extension(ext) or "bin"


# Scraper Class wrapping Page-level evaluations
class Scraper:
    def __init__(self, page):
        self.page = page

    async def get_json(self, url):
        return await self.page.evaluate("""
            async (url) => {
                try {
                    const response = await fetch(url);
                    if (response.ok) {
                        const json = await response.json();
                        return { data: json, status: response.status };
                    } else {
                        return { data: null, status: response.status };
                    }
                } catch (e) {
                    return { data: null, status: 500, error: e.toString() };
                }
            }
        """, url)

    async def download_file(self, url):
        # Fetch file and convert to base64
        result = await self.page.evaluate("""
            async (url) => {
                try {
                    const response = await fetch(url);
                    if (!response.ok) {
                        return { data: null, status: response.status };
                    }
                    const blob = await response.blob();
                    const base64 = await new Promise((resolve, reject) => {
                        const reader = new FileReader();
                        reader.onloadend = () => resolve(reader.result.split(',')[1]);
                        reader.onerror = reject;
                        reader.readAsDataURL(blob);
                    });
                    return { data: base64, status: 200 };
                } catch (e) {
                    return { data: null, status: 500, error: e.toString() };
                }
            }
        """, url)

        if result["status"] == 200 and result["data"] is not None:
            # Decode base64 back to bytes
            return base64.b64decode(result["data"]), 200
        return None, result["status"]


async def _pregenerate_thumbnail(post_id, file_data, media_path, is_video):
    """Build the gallery thumbnail now, while the bytes are already in hand."""
    if not config.get("pregenerate_thumbnails", True):
        return
    try:
        if is_video:
            await asyncio.to_thread(thumbnail_from_video, post_id, media_path)
        else:
            await asyncio.to_thread(thumbnail_from_bytes, post_id, file_data)
    except Exception as e:
        logger.debug("Thumbnail pregeneration failed for %s: %s", post_id, e)


# Core scrape worker
async def scrape_post(scraper, post_id):
    # Check if already processed
    status = db.get_post_status(post_id)
    if status in ['completed', 'skipped', 'empty']:
        return "skipped"

    meta_url = f"{BOORU_ORIGIN}/api/booru/posts/{post_id}"
    file_url = f"{BOORU_ORIGIN}/api/booru/posts/{post_id}/file"

    try:
        res = await scraper.get_json(meta_url)

        status_code = res.get("status")
        data = res.get("data")

        if status_code == 404 or (data is not None and data.get("id") is None):
            db.save_post(post_id, "empty")
            return "empty"

        # Retry on rate limit / server overload (exponential backoff)
        max_retries = 3
        retries = 0
        while status_code in (429, 503) and retries < max_retries:
            retries += 1
            wait = (2 ** retries) * 5
            logger.info("Post %s: got %s, retrying in %ss (attempt %s/%s)",
                        post_id, status_code, wait, retries, max_retries)
            await asyncio.sleep(wait)
            res = await scraper.get_json(meta_url)
            status_code = res.get("status")
            data = res.get("data")

        if status_code != 200 or data is None:
            db.save_post(post_id, "failed", error_message=f"HTTP Status {status_code}: {res.get('error')}")
            return "failed"

        # Parse tags — separate general tags from variant/subvariant
        tags_list   = data.get("tags", [])
        variants    = [t.get("name") for t in tags_list if t.get("category") == "variant"]
        subvariants = [t.get("name") for t in tags_list if t.get("category") == "subvariant"]
        # Only general tags go into the tags column (exclude variant and subvariant)
        general_tags = [t.get("name") for t in tags_list if t.get("name") and t.get("category") not in ("variant", "subvariant")]

        variant_str    = ",".join(variants)    if variants    else None
        subvariant_str = ",".join(subvariants) if subvariants else None
        tags_str       = " ".join(general_tags) if general_tags else None

        # Safe mime type and original filename
        mime_type = data.get("mimeType", "application/octet-stream")
        orig_filename = data.get("originalFileName", "")
        ext = resolve_extension(mime_type, orig_filename)

        # Download the file
        file_data, file_status = await scraper.download_file(file_url)

        # Retry file download on rate limit / server overload
        retries = 0
        while file_status in (429, 503) and retries < max_retries:
            retries += 1
            wait = (2 ** retries) * 5
            logger.info("Post %s: file download got %s, retrying in %ss (attempt %s/%s)",
                        post_id, file_status, wait, retries, max_retries)
            await asyncio.sleep(wait)
            file_data, file_status = await scraper.download_file(file_url)

        if file_status != 200 or file_data is None:
            db.save_post(post_id, "failed", error_message=f"File download HTTP Status {file_status}")
            return "failed"

        # Verify and sanitize file
        try:
            file_data = verify_and_sanitize_file(file_data, mime_type, ext)
        except Exception as e:
            db.save_post(post_id, "failed", error_message=f"Verification error: {e}")
            logger.warning("Post %s: threat mitigation validation failed: %s", post_id, e)
            return "failed"

        # Write files
        # 1. Image or video
        is_video = is_video_extension(ext, mime_type)
        media_path = (VIDEOS_DIR if is_video else IMAGES_DIR) / f"{post_id}.{ext}"
        with open(media_path, "wb") as f:
            f.write(file_data)

        # 2. Simplified JSON metadata — tags is general tags only, variant/subvariant separate
        simplified_meta = {
            "postNumber": post_id,
            "originalFileName": orig_filename,
            "mimeType": mime_type,
            "fileSize": data.get("fileSize") or len(file_data),
            "width": data.get("width"),
            "height": data.get("height"),
            "uploadedAt": data.get("uploadedAt"),
            "uploader": data.get("uploader", {}).get("userName"),
            "tags": general_tags,
            "variants": variants,
            "subvariants": subvariants
        }
        meta_path = METADATA_DIR / f"{post_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(simplified_meta, f, indent=2)

        # 3. Database
        db.save_post(
            post_id=post_id,
            status="completed",
            variant=variant_str,
            subvariant=subvariant_str,
            tags=tags_str,
            date_uploaded=data.get("uploadedAt"),
            file_url=file_url,
            width=data.get("width"),
            height=data.get("height"),
            file_size=data.get("fileSize") or len(file_data),
            image_hash=data.get("imageHash"),
            mime_type=mime_type,
            extension=ext,
            uploader=data.get("uploader", {}).get("userName"),
            original_filename=orig_filename
        )

        # 4. Thumbnail, so the gallery never has to generate one while browsing
        await _pregenerate_thumbnail(post_id, file_data, media_path, is_video)
        return "completed"

    except Exception as e:
        db.save_post(post_id, "failed", error_message=str(e))
        logger.error("Error scraping post %s: %s", post_id, e)
        return "failed"


def pending_ids(start_id, end_id, limit=None):
    """Ids in the range that still need scraping.

    One range query beats asking the database about every id in turn: a full
    catalog sweep is hundreds of thousands of ids.
    """
    done = db.get_done_ids(start_id, end_id)
    pending = []
    for pid in range(start_id, end_id + 1):
        if pid in done:
            continue
        pending.append(pid)
        if limit and len(pending) >= limit:
            break
    return pending


def resume_start_id():
    """Default scrape start: just after the highest post already recorded."""
    resume = db.get_resume_id()
    if resume:
        return resume

    known = set()
    for directory in (IMAGES_DIR, VIDEOS_DIR):
        if not directory.exists():
            continue
        for f in directory.iterdir():
            if f.is_file() and f.stem.isdigit():
                known.add(int(f.stem))
    return max(known) if known else 1


async def detect_latest_id(page):
    html = await page.content()
    post_ids = [int(m) for m in re.findall(r'/post/view/(\d+)', html)]
    return max(post_ids) if post_ids else None


def scrape_delay(delay_ms):
    """Politeness delay with ±30% jitter so requests don't arrive in lockstep."""
    jitter = random.uniform(-delay_ms * 0.3, delay_ms * 0.3)
    return max(100, delay_ms + jitter) / 1000.0


class ScrapeJob:
    """Manages a scrape session with progress tracking and real-time updates."""

    def __init__(self, start_id=None, end_id=None, limit=None, progress_queue=None, main_loop=None):
        self.start_id = start_id
        self.end_id = end_id
        self.limit = limit
        self.progress_queue = progress_queue or asyncio.Queue()
        self.main_loop = main_loop or asyncio.get_event_loop()
        self.running = False
        self.cancelled = False
        self.stats = dict(STATS_TEMPLATE)
        self.current_id = None
        self.total_queue = 0
        self.message = "Initializing..."

    async def _emit(self, event_type, data=None):
        event = {"type": event_type, "data": data or {}}
        # The scrape runs on its own event loop; hand events back to the server's.
        try:
            asyncio.run_coroutine_threadsafe(self.progress_queue.put(event), self.main_loop)
        except RuntimeError:
            pass

    def _progress_message(self, post_id, result, total_done):
        if self.end_id:
            pct = min(100.0, (post_id / self.end_id) * 100)
            return f"[{total_done}] Scraped {post_id}: {result.upper()} ({pct:.2f}%)"
        return f"[{total_done}] Scraped {post_id}: {result.upper()}"

    async def _worker(self, queue, scraper, delay_ms):
        while True:
            if self.cancelled:
                # Drain remaining items without processing
                while not queue.empty():
                    try:
                        queue.get_nowait()
                        queue.task_done()
                    except asyncio.QueueEmpty:
                        break
                return

            try:
                post_id = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                self.current_id = post_id
                await self._emit("post_start", {"id": post_id})
                res = await scrape_post(scraper, post_id)
                self.stats[res] = self.stats.get(res, 0) + 1
                total_done = sum(self.stats.values())

                if res in ("completed", "empty", "failed"):
                    msg = self._progress_message(post_id, res, total_done)
                    logger.info(msg)
                    level = {"completed": "success", "empty": "warning"}.get(res, "error")
                    await self._emit("console", {"message": msg, "level": level})
                    await self._emit("post_done", {
                        "id": post_id,
                        "status": res,
                        "total_done": total_done,
                        "stats": dict(self.stats)
                    })

                if res != "skipped":
                    await asyncio.sleep(scrape_delay(delay_ms))
            finally:
                queue.task_done()

    async def run(self):
        self.running = True
        self.cancelled = False
        self.stats = dict(STATS_TEMPLATE)

        logger.info("ScrapeJob starting: range %s to %s", self.start_id, self.end_id)
        await self._emit("status", {"message": "Launching browser..."})

        # Playwright drives subprocesses, which needs a loop that supports them
        # (ProactorEventLoop on Windows), so it gets a thread and loop of its own.
        error_holder = [None]
        done_event = threading.Event()

        def _run_playwright_thread():
            if sys.platform == "win32" and hasattr(asyncio, "ProactorEventLoop"):
                loop = asyncio.ProactorEventLoop()
            else:
                loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._run_playwright())
            except Exception as e:
                error_holder[0] = e
                logger.exception("Playwright thread exception: %s", e)
            finally:
                loop.close()
                done_event.set()

        thread = threading.Thread(target=_run_playwright_thread, daemon=True)
        thread.start()
        await asyncio.to_thread(done_event.wait)

        if error_holder[0]:
            await self._emit("error", {"message": str(error_holder[0])})

        self.running = False
        await self._emit("complete", {"stats": dict(self.stats)})
        logger.info("Scrape session ended. Stats: %s", self.stats)

    async def _run_playwright(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            try:
                await self._emit("status", {"message": "Activating Turnstile & PoW..."})
                await page.goto(f"{BOORU_ORIGIN}/booru", timeout=60000)
                await asyncio.sleep(5)

                scraper = Scraper(page)

                if self.start_id is None:
                    self.start_id = resume_start_id()
                if self.end_id is None:
                    self.end_id = await detect_latest_id(page) or FALLBACK_LATEST_ID

                await self._emit("status", {
                    "message": f"Scraping range {self.start_id} to {self.end_id}",
                    "range_start": self.start_id,
                    "range_end": self.end_id
                })

                pending = await asyncio.to_thread(
                    pending_ids, self.start_id, self.end_id, self.limit)
                if self.cancelled:
                    return

                queue = asyncio.Queue()
                for pid in pending:
                    queue.put_nowait(pid)

                self.total_queue = len(pending)
                await self._emit("status", {"message": f"Queue populated: {self.total_queue} posts"})

                if not pending:
                    await self._emit("status", {"message": "No new posts to scrape."})
                    return

                tasks = [
                    asyncio.create_task(self._worker(queue, scraper, config["delay_ms"]))
                    for _ in range(config["concurrency"])
                ]

                try:
                    await queue.join()
                finally:
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                await browser.close()

    def cancel(self):
        self.cancelled = True
        self.message = "Cancelling..."

    def get_status(self):
        return {
            "running": self.running,
            "cancelled": self.cancelled,
            "current_id": self.current_id,
            "total_queue": self.total_queue,
            "stats": dict(self.stats),
            "message": self.message,
            "range": {"start": self.start_id, "end": self.end_id}
        }


# CLI entry point
async def worker(queue, scraper, stats, delay_ms, end_id=None):
    while True:
        post_id = await queue.get()
        try:
            res = await scrape_post(scraper, post_id)
            stats[res] = stats.get(res, 0) + 1
            total_done = sum(stats.values())

            if res in ("completed", "empty", "failed"):
                if end_id:
                    pct = min(100.0, (post_id / end_id) * 100)
                    print(f"[{total_done}] Scraped {post_id}: {res.upper()} ({pct:.2f}%)")
                else:
                    print(f"[{total_done}] Scraped {post_id}: {res.upper()}")

            if res != "skipped":
                await asyncio.sleep(scrape_delay(delay_ms))
        finally:
            queue.task_done()


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Soybooru Downloader Scraper (Headless)")
    parser.add_argument("--start", type=int, default=None, help="Post ID to start scraping from (default: resumes after the highest post already stored)")
    parser.add_argument("--end", type=int, help="Post ID to stop scraping at (default: latest)")
    parser.add_argument("--limit", type=int, help="Limit total posts to scrape in this run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    start_id = args.start
    if start_id is None:
        start_id = resume_start_id()
        print(f"Resuming from post ID: {start_id}")

    end_id = args.end

    print("Launching headless Playwright context...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Use standard, unmodified context to prevent fingerprint detection
        context = await browser.new_context()
        page = await context.new_page()

        print(f"Navigating to {BOORU_ORIGIN}/booru (Turnstile & PoW activation)...")
        await page.goto(f"{BOORU_ORIGIN}/booru", timeout=60000)
        # Wait for page scripts to load and verify
        await asyncio.sleep(5)

        scraper = Scraper(page)

        if end_id is None:
            end_id = await detect_latest_id(page)
            if end_id:
                print(f"Latest post ID auto-detected: {end_id}")
            else:
                end_id = FALLBACK_LATEST_ID
                print(f"Failed to auto-detect latest post, defaulting stop to {end_id}")

        print(f"Scrape range: {start_id} to {end_id}")

        pending = pending_ids(start_id, end_id, args.limit)
        print(f"Queue size populated: {len(pending)} posts to process")
        if not pending:
            print("No new posts to scrape.")
            await browser.close()
            db.close()
            return

        queue = asyncio.Queue()
        for pid in pending:
            queue.put_nowait(pid)

        stats = dict(STATS_TEMPLATE)
        concurrency = config["concurrency"]
        delay_ms = config["delay_ms"]

        print(f"Starting scraper with {concurrency} workers (delay: {delay_ms}ms)...")
        tasks = [
            asyncio.create_task(worker(queue, scraper, stats, delay_ms, end_id))
            for _ in range(concurrency)
        ]

        try:
            await queue.join()
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nShutdown requested by user. Terminating workers...")
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await browser.close()
            db.close()

    print("\nScraper session ended.")
    print(f"Stats - Completed: {stats['completed']}, Empty/404: {stats['empty']}, "
          f"Failed: {stats['failed']}, Skipped: {stats['skipped']}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
