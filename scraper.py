import os
import io
import re
import json
import base64
import asyncio
import random
import sqlite3
import datetime
import threading
import traceback
from pathlib import Path
from PIL import Image
import subprocess
import tempfile
import shutil

# Import playwright
from playwright.async_api import async_playwright

CONFIG_FILE = "config.json"

# Default configuration
DEFAULT_CONFIG = {
    "concurrency": 3,
    "delay_ms": 2000,
    "data_dir": "./data",
    "validate_images": True,
    "sanitize_images": False,
    "validate_videos": True,
    "sanitize_videos": False
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                # Merge with default keys
                for k, v in DEFAULT_CONFIG.items():
                    if k not in config:
                        config[k] = v
                return config
        except Exception as e:
            print(f"Error loading config.json, using defaults. Error: {e}")
    return DEFAULT_CONFIG

config = load_config()
DATA_DIR = Path(config["data_dir"])
IMAGES_DIR = DATA_DIR / "images"
VIDEOS_DIR = DATA_DIR / "videos"
METADATA_DIR = DATA_DIR / "metadata"
DB_PATH = DATA_DIR / "soybooru.db"

# Create directories
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Database Manager
class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self._local = threading.local()

    def _get_conn(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, timeout=10.0)
        return self._local.conn

    def setup(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY,
                status TEXT,
                variant TEXT,
                subvariant TEXT,
                tags TEXT,
                date_uploaded TEXT,
                file_url TEXT,
                width INTEGER,
                height INTEGER,
                file_size INTEGER,
                image_hash TEXT,
                mime_type TEXT,
                extension TEXT,
                uploader TEXT,
                original_filename TEXT,
                last_scraped TEXT,
                error_message TEXT
            )
        """)
        conn.commit()

    def get_post_status(self, post_id):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM posts WHERE id = ?", (post_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def save_post(self, post_id, status, variant=None, subvariant=None, tags=None,
                  date_uploaded=None, file_url=None, width=None, height=None,
                  file_size=None, image_hash=None, mime_type=None, extension=None,
                  uploader=None, original_filename=None, error_message=None):
        conn = self._get_conn()
        cursor = conn.cursor()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cursor.execute("""
            INSERT INTO posts (
                id, status, variant, subvariant, tags, date_uploaded, file_url,
                width, height, file_size, image_hash, mime_type, extension,
                uploader, original_filename, last_scraped, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                variant = excluded.variant,
                subvariant = excluded.subvariant,
                tags = excluded.tags,
                date_uploaded = excluded.date_uploaded,
                file_url = excluded.file_url,
                width = excluded.width,
                height = excluded.height,
                file_size = excluded.file_size,
                image_hash = excluded.image_hash,
                mime_type = excluded.mime_type,
                extension = excluded.extension,
                uploader = excluded.uploader,
                original_filename = excluded.original_filename,
                last_scraped = excluded.last_scraped,
                error_message = excluded.error_message
        """, (
            post_id, status, variant, subvariant, tags, date_uploaded, file_url,
            width, height, file_size, image_hash, mime_type, extension,
            uploader, original_filename, now, error_message
        ))
        conn.commit()

    def close(self):
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()

db = Database(DB_PATH)
db.setup()

# Magic Bytes Signatures
def validate_magic_bytes(data: bytes, mime_type: str, ext: str) -> bool:
    mime_type = mime_type.lower()
    ext = ext.lower()
    
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
        print("WARNING: ffprobe not found. Falling back to magic-bytes-only validation for video.")
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
                print(f"WARNING: ffmpeg sanitization failed, returning original: {sanitize_result.stderr.strip()}")
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
def verify_and_sanitize_image(file_data: bytes, mime_type: str, ext: str) -> bytes:
    # 1. Magic bytes check
    if not validate_magic_bytes(file_data, mime_type, ext):
        raise ValueError(f"Magic bytes signature mismatch for {mime_type} / .{ext}")
        
    # 2. Image structure verification (Pillow)
    if mime_type in ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif']:
        try:
            img = Image.open(io.BytesIO(file_data))
            img.verify()
            
            # 3. Optional Sanitization (Strip metadata by re-encoding)
            if config["sanitize_images"] and mime_type in ['image/png', 'image/jpeg', 'image/jpg', 'image/webp']:
                # Re-open because verify() closes the stream and prohibits saving
                img = Image.open(io.BytesIO(file_data))
                out_buf = io.BytesIO()
                # Saving without original EXIF/metadata
                img.save(out_buf, format=img.format)
                return out_buf.getvalue()
        except Exception as e:
            raise ValueError(f"Pillow verification failed: {e}")
            
    return file_data

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

# Core scrape worker
async def scrape_post(scraper, post_id):
    # Check if already processed
    status = db.get_post_status(post_id)
    if status in ['completed', 'skipped', 'empty']:
        return "skipped"

    meta_url = f"https://soybooru.com/api/booru/posts/{post_id}"
    file_url = f"https://soybooru.com/api/booru/posts/{post_id}/file"
    
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
            print(f"Post {post_id}: Got {status_code}, retrying in {wait}s (attempt {retries}/{max_retries})")
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
        
        # Deduce extension
        ext = mime_type.split("/")[-1] if "/" in mime_type else "bin"
        if ext == "jpeg":
            ext = "jpg"
        if orig_filename and "." in orig_filename:
            ext = orig_filename.split(".")[-1]
            
        # Download the file
        file_data, file_status = await scraper.download_file(file_url)
        
        # Retry file download on rate limit / server overload
        retries = 0
        while file_status in (429, 503) and retries < max_retries:
            retries += 1
            wait = (2 ** retries) * 5
            print(f"Post {post_id}: File download got {file_status}, retrying in {wait}s (attempt {retries}/{max_retries})")
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
            print(f"Post {post_id}: Threat mitigation validation failed! Error: {e}")
            return "failed"
                
        # Write files
        # 1. Image or Video
        if mime_type.startswith("video/"):
            media_path = VIDEOS_DIR / f"{post_id}.{ext}"
        else:
            media_path = IMAGES_DIR / f"{post_id}.{ext}"
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
        return "completed"
        
    except Exception as e:
        db.save_post(post_id, "failed", error_message=str(e))
        print(f"Error scraping post {post_id}: {e}")
        return "failed"


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
        self.stats = {"completed": 0, "skipped": 0, "empty": 0, "failed": 0}
        self.current_id = None
        self.total_queue = 0
        self.message = "Initializing..."
        self._task = None
        
    async def _emit(self, event_type, data=None):
        event = {"type": event_type, "data": data or {}}
        # Use run_coroutine_threadsafe to put events into the main loop's queue
        asyncio.run_coroutine_threadsafe(
            self.progress_queue.put(event),
            self.main_loop
        )
        
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
                self.stats[res] += 1
                total_done = sum(self.stats.values())
                
                if res in ["completed", "empty", "failed"]:
                    pct = (post_id / self.end_id) * 100
                    msg = f"[{total_done}] Scraped {post_id}: {res.upper()} ({pct:.2f}%)"
                    print(msg)
                    await self._emit("console", {"message": msg, "level": "success" if res == "completed" else ("warning" if res == "empty" else "error")})
                    await self._emit("post_done", {
                        "id": post_id, 
                        "status": res,
                        "total_done": total_done,
                        "stats": dict(self.stats)
                    })
                
                if res != "skipped":
                    jitter = random.uniform(-delay_ms * 0.3, delay_ms * 0.3)
                    actual_delay = max(100, delay_ms + jitter) / 1000.0
                    await asyncio.sleep(actual_delay)
            except asyncio.CancelledError:
                queue.task_done()
                return
            finally:
                queue.task_done()
                
    async def run(self):
        self.running = True
        self.cancelled = False
        self.stats = {"completed": 0, "skipped": 0, "empty": 0, "failed": 0}
        
        print(f"ScrapeJob.run() starting: range {self.start_id} to {self.end_id}")
        await self._emit("status", {"message": "Launching browser..."})
        
        # Run Playwright in a separate thread with ProactorEventLoop
        # because the main loop (SelectorEventLoop on Windows) doesn't support subprocesses
        import threading
        import concurrent.futures
        
        error_holder = [None]
        done_event = threading.Event()
        
        def _run_in_proactor():
            loop = asyncio.ProactorEventLoop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._run_playwright())
            except Exception as e:
                error_holder[0] = e
                logger.error(f"Playwright thread exception: {e}")
                logger.exception("Full traceback:")
            finally:
                loop.close()
                done_event.set()
        
        thread = threading.Thread(target=_run_in_proactor, daemon=True)
        thread.start()
        
        # Wait for completion in the main async context
        while not done_event.is_set():
            await asyncio.sleep(0.1)
            if self.cancelled:
                break
        
        if error_holder[0]:
            await self._emit("error", {"message": str(error_holder[0])})
        
        self.running = False
        await self._emit("complete", {"stats": dict(self.stats)})
        print(f"\nScrape session ended. Stats: {self.stats}")

    async def _run_playwright(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            await self._emit("status", {"message": "Activating Turnstile & PoW..."})
            await page.goto("https://soybooru.com/booru", timeout=60000)
            await asyncio.sleep(5)
            
            scraper = Scraper(page)
            
            if self.end_id is None:
                html = await page.content()
                post_ids = [int(m) for m in re.findall(r'/post/view/(\d+)', html)]
                if post_ids:
                    self.end_id = max(post_ids)
                else:
                    self.end_id = 245000
                    
            await self._emit("status", {
                "message": f"Scraping range {self.start_id} to {self.end_id}",
                "range_start": self.start_id,
                "range_end": self.end_id
            })
            
            queue = asyncio.Queue()
            count = 0
            for pid in range(self.start_id, self.end_id + 1):
                if self.cancelled:
                    break
                status = db.get_post_status(pid)
                if status in ['completed', 'skipped', 'empty']:
                    continue
                await queue.put(pid)
                count += 1
                if self.limit and count >= self.limit:
                    break
                    
            self.total_queue = count
            await self._emit("status", {"message": f"Queue populated: {count} posts"})
            
            if count == 0:
                await self._emit("status", {"message": "No new posts to scrape."})
                await browser.close()
                return
                
            concurrency = config["concurrency"]
            delay_ms = config["delay_ms"]
            
            tasks = []
            for _ in range(concurrency):
                t = asyncio.create_task(self._worker(queue, scraper, delay_ms))
                tasks.append(t)
                
            try:
                await queue.join()
            except asyncio.CancelledError:
                for t in tasks:
                    t.cancel()
                    
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
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


# Legacy CLI entry point (unchanged behavior)
async def worker(queue, scraper, stats, delay_ms, end_id):
    while True:
        post_id = await queue.get()
        try:
            res = await scrape_post(scraper, post_id)
            stats[res] += 1
            total_done = stats["completed"] + stats["skipped"] + stats["empty"] + stats["failed"]
            
            # Progress print
            if res in ["completed", "empty", "failed"]:
                pct = (post_id / end_id) * 100
                print(f"[{total_done}] Scraped {post_id}: {res.upper()} ({pct:.2f}%)")
                
            # Rate limit delay with random jitter (±30%)
            if res != "skipped":
                jitter = random.uniform(-delay_ms * 0.3, delay_ms * 0.3)
                actual_delay = max(100, delay_ms + jitter) / 1000.0
                await asyncio.sleep(actual_delay)
        finally:
            queue.task_done()

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Soybooru Downloader Scraper (Headless)")
    parser.add_argument("--start", type=int, default=None, help="Post ID to start scraping from (default: 1, or auto-resumes from highest matched file if previous runs are found)")
    parser.add_argument("--end", type=int, help="Post ID to stop scraping at (default: latest)")
    parser.add_argument("--limit", type=int, help="Limit total posts to scrape in this run")
    args = parser.parse_args()

    # Get range
    start_id = args.start
    if start_id is None:
        meta_ids = set()
        for f in METADATA_DIR.iterdir():
            if f.is_file() and f.suffix == '.json':
                if f.stem.isdigit():
                    meta_ids.add(int(f.stem))

        image_ids = set()
        for f in IMAGES_DIR.iterdir():
            if f.is_file():
                if f.stem.isdigit():
                    image_ids.add(int(f.stem))

        video_ids = set()
        for f in VIDEOS_DIR.iterdir():
            if f.is_file():
                if f.stem.isdigit():
                    video_ids.add(int(f.stem))

        matching_ids = meta_ids.intersection(image_ids.union(video_ids))
        if matching_ids:
            start_id = max(matching_ids)
            print(f"Auto-detect: Previous run found. Starting default scrape from highest matched post ID: {start_id}")
        else:
            start_id = 1
            print(f"Auto-detect: No previous runs detected. Starting scrape from post ID: {start_id}")

    end_id = args.end

    print("Launching headless Playwright context...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Use standard, unmodified context to prevent fingerprint detection
        context = await browser.new_context()
        page = await context.new_page()
        
        print("Navigating to https://soybooru.com/booru (Turnstile & PoW activation)...")
        await page.goto("https://soybooru.com/booru", timeout=60000)
        # Wait for page scripts to load and verify
        await asyncio.sleep(5)
        
        # Instantiate Scraper
        scraper = Scraper(page)
        
        if end_id is None:
            # Detect latest post ID
            html = await page.content()
            post_ids = [int(m) for m in re.findall(r'/post/view/(\d+)', html)]
            if post_ids:
                end_id = max(post_ids)
                print(f"Latest post ID auto-detected: {end_id}")
            else:
                print("Failed to auto-detect latest post, defaulting stop to 245000")
                end_id = 245000

        print(f"Scrape range: {start_id} to {end_id}")
        
        # Populate queue
        queue = asyncio.Queue()
        count = 0
        for pid in range(start_id, end_id + 1):
            status = db.get_post_status(pid)
            if status in ['completed', 'skipped', 'empty']:
                continue
                
            await queue.put(pid)
            count += 1
            if args.limit and count >= args.limit:
                break
                
        print(f"Queue size populated: {count} posts to process")
        if count == 0:
            print("No new posts to scrape.")
            await browser.close()
            db.close()
            return

        stats = {"completed": 0, "skipped": 0, "empty": 0, "failed": 0}
        
        # Concurrency workers
        concurrency = config["concurrency"]
        delay_ms = config["delay_ms"]
        
        print(f"Starting scraper with {concurrency} workers (delay: {delay_ms}ms)...")
        tasks = []
        for _ in range(concurrency):
            t = asyncio.create_task(worker(queue, scraper, stats, delay_ms, end_id))
            tasks.append(t)
            
        # Wait for queue to be empty
        try:
            await queue.join()
        except KeyboardInterrupt:
            print("\nShutdown requested by user. Terminating workers...")
        finally:
            # Cancel workers
            for t in tasks:
                t.cancel()
            # Wait for cancellations
            await asyncio.gather(*tasks, return_exceptions=True)
            await browser.close()
            db.close()
            
    print("\nScraper session ended.")
    print(f"Stats - Completed: {stats['completed']}, Empty/404: {stats['empty']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
