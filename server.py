from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from contextlib import asynccontextmanager
from pathlib import Path
import sqlite3
import asyncio
import json
import bisect
import logging
import subprocess
import tempfile
import os
from typing import Optional

from scraper import ScrapeJob
from PIL import Image as PILImage

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "soybooru.db"
STATIC_DIR = BASE_DIR / "static"
IMAGES_DIR = BASE_DIR / "data" / "images"
VIDEOS_DIR = BASE_DIR / "data" / "videos"
THUMBNAILS_DIR = BASE_DIR / "data" / "thumbnails"

# Setup logging (console only)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Ensure directories exist
STATIC_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR.mkdir(exist_ok=True)

# Global scrape job manager
scrape_job: Optional[ScrapeJob] = None
scrape_event_queues: list[asyncio.Queue] = []

# ── In-memory tag index ───────────────────────────────────────────────────────
# Two structures kept in sync:
#   _tag_sorted  – alphabetically sorted list of all unique tag strings (for bisect)
#   _tag_freq    – dict mapping tag -> post count (for ranking matches by popularity)
_tag_sorted: list[str] = []
_tag_freq: dict[str, int] = {}
_tag_index_dirty: bool = True   # set True to trigger a rebuild on next query


def _build_tag_index():
    """Scan the DB, count tag frequencies, build sorted list + freq map.
    Includes general tags, variant:, and subvariant: prefixed entries."""
    global _tag_sorted, _tag_freq, _tag_index_dirty
    logger.info("Building tag index...")
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    cursor = conn.cursor()
    counts: dict[str, int] = {}

    # General tags column (space-separated)
    logger.info("Scanning general tags...")
    cursor.execute("SELECT tags FROM posts WHERE status = 'completed' AND tags IS NOT NULL")
    for (row,) in cursor.fetchall():
        for tok in row.split():
            t = tok.lower()
            counts[t] = counts.get(t, 0) + 1

    # Variant column (comma-separated) — index with "variant:" prefix AND without prefix
    logger.info("Scanning variant tags...")
    cursor.execute("SELECT variant FROM posts WHERE status = 'completed' AND variant IS NOT NULL")
    for (row,) in cursor.fetchall():
        for tok in row.split(","):
            t = tok.strip().lower()
            if t:
                key = f"variant:{t}"
                counts[key] = counts.get(key, 0) + 1
                counts[t] = counts.get(t, 0) + 1

    # Subvariant column (comma-separated) — index with "subvariant:" prefix AND without prefix
    logger.info("Scanning subvariant tags...")
    cursor.execute("SELECT subvariant FROM posts WHERE status = 'completed' AND subvariant IS NOT NULL")
    for (row,) in cursor.fetchall():
        for tok in row.split(","):
            t = tok.strip().lower()
            if t:
                key = f"subvariant:{t}"
                counts[key] = counts.get(key, 0) + 1
                counts[t] = counts.get(t, 0) + 1

    conn.close()

    _tag_freq = counts
    _tag_sorted = sorted(counts.keys())   # alphabetical — used only for bisect range
    _tag_index_dirty = False
    logger.info(f"Tag index built with {len(_tag_sorted)} unique tags")


def _prefix_search(prefix: str, limit: int = 20) -> list[str]:
    """Return up to `limit` tags starting with `prefix`, ranked by post frequency."""
    if _tag_index_dirty or not _tag_sorted:
        _build_tag_index()
    p = prefix.lower()
    lo = bisect.bisect_left(_tag_sorted, p)
    # Collect ALL tags in the prefix range, then sort by frequency descending
    matches = []
    for i in range(lo, len(_tag_sorted)):
        tag = _tag_sorted[i]
        if not tag.startswith(p):
            break
        matches.append(tag)
    matches.sort(key=lambda t: -_tag_freq.get(t, 0))
    return matches[:limit]


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


VIDEO_EXTENSIONS = {'mp4', 'webm', 'mov', 'avi', 'mkv', 'ogv', 'ogg', 'flv', 'wmv'}


def get_or_create_thumbnail(post_id: int, extension: str, mime_type: str = None):
    """Generate a 300px max thumbnail if it doesn't exist."""
    thumb_path = THUMBNAILS_DIR / f"{post_id}.jpg"
    if thumb_path.exists():
        return thumb_path

    ext_lower = (extension or "").lower()
    is_video = ext_lower in VIDEO_EXTENSIONS or (mime_type and mime_type.startswith("video/"))

    if is_video:
        return _thumbnail_from_video(post_id, extension, thumb_path)

    img_path = IMAGES_DIR / f"{post_id}.{extension}"
    if not img_path.exists():
        return None

    try:
        with PILImage.open(img_path) as img:
            img.thumbnail((300, 300), PILImage.LANCZOS)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.save(thumb_path, 'JPEG', quality=85)
        return thumb_path
    except Exception:
        # Pillow can't open this format (swf, cbz, psd, etc.) — no thumbnail possible
        return None


def _thumbnail_from_video(post_id: int, extension: str, thumb_path: Path):
    """Extract the first frame of a video as a JPEG thumbnail using ffmpeg."""
    vid_path = VIDEOS_DIR / f"{post_id}.{extension}"
    if not vid_path.exists():
        return None

    # Write to a temp file first, then move atomically so a failed extraction
    # doesn't leave a corrupt thumbnail on disk.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=THUMBNAILS_DIR)
    os.close(tmp_fd)
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", "0",
                "-i", str(vid_path),
                "-vframes", "1",
                "-update", "1",
                "-vf", "scale=300:300:force_original_aspect_ratio=decrease",
                "-q:v", "5",
                tmp_path,
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and Path(tmp_path).stat().st_size > 0:
            os.chmod(tmp_path, 0o644)
            Path(tmp_path).rename(thumb_path)
            return thumb_path
        else:
            logger.warning("ffmpeg failed for post %d: %s", post_id, result.stderr[-200:])
            return None
    except Exception as e:
        logger.warning("Video thumbnail error for post %d: %s", post_id, e)
        return None
    finally:
        # Clean up temp file if it wasn't moved
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def enrich_post(post: dict) -> dict:
    post["thumbnail_url"] = f"/thumbnails/{post['id']}.jpg"
    mime_type = post.get("mime_type", "")
    extension = (post.get("extension") or "").lower()
    # Route by extension first (handles mime_type mismatches in scraped data),
    # then fall back to mime_type for edge cases.
    is_video = extension in VIDEO_EXTENSIONS or mime_type.startswith("video/")
    post["is_video"] = is_video
    if is_video:
        post["image_url"] = f"/videos/{post['id']}.{post['extension']}"
    else:
        post["image_url"] = f"/images/{post['id']}.{post['extension']}"
    return post


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - pre-build tag index on startup."""
    import concurrent.futures
    logger.info("Application startup: building tag index...")
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, _build_tag_index)
    logger.info("Tag index ready - application startup complete")
    yield
    logger.info("Application shutdown")


# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/images/{post_id}.{extension}")
def get_image(post_id: int, extension: str):
    img_path = IMAGES_DIR / f"{post_id}.{extension}"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img_path)


@app.get("/videos/{post_id}.{extension}")
def get_video(post_id: int, extension: str):
    video_path = VIDEOS_DIR / f"{post_id}.{extension}"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path)


@app.get("/thumbnails/{post_id}.jpg")
def get_thumbnail(post_id: int):
    thumb_path = THUMBNAILS_DIR / f"{post_id}.jpg"
    if not thumb_path.exists():
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT extension, mime_type FROM posts WHERE id = ?", (post_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail="Post not found")
        ext = row["extension"]
        mime_type = row["mime_type"]
        result = get_or_create_thumbnail(post_id, ext, mime_type)
        if result is None:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        thumb_path = result
    return FileResponse(thumb_path)


@app.get("/api/posts")
def get_posts(
    q: str = Query(default=""),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
):
    conn = get_db()
    cursor = conn.cursor()

    fields = "id, width, height, extension, mime_type, tags, variant, subvariant, uploader, date_uploaded"
    base_sql = f"SELECT {fields} FROM posts WHERE status = 'completed'"
    count_sql = "SELECT COUNT(*) FROM posts WHERE status = 'completed'"
    params = []

    if q and q.strip():
        terms = q.strip().split()
        conditions = []
        for term in terms:
            term_like = f"%{term}%"
            # Check for prefix syntax like "variant:wide" or "subvariant:hdr"
            if term.startswith("variant:") and len(term) > 8:
                sub_term = term[8:]
                sub_like = f"%{sub_term}%"
                conditions.append("(variant LIKE ?)")
                params.extend([sub_like])
            elif term.startswith("subvariant:") and len(term) > 11:
                sub_term = term[11:]
                sub_like = f"%{sub_term}%"
                conditions.append("(subvariant LIKE ?)")
                params.extend([sub_like])
            elif term.isdigit():
                conditions.append(
                    "(id = ? OR tags LIKE ? OR variant LIKE ? OR subvariant LIKE ? OR uploader LIKE ?)"
                )
                params.extend([int(term), term_like, term_like, term_like, term_like])
            else:
                conditions.append(
                    "(tags LIKE ? OR variant LIKE ? OR subvariant LIKE ? OR uploader LIKE ?)"
                )
                params.extend([term_like, term_like, term_like, term_like])
        where_clause = " AND " + " AND ".join(conditions)
        base_sql += where_clause
        count_sql += where_clause

    base_sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    offset = (page - 1) * limit
    query_params = params + [limit, offset]

    cursor.execute(base_sql, query_params)
    rows = cursor.fetchall()
    posts = [enrich_post(dict(row)) for row in rows]

    cursor.execute(count_sql, params)
    total = cursor.fetchone()[0]

    conn.close()
    return {"posts": posts, "total": total, "page": page, "limit": limit}


@app.get("/api/recent")
def get_recent(after_id: int = Query(default=0, ge=0)):
    conn = get_db()
    cursor = conn.cursor()
    fields = "id, width, height, extension, mime_type, tags, variant, subvariant, uploader, date_uploaded"
    cursor.execute(
        f"SELECT {fields} FROM posts WHERE status = 'completed' AND id > ? ORDER BY id ASC LIMIT 20",
        (after_id,),
    )
    rows = cursor.fetchall()
    posts = [enrich_post(dict(row)) for row in rows]
    conn.close()
    return {"posts": posts}


@app.get("/api/tags")
def get_tags(prefix: str = Query(default="", min_length=1)):
    """Return up to 20 tag suggestions that start with the given prefix."""
    matches = _prefix_search(prefix, limit=20)
    return {"tags": matches}


# Scrape management endpoints
@app.post("/api/scrape/start")
async def start_scrape(
    start_id: Optional[int] = Query(default=None),
    end_id: Optional[int] = Query(default=None),
    limit: Optional[int] = Query(default=None),
):
    global scrape_job
    logger.info("Scrape start request received")

    if scrape_job and scrape_job.running:
        return {"error": "Scrape already running", "status": scrape_job.get_status()}

    # Auto-detect start_id if not provided
    if start_id is None:
        from scraper import METADATA_DIR, IMAGES_DIR, VIDEOS_DIR
        meta_ids = set()
        for f in METADATA_DIR.iterdir():
            if f.is_file() and f.suffix == '.json' and f.stem.isdigit():
                meta_ids.add(int(f.stem))
        image_ids = set()
        for f in IMAGES_DIR.iterdir():
            if f.is_file() and f.stem.isdigit():
                image_ids.add(int(f.stem))
        video_ids = set()
        for f in VIDEOS_DIR.iterdir():
            if f.is_file() and f.stem.isdigit():
                video_ids.add(int(f.stem))
        matching_ids = meta_ids.intersection(image_ids.union(video_ids))
        start_id = max(matching_ids) if matching_ids else 1

    logger.info(f"Scrape range: {start_id} to {end_id}, limit={limit}")

    try:
        progress_queue = asyncio.Queue()
        main_loop = asyncio.get_event_loop()
        scrape_job = ScrapeJob(
            start_id=start_id,
            end_id=end_id,
            limit=limit,
            progress_queue=progress_queue,
            main_loop=main_loop,
        )

        async def forward_events():
            global _tag_index_dirty
            try:
                while True:
                    event = await progress_queue.get()
                    for q in scrape_event_queues:
                        await q.put(event)
                    if isinstance(event, dict) and event.get("type") == "complete":
                        _tag_index_dirty = True
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(scrape_job.run())
        asyncio.create_task(forward_events())

        # Wait for the job to actually start (or fail)
        for i in range(30):
            await asyncio.sleep(0.1)
            if scrape_job.running:
                return {"message": "Scrape started", "status": scrape_job.get_status()}
            if scrape_job.cancelled:
                return {"error": "Scrape was cancelled during startup"}
            if task.done():
                try:
                    task.result()
                except Exception as e:
                    logger.error(f"Scrape task failed immediately: {e}")
                    return {"error": f"Scrape failed to start: {str(e)}"}
                return {"error": "Scrape job finished unexpectedly."}

        logger.error("Scrape job failed to start within 3s")
        return {"error": "Scrape job failed to start."}
    except Exception as e:
        logger.exception("Exception in start_scrape endpoint:")
        return {"error": f"Failed to start scrape: {str(e)}"}


@app.post("/api/scrape/stop")
async def stop_scrape():
    global scrape_job
    if scrape_job and scrape_job.running:
        scrape_job.cancel()
        return {"message": "Scrape cancellation requested"}
    return {"message": "No active scrape to stop"}


@app.get("/api/scrape/status")
async def get_scrape_status():
    global scrape_job
    if scrape_job:
        return scrape_job.get_status()
    return {"running": False, "message": "No scrape job"}


@app.get("/api/events")
async def events():
    """Server-Sent Events stream for real-time scrape progress."""
    queue = asyncio.Queue()
    scrape_event_queues.append(queue)

    async def event_generator():
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"

            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in scrape_event_queues:
                scrape_event_queues.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
