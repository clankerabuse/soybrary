import asyncio
import json
import logging
import mimetypes
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import library
import search
from library import (
    IMAGES_DIR,
    VIDEOS_DIR,
    config,
    ensure_thumbnail,
    find_media,
    is_video_extension,
    read_connection,
    safe_extension,
    thumbnail_path,
)
from scraper import ScrapeJob

BASE_DIR = library.BASE_DIR
STATIC_DIR = BASE_DIR / "static"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

STATIC_DIR.mkdir(exist_ok=True)
library.ensure_dirs()

POST_FIELDS = ", ".join(search.POST_FIELDS)

# Media files are named after an immutable post id, so browsers can keep them
# forever instead of revalidating every thumbnail on each visit.
IMMUTABLE_CACHE = {"Cache-Control": "public, max-age=31536000, immutable"}

# Global scrape job manager
scrape_job: Optional[ScrapeJob] = None
scrape_event_queues: list[asyncio.Queue] = []
# Held module-side: the event loop only keeps weak references to tasks, and a
# scrape that gets garbage collected mid-run is a very confusing bug.
_scrape_task: Optional[asyncio.Task] = None
_event_forwarder: Optional[asyncio.Task] = None

SSE_QUEUE_LIMIT = 500


def _media_response(path: Path) -> FileResponse:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, headers=IMMUTABLE_CACHE)


def enrich_post(post: dict) -> dict:
    mime_type = post.get("mime_type") or ""
    extension = safe_extension(post.get("extension"))
    is_video = is_video_extension(extension, mime_type)
    post["thumbnail_url"] = f"/thumbnails/{post['id']}.jpg"
    post["is_video"] = is_video
    post["is_gif"] = extension == "gif" or mime_type == "image/gif"
    # Include the extension so serving the file needs no database lookup.
    post["image_url"] = f"/media/{post['id']}.{extension}" if extension else f"/media/{post['id']}"
    return post


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the tag index in the background; the catalog is usable immediately."""
    logger.info("Application startup (data dir: %s)", library.DATA_DIR)
    loop = asyncio.get_running_loop()
    index_task = loop.run_in_executor(None, search.tag_index.build)
    try:
        yield
    finally:
        index_task.cancel()
        if _event_forwarder is not None:
            _event_forwarder.cancel()
        logger.info("Application shutdown")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/media/{post_id}.{extension}")
def get_media_with_extension(post_id: int, extension: str):
    found = find_media(post_id, extension, lookup_db=False)
    if found is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return _media_response(found[0])


@app.get("/media/{post_id}")
def get_media(post_id: int):
    found = find_media(post_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return _media_response(found[0])


@app.get("/images/{post_id}.{extension}")
def get_image(post_id: int, extension: str):
    found = find_media(post_id, extension)
    if found is None or found[1]:
        raise HTTPException(status_code=404, detail="Image not found")
    return _media_response(found[0])


@app.get("/videos/{post_id}.{extension}")
def get_video(post_id: int, extension: str):
    found = find_media(post_id, extension)
    if found is None or not found[1]:
        raise HTTPException(status_code=404, detail="Video not found")
    return _media_response(found[0])


@app.get("/thumbnails/{post_id}.jpg")
def get_thumbnail(post_id: int):
    path = thumbnail_path(post_id)
    if path.exists():
        return FileResponse(path, media_type="image/jpeg", headers=IMMUTABLE_CACHE)

    row = read_connection().execute(
        "SELECT extension, mime_type FROM posts WHERE id = ?", (post_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Post not found")

    generated = ensure_thumbnail(post_id, row["extension"], row["mime_type"])
    if generated is None:
        raise HTTPException(status_code=404, detail="Thumbnail not available")
    return FileResponse(generated, media_type="image/jpeg", headers=IMMUTABLE_CACHE)


@app.get("/api/posts")
def get_posts(
    q: str = Query(default=""),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
):
    conn = read_connection()
    query = (q or "").strip()
    offset = (page - 1) * limit

    def count(plan: search.CatalogQuery) -> int:
        cached = search.cached_count(plan.cache_key)
        if cached is not None:
            return cached
        total = conn.execute(plan.count_sql, plan.params).fetchone()[0]
        search.store_count(plan.cache_key, total)
        return total

    plan = search.build_query(query, conn)
    total = count(plan)

    # Prefix matching can miss a substring the user meant; only then pay for a scan.
    if total == 0 and plan.used_fts and query:
        plan = search.build_like_query(query)
        total = count(plan)

    posts = []
    if total:
        rows = conn.execute(
            plan.select_sql + " LIMIT ? OFFSET ?", (*plan.params, limit, offset)
        ).fetchall()
        posts = [enrich_post(dict(row)) for row in rows]
    return {"posts": posts, "total": total, "page": page, "limit": limit}


@app.get("/api/recent")
def get_recent(after_id: int = Query(default=0, ge=0)):
    rows = read_connection().execute(
        f"SELECT {POST_FIELDS} FROM posts WHERE status = 'completed' AND id > ?"
        " ORDER BY id ASC LIMIT 20",
        (after_id,),
    ).fetchall()
    return {"posts": [enrich_post(dict(row)) for row in rows]}


@app.get("/api/tags")
def get_tags(prefix: str = Query(default="", min_length=1)):
    """Return up to 20 tag suggestions that start with the given prefix."""
    return {"tags": search.tag_index.prefix_search(prefix, limit=20)}


# Scrape management endpoints
@app.post("/api/scrape/start")
async def start_scrape(
    start_id: Optional[int] = Query(default=None),
    end_id: Optional[int] = Query(default=None),
    limit: Optional[int] = Query(default=None),
    backfill: bool = Query(default=False),
):
    global scrape_job, _scrape_task, _event_forwarder
    logger.info("Scrape start request received")

    if scrape_job and scrape_job.running:
        return {"error": "Scrape already running", "status": scrape_job.get_status()}

    if start_id is None:
        start_id = await asyncio.to_thread(_resume_start_id)

    logger.info(f"Scrape range: {start_id} to {end_id}, limit={limit}, backfill={backfill}")

    try:
        progress_queue = asyncio.Queue()
        scrape_job = ScrapeJob(
            start_id=start_id,
            end_id=end_id,
            limit=limit,
            progress_queue=progress_queue,
            main_loop=asyncio.get_running_loop(),
            backfill=backfill,
        )

        task = _scrape_task = asyncio.create_task(scrape_job.run())
        if _event_forwarder is not None:
            _event_forwarder.cancel()
        _event_forwarder = asyncio.create_task(_forward_events(progress_queue))

        # Wait for the job to actually start (or fail)
        for _ in range(30):
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


def _resume_start_id() -> int:
    """Resume from the highest stored post instead of walking the media dirs."""
    try:
        row = read_connection().execute(
            "SELECT MAX(id) FROM posts WHERE status IN ('completed', 'empty')"
        ).fetchone()
        if row and row[0]:
            return int(row[0])
    except sqlite3.Error as e:
        logger.warning("Could not read resume point from database: %s", e)

    known = set()
    for directory in (IMAGES_DIR, VIDEOS_DIR):
        for f in directory.iterdir():
            if f.is_file() and f.stem.isdigit():
                known.add(int(f.stem))
    return max(known) if known else 1


async def _forward_events(progress_queue: asyncio.Queue):
    """Fan scrape events out to every connected SSE client."""
    try:
        while True:
            event = await progress_queue.get()
            for q in list(scrape_event_queues):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass  # a stalled browser tab must not stall the scrape
            if isinstance(event, dict) and event.get("type") in ("post_done", "complete"):
                search.invalidate_counts()
            if isinstance(event, dict) and event.get("type") == "complete":
                search.tag_index.dirty = True
                asyncio.get_running_loop().run_in_executor(None, search.tag_index.build)
    except asyncio.CancelledError:
        pass


@app.post("/api/scrape/stop")
async def stop_scrape():
    if scrape_job and scrape_job.running:
        scrape_job.cancel()
        return {"message": "Scrape cancellation requested"}
    return {"message": "No active scrape to stop"}


@app.get("/api/scrape/status")
async def get_scrape_status():
    if scrape_job:
        return scrape_job.get_status()
    return {"running": False, "message": "No scrape job"}


@app.get("/api/events")
async def events():
    """Server-Sent Events stream for real-time scrape progress."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=SSE_QUEUE_LIMIT)
    scrape_event_queues.append(queue)

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=str(config.get("host", "127.0.0.1")),
        port=int(config.get("port", 8000)),
        log_level="info",
    )
