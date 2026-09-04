"""Shared library layer: configuration, paths, database access and media helpers.

Both the scraper and the web server build on this module so that a library
scraped from the CLI and a library browsed in the UI always agree on where
files live and how they are indexed.
"""

import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"

DEFAULT_CONFIG = {
    # Scraper
    "concurrency": 3,
    "delay_ms": 2000,
    "data_dir": "./data",
    "validate_images": True,
    "sanitize_images": True,
    "validate_videos": True,
    "sanitize_videos": False,
    "pregenerate_thumbnails": True,
    # Server
    "host": "127.0.0.1",
    "port": 8000,
    "thumbnail_size": 300,
}

THUMBNAIL_QUALITY = 82
MAX_EXTENSION_LENGTH = 8
VIDEO_EXTENSIONS = {"mp4", "webm", "mov", "avi", "mkv", "ogv", "flv", "wmv"}
_SAFE_EXT = re.compile(r"^[A-Za-z0-9]{1,%d}$" % MAX_EXTENSION_LENGTH)

DONE_STATUSES = ("completed", "skipped", "empty")


def load_config(path=None):
    """Read config.json, filling in any missing keys from DEFAULT_CONFIG."""
    path = Path(path) if path else CONFIG_FILE
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                config.update(json.load(f))
        except Exception as e:
            print(f"Error loading {path}, using defaults. Error: {e}")
            return dict(DEFAULT_CONFIG)
    return config


config = load_config()


def _resolve_data_dir(raw: str) -> Path:
    """Resolve the data directory relative to the project, not the shell's cwd."""
    env_override = os.environ.get("SOYBRARY_DATA_DIR")
    path = Path(env_override or raw).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


DATA_DIR = _resolve_data_dir(config["data_dir"])
IMAGES_DIR = DATA_DIR / "images"
VIDEOS_DIR = DATA_DIR / "videos"
METADATA_DIR = DATA_DIR / "metadata"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
DB_PATH = DATA_DIR / "soybooru.db"
# Post IDs that already failed a missing-entry (backfill) recovery pass.
BACKFILL_EXHAUSTED = DATA_DIR / ".backfill_exhausted"


def ensure_dirs():
    for d in (IMAGES_DIR, VIDEOS_DIR, METADATA_DIR, THUMBNAILS_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()


# ── Extensions ────────────────────────────────────────────────────────────────
def safe_extension(extension: Optional[str]) -> Optional[str]:
    """Return a filesystem-safe lowercase extension, or None if unusable.

    Extensions are derived from remote filenames, so they must never be able to
    escape the media directories.
    """
    if not extension:
        return None
    ext = str(extension).strip().lstrip(".").lower()
    return ext if _SAFE_EXT.match(ext) else None


def is_video_extension(extension: Optional[str], mime_type: Optional[str] = None) -> bool:
    ext = (extension or "").strip().lstrip(".").lower()
    if ext in VIDEO_EXTENSIONS:
        return True
    return bool(mime_type) and mime_type.startswith("video/")


# ── Database ──────────────────────────────────────────────────────────────────
SCHEMA_VERSION = 2

_SCHEMA = """
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
);

-- Every catalog query filters on status and orders by id; this index turns the
-- listing, the COUNT(*) and the resume scan into index-only lookups.
CREATE INDEX IF NOT EXISTS idx_posts_status_id ON posts(status, id);

CREATE TABLE IF NOT EXISTS library_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE posts_fts USING fts5(
    tags, variant, subvariant, uploader,
    content='posts', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER posts_fts_ai AFTER INSERT ON posts BEGIN
    INSERT INTO posts_fts(rowid, tags, variant, subvariant, uploader)
    VALUES (new.id, new.tags, new.variant, new.subvariant, new.uploader);
END;

CREATE TRIGGER posts_fts_ad AFTER DELETE ON posts BEGIN
    INSERT INTO posts_fts(posts_fts, rowid, tags, variant, subvariant, uploader)
    VALUES ('delete', old.id, old.tags, old.variant, old.subvariant, old.uploader);
END;

CREATE TRIGGER posts_fts_au AFTER UPDATE ON posts BEGIN
    INSERT INTO posts_fts(posts_fts, rowid, tags, variant, subvariant, uploader)
    VALUES ('delete', old.id, old.tags, old.variant, old.subvariant, old.uploader);
    INSERT INTO posts_fts(rowid, tags, variant, subvariant, uploader)
    VALUES (new.id, new.tags, new.variant, new.subvariant, new.uploader);
END;
"""

_WRITE_PRAGMAS = (
    "PRAGMA journal_mode=WAL",       # readers (the UI) never block on the scraper
    "PRAGMA synchronous=NORMAL",
    "PRAGMA busy_timeout=10000",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA cache_size=-16000",
)

_READ_PRAGMAS = (
    "PRAGMA busy_timeout=5000",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA cache_size=-32000",
    "PRAGMA mmap_size=268435456",
)


def fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts_probe USING fts5(x)")
        conn.execute("DROP TABLE temp.__fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


class Database:
    """Writer-side database handle used by the scraper.

    A single connection guarded by a lock: SQLite serialises writes anyway, and
    a shared connection keeps in-memory databases usable across threads.
    """

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(db_path), timeout=10.0, check_same_thread=False)
        self.has_fts = False
        for pragma in _WRITE_PRAGMAS:
            try:
                self.conn.execute(pragma)
            except sqlite3.OperationalError:
                pass
        self.setup()

    def setup(self):
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self._setup_fts()
            self.conn.execute(
                "INSERT INTO library_meta(key, value) VALUES('schema_version', ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(SCHEMA_VERSION),),
            )
            self.conn.commit()

    def _setup_fts(self):
        """Create the search index, rebuilding it once for pre-existing libraries."""
        if not fts5_available(self.conn):
            return
        existed = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='posts_fts'"
        ).fetchone()
        if not existed:
            self.conn.executescript(_FTS_SCHEMA)
        self.has_fts = True

        indexed_version = self.conn.execute(
            "SELECT value FROM library_meta WHERE key='fts_version'"
        ).fetchone()
        if existed and indexed_version and indexed_version[0] == str(SCHEMA_VERSION):
            return
        self.conn.execute("INSERT INTO posts_fts(posts_fts) VALUES('rebuild')")
        self.conn.execute(
            "INSERT INTO library_meta(key, value) VALUES('fts_version', ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )

    def get_post_status(self, post_id):
        with self._lock:
            row = self.conn.execute(
                "SELECT status FROM posts WHERE id = ?", (post_id,)
            ).fetchone()
        return row[0] if row else None

    def get_done_ids(self, start_id, end_id):
        """Every id in [start_id, end_id] that never needs to be fetched again.

        One indexed range scan replaces a per-id query while queueing work.
        """
        placeholders = ",".join("?" * len(DONE_STATUSES))
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id FROM posts WHERE status IN ({placeholders})"
                " AND id BETWEEN ? AND ?",
                (*DONE_STATUSES, start_id, end_id),
            ).fetchall()
        return {row[0] for row in rows}

    def get_resume_id(self):
        """Highest post id already stored, used as the default scrape start."""
        with self._lock:
            row = self.conn.execute(
                "SELECT MAX(id) FROM posts WHERE status IN ('completed', 'empty')"
            ).fetchone()
        return row[0] if row and row[0] else None

    def save_post(self, post_id, status, variant=None, subvariant=None, tags=None,
                  date_uploaded=None, file_url=None, width=None, height=None,
                  file_size=None, image_hash=None, mime_type=None, extension=None,
                  uploader=None, original_filename=None, error_message=None):
        import datetime

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            self.conn.execute("""
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
            self.conn.commit()

    def close(self):
        with self._lock:
            if self.conn is not None:
                self.conn.close()
                self.conn = None


_read_local = threading.local()


def read_connection(db_path=None) -> sqlite3.Connection:
    """Per-thread read connection. WAL lets these run while a scrape writes."""
    path = str(db_path or DB_PATH)
    conn = getattr(_read_local, "conn", None)
    if conn is not None and getattr(_read_local, "path", None) == path:
        return conn
    if conn is not None:
        conn.close()
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    for pragma in _READ_PRAGMAS:
        try:
            conn.execute(pragma)
        except sqlite3.OperationalError:
            pass
    _read_local.conn = conn
    _read_local.path = path
    return conn


def close_read_connection():
    conn = getattr(_read_local, "conn", None)
    if conn is not None:
        conn.close()
        _read_local.conn = None


# ── Media files ───────────────────────────────────────────────────────────────
_media_cache: dict[int, tuple[Path, bool]] = {}
_media_cache_lock = threading.Lock()
MEDIA_CACHE_LIMIT = 8192


def _cache_media(post_id: int, found: tuple[Path, bool]) -> tuple[Path, bool]:
    with _media_cache_lock:
        if len(_media_cache) >= MEDIA_CACHE_LIMIT:
            _media_cache.clear()
        _media_cache[post_id] = found
    return found


def media_path_for(post_id: int, extension: Optional[str], is_video: Optional[bool] = None) -> Optional[Path]:
    """Direct path for a known post id/extension pair, without touching disk."""
    ext = safe_extension(extension)
    if ext is None:
        return None
    base = VIDEOS_DIR if (is_video if is_video is not None else is_video_extension(ext)) else IMAGES_DIR
    return base / f"{post_id}.{ext}"


def find_media(post_id: int, extension: Optional[str] = None,
               lookup_db: bool = True) -> Optional[tuple[Path, bool]]:
    """Locate a post's media file. Returns (path, is_video) or None.

    Media files never change once scraped, so hits are memoised; scanning the
    media directories is the last resort because they hold one file per post.
    """
    cached = _media_cache.get(post_id)
    if cached is not None and cached[0].exists():
        return cached

    ext = safe_extension(extension)
    if extension and ext is None:
        return None  # an extension that can't be trusted is not worth guessing around
    if ext is None and lookup_db:
        try:
            row = read_connection().execute(
                "SELECT extension FROM posts WHERE id = ?", (post_id,)
            ).fetchone()
            if row:
                ext = safe_extension(row["extension"])
        except sqlite3.Error:
            ext = None

    if ext is not None:
        video_first = is_video_extension(ext)
        for base_dir, is_video in (((VIDEOS_DIR, True), (IMAGES_DIR, False)) if video_first
                                   else ((IMAGES_DIR, False), (VIDEOS_DIR, True))):
            candidate = base_dir / f"{post_id}.{ext}"
            if candidate.exists():
                return _cache_media(post_id, (candidate, is_video))

    for base_dir, is_video in ((IMAGES_DIR, False), (VIDEOS_DIR, True)):
        matches = sorted(base_dir.glob(f"{post_id}.*"))
        if matches:
            return _cache_media(post_id, (matches[0], is_video))
    return None


# ── Thumbnails ────────────────────────────────────────────────────────────────
_unthumbnailable: set[int] = set()
_unthumbnailable_lock = threading.Lock()


def thumbnail_path(post_id: int) -> Path:
    return THUMBNAILS_DIR / f"{post_id}.jpg"


def _mark_unthumbnailable(post_id: int):
    with _unthumbnailable_lock:
        if len(_unthumbnailable) >= MEDIA_CACHE_LIMIT:
            _unthumbnailable.clear()
        _unthumbnailable.add(post_id)


def _thumbnail_size() -> tuple[int, int]:
    size = int(config.get("thumbnail_size") or 300)
    return size, size


def _write_thumbnail(img: Image.Image, dest: Path) -> bool:
    """Downscale and save atomically so a crash can't leave a broken thumbnail."""
    # Palette and alpha modes fall back to nearest-neighbour when resampled, so
    # flatten first and let LANCZOS do the downscale.
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail(_thumbnail_size(), Image.LANCZOS, reducing_gap=2.0)
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".jpg", dir=str(THUMBNAILS_DIR))
    os.close(tmp_fd)
    try:
        img.save(tmp_name, "JPEG", quality=THUMBNAIL_QUALITY, optimize=True)
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, dest)
        return True
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _open_for_thumbnail(source) -> Image.Image:
    img = Image.open(source)
    # draft() lets libjpeg decode straight to a reduced size — far cheaper than
    # decoding a full-resolution frame just to shrink it.
    try:
        img.draft("RGB", _thumbnail_size())
    except Exception:
        pass
    return img


def thumbnail_from_bytes(post_id: int, data: bytes) -> Optional[Path]:
    """Build a thumbnail straight from freshly downloaded bytes."""
    dest = thumbnail_path(post_id)
    try:
        with _open_for_thumbnail(io.BytesIO(data)) as img:
            _write_thumbnail(img, dest)
        return dest
    except Exception:
        return None


def thumbnail_from_video(post_id: int, video_path: Path) -> Optional[Path]:
    """Extract the first frame of a video with ffmpeg."""
    if shutil.which("ffmpeg") is None:
        return None
    dest = thumbnail_path(post_id)
    width, height = _thumbnail_size()
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".jpg", dir=str(THUMBNAILS_DIR))
    os.close(tmp_fd)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", "0", "-i", str(video_path), "-vframes", "1", "-update", "1",
             "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease",
             "-q:v", "5", tmp_name],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and os.path.getsize(tmp_name) > 0:
            os.chmod(tmp_name, 0o644)
            os.replace(tmp_name, dest)
            return dest
        return None
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def ensure_thumbnail(post_id: int, extension: Optional[str] = None,
                     mime_type: Optional[str] = None) -> Optional[Path]:
    """Return the post's thumbnail, generating it on first use."""
    dest = thumbnail_path(post_id)
    if dest.exists():
        return dest
    if post_id in _unthumbnailable:
        return None

    found = find_media(post_id, extension)
    if found is None:
        return None
    path, is_video = found

    if is_video or is_video_extension(extension, mime_type):
        result = thumbnail_from_video(post_id, path)
    else:
        try:
            with _open_for_thumbnail(path) as img:
                _write_thumbnail(img, dest)
            result = dest
        except Exception:
            # Formats Pillow can't decode (swf, cbz, psd, ...) have no thumbnail.
            result = None

    if result is None:
        _mark_unthumbnailable(post_id)
    return result
