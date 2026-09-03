"""Catalog search: full-text query building, result counting and tag suggestions."""

import bisect
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from typing import Optional

import library

logger = logging.getLogger(__name__)

SEARCH_COLUMNS = ("tags", "variant", "subvariant", "uploader")
POST_FIELDS = ("id", "width", "height", "extension", "mime_type", "tags", "variant",
               "subvariant", "uploader", "date_uploaded")
_HAS_ALNUM = re.compile(r"[0-9A-Za-z]")

# The full-text index is joined with CROSS JOIN on purpose: it pins the index as
# the outer loop. Left to its own devices SQLite sometimes scans posts instead
# and re-runs the match for every row, which turns a 2ms query into a 20s one.
_FTS_FROM = "posts_fts f CROSS JOIN posts p ON p.id = f.rowid"
_PLAIN_FROM = "posts p"


def _fts_prefix(term: str) -> str:
    """Quote a user term as an FTS5 prefix phrase."""
    return '"%s"*' % term.replace('"', '""')


def _fts_supported(conn: sqlite3.Connection) -> bool:
    try:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='posts_fts'"
        ).fetchone() is not None
    except sqlite3.Error:
        return False


def parse_terms(q: str) -> list[tuple[str, str]]:
    """Split a query into (kind, value) pairs: variant, subvariant, id or general."""
    terms = []
    for term in (q or "").strip().split():
        if term.startswith("variant:") and len(term) > 8:
            terms.append(("variant", term[8:]))
        elif term.startswith("subvariant:") and len(term) > 11:
            terms.append(("subvariant", term[11:]))
        elif term.isdigit():
            terms.append(("id", term))
        else:
            terms.append(("general", term))
    return terms


def _like_conditions(terms) -> tuple[list[str], list]:
    """Substring matching. Correct for any input but scans the whole table."""
    conditions, params = [], []
    for kind, value in terms:
        like = f"%{value}%"
        if kind in ("variant", "subvariant"):
            conditions.append(f"(p.{kind} LIKE ?)")
            params.append(like)
        elif kind == "id":
            conditions.append("(p.id = ? OR " + " OR ".join(
                f"p.{col} LIKE ?" for col in SEARCH_COLUMNS) + ")")
            params.extend([int(value), like, like, like, like])
        else:
            conditions.append("(" + " OR ".join(
                f"p.{col} LIKE ?" for col in SEARCH_COLUMNS) + ")")
            params.extend([like, like, like, like])
    return conditions, params


def _fts_match_all(value: str) -> str:
    return "{%s} : %s" % (" ".join(SEARCH_COLUMNS), _fts_prefix(value))


def _fts_conditions(terms) -> Optional[tuple[str, list[str], list]]:
    """Index-backed prefix matching.

    Returns (match_expression, extra_conditions, extra_params), or None when a
    term holds nothing the tokeniser can index.
    """
    match_parts, conditions, params = [], [], []
    for kind, value in terms:
        if not _HAS_ALNUM.search(value):
            return None
        if kind in ("variant", "subvariant"):
            match_parts.append(f"{kind} : {_fts_prefix(value)}")
        elif kind == "id":
            # An id term also matches posts whose tags contain those digits.
            conditions.append(
                "(p.id = ? OR p.id IN (SELECT rowid FROM posts_fts WHERE posts_fts MATCH ?))")
            params.extend([int(value), _fts_match_all(value)])
        else:
            match_parts.append(_fts_match_all(value))

    return " AND ".join(match_parts), conditions, params


@dataclass
class CatalogQuery:
    """A prepared catalog listing: the page query, its COUNT and their params."""
    select_sql: str
    count_sql: str
    params: list
    used_fts: bool
    cache_key: str


def _assemble(from_sql: str, conditions: list[str], params: list, order_by: str,
              used_fts: bool, cache_key: str) -> CatalogQuery:
    fields = ", ".join(f"p.{c}" for c in POST_FIELDS)
    where = " AND ".join(conditions)
    return CatalogQuery(
        select_sql=f"SELECT {fields} FROM {from_sql} WHERE {where} ORDER BY {order_by}",
        count_sql=f"SELECT COUNT(*) FROM {from_sql} WHERE {where}",
        params=params,
        used_fts=used_fts,
        cache_key=cache_key,
    )


def build_like_query(q: str) -> CatalogQuery:
    conditions, params = _like_conditions(parse_terms(q))
    return _assemble(_PLAIN_FROM, ["p.status = 'completed'"] + conditions, params,
                     "p.id DESC", False, f"like:{q}")


def build_query(q: str, conn: sqlite3.Connection) -> CatalogQuery:
    """Plan a catalog query, preferring the full-text index when it applies."""
    query = (q or "").strip()
    terms = parse_terms(query)
    if not terms:
        return _assemble(_PLAIN_FROM, ["p.status = 'completed'"], [], "p.id DESC",
                         False, "all:")

    if _fts_supported(conn):
        fts = _fts_conditions(terms)
        if fts is not None:
            match_expr, conditions, params = fts
            if match_expr:
                # Ordering by the index rowid lets FTS stream matches newest
                # first and stop at LIMIT instead of sorting every match.
                return _assemble(
                    _FTS_FROM,
                    ["f.posts_fts MATCH ?", "p.status = 'completed'"] + conditions,
                    [match_expr] + params, "f.rowid DESC", True, f"fts:{query}")
            return _assemble(_PLAIN_FROM, ["p.status = 'completed'"] + conditions,
                             params, "p.id DESC", True, f"fts:{query}")

    return build_like_query(query)


# ── Result counts ─────────────────────────────────────────────────────────────
# Counting matches means visiting every match, so the totals that drive
# pagination are cached until the library changes.
_count_cache: dict[str, int] = {}
_count_lock = threading.Lock()
COUNT_CACHE_LIMIT = 512


def cached_count(key: str) -> Optional[int]:
    with _count_lock:
        return _count_cache.get(key)


def store_count(key: str, total: int):
    with _count_lock:
        if len(_count_cache) >= COUNT_CACHE_LIMIT:
            _count_cache.clear()
        _count_cache[key] = total


def invalidate_counts():
    with _count_lock:
        _count_cache.clear()


# ── Tag suggestions ───────────────────────────────────────────────────────────
class TagIndex:
    """Alphabetically sorted tag list plus post counts, for autocomplete.

    Rebuilt in the background so a scrape finishing never stalls a keystroke.
    """

    def __init__(self):
        self._sorted: list[str] = []
        self._freq: dict[str, int] = {}
        self._lock = threading.Lock()
        self._building = False
        self.dirty = True

    def _scan(self, conn: sqlite3.Connection) -> dict[str, int]:
        counts: dict[str, int] = {}
        for (row,) in conn.execute(
                "SELECT tags FROM posts WHERE status = 'completed' AND tags IS NOT NULL"):
            for tok in row.split():
                t = tok.lower()
                counts[t] = counts.get(t, 0) + 1

        for column in ("variant", "subvariant"):
            for (row,) in conn.execute(
                    f"SELECT {column} FROM posts WHERE status = 'completed'"
                    f" AND {column} IS NOT NULL"):
                for tok in row.split(","):
                    t = tok.strip().lower()
                    if not t:
                        continue
                    # Indexed twice: bare for plain search, prefixed so that
                    # "variant:" autocompletes only within that category.
                    counts[f"{column}:{t}"] = counts.get(f"{column}:{t}", 0) + 1
                    counts[t] = counts.get(t, 0) + 1
        return counts

    def build(self, db_path=None):
        with self._lock:
            if self._building:
                return
            self._building = True
        try:
            logger.info("Building tag index...")
            conn = sqlite3.connect(str(db_path or library.DB_PATH), timeout=10.0)
            try:
                counts = self._scan(conn)
            finally:
                conn.close()
            ordered = sorted(counts.keys())
            with self._lock:
                self._freq = counts
                self._sorted = ordered
                self.dirty = False
            logger.info("Tag index built with %d unique tags", len(ordered))
        except sqlite3.Error as e:
            logger.warning("Tag index build failed: %s", e)
        finally:
            with self._lock:
                self._building = False

    def prefix_search(self, prefix: str, limit: int = 20) -> list[str]:
        """Tags starting with `prefix`, most used first."""
        if self.dirty and not self._sorted:
            self.build()
        with self._lock:
            tags, freq = self._sorted, self._freq
        p = prefix.lower()
        lo = bisect.bisect_left(tags, p)
        matches = []
        for i in range(lo, len(tags)):
            tag = tags[i]
            if not tag.startswith(p):
                break
            matches.append(tag)
        matches.sort(key=lambda t: -freq.get(t, 0))
        return matches[:limit]


tag_index = TagIndex()
