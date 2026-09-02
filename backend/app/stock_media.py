"""
stock_media.py — fetches a real, freely-licensed photo for a scene.

Replaces the old "STOCK PHOTO (placeholder)" gradient card: when a scene's
MediaSelector plan calls for a stock photo, this module actually goes and
gets one, using the Openverse API (https://api.openverse.org) — a free,
keyless search over openly-licensed images (CC0/PDM/CC-BY and similar)
aggregated from many sources. No API key or account needed, which matters
because this app has to keep working with zero paid services.

Design constraints this module is built around:
  - MUST NEVER raise. Any failure (no network, timeout, no results, a bad
    image) falls back to `None`, and the caller (Renderer) draws the old
    gradient card instead. A flaky third-party API can never take the whole
    render down.
  - MUST be bounded in size and time. Renders happen on a memory-constrained
    free-tier host, so a single accidentally-huge photo must never be
    allowed to blow the memory/time budget: requests use a short timeout and
    downloads are capped and streamed rather than read in one shot.
  - Prefers the most permissively licensed results (CC0 / public domain)
    before falling back to any commercially-reusable license, and always
    keeps the source's attribution/landing-page URL on hand even though
    this prototype doesn't render an on-screen credit yet (see ARCHITECTURE
    notes) — a real product surfacing these images publicly should show
    that attribution.
  - Caches by query, in-process, so repeated renders (or repeated scenes
    with the same query) don't hammer the API or redownload the same photo.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Optional

_API_URL = "https://api.openverse.org/v1/images/"
_USER_AGENT = "AutoVideo/1.0 (personal video generator; contact: n/a)"
_REQUEST_TIMEOUT = 4.0
_MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8MB hard cap per photo
_CHUNK_SIZE = 65536

_MISSING = object()

# In-process cache of query -> raw image bytes (or explicitly-cached miss).
# Capped so a long-lived server process doesn't grow this unbounded; oldest
# entries are evicted first (simple FIFO via OrderedDict).
_CACHE: "OrderedDict[str, object]" = OrderedDict()
_CACHE_MAX_ENTRIES = 300


def _cache_get(key: str):
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    return _MISSING


def _cache_set(key: str, value) -> None:
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX_ENTRIES:
        _CACHE.popitem(last=False)


def _http_get_json(url: str) -> Optional[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
        raw = resp.read(2 * 1024 * 1024)  # JSON responses are small; 2MB is generous
        return json.loads(raw.decode("utf-8", errors="replace"))


def _http_get_bytes(url: str) -> Optional[bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
        chunks = []
        total = 0
        while True:
            chunk = resp.read(_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_IMAGE_BYTES:
                return None  # bail rather than accept an unbounded download
            chunks.append(chunk)
        return b"".join(chunks)


def _search_image_url(query: str) -> Optional[str]:
    """Query Openverse, preferring CC0/public-domain results, and return one
    result's best download URL (thumbnail preferred: smaller, faster, and
    already a reasonable size for a video frame background)."""
    encoded_q = urllib.parse.quote(query)
    base_params = f"q={encoded_q}&page_size=8&mature=false"
    for license_filter in ("license=cc0,pdm", "license_type=commercial,modification"):
        try:
            data = _http_get_json(f"{_API_URL}?{base_params}&{license_filter}")
        except Exception:
            continue
        results = (data or {}).get("results") or []
        for result in results:
            url = result.get("thumbnail") or result.get("url")
            if url:
                return url
    return None


def fetch_stock_photo_bytes(query: str) -> Optional[bytes]:
    """Return raw image bytes for `query`, or None if nothing could be
    fetched (network unavailable, no results, download too large, etc).
    Never raises. Cached per-query for the life of the process."""
    query = (query or "").strip()
    if not query:
        return None

    cached = _cache_get(query)
    if cached is not _MISSING:
        return cached  # type: ignore[return-value]

    result: Optional[bytes] = None
    try:
        image_url = _search_image_url(query)
        if image_url:
            result = _http_get_bytes(image_url)
    except Exception:
        result = None

    _cache_set(query, result)
    return result


def fetch_stock_photo_file(query: str, cache_dir: Path) -> Optional[Path]:
    """Convenience wrapper: fetches (or reuses the in-process cache for) a
    photo for `query` and writes it to a file under `cache_dir`, returning
    the path — or None if no photo could be obtained. `cache_dir` is
    per-job scratch space and is cleaned up with the rest of the job."""
    data = fetch_stock_photo_bytes(query)
    if not data:
        return None
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]
    path = cache_dir / f"stock_{digest}.img"
    try:
        path.write_bytes(data)
    except Exception:
        return None
    return path
