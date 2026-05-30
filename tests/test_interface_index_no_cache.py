"""The root HTML document must be served no-cache.

Asset references in index.html are cache-busted with a `?v=` query
(e.g. `style.css?v=2.7`). If the browser caches the *document* itself, it
keeps serving the old `?v=` link and never fetches updated CSS/JS until a
manual hard refresh. Serving `/` with `Cache-Control: no-cache` forces the
browser to revalidate the document each load, so version bumps take effect
on a plain refresh.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("GHOST_API_KEY", "test-ghost-key")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import interface.server as server  # noqa: E402
from interface.server import GHOST_API_KEY, get  # noqa: E402


@pytest.mark.asyncio
async def test_root_html_sets_no_cache():
    resp = await get(key=GHOST_API_KEY)
    assert resp.status_code == 200
    cc = resp.headers.get("cache-control", "")
    assert "no-cache" in cc
    # Document carries the injected API key, so it must not be shared-cached.
    assert "private" in cc


@pytest.mark.asyncio
async def test_root_html_injects_key_and_serves_fresh_from_disk():
    """Body is read from disk per request (no restart needed) and carries
    the injected key global the front-end reads."""
    resp = await get(key=GHOST_API_KEY)
    body = resp.body.decode("utf-8")
    assert "window.GHOST_API_KEY=" in body
    # Served body reflects the current on-disk index.html (proves the
    # per-request read at server.py — no restart needed for edits).
    on_disk = (server.static_dir / "index.html").read_text()
    assert "chat-container" in on_disk
    assert "chat-container" in body


@pytest.mark.asyncio
async def test_root_unauthorized_without_key():
    resp = await get(key=None)
    assert resp.status_code == 401
