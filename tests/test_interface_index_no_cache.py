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
from interface.server import get  # noqa: E402
# ⚠ THE KEY IS READ FROM THE MODULE AT ASSERT TIME, never bound by value at
# import. `interface.server` computes `GHOST_API_KEY` from the environment
# at import, and `test_interface_chat_timeout.py` calls
# `importlib.reload(server)` — so a module-level `from ... import
# GHOST_API_KEY` here freezes whatever the env held when THIS file was
# imported, while the proxies under test read the module's CURRENT value.
# When a neighbour left `GHOST_API_KEY=test-key` in the environment, the two
# diverged and six interface tests failed under xdist:
#
#   assert {'X-Ghost-Key': 'test-key'} == {'X-Ghost-Key': '0dc28f40...'}
#
# The leak is fixed at its source (the Slack suites now restore the env),
# but reading it dynamically is what makes this file's assertion state what
# it means: the proxy forwards THE SERVER'S key, whatever it is.

def _fake_request(cookies=None, scheme="http", headers=None):
    """Minimal Request for calling the `/` handler directly.

    §4DW gave the handler a `request` parameter: `?key=` is how the key
    ARRIVES, but a cookie is how it PERSISTS, so the URL can be scrubbed
    without making the page a one-shot (a reload used to land on 401).
    """
    from starlette.requests import Request
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    if cookies:
        raw.append((b"cookie", "; ".join(
            f"{k}={v}" for k, v in cookies.items()).encode()))
    return Request({"type": "http", "method": "GET", "path": "/",
                    "scheme": scheme, "headers": raw, "query_string": b"",
                    "server": ("testserver", 80), "client": ("1.2.3.4", 1)})



@pytest.mark.asyncio
async def test_root_html_sets_no_cache():
    resp = await get(_fake_request(), key=server.GHOST_API_KEY)
    assert resp.status_code == 200
    cc = resp.headers.get("cache-control", "")
    assert "no-cache" in cc
    # Document carries the injected API key, so it must not be shared-cached.
    assert "private" in cc


@pytest.mark.asyncio
async def test_root_html_injects_key_and_serves_fresh_from_disk():
    """Body is read from disk per request (no restart needed) and carries
    the injected key global the front-end reads."""
    resp = await get(_fake_request(), key=server.GHOST_API_KEY)
    body = resp.body.decode("utf-8")
    assert "window.GHOST_API_KEY=" in body
    # Served body reflects the current on-disk index.html (proves the
    # per-request read at server.py — no restart needed for edits).
    on_disk = (server.static_dir / "index.html").read_text()
    assert "chat-container" in on_disk
    assert "chat-container" in body


@pytest.mark.asyncio
async def test_root_unauthorized_without_key():
    resp = await get(_fake_request(), key=None)
    assert resp.status_code == 401
