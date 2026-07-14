"""Tor fetch DNS-leak fixes — 2026-07-14 search/darkweb audit.

Both fetch helpers validated the target URL with the shared SSRF guard's
DEFAULT resolve=True, which does a HOST-SIDE getaddrinfo of the hostname —
leaking the DNS query for the very site being visited anonymously (and, for
a .onion, leaking which hidden service). The fetches always go over Tor
(routing/resolution happens at the exit node), so the host lookup bought no
protection and only leaked. The fix: resolve=False when a Tor proxy is in
use — mirroring the browser/download tools' `resolve=not anonymous`.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.asyncio


async def test_clearnet_fetch_over_tor_does_not_resolve_host(monkeypatch):
    """helper_fetch_url_content over the default Tor proxy must NOT call the
    host resolver (getaddrinfo) for the target URL."""
    from ghost_agent.utils import helpers

    monkeypatch.setenv("TOR_PROXY", "socks5://127.0.0.1:9050")
    calls = {"resolve": []}

    def _fake_ssrf(url, *, resolve=True):
        calls["resolve"].append(resolve)
        return None  # allow

    monkeypatch.setattr(helpers, "url_ssrf_reason", _fake_ssrf)

    with patch("ghost_agent.utils.helpers.httpx.AsyncClient") as mock_cls, \
         patch.dict("sys.modules", {"curl_cffi": None, "curl_cffi.requests": None}):
        client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = client
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/html"}
        resp.text = "<html><body><p>hi</p></body></html>"
        client.get.return_value = resp
        await helpers.helper_fetch_url_content("http://example.com")

    assert calls["resolve"] == [False], (
        "Tor fetch must pass resolve=False so the host resolver never sees "
        "the target hostname")


async def test_clearnet_fetch_without_proxy_still_resolves(monkeypatch):
    """With no Tor proxy configured, the resolve check is kept (no leak
    concern, and the resolves-to-internal SSRF check is worth running)."""
    from ghost_agent.utils import helpers

    monkeypatch.setenv("TOR_PROXY", "")  # explicitly no proxy
    calls = {"resolve": []}

    def _fake_ssrf(url, *, resolve=True):
        calls["resolve"].append(resolve)
        return None

    monkeypatch.setattr(helpers, "url_ssrf_reason", _fake_ssrf)

    with patch("ghost_agent.utils.helpers.httpx.AsyncClient") as mock_cls, \
         patch.dict("sys.modules", {"curl_cffi": None, "curl_cffi.requests": None}):
        client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = client
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/html"}
        resp.text = "<html><body>x</body></html>"
        client.get.return_value = resp
        await helpers.helper_fetch_url_content("http://example.com")

    assert calls["resolve"] == [True]


async def test_ssrf_block_still_short_circuits(monkeypatch):
    """A refused URL still returns the error and never fetches."""
    from ghost_agent.utils import helpers

    monkeypatch.setattr(helpers, "url_ssrf_reason",
                        lambda url, *, resolve=True: "refused internal host")

    with patch("ghost_agent.utils.helpers.httpx.AsyncClient") as mock_cls:
        out = await helpers.helper_fetch_url_content("http://169.254.169.254/")
        assert out.startswith("Error")
        assert "refused" in out
        mock_cls.assert_not_called()


async def test_onion_fetch_never_resolves(monkeypatch):
    """_fetch_onion_text must pass resolve=False — a getaddrinfo on a .onion
    leaks which hidden service is being visited to the host resolver."""
    from ghost_agent.tools import darkweb_search

    calls = {"resolve": []}

    def _fake_ssrf(url, *, resolve=True):
        calls["resolve"].append(resolve)
        return None

    monkeypatch.setattr(darkweb_search, "url_ssrf_reason", _fake_ssrf)

    async def _fake_raw(url, proxy, timeout):
        return 200, "<html><body>onion text</body></html>"

    monkeypatch.setattr(darkweb_search, "_fetch_raw_html", _fake_raw)

    out = await darkweb_search._fetch_onion_text(
        "http://abcdefabcdef1234.onion/page", "socks5://127.0.0.1:9050")

    assert calls["resolve"] == [False]
    assert "onion text" in out


async def test_onion_fetch_ssrf_block_short_circuits(monkeypatch):
    from ghost_agent.tools import darkweb_search

    monkeypatch.setattr(darkweb_search, "url_ssrf_reason",
                        lambda url, *, resolve=True: "refused non-http(s)")
    called = {"raw": False}

    async def _fake_raw(url, proxy, timeout):
        called["raw"] = True
        return 200, ""

    monkeypatch.setattr(darkweb_search, "_fetch_raw_html", _fake_raw)
    out = await darkweb_search._fetch_onion_text("file:///etc/passwd", "socks5://x")
    assert out.startswith("Error")
    assert called["raw"] is False


def test_filter_junk_survives_none_href():
    """A result with href=None must be skipped, not crash the whole batch."""
    from ghost_agent.tools.search import _filter_junk

    rows = [
        {"href": None, "title": "malformed"},
        {"href": "https://good.example.com/a", "title": "ok"},
        {"url": "https://also-good.example.com/b"},
        {"href": "", "title": "empty"},
    ]
    out = _filter_junk(rows)
    urls = {r.get("href") or r.get("url") for r in out}
    assert urls == {"https://good.example.com/a", "https://also-good.example.com/b"}
