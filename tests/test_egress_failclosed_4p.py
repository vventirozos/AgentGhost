"""§4P — egress fail-closed at the curl_cffi / browser CALL SITES.

Context (reproduced during the audit): the process-wide `egress_guard`
monkeypatches `socket.socket`, but `curl_cffi` opens its sockets in
libcurl/C (measured: a curl_cffi request produces ZERO
`socket.socket.connect` calls), and Chromium runs in a subprocess — so the
socket guard is BLIND to every real egress path. The fail-closed guarantee
therefore has to live at the call site: `resolve_egress_proxy` forces the
loopback Tor proxy for a public target when the guard is installed and no
proxy was threaded.

These tests pin that WIRING per call site — each is red-on-revert of the
respective `resolve_egress_proxy(...)` insertion (revert it and the site
hands curl_cffi `proxies=None` / Chromium no `--proxy-server` under the
guard, i.e. a direct cleartext connect).

CRITICAL: every install() is paired with uninstall() in finally so the
socket monkeypatch never leaks into the rest of the suite.
"""

import asyncio
from pathlib import Path

import curl_cffi.requests as _cr
import pytest

from ghost_agent.utils import egress_guard


@pytest.fixture(autouse=True)
def _guard_cleanup():
    yield
    if egress_guard.is_installed():
        o = egress_guard._ORIGINALS
        if o:
            import socket
            socket.socket.connect = o["connect"]
            socket.socket.connect_ex = o["connect_ex"]
            socket.socket.sendto = o["sendto"]
            socket.socket.sendmsg = o["sendmsg"]
        egress_guard._INSTALLED = False


class _FakeAsyncSession:
    """Records the `proxies` kwarg curl_cffi would use, then aborts before
    any real network I/O."""
    last_proxies = "UNSET"

    def __init__(self, *a, **kw):
        type(self).last_proxies = kw.get("proxies", None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, *a, **kw):
        raise RuntimeError("stop-before-network")


def _proxies_from(monkeypatch, module, coro_factory):
    """Patch a tool module's curl_cffi AsyncSession, run the coroutine, and
    return the `proxies` dict the tool handed curl_cffi."""
    _FakeAsyncSession.last_proxies = "UNSET"
    monkeypatch.setattr(module.curl_requests, "AsyncSession", _FakeAsyncSession)
    try:
        asyncio.run(coro_factory())
    except Exception:
        pass
    return _FakeAsyncSession.last_proxies


# ── curl_cffi really does bypass the socket guard (the premise) ───────────

def test_curl_cffi_bypasses_socket_guard():
    """Documents WHY the call-site backstop exists. If curl_cffi ever starts
    routing through Python sockets this fails — signalling the socket guard
    now DOES cover it (belt-and-suspenders rather than sole guarantee)."""
    import socket
    hits = {"n": 0}
    orig = socket.socket.connect

    def _tw(self, addr):
        hits["n"] += 1
        return orig(self, addr)

    socket.socket.connect = _tw
    try:
        try:
            _cr.get("http://127.0.0.1:9", timeout=0.3)
        except Exception:
            pass
    finally:
        socket.socket.connect = orig
    assert hits["n"] == 0


# ── weather (system.py) ───────────────────────────────────────────────────

def test_weather_failclosed_under_guard(monkeypatch):
    from ghost_agent.tools import system as sysmod
    u = egress_guard.install("socks5://127.0.0.1:9050")
    try:
        proxies = _proxies_from(
            monkeypatch, sysmod,
            lambda: sysmod.tool_get_weather(tor_proxy=None, location="London"),
        )
    finally:
        u()
    assert isinstance(proxies, dict) and "socks5h://127.0.0.1:9050" in proxies.values(), \
        f"weather leaked cleartext under --mandatory-tor: proxies={proxies!r}"


def test_weather_direct_without_guard(monkeypatch):
    # --no-mandatory-tor: unchanged (direct). No regression to the non-Tor path.
    from ghost_agent.tools import system as sysmod
    assert not egress_guard.is_installed()
    proxies = _proxies_from(
        monkeypatch, sysmod,
        lambda: sysmod.tool_get_weather(tor_proxy=None, location="London"),
    )
    assert proxies is None


# ── download (file_system.py) ─────────────────────────────────────────────

def test_download_failclosed_under_guard(monkeypatch):
    from ghost_agent.tools import file_system as fsmod
    u = egress_guard.install("socks5://127.0.0.1:9050")
    try:
        proxies = _proxies_from(
            monkeypatch, fsmod,
            lambda: fsmod.tool_download_file(
                url="http://example.com/x", sandbox_dir=Path("/tmp"),
                tor_proxy=None, filename="x"),
        )
    finally:
        u()
    assert isinstance(proxies, dict) and "socks5h://127.0.0.1:9050" in proxies.values(), \
        f"download leaked cleartext under --mandatory-tor: proxies={proxies!r}"


def test_download_direct_without_guard(monkeypatch):
    from ghost_agent.tools import file_system as fsmod
    assert not egress_guard.is_installed()
    proxies = _proxies_from(
        monkeypatch, fsmod,
        lambda: fsmod.tool_download_file(
            url="http://example.com/x", sandbox_dir=Path("/tmp"),
            tor_proxy=None, filename="x"),
    )
    assert proxies is None


# ── search (search.py) ────────────────────────────────────────────────────

def test_search_failclosed_under_guard(monkeypatch):
    """tool_search_ddgs resolves the proxy before fanning out — pin that the
    (always-public) engine race receives a non-empty proxy under the guard."""
    from ghost_agent.tools import search as searchmod
    seen = {}

    async def _fake_wave(query, tor_proxy, wave, *a, **kw):
        seen["proxy"] = tor_proxy
        return [{"href": "http://x", "title": "t", "body": "b"}]

    monkeypatch.setattr(searchmod, "_race_search_wave", _fake_wave)
    u = egress_guard.install("socks5://127.0.0.1:9050")
    try:
        asyncio.run(searchmod.tool_search_ddgs("hello world", tor_proxy=None))
    finally:
        u()
    assert seen.get("proxy"), f"search fanned out with no proxy under guard: {seen!r}"
    assert "127.0.0.1:9050" in seen["proxy"]


def test_search_direct_without_guard(monkeypatch):
    from ghost_agent.tools import search as searchmod
    seen = {}

    async def _fake_wave(query, tor_proxy, wave, *a, **kw):
        seen["proxy"] = tor_proxy
        return [{"href": "http://x", "title": "t", "body": "b"}]

    monkeypatch.setattr(searchmod, "_race_search_wave", _fake_wave)
    assert not egress_guard.is_installed()
    asyncio.run(searchmod.tool_search_ddgs("hello world", tor_proxy=None))
    assert not seen.get("proxy")  # unchanged (None): no forced Tor off-guard


def test_deep_research_failclosed_under_guard(monkeypatch):
    """deep_research fans out to _race_search_wave DIRECTLY (bypassing
    tool_search_ddgs) — pin that it resolves the proxy under the guard too."""
    from ghost_agent.tools import search as searchmod
    seen = {}

    async def _fake_wave(query, tor_proxy, wave, *a, **kw):
        seen["proxy"] = tor_proxy
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(searchmod, "_race_search_wave", _fake_wave)
    u = egress_guard.install("socks5://127.0.0.1:9050")
    try:
        try:
            asyncio.run(searchmod.tool_deep_research("hello world", tor_proxy=None))
        except Exception:
            pass
    finally:
        u()
    assert seen.get("proxy") and "127.0.0.1:9050" in seen["proxy"], \
        f"deep_research fanned out with no proxy under guard: {seen!r}"


# ── connectivity probe (system.py tool_check_health) ──────────────────────

def test_check_health_probe_failclosed_under_guard(monkeypatch):
    """The 1.1.1.1 internet probe used context.tor_proxy directly; pin that a
    missing context proxy is replaced with the Tor default under the guard
    (so the diagnostic itself can't leak cleartext)."""
    from types import SimpleNamespace
    from ghost_agent.tools import system as sysmod
    ctx = SimpleNamespace(tor_proxy=None)
    u = egress_guard.install("socks5://127.0.0.1:9050")
    try:
        proxies = _proxies_from(
            monkeypatch, sysmod,
            lambda: sysmod.tool_check_health(context=ctx),
        )
    finally:
        u()
    assert isinstance(proxies, dict) and "socks5h://127.0.0.1:9050" in proxies.values(), \
        f"connectivity probe leaked cleartext under --mandatory-tor: {proxies!r}"


# ── browser (browser.py) ──────────────────────────────────────────────────

def _browser_payload_proxy(monkeypatch, url):
    """Drive tool_browser far enough to build the op payload and capture the
    tor_proxy that would reach Chromium's --proxy-server."""
    from ghost_agent.tools import browser as browsermod
    captured = {}
    real_build = browsermod._build_op_payload

    def _spy(**kw):
        captured["tor_proxy"] = kw.get("tor_proxy")
        raise RuntimeError("stop-before-subprocess")

    monkeypatch.setattr(browsermod, "_build_op_payload", _spy)

    class _SM:  # minimal sandbox_manager stand-in
        tor_proxy = None
    try:
        asyncio.run(browsermod.tool_browser(
            operation="navigate", url=url,
            sandbox_dir=Path("/tmp"), sandbox_manager=_SM(),
            tor_proxy=None))
    except Exception:
        pass
    _ = real_build  # keep a ref (avoid lint drop); not called
    return captured.get("tor_proxy", "NO-BUILD")


def test_browser_public_failclosed_under_guard(monkeypatch):
    u = egress_guard.install("socks5://127.0.0.1:9050")
    try:
        proxy = _browser_payload_proxy(monkeypatch, "http://example.com/page")
    finally:
        u()
    assert proxy and "127.0.0.1:9050" in str(proxy), \
        f"browser navigated a public URL with no proxy under guard: {proxy!r}"


def test_browser_loopback_stays_direct_under_guard(monkeypatch):
    # A supervised sandbox service on loopback must NOT be forced through Tor
    # (Tor can't route loopback) — resolve returns unchanged (None).
    u = egress_guard.install("socks5://127.0.0.1:9050")
    try:
        proxy = _browser_payload_proxy(monkeypatch, "http://127.0.0.1:8100/app")
    finally:
        u()
    assert proxy in (None, "NO-BUILD"), \
        f"loopback nav should stay direct, got proxy={proxy!r}"
