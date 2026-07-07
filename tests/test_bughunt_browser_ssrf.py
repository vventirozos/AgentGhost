"""Regression test for the deferred browser SSRF finding (HIGH).

The sandbox runs under HOST networking, so a page the agent navigates to that
302-redirects to an internal address (http://127.0.0.1:9051 Tor control,
169.254.169.254 cloud metadata, a LAN host) — or that pulls an internal
subresource (blind SSRF) — would reach host-local services. The host-side
guard only vetted the INITIAL url and Chromium does not re-vet redirects.

The fix installs a Playwright request interceptor in the in-sandbox runner that
aborts every http(s) request to an internal host. These tests exec the REAL
embedded runner source (with playwright mocked, since it isn't installed here)
and exercise the actual decision function + route handler.
"""

import sys
import types

import pytest
from unittest.mock import AsyncMock, MagicMock


def _load_runner_namespace():
    # The runner does `from playwright.async_api import async_playwright` at
    # top level; stub it so we can exec the source and test its functions.
    if "playwright" not in sys.modules:
        sys.modules["playwright"] = types.ModuleType("playwright")
    if "playwright.async_api" not in sys.modules:
        m = types.ModuleType("playwright.async_api")
        m.async_playwright = lambda: None
        sys.modules["playwright.async_api"] = m
    import ghost_agent.tools.browser as browser
    ns: dict = {}
    exec(compile(browser._runner_script(), "<browser_runner>", "exec"), ns)
    return ns


NS = _load_runner_namespace()


# ══════════════════════════════════════════════════════════════════════
# Host classification
# ══════════════════════════════════════════════════════════════════════

class TestHostClassification:
    @pytest.mark.parametrize("host", [
        "127.0.0.1", "127.1.2.3", "::1", "localhost", "ip6-localhost",
        "169.254.169.254", "169.254.0.1",       # link-local + cloud metadata
        "10.0.0.5", "172.16.0.9", "192.168.1.1",  # RFC1918 private
        "0.0.0.0",                                 # unspecified
        "metadata.google.internal",
    ])
    def test_internal_hosts_flagged(self, host):
        assert NS["_host_is_internal"](host) is True

    @pytest.mark.parametrize("host", [
        "8.8.8.8", "1.1.1.1", "93.184.216.34", "example.com",
        "duckduckgo.com", "",
    ])
    def test_external_hosts_not_flagged(self, host):
        assert NS["_host_is_internal"](host) is False


# ══════════════════════════════════════════════════════════════════════
# Per-URL block decision
# ══════════════════════════════════════════════════════════════════════

class TestShouldBlock:
    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:9051/",              # Tor control
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "https://10.0.0.5/internal",
        "http://[::1]:8000/",
        "http://localhost:5432/",
    ])
    def test_internal_http_blocked(self, url):
        assert NS["_ssrf_should_block"](url) is True

    @pytest.mark.parametrize("url", [
        "https://example.com/page",
        "http://93.184.216.34/",
        "file:///workspace/game/index.html",   # self-play fixture — never blocked
        "about:blank",
        "data:text/html,<h1>hi</h1>",
        "",
    ])
    def test_safe_urls_pass(self, url):
        assert NS["_ssrf_should_block"](url) is False


# ══════════════════════════════════════════════════════════════════════
# Route handler behaviour (the actual interceptor)
# ══════════════════════════════════════════════════════════════════════

class TestRouteHandler:
    async def _handler(self):
        captured = {}

        class FakeCtx:
            async def route(self, pattern, handler):
                captured["pattern"] = pattern
                captured["handler"] = handler

        await NS["_install_ssrf_guard"](FakeCtx())
        assert captured["pattern"] == "**/*"
        return captured["handler"]

    async def test_aborts_internal_redirect_target(self):
        handler = await self._handler()
        route = MagicMock()
        route.request.url = "http://127.0.0.1:9051/"
        route.abort = AsyncMock()
        route.continue_ = AsyncMock()
        await handler(route)
        route.abort.assert_awaited_once()
        route.continue_.assert_not_called()

    async def test_continues_external(self):
        handler = await self._handler()
        route = MagicMock()
        route.request.url = "https://example.com/"
        route.abort = AsyncMock()
        route.continue_ = AsyncMock()
        await handler(route)
        route.continue_.assert_awaited_once()
        route.abort.assert_not_called()

    async def test_continues_file_fixture(self):
        handler = await self._handler()
        route = MagicMock()
        route.request.url = "file:///workspace/game/index.html"
        route.abort = AsyncMock()
        route.continue_ = AsyncMock()
        await handler(route)
        route.continue_.assert_awaited_once()
        route.abort.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# Wiring pin
# ══════════════════════════════════════════════════════════════════════

def test_guard_installed_before_navigation():
    import ghost_agent.tools.browser as browser
    src = browser._runner_script()
    # Installed inside _with_context, before op_fn runs. (The call now passes
    # sandbox_root/anonymous kwargs, so match the prefix, not the bare `(ctx)`.)
    assert "await _install_ssrf_guard(ctx," in src
    ctx_idx = src.index("launch_persistent_context")
    guard_idx = src.index("await _install_ssrf_guard(ctx")
    op_idx = src.index("result = await op_fn(page)")
    assert ctx_idx < guard_idx < op_idx
