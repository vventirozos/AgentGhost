"""Regression tests for the two RESIDUAL browser-SSRF gaps closed on top of
the 2026-07-04 in-sandbox interceptor.

The interceptor already aborted http(s) requests to an internal HOST STRING
(literal IP / known local name), covering redirects + subresources. Two gaps
stayed open:

  1. file:// container-read — the sandbox runs under HOST mounts, so a
     file:///etc/passwd (or any path above the mount) let the browser read
     files OUTSIDE the sandbox subtree. The interceptor now allows file:// only
     when its resolved REAL path stays inside `sandbox_root`.

  2. DNS-rebind of a subresource host — the top-level host was vetted while it
     pointed at a public IP, but a later fetch re-resolves the SAME name to an
     internal IP (169.254.169.254, 127.0.0.1, RFC1918), which the host-STRING
     check can't see. In non-Tor mode the interceptor now re-resolves each
     request's host and blocks an internal result. Over Tor (anonymous) DNS is
     at the exit node, so the local lookup is skipped (no query leak).

Like tests/test_bughunt_browser_ssrf.py, these exec the REAL embedded runner
source (playwright + DNS mocked, no real network I/O) and drive the actual
decision function + route handler.
"""

import os
import socket
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


def _addrinfo(ip, port=80):
    """A minimal getaddrinfo() result row: (family, type, proto, canon, addr)."""
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]


# ══════════════════════════════════════════════════════════════════════
# Gap 1 — file:// container-read must stay inside the sandbox subtree
# ══════════════════════════════════════════════════════════════════════

class TestFileEscapesSandbox:
    def test_inside_subtree_allowed(self, tmp_path):
        root = tmp_path / "workspace"
        (root / "game").mkdir(parents=True)
        fixture = root / "game" / "index.html"
        fixture.write_text("<h1>fixture</h1>")
        assert NS["_file_escapes_sandbox"](str(fixture), str(root)) is False

    def test_root_itself_allowed(self, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        assert NS["_file_escapes_sandbox"](str(root), str(root)) is False

    def test_absolute_escape_blocked(self, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        assert NS["_file_escapes_sandbox"]("/etc/passwd", str(root)) is True

    def test_dotdot_escape_blocked(self, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        # `../../etc/passwd` normalises above the mount.
        assert NS["_file_escapes_sandbox"](str(root / ".." / ".." / "etc" / "passwd"), str(root)) is True

    def test_sibling_prefix_not_confused(self, tmp_path):
        # `<root>-evil` shares a string prefix with the root but is a DIFFERENT
        # directory — commonpath is component-wise, so it must be blocked.
        root = tmp_path / "workspace"
        root.mkdir()
        assert NS["_file_escapes_sandbox"](str(tmp_path / "workspace-evil" / "x"), str(root)) is True

    def test_no_root_declared_passes(self, tmp_path):
        # No declared subtree → check not applicable, file:// passes (matches
        # the bare interceptor install path used by the older tests).
        assert NS["_file_escapes_sandbox"]("/etc/passwd", None) is False


class TestShouldBlockFileScheme:
    def test_escaping_file_url_blocked(self, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        assert NS["_ssrf_should_block"]("file:///etc/passwd", str(root), False) is True

    def test_inside_file_url_allowed(self, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        fixture = root / "index.html"
        fixture.write_text("<h1>ok</h1>")
        url = "file://" + str(fixture)
        assert NS["_ssrf_should_block"](url, str(root), False) is False


# ══════════════════════════════════════════════════════════════════════
# Gap 2 — DNS-rebind: re-resolve each request's host in non-Tor mode
# ══════════════════════════════════════════════════════════════════════

class TestDnsRebind:
    @pytest.mark.parametrize("internal_ip", [
        "127.0.0.1", "169.254.169.254", "10.0.0.5", "172.16.0.9", "192.168.1.1",
    ])
    def test_rebind_to_internal_blocked_non_tor(self, monkeypatch, internal_ip):
        # The name looks public to the host-string check but RESOLVES internal.
        monkeypatch.setattr(NS["socket"], "getaddrinfo",
                            lambda h, p, *a, **k: _addrinfo(internal_ip, p))
        assert NS["_ssrf_should_block"]("http://rebind.evil.example/", None, False) is True

    def test_rebind_to_public_allowed_non_tor(self, monkeypatch):
        monkeypatch.setattr(NS["socket"], "getaddrinfo",
                            lambda h, p, *a, **k: _addrinfo("93.184.216.34", p))
        assert NS["_ssrf_should_block"]("http://legit.example/", None, False) is False

    def test_rebind_skipped_in_tor_mode(self, monkeypatch):
        # Over Tor (anonymous=True) we must NOT resolve — no query leaks, and a
        # host that resolves internal is left to Tor (which can't route there).
        called = {"n": 0}

        def _boom(*a, **k):
            called["n"] += 1
            return _addrinfo("127.0.0.1")

        monkeypatch.setattr(NS["socket"], "getaddrinfo", _boom)
        assert NS["_ssrf_should_block"]("http://rebind.evil.example/", None, True) is False
        assert called["n"] == 0  # DNS never touched in Tor mode

    def test_resolution_failure_fails_open(self, monkeypatch):
        # A transient resolver error must not brick a legitimate fetch.
        def _raise(*a, **k):
            raise socket.gaierror("no network")

        monkeypatch.setattr(NS["socket"], "getaddrinfo", _raise)
        assert NS["_ssrf_should_block"]("http://legit.example/", None, False) is False

    def test_literal_internal_ip_still_blocked_without_dns(self, monkeypatch):
        # A literal internal IP is caught by the string check; DNS is never
        # needed (and must not be the only thing standing between us and it).
        def _boom(*a, **k):
            raise AssertionError("getaddrinfo must not be called for a literal IP")

        monkeypatch.setattr(NS["socket"], "getaddrinfo", _boom)
        assert NS["_ssrf_should_block"]("http://10.0.0.5/internal", None, False) is True


# ══════════════════════════════════════════════════════════════════════
# Route handler — the residual rules through the real interceptor
# ══════════════════════════════════════════════════════════════════════

class TestRouteHandler:
    async def _handler(self, sandbox_root=None, anonymous=False):
        captured = {}

        class FakeCtx:
            async def route(self, pattern, handler):
                captured["pattern"] = pattern
                captured["handler"] = handler

        await NS["_install_ssrf_guard"](FakeCtx(), sandbox_root=sandbox_root, anonymous=anonymous)
        assert captured["pattern"] == "**/*"
        return captured["handler"]

    async def test_aborts_file_escape(self, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        handler = await self._handler(sandbox_root=str(root))
        route = MagicMock()
        route.request.url = "file:///etc/passwd"
        route.abort = AsyncMock()
        route.continue_ = AsyncMock()
        await handler(route)
        route.abort.assert_awaited_once()
        route.continue_.assert_not_called()

    async def test_continues_file_inside_subtree(self, tmp_path):
        root = tmp_path / "workspace"
        root.mkdir()
        fixture = root / "index.html"
        fixture.write_text("<h1>ok</h1>")
        handler = await self._handler(sandbox_root=str(root))
        route = MagicMock()
        route.request.url = "file://" + str(fixture)
        route.abort = AsyncMock()
        route.continue_ = AsyncMock()
        await handler(route)
        route.continue_.assert_awaited_once()
        route.abort.assert_not_called()

    async def test_aborts_dns_rebind_subresource(self, monkeypatch):
        monkeypatch.setattr(NS["socket"], "getaddrinfo",
                            lambda h, p, *a, **k: _addrinfo("169.254.169.254", p))
        handler = await self._handler(anonymous=False)
        route = MagicMock()
        route.request.url = "http://rebind.evil.example/meta-data/"
        route.abort = AsyncMock()
        route.continue_ = AsyncMock()
        await handler(route)
        route.abort.assert_awaited_once()
        route.continue_.assert_not_called()
