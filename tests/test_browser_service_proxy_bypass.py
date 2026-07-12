"""Regression: the browser must BYPASS the Tor proxy for supervised loopback
services (found live 2026-07-12).

`--proxy-server=socks5://…` routes EVERY http(s) request through SOCKS,
including `http://127.0.0.1:8100`. Tor cannot route loopback, so navigating to
a service the agent itself started died with net::ERR_SOCKS_CONNECTION_FAILED
— breaking the whole "host an app, then drive it with the browser" capability
(Feature 4) under --mandatory-tor. Observed live: the chess-coach service came
up on :8100 and every navigate failed.

The `--host-resolver-rules … EXCLUDE localhost` flag did NOT cover this — it
governs DNS resolution only, never proxy ROUTING. The fix is Playwright's
launch-time `proxy.bypass`, scoped to the exact allowed service ports.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import types
import unittest.mock as um
from pathlib import Path

import pytest

from ghost_agent.tools.browser import _runner_script


def _runner_ns():
    """Exec the in-sandbox runner (playwright stubbed) and return its module
    namespace — `_proxy_bypass_for_ports` lives INSIDE the runner string
    (that is where it runs), so it is tested there, not by import."""
    fake_api = types.ModuleType("playwright.async_api")
    fake_api.async_playwright = None
    fake_pw = types.ModuleType("playwright")
    fake_pw.async_api = fake_api
    ns = {}
    with um.patch.dict(sys.modules, {"playwright": fake_pw,
                                     "playwright.async_api": fake_api}):
        src = _runner_script().replace(
            'if __name__ == "__main__":', 'if False:')
        exec(compile(src, "browser_runner", "exec"), ns)
    return ns


class TestProxyBypassList:
    def setup_method(self):
        self.bypass = _runner_ns()["_proxy_bypass_for_ports"]

    def test_empty_ports_yields_empty(self):
        # No service running ⇒ no bypass; loopback stays blocked by the proxy
        # AND the SSRF interceptor.
        assert self.bypass(set()) == ""
        assert self.bypass(None) == ""

    def test_any_service_bypasses_loopback(self):
        # EMPIRICALLY VERIFIED: host:port bypass entries are IGNORED by
        # Chromium; only `<loopback>` (or a bare host) actually reaches a
        # local server through a SOCKS proxy. Port-level access is enforced
        # by the SSRF interceptor, not the proxy bypass.
        assert self.bypass({8100}) == "<loopback>"
        assert self.bypass({5055, 5056}) == "<loopback>"


class TestRunnerAppliesBypass:
    """Exec the ACTUAL in-sandbox runner string (playwright stubbed) and
    confirm the launch config it builds carries the bypass."""

    def _run_with_context(self, proxy, allowed_ports):
        captured = {}

        # ---- fake playwright that records launch_persistent_context kwargs --
        class _FakeCtx:
            pages = []

            async def route(self, *a, **k):
                pass

            async def new_page(self):
                class _P:
                    def set_default_timeout(self, *a): pass
                    def on(self, *a, **k): pass
                    async def wait_for_timeout(self, *a): pass
                    async def close(self): pass
                return _P()

            async def close(self):
                pass

        class _Chromium:
            async def launch_persistent_context(self, **kw):
                captured.update(kw)
                return _FakeCtx()

        class _PW:
            chromium = _Chromium()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        fake_api = types.ModuleType("playwright.async_api")
        fake_api.async_playwright = lambda: _PW()
        fake_pw = types.ModuleType("playwright")
        fake_pw.async_api = fake_api

        ns = {}
        with um.patch.dict(sys.modules, {"playwright": fake_pw,
                                         "playwright.async_api": fake_api}):
            src = _runner_script().replace(
                'if __name__ == "__main__":', 'if False:')
            exec(compile(src, "browser_runner", "exec"), ns)
            # ALLOWED_LOCAL_PORTS is module-level in the runner namespace;
            # reset per call so a prior test can't leak ports into this one.
            ns["ALLOWED_LOCAL_PORTS"].clear()
            ns["ALLOWED_LOCAL_PORTS"].update(allowed_ports)
            import asyncio

            async def _op(page):
                return {"ok": True}

            # asyncio.run (fresh loop) — get_event_loop() picks up a closed
            # loop left by an earlier test in the same process.
            asyncio.run(ns["_with_context"]("/tmp/prof", proxy, 1000, _op))
        return captured

    def test_bypass_set_for_allowed_service_port(self):
        cap = self._run_with_context("socks5://127.0.0.1:9050", {8100})
        assert cap["proxy"]["server"] == "socks5://127.0.0.1:9050"
        assert cap["proxy"]["bypass"] == "<loopback>"

    def test_no_bypass_when_no_services(self):
        cap = self._run_with_context("socks5://127.0.0.1:9050", set())
        # proxy is still set (Tor), but no bypass key when nothing to bypass.
        assert cap["proxy"]["server"] == "socks5://127.0.0.1:9050"
        assert "bypass" not in cap["proxy"]

    def test_no_proxy_no_proxy_config(self):
        cap = self._run_with_context(None, {8100})
        assert "proxy" not in cap           # file:// / no-Tor mode unaffected


class TestSourceIntent:
    SRC = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
           / "tools" / "browser.py").read_text()

    def test_bypass_wired_at_launch(self):
        assert '_proxy_bypass_for_ports(ALLOWED_LOCAL_PORTS)' in self.SRC
        assert 'launch_kwargs["proxy"]["bypass"]' in self.SRC

    def test_stale_comment_corrected(self):
        # The old comment claimed EXCLUDE localhost made services "reachable
        # without routing through Tor" — it doesn't (DNS only). Make sure the
        # corrected note about proxy.bypass is present.
        assert "governs DNS RESOLUTION only" in self.SRC

    def test_port_enforcement_still_lives_in_the_ssrf_interceptor(self):
        # `<loopback>` bypasses ALL loopback at the PROXY layer; the guarantee
        # that only ALLOWED ports are reachable must therefore come from the
        # SSRF interceptor, which runs on every request regardless of proxy.
        ns = _runner_ns()
        ns["ALLOWED_LOCAL_PORTS"].clear()
        ns["ALLOWED_LOCAL_PORTS"].add(8100)
        block = ns["_ssrf_should_block"]
        assert block("http://127.0.0.1:8100/") is False   # allowed service
        assert block("http://127.0.0.1:9051/") is True    # Tor control — blocked
        assert block("http://127.0.0.1:8000/") is True    # agent API — blocked
