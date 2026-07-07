"""navigate/click return a page-text preview (IMPROVEMENTS.md #9).

Every browser op opens a fresh Chromium context and reloads the page, so a
bare `navigate` that returned only status/title forced a second `extract_text`
op that re-launched Playwright and re-fetched the SAME page over Tor (5-20s).
navigate and click now include a capped ~8 KB innerText preview, collapsing the
dominant navigate→extract_text→read flow into one op.

The browser ops live inside the in-sandbox runner string (`_runner_script()`),
so this test exec's that source with a stubbed playwright and drives the op
functions against a fake page.
"""
import sys
import types

import pytest

from ghost_agent.tools.browser import _runner_script


@pytest.fixture(scope="module")
def runner_ns():
    # Stub playwright so the runner source imports on the host.
    fake_pw = types.ModuleType("playwright")
    fake_async = types.ModuleType("playwright.async_api")
    fake_async.async_playwright = lambda: None
    fake_pw.async_api = fake_async
    sys.modules.setdefault("playwright", fake_pw)
    sys.modules.setdefault("playwright.async_api", fake_async)

    ns = {}
    exec(compile(_runner_script(), "<browser_runner>", "exec"), ns)
    return ns


class _FakePage:
    def __init__(self, body_text, url="https://example.com/", title="Example"):
        self._body = body_text
        self.url = url
        self._title = title

    async def goto(self, url, wait_until=None):
        self.url = url
        return type("Resp", (), {"status": 200})()

    async def title(self):
        return self._title

    async def evaluate(self, script):
        return self._body

    async def click(self, selector):
        return None

    async def wait_for_load_state(self, *a, **k):
        return None


def _install_fakes(ns, page):
    """Patch the runner's _with_context / sidecar / probe so ops run against
    the fake page with no real Chromium."""
    async def fake_with_context(profile_dir, proxy, timeout_ms, op_fn):
        return await op_fn(page)

    async def fake_probe(page):
        return {"pre_interaction": None}

    ns["_with_context"] = fake_with_context
    ns["_write_last_url"] = lambda *a, **k: None
    ns["_probe_pre_interaction"] = fake_probe
    ns["_resolve_url_or_error"] = lambda op, label: (op.get("url"), False)


async def test_body_excerpt_caps_and_reports_length(runner_ns):
    page = _FakePage("A" * 20000)
    text, truncated, full = await runner_ns["_body_excerpt"](page, 8000)
    assert len(text) == 8000 and truncated is True and full == 20000


async def test_body_excerpt_no_truncation_when_short(runner_ns):
    page = _FakePage("short body")
    text, truncated, full = await runner_ns["_body_excerpt"](page, 8000)
    assert text == "short body" and truncated is False and full == 10


async def test_navigate_includes_text_preview(runner_ns):
    page = _FakePage("Hello from the page body")
    _install_fakes(runner_ns, page)
    op = {"url": "https://example.com", "wait_until": "load",
          "profile_dir": "/tmp/p", "timeout_ms": 1000}
    result = await runner_ns["op_navigate"](op)
    assert result["status"] == 200
    assert result["title"] == "Example"
    assert result["text"] == "Hello from the page body"
    assert result["length"] == len("Hello from the page body")
    assert result["truncated"] is False


async def test_navigate_text_can_be_disabled(runner_ns):
    page = _FakePage("body text")
    _install_fakes(runner_ns, page)
    op = {"url": "https://example.com", "wait_until": "load",
          "profile_dir": "/tmp/p", "timeout_ms": 1000, "nav_text_chars": 0}
    result = await runner_ns["op_navigate"](op)
    assert "text" not in result


async def test_click_includes_post_click_text(runner_ns):
    page = _FakePage("state after click")
    _install_fakes(runner_ns, page)
    op = {"url": "https://example.com", "selector": "#go", "wait_until": "load",
          "profile_dir": "/tmp/p", "timeout_ms": 1000}
    result = await runner_ns["op_click"](op)
    assert result["text"] == "state after click"
    assert result["title"] == "Example"


def test_runner_source_documents_navigate_text():
    """Guard: the runner must actually carry the preview logic."""
    src = _runner_script()
    assert "_body_excerpt" in src
    assert "nav_text_chars" in src
