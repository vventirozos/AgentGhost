"""Atomic click fail-fast probe (2026-07-14).

Incident: two repair turns on the WebOS project each died at the same wall —
`browser click` on `.wp-option` timed out after the FULL 30s and the
no-progress loop breaker then force-ended the turn before the agent could
verify or fix the user's bug. Root cause: every atomic op runs in a fresh
context and re-navigates, so `.wp-option` (which only exists after clicking
the Wallpapers icon) can never appear; `page.click` silently waited out the
whole timeout on an impossible selector, and the opaque TimeoutError taught
the model nothing — it retried variants of the same doomed call.

Fix: `op_click` probes the selector with a short bounded `wait_for_selector`
(state='attached', ≤8s) right after goto. When absent it fails fast with an
error that names the escape: `operation='interact'` runs the whole
multi-step flow in ONE context. A present-but-animating element still
proceeds to `page.click`'s own actionability wait (probe is 'attached', not
'visible').

Harness: exec the runner-string-as-source (same shape as
test_browser_interact_stability_guard.py) so the tests hit the exact code
that ships into the sandbox.
"""

import sys
import types

import pytest

from ghost_agent.tools import browser as browser_mod


class _FakePage:
    def __init__(self, missing_selectors=()):
        self.url = "about:blank"
        self._title = "fake"
        self.missing = set(missing_selectors)
        self.click_calls = []
        self.wait_for_selector_calls = []  # (selector, state, timeout)

    def set_default_timeout(self, ms):
        pass

    async def title(self):
        return self._title

    async def goto(self, url, wait_until=None):
        self.url = url
        return None

    async def click(self, selector, force=False, timeout=None):
        self.click_calls.append(selector)

    async def wait_for_selector(self, selector, state="visible", timeout=None):
        self.wait_for_selector_calls.append((selector, state, timeout))
        if selector in self.missing:
            raise TimeoutError(f"waiting for selector {selector!r}")

    async def wait_for_load_state(self, state="load", timeout=None):
        pass

    async def evaluate(self, fn):
        return ""

    async def query_selector(self, sel):
        return None


def _load_runner_namespace():
    if "playwright" not in sys.modules:
        fake_pw = types.ModuleType("playwright")
        fake_pw_async = types.ModuleType("playwright.async_api")

        class _StubAsyncPlaywright:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        def async_playwright():
            return _StubAsyncPlaywright()

        fake_pw_async.async_playwright = async_playwright
        sys.modules["playwright"] = fake_pw
        sys.modules["playwright.async_api"] = fake_pw_async

    src = browser_mod._runner_script()
    cutoff = src.find("async def main():")
    assert cutoff != -1
    ns: dict = {}
    exec(compile(src[:cutoff], "<runner-snippet>", "exec"), ns)
    return ns


_RUNNER_NS = _load_runner_namespace()


def _click_op(tmp_path, selector, timeout_ms=30000):
    return {
        "op": "click",
        "profile_dir": str(tmp_path / "profile"),
        "timeout_ms": timeout_ms,
        "url": "http://127.0.0.1:8100/",
        "selector": selector,
        "nav_text_chars": 0,   # skip the body-excerpt path in the fake
    }


async def _run_click(op, page):
    async def fake_with_context(profile_dir, proxy, timeout_ms, run):
        return await run(page)

    original = _RUNNER_NS["_with_context"]
    _RUNNER_NS["_with_context"] = fake_with_context
    try:
        return await _RUNNER_NS["op_click"](op)
    finally:
        _RUNNER_NS["_with_context"] = original


@pytest.mark.asyncio
async def test_missing_selector_fails_fast_with_interact_steer(tmp_path):
    page = _FakePage(missing_selectors={".wp-option:nth-child(1)"})
    op = _click_op(tmp_path, ".wp-option:nth-child(1)")
    with pytest.raises(RuntimeError) as ei:
        await _run_click(op, page)
    msg = str(ei.value)
    assert "not found" in msg
    assert "interact" in msg           # names the escape
    assert "freshly-loaded" in msg     # explains the state reset
    assert page.click_calls == []      # never attempted the doomed click
    # Probe was bounded (≤8s), not the full 30s budget.
    (sel, state, timeout) = page.wait_for_selector_calls[0]
    assert sel == ".wp-option:nth-child(1)"
    assert state == "attached"
    assert timeout <= 8000


@pytest.mark.asyncio
async def test_probe_respects_smaller_op_timeout(tmp_path):
    page = _FakePage(missing_selectors={"#gone"})
    op = _click_op(tmp_path, "#gone", timeout_ms=5000)
    with pytest.raises(RuntimeError):
        await _run_click(op, page)
    (_sel, _state, timeout) = page.wait_for_selector_calls[0]
    assert timeout == 5000


@pytest.mark.asyncio
async def test_present_selector_clicks_as_before(tmp_path):
    page = _FakePage()
    op = _click_op(tmp_path, "#start-btn")
    result = await _run_click(op, page)
    assert page.click_calls == ["#start-btn"]
    assert result["url"] == "http://127.0.0.1:8100/"


def test_shipped_runner_contains_probe_and_hint():
    src = browser_mod._runner_script()
    assert "FAIL-FAST SELECTOR PROBE" in src
    assert "state=\"attached\"" in src or "state='attached'" in src
