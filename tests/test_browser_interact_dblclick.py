"""Regression test for the `dblclick` action in `browser.interact`.

Incident context (2026-04-24, webOS session)
--------------------------------------------
The agent's own webOS binds `icon.addEventListener('dblclick', openApp)`
— a plain `click` does nothing. During a "create 10 apps and screenshot
each" session the LLM fired 8 parallel `interact` calls, each emitting
a single-`click` followed by a screenshot. All 8 screenshots showed
the same picture (the desktop before any app opened). Root cause: the
`interact` sub-action list supported `click`, `fill`, `goto`, ..., but
NOT `dblclick`. The LLM had no way to fire a proper double-click
gesture that desktop-icon-style UIs require.

Fix
---
`dblclick` is now a first-class sub-action in `op_interact`. It calls
`page.dblclick(selector)` — Playwright's proper mousedown-mouseup-
mousedown-mouseup sequence — so native `ondblclick` listeners fire.
The registry tool description advertises the action and gives a
desktop-icon example so the LLM picks it up without extra prompting.

These tests exercise the runner (shared exec'd namespace, same pattern
as test_browser_interact_abort.py) so the tests hit the exact code
that ships into the sandbox.
"""

import sys
import types

import pytest

from ghost_agent.tools import browser as browser_mod


# ------------------------------------------------------------------
# Shared harness (mirrors test_browser_interact_abort.py)
# ------------------------------------------------------------------


class _FakePage:
    def __init__(self):
        self.url = "about:blank"
        self._title = ""
        self.default_timeout = 30000
        self.click_calls = 0
        self.dblclick_calls = 0
        self.dblclick_selectors = []

    def set_default_timeout(self, ms):
        self.default_timeout = ms

    async def title(self):
        return self._title

    async def goto(self, url, wait_until=None):
        self.url = url
        self._title = "fake"
        return None

    async def click(self, selector):
        self.click_calls += 1

    async def dblclick(self, selector):
        self.dblclick_calls += 1
        self.dblclick_selectors.append(selector)

    async def fill(self, selector, text):
        pass

    async def wait_for_selector(self, selector, timeout=None):
        pass

    async def wait_for_timeout(self, ms):
        pass

    async def screenshot(self, path=None, full_page=False):
        pass

    async def query_selector(self, sel):
        return None

    async def evaluate(self, fn):
        return ""


def _load_runner_namespace():
    if "playwright" not in sys.modules:
        fake_pw = types.ModuleType("playwright")
        fake_pw_async = types.ModuleType("playwright.async_api")

        class _StubAsyncPlaywright:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        def async_playwright():  # noqa: N802
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


async def _run_interact(op, page):
    async def fake_with_context(profile_dir, proxy, timeout_ms, run):
        return await run(page)

    original = _RUNNER_NS["_with_context"]
    _RUNNER_NS["_with_context"] = fake_with_context
    try:
        return await _RUNNER_NS["op_interact"](op)
    finally:
        _RUNNER_NS["_with_context"] = original


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


async def test_dblclick_action_invokes_page_dblclick():
    """A dblclick sub-action must call `page.dblclick(selector)` —
    NOT `page.click`. Desktop-icon handlers bound to `ondblclick`
    do not fire on a plain click."""
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/profile",
        "timeout_ms": 1000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {"action": "dblclick", "selector": ".desktop-icon[data-app='calc']"},
        ],
    }
    result = await _run_interact(op, page)

    assert page.dblclick_calls == 1
    assert page.click_calls == 0, "must not fall back to plain click"
    assert page.dblclick_selectors == [".desktop-icon[data-app='calc']"]

    actions = result["actions"]
    # Find the dblclick result
    dbl = [a for a in actions if a.get("action") == "dblclick"]
    assert len(dbl) == 1
    assert dbl[0]["ok"] is True
    assert dbl[0]["selector"] == ".desktop-icon[data-app='calc']"


async def test_dblclick_missing_selector_raises():
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/profile",
        "timeout_ms": 1000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {"action": "dblclick"},  # no selector
        ],
    }
    result = await _run_interact(op, page)
    dbl = [a for a in result["actions"] if a.get("action") == "dblclick"]
    assert len(dbl) == 1
    assert dbl[0]["ok"] is False
    assert "selector" in dbl[0]["error"].lower()
    assert page.dblclick_calls == 0


async def test_dblclick_listed_in_unknown_action_error():
    """If someone types a typo like `doubleclick`, the error message
    must surface the real action name so the LLM can self-correct."""
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/profile",
        "timeout_ms": 1000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {"action": "doubleclick", "selector": "#x"},  # invalid
        ],
    }
    result = await _run_interact(op, page)
    bad = [a for a in result["actions"] if a.get("action") == "doubleclick"]
    assert len(bad) == 1
    assert bad[0]["ok"] is False
    assert "dblclick" in bad[0]["error"], (
        "unknown-action error must mention `dblclick` so LLMs type the "
        "canonical name; otherwise self-correction loops typo-thrash"
    )


async def test_dblclick_exposed_in_registry_description():
    """The tool registry description is what the LLM sees when it
    picks actions. It must mention `dblclick` AND signal when to use
    it — otherwise the LLM defaults to `click` on icon UIs and
    produces the all-identical-screenshots bug.
    """
    from ghost_agent.tools import registry as registry_mod

    browser_def = next(
        t for t in registry_mod.TOOL_DEFINITIONS
        if t.get("function", {}).get("name") == "browser"
    )
    actions_desc = browser_def["function"]["parameters"]["properties"]["actions"]["description"]
    assert "dblclick" in actions_desc
    # Must explicitly hint at ondblclick-bound UIs; otherwise the LLM
    # still defaults to `click` for icons.
    assert "ondblclick" in actions_desc or "double" in actions_desc.lower()


async def test_dblclick_and_click_are_independent_in_source():
    """Belt-and-braces: grep the browser.py runner string for both
    branches. A refactor that accidentally routes `dblclick` through
    `page.click` would silently regress the incident. This guards the
    structural invariant: `page.dblclick` MUST appear somewhere in the
    runner, and the `dblclick` action branch must live near it.
    """
    src = browser_mod._runner_script()
    assert "page.dblclick(" in src, (
        "runner must call page.dblclick directly; delegating to "
        "page.click would not fire ondblclick listeners"
    )
    assert 'name == "dblclick"' in src
