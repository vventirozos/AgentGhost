"""Regression tests for the click/dblclick/fill stability guard and
the wait_for_selector ``state`` parameter.

Incident context (2026-04-26, webOS session)
--------------------------------------------
The agent built a Web OS with a fading lock-screen overlay. The
sequence:
   click(#unlock-btn)   -> JS handler fires display:none on the lock
   click(#start-btn)    -> Playwright reports the lock screen still
                           intercepts pointer events
…sent the agent into a 70-minute repair loop. The bare
``page.click`` auto-waits for the TARGET to be actionable but knows
nothing about an overlapping overlay. ``wait_for_selector`` was the
only escape hatch and didn't accept a ``state`` arg, so the LLM had
no way to wait for the overlay to GO AWAY.

Fixes
-----
1. click / dblclick / fill accept an optional ``wait_for_hidden``
   selector. When set, the action waits for THAT element to leave
   the visible state before issuing the click — surfaces the
   "overlay didn't disappear" failure with a clear message rather
   than the generic "element intercepts pointer events" Playwright
   default.
2. click / dblclick accept an optional ``force=True`` to skip
   Playwright's actionability check entirely.
3. wait_for_selector accepts ``state`` ∈ {visible, hidden, attached,
   detached}. Without ``state=hidden`` the LLM cannot express
   "wait for this thing to go away".
"""

import sys
import types

import pytest

from ghost_agent.tools import browser as browser_mod


# ---------------------------------------------------------------- harness
# Shared shape with test_browser_interact_dblclick.py /
# test_browser_interact_abort.py — exec the runner-string-as-source so
# the tests hit the exact code that ships into the sandbox.


class _FakePage:
    def __init__(self):
        self.url = "about:blank"
        self._title = ""
        self.default_timeout = 30000
        self.click_calls = []           # list of (selector, force) tuples
        self.dblclick_calls = []
        self.fill_calls = []
        self.wait_for_selector_calls = []  # (selector, state, timeout)
        self._wait_for_selector_should_raise: dict = {}  # (selector,state) -> exc

    def set_default_timeout(self, ms):
        self.default_timeout = ms

    async def title(self):
        return self._title

    async def goto(self, url, wait_until=None):
        self.url = url
        self._title = "fake"
        return None

    async def click(self, selector, force=False):
        self.click_calls.append((selector, force))

    async def dblclick(self, selector, force=False):
        self.dblclick_calls.append((selector, force))

    async def fill(self, selector, text):
        self.fill_calls.append((selector, text))

    async def wait_for_selector(self, selector, state="visible", timeout=None):
        self.wait_for_selector_calls.append((selector, state, timeout))
        exc = self._wait_for_selector_should_raise.get((selector, state))
        if exc is not None:
            raise exc

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


async def _run_interact(op, page):
    async def fake_with_context(profile_dir, proxy, timeout_ms, run):
        return await run(page)

    original = _RUNNER_NS["_with_context"]
    _RUNNER_NS["_with_context"] = fake_with_context
    try:
        return await _RUNNER_NS["op_interact"](op)
    finally:
        _RUNNER_NS["_with_context"] = original


# ---------------------------------------------------------------- click + wait_for_hidden


async def test_click_with_wait_for_hidden_invokes_wait_first():
    """The exact incident shape: click an unlock button, then click
    a start button — the start-button click MUST wait for the lock
    screen to be hidden first, surfacing a clean error if it
    doesn't disappear."""
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/p",
        "timeout_ms": 5000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {"action": "click", "selector": "#unlock-btn"},
            {
                "action": "click", "selector": "#start-btn",
                "wait_for_hidden": "#lock-screen",
            },
        ],
    }
    result = await _run_interact(op, page)

    assert [c[0] for c in page.click_calls] == ["#unlock-btn", "#start-btn"]
    # The wait_for_selector(#lock-screen, state=hidden) must be on
    # the call list and must have happened BEFORE the second click.
    sels = [c[0] for c in page.wait_for_selector_calls]
    states = [c[1] for c in page.wait_for_selector_calls]
    assert "#lock-screen" in sels
    idx = sels.index("#lock-screen")
    assert states[idx] == "hidden"

    actions = result["actions"]
    second_click = [
        a for a in actions
        if a.get("action") == "click" and a.get("selector") == "#start-btn"
    ]
    assert len(second_click) == 1
    assert second_click[0]["ok"] is True


async def test_click_wait_for_hidden_failure_surfaces_clear_error():
    """When the overlay doesn't disappear in time the click action
    must report the wait failure clearly — better than Playwright's
    generic "element intercepts pointer events"."""
    page = _FakePage()
    page._wait_for_selector_should_raise[("#lock-screen", "hidden")] = TimeoutError(
        "Timeout 5000ms exceeded."
    )
    op = {
        "profile_dir": "/tmp/p",
        "timeout_ms": 5000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {
                "action": "click", "selector": "#start-btn",
                "wait_for_hidden": "#lock-screen",
            },
        ],
    }
    result = await _run_interact(op, page)
    bad = [a for a in result["actions"] if a.get("action") == "click"]
    assert len(bad) == 1
    assert bad[0]["ok"] is False
    err = bad[0]["error"]
    assert "wait_for_hidden" in err
    assert "#lock-screen" in err
    # The actual click must NOT have fired (we didn't get past the wait)
    assert page.click_calls == []


async def test_click_force_skips_actionability_check():
    """force=True is forwarded to page.click."""
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/p",
        "timeout_ms": 5000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {"action": "click", "selector": "#x", "force": True},
        ],
    }
    await _run_interact(op, page)
    assert page.click_calls == [("#x", True)]


async def test_click_default_force_is_false():
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/p",
        "timeout_ms": 5000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {"action": "click", "selector": "#x"},
        ],
    }
    await _run_interact(op, page)
    assert page.click_calls == [("#x", False)]


# ---------------------------------------------------------------- dblclick parity


async def test_dblclick_supports_wait_for_hidden_and_force():
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/p",
        "timeout_ms": 5000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {
                "action": "dblclick", "selector": ".icon",
                "wait_for_hidden": "#splash",
                "force": True,
            },
        ],
    }
    await _run_interact(op, page)
    assert ("#splash", "hidden", 5000) in page.wait_for_selector_calls
    assert page.dblclick_calls == [(".icon", True)]


async def test_fill_supports_wait_for_hidden():
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/p",
        "timeout_ms": 5000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {
                "action": "fill", "selector": "#input", "text": "x",
                "wait_for_hidden": "#busy-indicator",
            },
        ],
    }
    await _run_interact(op, page)
    assert ("#busy-indicator", "hidden", 5000) in page.wait_for_selector_calls
    assert page.fill_calls == [("#input", "x")]


# ---------------------------------------------------------------- wait_for_selector state


async def test_wait_for_selector_default_state_is_visible():
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/p",
        "timeout_ms": 1000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {"action": "wait_for_selector", "selector": "#thing"},
        ],
    }
    await _run_interact(op, page)
    assert ("#thing", "visible", 1000) in page.wait_for_selector_calls


async def test_wait_for_selector_accepts_hidden_state():
    """The whole point of fix 3: an LLM can wait for an overlay to
    GO AWAY, not just to appear."""
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/p",
        "timeout_ms": 2000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {
                "action": "wait_for_selector",
                "selector": "#lock-screen",
                "state": "hidden",
            },
        ],
    }
    result = await _run_interact(op, page)
    assert ("#lock-screen", "hidden", 2000) in page.wait_for_selector_calls
    res = [a for a in result["actions"] if a.get("action") == "wait_for_selector"]
    assert res[0]["ok"] is True
    assert res[0]["state"] == "hidden"


@pytest.mark.parametrize("state", ["visible", "hidden", "attached", "detached"])
async def test_wait_for_selector_all_states_accepted(state):
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/p",
        "timeout_ms": 1000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {"action": "wait_for_selector", "selector": "#x", "state": state},
        ],
    }
    result = await _run_interact(op, page)
    res = [a for a in result["actions"] if a.get("action") == "wait_for_selector"]
    assert res[0]["ok"] is True


async def test_wait_for_selector_rejects_unknown_state():
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/p",
        "timeout_ms": 1000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {"action": "wait_for_selector", "selector": "#x", "state": "fluffy"},
        ],
    }
    result = await _run_interact(op, page)
    res = [a for a in result["actions"] if a.get("action") == "wait_for_selector"]
    assert res[0]["ok"] is False
    assert "fluffy" in res[0]["error"] or "invalid" in res[0]["error"].lower()


async def test_wait_for_selector_custom_timeout_propagates():
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/p",
        "timeout_ms": 30000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {
                "action": "wait_for_selector", "selector": "#x",
                "state": "hidden", "timeout_ms": 1234,
            },
        ],
    }
    await _run_interact(op, page)
    assert ("#x", "hidden", 1234) in page.wait_for_selector_calls


# ---------------------------------------------------------------- registry


async def test_registry_documents_wait_for_hidden_and_state():
    """The LLM only sees the registry description for actions; the
    new fields must be advertised there or they're invisible."""
    from ghost_agent.tools import registry as registry_mod

    browser_def = next(
        t for t in registry_mod.TOOL_DEFINITIONS
        if t.get("function", {}).get("name") == "browser"
    )
    actions_desc = browser_def["function"]["parameters"]["properties"]["actions"]["description"]
    assert "wait_for_hidden" in actions_desc, (
        "LLM cannot use wait_for_hidden without registry advertising it"
    )
    # state docs must list at least visible+hidden
    assert "state" in actions_desc and "hidden" in actions_desc
    assert "force" in actions_desc


async def test_runner_source_contains_state_arg():
    """Belt-and-braces: the runner source string actually contains
    the state= keyword wired into wait_for_selector. A refactor that
    drops the state forward would silently regress fix 3."""
    src = browser_mod._runner_script()
    assert "state=state" in src, (
        "wait_for_selector must forward state= to page.wait_for_selector"
    )
    assert "wait_for_hidden" in src
