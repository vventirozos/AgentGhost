"""Tests for the interact-sequence abort behaviour.

The original hang came from ``page.goto("file:///...-not-exist")``
raising ``ERR_FILE_NOT_FOUND``, the per-action `except` catching it,
and the loop marching on through 53 more click / fill / screenshot
actions. Each one hit a timeout because the page was Chromium's error
page, not the real app. 54 × 120 s ≈ 108 min of silent hang before
the subprocess watchdog finally killed it.

After the fix: a failed ``goto`` aborts the sequence IMMEDIATELY
regardless of ``stop_on_error``. These tests exercise that contract
without touching Playwright — we run ``op_interact`` with a fake
page object that drives the failure shape we care about.
"""

import pytest

from ghost_agent.tools import browser as browser_mod


class _FakePage:
    """Minimal Playwright-page shape: just enough for op_interact
    to drive goto/click/fill through it.

    `goto_behavior` is a dict: key = URL prefix match, value = either
    a raised exception class-name or None (success).
    """

    def __init__(self, goto_behavior=None):
        self.url = "about:blank"
        self._title = ""
        self.goto_behavior = goto_behavior or {}
        self.default_timeout = 30000
        self.click_calls = 0
        self.fill_calls = 0

    def set_default_timeout(self, ms):
        self.default_timeout = ms

    async def title(self):
        return self._title

    async def goto(self, url, wait_until=None):
        for prefix, behavior in self.goto_behavior.items():
            if url.startswith(prefix):
                if behavior is None:
                    self.url = url
                    self._title = "fake"
                    return None
                raise behavior(f"simulated failure at {url}")
        self.url = url
        self._title = "fake-default"
        return None

    async def click(self, selector):
        self.click_calls += 1

    async def fill(self, selector, text):
        self.fill_calls += 1

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
    """Exec the runner source (which lives as a string inside
    `browser_mod._runner_script`) into an isolated namespace so tests
    can call `op_interact` directly without spawning a Playwright
    subprocess. This is important because `op_interact` is defined
    INSIDE the runner string — it's the code that actually ships
    into the sandbox — so exec'ing it guarantees the tests are
    exercising the exact code path production runs.

    The runner does `from playwright.async_api import async_playwright`
    at top-level; the host venv doesn't ship Playwright (it's installed
    inside the sandbox container, not on the host). We stub a fake
    module so the import succeeds. The fake is never actually called —
    tests monkey-patch `_with_context` to bypass the Playwright launch
    entirely.
    """
    import sys, types
    if "playwright" not in sys.modules:
        fake_pw = types.ModuleType("playwright")
        fake_pw_async = types.ModuleType("playwright.async_api")

        class _StubAsyncPlaywright:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        def async_playwright():  # noqa: N802 — mimic real API
            return _StubAsyncPlaywright()

        fake_pw_async.async_playwright = async_playwright
        sys.modules["playwright"] = fake_pw
        sys.modules["playwright.async_api"] = fake_pw_async

    src = browser_mod._runner_script()
    # Only exec the top half — async def main + the `if __name__`
    # tail would try to read argv at import time.
    cutoff = src.find("async def main():")
    assert cutoff != -1, "runner template should have a main() entry point"
    ns: dict = {}
    exec(compile(src[:cutoff], "<runner-snippet>", "exec"), ns)
    return ns


_RUNNER_NS = _load_runner_namespace()


async def _run_interact(op, page):
    """Invoke op_interact with a fake page. We monkey-patch
    `_with_context` in the runner's namespace so the call never
    touches Playwright — it just runs the inner `run` function
    directly on our fake page."""
    async def fake_with_context(profile_dir, proxy, timeout_ms, run):
        return await run(page)

    original = _RUNNER_NS["_with_context"]
    _RUNNER_NS["_with_context"] = fake_with_context
    try:
        return await _RUNNER_NS["op_interact"](op)
    finally:
        _RUNNER_NS["_with_context"] = original


# ------------------------------------------------------------------
# Initial goto failure aborts sequence
# ------------------------------------------------------------------

async def test_initial_goto_failure_aborts_whole_sequence():
    """The canonical hang scenario: goto → ERR_FILE_NOT_FOUND, then
    54 more actions each timeout. With the fix, the sequence aborts
    after the failed goto; click/fill/screenshot are NOT attempted."""

    class FileNotFound(Exception):
        pass

    page = _FakePage(goto_behavior={
        "file:///workspace/webos/index.html": FileNotFound,
    })

    op = {
        "profile_dir": "/tmp/profile",
        "timeout_ms": 10000,
        "actions": [
            {"action": "goto", "url": "file:///workspace/webos/index.html", "wait_until": "load"},
            {"action": "click", "selector": "#start-btn"},
            {"action": "click", "selector": ".calc-btn-7"},
            {"action": "screenshot", "out_path": "/workspace/s.png"},
            {"action": "fill", "selector": "#term", "text": "x"},
        ],
    }

    result = await _run_interact(op, page)

    # Only the goto attempt was recorded; no subsequent click/fill/screenshot
    assert page.click_calls == 0, "click must NOT be called after a failed goto"
    assert page.fill_calls == 0, "fill must NOT be called after a failed goto"
    actions = result["actions"]
    assert len(actions) == 1
    assert actions[0]["action"] == "goto"
    assert actions[0]["ok"] is False
    assert "FileNotFound" in actions[0]["error"] or "simulated failure" in actions[0]["error"]
    assert actions[0].get("aborted_sequence") is True
    assert result["aborted"] is True
    assert result["abort_reason"] == "goto_failed"


async def test_initial_goto_failure_aborts_even_without_stop_on_error():
    """The abort is UNCONDITIONAL for goto — stop_on_error=False is
    specifically the setting that used to cause the hang."""
    class TimeoutErr(Exception):
        pass

    page = _FakePage(goto_behavior={"http://": TimeoutErr})
    op = {
        "profile_dir": "/tmp/profile",
        "timeout_ms": 10000,
        "stop_on_error": False,   # THE bug-triggering setting
        "actions": [
            {"action": "goto", "url": "http://example.invalid"},
            {"action": "click", "selector": "#anything"},
            {"action": "click", "selector": "#anywhere"},
        ],
    }

    result = await _run_interact(op, page)
    assert page.click_calls == 0
    assert result["aborted"] is True


async def test_implicit_initial_navigation_failure_aborts():
    """When the first action is NOT a goto, op_interact still
    navigates to `op["url"]` first. That navigation failure must
    abort the same way an explicit goto failure does."""
    class FileNotFound(Exception):
        pass

    page = _FakePage(goto_behavior={"file://": FileNotFound})
    op = {
        "profile_dir": "/tmp/profile",
        "timeout_ms": 10000,
        "url": "file:///does/not/exist.html",
        "actions": [
            {"action": "click", "selector": "#btn"},
            {"action": "screenshot", "out_path": "/workspace/s.png"},
        ],
    }

    result = await _run_interact(op, page)
    assert page.click_calls == 0
    assert result["aborted"] is True
    assert result["abort_reason"] == "initial_goto_failed"
    assert len(result["actions"]) == 1
    assert result["actions"][0]["ok"] is False


# ------------------------------------------------------------------
# Non-goto failures still honour the per-action stop_on_error contract
# ------------------------------------------------------------------

async def test_click_failure_does_not_abort_by_default():
    """A click that can't find its selector is NOT a sequence-wide
    failure. With the default stop_on_error=False, the loop continues
    so the caller can still get the screenshots and other diagnostics
    it asked for."""
    page = _FakePage()

    async def broken_click(sel):
        raise ValueError(f"selector {sel!r} not found")
    page.click = broken_click

    op = {
        "profile_dir": "/tmp/profile",
        "timeout_ms": 1000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {"action": "click", "selector": "#missing"},
            {"action": "sleep", "ms": 0},
            {"action": "sleep", "ms": 0},
        ],
    }
    result = await _run_interact(op, page)
    # Got through all 4 actions
    assert len(result["actions"]) == 4
    assert result["aborted"] is False
    # Click failed but subsequent sleeps succeeded
    assert result["actions"][1]["ok"] is False
    assert result["actions"][2]["ok"] is True
    assert result["actions"][3]["ok"] is True


async def test_stop_on_error_true_breaks_on_non_goto_failure():
    """With stop_on_error=True, any failure stops the loop (same
    behaviour as before the goto-abort fix)."""
    page = _FakePage()

    async def broken_click(sel):
        raise ValueError("never matches")
    page.click = broken_click

    op = {
        "profile_dir": "/tmp/profile",
        "timeout_ms": 1000,
        "stop_on_error": True,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {"action": "click", "selector": "#x"},
            {"action": "click", "selector": "#y"},
        ],
    }
    result = await _run_interact(op, page)
    assert len(result["actions"]) == 2
    assert result["actions"][0]["ok"] is True
    assert result["actions"][1]["ok"] is False
    assert page.click_calls == 0  # broken_click raises before counter increments


# ------------------------------------------------------------------
# Mid-sequence goto failures
# ------------------------------------------------------------------

async def test_mid_sequence_goto_failure_stops_loop():
    """A goto that comes mid-sequence also aborts: every action after
    it would be on the error page. Actions BEFORE it are still valid
    and reported."""
    class NavFailure(Exception):
        pass

    # First goto ok; second goto fails.
    state = {"goto_idx": 0}
    async def selective_goto(url, wait_until=None):
        state["goto_idx"] += 1
        if state["goto_idx"] >= 2:
            raise NavFailure(f"second goto to {url} failed")
        page.url = url
        page._title = "first"
        return None

    page = _FakePage()
    page.goto = selective_goto

    op = {
        "profile_dir": "/tmp/profile",
        "timeout_ms": 1000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {"action": "click", "selector": "#a"},    # still on first page
            {"action": "goto", "url": "file:///bad"}, # fails → abort
            {"action": "click", "selector": "#b"},    # skipped
            {"action": "click", "selector": "#c"},    # skipped
        ],
    }
    result = await _run_interact(op, page)
    assert len(result["actions"]) == 3
    assert result["actions"][0]["ok"] is True
    assert result["actions"][1]["ok"] is True
    assert result["actions"][2]["ok"] is False
    assert result["actions"][2].get("aborted_sequence") is True
    assert result["aborted"] is True


async def test_screenshot_after_failed_nav_does_not_execute():
    """Specific regression: the user's log showed a failed goto
    followed by a screenshot. Without the fix, the screenshot would
    execute on the error page (ok) and be misleadingly listed in
    results. With the fix, the sequence aborts before the screenshot."""
    class NavFailure(Exception):
        pass

    screenshot_calls = {"n": 0}
    async def record_screenshot(path=None, full_page=False):
        screenshot_calls["n"] += 1

    page = _FakePage(goto_behavior={"file://": NavFailure})
    page.screenshot = record_screenshot
    op = {
        "profile_dir": "/tmp/profile",
        "timeout_ms": 1000,
        "actions": [
            {"action": "goto", "url": "file:///bad"},
            {"action": "screenshot", "out_path": "/workspace/a.png"},
            {"action": "screenshot", "out_path": "/workspace/b.png"},
        ],
    }
    await _run_interact(op, page)
    assert screenshot_calls["n"] == 0


# ------------------------------------------------------------------
# Backwards compat: empty actions still errors, shape preserved
# ------------------------------------------------------------------

async def test_empty_actions_list_still_raises_value_error():
    page = _FakePage()
    with pytest.raises(ValueError, match="non-empty"):
        await _run_interact({"profile_dir": "/tmp/p", "timeout_ms": 1000,
                             "actions": []}, page)


async def test_happy_path_still_runs_all_actions():
    """No regressions: when every action succeeds, aborted is False
    and the actions list is complete."""
    page = _FakePage()
    op = {
        "profile_dir": "/tmp/profile",
        "timeout_ms": 1000,
        "actions": [
            {"action": "goto", "url": "about:blank"},
            {"action": "sleep", "ms": 0},
            {"action": "sleep", "ms": 0},
        ],
    }
    result = await _run_interact(op, page)
    assert len(result["actions"]) == 3
    assert all(r["ok"] for r in result["actions"])
    assert result["aborted"] is False
    assert result["abort_reason"] is None
