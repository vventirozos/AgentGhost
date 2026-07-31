"""The `evaluate` action in `browser.interact` — exact-state probes.

Incident context (2026-07-31, pinball session, req 66d64313)
------------------------------------------------------------
The agent spent 550s and five ~30s vision round-trips trying to answer one
question about its own canvas game: "did the ball exit the launcher
channel?" Ground truth was a JS expression away (`ball.x, ball.y`), but the
interact sub-action vocabulary had no way to run one — `page.evaluate` was
used internally by extract_text yet never exposed. The agent fell back to
screenshot + describe_picture captions, misjudged one frame, and oscillated
through five failed geometry fixes on low-bandwidth feedback.

`evaluate` closes that gap: run a JS expression (or arrow function) in the
page, get its JSON-serialised value back, capped like extract_text so one
call can't flood the model context. Same trust domain as click/fill (the
page is agent-authored or already navigable); JS-initiated network calls
still pass the context's SSRF route guard installed by _with_context.

Tests exercise the runner via the shared exec'd-namespace pattern
(test_browser_interact_dblclick.py) so they hit the exact code that ships
into the sandbox.
"""

import sys
import types

from ghost_agent.tools import browser as browser_mod


# ------------------------------------------------------------------
# Shared harness (mirrors test_browser_interact_dblclick.py)
# ------------------------------------------------------------------


class _FakePage:
    def __init__(self, eval_result=None):
        self.url = "about:blank"
        self._title = ""
        self.default_timeout = 30000
        self.eval_calls = []
        self._eval_result = eval_result

    def set_default_timeout(self, ms):
        self.default_timeout = ms

    async def title(self):
        return self._title

    async def goto(self, url, wait_until=None):
        self.url = url
        self._title = "fake"
        return None

    async def click(self, selector):
        pass

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

    async def evaluate(self, js):
        self.eval_calls.append(js)
        return self._eval_result


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


def _op(actions):
    return {"profile_dir": "/tmp/profile", "timeout_ms": 1000,
            "actions": actions}


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


async def test_evaluate_returns_json_value():
    page = _FakePage(eval_result={"x": 370, "y": 150})
    result = await _run_interact(_op([
        {"action": "goto", "url": "about:blank"},
        {"action": "evaluate", "js": "({x: ball.x, y: ball.y})"},
    ]), page)

    assert page.eval_calls == ["({x: ball.x, y: ball.y})"]
    ev = [a for a in result["actions"] if a.get("action") == "evaluate"]
    assert len(ev) == 1
    assert ev[0]["ok"] is True
    assert ev[0]["value"] == '{"x": 370, "y": 150}'
    assert ev[0]["truncated"] is False


async def test_evaluate_missing_js_fails_with_clear_error():
    page = _FakePage()
    result = await _run_interact(_op([
        {"action": "goto", "url": "about:blank"},
        {"action": "evaluate"},  # no js
    ]), page)
    ev = [a for a in result["actions"] if a.get("action") == "evaluate"]
    assert ev[0]["ok"] is False
    assert "js" in ev[0]["error"]
    assert page.eval_calls == []


async def test_evaluate_accepts_expression_and_script_aliases():
    """LLMs reach for `expression`/`script` — heal them like the outer
    tool heals its own parameter names."""
    page = _FakePage(eval_result=1)
    result = await _run_interact(_op([
        {"action": "goto", "url": "about:blank"},
        {"action": "evaluate", "expression": "1"},
        {"action": "evaluate", "script": "1"},
    ]), page)
    ev = [a for a in result["actions"] if a.get("action") == "evaluate"]
    assert [e["ok"] for e in ev] == [True, True]
    assert page.eval_calls == ["1", "1"]


async def test_evaluate_output_is_capped():
    """One evaluate must not flood the model context — same policy as
    extract_text. The cap must report the TRUE length, not the capped one."""
    big = "a" * 50_000
    page = _FakePage(eval_result=big)
    result = await _run_interact(_op([
        {"action": "goto", "url": "about:blank"},
        {"action": "evaluate", "js": "bigString", "max_chars": 1000},
    ]), page)
    ev = [a for a in result["actions"] if a.get("action") == "evaluate"][0]
    assert ev["ok"] is True
    assert len(ev["value"]) == 1000
    assert ev["truncated"] is True
    assert ev["length"] == len('"' + big + '"')


async def test_evaluate_survives_non_json_serialisable_value():
    """A set is handled by json.dumps' default=str hook (NOT the except
    fallback — default converts it inline). Content must be preserved and
    the sequence must continue."""
    page = _FakePage(eval_result={1, 2, 3})  # a set: default=str converts
    result = await _run_interact(_op([
        {"action": "goto", "url": "about:blank"},
        {"action": "evaluate", "js": "someSet"},
        {"action": "sleep", "ms": 1},
    ]), page)
    ev = [a for a in result["actions"] if a.get("action") == "evaluate"][0]
    assert ev["ok"] is True
    assert "1" in ev["value"]  # stringified somehow, content preserved
    # The sequence continued past it.
    assert any(a.get("action") == "sleep" and a.get("ok")
               for a in result["actions"])


async def test_evaluate_survives_circular_reference():
    """The except (TypeError, ValueError) fallback path: a circular
    structure raises ValueError even WITH default=str, and must degrade
    to a stringified value rather than crash the action."""
    circular = {}
    circular["self"] = circular
    page = _FakePage(eval_result=circular)
    result = await _run_interact(_op([
        {"action": "goto", "url": "about:blank"},
        {"action": "evaluate", "js": "circularThing"},
    ]), page)
    ev = [a for a in result["actions"] if a.get("action") == "evaluate"][0]
    assert ev["ok"] is True
    assert "self" in ev["value"]


async def test_evaluate_hang_is_bounded_per_action():
    """page.evaluate is NOT governed by set_default_timeout; an unsettled
    promise used to hang until the subprocess kill, which discards the
    [BROWSER_OK] payload — and with it every EARLIER action's result. A
    stuck evaluate must instead fail per-action with a self-explanatory
    message, preserving the rest of the sequence."""
    import asyncio as _asyncio

    class _HangingPage(_FakePage):
        async def evaluate(self, js):
            self.eval_calls.append(js)
            await _asyncio.sleep(999)

    page = _HangingPage()
    result = await _run_interact(_op([
        {"action": "goto", "url": "about:blank"},
        {"action": "evaluate", "js": "new Promise(() => {})",
         "timeout_ms": 100},  # floored to 1s by the runner
        {"action": "sleep", "ms": 1},
    ]), page)
    ev = [a for a in result["actions"] if a.get("action") == "evaluate"][0]
    assert ev["ok"] is False
    assert "timed out" in ev["error"]
    # Earlier and later results survive.
    assert any(a.get("action") == "goto" and a.get("ok")
               for a in result["actions"])
    assert any(a.get("action") == "sleep" and a.get("ok")
               for a in result["actions"])


async def test_evaluate_listed_in_unknown_action_error():
    """A typo'd action name must surface `evaluate` in the valid list so
    the LLM can self-correct instead of typo-thrashing."""
    page = _FakePage()
    result = await _run_interact(_op([
        {"action": "goto", "url": "about:blank"},
        {"action": "eval", "js": "1"},  # invalid name
    ]), page)
    bad = [a for a in result["actions"] if a.get("action") == "eval"][0]
    assert bad["ok"] is False
    assert "evaluate" in bad["error"]


async def test_evaluate_exposed_in_registry_description():
    """The registry actions description is what the LLM sees when picking
    sub-actions. It must (a) list evaluate and (b) steer state questions
    to it instead of screenshot-plus-vision eyeballing."""
    from ghost_agent.tools import registry as registry_mod

    browser_def = next(
        t for t in registry_mod.TOOL_DEFINITIONS
        if t.get("function", {}).get("name") == "browser"
    )
    actions_desc = browser_def["function"]["parameters"]["properties"]["actions"]["description"]
    assert '"evaluate"' in actions_desc
    assert "state" in actions_desc.lower()


async def test_evaluate_branch_in_runner_source():
    """Belt-and-braces: the shipped runner string must contain the evaluate
    branch and its page.evaluate call — a refactor that drops either
    silently regresses the pinball incident."""
    src = browser_mod._runner_script()
    assert 'name == "evaluate"' in src
    assert "await page.evaluate(js)" in src


async def test_evaluate_value_rendered_in_formatter():
    """The outer per-action formatter must surface the VALUE in full —
    the value IS the point of the action; a dropped field would recreate
    the navigate-preview formatter gap all over again."""
    import inspect
    src = inspect.getsource(browser_mod)
    # The formatter lives OUTSIDE the runner string; strip the runner
    # first so we assert on the outer code specifically.
    outer = src.replace(browser_mod._runner_script(), "")
    assert 'act == "evaluate"' in outer
    assert "VALUE:" in outer
