"""§4RI (2026-10-25, operator: "look at request 43fe95a2 and evaluate" →
"proceed with all 4 improvements"). Three of the four, each driven through
the real code (the fourth — the in-line pivot — lives in
tests/test_system3_crisis_pivot.py):

  * `manage_services start` on a server that was already there was booked
    failure 1 of the 4 that fired the recovery plan.
  * write + two appends of one parts file counted as "3 unverified edits".
  * two browser interact timeouts on a page with a 3.3 s boot animation; the
    model was never told the budget is its own timeout_ms.
"""
import asyncio
import json
from unittest.mock import MagicMock

import pytest

import ghost_agent.core.agent as A
import ghost_agent.tools.browser_runner as R
from ghost_agent.core.strikes import StrikeLedger
from ghost_agent.sandbox.services import ServiceSupervisor
from ghost_agent.tools.outcome import OutcomeStatus
from ghost_agent.tools.sandbox_services import tool_manage_services
from tests.test_4jj_search_yield_steer import _agent as _sy_agent, _ts as _sy_ts
from tests.test_sandbox_services import FakeSandbox, happy_handler


# ── already running = information, not a strike ─────────────────────────────

def test_a_start_on_a_running_service_is_a_note_that_names_where_it_serves(tmp_path):
    """Fails in the measured world: an `Error:` head, declared REJECTED, strike 1."""
    sup = ServiceSupervisor(FakeSandbox(tmp_path, happy_handler()))
    sup.start("webos", "cd /workspace/webos && python3 -m http.server ${PORT:-8100}", port=8101,
              workdir="/workspace/webos")
    out = sup.start("webos", "python3 -m http.server 8100", port=8100, workdir="/workspace/webos")
    assert out.startswith("Note: service 'webos' is already running")
    assert not out.lstrip().startswith(("Error:", "ERROR:"))
    assert "http://127.0.0.1:8101/" in out and "/workspace/webos" in out
    assert "DIFFERENT command" in out and "action='stop'" in out        # the §4NW guidance stays


async def test_the_tool_declares_a_running_service_ok(tmp_path):
    """Through the tool: the second start is OK (not REJECTED), so it is no
    failure strike; a true refusal (a port the sandbox reserves) still is."""
    sb = FakeSandbox(tmp_path, happy_handler())
    first = await tool_manage_services(action="start", name="webos", command="python3 -m http.server 8100",
                                       port=8101, sandbox_manager=sb)
    assert first.status == OutcomeStatus.OK, str(first)
    again = await tool_manage_services(action="start", name="webos", command="python3 -m http.server 8100",
                                       port=8100, sandbox_manager=sb)
    assert again.status == OutcomeStatus.OK and not again.is_failure, str(again)
    assert "already running" in str(again) and "nothing was launched" in str(again)
    refused = await tool_manage_services(action="stop", name="no-such-service", sandbox_manager=sb)
    assert refused.status == OutcomeStatus.REJECTED and refused.is_failure, str(refused)   # a real refusal still is one


# ── a parts file is one write ────────────────────────────────────────────────

def test_an_append_part_is_not_another_edit():
    assert A._is_parts_continuation({"operation": "write", "path": "a.html", "append": True})
    assert A._is_parts_continuation({"operation": "write", "append": "true", "more_parts": True})
    assert not A._is_parts_continuation({"operation": "write", "path": "a.html", "more_parts": True})   # the FIRST part counts
    assert not A._is_parts_continuation({"operation": "write", "append": False})
    assert not A._is_parts_continuation({"operation": "edit", "append": True})
    assert not A._is_parts_continuation(None)


async def _three_batches(calls):
    """Run three one-call batches against one request state; return the
    user-role steers the dispatcher appended."""
    agent = _sy_agent()

    async def file_system(**kw):
        return f"Wrote {len(str(kw.get('content') or ''))} bytes to {kw.get('path')}"
    agent.available_tools = {"file_system": file_system}
    shared = MagicMock()
    shared._edit_churn = None
    steers = []
    for call in calls:
        ts = _sy_ts([call], StrikeLedger(), set())
        ts.request_state = shared
        await agent._dispatch_and_process_tool_batch(ts)
        steers += [str(m.get("content")) for m in ts.messages
                   if m.get("role") == "user" and "you have now edited" in str(m.get("content"))]
    return steers


async def test_the_churn_brake_lets_a_file_be_written_in_parts():
    """Fails where write + 2 appends of webos/index.html steered 'verify
    before further edits' in the middle of the file (43fe95a2, +1882 s)."""
    parts = [("file_system", {"operation": "write", "path": "webos/index.html", "content": "<html>", "more_parts": True}),
             ("file_system", {"operation": "write", "path": "webos/index.html", "content": "<script>", "append": True, "more_parts": True}),
             ("file_system", {"operation": "write", "path": "webos/index.html", "content": "</script>", "append": True})]
    assert await _three_batches(parts) == []


async def test_the_churn_brake_still_catches_three_blind_edits():
    """The control: three real edits of one file, no run between them, steer."""
    edits = [("file_system", {"operation": "edit", "path": "app.py", "content": "a", "replace_with": "b"})] * 3
    steers = await _three_batches(edits)
    assert steers and "app.py" in steers[0]


# ── the browser budget names its own remedy ──────────────────────────────────

def test_the_budget_overrun_tells_the_model_to_raise_timeout_ms(monkeypatch):
    """Driven through the runner's own deadline (the §4PC harness): the error
    the model reads must name timeout_ms and the per-action split."""
    class _Page:
        url = "https://a.org/"
        def set_default_timeout(self, ms): pass
        async def wait_for_timeout(self, ms): pass
    class _Ctx:
        pages = [_Page()]
        async def route(self, *a, **k): pass
        async def route_web_socket(self, *a, **k): pass
        def on(self, *a, **k): pass
        async def close(self): pass
    class _Chromium:
        async def launch_persistent_context(self, **kw): return _Ctx()
    class _PW:
        chromium = _Chromium()
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    async def _fake_guard(ctx, sandbox_root=None, anonymous=False):
        return None
    monkeypatch.setattr(R, "async_playwright", lambda: _PW())
    monkeypatch.setattr(R, "_attach_diagnostics", lambda page: ([], []))
    monkeypatch.setattr(R, "_install_ssrf_guard", _fake_guard)

    async def spins(page):
        await asyncio.sleep(10)
    with pytest.raises(RuntimeError) as e:
        asyncio.run(R._with_context("/tmp/p", None, 200, spins))
    msg = str(e.value)
    assert "exceeded its 200 ms budget" in msg and "the page kept the browser busy" in msg   # browser_support still classifies it
    assert "timeout_ms=60000" in msg and "split across the actions" in msg and "boot animation" in msg


# ── §4RJ: a throw inside the model's own evaluate is a result, not a failure ─

from tests.test_4ln_browser import _run as _run_browser  # noqa: E402  (the §4LN interact harness)


def _interact(tmp_path, actions):
    return _run_browser(tmp_path, {"final_url": "file:///workspace/webos/index.html", "final_title": "NebulaOS",
                                   "actions": actions},
                        operation="interact", url="file:///workspace/webos/index.html",
                        actions=[{"action": a["action"], "selector": "#x", "js": "1"} for a in actions])


async def test_an_evaluate_that_threw_in_the_models_script_leaves_the_interact_ok(tmp_path):
    """Fails in the measured world: PARTIAL (1 of 9 actions failed) → failure
    strike → the final turn booked failed over a page the verifier found clean."""
    res = await _interact(tmp_path, [
        {"index": 0, "action": "click", "ok": True},
        {"index": 1, "action": "extract_text", "ok": True, "text": "9-4=5", "length": 5, "selector": "#calc"},
        {"index": 2, "action": "evaluate", "ok": False,
         "error": "Error: SyntaxError: Unexpected token '{' — if(readme)readme.click();{clicked:!!readme}"}])
    assert res.status == OutcomeStatus.OK and not res.is_failure, str(res)
    assert "STATUS: OK (1 evaluate step threw inside YOUR script" in str(res)
    assert "ACTIONS: 2 OK, 1 error" in str(res) and "ERR evaluate" in str(res)    # the throw is still shown
    from ghost_agent.core.agent import _has_interaction_evidence
    assert _has_interaction_evidence([{"name": "browser", "content": str(res)}])


async def test_a_failed_click_or_a_timed_out_evaluate_is_still_partial(tmp_path):
    click = await _interact(tmp_path, [
        {"index": 0, "action": "click", "ok": False, "error": "selector '#add' did not match any element"},
        {"index": 1, "action": "evaluate", "ok": False, "error": "Error: ReferenceError: x is not defined"},
        {"index": 2, "action": "extract_text", "ok": True, "text": "t", "length": 1, "selector": "#c"}])
    assert click.status == OutcomeStatus.PARTIAL and "STATUS: PARTIAL (2 of 3 actions failed)" in str(click)
    timed = await _interact(tmp_path, [
        {"index": 0, "action": "click", "ok": True},
        {"index": 1, "action": "evaluate", "ok": False,
         "error": "RuntimeError: evaluate timed out after 27s — the expression never settled"}])
    assert timed.status == OutcomeStatus.PARTIAL, str(timed)
    alone = await _interact(tmp_path, [
        {"index": 0, "action": "evaluate", "ok": False, "error": "Error: TypeError: null"}])
    assert alone.is_failure and "STATUS: ERROR (every action failed)" in str(alone)
