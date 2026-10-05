"""§4LN — the browser tool: behaviour pins for the review's fixes.

Driven through the real `tool_browser` with a stub sandbox that returns the
runner's sentinel line, and through the real runner helpers with fake pages
(live probes B2/B5, traffic rows, reviewer harness cases)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools import browser as B
from ghost_agent.tools import browser_runner as R
from ghost_agent.tools.outcome import ToolOutcome


def _stub(payload, rc=0):
    stub = MagicMock()
    stub.cmds = []

    def _execute(cmd, timeout=300, **kw):
        stub.cmds.append(cmd)
        return (f"[BROWSER_OK] {json.dumps(payload, ensure_ascii=False)}\n", rc)
    stub.execute = _execute
    return stub


def _op_payload(stub) -> dict:
    import shlex
    return json.loads(shlex.split(stub.cmds[-1].split(" ", 3)[-1])[0])


async def _run(tmp_path, payload, **kw):
    return await B.tool_browser(sandbox_dir=tmp_path, sandbox_manager=_stub(payload), **kw)


# ── A: a read page is a declared success; text checks read the header ──

@pytest.mark.parametrize("title,text", [
    ("Built-in Exceptions", "Exception handling: Traceback (most recent call last) …"),
    ("TypeError: Cannot read properties of undefined", "Error: this happens when the array is empty"),
    ("CI log", "build step 3\nEXIT CODE: 1\nSYSTEM ERROR in the job above"),
])
async def test_a_page_about_errors_is_a_successful_read(tmp_path, title, text):
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    from ghost_agent.core.claim_binding import evidence_all_failed
    from ghost_agent.core.evidence_gate import assess_turn_evidence
    res = await _run(tmp_path, {"status": 200, "url": "https://docs.python.org/3/library/exceptions.html",
                                "title": title, "text": text * 3, "length": len(text) * 3},
                     operation="extract_text", url="https://docs.python.org/3/library/exceptions.html")
    assert isinstance(res, ToolOutcome) and res.declared and not res.is_failure and not res.exit_code_failed
    assert not _looks_like_tool_error(str(res), "browser")                 # a stored row (text only)
    assert not evidence_all_failed(f"[browser] {res}")
    a = assess_turn_evidence([{"name": "browser", "content": res}])
    assert a.substantive == 1


@pytest.mark.parametrize("text,failed", [
    ("--- BROWSER RESULT ---\nSTATUS: ERROR\nRunner failed (exit 1): x", True),
    ("[FAILURE BANNER] x\n--- BROWSER RESULT ---\n--- BROWSER RESULT ---\nSTATUS: BLOCKED (HTTP 403)", True),
    ("--- BROWSER RESULT ---\nSTATUS: PARTIAL (1 of 2 actions failed)\nOP: interact", False),   # some steps read the page
    ("Error: refused non-http(s) URL (scheme='file')", True),
    ("--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\nTITLE: Exception handling", False),
])
def test_the_text_sniffer_reads_the_browsers_own_verdict(text, failed):
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    assert _looks_like_tool_error(text, "browser") is failed


# ── B / C: interact says what happened, and shows what it read ──

async def test_an_interact_whose_actions_failed_does_not_say_ok(tmp_path):
    from ghost_agent.core.agent import _has_interaction_evidence
    res = await _run(tmp_path, {"final_url": "file:///workspace/c.html", "final_title": "C", "actions": [
        {"index": 0, "action": "click", "ok": False, "error": "selector '#add-btn' did not match any element"},
        {"index": 1, "action": "extract_text", "ok": True, "text": "0", "length": 1, "selector": "#count"}]},
        operation="interact", url="file:///workspace/c.html",
        actions=[{"action": "click", "selector": "#add-btn"}, {"action": "extract_text", "selector": "#count"}])
    assert "STATUS: OK" not in res and "STATUS: PARTIAL (1 of 2 actions failed)" in res
    assert not _has_interaction_evidence([{"name": "browser", "content": res}])
    every = await _run(tmp_path, {"final_url": "x", "final_title": "x", "actions": [
        {"index": 0, "action": "click", "ok": False, "error": "e"}]},
        operation="interact", url="file:///workspace/c.html", actions=[{"action": "click", "selector": "#a"}])
    assert "STATUS: ERROR (every action failed)" in every and every.is_failure


async def test_an_interact_extract_shows_its_whole_text(tmp_path):
    text = "x" * 4990 + "THE_ANSWER_IS_HERE"
    res = await _run(tmp_path, {"final_url": "u", "final_title": "t", "actions": [
        {"index": 0, "action": "extract_text", "ok": True, "text": text, "length": len(text), "selector": "body"}]},
        operation="interact", url="file:///workspace/p.html", actions=[{"action": "extract_text", "selector": "body"}])
    assert "THE_ANSWER_IS_HERE" in res and res.declared and not res.is_failure


# ── D: the url-less page is this request's, and says so ──

async def test_the_sidecar_is_per_request(tmp_path):
    from ghost_agent.utils.logging import request_id_context
    tok = request_id_context.set("req-abc123")
    try:
        stub = _stub({"status": 200, "url": "https://a.org/", "title": "A", "text": "hello " * 20, "length": 120})
        await B.tool_browser(operation="navigate", url="https://a.org/", sandbox_dir=tmp_path, sandbox_manager=stub)
        assert _op_payload(stub)["last_url_file"] == ".last_url.req-abc123"
    finally:
        request_id_context.reset(tok)
    assert B._last_url_filename() == ".last_url"                 # outside a request: the shared one


async def test_a_urlless_read_names_the_page_it_reopened(tmp_path):
    res = await _run(tmp_path, {"status": 200, "url": "https://www.postgresql.org/docs/", "title": "PG",
                                "text": "release notes " * 10, "length": 140, "used_last_url": True},
                     operation="extract_text")
    assert "re-opened the last page asked for (now at https://www.postgresql.org/docs/)" in res


async def test_a_failed_navigate_leaves_the_page_it_was_asked_for(tmp_path, monkeypatch):
    prof = tmp_path / "prof"
    prof.mkdir()
    R._set_last_url_file(".last_url.r1")
    (prof / ".last_url.r1").write_text("https://old.example/page")

    class _Page:
        async def goto(self, url, wait_until=None):
            raise RuntimeError("Page.goto: Timeout 30000ms exceeded")

    async def _ctx(profile_dir, proxy, timeout_ms, run):
        return await run(_Page())
    monkeypatch.setattr(R, "_with_context", _ctx)
    with pytest.raises(RuntimeError):
        await R.op_navigate({"url": "https://new.example/asked", "profile_dir": str(prof), "timeout_ms": 1000})
    # the next url-less read re-opens what was ASKED for, never the old page
    assert (prof / ".last_url.r1").read_text() == "https://new.example/asked"
    R._set_last_url_file(".last_url")


# ── E: a big or non-ASCII result reaches the host intact ──

def test_a_long_greek_page_fits_the_exec_line_and_parses():
    text = "Ελληνικό κείμενο \u2028 με διαχωριστικό " * 8000     # 288 K chars: over the line cap
    line = "[BROWSER_OK] " + R._trim_payload({"text": text, "url": "https://x.gr/", "length": len(text)})
    assert len(line) <= R._EMIT_MAX_CHARS + 20
    ok, parsed = B._parse_runner_output("Chromium warning\n" + line + "\n")
    assert ok and parsed["text"].startswith("Ελληνικό") and parsed.get("truncated") is True


def test_a_small_payload_is_untouched():
    p = {"text": "short", "url": "u"}
    assert json.loads(R._trim_payload(dict(p))) == p


# ── F: deterministic failures are not "will retry" ──

@pytest.mark.parametrize("err,cls", [
    ("TimeoutError: Page.click: Timeout 30000ms exceeded.\nCall log:\n  - waiting for locator(\"#add-btn\")", "diagnostic"),
    ("x.org failed 2 times in the last few hours (y) — fetching it again will fail the same way.", "diagnostic"),
    ("net::ERR_CONNECTION_REFUSED at http://127.0.0.1:8103/", "diagnostic"),
    ("Page.goto: Timeout 30000ms exceeded", "retryable"),
])
def test_retryable_only_when_a_retry_can_differ(err, cls):
    from ghost_agent.tools.tool_failure import classify_tool_failure
    assert classify_tool_failure(err)[0].value == cls


# ── G: screenshots ──

async def test_a_screenshot_of_a_page_titled_cloudflare_is_not_blocked(tmp_path):
    res = await _run(tmp_path, {"url": "https://www.cloudflare.com/", "title": "Cloudflare | Connect, protect",
                                "path": "/workspace/shot.png", "dom_text_chars": 9000},
                     operation="screenshot", url="https://www.cloudflare.com/", out_path="shot.png")
    assert "BLOCKED" not in res and not res.is_failure
    assert "you have NOT seen this image" in res


def test_a_challenge_title_with_unknown_size_is_not_a_challenge():
    from ghost_agent.tools.browser_routes import blocked_page_reason
    assert blocked_page_reason({"title": "Cloudflare", "url": "https://cloudflare.com/"}) == ""
    assert blocked_page_reason({"title": "Just a moment...", "url": "https://x/", "length": 0}) != ""
    assert blocked_page_reason({"title": "x", "url": "https://x/cdn-cgi/challenge-platform/h"}) != ""


# ── H: a cancelled call keeps the profile lock until its runner returns ──

async def test_the_profile_lock_outlives_a_cancel_until_the_runner_returns():
    import threading
    release = threading.Event()
    lock = asyncio.Lock()

    async def call():
        async with lock:
            await B._exec_holding_lock(lambda: release.wait(5) or ("", 0))
    t = asyncio.create_task(call())
    await asyncio.sleep(0.05)
    t.cancel()
    await asyncio.sleep(0.1)
    assert lock.locked()                      # the runner is still going
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert not lock.locked()


# ── I: the corpus selector label ──

def _tc(name, args, result="--- BROWSER RESULT ---\nSTATUS: OK\nOP: interact", error=""):
    return SimpleNamespace(name=name, arguments=args, result=result, error=error)


def test_edit_then_retest_is_not_stuck_clicking():
    from ghost_agent.distill import outcome_heuristics as OH
    calls = []
    for _ in range(5):
        calls.append(_tc("file_system", {"operation": "replace", "path": "app.js"}, "SUCCESS: replaced"))
        calls.append(_tc("browser", {"operation": "interact", "actions": [{"action": "click", "selector": "#restart"}]}))
    traj = SimpleNamespace(tool_calls=calls, final_response="Fixed — restart works now.", outcome=None,
                           failure_reason="", user_request="fix restart", task_kind="user_request", extra={})
    res = OH.classify_chat_outcome(traj)
    reason = getattr(res, "reason", "") or ""
    assert "selector" not in reason


def test_the_same_click_five_times_with_no_change_is_still_stuck():
    from ghost_agent.distill import outcome_heuristics as OH
    calls = [_tc("browser", {"operation": "click", "selector": "#go"}) for _ in range(6)]
    traj = SimpleNamespace(tool_calls=calls, final_response="done", outcome=None, failure_reason="",
                           user_request="x", task_kind="user_request", extra={})
    res = OH.classify_chat_outcome(traj)
    assert "browser selector '#go' used" in (getattr(res, "reason", "") or "")


# ── J: the extract steer's trigger ──

def test_a_urlless_extract_counts_as_reading_the_page():
    from ghost_agent.core.agent import _browser_loaded_but_never_extracted
    nav = ToolOutcome.ok("nav"); nav.call_args = {"operation": "navigate", "url": "https://a.org/x"}
    ext = ToolOutcome.ok("ext"); ext.call_args = {"operation": "extract_text"}
    rows = [{"name": "browser", "content": nav}, {"name": "browser", "content": nav}]
    assert _browser_loaded_but_never_extracted(rows, "https://a.org/x") is True
    assert _browser_loaded_but_never_extracted(rows + [{"name": "browser", "content": ext}], "https://a.org/x") is False


# ── K: hints by cause; a missing selector fails fast and names what is there ──

def test_hints_by_cause():
    from ghost_agent.tools.browser_routes import blocked_page_hint, BLOCKED_PAGE_HINT
    assert "YOUR OWN app" in blocked_page_hint("HTTP 500", "http://127.0.0.1:8103/x")
    assert "no page at this URL" in blocked_page_hint("HTTP 404", "https://uni.ac.uk/fees-2026")
    assert blocked_page_hint("HTTP 403 — access refused", "https://ft.com/x") == BLOCKED_PAGE_HINT


async def test_a_missing_selector_fails_fast_and_lists_the_real_ones():
    class TimeoutError_(Exception):
        pass
    TimeoutError_.__name__ = "TimeoutError"

    class _Page:
        waited = None

        async def wait_for_selector(self, sel, state=None, timeout=None):
            _Page.waited = timeout
            raise TimeoutError_("Timeout")

        async def evaluate(self, js):
            return ['#add "Add"', 'a[href] "Home"']
    R._MISSING_SELECTORS.clear()
    with pytest.raises(ValueError) as e:
        await R._require_selector(_Page(), "#add-btn", 30000)
    assert _Page.waited == 30000 and '#add "Add"' in str(e.value)     # the full wait, once
    with pytest.raises(ValueError):
        await R._require_selector(_Page(), "#add-btn", 30000)
    assert _Page.waited <= 1                                            # the same guess again: at once
    R._MISSING_SELECTORS.clear()


def test_runner_failure_hint_fits_the_error():
    h = B._runner_failure_hint("ValueError: selector 'h1' did not match any element")
    assert "pick a selector from what is there" in h and "Tor" not in h and "supercharged" not in h
    assert "over Tor" in B._runner_failure_hint("Page.goto: Timeout 30000ms exceeded")


# ── L: budgets ──

async def test_more_than_sixty_actions_is_refused(tmp_path):
    res = await _run(tmp_path, {}, operation="interact", url="file:///workspace/p.html",
                     actions=[{"action": "sleep", "ms": 10}] * 61)
    assert "at most 60 actions" in str(res)


async def test_a_step_wait_is_clamped_to_its_share(tmp_path):
    stub = _stub({"final_url": "u", "final_title": "t", "actions": []})
    await B.tool_browser(operation="interact", url="file:///workspace/p.html", sandbox_dir=tmp_path,
                         sandbox_manager=stub, actions=[{"action": "sleep", "ms": 900000}])
    op = _op_payload(stub)
    assert op["actions"][0]["ms"] <= op["timeout_ms"]


# ── M: link grounding ──

def test_a_url_that_failed_to_load_does_not_vouch_for_itself():
    from ghost_agent.core.link_grounding import haystack_from
    msgs = [{"role": "user", "content": "find the fees page"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "browser",
                                          "arguments": '{"operation":"navigate","url":"https://uni.ac.uk/fees-2026-invented"}'}},
                {"id": "c2", "function": {"name": "browser",
                                          "arguments": '{"operation":"navigate","url":"https://uni.ac.uk/real-page"}'}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": "--- BROWSER RESULT ---\nSTATUS: BLOCKED (HTTP 404)\nURL: https://uni.ac.uk/fees-2026-invented"},
            {"role": "tool", "tool_call_id": "c2", "content": "--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate"}]
    hay = haystack_from(msgs)
    assert "fees-2026-invented" not in hay and "real-page" in hay


# ── N: the research ledger ──

async def test_a_blocked_fetch_is_not_booked_as_pulled(tmp_path):
    wm = SimpleNamespace(enabled=True, record_research_artifact=MagicMock(), record_navigation=MagicMock(return_value=""))
    await B.tool_browser(operation="navigate", url="https://ft.com/x", sandbox_dir=tmp_path, workspace_model=wm,
                         sandbox_manager=_stub({"status": 403, "url": "https://ft.com/x", "title": "Just a moment...",
                                                "text": "", "length": 0}))
    wm.record_research_artifact.assert_not_called()


# ── P: files ──

async def test_a_screenshot_never_overwrites_a_users_photo(tmp_path):
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xffJPEGDATA")
    res = await _run(tmp_path, {"url": "u", "path": "/workspace/photo.jpg", "dom_text_chars": 500},
                     operation="screenshot", url="https://a.org/", out_path="photo.jpg")
    assert "will not overwrite" in str(res)
    assert (tmp_path / "photo.jpg").read_bytes() == b"\xff\xd8\xffJPEGDATA"


async def test_an_earlier_screenshot_may_be_retaken(tmp_path):
    (tmp_path / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\nold")
    res = await _run(tmp_path, {"url": "u", "path": "/workspace/shot.png", "dom_text_chars": 500},
                     operation="screenshot", url="https://a.org/", out_path="shot.png")
    assert "will not overwrite" not in str(res)


def test_close_reports_whether_the_profile_is_gone(tmp_path):
    prof = tmp_path / "p"
    prof.mkdir()
    out = asyncio.run(R.OPS["close"]({"profile_dir": str(prof)}))
    assert out["closed"] is True


# ── Q: interact landing on a refusal is BLOCKED ──

async def test_an_interact_goto_onto_a_403_is_blocked(tmp_path):
    res = await _run(tmp_path, {"final_url": "https://zoopla.co.uk/x", "final_title": "Just a moment...", "actions": [
        {"index": 0, "action": "goto", "ok": True, "url": "https://zoopla.co.uk/x", "title": "Just a moment...", "status": 403},
        {"index": 1, "action": "extract_text", "ok": True, "text": "Checking your browser", "length": 21, "selector": "body"}]},
        operation="interact", actions=[{"action": "goto", "url": "https://zoopla.co.uk/x"},
                                       {"action": "extract_text", "selector": "body"}])
    assert "STATUS: BLOCKED (HTTP 403" in res and res.is_failure


# ── F9 / F10 ──

def test_a_failed_screenshot_is_not_the_after_image(tmp_path):
    from ghost_agent.core.agent import _select_visual_evidence
    (tmp_path / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\nyesterday")
    msgs = [{"role": "user", "content": "fix the layout"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "s1", "function": {
                "name": "browser", "arguments": '{"operation":"screenshot","out_path":"shot.png"}'}}]},
            {"role": "tool", "tool_call_id": "s1", "content": "--- BROWSER RESULT ---\nSTATUS: ERROR\nRunner failed: timeout"}]
    assert _select_visual_evidence(msgs, "fix the layout", tmp_path)[1] is None


def test_a_refused_navigate_is_not_progress_for_the_selector_window():
    from ghost_agent.distill import outcome_heuristics as OH
    calls = [_tc("browser", {"operation": "click", "selector": "#go"})]
    for _ in range(2):               # two refusals: under the same-error rule's 3
        calls.append(_tc("browser", {"operation": "navigate", "url": "https://dead.example/"},
                         "dead.example failed 2 times — fetching it again will fail the same way", error="refused"))
        calls.append(_tc("browser", {"operation": "click", "selector": "#go"}))
    calls.append(_tc("browser", {"operation": "click", "selector": "#go"}))
    traj = SimpleNamespace(tool_calls=calls, final_response="done", outcome=None, failure_reason="",
                           user_request="x", task_kind="user_request", extra={})
    assert "browser selector '#go' used" in (getattr(OH.classify_chat_outcome(traj), "reason", "") or "")


async def test_browser_calls_in_one_batch_run_in_the_order_written(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock
    from tests.test_requester_role import _agent as _loop_agent, _tc as _call
    from tests.test_4kl_member_capability import _resp, FakeBgTasks
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    order = []

    async def browser(**kw):
        name = kw.get("url") or kw.get("operation")
        order.append(("start", name))
        await asyncio.sleep(0.2 if name == "https://a.org/" else 0.0)     # the first is the slow one
        order.append(("end", name))
        return ToolOutcome.ok(f"--- BROWSER RESULT ---\nSTATUS: OK\nOP: x\nURL: {name}")
    agent.available_tools = {"browser": browser}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_call("c0", "browser", {"operation": "navigate", "url": "https://a.org/"}),
                   _call("c1", "browser", {"operation": "extract_text"})]),
        _resp("Done."), _resp("Done.")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "read a.org"}]},
                            FakeBgTasks(), request_id="web-4ln-order")
    assert order[:4] == [("start", "https://a.org/"), ("end", "https://a.org/"),
                         ("start", "extract_text"), ("end", "extract_text")]


def test_a_bannered_runner_failure_still_reads_failed():
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    t = ("[FAILURE BANNER] Runner failed (exit 1)\n--- BROWSER RESULT ---\n--- BROWSER RESULT ---\n"
         "STATUS: ERROR\nRunner failed (exit 1): x")
    assert _looks_like_tool_error(t, "browser") is True


async def test_a_screenshot_of_a_real_challenge_page_is_blocked(tmp_path):
    res = await _run(tmp_path, {"url": "https://ft.com/x", "title": "Just a moment...",
                                "path": "/workspace/ft.png", "dom_text_chars": 40},
                     operation="screenshot", url="https://ft.com/x", out_path="ft.png")
    assert "STATUS: BLOCKED" in res and res.is_failure


async def test_a_non_timeout_selector_error_is_left_to_the_action():
    class _Page:
        async def wait_for_selector(self, sel, state=None, timeout=None):
            raise ValueError("invalid selector syntax")

        async def evaluate(self, js):
            raise AssertionError("must not list candidates for a non-timeout error")
    assert await R._require_selector(_Page(), "div[", 30000) is None



# ── second fresh-eye review of the §4LN diff ──

def test_a_partial_interact_with_a_good_read_is_evidence():
    from ghost_agent.core.claim_binding import evidence_all_failed
    body = ("--- BROWSER RESULT ---\nSTATUS: PARTIAL (1 of 2 actions failed)\nOP: interact\n"
            "--- PER-ACTION RESULTS ---\n  [0] ERR click: x\n  [1] OK extract_text: len=40\n      TEXT: the price is 17.25 euro")
    assert not evidence_all_failed(f"[browser] {body}")


def test_a_textless_navigate_of_an_app_is_still_evidence():
    from ghost_agent.core.evidence_gate import assess_turn_evidence
    row = "--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\nURL: file:///workspace/game/index.html\nHTTP_STATUS: None\nTITLE: Game"
    assert assess_turn_evidence([{"name": "browser", "content": ToolOutcome.ok(row)}]).substantive == 1


async def test_a_blocked_interact_strikes_the_blocked_host_and_a_later_real_read_counts(tmp_path, monkeypatch):
    struck = []
    monkeypatch.setattr(B, "_mark_host_failed", lambda url, cause: struck.append(url))
    res = await _run(tmp_path, {"final_url": "https://reuters.com/x", "final_title": "Just a moment...", "actions": [
        {"index": 0, "action": "goto", "ok": True, "url": "https://news.ycombinator.com/", "title": "HN", "status": 200},
        {"index": 1, "action": "goto", "ok": True, "url": "https://reuters.com/x", "title": "Just a moment...", "status": 403}]},
        operation="interact", actions=[{"action": "goto", "url": "https://news.ycombinator.com/"},
                                       {"action": "goto", "url": "https://reuters.com/x"}])
    assert "BLOCKED" in res and struck == ["https://reuters.com/x"]
    ok = await _run(tmp_path, {"final_url": "https://hn/", "final_title": "HN", "actions": [
        {"index": 0, "action": "goto", "ok": True, "url": "https://reuters.com/x", "title": "Just a moment...", "status": 403},
        {"index": 1, "action": "goto", "ok": True, "url": "https://news.ycombinator.com/", "title": "HN", "status": 200},
        {"index": 2, "action": "extract_text", "ok": True, "text": "stories", "length": 7, "selector": "body"}]},
        operation="interact", actions=[{"action": "goto", "url": "https://reuters.com/x"},
                                       {"action": "goto", "url": "https://news.ycombinator.com/"},
                                       {"action": "extract_text", "selector": "body"}])
    assert "BLOCKED" not in ok and not ok.is_failure


def test_a_follow_up_turn_can_still_reopen_the_last_page(tmp_path):
    prof = tmp_path / "p"
    R._set_last_url_file(".last_url.turn1")
    R._write_last_url(str(prof), "https://a.org/article")
    R._set_last_url_file(".last_url.turn2")                  # the next request: no file of its own yet
    assert R._read_last_url(str(prof)) == "https://a.org/article"
    R._set_last_url_file(".last_url")


def test_the_extract_steer_counts_a_urlless_extract_only_after_that_navigate():
    from ghost_agent.core.agent import _browser_loaded_but_never_extracted
    def row(**a):
        o = ToolOutcome.ok("x"); o.call_args = a; return {"name": "browser", "content": o}
    other = [row(operation="navigate", url="https://b.org/"), row(operation="extract_text"),
             row(operation="navigate", url="https://a.org/x"), row(operation="navigate", url="https://a.org/x")]
    assert _browser_loaded_but_never_extracted(other, "https://a.org/x") is True


def test_the_strike_line_of_a_failed_interact_names_the_action():
    from ghost_agent.core.strikes import _content_failure_head
    lines = ["--- BROWSER RESULT ---", "STATUS: ERROR (every action failed)", "OP: interact",
             "--- PER-ACTION RESULTS ---", "[0] ERR click: selector '#x' did not match any element"]
    assert _content_failure_head(lines, declared=True).startswith("[0] ERR click")


def test_page_text_does_not_classify_a_browser_failure():
    from ghost_agent.tools.tool_failure import classify_tool_failure
    t = ("--- BROWSER RESULT ---\nSTATUS: PARTIAL (1 of 2 actions failed)\nOP: interact\n"
         "  [0] ERR click: selector '#go' did not match any element\n"
         "  [1] OK extract_text: len=60\n      TEXT: connection timed out errors and rate limit advice")
    assert classify_tool_failure(t)[0].value == "diagnostic"


def test_the_trim_never_emits_invalid_json():
    p = {"actions": [{"text": "x" * 300000}, {"value": "y" * 300000}], "text": "z" * 300000}
    line = R._trim_payload(p, limit=1000)
    assert json.loads(line)["payload_trimmed"] is True


async def test_an_atomic_settle_wait_is_clamped(tmp_path):
    stub = _stub({"url": "u", "path": "/workspace/s.png", "dom_text_chars": 500})
    await B.tool_browser(operation="screenshot", url="https://a.org/", out_path="s.png", sandbox_dir=tmp_path,
                         sandbox_manager=stub, settle_ms=900000)
    op = _op_payload(stub)
    assert op["settle_ms"] <= op["timeout_ms"]


def test_the_agents_own_app_url_survives_its_own_500():
    from ghost_agent.core.link_grounding import haystack_from
    msgs = [{"role": "user", "content": "check my app"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {
                "name": "browser", "arguments": '{"operation":"navigate","url":"http://127.0.0.1:8103/escape_beacon.html"}'}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": "--- BROWSER RESULT ---\nSTATUS: BLOCKED (HTTP 500)\nOP: navigate\nURL: http://127.0.0.1:8103/escape_beacon.html"}]
    assert "escape_beacon" in haystack_from(msgs)


def test_page_text_saying_timed_out_does_not_make_a_failure_retryable():
    from ghost_agent.tools.tool_failure import classify_tool_failure
    t = ("--- BROWSER RESULT ---\nSTATUS: PARTIAL (1 of 2 actions failed)\nOP: interact\n"
         "  [0] ERR click: Element is outside of the viewport\n"
         "  [1] OK extract_text: len=60\n      TEXT: the request timed out; connection reset by peer")
    assert classify_tool_failure(t)[0].value != "retryable"


def test_the_trim_falls_back_to_valid_json_when_it_cannot_converge():
    p = {"actions": [{"text": "x" * 2000} for _ in range(100)]}
    line = R._trim_payload(p, limit=1000)
    assert json.loads(line)["payload_trimmed"] is True


# ── live probe B2 after deploy: the verifier refuted "3" with the pre-click "0" ──

_TWO_READS = """[browser] --- BROWSER RESULT ---
STATUS: PARTIAL (3 of 4 actions failed)
OP: interact
FINAL_URL: file:///workspace/probe_4ln/counter.html
      TEXT: Count: 0
[browser] --- BROWSER RESULT ---
STATUS: OK
OP: interact
FINAL_URL: file:///workspace/probe_4ln/counter.html
      TEXT: Count: 3"""


def test_a_later_reading_of_the_same_page_supersedes_the_earlier_one():
    from ghost_agent.core.claim_binding import find_conflicting_line
    assert find_conflicting_line(_TWO_READS, "Count: 3", "the count shown is 3") is None


def test_reporting_the_stale_reading_is_still_a_conflict():
    from ghost_agent.core.claim_binding import find_conflicting_line
    assert find_conflicting_line(_TWO_READS, "Count: 0", "the count shown is 0") == "TEXT: Count: 3"
