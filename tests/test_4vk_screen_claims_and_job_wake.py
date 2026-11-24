"""§4VK (2026-11-24) — req 16faf8b4 and its job wake job-958c (webOS DOOM).

16faf8b4 replied "I confirmed the engine's own UI renders" after its last two
screenshot checks saw a solid black window and then no DOOM window at all.
job-958c (the wake of job-958c1ed3) ran 3.5 h unattended; its reply opened with
a 1,302-char note written beside a probe at turn 14, the away digest quoted that
note under "[scheduled task] … FAILED" (the FAILED was the JOB's exit 1), and its
unverified-claim footer named `game.html`, `game.html` (test pages under /tmp)
ahead of `serve.py`.

Behavioural pins, plus AST wiring pins for the two delivery-path call sites
(review r1: the screen check sat inside the no-verdict branch).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.core import agent as A
from ghost_agent.core import code_claims as CC
from ghost_agent.tools.outcome import ToolOutcome

from tests import pin_ast as P


def _vision(target, shows_problem, evidence="what was seen", answer="UNCERTAIN"):
    v = {"answer": answer, "confidence": 0.95, "evidence": evidence, "details": ""}
    if shows_problem is not None:
        v["shows_problem"] = shows_problem
    text = "UI VERIFICATION RESULT (judged from pixels only):\n" + json.dumps(v, indent=2)
    return {"role": "tool", "name": "vision_analysis",
            "content": ToolOutcome.ok(text, world_changed=False),
            "call_args": {"action": "verify_ui", "target": target, "prompt": "what is in the window?"}}


def _fs(path, n=3000):
    return {"role": "tool", "name": "file_system",
            "content": ToolOutcome(f"SUCCESS: Wrote {n} chars to '{path}'.",
                                   call_args={"operation": "write", "path": path})}


REPLY_16FAF8B4 = ("- **The game now reaches the engine** inside the webOS window. I confirmed the engine's "
                  "own UI renders in the iframe with its \"Ready — click Launch\" panel.")


# ── the screen arm ────────────────────────────────────────────────────

def test_a_success_claim_after_a_black_screen_is_contradicted():
    tools = [_vision("doom-repro/coi.png", True, "The game window area is a solid black rectangle")]
    note = CC.screen_contradiction_note(REPLY_16FAF8B4, tools)
    assert "Contradicted by the screen" in note
    assert "coi.png" in note and "solid black rectangle" in note


def test_a_later_clean_check_of_the_same_picture_clears_its_problem():
    tools = [_vision("a.png", True, "error card"), _vision("a.png", False, "the game menu is visible")]
    assert CC.screen_contradiction_note(REPLY_16FAF8B4, tools) == ""


def test_a_clean_after_shot_under_a_new_name_clears_it():
    """Review r2 MAJOR: before/after screenshots carry different names (coi.png → coi2.png); a fix that
    re-checks under a new name must not be footered as contradicted."""
    tools = [_vision("shot1.png", True, "solid black rectangle"), _vision("shot2.png", False, "DOOM title menu")]
    assert CC.screen_contradiction_note("I fixed the WASM path and the game now renders its title menu.",
                                        tools) == ""


def test_a_later_problem_overrides_an_earlier_clean_check():
    tools = [_vision("a.png", False, "menu"), _vision("b.png", True, "no window open")]
    assert "b.png" in CC.screen_contradiction_note(REPLY_16FAF8B4, tools)


def test_a_check_without_the_verdict_field_says_nothing():
    """An answer from before §4VK (or a model that dropped the field) is not read as a problem."""
    assert CC.screen_contradiction_note(REPLY_16FAF8B4, [_vision("a.png", None, "black")]) == ""


def test_an_honest_report_of_the_problem_is_not_a_claim():
    tools = [_vision("a.png", True, "black window")]
    assert CC.screen_contradiction_note("The DOOM window still shows a black screen; it never renders.",
                                        tools) == ""


@pytest.mark.parametrize("reply", [
    "Root cause found and fixed: the server sends the headers.",
    "No more CROSS-ORIGIN ISOLATION FAILED stuck state.",
    "The game now loads inside the window.",
    "I confirmed the engine's own UI renders in the iframe.",
])
def test_on_screen_success_forms_are_claims(reply):
    assert CC.claims_code_works(reply) == ""                      # not the old detector's: this arm's own
    assert CC.claims_screen_success(reply)


@pytest.mark.parametrize("reply", [                                # review r1, each verified a false alarm
    "I fixed the path; however the screen is still black.",
    "Here is what the screenshot shows: an error dialog.",
    "What the check shows: no window open.",
    "The table below shows the numbers per month.",
    "I used a fixed-width font for the log.",
    "Unfortunately the canvas is black. It displays the HUD only.",
    "The window never renders.",
    "The page should load once the server restarts.",
    "The page loads, but the canvas stays black.",              # a claim word in a sentence reporting the problem
    "The window now renders the error dialog again.",
    "It loads the WAD file and then exits with code 1.",          # review r2
    "It is still black in index.html, though it loads.",          # review r2: a dot in a filename ends nothing
])
def test_reports_and_unrelated_sentences_are_not_screen_claims(reply):
    assert CC.claims_screen_success(reply) == ""


def test_the_footer_function():
    tools = [_vision("coi2.png", True, "no DOOM window is open")]
    assert "Contradicted by the screen" in A.screen_contradiction_footer(REPLY_16FAF8B4, tools)
    assert A.screen_contradiction_footer("", tools) == ""
    once = REPLY_16FAF8B4 + A.screen_contradiction_footer(REPLY_16FAF8B4, tools)
    assert A.screen_contradiction_footer(once, tools) == ""          # never twice


def _fn(cls_or_mod, name):
    import ast
    for n in ast.walk(P.tree(A.GhostAgent)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    raise AssertionError(name)


def _enclosing_ifs(fn, call):
    import ast
    out = []

    def walk(node, stack):
        for ch in ast.iter_child_nodes(node):
            if ch is call:
                out.extend(stack)
                return True
            if walk(ch, stack + ([P.U(node.test)] if isinstance(node, ast.If) and ch in node.body + node.orelse
                                 else [])):
                return True
        return False
    walk(fn, [])
    return out


def test_finalize_runs_the_screen_check_whatever_the_verifier_said():
    """Review r1 MAJOR: inside the verifier's no-verdict branch a CONFIRMED skipped it."""
    fn = _fn(A, "_finalize_and_return")
    c = P.calls(fn, "screen_contradiction_footer")
    assert len(c) == 1
    assert not any("v_result" in t or "verifier" in t.lower() for t in _enclosing_ifs(fn, c[0]))
    assert P.assigned_values(fn, "verifier_backfill").count("('failed', SCREEN_CONTRADICTION_REASON)") == 1


def test_the_streamed_final_runs_it_and_books_it_failed():
    import ast
    fn = _fn(A, "stream_wrapper")
    assert len(P.calls(fn, "screen_contradiction_footer")) == 1
    assert "True" in P.assigned_values(fn, "_screen_contradicted_stream")
    import ast as _ast
    booked = [n for n in _ast.walk(fn) if isinstance(n, _ast.IfExp)
              and P.U(n.test) == "_um_failed_stream or _screen_contradicted_stream"]
    assert booked, "the streamed trajectory's verifier label must read the screen flag"


def test_verify_ui_asks_for_the_verdict_field(monkeypatch, tmp_path):
    """The field the arm reads must be asked for — else the arm never fires."""
    from ghost_agent.tools import vision as V
    sent = {}

    class _LLM:
        async def chat_completion(self, payload, **kw):
            sent["p"] = payload
            return {"choices": [{"message": {"content": '{"answer":"NO","shows_problem":true}'}}]}
    img = str(tmp_path / "px.png")
    import base64
    with open(img, "wb") as f:
        f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="))
    asyncio.run(V.tool_vision_analysis(action="verify_ui", target="px.png", prompt="is it there?",
                                       llm_client=_LLM(), sandbox_dir=tmp_path))
    parts = sent["p"]["messages"][1]["content"]
    text = " ".join(c.get("text", "") for c in parts if isinstance(c, dict))
    assert '"shows_problem": true | false' in text          # the schema line, not just the explanation
    assert "is not a problem" in text                         # review r1: an expected error/empty screen


# ── the footer's file names ───────────────────────────────────────────

def test_names_show_each_path_once_and_tell_same_named_files_apart():
    assert CC._names(["serve.py", "serve.py", "doom.js"]) == "`serve.py`, `doom.js`"
    assert CC._names(["src/index.html", "dist/index.html"]) == "`src/index.html`, `dist/index.html`"


def test_scratch_files_drop_out_when_project_files_were_written(monkeypatch):
    led = {1: {"path": "/tmp/coib/game.html", "write_idx": 1, "run_idx": None, "run_ok": None},
           2: {"path": "serve.py", "write_idx": 2, "run_idx": None, "run_ok": None},
           3: {"path": "/private/tmp/x/game.html", "write_idx": 3, "run_idx": None, "run_ok": None}}
    monkeypatch.setattr(CC, "code_evidence", lambda tools: led)
    never, failed, _ = CC._unbacked([])
    assert never == ["serve.py"] and failed == []


def test_a_failed_scratch_script_still_counts(monkeypatch):
    """Review r1: dropping it turned BROKEN into UNTESTED."""
    led = {1: {"path": "serve.py", "write_idx": 1, "run_idx": None, "run_ok": None},
           2: {"path": "/tmp/test_serve.py", "write_idx": 2, "run_idx": 3, "run_ok": False},
           3: {"path": "/var/folders/ab/cd1234/T/probe.html", "write_idx": 4, "run_idx": None, "run_ok": None}}
    monkeypatch.setattr(CC, "code_evidence", lambda tools: led)
    never, failed, _ = CC._unbacked([])
    assert never == ["serve.py"] and failed == ["/tmp/test_serve.py"]


def test_scratch_files_stay_when_they_are_all_there_is(monkeypatch):
    led = {1: {"path": "/tmp/probe.py", "write_idx": 1, "run_idx": None, "run_ok": None}}
    monkeypatch.setattr(CC, "code_evidence", lambda tools: led)
    assert CC._unbacked([])[0] == ["/tmp/probe.py"]


# ── the reply's opening note ─────────────────────────────────────────

def test_a_long_work_note_beside_a_probe_is_dropped_when_the_report_outweighs_it():
    from ghost_agent.core.reply_smoothing import WorkSegment, drop_pre_tool_segments
    note = ("The matrix gave me the decisive architectural fact. Let me interpret it and get the one "
            "missing piece of evidence. " + "Isolated top plus a plain child is blocked by the response. " * 20).strip()
    assert 1000 < len(note) < 4000
    report = ("Here's where the DOOM investigation landed, and exactly what's on disk now. "
              + "The frame is served same-origin through serve.py and the isolation check passes. " * 30)
    text = note + "\n\n" + report
    out = drop_pre_tool_segments(text, [WorkSegment("Reading the files."), note])
    assert out.startswith("Here's where the DOOM investigation landed")


def test_let_me_know_is_not_a_step_beat():
    """Review r2: "let me know if you want EUR" inside a comparison dropped it."""
    from ghost_agent.core.reply_smoothing import WorkSegment, drop_pre_tool_segments
    cmp_ = ("Here is the full comparison of the two funds. Let me know if you want the costs in EUR. "
            + "Fund A charges 0.07% and tracks the index within 0.02% over five years. " * 22).strip()
    assert 1000 < len(cmp_) < 4000
    rest = "My recommendation is to keep Fund A as the core and add monthly. " * 55
    out = drop_pre_tool_segments(cmp_ + "\n\n" + rest, [WorkSegment("Saving."), cmp_])
    assert out.startswith("Here is the full comparison")


def test_let_me_inside_a_sentence_is_not_a_step_beat():
    from ghost_agent.core.reply_smoothing import WorkSegment, drop_pre_tool_segments
    res = ("Results: the totals below, let me stress, include every fee charged in 2026. "
           + "Account B returned 4.1% after fees across the twelve months measured. " * 22).strip()
    rest = "My recommendation is to keep Account B and review it in March next year. " * 50
    out = drop_pre_tool_segments(res + "\n\n" + rest, [WorkSegment("Saving."), res])
    assert out.startswith("Results:")


def test_a_long_results_note_without_a_step_beat_stays():
    """Review r1: a 2,158-char table written beside a save, then the recommendations."""
    from ghost_agent.core.reply_smoothing import WorkSegment, drop_pre_tool_segments
    table = ("| month | total |\n|---|---|\n" + "| 2026-01 | 1,234.56 euro across all accounts |\n" * 50).strip()
    assert 1000 < len(table) < 4000
    recs = "Recommendation: keep the monthly transfer and rebalance once a quarter as planned. " * 45
    text = table + "\n\n" + recs
    out = drop_pre_tool_segments(text, [WorkSegment("Saving."), table])
    assert out.startswith("| month | total |")


# ── the away digest ───────────────────────────────────────────────────

def _ledger(tmp_path):
    from ghost_agent.core.autonomous_activity import ActivityLog
    return ActivityLog(str(tmp_path / "activity.jsonl"))


def _records(log):
    recs, _ = log.read_since(0)
    return recs


def test_a_job_wake_is_labelled_a_background_job_with_the_jobs_own_exit(tmp_path):
    import ghost_agent.core.autonomous_activity as AA
    log = _ledger(tmp_path)
    AA.record_scheduled_result(log, job_id="job-958c1ed3", task_name="background job job-958c1ed3",
                               content="Here's where it landed.", ok=False,
                               status="done (exit 1) — the follow-up turn says")
    recs = _records(log)
    for text in (AA.render_activity_digest(recs), AA.render_activity_brief(recs),
                 AA.render_activity_report(recs)):
        assert "[background job]" in text and "[scheduled task]" not in text
        assert "done (exit 1) — the follow-up turn says: Here's where it landed." in text
        assert "FAILED" not in text


def test_a_scheduled_task_keeps_its_label_and_failed_wording(tmp_path):
    import ghost_agent.core.autonomous_activity as AA
    log = _ledger(tmp_path)
    AA.record_scheduled_result(log, job_id="sched-abc", task_name="nightly", content="no data", ok=False)
    brief = AA.render_activity_brief(_records(log))
    assert "[scheduled task]" in brief and "'nightly' FAILED: no data" in brief


@pytest.mark.parametrize("reply,ok,said", [
    ("the follow-up reply", True, "done (exit 0) — the follow-up turn says"),
    ("partial notes\n\n_(Turn cancelled: this request reached its time limit (45 min).)_", False,
     "done (exit 0) — the follow-up turn was stopped at its time limit; it says"),
    ("partial notes\n\n_(Turn cancelled: stopped by the owner.)_", False,
     "done (exit 0) — the follow-up turn was stopped; it says"),
    ("Earlier a turn said this request reached its time limit; this one finished.", True,   # a quote
     "done (exit 0) — the follow-up turn says"),
])
def test_a_wake_stopped_by_its_clock_is_not_booked_ok(monkeypatch, reply, ok, said):
    from unittest.mock import MagicMock
    from ghost_agent import main as M
    import ghost_agent.tools.tasks as T
    import ghost_agent.core.autonomous_activity as AA
    rec = []

    async def _fg(ctx, body, rid, project_id=None):
        return reply, None, None
    monkeypatch.setattr(M, "_handle_chat_foreground", _fg)
    monkeypatch.setattr(T, "should_defer_scheduled_task", lambda *a, **k: False)
    monkeypatch.setattr(AA, "record_scheduled_result", lambda *a, **k: rec.append(k))
    M._RESUMED_JOBS.discard("job-0c10c4ed")
    assert asyncio.run(M._resume_after_job(MagicMock(sandbox_manager=None),
                                           {"id": "job-0c10c4ed", "state": "done", "exit_code": 0}))
    assert rec[0]["ok"] is ok and rec[0]["status"] == said


def test_the_wake_records_the_jobs_exit_as_status(monkeypatch):
    from unittest.mock import MagicMock
    from ghost_agent import main as M
    import ghost_agent.tools.tasks as T
    import ghost_agent.core.autonomous_activity as AA
    rec = []

    async def _fg(ctx, body, rid, project_id=None):
        return "the follow-up reply", None, None
    monkeypatch.setattr(M, "_handle_chat_foreground", _fg)
    monkeypatch.setattr(T, "should_defer_scheduled_task", lambda *a, **k: False)
    monkeypatch.setattr(AA, "record_scheduled_result", lambda *a, **k: rec.append(k))
    M._RESUMED_JOBS.discard("job-0badc0de")
    ctx = MagicMock(sandbox_manager=None)
    assert asyncio.run(M._resume_after_job(ctx, {"id": "job-0badc0de", "state": "done", "exit_code": 1}))
    assert rec and rec[0]["status"].startswith("done (exit 1)")


# ── the wake's clock ──────────────────────────────────────────────────

@pytest.mark.parametrize("rid,env,expect", [
    ("job-958c1ed3", None, 2700.0), ("sub-job-958c1ed3", None, 2700.0),
    ("sched-nightly", None, 0.0), ("chatcmpl-1", None, 0.0),
    ("job-958c1ed3", "600", 600.0), ("job-958c1ed3", "0", 0.0), ("job-958c1ed3", "inf", 0.0),
    ("job-958c1ed3", "junk", 2700.0),
])
def test_the_wake_deadline(monkeypatch, rid, env, expect):
    from ghost_agent import main as M
    if env is None:
        monkeypatch.delenv("GHOST_JOB_WAKE_S", raising=False)
    else:
        monkeypatch.setenv("GHOST_JOB_WAKE_S", env)
    assert M.unattended_turn_deadline_s(rid) == expect


@pytest.mark.parametrize("rid,expect", [("job-958c1ed3", 2700.0), ("sched-x", 0.0)])
def test_a_job_wake_runs_under_the_deadline_and_a_scheduled_task_does_not(monkeypatch, rid, expect):
    import ghost_agent.main as M
    from ghost_agent.utils.logging import client_deadline_context
    monkeypatch.delenv("GHOST_JOB_WAKE_S", raising=False)
    monkeypatch.setattr(M, "_project_turn_scope", lambda ctx, pid: (None, ""))
    seen = {}

    async def _hc(body, bg, request_id=None, **kw):
        seen["deadline"] = client_deadline_context.get()
        return "ok", None, None
    ctx = SimpleNamespace(agent=SimpleNamespace(handle_chat=_hc), llm_client=SimpleNamespace(foreground_requests=0))
    asyncio.run(M._handle_chat_foreground(ctx, {"messages": []}, rid))
    assert seen["deadline"] == expect
    assert client_deadline_context.get() == 0.0                  # restored after the turn


def test_the_wake_deadline_is_the_budget_in_force(monkeypatch):
    """The 2700 s reaches the loop's budget: job kinds have no server cap and no extension."""
    from ghost_agent.utils import logging as L
    monkeypatch.delenv("GHOST_JOB_WAKE_S", raising=False)
    monkeypatch.setenv("GHOST_MAX_REQUEST_EXTENDED_S", "inf")
    from ghost_agent import main as M
    tok = L.client_deadline_context.set(M.unattended_turn_deadline_s("job-958c1ed3"))
    try:
        for rid in ("job-958c1ed3", "sub-job-958c1ed3"):
            assert L.base_deadline_s(rid) == 2700.0
            assert L.extension_ceiling_s(rid) == 2700.0
    finally:
        L.client_deadline_context.reset(tok)
