"""§4OE-1 — the forced-final path of a LONG task: a report reserve sized from
the request's own replies, a system-written TASK STATE every turn, a STATE OF
WORK that survives a forced or cancelled turn, and a fallback that carries all
the evidence."""
import time

import pytest
from unittest.mock import AsyncMock

from ghost_agent.core import task_state as TS
from ghost_agent.core import agent as ag
from ghost_agent.core.agent import GhostAgent, effective_report_floor, evidence_digest
from ghost_agent.tools.outcome import ToolOutcome
from tests.helpers import FakeBgTasks, make_context
from tests.test_4ji_request_echo_narration import _resp, _tc


def _run(name, content, **args):
    o = ToolOutcome.ok(content) if hasattr(ToolOutcome, "ok") else content
    try:
        o.call_args = dict(args)
    except Exception:
        pass
    return {"name": name, "content": o}


def test_report_floor_grows_with_the_requests_own_reply_time():
    """Fails where the reserve was a constant 300 s while a report turn took
    240–253 s + a retry: 0 of 5 forced reports landed in time (§4OD)."""
    TS.forget_reply_seconds("r1")
    assert effective_report_floor(1800.0, "r1") == 300.0                 # nothing measured: the constant
    TS.note_reply_seconds("r1", 95.0); TS.note_reply_seconds("r1", 253.0); TS.note_reply_seconds("r1", 40.0)
    assert effective_report_floor(1800.0, "r1") == min(2 * 253.0 + 60.0, 450.0)   # capped at a quarter
    assert effective_report_floor(6000.0, "r1") == 2 * 253.0 + 60.0
    assert effective_report_floor(1800.0) == 300.0                        # no req_id: the old rule
    assert effective_report_floor(300.0, "r1") == 75.0                    # a short deadline keeps its first turn
    TS.forget_reply_seconds("r1")
    assert TS.max_reply_seconds("r1") == 0.0


def test_task_state_names_files_last_failure_and_counts():
    runs = [_run("file_system", "SUCCESS: Wrote 120 chars to 'app/main.py'. Script-side path (from sandbox cwd): 'app/main.py'.", path="app/main.py"),
            _run("execute", "Error: Traceback (most recent call last):\n  File 'x'\nValueError: boom", command="python app/main.py"),
            _run("web_search", "Results: 1. a\n2. b", query="q")]
    block = TS.render_task_state(runs, ask="build the app", turn=3)
    assert block.startswith("### TASK STATE") and "`app/main.py`" in block and "do not rewrite them from memory" in block
    assert "Last failure (1 so far): `execute` (python app/main.py): " in block and "ValueError: boom" in block
    assert "Tool calls so far: 3 over 3 turn(s)." in block
    assert TS.render_task_state([], ask="x") == ""                        # nothing ran: no block


def test_state_of_work_round_trips_through_the_scratchpad_and_expires(tmp_path):
    from ghost_agent.memory.scratchpad import Scratchpad
    ctx = make_context()
    ctx.scratchpad = Scratchpad()
    runs = [_run("file_system", "SUCCESS: Wrote 9 chars to 'notes.md'. Script-side path (from sandbox cwd): 'notes.md'.", path="notes.md")]
    text = TS.state_of_work_text(runs, ask="write notes", why="it ran out of its budget")
    assert "did not finish (it ran out of its budget)" in text and "`notes.md`" in text
    assert TS.persist_state_of_work(ctx.scratchpad, "sess-1", text)
    assert TS.recall_state_of_work(ctx.scratchpad, "sess-1") == text
    assert TS.recall_state_of_work(ctx.scratchpad, "sess-2") == ""
    assert "STATE OF WORK" in TS.render_state_of_work(text) and TS.render_state_of_work("") == ""
    ctx.scratchpad.set(TS._sow_key("sess-1"), {"text": text, "at": time.time() - TS.STATE_OF_WORK_TTL_S - 1}, namespace=None)
    assert TS.recall_state_of_work(ctx.scratchpad, "sess-1") == ""      # expired
    TS.clear_state_of_work(ctx.scratchpad, "sess-1")
    assert ctx.scratchpad.get(TS._sow_key("sess-1")) is None


def test_the_fallback_digest_carries_every_evidence_head_not_the_last_one():
    """Fails where 49 fetched pages were reduced to the last clean output,
    700 chars (probe-mx, §4OD)."""
    runs = [_run("browser", f"--- BROWSER RESULT ---\nSTATUS: OK\nPage {i} says something distinct number {i}.", url=f"https://ex.org/p{i}")
            for i in range(12)]
    runs.append(_run("browser", "--- BROWSER RESULT ---\nSTATUS: ERROR\nTimeout", url="https://ex.org/bad"))
    d = evidence_digest(runs, ask="research it")
    assert "Evidence gathered (newest first, heads only):" in d
    assert "distinct number 0" in d and "distinct number 11" in d and "ex.org/bad" not in d.split("Evidence gathered")[1].split("Tools run")[0]
    heads = TS.evidence_heads(runs, limit_chars=400)
    assert heads.count("\n- ") <= 3 and "distinct number 11" in heads   # bounded, newest first


def _scripted(monkeypatch, scripted, tmp_path):
    from ghost_agent.memory.scratchpad import Scratchpad
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    ctx.scratchpad = Scratchpad()
    agent = GhostAgent(ctx)
    agent.available_tools = {
        "web_search": AsyncMock(return_value="Results:\n1. https://ex.org/a — a page about it\n2. https://ex.org/b — another"),
        "file_system": AsyncMock(return_value="SUCCESS: Wrote 40 chars to 'notes.md'. Script-side path (from sandbox cwd): 'notes.md'."),
    }
    ctx.llm_client.chat_completion = AsyncMock(side_effect=scripted)
    return agent, ctx


def _payload_msgs(c):
    p = c.kwargs.get("messages")
    if p is None and c.args:
        p = c.args[0].get("messages") if isinstance(c.args[0], dict) else c.args[0]
    return p or []


def _payload(c):
    return c.kwargs if c.kwargs.get("messages") is not None else (c.args[0] if c.args and isinstance(c.args[0], dict) else {})


async def test_the_task_state_block_rides_every_turn_after_a_tool_ran(monkeypatch, tmp_path):
    """Fails where a 30-call turn carried no plan: only 200-char checkpoint
    heads at turns 15/30 (§4OD) — the TASK STATE block names the files and
    the last failure from turn 2 on."""
    agent, ctx = _scripted(monkeypatch, [
        _resp("writing", [_tc("c0", "file_system", {"operation": "write", "path": "notes.md", "content": "x"})]),
        _resp("searching", [_tc("c1", "web_search", {"query": "q"})]),
        _resp("Done: notes.md holds the notes."),
    ], tmp_path)
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "write notes.md then search"}],
                                         "session_id": "sess-ts"}, FakeBgTasks())
    calls = ctx.llm_client.chat_completion.call_args_list
    first = "\n".join(str(m.get("content")) for m in _payload_msgs(calls[0]))
    later = "\n".join(str(m.get("content")) for m in _payload_msgs(calls[1]))
    assert "### TASK STATE" not in first                                   # nothing ran yet
    assert "### TASK STATE" in later and "`notes.md`" in later and "file_system ×1" in later
    assert ctx.scratchpad.get(TS._sow_key("sess-ts")) is None             # a finished turn leaves no state of work


async def test_a_budget_forced_final_persists_the_state_of_work_and_the_next_turn_sees_it(monkeypatch, tmp_path):
    """Fails where a forced final left nothing a later turn could resume
    from: sessions keep only user+assistant text (§4OD)."""
    agent, ctx = _scripted(monkeypatch, [
        _resp("step", [_tc("c0", "file_system", {"operation": "write", "path": "notes.md", "content": "x"})]),
        _resp("step", [_tc("c1", "web_search", {"query": "q1"})]),
        _resp("step", [_tc("c2", "web_search", {"query": "q2"})]),
        _resp("Where things stand: notes.md written; searches done."),      # the forced report
        _resp("(unreachable)"),
    ], tmp_path)
    agent.max_turns_override = 5                                          # turns 4–5 are the report
    ctx.args.tool_turn_max_tokens = 20000                                 # the cap must be the ONLY thing lowering it
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "do the long thing"}],
                                         "session_id": "sess-sow"}, FakeBgTasks())
    stored = ctx.scratchpad.get(TS._sow_key("sess-sow"))
    assert isinstance(stored, dict) and "did not finish (it ran out of its budget)" in stored["text"] and "`notes.md`" in stored["text"]
    # the forced report turn is capped at 8k tokens; the tool turns before it were not
    calls = ctx.llm_client.chat_completion.call_args_list
    assert _payload(calls[0]).get("max_tokens") == 20000
    assert _payload(calls[3]).get("max_tokens", 10**9) <= TS.FORCED_REPORT_MAX_TOKENS
    # next turn of the same session: the state of work is shown once, then cleared
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[_resp("Continuing — notes.md is there.")])
    out2, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "continue"}],
                                          "session_id": "sess-sow"}, FakeBgTasks())
    shown = "\n".join(str(m.get("content")) for m in _payload_msgs(ctx.llm_client.chat_completion.call_args_list[0]))
    assert "### STATE OF WORK" in shown and "`notes.md`" in shown
    assert ctx.scratchpad.get(TS._sow_key("sess-sow")) is None
    # a different session never sees another session's note
    TS.persist_state_of_work(ctx.scratchpad, "sess-sow", "The previous turn of this conversation did not finish (x).")
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[_resp("hi")])
    await agent.handle_chat({"messages": [{"role": "user", "content": "hello"}], "session_id": "other"}, FakeBgTasks())
    assert "STATE OF WORK" not in "\n".join(str(m.get("content")) for m in _payload_msgs(ctx.llm_client.chat_completion.call_args_list[0]))
    assert ctx.scratchpad.get(TS._sow_key("sess-sow")) is not None       # and does not clear it


def test_a_declared_failure_with_neutral_prose_is_a_failure_not_evidence():
    """Review: `str()` on a ToolOutcome dropped its status — a declared
    failure with neutral prose vanished from the last-failure line and was
    listed as evidence."""
    bad = ToolOutcome.failed("The page could not be fetched.")
    runs = [{"name": "browser", "content": bad}, _run("execute", "=== banner ===\n" + "x " * 300 + "\nFINAL: 12 passed", command="pytest")]
    facts = TS.task_state_facts(runs)
    assert facts["failures"] == 1 and facts["last_failure"]["tool"] == "browser"
    heads = TS.evidence_heads(runs)
    assert "could not be fetched" not in heads and "FINAL: 12 passed" in heads   # execute shows its TAIL


async def test_every_reply_wall_time_is_recorded_for_the_reserve(monkeypatch, tmp_path):
    """Fails where the thought-log site did not record the reply's wall time —
    the sized reserve would then always be the constant."""
    seen = []
    monkeypatch.setattr(TS, "note_reply_seconds", lambda rid, s: seen.append((str(rid), float(s))))
    agent, ctx = _scripted(monkeypatch, [
        _resp("step", [_tc("c0", "web_search", {"query": "q1"})]),
        _resp("Done."),
    ], tmp_path)
    await agent.handle_chat({"messages": [{"role": "user", "content": "look it up"}], "session_id": "sess-rt"},
                            FakeBgTasks())
    assert len(seen) >= 2 and all(rid and s >= 0.0 for rid, s in seen) and len({rid for rid, _ in seen}) == 1


async def test_a_cancelled_turn_keeps_its_state_of_work_too(monkeypatch, tmp_path):
    """Fails where only the budget-forced path persisted the state of work —
    a Stop pressed mid-task (the shape of 1 of 4 long owner turns, §4OD)
    left nothing to resume from."""
    from ghost_agent.core.turns import get_turn_registry
    agent, ctx = _scripted(monkeypatch, [
        _resp("step", [_tc("c0", "file_system", {"operation": "write", "path": "notes.md", "content": "x"})]),
        _resp("step", [_tc("c1", "web_search", {"query": "q"})]),
        _resp("(unreachable)"),
    ], tmp_path)

    async def _search_then_stop(**kw):                                    # the Stop button lands mid-turn
        get_turn_registry(agent).cancel("rid-cancel")
        return "Results:\n1. https://ex.org/a — a page"
    agent.available_tools["web_search"] = _search_then_stop
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "do the long thing"}],
                                         "session_id": "sess-cancel"}, FakeBgTasks(), request_id="rid-cancel")
    stored = ctx.scratchpad.get(TS._sow_key("sess-cancel"))
    assert isinstance(stored, dict) and "did not finish (the turn was cancelled)" in stored["text"] and "`notes.md`" in stored["text"]


async def test_a_members_forced_turn_persists_no_state_of_work(monkeypatch, tmp_path):
    """A channel member's unfinished turn must leave nothing server-side."""
    monkeypatch.setattr(ag, "requester_is_member", lambda: True)
    agent, ctx = _scripted(monkeypatch, [
        _resp("step", [_tc("c0", "file_system", {"operation": "write", "path": "notes.md", "content": "x"})]),
        _resp("step", [_tc("c1", "web_search", {"query": "q1"})]),
        _resp("step", [_tc("c2", "web_search", {"query": "q2"})]),
        _resp("Where things stand."),
        _resp("(unreachable)"),
    ], tmp_path)
    agent.max_turns_override = 5
    await agent.handle_chat({"messages": [{"role": "user", "content": "do the long thing"}],
                             "session_id": "sess-member"}, FakeBgTasks())
    assert ctx.scratchpad.get(TS._sow_key("sess-member")) is None


def _build_loop(n_writes, run_outputs):
    """A scripted rewrite/rerun cycle on app.py: write, run, write, run …"""
    steps = []
    runs = iter(run_outputs)
    for i in range(n_writes):
        steps.append(_resp(f"edit {i}", [_tc(f"w{i}", "file_system", {"operation": "write", "path": "app.py", "content": f"v{i}"})]))
        steps.append(_resp(f"run {i}", [_tc(f"r{i}", "execute", {"command": "python -m pytest -q app.py"})]))
    steps.append(_resp("Done: the tests pass now."))
    steps.append(_resp("(unreachable)"))
    return steps


async def _run_build(monkeypatch, tmp_path, run_outputs):
    agent, ctx = _scripted(monkeypatch, _build_loop(len(run_outputs), run_outputs), tmp_path)
    agent.available_tools["execute"] = AsyncMock(side_effect=list(run_outputs))
    agent.max_turns_override = 40
    await agent.handle_chat({"messages": [{"role": "user", "content": "make the tests pass"}],
                             "session_id": "sess-build"}, FakeBgTasks())
    return agent, ctx


async def test_the_futility_breaker_lets_a_converging_test_fix_loop_run(monkeypatch, tmp_path):
    """Fails where 6 rewrites + 4 reruns forced a blocker report although
    every run passed more tests (§4OD)."""
    falling = [f"=== {n} failed, {12 - n} passed in 0.1s ===" for n in (9, 8, 7, 6, 5, 4, 3)]
    agent, ctx = await _run_build(monkeypatch, tmp_path, falling)
    assert not getattr(agent.context, "_futility_report_done", False)
    assert not getattr(agent.context, "_breaker_forced_final", False)   # the converging loop ran to its own end
    assert ctx.llm_client.chat_completion.await_count == 2 * len(falling) + 1
    # the same cycle with NO progress: §4UI — the futility breaker steers once and no
    # longer forces a report; the progress flag is False (what the steer reads)
    # production shape: a failing pytest EXITS 1 → the tool result is a declared failure
    stuck = [ToolOutcome.failed(f"=== 9 failed, 3 passed in 0.{i}s ===") for i in range(1, 8)]
    agent2, ctx2 = await _run_build(monkeypatch, tmp_path, stuck)
    assert not getattr(agent2.context, "_futility_report_done", False)
    rec = (getattr(agent2.context, "_script_iter", None) or {}).get("app.py") or {}
    assert rec.get("progress") is False and rec.get("fails") == 9


# ── §4OE-1b: the report written as a file on the report turn ──────────────── #

_FALLBACK_HEAD = "ran out of this turn's budget"


def _write(cid, path):
    return _tc(cid, "file_system", {"operation": "write", "path": path, "content": "# report\n\nfindings"})


def _fs_by_path(refuse=()):
    """A file_system mock whose result names the path it was given (the
    ledger keys on the RESULT, review MAJOR 2); ``refuse`` paths are rejected."""
    async def _fs(**kw):
        path = kw.get("path")
        if path in refuse:
            return f"REJECTED: '{path}' already exists (900 bytes) and this request has not read it. Nothing was changed."
        return f"SUCCESS: Wrote 40 chars to '{path}'. Script-side path (from sandbox cwd): '{path}'."
    return AsyncMock(side_effect=_fs)


def _six_turn_script(report_turn, answer_turn):
    return [
        _resp("step", [_write("c0", "notes.md")]),
        _resp("step", [_tc("c1", "web_search", {"query": "q1"})]),
        _resp("step", [_tc("c2", "web_search", {"query": "q2"})]),
        _resp("step", [_tc("c3", "web_search", {"query": "q3"})]),
        report_turn, answer_turn, _resp("(unreachable)"),
    ]


async def test_a_report_written_as_a_file_on_the_report_turn_runs_and_the_next_turn_links_it(monkeypatch, tmp_path):
    """Fails where the final-generation drop ate the write and the fallback
    shipped at 457 s (tight §4OE probe): the write RUNS, the next turn is told
    the file exists and answers with its link."""
    agent, ctx = _scripted(monkeypatch, [
        _resp("step", [_write("c0", "notes.md")]),
        _resp("step", [_tc("c1", "web_search", {"query": "q1"})]),
        _resp("step", [_tc("c2", "web_search", {"query": "q2"})]),
        _resp("step", [_tc("c3", "web_search", {"query": "q3"})]),
        _resp("I now have enough data. Let me compile the findings into the report.", [_write("c4", "report.md")]),   # the forced report turn: a file
        _resp("The report is in report.md: four servers compared. Download: /api/download/report.md"),
        _resp("(unreachable)"),
    ], tmp_path)
    agent.max_turns_override = 6                                          # turns 5–6 are the report
    agent.available_tools["file_system"] = _fs_by_path()
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "research the four servers"}],
                                         "session_id": "sess-rw"}, FakeBgTasks())
    fs = agent.available_tools["file_system"]
    assert fs.await_count == 2 and fs.await_args_list[-1].kwargs.get("path", fs.await_args_list[-1].args[0] if fs.await_args_list[-1].args else None) in ("report.md", None)
    assert any("report.md" in str(c) for c in fs.await_args_list)        # the report write ran
    calls = ctx.llm_client.chat_completion.call_args_list
    assert len(calls) == 6
    last = "\n".join(str(m.get("content")) for m in _payload_msgs(calls[5]))
    assert "SYSTEM ALERT (report written)" in last and "/api/download/report.md" in last
    assert "### TASK STATE" in last and "`report.md`" in last           # the answer turn sees the file in the ledger
    assert "SYSTEM ALERT: this is the FINAL turn" not in last           # not the no-answer retry
    assert "report.md" in str(out) and _FALLBACK_HEAD not in str(out)
    assert "Let me compile" not in str(out)                              # the write turn's narration never ships (review MAJOR 3)
    assert getattr(agent.context, "_report_write_used", False) is True
    assert getattr(agent.context, "_report_write_paths", None) is None


async def test_a_report_turn_that_writes_AND_calls_another_tool_is_still_dropped(monkeypatch, tmp_path):
    """Fails where the allowance opened the whole tool set on the report
    turn: a write beside a search is work, not the report — dropped, and the
    no-answer retry runs as before."""
    agent, ctx = _scripted(monkeypatch, [
        _resp("step", [_write("c0", "notes.md")]),
        _resp("step", [_tc("c1", "web_search", {"query": "q1"})]),
        _resp("step", [_tc("c2", "web_search", {"query": "q2"})]),
        _resp("step", [_tc("c3", "web_search", {"query": "q3"})]),
        _resp("Let me finish.", [_write("c4", "report.md"), _tc("c5", "web_search", {"query": "q4"})]),
        _resp("Here is the report in text: four servers compared, sources ex.org/a and ex.org/b."),
        _resp("(unreachable)"),
    ], tmp_path)
    agent.max_turns_override = 6
    agent.available_tools["file_system"] = _fs_by_path()
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "research the four servers"}],
                                         "session_id": "sess-rw2"}, FakeBgTasks())
    assert agent.available_tools["file_system"].await_count == 1           # notes.md only
    calls = ctx.llm_client.chat_completion.call_args_list
    last = "\n".join(str(m.get("content")) for m in _payload_msgs(calls[5]))
    assert "SYSTEM ALERT: this is the FINAL turn" in last and "(report written)" not in last
    assert not getattr(agent.context, "_report_write_used", False)


async def test_the_allowance_is_once_per_request_and_never_on_the_last_turn(monkeypatch, tmp_path):
    """Fails where a model that keeps writing files on report turns could
    spend the whole reserve on writes: the second write is dropped (and the
    last turn never runs one — nothing could answer after it)."""
    agent, ctx = _scripted(monkeypatch, [
        _resp("step", [_write("c0", "notes.md")]),
        _resp("step", [_tc("c1", "web_search", {"query": "q1"})]),
        _resp("step", [_tc("c2", "web_search", {"query": "q2"})]),
        _resp("step", [_tc("c3", "web_search", {"query": "q3"})]),
        _resp("", [_write("c4", "a.md")]),                                    # allowed (turn 5 of 6)
        _resp("", [_write("c5", "b.md")]),                                    # last turn: dropped
        _resp("(unreachable)"),
    ], tmp_path)
    agent.max_turns_override = 6
    agent.available_tools["file_system"] = _fs_by_path()
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "research it"}],
                                         "session_id": "sess-rw3"}, FakeBgTasks())
    fs = agent.available_tools["file_system"]
    assert fs.await_count == 2 and not any("b.md" in str(c) for c in fs.await_args_list)
    assert len(ctx.llm_client.chat_completion.call_args_list) == 6


def test_the_report_write_allowance_rules():
    from types import SimpleNamespace as NS
    from ghost_agent.core.agent import _report_write_allowed, report_written_alert, forced_final_answer_directive
    w = [_tc("c", "file_system", {"operation": "write", "path": "r.md", "content": "x"})]
    ap = [_tc("c", "file_system", {"operation": "append", "path": "r.md", "content": "x"})]
    ed = [_tc("c", "file_system", {"operation": "edit", "path": "r.md", "old_string": "a", "new_string": "b"})]
    armed = lambda **k: NS(_breaker_forced_final=True, **k)
    assert _report_write_allowed(w, armed(), turn=4, max_turns=6)
    assert _report_write_allowed(ap, armed(), turn=4, max_turns=6)
    assert not _report_write_allowed(ed, armed(), turn=4, max_turns=6)                 # an edit is not a report
    assert not _report_write_allowed(w, armed(), turn=5, max_turns=6)                  # nothing could answer after it
    assert not _report_write_allowed(w, armed(_report_write_used=True), turn=4, max_turns=6)
    assert not _report_write_allowed(w, NS(_breaker_forced_final=False), turn=4, max_turns=6)   # a planner final, not a breaker
    assert not _report_write_allowed(w + [_tc("d", "web_search", {"query": "q"})], armed(), turn=4, max_turns=6)
    assert not _report_write_allowed([], armed(), turn=4, max_turns=6)
    a = report_written_alert(["out/r.md"], "research it")
    assert "(report written)" in a and "/api/download/out/r.md" in a and "Tools are OFF" in a and '"research it"' in a
    d = forced_final_answer_directive([], "q", file_report=True)
    assert "NOT written" in d and "put the report itself in your reply text" in d
    assert "NOT written" not in forced_final_answer_directive([], "q")


def test_the_fallback_digest_links_the_files_it_names():
    runs = [_run("file_system", "SUCCESS: Wrote 9 chars to 'out/r.md'. Script-side path (from sandbox cwd): 'out/r.md'.", path="out/r.md")]
    d = evidence_digest(runs, ask="x")
    assert "`out/r.md` (/api/download/out/r.md)" in d


async def test_a_refused_report_write_gets_the_answer_now_directive_not_a_link(monkeypatch, tmp_path):
    """Fails where the answer turn was told "the file exists" from the CALL
    (review MAJOR 2): the receipts gate refused the write, so the next turn
    gets the plain answer-now directive and no link to a missing file."""
    agent, ctx = _scripted(monkeypatch, _six_turn_script(
        _resp("", [_write("c4", "report.md")]),
        _resp("Here is the report in text: four servers compared, sources ex.org/a and ex.org/b."),
    ), tmp_path)
    agent.max_turns_override = 6
    agent.available_tools["file_system"] = _fs_by_path(refuse=("report.md",))
    out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "research it"}],
                                         "session_id": "sess-rw4"}, FakeBgTasks())
    calls = ctx.llm_client.chat_completion.call_args_list
    last = "\n".join(str(m.get("content")) for m in _payload_msgs(calls[5]))
    assert "SYSTEM ALERT: this is the FINAL turn" in last and "`file_system`" in last and "(report written)" not in last
    assert "/api/download/report.md" not in str(out)


async def test_the_allowance_resets_per_request_on_the_shared_context(monkeypatch, tmp_path):
    """Fails where `_report_write_used` / `_report_write_paths` were process-
    sticky (review CRIT 1): the allowance fired once per deploy, and a stale
    path could force the NEXT request's first turn with another request's
    file."""
    agent, ctx = _scripted(monkeypatch, _six_turn_script(
        _resp("", [_write("c4", "report.md")]),
        _resp("The report is in report.md: /api/download/report.md"),
    ), tmp_path)
    agent.max_turns_override = 6
    agent.available_tools["file_system"] = _fs_by_path()
    await agent.handle_chat({"messages": [{"role": "user", "content": "research it"}], "session_id": "s1"}, FakeBgTasks())
    assert getattr(agent.context, "_report_write_used", False) is True
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_six_turn_script(
        _resp("", [_write("d4", "report2.md")]),
        _resp("The report is in report2.md: /api/download/report2.md"),
    ))
    out2, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "research it again"}], "session_id": "s2"}, FakeBgTasks())
    first = "\n".join(str(m.get("content")) for m in _payload_msgs(ctx.llm_client.chat_completion.call_args_list[0]))
    assert "(report written)" not in first                              # no stale path forces turn 0
    last = "\n".join(str(m.get("content")) for m in _payload_msgs(ctx.llm_client.chat_completion.call_args_list[5]))
    assert "(report written)" in last and "/api/download/report2.md" in last   # the allowance fired again
    assert any("report2.md" in str(c) for c in agent.available_tools["file_system"].await_args_list)
