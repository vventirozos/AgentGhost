"""§4EE pins — the label producers as a unit (driven).

Each pin names the world where the fix decides: the operator line for a
shape-heuristic failure, a verified turn that exhausted its budget, a late
verdict on a turn the corpus will never upgrade. (§4PJ: the pins on a
human thumb reaching the calibration fit went with the calibration corpus.)
"""
from __future__ import annotations

import itertools
from collections import OrderedDict

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.distill.outcome_heuristics import (
    resolve_turn_outcome, STRUCTURAL_FAILURE_REASON)
from ghost_agent.distill.schema import Outcome
from tests.test_finalize_stream_pins import _fs, make_fin_agent
from tests.test_outcome_consolidation import _FakeCollector

L = GhostAgent._turn_outcome_label


# ── the operator label IS the corpus ladder, over every input ─────────── #

CURRENTS = [("unknown", ""), ("failed", STRUCTURAL_FAILURE_REASON),
            ("failed", STRUCTURAL_FAILURE_REASON + ": execute: EXIT CODE: 1"),
            ("failed", "browser selector 'x' used 4× in one turn"),
            ("failed", "verifier refuted")]


@pytest.mark.parametrize("verifier,exec_terminal,unacked,budget,current",
                         list(itertools.product([None, "passed", "failed"],
                                                [False, True], [False, True],
                                                [False, True], CURRENTS)),
                         ids=lambda v: str(v)[:12])
def test_operator_label_valence_equals_the_corpus_label(
        verifier, exec_terminal, unacked, budget, current):
    cur, reason = current
    if unacked and not exec_terminal:
        pytest.skip("finalize couples them: unacked ⇒ terminal execution failure")
    from ghost_agent.distill.outcome_heuristics import is_structural_reason
    if cur == "failed" and is_structural_reason(reason) and not exec_terminal:
        pytest.skip("a structural FAILED is stamped only when execution failed "
                    "(rule 4), so this cell does not exist")
    corpus = resolve_turn_outcome(current=cur, verifier=verifier,
                                  execution_failed=exec_terminal,
                                  current_reason=reason,
                                  unacked_total_failure=unacked)
    # the production derivation, not a private copy of the rule
    import types
    shape_failed = GhostAgent._row_shape_failed(
        types.SimpleNamespace(outcome=cur, failure_reason=reason))
    line = L(verifier_failed=(verifier == "failed"),
             verifier_passed=(verifier == "passed"),
             budget_exhausted=budget, exec_terminal=exec_terminal,
             unacked_total_failure=unacked, shape_failed=shape_failed)
    if corpus == Outcome.PASSED.value:
        assert line == "verified", (corpus, line)
    elif corpus == Outcome.FAILED.value:
        assert line == "failed", (corpus, line)
    else:
        assert line == ("partial (budget exhausted)" if budget else "ok"), (corpus, line)


# ── the late correction carries the shape verdict ─────────────────────── #

def _agent_with_ring(snap, tid="t-late"):
    agent = GhostAgent.__new__(GhostAgent)
    import types
    agent.context = types.SimpleNamespace(_recent_turn_outcome={tid: snap})
    return agent, tid


def _snap(**over):
    base = {"state": "failed", "confidence": 0.9, "tools": ["execute"],
            "chars": 4, "exec_failures": 1, "exec_terminal": True,
            "budget_exhausted": False, "unacked_total_failure": False,
            "shape_failed": False}
    base.update(over)
    return base


def test_late_pass_on_a_shape_failed_turn_stays_silent(capsys):
    """The corpus keeps FAILED for a selector-thrash turn whatever the
    verifier says (rule 2); the stream must not announce a recovery the
    corpus does not record."""
    agent, tid = _agent_with_ring(_snap(shape_failed=True))
    agent._emit_late_outcome_correction(tid, "passed")
    assert "CORRECTED" not in capsys.readouterr().out


def test_late_pass_on_a_structural_failure_still_corrects(capsys):
    agent, tid = _agent_with_ring(_snap(shape_failed=False))
    agent._emit_late_outcome_correction(tid, "passed")
    assert "CORRECTED failed → verified" in capsys.readouterr().out


# ── finalize, driven: the line reads the row it just wrote ────────────── #

def _tool_msgs(n, content):
    calls = [{"id": f"c{i}", "type": "function",
              "function": {"name": "execute", "arguments": "{}"}} for i in range(n)]
    msgs = [{"role": "user", "content": "run it"},
            {"role": "assistant", "content": "", "tool_calls": calls}]
    for i in range(n):
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "execute",
                     "content": content})
    return msgs


def _fin_agent():
    a = make_fin_agent()
    a.context.verifier = None
    a.context.trajectory_collector = _FakeCollector()
    a.context._recent_trajectories_for_correction = OrderedDict()
    a.context._recent_turn_outcome = OrderedDict()   # a MagicMock ring swallows the snapshot
    a.context.trajectory_task_kind = "user_request"
    a.context.trajectory_user_request_override = None
    a.context.trajectory_extra_static = None
    a.context.self_model = None
    a.context.memory_dir = None
    return a


@pytest.mark.asyncio
async def test_a_repeated_error_turn_prints_failed_even_with_a_clean_ledger(capsys):
    """Three identical execute errors → the corpus row is FAILED by the
    shape heuristics. The strike ledger says nothing terminal (the last
    call is not marked a failure), so before §4EE the line said "ok"."""
    a = _fin_agent()
    msgs = _tool_msgs(3, "Error: ENOENT no such file")
    tools = [{"name": "execute", "content": "Error: ENOENT no such file"}] * 3
    await a._finalize_and_return(_fs(
        messages=msgs, tools_run_this_turn=tools, final_ai_content="I fixed it.",
        execution_failure_count=0, last_was_failure=False,
        current_trajectory_id="t-shape"))
    col = a.context.trajectory_collector
    assert col.appended and col.appended[-1].outcome == Outcome.FAILED.value, \
        [(t.outcome, t.failure_reason) for t in col.appended]
    out = capsys.readouterr().out.lower()
    assert "turn outcome" in out and "failed" in out.split("turn outcome", 1)[1][:80], out
    ring = a.context._recent_turn_outcome
    assert ring and list(ring.values())[-1]["shape_failed"] is True, dict(ring)


@pytest.mark.asyncio
async def test_a_terminal_structural_failure_prints_failed_and_is_upgradable(capsys):
    a = _fin_agent()
    msgs = _tool_msgs(1, "Error: EXIT CODE: 1")
    tools = [{"name": "execute", "content": "Error: EXIT CODE: 1"}]
    await a._finalize_and_return(_fs(
        messages=msgs, tools_run_this_turn=tools,
        final_ai_content="The command failed with exit code 1.",
        execution_failure_count=1, last_was_failure=True,
        current_trajectory_id="t-struct"))
    col = a.context.trajectory_collector
    assert col.appended[-1].outcome == Outcome.FAILED.value
    assert col.appended[-1].failure_reason.startswith(STRUCTURAL_FAILURE_REASON)
    ring = a.context._recent_turn_outcome
    assert list(ring.values())[-1]["shape_failed"] is False
    assert list(ring.values())[-1]["state"] == "failed"


@pytest.mark.asyncio
async def test_unacknowledged_needs_a_terminal_strike_not_just_a_sniffer(capsys):
    """The sniffer sees every tool result as an error, but the ledger is
    clean: finalize must NOT call the turn unacknowledged (the coupling
    the mirror table relies on), so the corpus stays UNKNOWN and the
    line reads ok."""
    a = _fin_agent()
    msgs = _tool_msgs(1, "Error: nested banner inside a successful payload")
    tools = [{"name": "execute", "content": "Error: nested banner inside a successful payload"}]
    await a._finalize_and_return(_fs(
        messages=msgs, tools_run_this_turn=tools, final_ai_content="42",
        execution_failure_count=0, last_was_failure=False,
        current_trajectory_id="t-sniff"))
    ring = a.context._recent_turn_outcome
    snap = list(ring.values())[-1]
    assert snap["unacked_total_failure"] is False and snap["exec_terminal"] is False
    assert snap["state"] == "ok", snap


# ── calibration: the human label reaches the fit ───────────────────────── #

# ── the feedback route wires the tracker ──────────────────────────────── #

# ── the late write updates the in-process row the next turn will read ─── #

def _cached(tid="t-c", outcome="unknown", reason=""):
    import types
    return types.SimpleNamespace(id=tid, outcome=outcome, failure_reason=reason,
                                 extra={}, tool_calls=[], final_response="x",
                                 user_request="q")


def test_late_refute_writes_its_reason_into_the_cached_row():
    from tests.test_selfhood_late_verdict_backfill import _fake_agent, _run_late
    cached = _cached()
    fake, writes = _fake_agent(True, cached, None)
    _run_late(fake, "t-c", "failed", reason="verifier refuted: wrong total")
    assert writes and writes[-1][1] == "failed"
    assert cached.outcome == "failed"
    assert cached.failure_reason == "verifier refuted: wrong total"


def test_late_pass_clears_the_cached_reason():
    from tests.test_selfhood_late_verdict_backfill import _fake_agent, _run_late
    cached = _cached(outcome="failed", reason=STRUCTURAL_FAILURE_REASON)
    fake, writes = _fake_agent(True, cached, None)
    _run_late(fake, "t-c", "passed")
    assert cached.outcome == "passed" and cached.failure_reason == ""


# ── fix-round survivors: the emitter's own edges ───────────────────────── #

def test_recorded_shape_failed_looks_up_by_id_not_by_position():
    import types
    agent = GhostAgent.__new__(GhostAgent)
    other = types.SimpleNamespace(id="other", outcome="failed",
                                  failure_reason="browser selector thrash")
    agent.context = types.SimpleNamespace(
        _recent_trajectories_for_correction=OrderedDict([("fp1", other)]))
    assert agent._recorded_shape_failed("mine") is False
    assert agent._recorded_shape_failed("other") is True
    assert agent._recorded_shape_failed("") is False
    other.failure_reason = STRUCTURAL_FAILURE_REASON + ": execute: boom"
    assert agent._recorded_shape_failed("other") is False


def test_a_bogus_verdict_tag_never_prints_a_correction(capsys):
    agent, tid = _agent_with_ring(_snap(state="failed", exec_terminal=False))
    agent._emit_late_outcome_correction(tid, "bogus")
    agent._emit_late_outcome_correction(tid, None)
    assert "CORRECTED" not in capsys.readouterr().out


def _capture_log(monkeypatch):
    """pretty_log truncates the console line; the durable mirror gets the
    whole message. Capture the message itself."""
    from ghost_agent.core import agent as A
    seen = []
    monkeypatch.setattr(A, "pretty_log",
                        lambda title, msg, **kw: seen.append((title, msg)))
    return seen


@pytest.mark.parametrize("exec_terminal,fails,conf,expect,absent", [
    (True, 2, 0.91, "2 tool failure(s), honestly reported", "recovered"),
    (False, 2, None, "recovered 2 strike(s)", "confidence"),
    # §4PJ: a ring entry's stale `confidence` (rows written before the
    # cut) is never printed — the score is gone from the line
    (True, 0, 0.5, "· 4 chars", "confidence"),
    (True, 0, None, "· 4 chars", "budget"),          # no budget note unless exhausted
], ids=["honest", "recovered", "no-fails", "no-budget-note"])
def test_correction_note_names_what_happened(monkeypatch, exec_terminal, fails,
                                             conf, expect, absent):
    seen = _capture_log(monkeypatch)
    agent, tid = _agent_with_ring(_snap(state="failed", exec_terminal=exec_terminal,
                                        exec_failures=fails, confidence=conf))
    agent._emit_late_outcome_correction(tid, "passed")
    msgs = [m for t, m in seen if t == "Turn Outcome"]
    assert msgs and "CORRECTED failed → verified" in msgs[-1], seen
    assert expect in msgs[-1] and absent not in msgs[-1], msgs[-1]
    # …and exactly once: a second identical verdict re-announces nothing
    agent._emit_late_outcome_correction(tid, "passed")
    assert len([m for t, m in seen if t == "Turn Outcome"]) == 1


def test_a_refute_correction_carries_no_recovery_note(monkeypatch):
    seen = _capture_log(monkeypatch)
    agent, tid = _agent_with_ring(_snap(state="ok", exec_terminal=False, exec_failures=1))
    agent._emit_late_outcome_correction(tid, "failed")
    msg = [m for t, m in seen if t == "Turn Outcome"][-1]
    assert "CORRECTED ok → failed" in msg and "recovered" not in msg and "honestly" not in msg


# ── round 2: the third mirror, the row source, the budget note ────────── #

@pytest.mark.asyncio
async def test_a_bench_row_still_drives_the_line_from_the_returned_row(monkeypatch):
    """Bench rows are never stashed in the correction cache; the line must
    read the row the recorder RETURNED."""
    seen = []
    from ghost_agent.core import agent as A
    monkeypatch.setattr(A, "pretty_log", lambda t, m, **kw: seen.append((t, m)))
    a = _fin_agent(); a.context.trajectory_task_kind = "bench"
    msgs = _tool_msgs(3, "Error: ENOENT no such file")
    tools = [{"name": "execute", "content": "Error: ENOENT no such file"}] * 3
    await a._finalize_and_return(_fs(messages=msgs, tools_run_this_turn=tools,
                                     final_ai_content="I fixed it.",
                                     execution_failure_count=0, last_was_failure=False,
                                     current_trajectory_id="t-bench"))
    assert a.context._recent_trajectories_for_correction == {}   # not stashed
    line = [m for t, m in seen if t == "Turn Outcome"][-1]
    assert line.startswith("failed"), line


@pytest.mark.asyncio
async def test_budget_exhaustion_is_a_note_at_warning_not_a_valence(monkeypatch):
    seen = []
    from ghost_agent.core import agent as A
    monkeypatch.setattr(A, "pretty_log", lambda t, m, **kw: seen.append((t, m, kw)))
    a = _fin_agent()
    await a._finalize_and_return(_fs(messages=_tool_msgs(1, "ok"),
                                     tools_run_this_turn=[{"name": "execute", "content": "ok"}],
                                     final_ai_content="[TURN BUDGET EXHAUSTED] partial work",
                                     turn_budget_exhausted=True,
                                     current_trajectory_id="t-budget"))
    t, m, kw = [x for x in seen if x[0] == "Turn Outcome"][-1]
    assert m.startswith("partial (budget exhausted)") and kw.get("level") == "WARNING", (m, kw)
    # …and a late PASS keeps the note while correcting nothing loud
    agent, tid = _agent_with_ring(_snap(state="failed", budget_exhausted=True))
    seen.clear()
    agent._emit_late_outcome_correction(tid, "passed")
    msg = [m for t, m, kw in seen if t == "Turn Outcome"][-1]
    assert "CORRECTED failed → verified" in msg and "budget exhausted" in msg


# ── the late verdict re-labels calibration through the cached row's req_id ─ #

# ── round-2 survivors: the emitter's text and severity, the tiers' joins ─ #

def _capture_kw(monkeypatch):
    from ghost_agent.core import agent as A
    seen = []
    monkeypatch.setattr(A, "pretty_log",
                        lambda title, msg, **kw: seen.append((title, msg, kw)))
    return seen


@pytest.mark.parametrize("fails,exec_terminal,present,absent", [
    (1, True, "1 tool failure(s), honestly reported", "strike"),
    (1, False, "recovered 1 strike(s)", "honestly"),
    (0, True, "· 4 chars", "honestly"),
    (0, False, "· 4 chars", "strike"),
], ids=["one-honest", "one-recovered", "zero-terminal", "zero-clean"])
def test_correction_note_counts_from_zero(monkeypatch, fails, exec_terminal, present, absent):
    seen = _capture_kw(monkeypatch)
    agent, tid = _agent_with_ring(_snap(state="failed", exec_failures=fails,
                                        exec_terminal=exec_terminal, confidence=None))
    agent._emit_late_outcome_correction(tid, "passed")
    msg = [m for t, m, kw in seen if t == "Turn Outcome"][-1]
    assert present in msg and absent not in msg and "budget" not in msg, msg


def test_correction_names_the_tools_or_says_no_tools(monkeypatch):
    seen = _capture_kw(monkeypatch)
    agent, tid = _agent_with_ring(_snap(tools=["execute", "web_search"]))
    agent._emit_late_outcome_correction(tid, "passed")
    msg = [m for t, m, kw in seen if t == "Turn Outcome"][-1]
    assert "tools: execute, web_search" in msg and "no tools" not in msg
    agent, tid = _agent_with_ring(_snap(tools=[]))
    agent._emit_late_outcome_correction(tid, "passed")
    msg = [m for t, m, kw in seen if t == "Turn Outcome"][-1]
    assert "no tools" in msg and "tools:" not in msg
    snap = _snap(); snap.pop("chars")
    agent, tid = _agent_with_ring(snap)
    agent._emit_late_outcome_correction(tid, "passed")
    assert "· 0 chars" in [m for t, m, kw in seen if t == "Turn Outcome"][-1]


def test_correction_severity_follows_the_new_valence(monkeypatch):
    from ghost_agent.utils.logging import Icons
    seen = _capture_kw(monkeypatch)
    from ghost_agent.utils.logging import request_id_context
    agent, tid = _agent_with_ring(_snap(state="ok", exec_terminal=False, exec_failures=0))
    tok = request_id_context.set("a1b2c3d4e5f60708")      # §4NV: an OWNER turn's late refute
    try:
        agent._emit_late_outcome_correction(tid, "failed")
    finally:
        request_id_context.reset(tok)
    t, m, kw = [x for x in seen if x[0] == "Turn Outcome"][-1]
    assert kw["level"] == "INFO" and kw["icon"] == Icons.VERDICT_NOTE, kw    # §4OK (operator): information
    agent, tid = _agent_with_ring(_snap(state="failed"))
    agent._emit_late_outcome_correction(tid, "passed")
    t, m, kw = [x for x in seen if x[0] == "Turn Outcome"][-1]
    assert kw["level"] == "INFO" and kw["icon"] == Icons.OK, kw


# ── decider survivors from the whole-function battery ─────────────────── #

def test_drop_pending_corrections_removes_only_that_trajectorys_banners(monkeypatch):
    seen = _capture_kw(monkeypatch)
    agent = GhostAgent.__new__(GhostAgent)
    import types
    agent.context = types.SimpleNamespace()
    agent._pending_corrections = [{"traj": "t"}, {"traj": "u"}, {"traj": "t"}, "junk"]
    assert agent._drop_pending_corrections_for("t") == 2
    assert agent._pending_corrections == [{"traj": "u"}, "junk"]
    assert any("revoked 2" in m for t, m, kw in seen if t == "Human Feedback"), seen
    assert agent._drop_pending_corrections_for("zzz") == 0
    assert agent._drop_pending_corrections_for("") == 0


def test_late_verdict_targets_the_cached_row_with_that_id_not_the_first():
    from tests.test_selfhood_late_verdict_backfill import _fake_agent, _run_late
    first = _cached(tid="t-first"); second = _cached(tid="t-second")
    fake, writes = _fake_agent(True, first, None)
    fake.context._recent_trajectories_for_correction = {"fp1": first, "fp2": second}
    _run_late(fake, "t-second", "failed", reason="late refute")
    assert writes and writes[-1][0] == "t-second"
    assert second.outcome == "failed" and first.outcome == "unknown"


def test_late_pass_on_an_already_passed_row_writes_nothing():
    from tests.test_selfhood_late_verdict_backfill import _fake_agent, _run_late
    cached = _cached(outcome="passed")
    fake, writes = _fake_agent(True, cached, None)
    _run_late(fake, "t-c", "passed")
    assert writes == []


def test_verifier_evidence_scan_skips_a_None_entry():
    from ghost_agent.core.agent import _find_substantive_tool_for_verifier
    real = {"name": "execute", "content": "SUCCESS: 42 lines"}
    out = _find_substantive_tool_for_verifier([real, None])
    assert out is real
    assert _find_substantive_tool_for_verifier([None]) is None


# ── the failure-report negative tier ───────────────────────────────────── #

def _verdict():
    import types
    return types.SimpleNamespace(signals=["broken", "does not work"])


# ── the late write's reason rule on the cached row ─────────────────────── #

def test_late_refute_keeps_an_existing_cached_reason_and_fills_a_missing_one():
    from tests.test_selfhood_late_verdict_backfill import _fake_agent, _run_late
    kept = _cached(reason="browser selector thrash")
    fake, writes = _fake_agent(True, kept, None)
    _run_late(fake, "t-c", "failed", reason="verifier refuted: x")
    assert kept.outcome == "failed" and kept.failure_reason == "browser selector thrash"
    empty = _cached(reason="")
    fake, writes = _fake_agent(True, empty, None)
    _run_late(fake, "t-c", "failed", reason="")
    assert empty.outcome == "failed" and empty.failure_reason == ""


# ── backfill / drop / lock guards, and the flush wrong-traj window ─────── #

def test_backfill_returns_without_writing_when_the_collector_is_absent():
    from tests.test_selfhood_late_verdict_backfill import _fake_agent, _run_late
    fake, writes = _fake_agent(True, None, None)
    fake.context.trajectory_collector = None
    _run_late(fake, "t-c", "failed", reason="x")
    assert writes == []
    fake, writes = _fake_agent(True, _cached(), None)
    _run_late(fake, "", "failed", reason="x")            # no trajectory id
    assert writes == []


def test_late_pass_with_no_cached_row_writes_nothing():
    from tests.test_selfhood_late_verdict_backfill import _fake_agent, _run_late
    fake, writes = _fake_agent(True, None, None)          # collector present, cache empty
    _run_late(fake, "t-missing", "passed")
    assert writes == []


def test_drop_pending_needs_both_a_list_and_an_id():
    import types
    agent = GhostAgent.__new__(GhostAgent); agent.context = types.SimpleNamespace()
    agent._pending_corrections = []
    assert agent._drop_pending_corrections_for("t") == 0
    agent._pending_corrections = None
    assert agent._drop_pending_corrections_for("t") == 0
    agent._pending_corrections = [{"traj": "t"}]
    assert agent._drop_pending_corrections_for("") == 0
    assert agent._pending_corrections == [{"traj": "t"}], "nothing dropped on an empty id"


def test_human_label_locked_stops_at_the_first_id_match():
    import types
    stamped = _cached(tid="t"); stamped.extra = {"human_labeled": True}
    unstamped = _cached(tid="t"); unstamped.extra = {}
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = types.SimpleNamespace(
        _recent_trajectories_for_correction={"fp1": unstamped, "fp2": stamped})
    # the FIRST entry with id "t" is unstamped → break, so locked is False
    assert agent._human_label_locked("t") is False
    agent.context._recent_trajectories_for_correction = {"fp": stamped}
    assert agent._human_label_locked("t") is True
    assert agent._human_label_locked("nobody") is False


# ── retro tiers must scan PAST a foreign sample, not break on it ────────── #

# ── the finalize Turn Outcome line's icon/level/note follow the state ──── #

def _fin_line(monkeypatch, **fs_over):
    """Drive finalize and capture the Turn Outcome pretty_log call (title,
    message, kwargs) — the icon and level, not just the state word."""
    seen = []
    from ghost_agent.core import agent as A
    monkeypatch.setattr(A, "pretty_log", lambda t, m, **kw: seen.append((t, m, kw)))
    import asyncio
    a = _fin_agent()
    asyncio.get_event_loop().run_until_complete(a._finalize_and_return(_fs(**fs_over)))
    return [x for x in seen if x[0] == "Turn Outcome"][-1]


def test_finalize_line_failed_is_fail_icon_at_warning(monkeypatch):
    from ghost_agent.utils.logging import Icons
    t, m, kw = _fin_line(monkeypatch,
                         messages=_tool_msgs(1, "Error: EXIT CODE: 1"),
                         tools_run_this_turn=[{"name": "execute", "content": "Error: EXIT CODE: 1"}],
                         final_ai_content="it failed", execution_failure_count=1,
                         last_was_failure=True, current_trajectory_id="tf")
    assert m.startswith("failed") and kw["icon"] == Icons.FAIL and kw["level"] == "WARNING"


def test_finalize_line_ok_is_ok_icon_at_info(monkeypatch):
    from ghost_agent.utils.logging import Icons
    t, m, kw = _fin_line(monkeypatch,
                         messages=_tool_msgs(1, "SUCCESS"),
                         tools_run_this_turn=[{"name": "execute", "content": "SUCCESS"}],
                         final_ai_content="all good", current_trajectory_id="to")
    assert m.startswith("ok") and kw["icon"] == Icons.OK and kw["level"] == "INFO"


def test_finalize_line_budget_is_a_note_at_warning_with_the_stop_icon(monkeypatch):
    from ghost_agent.utils.logging import Icons
    t, m, kw = _fin_line(monkeypatch,
                         messages=_tool_msgs(1, "SUCCESS"),
                         tools_run_this_turn=[{"name": "execute", "content": "SUCCESS"}],
                         final_ai_content="[TURN BUDGET EXHAUSTED] partial",
                         turn_budget_exhausted=True, current_trajectory_id="tb")
    assert m.startswith("partial (budget exhausted)")
    assert kw["icon"] == Icons.STOP and kw["level"] == "WARNING"


def test_finalize_line_has_no_budget_note_without_exhaustion(monkeypatch):
    t, m, kw = _fin_line(monkeypatch,
                         messages=_tool_msgs(1, "SUCCESS"),
                         tools_run_this_turn=[{"name": "execute", "content": "SUCCESS"}],
                         final_ai_content="done", turn_budget_exhausted=False,
                         current_trajectory_id="tn")
    assert m.startswith("ok") and "budget exhausted" not in m


def test_finalize_line_carries_the_honest_report_note_not_recovered(monkeypatch):
    t, m, kw = _fin_line(monkeypatch,
                         messages=_tool_msgs(1, "Error: EXIT CODE: 1"),
                         tools_run_this_turn=[{"name": "execute", "content": "Error: EXIT CODE: 1"}],
                         final_ai_content="the command exited nonzero",
                         execution_failure_count=1, last_was_failure=True,
                         current_trajectory_id="th")
    # a terminal failure with an honest reply is "failed"; the note must not
    # invent a recovery
    assert m.startswith("failed") and "recovered" not in m
