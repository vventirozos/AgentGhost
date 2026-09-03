"""§4EE pins — the label producers as a unit (driven).

Each pin names the world where the fix decides: the operator line for a
shape-heuristic failure, a verified turn that exhausted its budget, a late
verdict on a turn the corpus will never upgrade, and a human thumb that
must reach the calibration fit.
"""
from __future__ import annotations

import itertools
import json
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.core import calibration as CAL
from ghost_agent.core.calibration import CalibrationTracker
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

def _tracker(tmp_path):
    return CalibrationTracker(tmp_path / "calibration")


def _seed(ct, req_id, outcome, source="turn"):
    ct.record(composite=0.8, entropy_component=0.5, competence_component=0.6,
              outcome=outcome, source=source, req_id=req_id)


def test_human_feedback_has_the_top_calibration_rank():
    assert CAL._SOURCE_RANK["human_feedback"] > max(
        v for k, v in CAL._SOURCE_RANK.items() if k != "human_feedback")


def test_human_label_relabels_the_turn_sample_and_wins_over_a_late_verdict(tmp_path):
    ct = _tracker(tmp_path)
    _seed(ct, "r1", 1.0)
    assert ct.record_human_label("r1", passed=False) is True
    resolved = CAL._resolve_superseded(ct._load_samples())
    rows = [s for s in resolved if s.req_id == "r1"]
    assert len(rows) == 1 and rows[0].source == "human_feedback" and rows[0].outcome == 0.0
    # a late machine verdict arriving afterwards does not overturn the human
    ct.record_late_verdict_correction("r1", 1.0)
    resolved = CAL._resolve_superseded(ct._load_samples())
    rows = [s for s in resolved if s.req_id == "r1"]
    assert rows[0].source == "human_feedback" and rows[0].outcome == 0.0


def test_human_label_joins_a_late_verdict_sample_when_that_is_all_there_is(tmp_path):
    ct = _tracker(tmp_path)
    _seed(ct, "r2", 1.0)
    _seed(ct, "r2", 0.0, source="verifier_late")
    assert ct.record_human_label("r2", passed=True) is True
    rows = [s for s in CAL._resolve_superseded(ct._load_samples()) if s.req_id == "r2"]
    assert rows[0].source == "human_feedback" and rows[0].outcome == 1.0


def test_human_label_is_idempotent_but_a_changed_thumb_writes_again(tmp_path):
    ct = _tracker(tmp_path)
    _seed(ct, "r3", 0.5)
    assert ct.record_human_label("r3", passed=True) is True
    assert ct.record_human_label("r3", passed=True) is False
    assert ct.record_human_label("r3", passed=False) is True
    rows = [s for s in CAL._resolve_superseded(ct._load_samples()) if s.req_id == "r3"]
    assert rows[0].outcome == 0.0


def test_human_label_without_a_sample_to_join_writes_nothing(tmp_path):
    ct = _tracker(tmp_path)
    assert ct.record_human_label("nope", passed=True) is False
    assert ct.record_human_label("", passed=True) is False
    assert ct._load_samples() == []


# ── the feedback route wires the tracker ──────────────────────────────── #

class _Traj:
    def __init__(self, tid, rid):
        self.id, self.extra, self.session_id = tid, {"req_id": rid}, ""
        self.outcome, self.failure_reason = "unknown", ""


class _Col:
    redaction = None

    def __init__(self, traj):
        self._t = traj
        self.updates = []

    def iter_trajectories(self, day=None):
        return iter([self._t])

    def update_outcome(self, tid, outcome, reason="", source="", **kw):
        self.updates.append((tid, outcome, source))
        return True


def _feedback_agent(tracker):
    import types
    ctx = types.SimpleNamespace(trajectory_collector=_Col(_Traj("t1", "req-9")),
                                calibration_tracker=tracker,
                                _recent_trajectories_for_correction=None,
                                self_model=None, args=None)
    return types.SimpleNamespace(context=ctx)


def test_apply_human_label_relabels_calibration_and_survives_a_raising_tracker():
    from ghost_agent.core.feedback import apply_human_label
    seen = []

    class _T:
        def record_human_label(self, rid, passed):
            seen.append((rid, passed)); return True
    out = apply_human_label(_feedback_agent(_T()), "chatcmpl-req-9", "negative", "wrong")
    assert out["ok"] is True and seen == [("req-9", False)], (out, seen)

    class _Boom:
        def record_human_label(self, rid, passed):
            raise RuntimeError("disk")
    agent = _feedback_agent(_Boom())
    out = apply_human_label(agent, "req-9", "positive")
    assert out["ok"] is True
    assert agent.context.trajectory_collector.updates[-1][1] == "passed"


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
    (True, 0, 0.5, "confidence 0.50", "strike"),
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

def test_grade_of_a_shape_failure_is_below_the_prior_even_with_a_pass():
    g = CAL.grade_turn_outcome(verifier_verdict="passed", shape_failed=True)
    assert g == CAL._SHAPE_FAILURE_GRADE < CAL._UNVERIFIED_PRIOR
    assert CAL.grade_turn_outcome(verifier_verdict="failed", shape_failed=True) == 0.0
    assert CAL.grade_turn_outcome(shape_failed=False) == CAL._UNVERIFIED_PRIOR


def test_shape_failure_relabels_the_turn_sample_once_and_yields_to_a_refute(tmp_path):
    ct = _tracker(tmp_path)
    _seed(ct, "s1", 1.0)
    assert ct.record_shape_failure("s1") is True
    assert ct.record_shape_failure("s1") is False
    rows = [s for s in CAL._resolve_superseded(ct._load_samples()) if s.req_id == "s1"]
    assert rows[0].source == "shape_failure" and rows[0].outcome == CAL._SHAPE_FAILURE_GRADE
    ct.record_late_verdict_correction("s1", 0.0)
    rows = [s for s in CAL._resolve_superseded(ct._load_samples()) if s.req_id == "s1"]
    assert rows[0].source == "verifier_late" and rows[0].outcome == 0.0
    assert ct.record_shape_failure("nope") is False


def test_rank_order_is_the_authority_order():
    r = CAL._SOURCE_RANK
    assert r["turn"] < r["shape_failure"] < r["verifier_late"] < r["task_reopened"] \
        < r["failure_report"] < r["bench_validator"] < r["user_correction"] \
        < r["human_feedback"]


class _ShapeTracker:
    def __init__(self):
        self.calls = []

    def record_shape_failure(self, rid):
        self.calls.append(rid); return True


@pytest.mark.asyncio
async def test_finalize_relabels_calibration_for_a_shape_failure_and_not_otherwise():
    a = _fin_agent(); a.context.calibration_tracker = _ShapeTracker()
    msgs = _tool_msgs(3, "Error: ENOENT no such file")
    tools = [{"name": "execute", "content": "Error: ENOENT no such file"}] * 3
    await a._finalize_and_return(_fs(messages=msgs, tools_run_this_turn=tools,
                                     final_ai_content="I fixed it.", req_id="rq-shape",
                                     execution_failure_count=0, last_was_failure=False,
                                     current_trajectory_id="t-shape2"))
    assert a.context.calibration_tracker.calls == ["rq-shape"]
    b = _fin_agent(); b.context.calibration_tracker = _ShapeTracker()
    await b._finalize_and_return(_fs(messages=_tool_msgs(1, "ok"),
                                     tools_run_this_turn=[{"name": "execute", "content": "ok"}],
                                     final_ai_content="done", req_id="rq-clean",
                                     current_trajectory_id="t-clean"))
    assert b.context.calibration_tracker.calls == []


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


def test_feedback_line_says_whether_the_fit_took_the_label(monkeypatch):
    from ghost_agent.core import feedback as FB
    seen = []
    monkeypatch.setattr(FB, "pretty_log", lambda t, m, **kw: seen.append((t, m)))

    class _T:
        def __init__(self, ok): self.ok = ok
        def record_human_label(self, rid, passed): return self.ok
    FB.apply_human_label(_feedback_agent(_T(True)), "req-9", "negative", "wrong")
    assert any("calibration relabelled" in m for t, m in seen), seen
    seen.clear()
    FB.apply_human_label(_feedback_agent(_T(False)), "req-9", "negative", "wrong")
    assert any("calibration unchanged" in m for t, m in seen), seen


# ── the late verdict re-labels calibration through the cached row's req_id ─ #

class _LateTracker:
    def __init__(self):
        self.calls = []

    def record_late_verdict_correction(self, rid, outcome):
        self.calls.append((rid, outcome)); return True


@pytest.mark.parametrize("outcome,cached_outcome,cached_reason,expect", [
    ("failed", "unknown", "", 0.0),
    ("passed", "failed", STRUCTURAL_FAILURE_REASON, 1.0),
], ids=["late-refute", "late-pass"])
def test_late_verdict_relabels_calibration_by_the_rows_req_id(outcome, cached_outcome,
                                                              cached_reason, expect):
    from tests.test_selfhood_late_verdict_backfill import _fake_agent, _run_late
    cached = _cached(outcome=cached_outcome, reason=cached_reason)
    cached.extra = {"req_id": "rq-late"}
    fake, writes = _fake_agent(True, cached, None)
    fake.context.calibration_tracker = _LateTracker()
    _run_late(fake, "t-c", outcome, reason="verifier says so")
    assert writes and writes[-1][1] == outcome
    assert fake.context.calibration_tracker.calls == [("rq-late", expect)]


def test_late_verdict_without_a_req_id_touches_no_calibration_sample():
    from tests.test_selfhood_late_verdict_backfill import _fake_agent, _run_late
    cached = _cached()                       # extra has no req_id
    fake, writes = _fake_agent(True, cached, None)
    fake.context.calibration_tracker = _LateTracker()
    _run_late(fake, "t-c", "failed", reason="x")
    assert writes and fake.context.calibration_tracker.calls == []


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
    agent, tid = _agent_with_ring(_snap(state="ok", exec_terminal=False, exec_failures=0))
    agent._emit_late_outcome_correction(tid, "failed")
    t, m, kw = [x for x in seen if x[0] == "Turn Outcome"][-1]
    assert kw["level"] == "WARNING" and kw["icon"] == Icons.FAIL, kw
    agent, tid = _agent_with_ring(_snap(state="failed"))
    agent._emit_late_outcome_correction(tid, "passed")
    t, m, kw = [x for x in seen if x[0] == "Turn Outcome"][-1]
    assert kw["level"] == "INFO" and kw["icon"] == Icons.OK, kw


def test_grade_treats_a_non_numeric_failure_count_as_zero():
    assert CAL.grade_turn_outcome(execution_failure_count="abc") == CAL._UNVERIFIED_PRIOR
    assert CAL.grade_turn_outcome(execution_failure_count=None) == CAL._UNVERIFIED_PRIOR
    assert CAL.grade_turn_outcome(execution_failure_count=1) == pytest.approx(
        CAL._UNVERIFIED_PRIOR - CAL._EXEC_FAILURE_PENALTY)


def _seed_c(ct, req_id, outcome, composite, source="turn"):
    ct.record(composite=composite, entropy_component=0.5, competence_component=0.6,
              outcome=outcome, source=source, req_id=req_id)


def _rows(ct, rid, source):
    return [s for s in ct._load_samples() if s.req_id == rid and s.source == source]


def test_shape_tier_joins_only_its_own_request_and_its_turn_row(tmp_path):
    ct = _tracker(tmp_path)
    _seed_c(ct, "b", 1.0, 0.33)
    _seed_c(ct, "b", 0.0, 0.11, source="verifier_late")   # a later, higher-ranked row
    _seed_c(ct, "a", 1.0, 0.81)                            # ANOTHER request, last in file
    assert ct.record_shape_failure("b") is True
    row = _rows(ct, "b", "shape_failure")[0]
    assert row.composite == 0.33, "features must come from b's TURN row"
    assert _rows(ct, "a", "shape_failure") == []
    assert ct.record_shape_failure("") is False
    _seed_c(ct, "5", 1.0, 0.42)
    assert ct.record_shape_failure(5) is True, "a non-str request id is normalised"
    assert _rows(ct, "5", "shape_failure")[0].composite == 0.42


def test_shape_tier_writes_nothing_when_the_turn_grade_already_says_so(tmp_path):
    ct = _tracker(tmp_path)
    _seed_c(ct, "c", CAL._SHAPE_FAILURE_GRADE, 0.5)
    assert ct.record_shape_failure("c") is False
    assert _rows(ct, "c", "shape_failure") == []


def test_human_tier_joins_the_highest_ranked_row_of_its_own_request(tmp_path):
    ct = _tracker(tmp_path)
    _seed_c(ct, "other", 1.0, 0.01)                        # a foreign row FIRST
    _seed_c(ct, "h", 1.0, 0.71)
    _seed_c(ct, "h", 0.0, 0.22, source="verifier_late")
    _seed_c(ct, "h", 0.5, 0.99, source="turn")             # a later turn row: lower rank
    _seed_c(ct, "other", 0.0, 0.55, source="verifier_late")  # a foreign, high-rank row LAST
    assert ct.record_human_label("h", passed=True) is True
    row = _rows(ct, "h", "human_feedback")[0]
    assert row.composite == 0.22, "the verifier_late row of THIS request outranks both turn rows"
    assert _rows(ct, "other", "human_feedback") == []
    _seed_c(ct, "7", 1.0, 0.61)
    assert ct.record_human_label(7, passed=True) is True
    assert _rows(ct, "7", "human_feedback")[0].composite == 0.61


def test_human_tier_keeps_scanning_past_a_standing_human_row(tmp_path):
    ct = _tracker(tmp_path)
    _seed_c(ct, "k", 1.0, 0.71)
    _seed_c(ct, "k", 1.0, 0.50, source="human_feedback")   # standing thumb (up)
    _seed_c(ct, "k", 0.0, 0.22, source="verifier_late")    # lands after the thumb
    assert ct.record_human_label("k", passed=False) is True  # a CHANGED thumb
    rows = _rows(ct, "k", "human_feedback")
    assert rows[-1].outcome == 0.0 and rows[-1].composite == 0.22


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


def test_stashed_lesson_outcome_flushes_once_and_rebooks_on_a_sign_flip():
    import types
    calls = []
    sm = types.SimpleNamespace(record_surfaced_outcomes=lambda trig, ok: calls.append((tuple(trig), ok)))
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = types.SimpleNamespace(_surfaced_triggers_by_traj={"t": ["L1", "L2"]},
                                          skill_memory=sm)
    import asyncio
    async def go():
        agent._flush_stashed_lesson_outcome("t", True)
        await asyncio.sleep(0.2)
        agent._flush_stashed_lesson_outcome("t", True)      # same sign: no-op
        await asyncio.sleep(0.2)
        agent._flush_stashed_lesson_outcome("t", False)     # flip: re-booked
        await asyncio.sleep(0.2)
        agent._flush_stashed_lesson_outcome("nope", True)   # nothing stashed
        await asyncio.sleep(0.2)
    asyncio.run(go())
    assert calls == [(("L1", "L2"), True), (("L1", "L2"), False)], calls
    assert "t" not in agent.context._surfaced_triggers_by_traj


def test_verifier_evidence_scan_skips_a_None_entry():
    from ghost_agent.core.agent import _find_substantive_tool_for_verifier
    real = {"name": "execute", "content": "SUCCESS: 42 lines"}
    out = _find_substantive_tool_for_verifier([real, None])
    assert out is real
    assert _find_substantive_tool_for_verifier([None]) is None


# ── the failure-report negative tier ───────────────────────────────────── #

class _RecTracker:
    def __init__(self, boom=False):
        self.rows, self.boom = [], boom

    def record(self, **kw):
        if self.boom:
            raise RuntimeError("disk")
        self.rows.append(kw); return True


def _fr_agent(stash, tracker):
    import types
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = types.SimpleNamespace(_recent_calib_for_correction=stash,
                                          calibration_tracker=tracker)
    return agent


def _verdict():
    import types
    return types.SimpleNamespace(signals=["broken", "does not work"])


def test_failure_report_negative_needs_stash_tracker_and_a_matching_fingerprint():
    t = _RecTracker()
    assert _fr_agent(None, t)._record_failure_report_negative("fp", _verdict()) is False
    assert _fr_agent({"fp": {"composite": 0.7}}, None)._record_failure_report_negative("fp", _verdict()) is False
    assert _fr_agent({"other": {"composite": 0.7}}, t)._record_failure_report_negative("fp", _verdict()) is False
    assert t.rows == []


def test_failure_report_negative_records_once_at_the_fractional_grade(monkeypatch):
    seen = _capture_kw(monkeypatch)
    t = _RecTracker()
    agent = _fr_agent({"fp": {"composite": 0.7, "entropy_component": 0.5,
                              "competence_component": 0.6}}, t)
    assert agent._record_failure_report_negative("fp", _verdict()) is True
    assert t.rows == [{"outcome": CAL._FAILURE_REPORT_GRADE, "source": "failure_report",
                       "composite": 0.7, "entropy_component": 0.5, "competence_component": 0.6}]
    assert [x[0] for x in seen] == ["Failure Report"]
    # the stash entry is consumed: a correction on the same turn cannot double-count
    assert agent._record_failure_report_negative("fp", _verdict()) is False
    assert len(t.rows) == 1


def test_user_correction_through_the_same_tier_uses_its_grade_and_title(monkeypatch):
    seen = _capture_kw(monkeypatch)
    t = _RecTracker()
    agent = _fr_agent({"fp": {"composite": 0.7}}, t)
    assert agent._record_failure_report_negative("fp", _verdict(), source="user_correction",
                                                 grade=0.0) is True
    assert t.rows[-1]["outcome"] == 0.0 and t.rows[-1]["source"] == "user_correction"
    assert [x[0] for x in seen] == ["User Correction"]
    assert "grade 0.0" in seen[-1][1]


def test_failure_report_negative_is_False_when_the_tracker_raises():
    agent = _fr_agent({"fp": {"composite": 0.7}}, _RecTracker(boom=True))
    assert agent._record_failure_report_negative("fp", _verdict()) is False


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


def test_flush_on_a_wrong_traj_stash_does_not_drop_a_pending_rebook():
    """stash is non-empty but for ANOTHER trajectory, and THIS one has a
    retained set of the opposite sign — the re-book must still fire, not be
    lost to a KeyError (the `and` guard, not `or`)."""
    import types, asyncio
    calls = []
    sm = types.SimpleNamespace(record_surfaced_outcomes=lambda trig, ok: calls.append((tuple(trig), ok)))
    from collections import OrderedDict
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = types.SimpleNamespace(
        _surfaced_triggers_by_traj={"other": ["Z"]},        # non-empty, different id
        _flushed_triggers_by_traj=OrderedDict([("t", (["L1"], True))]),  # prior, sign True
        skill_memory=sm)
    async def go():
        agent._flush_stashed_lesson_outcome("t", False)     # sign flip → re-book
        await asyncio.sleep(0.2)
    asyncio.run(go())
    assert calls == [(("L1",), False)], calls


# ── retro tiers must scan PAST a foreign sample, not break on it ────────── #

def test_late_verdict_correction_finds_a_target_that_is_not_first(tmp_path):
    ct = _tracker(tmp_path)
    _seed_c(ct, "foreign", 1.0, 0.11)          # a DIFFERENT request, first in file
    _seed_c(ct, "r", 1.0, 0.71)                # the target's turn row
    assert ct.record_late_verdict_correction("r", 0.0) is True
    rows = [s for s in CAL._resolve_superseded(ct._load_samples()) if s.req_id == "r"]
    assert rows[0].source == "verifier_late" and rows[0].outcome == 0.0
    assert rows[0].composite == 0.71


def test_task_reopened_negative_finds_a_target_that_is_not_first(tmp_path):
    ct = _tracker(tmp_path)
    _seed_c(ct, "foreign", 1.0, 0.11)
    _seed_c(ct, "closed", 1.0, 0.71)
    assert ct.record_task_reopened_negative("closed") is True
    rows = [s for s in CAL._resolve_superseded(ct._load_samples()) if s.req_id == "closed"]
    assert rows[0].source == "task_reopened"


def test_bench_validator_verdict_finds_a_target_that_is_not_first(tmp_path):
    ct = _tracker(tmp_path)
    ct.record(composite=0.5, entropy_component=0.5, competence_component=0.6,
              outcome=1.0, source="turn", req_id="foreign", origin="bench")
    ct.record(composite=0.71, entropy_component=0.5, competence_component=0.6,
              outcome=1.0, source="turn", req_id="bench-1", origin="bench")
    assert ct.record_bench_validator_verdict("bench-1", passed=False) is True
    rows = [s for s in CAL._resolve_superseded(ct._load_samples()) if s.req_id == "bench-1"]
    assert rows[0].source == "bench_validator" and rows[0].outcome == 0.0
    assert rows[0].composite == 0.71


def test_shape_failure_finds_a_target_that_is_not_first(tmp_path):
    ct = _tracker(tmp_path)
    _seed_c(ct, "foreign", 1.0, 0.11)
    _seed_c(ct, "s", 1.0, 0.71)
    assert ct.record_shape_failure("s") is True
    rows = [s for s in CAL._resolve_superseded(ct._load_samples()) if s.req_id == "s"]
    assert rows[0].source == "shape_failure" and rows[0].composite == 0.71


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
