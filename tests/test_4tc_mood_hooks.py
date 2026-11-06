"""§4TC (2026-11-06): what MOVES the mood, and who may.

The only writers besides `introspect(action="set_mood")`:

* the end of an OWNER's turn — failed tool calls and loop guards, a clean
  multi-tool turn, a research turn, a very long turn
  (`GhostAgent._record_turn_trajectory` → `SelfModel.note_owner_turn`);
* the owner's thumbs (`core/feedback.apply_human_label`), a correction of
  the last reply (the judge's positive verdict), thanks in the message;
* nothing else. A long quiet moves nothing (review): boredom is what a
  reader other than the owner sees over a neutral mood; an owner's request
  arriving only ends the quiet (`GhostAgent._mood_owner_arrived`).

Never: a probe, a channel member's turn, a scheduled task, a job wake, a
sub-agent, self-play, a bench or replay run, a coding leaf, a contained copy.

Each hook is driven through the real method that carries it. All owner data
here is fictional (Elena, her dog Bruno)."""
from __future__ import annotations

import ast
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core import feedback as F
from ghost_agent.core.agent import GhostAgent
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Outcome, Trajectory
from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood import mood as M
from ghost_agent.tools.outcome import ToolOutcome
from ghost_agent.utils.logging import (
    ORIGIN_PROBE, request_id_context, request_origin_context, requester_role_context)

ROOT = Path(__file__).resolve().parents[1]
OWNER_RID = "web-4tc-hooks"
ASK = "what did Bruno's vet say about the limp?"


@pytest.fixture
def owner():
    tok = request_id_context.set(OWNER_RID)
    yield OWNER_RID
    request_id_context.reset(tok)


def _traj_agent(tmp_path) -> GhostAgent:
    """A real agent object over a mock context with a real self-model — the
    harness the derived-mood hook tests used."""
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.context.trajectory_collector = MagicMock()
    agent.context.trajectory_collector.append = MagicMock(return_value=None)
    agent.context._ctx_pressure_lockdown = False
    agent.context._guard_events = None
    agent.context.turn_origin_label = None
    agent.context.owner_memory_isolated = False
    agent.context._replay_forbidden_tools = None
    agent.context.self_model = SelfModel(root=tmp_path)
    return agent


def _call(i, name="execute", args=None):
    return {"id": f"c{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {"command": "ls"})}}


def _turn(results, ask=ASK, history=()):
    """One turn's messages: the request, then one assistant call + tool row
    per result."""
    msgs = list(history) + [{"role": "user", "content": ask}]
    for i, (name, content) in enumerate(results):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [_call(i, name)]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": name, "content": content})
    return msgs


OK = ToolOutcome.ok("EXIT CODE: 0\nSTDOUT/STDERR:\nfine")
BAD = ToolOutcome.failed("EXIT CODE: 1\nSTDOUT/STDERR:\nTraceback (most recent call last):\nValueError: x")


def _page(i):
    return ToolOutcome.ok(f"URL: https://example.org/vet/{i}\nTITLE: page {i}\n\nBruno's vet page {i}.")


def _record(agent, msgs, rid=OWNER_RID, ask=ASK, **kw):
    return GhostAgent._record_turn_trajectory(
        agent, messages=msgs, final_content="done", req_id=rid, model="m", user_request=ask, **kw)


def _mood(agent):
    return agent.context.self_model.mood()


def _untouched(agent) -> bool:
    """No mood was ever stored and no journal row written (state.json itself
    may exist: every captured turn stamps the last-active time)."""
    st = agent.context.self_model.state
    return st.state.mood is None and not st.mood_history_path.exists()


def _rows(agent):
    p = agent.context.self_model.state.mood_history_path
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# ── the end of an owner's turn ──────────────────────────────────────────────

def test_a_plain_chat_turn_changes_nothing(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    _record(agent, [{"role": "user", "content": ASK}])
    assert _mood(agent).neutral and _untouched(agent)


def test_failed_tool_calls_in_the_turn_make_it_irritable(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    _record(agent, _turn([("execute", BAD), ("execute", BAD), ("execute", OK), ("execute", BAD)]))
    v = _mood(agent)
    assert (v.label, v.intensity) == ("irritable", 1)
    assert (v.p, v.e) == (pytest.approx(-0.36), pytest.approx(0.24))
    assert v.cause == "3 tool calls failed in this turn" and v.source == "auto"
    assert [(r["label"], r["kind"]) for r in _rows(agent)] == [("irritable", "turn")]


def test_a_clean_turn_of_several_calls_lifts_it_a_little(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    _record(agent, _turn([("execute", OK)] * 3))
    assert (_mood(agent).p, _mood(agent).e) == (pytest.approx(0.08), 0.0)
    two = _traj_agent(tmp_path / "two")
    _record(two, _turn([("execute", OK)] * 2))
    assert _mood(two).p == 0.0


def test_a_designed_stop_is_not_a_failure(tmp_path, owner):
    """Asking the owner to confirm is the tool doing its job (§4MI)."""
    agent = _traj_agent(tmp_path)
    stop = ToolOutcome.rejected("SYSTEM BLOCK: ask the user first.", reason_code="clarify_first")
    _record(agent, _turn([("execute", OK), ("execute", OK), ("manage_projects", stop)]))
    assert _mood(agent).p == pytest.approx(0.08)                        # three answered calls, none failed


def test_a_refused_call_counts_as_a_failure(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    refused = ToolOutcome.rejected("SYSTEM BLOCK: that path is outside the project.", reason_code="path_denied")
    _record(agent, _turn([("file_system", refused), ("execute", OK)]))
    assert (_mood(agent).p, _mood(agent).e) == (pytest.approx(-0.12), pytest.approx(0.08))


def test_only_this_turns_calls_count(tmp_path, owner):
    """Failures the client sent back as HISTORY belong to an earlier turn."""
    agent = _traj_agent(tmp_path)
    earlier = _turn([("execute", BAD)] * 4, ask="an earlier request about Bruno's food")
    earlier.append({"role": "assistant", "content": "It failed four times."})
    _record(agent, _turn([("execute", OK)] * 3, history=earlier))
    assert (_mood(agent).p, _mood(agent).e) == (pytest.approx(0.08), 0.0)


def test_a_research_turn_that_read_several_pages_makes_it_curious(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    _record(agent, _turn([("browser", _page(1)), ("browser", _page(2)), ("browser", _page(3))]))
    v = _mood(agent)
    assert (v.label, v.intensity) == ("curious", 1)
    assert (v.p, v.e) == (pytest.approx(0.15 + 0.08), pytest.approx(0.20))
    assert "read 3 sources in a research turn" in v.cause


def test_the_same_page_read_three_times_is_one_source(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    _record(agent, _turn([("browser", _page(1))] * 3))
    assert (_mood(agent).p, _mood(agent).e) == (pytest.approx(0.08), 0.0)       # a clean turn, no research


def test_a_failed_page_load_is_not_a_source(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    dead = ToolOutcome.failed("URL: https://example.org/vet/9\nERROR: timeout")
    sig = agent._mood_turn_signals(_turn([("browser", _page(1)), ("browser", _page(2)), ("browser", dead)]),
                                   ASK, OWNER_RID)
    assert (sig.tool_calls, sig.failures, sig.sources) == (3, 1, 2)


def test_a_very_long_turn_makes_it_tired(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    _record(agent, _turn([("execute", OK)] * 2), elapsed_s=1500.0)
    v = _mood(agent)
    assert (v.label, v.intensity) == ("tired", 1) and v.cause == "a very long turn (25 min, 2 tool calls)"


def test_the_signals_are_the_turns_own(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    A.note_guard_event(agent.context, OWNER_RID, "edit_churn")
    A.note_guard_event(agent.context, "web-someone-else", "futility")
    sig = agent._mood_turn_signals(_turn([("execute", OK), ("execute", BAD), ("browser", _page(1))]),
                                   ASK, OWNER_RID, duration_s=12.5, lockdown=True)
    assert sig == M.TurnSignals(tool_calls=3, failures=1, guards=1, sources=1, duration_s=12.5, completed=True,
                                pressure=True, owner_set=False)
    # a call that never got a result is not an answered call
    pending = _turn([("execute", OK)]) + [{"role": "assistant", "content": "", "tool_calls": [_call(9)]}]
    assert agent._mood_turn_signals(pending, ASK, "web-other").tool_calls == 1


# ── guard events ────────────────────────────────────────────────────────────

def test_a_guard_event_in_the_turn_moves_the_mood(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    A.note_guard_event(agent.context, OWNER_RID, "edit_churn")
    A.note_guard_event(agent.context, OWNER_RID, "futility")
    _record(agent, _turn([("execute", OK)] * 3))
    v = _mood(agent)
    assert (v.p, v.e) == (pytest.approx(-0.24), pytest.approx(0.16))
    assert v.cause == "a loop guard stepped in 2 times in this turn"


def test_another_requests_guard_events_do_not_count(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    A.note_guard_event(agent.context, "web-someone-else", "edit_churn")
    _record(agent, _turn([("execute", OK)] * 3))
    assert _mood(agent).p == pytest.approx(0.08)


def test_a_breaker_that_closes_the_request_is_a_guard_event():
    ctx = SimpleNamespace()
    A.stamp_loop_breaker(ctx, "r1", "thinking_loop")
    A.stamp_loop_breaker(ctx, "r1", "cross_turn")
    assert A.guard_events_for(ctx, "r1") == ["thinking_loop", "cross_turn"]
    assert A.loop_breaker_for(ctx, "r1") == "cross_turn"                # the breaker report is unchanged


def test_the_guard_ledger_is_keyed_bounded_and_poppable():
    ctx = SimpleNamespace()
    assert A.guard_events_for(ctx, "r1") == [] and A.guard_events_for(None, "r1") == []
    A.note_guard_event(ctx, "", "futility")                             # no request: not counted
    A.note_guard_event(ctx, None, "futility")
    assert ctx._guard_events == {}
    for i in range(A._GUARD_EVENTS_MAX + 5):
        A.note_guard_event(ctx, f"r{i}", "edit_churn")
    assert len(ctx._guard_events) == A._GUARD_EVENTS_MAX and A.guard_events_for(ctx, "r0") == []
    last = f"r{A._GUARD_EVENTS_MAX + 4}"
    assert A.guard_events_for(ctx, last, pop=True) == ["edit_churn"] and A.guard_events_for(ctx, last) == []
    A.note_guard_event(object(), "r1", "x")                             # a context that takes no attribute: no raise


def _calls_named(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]


def test_every_guard_the_brief_names_reports_itself():
    """The sites, read off the tree: an edit-churn steer and stop, a futility
    steer and report, a killed thinking loop — and every breaker close, through
    `stamp_loop_breaker`."""
    tree = ast.parse((ROOT / "src" / "ghost_agent" / "core" / "agent.py").read_text(encoding="utf-8"))
    kinds = sorted(c.args[2].value for c in _calls_named(tree, "note_guard_event")
                   if len(c.args) == 3 and isinstance(c.args[2], ast.Constant))
    assert kinds == ["edit_churn", "edit_churn", "futility", "futility", "thinking_loop"]
    stamp = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "stamp_loop_breaker")
    assert len(_calls_named(stamp, "note_guard_event")) == 1
    assert len(_calls_named(tree, "stamp_loop_breaker")) >= 5


def test_a_request_id_that_is_reused_starts_with_no_guard_event():
    """handle_chat pops the ledger at request start, beside the breaker report."""
    tree = ast.parse((ROOT / "src" / "ghost_agent" / "core" / "agent.py").read_text(encoding="utf-8"))
    hc = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_chat")
    pops = [c for c in _calls_named(hc, "guard_events_for")
            if any(k.arg == "pop" and getattr(k.value, "value", None) is True for k in c.keywords)]
    assert len(pops) == 1


# ── what the mood does not look at (review) ─────────────────────────────────

REFUSED_PAGE = ToolOutcome.rejected("SYSTEM BLOCK: this request read outside content — ask the user to confirm "
                                    "in a new message.", reason_code="untrusted_content")
MEMBER_BLOCK = ToolOutcome.rejected("SYSTEM BLOCK: this tool reads the owner's data.", reason_code="owner_data_blocked")
OWNER_ONLY = ToolOutcome.rejected("SYSTEM BLOCK: only the owner can set the mood or the manner.",
                                  reason_code="owner_only")
INTROSPECT_ERR = "SYSTEM ERROR: 'furious' is not one of my moods. Nothing was changed."


@pytest.mark.parametrize("name,content", [
    ("manage_tasks", REFUSED_PAGE), ("update_profile", MEMBER_BLOCK), ("introspect", OWNER_ONLY),
    ("introspect", INTROSPECT_ERR), ("introspect", ToolOutcome.failed("Introspection failed: OSError: x")),
    ("introspect", "Experiences on file: 3"),
])
def test_a_designed_refusal_and_any_introspect_call_are_not_trouble(tmp_path, owner, name, content):
    agent = _traj_agent(tmp_path)
    sig = agent._mood_turn_signals(_turn([("execute", OK), (name, content), (name, content)]), ASK, OWNER_RID)
    assert (sig.tool_calls, sig.failures) == (1, 0)                    # neither trouble nor a "clean" call
    _record(agent, _turn([(name, content)] * 4))
    assert _mood(agent).neutral


def test_a_refusal_of_another_kind_is_still_trouble(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    denied = ToolOutcome.rejected("SYSTEM BLOCK: that path is outside the project.", reason_code="path_denied")
    sig = agent._mood_turn_signals(_turn([("file_system", denied), ("execute", BAD)]), ASK, OWNER_RID)
    assert (sig.tool_calls, sig.failures) == (2, 2)
    assert A.MOOD_BLIND_REASONS == frozenset({"untrusted_content", "owner_data_blocked", "owner_only"})


def _set_call(i, args):
    return {"id": f"m{i}", "type": "function", "function": {"name": "introspect", "arguments": json.dumps(args)}}


async def _mood_turn(agent, calls, extra=()):
    """One owner turn in which the model calls set_mood through the REAL tool
    (`calls`: argument dicts), plus `extra` ordinary tool rows. Returns the
    turn's messages."""
    from ghost_agent.tools.introspect import tool_introspect
    msgs = [{"role": "user", "content": "cheer up"}]
    for i, args in enumerate(calls):
        out = await tool_introspect(self_model=agent.context.self_model, context=agent.context, **args)
        msgs.append({"role": "assistant", "content": "", "tool_calls": [_set_call(i, args)]})
        msgs.append({"role": "tool", "tool_call_id": f"m{i}", "name": "introspect", "content": out})
    for i, (name, content) in enumerate(extra):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [_call(i, name)]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": name, "content": content})
    return msgs


async def test_a_wrong_word_then_the_right_one_lands_exactly_on_the_anchor(tmp_path, owner):
    """Review: the refused first call counted as a failed tool call, and the
    end-of-turn nudge pulled the mood the owner had just set off its anchor."""
    agent = _traj_agent(tmp_path)
    msgs = await _mood_turn(agent, [{"action": "set_mood", "mood": "chipper"},
                                    {"action": "set_mood", "mood": "cheerful", "intensity": 2}])
    assert "is not one of my moods" in str(msgs[2]["content"])
    _record(agent, msgs, ask="cheer up")
    v = _mood(agent)
    assert (v.label, v.intensity, v.source) == ("cheerful", 2, "owner")
    assert (v.p, v.e) == (pytest.approx(0.60, abs=1e-3), pytest.approx(0.45, abs=1e-3))
    assert [r["kind"] for r in _rows(agent)] == ["owner"]


async def test_cheer_up_with_four_failing_calls_in_the_same_turn_stays_cheerful(tmp_path, owner):
    """The owner's word holds for the turn: no automatic nudge on top of it."""
    agent = _traj_agent(tmp_path)
    A.note_guard_event(agent.context, OWNER_RID, "futility")
    msgs = await _mood_turn(agent, [{"action": "set_mood", "mood": "cheerful"}], extra=[("execute", BAD)] * 4)
    sig = agent._mood_turn_signals(msgs, "cheer up", OWNER_RID, lockdown=True)
    assert sig.owner_set is True and sig.failures == 4
    _record(agent, msgs, ask="cheer up", pressure_lockdown=True)
    v = _mood(agent)
    assert (v.label, v.intensity, v.source) == ("cheerful", 2, "owner") and v.p == pytest.approx(0.60, abs=1e-3)
    _record(agent, _turn([("execute", BAD)] * 4))                      # the NEXT turn's failures do count
    assert _mood(agent).p < 0.2


@pytest.mark.parametrize("evidence", ["Error: the deploy failed, cheer up", "Traceback (most recent call last): cheer up",
                                      "EXIT CODE: 1 — cheer up anyway", "FAILED: 3 tests — be cheerful"])
async def test_the_owners_own_words_cannot_make_his_set_mood_look_like_a_failed_call(tmp_path, owner, evidence):
    """Review r2: the reply was a plain string, the error sniffer read the
    owner's evidence text in it, booked the call FAILED — and the automatic
    nudge landed on top of his word. The reply is a declared OK now."""
    agent = _traj_agent(tmp_path)
    msgs = await _mood_turn(agent, [{"action": "set_mood", "mood": "cheerful", "evidence": evidence}],
                            extra=[("execute", BAD)] * 2)
    out = msgs[2]["content"]
    assert out.status.value == "ok" and out.declared is True and evidence.split()[0] in out
    sig = agent._mood_turn_signals(msgs, "cheer up", OWNER_RID)
    assert sig.owner_set is True
    _record(agent, msgs, ask="cheer up")
    v = _mood(agent)
    assert (v.label, v.intensity, v.source) == ("cheerful", 2, "owner")
    assert (v.p, v.e) == (pytest.approx(0.60, abs=1e-3), pytest.approx(0.45, abs=1e-3))   # exactly the anchor


@pytest.mark.parametrize("args", [{"action": "set_mood", "manner": "sarcastic"}, {"action": "set_mood", "mood": "neutral"},
                                  {"action": "set_mood", "manner": "none"}])
async def test_setting_a_manner_or_neutral_also_holds_for_the_turn(tmp_path, owner, args):
    agent = _traj_agent(tmp_path)
    msgs = await _mood_turn(agent, [args], extra=[("execute", BAD)] * 4)
    _record(agent, msgs, ask="cheer up")
    assert _mood(agent).neutral and _mood(agent).p == 0.0


@pytest.mark.parametrize("args", [{"action": "set_mood", "mood": "chipper"}, {"action": "set_mood"},
                                  {"action": "mood"}, {"action": "stats"}])
async def test_a_set_mood_that_set_nothing_or_a_read_does_not_hold_the_turn(tmp_path, owner, args):
    agent = _traj_agent(tmp_path)
    msgs = await _mood_turn(agent, [args], extra=[("execute", BAD)] * 4)
    assert agent._mood_turn_signals(msgs, "cheer up", OWNER_RID).owner_set is False
    _record(agent, msgs, ask="cheer up")
    assert (_mood(agent).label, _mood(agent).intensity) == ("irritable", 2)


# ── the context-pressure lockdown is trouble of its own kind (the snapshot rule is §4CC's) ──

def test_a_turn_that_ended_under_the_lockdown_is_journalled_as_context_pressure(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    agent.context._ctx_pressure_lockdown = True
    _record(agent, [{"role": "user", "content": ASK}])
    assert (_mood(agent).p, _mood(agent).e) == (pytest.approx(-0.12), pytest.approx(0.08))
    cause = _mood(agent).recent[-1]["cause"]
    assert cause == "the context ran out of room (context pressure) in this turn" and "loop guard" not in cause


def test_the_hook_prefers_the_snapshot_over_the_live_flag(tmp_path, owner):
    """(Ported.) The SSE drain runs post-semaphore, where the live flag belongs
    to whichever request runs NOW. The snapshot wins — in both directions."""
    armed = _traj_agent(tmp_path / "a")
    armed.context._ctx_pressure_lockdown = False
    _record(armed, [{"role": "user", "content": ASK}], pressure_lockdown=True)
    assert _mood(armed).p == pytest.approx(-0.12)
    clear = _traj_agent(tmp_path / "b")
    clear.context._ctx_pressure_lockdown = True
    _record(clear, [{"role": "user", "content": ASK}], pressure_lockdown=False)
    assert _mood(clear).p == 0.0


async def test_the_stream_drain_uses_the_snapshot_not_the_live_flag(tmp_path, owner):
    """(Ported.) The REAL SSE drain, with a StreamState whose semaphore-time
    snapshot says False while the LIVE flag says True."""
    from tests.test_finalize_stream_pins import make_stream_agent, _make_stream_state
    a = make_stream_agent()
    a.context.self_model = SelfModel(root=tmp_path)
    seen = []
    real = a.context.self_model.note_owner_turn
    a.context.self_model.note_owner_turn = lambda sig, **kw: (seen.append(sig), real(sig, **kw))[1]
    a.context.trajectory_collector = MagicMock()
    a.context.trajectory_collector.append = MagicMock(return_value=None)
    a.context.turn_origin_label = None
    a.context.owner_memory_isolated = False
    a.context._replay_forbidden_tools = None
    a.context._guard_events = None
    a.context._ctx_pressure_lockdown = True

    async def final_stream(payload, use_coding=False):
        yield f"data: {json.dumps({'choices': [{'delta': {'content': 'A fine answer.'}}]})}\n\n".encode()
        yield b"data: [DONE]\n\n"

    a.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock()
    reg.is_cancelled.return_value = False
    ss = _make_stream_state(reg)                 # pressure_lockdown defaults False
    gen, _, _ = a._stream_final_generation(ss)
    async for _ in gen:
        pass
    assert seen, "the mood hook never fired on the streamed drain"
    assert seen[0].pressure is False and seen[0].guards == 0


# ── who may move it ─────────────────────────────────────────────────────────

BAD_TURN = [("execute", BAD)] * 4


@pytest.mark.parametrize("rid", ["probe-4tc", "sched-morning", "job-12", "sub-3", "sim-9", "bench-1",
                                 "replay-7", "SYSTEM", ""])
def test_only_an_owners_request_moves_the_mood(tmp_path, rid):
    agent = _traj_agent(tmp_path)
    A.note_guard_event(agent.context, rid, "futility")
    tok = request_id_context.set(rid)
    try:
        _record(agent, _turn(BAD_TURN), rid=rid)
        assert A.mood_moves_on_this_request(agent.context) is False
    finally:
        request_id_context.reset(tok)
    assert _mood(agent).neutral and _untouched(agent)


def test_a_probe_declared_by_its_origin_header_moves_nothing(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    tok = request_origin_context.set(ORIGIN_PROBE)
    try:
        _record(agent, _turn(BAD_TURN))
    finally:
        request_origin_context.reset(tok)
    assert _mood(agent).neutral


def test_a_members_turn_moves_nothing(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    tok = requester_role_context.set("member")
    try:
        _record(agent, _turn(BAD_TURN))
        assert A.mood_moves_on_this_request(agent.context) is False
    finally:
        requester_role_context.reset(tok)
    assert _mood(agent).neutral


@pytest.mark.parametrize("mark", ["sim", "bench", "leaf", "contained", "isolated", "read_only"])
def test_a_run_that_is_not_the_live_owner_population_moves_nothing(tmp_path, owner, mark):
    agent = _traj_agent(tmp_path)
    if mark in ("sim", "bench", "leaf"):
        agent.context.turn_origin_label = mark
    elif mark == "contained":
        agent.context._replay_forbidden_tools = frozenset({"deploy"})
    elif mark == "isolated":
        agent.context.owner_memory_isolated = True
    else:
        agent.context.simulated = True
    _record(agent, _turn(BAD_TURN))
    assert A.mood_moves_on_this_request(agent.context) is False
    assert _mood(agent).neutral


def test_the_owner_on_a_public_surface_still_moves_it(tmp_path, owner):
    """"A real owner request on any surface"."""
    from ghost_agent.utils.logging import reply_surface_context
    agent = _traj_agent(tmp_path)
    tok = reply_surface_context.set("public")
    try:
        _record(agent, _turn(BAD_TURN))
    finally:
        reply_surface_context.reset(tok)
    assert (_mood(agent).label, _mood(agent).intensity) == ("irritable", 2)


def test_a_turn_cut_from_outside_moves_nothing(tmp_path, owner):
    """(Ported from §4IC: an abort from outside is not a verdict on my work.)
    A Stop, a disconnect, a restart mid-request: recorded, and the mood stays."""
    agent = _traj_agent(tmp_path)
    A.note_guard_event(agent.context, OWNER_RID, "futility")
    GhostAgent._record_aborted_turn(agent, req_id=OWNER_RID, reason="cancelled: client disconnected",
                                    messages=_turn(BAD_TURN), user_request=ASK, model="m")
    assert agent.context.trajectory_collector.append.call_count == 1       # the turn IS recorded
    assert _mood(agent).neutral and _untouched(agent)
    _record(agent, _turn(BAD_TURN))                                        # the same turn, finished: it moves
    assert _mood(agent).label == "irritable"


def test_a_guard_that_cannot_be_asked_moves_nothing(tmp_path, owner, monkeypatch):
    agent = _traj_agent(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("no kind")
    import ghost_agent.utils.logging as GL
    monkeypatch.setattr(GL, "request_kind", boom)
    assert A.mood_moves_on_this_request(agent.context) is False
    assert A.mood_is_heard_on_this_request(agent.context) is False
    _record(agent, _turn(BAD_TURN))
    assert _mood(agent).neutral


def test_a_mood_fault_never_costs_the_turn_its_record(tmp_path, owner):
    agent = _traj_agent(tmp_path)

    def boom(*a, **k):
        raise OSError("disk gone")
    agent.context.self_model.note_owner_turn = boom
    _record(agent, _turn(BAD_TURN))
    assert agent.context.trajectory_collector.append.call_count == 1


def test_a_mock_self_model_is_not_a_self_model(tmp_path, owner):
    agent = _traj_agent(tmp_path)
    agent.context.self_model = MagicMock()
    assert A._mood_model(agent.context) is None
    _record(agent, _turn(BAD_TURN))
    agent.context.self_model.note_owner_turn.assert_not_called()
    agent.context.self_model = SelfModel(root=tmp_path, enabled=False)
    assert A._mood_model(agent.context) is None


# ── the owner's thumbs ──────────────────────────────────────────────────────

def _labelled(tmp_path, *, task_kind="user_request", role="", age_h=0.0, stamp=None):
    coll = TrajectoryCollector(root=tmp_path / "traj", session_id="t-session")
    t = Trajectory(session_id="r1", user_request=ASK, final_response="done", task_kind=task_kind,
                   outcome=Outcome.UNKNOWN.value,
                   extra={"req_id": "r1", **({"requester_role": role} if role else {})})
    t.timestamp = stamp if stamp is not None else (
        (dt.datetime.utcnow() - dt.timedelta(hours=age_h)).isoformat() + "Z")
    assert coll.append(t) is not None
    sm = SelfModel(root=tmp_path / "self")
    ctx = SimpleNamespace(trajectory_collector=coll, _recent_trajectories_for_correction={},
                          self_model=sm, turn_origin_label=None)
    return SimpleNamespace(context=ctx), sm


@pytest.mark.parametrize("signal,move,label", [("positive", (0.25, 0.0), "content"),
                                               ("negative", (-0.30, 0.0), "down")])
def test_the_owners_thumb_moves_the_mood(tmp_path, signal, move, label):
    agent, sm = _labelled(tmp_path)
    assert F.apply_human_label(agent, "r1", signal, source="web")["ok"]
    v = sm.mood()
    assert (v.p, v.e) == move and (v.label, v.intensity) == (label, 1)
    assert v.cause == f"the owner gave a thumbs-{'up' if signal == 'positive' else 'down'}"


def test_the_same_thumb_again_is_not_a_second_thumb(tmp_path):
    agent, sm = _labelled(tmp_path)
    for _ in range(3):
        assert F.apply_human_label(agent, "r1", "positive", source="web")["ok"]
    assert sm.mood().p == 0.25
    F.apply_human_label(agent, "r1", "negative", source="web")           # a changed mind is a new thumb
    assert sm.mood().p == pytest.approx(-0.05, abs=0.01)


@pytest.mark.parametrize("source", sorted(F.OWNER_LABEL_SOURCES))
def test_every_client_the_owner_uses_counts(tmp_path, source):
    agent, sm = _labelled(tmp_path)
    assert F.apply_human_label(agent, "r1", "positive", source=source)["ok"]
    assert sm.mood().p == 0.25


@pytest.mark.parametrize("age_h,moves", [(0.1, True), (23.5, True), (24.5, False), (24 * 9, False)])
def test_a_thumb_moves_the_mood_only_for_a_turn_of_the_last_day(tmp_path, age_h, moves):
    """Review: a thumb on last week's reply is housekeeping. The label itself
    is recorded either way."""
    agent, sm = _labelled(tmp_path, age_h=age_h)
    assert F.apply_human_label(agent, "r1", "negative", source="web")["ok"]
    assert (sm.mood().p == -0.30) is moves and (sm.state.state.mood is not None) is moves


@pytest.mark.parametrize("stamp", ["", "last tuesday", "2026-13-45T99:00:00Z"])
def test_a_thumb_on_a_turn_whose_time_cannot_be_read_moves_nothing(tmp_path, stamp):
    agent, sm = _labelled(tmp_path, stamp=stamp)
    F.apply_human_label(agent, "r1", "positive", source="web")
    assert sm.state.state.mood is None


@pytest.mark.parametrize("ahead_h", [0.1, 2.0, 24.0 * 3])
def test_a_thumb_on_a_turn_dated_in_the_future_moves_nothing(tmp_path, ahead_h):
    """Review r2: a negative age was read as "just now"."""
    agent, sm = _labelled(tmp_path, age_h=-ahead_h)
    assert F.apply_human_label(agent, "r1", "positive", source="web")["ok"]        # the label is recorded
    assert sm.state.state.mood is None


def test_the_day_is_twenty_four_hours():
    now = dt.datetime(2026, 11, 6, 12, 0, tzinfo=dt.timezone.utc)
    turn = lambda iso: SimpleNamespace(timestamp=iso)            # noqa: E731
    assert F.THUMB_MAX_TURN_AGE_S == 24 * 3600.0
    assert F.turn_is_recent_enough(turn("2026-11-05T12:00:00Z"), now=now) is True          # exactly a day
    assert F.turn_is_recent_enough(turn("2026-11-05T11:59:59Z"), now=now) is False
    assert F.turn_is_recent_enough(turn("2026-11-06T12:00:00Z"), now=now) is True           # this instant
    assert F.turn_is_recent_enough(turn("2026-11-06T12:00:01Z"), now=now) is False          # a second ahead
    assert F.turn_is_recent_enough(turn("2026-11-06T11:00:00Z"), now=now.replace(tzinfo=None)) is True
    assert F.turn_is_recent_enough(SimpleNamespace(), now=now) is False
    assert F.turn_is_recent_enough(None, now=now) is False


def test_the_owners_sources_are_a_closed_list():
    assert F.OWNER_LABEL_SOURCES == frozenset({"web", "clockwork", "cli", "slack:owner"})
    assert F.is_owner_label_source(" WEB ") and not F.is_owner_label_source(None)


@pytest.mark.parametrize("source", ["", "api", "slack:requester", "slack:U0AAAA", "script", "nightly-eval", "webhook"])
def test_a_script_or_another_persons_label_does_not_move_the_mood(tmp_path, source):
    agent, sm = _labelled(tmp_path)
    assert F.apply_human_label(agent, "r1", "negative", source=source)["ok"]      # the label itself is recorded
    assert sm.mood().neutral and not sm.state.path.exists()


def test_a_thumb_on_a_members_turn_is_refused_and_moves_nothing(tmp_path):
    agent, sm = _labelled(tmp_path, role="member")
    assert F.apply_human_label(agent, "r1", "negative", source="web")["code"] == "member_turn"
    assert sm.mood().neutral


@pytest.mark.parametrize("task_kind", ["probe", "internal", "bench", "leaf"])
def test_a_thumb_on_a_row_that_is_not_an_owner_turn_moves_nothing(tmp_path, task_kind):
    agent, sm = _labelled(tmp_path, task_kind=task_kind)
    assert F.apply_human_label(agent, "r1", "negative", source="web")["ok"]
    assert sm.mood().neutral


def test_a_thumb_in_a_simulated_context_moves_nothing(tmp_path):
    agent, sm = _labelled(tmp_path)
    agent.context.turn_origin_label = "sim"
    assert F.apply_human_label(agent, "r1", "positive", source="web")["ok"]
    assert sm.mood().neutral


def test_a_mood_fault_does_not_turn_a_recorded_label_into_an_error(tmp_path):
    agent, sm = _labelled(tmp_path)

    def boom(*a, **k):
        raise OSError("disk gone")
    sm.note_owner_reaction = boom
    assert F.apply_human_label(agent, "r1", "positive", source="web") == {
        "ok": True, "trajectory_id": agent.context.trajectory_collector and
        F.find_trajectory_for_request(agent.context.trajectory_collector, "r1").id, "outcome": "passed"}


# ── a correction of the last reply ──────────────────────────────────────────

HISTORY = [{"role": "user", "content": "how old is Bruno?"},
           {"role": "assistant", "content": "Bruno is seven."},
           {"role": "user", "content": "no, he is nine"}]


def _judge_agent(tmp_path, verdict):
    agent = _traj_agent(tmp_path)
    agent._prior_turn_changed_mid_turn = lambda messages: False
    agent._adjudicate_correction = AsyncMock(return_value=verdict)
    agent._maybe_promote_prior_turn_via_user_correction = MagicMock()
    return agent


async def test_a_confirmed_correction_lowers_the_mood(tmp_path, owner):
    agent = _judge_agent(tmp_path, True)
    await agent._promote_prior_turn_after_judge(HISTORY, "no, he is nine")
    v = _mood(agent)
    assert (v.p, v.e, v.label, v.intensity) == (-0.20, 0.0, "down", 1)
    assert v.cause == "the owner corrected my last reply"
    agent._maybe_promote_prior_turn_via_user_correction.assert_called_once()      # the promotion is untouched


@pytest.mark.parametrize("verdict", [False, None])
async def test_no_positive_verdict_no_move(tmp_path, owner, verdict):
    agent = _judge_agent(tmp_path, verdict)
    await agent._promote_prior_turn_after_judge(HISTORY, "no, he is nine")
    assert _mood(agent).neutral
    agent._maybe_promote_prior_turn_via_user_correction.assert_called_once()


async def test_a_prior_turn_that_changed_mid_turn_is_never_judged(tmp_path, owner):
    agent = _judge_agent(tmp_path, True)
    agent._prior_turn_changed_mid_turn = lambda messages: True
    await agent._promote_prior_turn_after_judge(HISTORY, "no, he is nine")
    assert _mood(agent).neutral


@pytest.mark.parametrize("rid", ["probe-4tc", "sched-morning", "job-12"])
async def test_a_correction_outside_an_owners_request_moves_nothing(tmp_path, rid):
    agent = _judge_agent(tmp_path, True)
    tok = request_id_context.set(rid)
    try:
        await agent._promote_prior_turn_after_judge(HISTORY, "no, he is nine")
    finally:
        request_id_context.reset(tok)
    assert _mood(agent).neutral


# ── thanks, through the real handle_chat ────────────────────────────────────

def _chat_agent(tmp_path, monkeypatch, kind="owner"):
    (tmp_path / "memory").mkdir(exist_ok=True)
    context = MagicMock()
    context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "You're welcome."}}]})
    context.args.smart_memory = 0.0
    context.args.temperature = 0.1
    context.args.max_context = 8000
    context.args.perfect_it = False
    context.sandbox_dir = "/tmp"
    context.memory_dir = str(tmp_path / "memory")
    context.memory_system = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.turn_origin_label = None
    context.owner_memory_isolated = False
    context._replay_forbidden_tools = None
    context.self_model = SelfModel(root=tmp_path / "self")
    agent = GhostAgent(context)
    agent.available_tools = {}
    from ghost_agent.utils import logging as L
    monkeypatch.setattr(L, "request_kind", lambda *a, **k: kind)
    monkeypatch.setattr(A, "reply_is_public", lambda: False)
    return agent


async def _say(agent, text, rid="web-4tc-thanks"):
    return await agent.handle_chat({"messages": [{"role": "user", "content": text}], "stream": False}, [],
                                   request_id=rid)


@pytest.mark.asyncio
async def test_thanks_in_the_owners_message_lifts_the_mood_once(tmp_path, monkeypatch):
    agent = _chat_agent(tmp_path, monkeypatch)
    await _say(agent, "thanks, perfect — great job!")                      # three praises, one request
    v = agent.context.self_model.mood()
    assert (v.p, v.e) == (pytest.approx(0.10, abs=1e-3), 0.0)
    assert [r["cause"] for r in v.recent] == ["the owner said thanks"]


@pytest.mark.asyncio
async def test_a_message_without_thanks_moves_nothing(tmp_path, monkeypatch):
    agent = _chat_agent(tmp_path, monkeypatch)
    for text in ("what time does Bruno's vet open?", "make the report look awesome",
                 "thanks to the cache it works now, right?", "no, thanks"):
        await _say(agent, text)
    assert agent.context.self_model.mood().p == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["probe", "member", "background", "job", "test", "system"])
async def test_thanks_from_anyone_but_the_owner_moves_nothing(tmp_path, monkeypatch, kind):
    agent = _chat_agent(tmp_path, monkeypatch, kind=kind)
    await _say(agent, "thanks, perfect")
    assert agent.context.self_model.mood().p == 0.0


# ── a long quiet: nothing is written into the mood; the owner's return ends it ──

class _as:
    """Read as a request of another kind (the fixture's id is the owner's)."""
    def __init__(self, rid, role=""):
        self.rid, self.role = rid, role

    def __enter__(self):
        self.t = (request_id_context.set(self.rid), requester_role_context.set(self.role))

    def __exit__(self, *exc):
        request_id_context.reset(self.t[0])
        requester_role_context.reset(self.t[1])


def _quiet_agent(tmp_path, hours):
    """An agent whose owner was last here `hours` ago (the persisted stamp)."""
    agent = _traj_agent(tmp_path)
    st = agent.context.self_model.state
    st.mark_owner_turn(now=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours))
    return agent, st


def _stamp(st):
    return json.loads(st.path.read_text())["last_owner_turn_at"]


def test_the_owners_return_ends_the_quiet_and_carries_no_boredom_into_his_turn(tmp_path, owner):
    """Review: boredom written into the pair at his return left 13 % of the
    owner's real turns starting neutral. Now the quiet simply ends."""
    agent, st = _quiet_agent(tmp_path, 13.0)
    with _as("sched-morning-news"):
        assert (_mood(agent).label, _mood(agent).intensity) == ("bored", 2)     # the night's report hears it
    before = _stamp(st)
    agent._mood_owner_arrived()
    assert _stamp(st) > before                                           # his presence is stamped …
    assert st.state.mood is None                                         # … and nothing went into the pair
    for rid in (OWNER_RID, "sched-morning-news", "probe-1"):
        with _as(rid):
            assert _mood(agent).neutral                                  # for anyone: the quiet is over
    assert [(r["label"], r["intensity"], r["kind"]) for r in _rows(agent)] == [
        ("bored", 1, "quiet"), ("bored", 2, "quiet"), ("neutral", 0, "quiet_end")]
    agent._mood_owner_arrived()                                          # his next message: no second row
    assert len(_rows(agent)) == 3


def test_the_owner_is_never_answered_bored_even_before_his_arrival_is_stamped(tmp_path, owner):
    agent, st = _quiet_agent(tmp_path, 13.0)
    assert A._mood_tone_block(agent.context) == "" and _mood(agent).neutral        # the owner's own request
    assert _rows(agent) == []                                            # and his read wrote no row
    with _as("slack-m-1", role="member"):
        assert A._mood_tone_block(agent.context) == M.tone_line("bored", 2) + "\n\n"


def test_an_owners_request_that_follows_no_boredom_writes_no_quiet_row(tmp_path, owner):
    agent, st = _quiet_agent(tmp_path, 2.0)
    agent._mood_owner_arrived()
    assert _rows(agent) == [] and st.state.mood is None


@pytest.mark.parametrize("mark", ["leaf", "contained", "isolated", "sim"])
def test_a_run_that_is_not_the_live_agent_ends_no_quiet_even_under_the_owners_id(tmp_path, owner, mark):
    agent, st = _quiet_agent(tmp_path, 13.0)
    before = _stamp(st)
    if mark in ("leaf", "sim"):
        agent.context.turn_origin_label = mark
    elif mark == "contained":
        agent.context._replay_forbidden_tools = frozenset({"deploy"})
    else:
        agent.context.owner_memory_isolated = True
    agent._mood_owner_arrived()
    _record(agent, _turn(BAD_TURN))
    assert _stamp(st) == before and _rows(agent) == []


@pytest.mark.parametrize("rid", ["probe-4tc", "sched-morning", "job-12", "sub-3", "sim-9", "bench-1"])
def test_a_request_that_is_not_the_owners_neither_ends_the_quiet_nor_stamps_a_turn(tmp_path, rid):
    agent, st = _quiet_agent(tmp_path, 13.0)
    before = _stamp(st)
    tok = request_id_context.set(rid)
    try:
        agent._mood_owner_arrived()
        _record(agent, _turn(BAD_TURN), rid=rid)
        assert _mood(agent).label == "bored"                              # still read off the running quiet
    finally:
        request_id_context.reset(tok)
    assert _stamp(st) == before
    assert [r["kind"] for r in _rows(agent)] == ["quiet", "quiet"]       # no "quiet ended"


def test_a_members_request_neither_ends_the_quiet_nor_stamps_a_turn(tmp_path, owner):
    agent, st = _quiet_agent(tmp_path, 13.0)
    before = _stamp(st)
    with _as("slack-m-1", role="member"):
        agent._mood_owner_arrived()
        _record(agent, _turn(BAD_TURN), rid="slack-m-1")
    assert _stamp(st) == before and st.state.mood is None


def test_the_end_of_an_owners_turn_stamps_it_so_the_next_quiet_counts_from_there(tmp_path, owner):
    agent, st = _quiet_agent(tmp_path, 13.0)
    before = _stamp(st)
    _record(agent, [{"role": "user", "content": ASK}])                   # a plain chat turn: no event …
    assert _stamp(st) > before and st.state.mood is None                 # … but the owner was here
    with _as("sched-x"):
        assert _mood(agent).neutral


def test_an_owners_turn_that_was_cut_still_says_the_owner_was_here(tmp_path, owner):
    agent, st = _quiet_agent(tmp_path, 13.0)
    before = _stamp(st)
    GhostAgent._record_aborted_turn(agent, req_id=OWNER_RID, reason="cancelled: client disconnected",
                                    messages=_turn(BAD_TURN), user_request=ASK, model="m")
    assert _stamp(st) > before and st.state.mood is None


def test_an_owners_request_also_writes_down_a_mood_that_wore_off(tmp_path, owner):
    """The journal gets its "it wore off" row when the owner next writes,
    dated when it happened."""
    agent, st = _quiet_agent(tmp_path, 0.2)
    then = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=9)
    st.nudge(-0.9, 0.45, "6 tool calls failed in this turn", M.KIND_TURN, now=then)
    agent._mood_owner_arrived()
    rows = _rows(agent)
    assert [(r["label"], r["intensity"], r["kind"]) for r in rows] == [("irritable", 3, "turn"), ("neutral", 0, "decay")]
    wore_off = M.parse_iso_utc(rows[1]["at"])
    assert dt.timedelta(hours=4, minutes=55) < wore_off - then < dt.timedelta(hours=5, minutes=5)


def test_the_heartbeat_writes_nothing_of_the_mood_and_keeps_its_own_clock(tmp_path, owner):
    """The agent's activity clocks are the loop's; the mood never writes them,
    and the heartbeat (called many times a turn) never writes the mood."""
    agent, st = _quiet_agent(tmp_path, 13.0)
    last = dt.datetime.now() - dt.timedelta(hours=13)
    agent.context.last_owner_activity_time = last
    raw = st.path.read_bytes()
    for _ in range(3):
        agent._heartbeat()
    assert agent.context.last_owner_activity_time > last                 # the loop's clock, as before
    assert st.path.read_bytes() == raw and _rows(agent) == []


def test_an_arrival_fault_never_stops_the_request(tmp_path, owner):
    agent, st = _quiet_agent(tmp_path, 13.0)

    def boom(**kw):
        raise OSError("disk gone")
    agent.context.self_model.state.owner_arrived = boom
    agent._mood_owner_arrived()                                          # no raise
    agent.context.self_model = MagicMock()
    agent._mood_owner_arrived()
    agent.context.self_model.owner_arrived.assert_not_called()           # a mock is no self-model


@pytest.mark.asyncio
async def test_through_handle_chat_the_owners_message_ends_the_quiet_before_he_is_answered(tmp_path, monkeypatch):
    agent = _chat_agent(tmp_path, monkeypatch)
    st = agent.context.self_model.state
    st.mark_owner_turn(now=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=13))
    monkeypatch.setattr(__import__("ghost_agent.utils.logging", fromlist=["x"]), "request_kind",
                        lambda *a, **k: "background")
    assert st.mood().label == "bored"                                    # a scheduled read in the night
    before = _stamp(st)
    monkeypatch.setattr(__import__("ghost_agent.utils.logging", fromlist=["x"]), "request_kind",
                        lambda *a, **k: "owner")
    await _say(agent, "good morning — what's new?")
    assert _stamp(st) > before and st.state.mood is None
    assert [r["kind"] for r in st.mood_history()] == ["quiet", "quiet", "quiet_end"]
    monkeypatch.setattr(__import__("ghost_agent.utils.logging", fromlist=["x"]), "request_kind",
                        lambda *a, **k: "background")
    assert st.mood().neutral


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["probe", "member", "background", "job", "test", "system"])
async def test_through_handle_chat_nobody_elses_message_ends_the_quiet(tmp_path, monkeypatch, kind):
    agent = _chat_agent(tmp_path, monkeypatch, kind=kind)
    st = agent.context.self_model.state
    st.mark_owner_turn(now=dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=13))
    before = _stamp(st)
    await _say(agent, "good morning — what's new?")
    assert _stamp(st) == before
    assert "quiet_end" not in [r["kind"] for r in st.mood_history()]


# ── nothing else writes the mood ────────────────────────────────────────────

#: every method that writes the mood or its owner stamp ("settle" is left out of the scan: other
#: objects of the loop have a method of that name; the state's own is called inside selfhood only)
WRITERS = {"note_owner_turn", "note_owner_reaction", "owner_arrived", "owner_turn_ended", "mark_owner_turn",
           "nudge", "set_owner_mood", "reset_mood", "set_manner"}


def _writer_calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in WRITERS:
                out.add((fn.name, n.func.attr))
    return out


def test_the_mood_has_exactly_these_writers_outside_its_own_module():
    """An enumeration of the tree: who calls a mood writer — in EVERY function
    of every file outside the selfhood package."""
    src = ROOT / "src" / "ghost_agent"
    found = {}
    for path in src.rglob("*.py"):
        if path.parent.name == "selfhood":
            continue
        calls = _writer_calls(path)
        if calls:
            found[str(path.relative_to(src))] = calls
    assert found == {
        "core/agent.py": {("_record_turn_trajectory", "owner_turn_ended"),
                          ("_record_turn_trajectory", "note_owner_turn"),
                          ("_mood_owner_arrived", "owner_arrived"),
                          ("_mood_owner_reaction", "note_owner_reaction")},
        "core/feedback.py": {("apply_human_label", "note_owner_reaction")},
        "tools/introspect.py": {("_set_mood", "set_owner_mood"), ("_set_mood", "reset_mood"),
                                ("_set_mood", "set_manner")},
    }


def test_an_owners_arrival_is_noted_once_where_the_request_arrives_and_not_in_the_heartbeat():
    tree = ast.parse((ROOT / "src" / "ghost_agent" / "core" / "agent.py").read_text(encoding="utf-8"))
    callers = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n = sum(1 for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "_mood_owner_arrived")
            if n:
                callers[fn.name] = n
    assert callers == {"handle_chat": 1}


def test_the_process_attaches_the_owner_clock_where_it_builds_the_self_model():
    """The fallback for a state file that holds no owner stamp yet (a first
    boot, an old file). `main.py` attaches the clock once, as a function of
    the context — read off the tree."""
    tree = ast.parse((ROOT / "src" / "ghost_agent" / "main.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "attach_owner_clock"]
    assert len(calls) == 1 and isinstance(calls[0].args[0], ast.Lambda)
    inner = [n.func.id for n in ast.walk(calls[0].args[0]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert inner == ["_owner_quiet_seconds"]


def test_the_owner_clock_is_written_in_one_place_and_never_by_the_mood():
    src = ROOT / "src" / "ghost_agent"
    writes = []
    for path in src.rglob("*.py"):
        for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if isinstance(t, ast.Attribute) and t.attr == "last_owner_activity_time":
                        writes.append(str(path.relative_to(src)))
    assert writes == ["core/agent.py"]
