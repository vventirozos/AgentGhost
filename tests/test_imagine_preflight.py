"""§4CL I1 — the Imagine pre-flight steer.

Today `foresight_note` appends precedent to a result AFTER the call has
already failed. I1 moves the same knowledge BEFORE dispatch: the call is
not run, the model is handed the precedent, and it gets one revision.

The whole mechanism is gated on `core/imagination.gate_allows`, which as
of 2026-08-22 enables ZERO buckets — so in production this path is inert
by construction. That is the design (the pre-registered stop rule fired,
see §4CL I0), which makes these pins the only place the behaviour is
observable at all. Two things they have to establish:

  * with the gate closed — the live state — the dispatch path is
    BIT-IDENTICAL to before: no deferral, no trigger mark, no log line;
  * with the gate open, every guard in the rule is independently
    necessary, and the deferral cannot become a loop.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core import imagination as IM
from ghost_agent.core.agent import GhostAgent, TurnState


# ------------------------------------------------------------------ #
# Fixtures                                                           #
# ------------------------------------------------------------------ #

@pytest.fixture(autouse=True)
def _imagine_env(monkeypatch, tmp_path):
    IM.reset_gate_cache_for_tests()
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_IMAGINE", "1")
    monkeypatch.delenv("GHOST_IMAGINE_PREFLIGHT", raising=False)
    yield
    IM.reset_gate_cache_for_tests()


def _open_the_gate(tmp_path, tool="execute", tclass="cmd:python3"):
    """Write a gate.json in which one bucket qualifies. Built through the
    REAL builder from a synthetic ledger, not hand-written, so the test
    cannot pass against a gate shape the builder no longer produces."""
    def _r(steerable, ok):
        support, fails = (6, 5) if steerable else (6, 1)
        return {"tool": tool, "tclass": tclass, "basis": "exact",
                "support": support, "fails": fails,
                "p_fail": round((fails + 1) / (support + 2), 4),
                "ok": ok, "pred_err": "filenotfounderror"}

    rows = ([_r(False, True)] * 38 + [_r(False, False)] * 2
            + [_r(True, True)] * 2 + [_r(True, False)] * 18)
    doc = IM.build_gate(rows, write=True)
    assert doc["buckets"][IM.bucket_key(tool, tclass)]["enabled"] is True
    IM.reset_gate_cache_for_tests()


class _Pred:
    """A foresight Prediction stand-in with the fields the rule reads."""

    def __init__(self, *, basis="exact", support=6, fails=4, p_fail=0.71,
                 predicted_error="filenotfounderror"):
        self.basis = basis
        self.support = support
        self.fails = fails
        self.p_fail = p_fail
        self.predicted_error = predicted_error
        self.simulation = True


def _agent(monkeypatch, *, arm="treatment", pred=None):
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)
    agent.available_tools = {}
    agent.disabled_tools = set()
    monkeypatch.setattr(
        "ghost_agent.core.foresight.predict_for_call",
        lambda **kw: (_Pred() if pred is None else pred))
    marks = []
    monkeypatch.setattr("ghost_agent.core.experiments.arm_for",
                        lambda *a, **k: arm)
    monkeypatch.setattr("ghost_agent.core.experiments.mark_trigger",
                        lambda *a, **k: marks.append(a))
    agent._marks = marks
    return agent


_ARGS = {"command": "python3 solve.py"}


# ------------------------------------------------------------------ #
# The gate is the master switch                                      #
# ------------------------------------------------------------------ #

def test_a_closed_gate_never_defers(monkeypatch, tmp_path):
    """THE live case: the gate enables nothing, so the strongest possible
    precedent must still dispatch. If this ever fails, the steer has
    escaped its own calibration."""
    agent = _agent(monkeypatch)
    assert IM.gate_allows("execute", "cmd:python3") is False
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "h1", "req-1") is None
    assert agent._marks == [], "control/treatment marked with no gate"


def test_an_open_gate_defers_and_names_the_precedent(monkeypatch, tmp_path):
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch)
    note = agent._imagine_preflight_note("execute", _ARGS, "h1", "req-1")
    assert note is not None
    assert note.startswith(agent.IMAGINE_PREFLIGHT_MARKER)
    assert "4 of 6" in note and "filenotfounderror" in note
    # and the model is told the re-issue WILL run — a steer that reads as
    # a permanent block is how a turn spins to its cap.
    assert "re-issue" in note
    assert agent._marks, "the trigger was not marked"


def test_a_different_bucket_is_still_closed(monkeypatch, tmp_path):
    """The gate is per (tool, tclass). Opening one bucket must not open
    the tool."""
    _open_the_gate(tmp_path, tool="execute", tclass="cmd:python3")
    agent = _agent(monkeypatch)
    assert agent._imagine_preflight_note(
        "execute", {"command": "curl https://x"}, "h1", "r") is None
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "h2", "r") is not None


# ------------------------------------------------------------------ #
# Kill switches                                                      #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("env,value", [
    ("GHOST_IMAGINE", "0"),
    ("GHOST_IMAGINE_PREFLIGHT", "0"),
])
def test_either_kill_switch_disarms_it(monkeypatch, tmp_path, env, value):
    _open_the_gate(tmp_path)
    monkeypatch.setenv(env, value)
    agent = _agent(monkeypatch)
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "h1", "r") is None


def test_the_master_flag_defaults_off(monkeypatch, tmp_path):
    """A feature that arrives armed is a feature nobody chose."""
    _open_the_gate(tmp_path)
    monkeypatch.delenv("GHOST_IMAGINE", raising=False)
    agent = _agent(monkeypatch)
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "h1", "r") is None


# ------------------------------------------------------------------ #
# Each precedent condition is independently necessary                #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("kw", [
    {"basis": "tool"},          # coarsest level — not specific enough
    {"basis": "none"},
    {"support": 2},             # below the support floor
    {"fails": 1},               # one failure can be an echoed false positive
    {"predicted_error": ""},    # nothing to tell the model
])
def test_weak_precedent_dispatches(monkeypatch, tmp_path, kw):
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch, pred=_Pred(**kw))
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "h1", "r") is None


def test_an_exact_tie_is_not_a_failure_claim(monkeypatch, tmp_path):
    """The C1 finding: `p_fail = (f+1)/(n+2) >= 0.5` reduces to `2f >= n`
    — a bare majority INCLUSIVE of the tie — because the Laplace prior's
    mean IS 0.5 and contributes zero shrinkage at the boundary. A cell
    with 2 failures in 4 tries reads p_fail = 0.5000 and used to steer.
    On the live ledger those coin-flip cells were 37.5% of the
    "predicted-fail" population and carried ALL of its apparent
    precision; the strict claims scored 0.067, below the base rate."""
    _open_the_gate(tmp_path)
    tie = _Pred(support=4, fails=2, p_fail=0.5)
    assert (2 * tie.fails) == tie.support          # the tie, exactly
    assert tie.p_fail == 0.5
    agent = _agent(monkeypatch, pred=tie)
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "h1", "r") is None
    # …while one more failure in the same cell IS a claim.
    agent2 = _agent(monkeypatch, pred=_Pred(support=4, fails=3, p_fail=0.67))
    assert agent2._imagine_preflight_note(
        "execute", _ARGS, "h1", "r") is not None


def test_the_gate_and_the_consumer_share_one_definition(monkeypatch,
                                                        tmp_path):
    """Two definitions of "the population a steer acts on", one measured
    and one executed, is how a gate certifies a statistic about rows the
    steer will never touch."""
    from ghost_agent.core.imagination import is_steerable_row

    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch)
    for kw, steerable in (
        ({"basis": "tool"}, False),
        ({"support": 2}, False),
        ({"fails": 1}, False),
        ({"predicted_error": ""}, False),
        ({"support": 4, "fails": 2}, False),      # the tie
        ({}, True),
    ):
        pred = _Pred(**kw)
        as_row = {"basis": pred.basis, "support": pred.support,
                  "fails": pred.fails, "p_fail": pred.p_fail,
                  "pred_err": pred.predicted_error}
        assert is_steerable_row(as_row) is steerable, kw
        agent._imagine_preflight_seen = None       # reset the loop guard
        fired = _agent(monkeypatch, pred=pred)._imagine_preflight_note(
            "execute", _ARGS, "h1", "r") is not None
        assert fired is steerable, kw


def test_a_missing_prediction_dispatches(monkeypatch, tmp_path):
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch)
    monkeypatch.setattr("ghost_agent.core.foresight.predict_for_call",
                        lambda **kw: None)
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "h1", "r") is None


def test_the_prediction_is_flagged_as_a_simulation(monkeypatch, tmp_path):
    """It is never resolved (the call does not run), but a stray resolve
    must not be able to write a synthetic transition into the production
    index — the §4J lesson, applied pre-emptively."""
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch)
    seen = {}

    def _capture(**kw):
        seen.update(kw)
        return _Pred()

    monkeypatch.setattr("ghost_agent.core.foresight.predict_for_call",
                        _capture)
    agent._imagine_preflight_note("execute", _ARGS, "h1", "r")
    assert seen["simulation"] is True


# ------------------------------------------------------------------ #
# The arm                                                            #
# ------------------------------------------------------------------ #

def test_control_dispatches_but_is_still_trigger_marked(monkeypatch,
                                                        tmp_path):
    """Both arms must be marked or the trigger rates are not comparable,
    and the §4CD lesson says the TRIGGERED subset is the only block worth
    reading."""
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch, arm="control")
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "h1", "r") is None
    assert agent._marks, "control was not marked — the arms are not comparable"


def test_an_unenrolled_turn_dispatches(monkeypatch, tmp_path):
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch, arm=None)
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "h1", "r") is None
    assert agent._marks == []


# ------------------------------------------------------------------ #
# The loop guard — a deferral must not become a spin                 #
# ------------------------------------------------------------------ #

def test_the_same_call_is_deferred_once_then_dispatched(monkeypatch,
                                                        tmp_path):
    """One revision, then the re-issue runs. Without this the model can
    re-propose the identical call forever and the turn never acts."""
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch)
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "same-hash", "req-1") is not None
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "same-hash", "req-1") is None


def test_a_request_cannot_be_deferred_more_than_twice(monkeypatch,
                                                      tmp_path):
    """A model that revises INTO a second flagged call would chain. Two
    deferrals is a steer; five is a turn spent being steered into a wall."""
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch)
    got = [agent._imagine_preflight_note("execute", _ARGS, f"h{i}", "req-1")
           for i in range(4)]
    assert [g is not None for g in got] == [True, True, False, False]


def test_the_budget_is_per_request(monkeypatch, tmp_path):
    """Two conversations must not consume each other's budget."""
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch)
    for i in range(2):
        assert agent._imagine_preflight_note(
            "execute", _ARGS, f"a{i}", "req-A") is not None
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "a9", "req-A") is None
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "b0", "req-B") is not None


# ------------------------------------------------------------------ #
# Corpus hygiene                                                     #
# ------------------------------------------------------------------ #

def test_the_deferral_reads_as_synthetic_to_the_corpus(monkeypatch,
                                                       tmp_path):
    """A deferred call NEVER EXECUTED. Its message pairs with a tool_call
    in the reconstructed trajectory exactly like a real result, and its
    label would be INVERTED — a deferral on a repeatedly-failing target
    reads as a success. Recognising it as synthetic is what stops a steer
    from teaching the index that produced it."""
    from ghost_agent.core.foresight import is_synthetic_result

    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch)
    note = agent._imagine_preflight_note("execute", _ARGS, "h1", "r")
    assert is_synthetic_result(note) is True


def test_an_ordinary_tool_result_is_not_mistaken_for_a_deferral():
    from ghost_agent.core.foresight import is_synthetic_result
    assert is_synthetic_result("OK: wrote 3 lines to solve.py") is False
    assert is_synthetic_result("") is False


def test_the_helper_never_raises(monkeypatch, tmp_path):
    """A steer that breaks a turn is worse than no steer."""
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch)
    monkeypatch.setattr(
        "ghost_agent.core.foresight.predict_for_call",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("index exploded")))
    assert agent._imagine_preflight_note(
        "execute", _ARGS, "h1", "r") is None


# ------------------------------------------------------------------ #
# End-to-end through the real dispatch pipeline                      #
# ------------------------------------------------------------------ #

def _ts(**over):
    fields = dict(
        _constraint_steer_pending=None, _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False, _request_sys3_prev_justification="",
        consecutive_parse_errors=0, current_plan_json="",
        execution_failure_count=0, final_ai_content="", fname="",
        force_final_response=False, force_stop=False,
        forget_was_called=False, last_was_failure=True,
        preflight_blocks_this_request=0, request_sandbox_state="",
        transient_failure_count=0, tool_calls=[],
        msg={"role": "assistant", "content": ""}, ui_content="",
        parse_failure_reason="", model="test-model",
        last_user_content="run it", char_budget=4000, strikes=MagicMock(),
        task_tree=MagicMock(), _user_batch_intent=None,
        _request_constraints=[], repeated_action_steered=False,
        messages=[], seen_tools=set(), executed_idempotent=set(),
        raw_tools_called=set(), tool_usage={}, tools_run_this_turn=[],
        request_state=MagicMock(),
    )
    fields.update(over)
    return TurnState(**fields)


def _exec_batch(agent, ran):
    async def _run(**kwargs):
        ran.append(kwargs)
        return "EXIT CODE: 0\nok"

    agent.available_tools = {"execute": _run}
    return _ts(tool_calls=[{
        "id": "t1", "type": "function",
        "function": {"name": "execute",
                     "arguments": json.dumps(_ARGS)}}])


async def test_a_closed_gate_leaves_dispatch_bit_identical(monkeypatch,
                                                           tmp_path):
    """The OFF path, pinned: with no gate on disk the call runs, nothing
    synthetic is appended, and no counter moves."""
    agent = _agent(monkeypatch)
    ran = []
    ts = _exec_batch(agent, ran)
    await agent._dispatch_and_process_tool_batch(ts)
    assert len(ran) == 1, "the call did not dispatch"
    assert not any(m.get("_synthetic")
                   for m in ts.tools_run_this_turn if isinstance(m, dict))
    assert ts.execution_failure_count == 0


async def test_an_open_gate_defers_the_real_dispatch(monkeypatch, tmp_path):
    """Executed end-to-end: the tool is NOT called, the model gets the
    precedent as a tool message, and the deferral is marked synthetic."""
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch)
    ran = []
    ts = _exec_batch(agent, ran)
    await agent._dispatch_and_process_tool_batch(ts)
    assert ran == [], "the call was dispatched despite the deferral"
    tool_msgs = [m for m in ts.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"].startswith(agent.IMAGINE_PREFLIGHT_MARKER)
    assert tool_msgs[0]["tool_call_id"] == "t1"
    synth = [m for m in ts.tools_run_this_turn
             if isinstance(m, dict) and m.get("_synthetic")]
    assert len(synth) == 1


async def test_a_deferral_is_not_charged_as_a_failure(monkeypatch, tmp_path):
    """A deliberate steer that counted as a strike would spend the turn's
    error budget on its own advice — and six strikes ends the turn."""
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch)
    ran = []
    ts = _exec_batch(agent, ran)
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.execution_failure_count == 0
    assert ts.last_was_failure is False
    assert ts.preflight_blocks_this_request == 0   # the OTHER guard's counter


def test_the_deferral_budget_ring_is_bounded(monkeypatch, tmp_path):
    """A plain dict on a long-lived process grows forever — one entry per
    request, never cleared. Every other per-request cache in the turn
    loop is a capped ring."""
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch)
    for i in range(agent._IMAGINE_PREFLIGHT_RING * 3):
        agent._imagine_preflight_note("execute", _ARGS, "h0", f"req-{i}")
    assert len(agent._imagine_preflight_seen) <= agent._IMAGINE_PREFLIGHT_RING


def test_a_request_without_an_id_is_never_deferred(monkeypatch, tmp_path):
    """All context-less callers would otherwise SHARE one bucket, which
    exhausts after two deferrals and then disables the steer for every
    one of them, permanently, with no way to tell from outside."""
    _open_the_gate(tmp_path)
    agent = _agent(monkeypatch)
    for req in ("", "   ", None):
        assert agent._imagine_preflight_note(
            "execute", _ARGS, "h1", req) is None
    # …and a real request still works, i.e. the empty ones consumed
    # nothing.
    for i in range(2):
        assert agent._imagine_preflight_note(
            "execute", _ARGS, f"h{i}", "req-real") is not None
