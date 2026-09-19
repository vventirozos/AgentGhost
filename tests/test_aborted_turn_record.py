"""§4IB — a turn that did not finish is still a turn.

THE VANISHED TURN (req 21b295ef, 2026-09-17): 40 turns, 1800.0 s, ~100 LLM
calls — then `request finished` 0.4 s after the last `llm request`, and
NOTHING: no reply, no trajectory row, no calibration sample, no cause in
the log. The client (or the interface's 1800 s GHOST_CHAT_TIMEOUT) dropped
the connection, the task was cancelled mid-flight, and no handler recorded
it. Six long user turns this month left the same hole; two more ended in a
process shutdown with not even a "finished" line.

`_record_aborted_turn` writes the row a finished turn would, with the
partial output + `[ATTEMPT_ABORTED_TURN] Turn aborted: <reason>` (the
existing shape rule files it FAILED), and logs one line. It is called from
the Stop-button `except TurnCancelled`, from a new
`except asyncio.CancelledError` (which re-raises), and from the process
shutdown for every turn still running.

World where each pin fails: the recorder drops the marker or the reason,
either except branch stops calling it, CancelledError is swallowed
instead of re-raised, or shutdown stops stamping the registry.
"""
import ast
import inspect
import types
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import GhostAgent
from ghost_agent.distill.outcome_heuristics import classify_chat_outcome
from ghost_agent.distill.schema import Outcome, Trajectory


def _agent():
    a = GhostAgent.__new__(GhostAgent)
    a.context = types.SimpleNamespace(_pressure_lockdown=False)
    a._record_turn_trajectory = MagicMock(return_value=None)
    return a


def test_records_partial_plus_marker_and_reason():
    a = _agent()
    with patch("ghost_agent.core.agent.pretty_log"):
        a._record_aborted_turn(req_id="21b295ef", reason="cancelled: client disconnected",
                               messages=[{"role": "user", "content": "x"}],
                               user_request="use eckit", model="m", partial="partial text")
    kw = a._record_turn_trajectory.call_args.kwargs
    assert kw["req_id"] == "21b295ef" and kw["user_request"] == "use eckit" and kw["model"] == "m"
    assert kw["final_content"].startswith("partial text")
    assert "[ATTEMPT_ABORTED_TURN] Turn aborted: cancelled: client disconnected." in kw["final_content"]
    assert kw["messages"] == [{"role": "user", "content": "x"}]


def test_no_partial_still_records_the_marker():
    a = _agent()
    with patch("ghost_agent.core.agent.pretty_log"):
        a._record_aborted_turn(req_id="r", reason="process shutdown", messages=None,
                               user_request="preview", model="", partial="")
    assert a._record_turn_trajectory.call_args.kwargs["final_content"].startswith("[ATTEMPT_ABORTED_TURN]")
    assert a._record_turn_trajectory.call_args.kwargs["messages"] == []


def test_the_marker_shapes_the_row_failed():
    """The existing rule 1 — a runtime abort marker — files it, so no new
    outcome path exists to drift."""
    traj = Trajectory(user_request="use eckit",
                      final_response="partial\n\n[ATTEMPT_ABORTED_TURN] Turn aborted: cancelled: client disconnected.")
    c = classify_chat_outcome(traj)
    assert c.outcome == Outcome.FAILED.value
    assert "ATTEMPT_ABORTED_TURN" in c.reason


def test_logs_one_operator_line():
    a = _agent()
    with patch("ghost_agent.core.agent.pretty_log") as pl:
        a._record_aborted_turn(req_id="r1", reason="cancelled: user", messages=[], partial="")
    titles = [c.args[0] for c in pl.call_args_list]
    assert "Turn Aborted" in titles
    assert any("cancelled: user" in str(c.args[1]) for c in pl.call_args_list)


def test_never_raises_when_the_recorder_fails():
    a = _agent()
    a._record_turn_trajectory = MagicMock(side_effect=RuntimeError("disk"))
    with patch("ghost_agent.core.agent.pretty_log", side_effect=RuntimeError("log")):
        a._record_aborted_turn(req_id="r", reason="x", messages=[], partial="")   # no exception


# --- the sites ---------------------------------------------------------------

def _handle_chat():
    for n in ast.walk(ast.parse(inspect.getsource(ag))):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "handle_chat":
            return n
    raise AssertionError("handle_chat not found")


def _handlers(fn):
    return [h for n in ast.walk(fn) if isinstance(n, ast.Try) for h in n.handlers]


def _records(node):
    return [c for c in ast.walk(node) if isinstance(c, ast.Call)
            and getattr(c.func, "attr", "") == "_record_aborted_turn"]


def test_cancelled_error_is_recorded_and_re_raised():
    fn = _handle_chat()
    hs = [h for h in _handlers(fn)
          if h.type is not None and ast.unparse(h.type) == "asyncio.CancelledError"
          and _records(h)]
    assert len(hs) == 1, "exactly one CancelledError handler records the aborted turn"
    h = hs[0]
    assert any(isinstance(s, ast.Raise) and s.exc is None for s in h.body), "must re-raise bare"
    kw = {k.arg: k.value for k in _records(h)[0].keywords}
    assert isinstance(kw["reason"], ast.Constant) and "cancelled" in kw["reason"].value


def test_stop_button_cancel_is_recorded_before_it_returns():
    fn = _handle_chat()
    hs = [h for h in _handlers(fn)
          if h.type is not None and ast.unparse(h.type) == "TurnCancelled" and _records(h)]
    assert len(hs) == 1
    body = hs[0].body
    rec_i = next(i for i, s in enumerate(body) if _records(s))
    ret_i = next(i for i, s in enumerate(body) if isinstance(s, ast.Return))
    assert rec_i < ret_i


def test_shutdown_stamps_every_running_turn():
    from ghost_agent import main as mn
    tree = ast.parse(inspect.getsource(mn))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan")
    recs = _records(fn)
    assert len(recs) == 1
    kw = {k.arg: k.value for k in recs[0].keywords}
    assert isinstance(kw["reason"], ast.Constant) and kw["reason"].value == "process shutdown"
    # inside the shutdown `finally`, iterating the registry
    finals = [t.finalbody for t in ast.walk(fn) if isinstance(t, ast.Try) and t.finalbody]
    assert any(any(c is recs[0] for c in ast.walk(ast.Module(body=fb, type_ignores=[])))
               for fb in finals)
    assert any(getattr(c.func, "id", "") == "_get_turn_registry" for c in ast.walk(fn)
               if isinstance(c, ast.Call))
