"""§4MS (2026-10-09): the idle cycle — what runs while the owner is away.
Each test names the measured defect it fails on.
§4NM: synthetic self-play / bench tests removed."""
from __future__ import annotations

import ast
import asyncio
import textwrap
import inspect
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.distill.schema import Trajectory

ROOT = Path(__file__).resolve().parents[1]


# ── owner safety ──────────────────────────────────────────────────────

def _bare_client():
    from ghost_agent.core.llm import LLMClient
    c = LLMClient.__new__(LLMClient)
    c.foreground_tasks = 0
    c.foreground_requests = 0
    c.main_slot_idle_windows = 0
    c._foreground_lock = asyncio.Lock()
    return c


async def test_a_parked_background_call_is_deferred_not_released_into_a_live_request(monkeypatch):
    """44 of 105 parks ran their call INSIDE a live owner request at 600 s;
    one owner call took 500 s instead of ~30 s."""
    from ghost_agent.core import llm as L
    c = _bare_client()
    c.foreground_requests = 1
    real = asyncio.sleep

    async def instant(_s):
        await real(0)
    monkeypatch.setattr(asyncio, "sleep", instant)
    with pytest.raises(L.BackgroundDeferred, match="selfplay-judge"):
        await c._wait_for_foreground_clear("selfplay-judge")


async def test_a_parked_call_runs_once_the_request_ends(monkeypatch):
    c = _bare_client()
    c.foreground_requests = 1
    real = asyncio.sleep
    n = {"i": 0}

    async def instant(_s):
        n["i"] += 1
        if n["i"] == 5:
            c.foreground_requests = 0
        await real(0)
    monkeypatch.setattr(asyncio, "sleep", instant)
    await c._wait_for_foreground_clear("x")          # returns, no raise


def _agent_tree():
    import ghost_agent.core.agent as A
    return ast.parse(inspect.getsource(A))


def test_self_play_and_counterfactual_run_only_through_the_bounded_runner():
    """No bare await of synthetic_self_play / run_counterfactual_batch in the
    idle tick."""
    tree = _agent_tree()
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_biological_tick")
    bad = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Await) and isinstance(n.value, ast.Call):
            f = n.value.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            if name in ("synthetic_self_play", "run_counterfactual_batch"):
                bad.append(name)
    assert bad == []


# ── memory quality gates ──────────────────────────────────────────────


def test_an_interrupted_turn_is_not_a_failure_and_teaches_nothing():
    """A client disconnect / restart was booked FAILED and taught: reflection
    reinforced a lesson from a turn the owner re-sent 2 s later."""
    from ghost_agent.distill.outcome_heuristics import TURN_INTERRUPTED_MARKER, classify_chat_outcome
    from ghost_agent.memory.population import trajectory_may_teach
    t = Trajectory(user_request="explain x",
                   final_response=f"partial…\n\n{TURN_INTERRUPTED_MARKER} Turn aborted: cancelled: client disconnected.")
    assert classify_chat_outcome(t).outcome == "unknown"
    assert trajectory_may_teach(t) is False and trajectory_may_teach(t, consumer="reflection") is False
    aborted = Trajectory(user_request="x", final_response="[ATTEMPT_ABORTED_TURN] Turn aborted: stop.")
    # rule 0 must hold even where another rule would promote the turn
    cut_in_a_loop = Trajectory(user_request="x", extra={"loop_breaker": "repeat"},
                               final_response=f"{TURN_INTERRUPTED_MARKER} Turn aborted: shutdown.")
    assert classify_chat_outcome(cut_in_a_loop).outcome == "unknown"
    assert classify_chat_outcome(aborted).outcome == "failed"      # an owner Stop stays as it was


def test_a_disconnect_records_the_interruption_marker():
    tree = _agent_tree()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "_record_aborted_turn"
             and any(k.arg == "reason" and "disconnected" in ast.unparse(k.value) for k in n.keywords)]
    assert calls and all(any(k.arg == "marker" and "interruption_marker" in ast.unparse(k.value)
                             for k in c.keywords) for c in calls)


def test_a_cancel_at_the_deadline_is_the_agents_failure(monkeypatch):
    """r1: the interface's 1800 s timeout reads like a disconnect — two of
    seven owner "disconnects" were the agent being too slow."""
    import ghost_agent.utils.logging as G
    from ghost_agent.core.agent import interruption_marker
    monkeypatch.setattr(G, "request_remaining_s", lambda rid: 2.0)
    assert interruption_marker("r1") == "[ATTEMPT_ABORTED_TURN]"
    monkeypatch.setattr(G, "request_remaining_s", lambda rid: 900.0)
    assert interruption_marker("r1") == "[TURN_INTERRUPTED]"
    monkeypatch.setattr(G, "request_remaining_s", lambda rid: None)
    assert interruption_marker("r1") == "[TURN_INTERRUPTED]"


# ── no-op churn ──────────────────────────────────────────────────────

# ── operator decisions ────────────────────────────────────────────────




def test_two_snapshots_in_one_second_both_publish(tmp_path, monkeypatch):
    """§4MS battery: a cleanup re-run inside the same second collided with
    the first snapshot's name and the publish failed (ENOTEMPTY)."""
    import time as _time
    from ghost_agent.memory import snapshot as S
    (tmp_path / "system" / "memory").mkdir(parents=True)
    (tmp_path / "system" / "memory" / "x.json").write_text("{}")
    _fixed = _time.gmtime(0)
    monkeypatch.setattr(S.time, "gmtime", lambda *a: _fixed)
    a = S.take_snapshot(tmp_path, None, "t")
    b = S.take_snapshot(tmp_path, None, "t")
    assert a["ok"] and b["ok"] and a["path"] != b["path"]


