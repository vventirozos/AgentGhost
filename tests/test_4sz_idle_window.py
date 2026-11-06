"""§4SZ: with the failure-replay slot removed, the idle window restarts on its own every hour of quiet, and
no idle phase writes the user clock."""
import ast
import datetime
import inspect
import textwrap
from unittest.mock import patch

import pytest

from ghost_agent.core import agent as A
from tests.test_biological_watchdog import _make_agent


def _idle_lines(agent):
    lines = []
    agent._safe_pretty_log = lambda title, content="", **kw: lines.append((title, content))
    return lines


@pytest.mark.asyncio
async def test_an_hour_of_quiet_restarts_the_window_and_leaves_the_user_clock_alone():
    """§4LF: the 10–60 min jobs are gated by `idle_secs <= 3600`; without a restart a long away stretch
    stops the narratives, the tidy and autoadvance for good. The restart moves only the WINDOW anchor."""
    agent = _make_agent(idle_seconds=2 * 3600 + 60)
    last_request = agent.context.last_activity_time
    lines = _idle_lines(agent)
    before = datetime.datetime.now()
    with patch("ghost_agent.core.agent.random.random", return_value=0.99):
        await agent._biological_tick()
    assert agent.context.last_activity_time == last_request, "an idle phase wrote the user clock"
    assert agent._idle_window_at >= before
    said = [c for t, c in lines if t == "Idle Cycle" and "the idle window restarted" in c]
    assert len(said) == 1
    assert said[0].startswith("no request for 12") and "replay" not in said[0]     # the REAL quiet time: ~120 min


@pytest.mark.asyncio
async def test_the_restart_reopens_the_window_for_the_next_tick():
    """What the restart is for: one tick later the window's age is small again, so a mid-window job's
    `900 < idle <= 3600` gate can open as the minutes pass — the age no longer climbs for ever."""
    agent = _make_agent(idle_seconds=3 * 3600)
    _idle_lines(agent)
    with patch("ghost_agent.core.agent.random.random", return_value=0.99):
        await agent._biological_tick()
    age = (datetime.datetime.now() - max(agent.context.last_activity_time, agent._idle_window_at)).total_seconds()
    assert age < 60
    # a second tick right after does NOT restart it again (the window is young)
    lines = _idle_lines(agent)
    stamp = agent._idle_window_at
    with patch("ghost_agent.core.agent.random.random", return_value=0.99):
        await agent._biological_tick()
    assert agent._idle_window_at == stamp and not [c for _t, c in lines if "the idle window restarted" in c]


@pytest.mark.asyncio
async def test_inside_the_hour_nothing_restarts():
    agent = _make_agent(idle_seconds=50 * 60)
    lines = _idle_lines(agent)
    with patch("ghost_agent.core.agent.random.random", return_value=0.99):
        await agent._biological_tick()
    assert not getattr(agent, "_idle_window_at", None) or agent._idle_window_at == datetime.datetime.min
    assert not [c for _t, c in lines if "the idle window restarted" in c]


@pytest.mark.asyncio
async def test_the_restart_line_says_the_real_quiet_time_not_the_windows_age():
    """After the second restart the window is one hour old and the owner has been away for three."""
    agent = _make_agent(idle_seconds=3 * 3600)
    agent._idle_window_at = datetime.datetime.now() - datetime.timedelta(seconds=3700)
    lines = _idle_lines(agent)
    with patch("ghost_agent.core.agent.random.random", return_value=0.99):
        await agent._biological_tick()
    said = [c for t, c in lines if t == "Idle Cycle" and "the idle window restarted" in c]
    assert len(said) == 1 and said[0].startswith("no request for 180m")


_USER_CLOCKS = ("last_activity_time", "last_owner_activity_time", "last_real_activity_time")


def test_no_idle_phase_writes_the_user_clock():
    """The enumeration (§4LF's rule, now without its one exception): nothing in the idle tick assigns a
    clock that means "the user was here" — the idle window's (`last_activity_time`), the quiet-hours
    hold's (`last_owner_activity_time`, §4MI) or the slot's gap clock (`last_real_activity_time`, §4SA).
    The replay slot's finally was the only writer, and it is gone."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(A.GhostAgent._biological_tick)))
    writes = [ast.unparse(n) for n in ast.walk(tree)
              if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))
              and any(isinstance(t, ast.Attribute) and t.attr in _USER_CLOCKS
                      for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    setters = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
               and getattr(n.func, "id", "") == "setattr" and len(n.args) >= 2
               and isinstance(n.args[1], ast.Constant) and n.args[1].value in _USER_CLOCKS]
    assert writes == [] and setters == [], (writes, setters)


def test_the_user_clocks_are_written_only_where_a_request_arrives():
    """The tick's own body is not the whole idle path — it awaits helpers in other functions and
    modules (review r2: a write in a helper's `finally` was invisible to the test above, which is how
    the replay slot wrote the clock in the first place). So: EVERY write of a user clock anywhere in
    the package, by the function it is in. Only the agent's constructor and the request heartbeat may."""
    from pathlib import Path
    src = Path(A.__file__).resolve().parents[1]
    where = set()
    for p in src.rglob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for n in ast.walk(tree):
            targets = (n.targets if isinstance(n, ast.Assign) else [n.target]
                       if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else [])
            hit = any(isinstance(t, ast.Attribute) and t.attr in _USER_CLOCKS for t in targets) or (
                isinstance(n, ast.Call) and getattr(n.func, "id", "") == "setattr" and len(n.args) >= 2
                and isinstance(n.args[1], ast.Constant) and n.args[1].value in _USER_CLOCKS)
            if hit:
                inside = [f for f in funcs if f.lineno <= n.lineno <= f.end_lineno]
                name = max(inside, key=lambda f: f.lineno).name if inside else "<module>"
                where.add((str(p.relative_to(src)), name))
    assert where == {("core/agent.py", "__init__"), ("core/agent.py", "_heartbeat")}, where
    # …and the heartbeat is called from request handling only, never from the idle tick
    tick = ast.parse(textwrap.dedent(inspect.getsource(A.GhostAgent._biological_tick)))
    assert not [n for n in ast.walk(tick) if isinstance(n, ast.Attribute) and n.attr == "_heartbeat"]


def test_the_replay_slot_is_gone_from_the_agent_and_the_job_list():
    names = {n.attr if isinstance(n, ast.Attribute) else n.id if isinstance(n, ast.Name) else
             n.name if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else ""
             for n in ast.walk(ast.parse(inspect.getsource(A)))}
    gone = {"_self_play_slot_body", "_self_play_slot_body_inner", "start_supervised_self_play_slot",
            "_SELFPLAY_COOLDOWN", "_current_selfplay_cooldown", "_last_selfplay_at", "_NoOwnerSeed",
            "_SlotBusy", "_ProofLegRan", "replay_tool_refusal", "is_replay_request", "judge_reactions"}
    assert names & gone == set(), names & gone
    assert "failure-replay" not in [label for label, _t, _f in A.IDLE_SCHEDULER_JOBS]
    assert "failure-replay" not in A.IDLE_JOB_PURPOSE


def test_the_operator_endpoint_for_the_slot_is_gone():
    from ghost_agent.api import routes
    paths = {getattr(r, "path", "") for r in routes.router.routes}
    assert "/api/operator/self-play-slot" not in paths


def test_nothing_expects_a_heartbeat_from_the_removed_slot():
    """A registered periodic phase with no producer reads DEAD for ever in the liveness report."""
    from ghost_agent.core.autonomous_activity import PHASE_EXPECTATION, EXPECT_PERIODIC
    assert "self_play" not in PHASE_EXPECTATION
    assert [p for p, e in PHASE_EXPECTATION.items() if e == EXPECT_PERIODIC] == ["maintenance"]


def test_the_removed_slots_old_heartbeats_are_history_not_an_unregistered_phase(tmp_path):
    """Review r1: `self_play` rows stay in the 7-day window after the slot is gone; unregistered and not
    retired, the liveness table printed them as "UNREGISTERED phase" for a week."""
    from ghost_agent.core.autonomous_activity import RETIRED_PHASES
    from ghost_agent.core.learning_health import activity_liveness
    from tests.test_activity_liveness import _ledger
    lv = activity_liveness(_ledger(tmp_path, ["self_play", "workspace_tidy"]))
    phases = [r["phase"] for r in lv["rows"]]
    assert "workspace_tidy" in phases and "self_play" not in phases
    assert lv["agent_silent"] is False                          # …and the old rows still show the agent ran
    assert "self_play" in RETIRED_PHASES
