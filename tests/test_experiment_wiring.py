"""End-to-end wiring for the live-arms framework and the risk governor.

The units are covered in test_experiments.py / test_risk_governor.py. What
breaks in this project is the SEAM — a framework that is built, tested, and
never actually reached on the live path. These tests drive `handle_chat` and
assert the arm is assigned, the steer respects it, and the stamp survives all
the way to the on-disk trajectory (without evicting the hydrated-lessons key
that already lived in `extra`).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core import experiments as ex
from ghost_agent.core import risk
from ghost_agent.distill.collector import TrajectoryCollector
from tests.helpers import make_agent, make_context


# Deliberately NOT a greeting: `_handle_trivial_chat` bypasses the turn
# loop AND the finalize chain, so a "hi" writes no trajectory at all and
# would make every assertion here vacuously pass.
_NON_TRIVIAL = "please investigate the failing build and report what you find"


def _final(text="done"):
    return {"choices": [{"message": {"content": text, "tool_calls": []}}]}


async def _run(agent, user=_NON_TRIVIAL, effects=None, request_id=None):
    agent.context.llm_client.chat_completion = AsyncMock(
        side_effect=list(effects or [_final()]))
    body = {"messages": [{"role": "user", "content": user}], "model": "Qwen-Test"}
    with patch("ghost_agent.core.agent.pretty_log"):
        result = await agent.handle_chat(body, background_tasks=MagicMock(),
                                         request_id=request_id)
    body["_result"] = result
    return body


def _agent_with_collector(tmp_path):
    collector = TrajectoryCollector(root=tmp_path / "trajectories",
                                    session_id="test")
    ctx = make_context(memory_dir=tmp_path / "memory",
                       trajectory_collector=collector)
    ctx.skill_memory.last_playbook_triggers = ["lesson-A"]
    return make_agent(ctx), collector


def _stored(collector):
    return list(collector.iter_trajectories())


@pytest.mark.asyncio
async def test_request_is_enrolled_and_stamped_on_the_trajectory(tmp_path, monkeypatch):
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, collector = _agent_with_collector(tmp_path)

    await _run(agent)

    trajs = _stored(collector)
    assert len(trajs) == 1
    stamped = trajs[0].extra.get(ex.EXTRA_KEY)
    assert stamped and stamped["risk_steer"] in (ex.CONTROL, ex.TREATMENT)


@pytest.mark.asyncio
async def test_stamp_does_not_evict_hydrated_lessons(tmp_path, monkeypatch):
    """`extra` already carried lesson attribution. Overwriting the dict
    instead of merging would silently break counterfactual attribution —
    which is exactly the kind of neighbour interaction that bites here."""
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, collector = _agent_with_collector(tmp_path)

    await _run(agent)

    extra = _stored(collector)[0].extra
    assert extra.get("hydrated_lessons") == ["lesson-A"]
    assert ex.EXTRA_KEY in extra


@pytest.mark.asyncio
async def test_kill_switch_leaves_no_stamp(tmp_path, monkeypatch):
    monkeypatch.setenv(ex.ENV_KILL, "0")
    ex.reset_registry_cache()
    agent, collector = _agent_with_collector(tmp_path)

    await _run(agent)

    extra = _stored(collector)[0].extra
    assert ex.EXTRA_KEY not in extra
    # ...and the rest of `extra` is untouched by the kill.
    assert extra.get("hydrated_lessons") == ["lesson-A"]


@pytest.mark.asyncio
async def test_two_requests_get_independent_assignments(tmp_path, monkeypatch):
    """The stash is per-request state on a process-global context. If it
    leaked, every turn after the first would inherit the first turn's arm."""
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, collector = _agent_with_collector(tmp_path)

    # PINNED req_ids, not uuid4: relying on six coin flips to show both arms
    # failed ~3% of the time (observed 1 in 15 runs). These two ids are
    # deterministically assigned to opposite arms.
    reg = ex.load_registry(ex.registry_path_for_context(agent.context))
    a = next(f"req-{i}" for i in range(100)
             if reg.assign("risk_steer", f"req-{i}") == ex.CONTROL)
    b = next(f"req-{i}" for i in range(100)
             if reg.assign("risk_steer", f"req-{i}") == ex.TREATMENT)
    seen = []
    for rid in (a, b):
        await _run(agent, request_id=rid)
        seen.append(_stored(collector)[-1].extra[ex.EXTRA_KEY]["risk_steer"])
    assert seen == [ex.CONTROL, ex.TREATMENT]


@pytest.mark.asyncio
async def test_steer_fires_in_treatment_only(tmp_path, monkeypatch):
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, _ = _agent_with_collector(tmp_path)

    high = risk.RiskReading(score=0.9, depth_prior=0.6, effort_struggle=0.8,
                            failure_pressure=0.5, step=9, band="high")
    monkeypatch.setattr(risk, "turn_risk", lambda **kw: high)
    monkeypatch.setattr(ex, "arm_for", lambda *a, **k: ex.TREATMENT)

    body = await _run(agent)
    assert any("risk governor" in str(m.get("content", "")).lower()
               for m in body["messages"])


@pytest.mark.asyncio
async def test_steer_suppressed_in_control(tmp_path, monkeypatch):
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, collector = _agent_with_collector(tmp_path)

    high = risk.RiskReading(score=0.9, depth_prior=0.6, effort_struggle=0.8,
                            failure_pressure=0.5, step=9, band="high")
    monkeypatch.setattr(risk, "turn_risk", lambda **kw: high)
    monkeypatch.setattr(ex, "arm_for", lambda *a, **k: ex.CONTROL)

    body = await _run(agent)
    assert not any("risk governor" in str(m.get("content", "")).lower()
                   for m in body["messages"])
    # Compliance is still recorded: the control arm must be distinguishable
    # from "the trigger never fired", or a null result is unreadable.
    assert _stored(collector)[0].extra.get("risk_steer_fired") is False


@pytest.mark.asyncio
async def test_steer_env_kill_beats_the_arm(tmp_path, monkeypatch):
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    monkeypatch.setenv(risk.ENV_STEER_KILL, "0")
    ex.reset_registry_cache()
    agent, _ = _agent_with_collector(tmp_path)

    high = risk.RiskReading(score=0.9, depth_prior=0.6, effort_struggle=0.8,
                            failure_pressure=0.5, step=9, band="high")
    monkeypatch.setattr(risk, "turn_risk", lambda **kw: high)
    monkeypatch.setattr(ex, "arm_for", lambda *a, **k: ex.TREATMENT)

    body = await _run(agent)
    assert not any("risk governor" in str(m.get("content", "")).lower()
                   for m in body["messages"])


@pytest.mark.asyncio
async def test_steer_fires_at_most_once_per_request(tmp_path, monkeypatch):
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, _ = _agent_with_collector(tmp_path)

    high = risk.RiskReading(score=0.9, depth_prior=0.6, effort_struggle=0.8,
                            failure_pressure=0.5, step=9, band="high")
    monkeypatch.setattr(risk, "turn_risk", lambda **kw: high)
    monkeypatch.setattr(ex, "arm_for", lambda *a, **k: ex.TREATMENT)

    # Two loop iterations: an unparseable reply forces a second pass.
    body = await _run(agent, effects=[
        {"choices": [{"message": {"content": "<tool_call>broken",
                                  "tool_calls": []}}]},
        _final(),
    ])
    hits = sum(1 for m in body["messages"]
               if "risk governor" in str(m.get("content", "")).lower())
    assert hits == 1


@pytest.mark.asyncio
async def test_risk_failure_never_breaks_the_turn(tmp_path, monkeypatch):
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, _ = _agent_with_collector(tmp_path)

    def _boom(**kw):
        raise RuntimeError("risk module on fire")

    monkeypatch.setattr(risk, "turn_risk", _boom)
    body = await _run(agent)
    # The turn still delivers its answer — the governor is best-effort.
    assert body["_result"][0] == "done"


@pytest.mark.asyncio
async def test_introspect_experiments_action_reads_the_corpus(tmp_path, monkeypatch):
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, collector = _agent_with_collector(tmp_path)
    await _run(agent)

    from ghost_agent.tools.introspect import tool_introspect
    out = await tool_introspect(action="experiments", context=agent.context)
    assert "EXPERIMENTS" in out and "risk_steer" in out


@pytest.mark.asyncio
async def test_introspect_rejects_unknown_actions_still():
    from ghost_agent.tools.introspect import tool_introspect
    out = await tool_introspect(action="bogus", context=MagicMock())
    assert "SYSTEM ERROR" in out and "experiments" in out


def test_introspect_schema_advertises_the_new_action():
    """A tool the model cannot see is a tool that does not exist — the
    `workdir` incident (2026-07-12) cost ~50s of a live turn to exactly this."""
    from ghost_agent.tools import registry
    spec = next(t for t in registry.TOOL_DEFINITIONS
                if t["function"]["name"] == "introspect")
    enum = spec["function"]["parameters"]["properties"]["action"]["enum"]
    assert "experiments" in enum
    assert "experiments" in spec["function"]["description"]


@pytest.mark.asyncio
async def test_turns_that_cannot_be_recorded_are_not_enrolled(tmp_path, monkeypatch):
    """The self-play/dream solver runs on an isolated context with NO
    trajectory collector. Enrolling it would perturb — and randomize the
    lesson keep/kill verdict of — a population the experiment can never see,
    measure, or exclude."""
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    ctx = make_context(memory_dir=tmp_path / "memory", trajectory_collector=None)
    agent = make_agent(ctx)

    await _run(agent)

    assert ex.assignments_for_request(agent.context, "any") == {}
    stash = getattr(agent.context, "_experiment_arms", None)
    assert stash is not None and stash[1] == {}


@pytest.mark.asyncio
async def test_selfplay_turns_are_not_enrolled(tmp_path, monkeypatch):
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, _ = _agent_with_collector(tmp_path)
    agent.thinking_budget_override = "selfplay"

    await _run(agent)

    stash = getattr(agent.context, "_experiment_arms", None)
    assert stash is not None and stash[1] == {}

# NOTE: the `force_final_response` / `force_stop` guard on the risk steer is
# NOT covered by an automated test. Reaching that state through handle_chat
# needs the no-progress breaker to trip, which needs real tool dispatch, and
# every cheap way to fake it ends up asserting the mock rather than the guard.
# It is a two-condition boolean read three lines from where it is set; an
# honest gap is recorded here rather than a test that would pass either way.


@pytest.mark.asyncio
async def test_control_arm_counterfactual_is_visible_in_the_stream(tmp_path, monkeypatch):
    """The control arm's "would have steered" line is the operator's evidence
    that the randomizer is live and roughly balanced. At debug level it was
    invisible in the pretty stream, so a half-dead experiment looked identical
    to a quiet one."""
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, _ = _agent_with_collector(tmp_path)

    high = risk.RiskReading(score=0.9, depth_prior=0.6, effort_struggle=0.8,
                            failure_pressure=0.5, step=9, band="high")
    monkeypatch.setattr(risk, "turn_risk", lambda **kw: high)
    monkeypatch.setattr(ex, "arm_for", lambda *a, **k: ex.CONTROL)

    seen = []
    with patch("ghost_agent.core.agent.pretty_log",
               side_effect=lambda *a, **k: seen.append(" ".join(str(x) for x in a))):
        body = {"messages": [{"role": "user", "content": _NON_TRIVIAL}],
                "model": "Qwen-Test"}
        agent.context.llm_client.chat_completion = AsyncMock(
            side_effect=[_final()])
        await agent.handle_chat(body, background_tasks=MagicMock())

    assert any("WOULD have" in line and "control arm" in line for line in seen)


def test_learning_health_reports_stamp_coverage(tmp_path):
    """`introspect action='learning'` is the standing "is my learning working"
    instrument; the experiment stamp belongs in it, so a regression surfaces
    without anyone thinking to ask."""
    from ghost_agent.core.learning_health import _experiment_health_lines
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.distill.schema import Trajectory

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    collector = TrajectoryCollector(root=tmp_path / "trajectories",
                                    session_id="s")
    collector.append(Trajectory(user_request="q", final_response="a",
                                task_kind="user_request",
                                extra={ex.EXTRA_KEY: {"risk_steer": "control"}}))
    collector.append(Trajectory(user_request="q2", final_response="a2",
                                task_kind="user_request"))
    out = "\n".join(_experiment_health_lines(memory_dir))
    assert "stamp coverage: 1/2" in out
    assert "risk_steer: control=1" in out


def test_learning_health_warns_when_nothing_is_stamped(tmp_path):
    from ghost_agent.core.learning_health import _experiment_health_lines
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.distill.schema import Trajectory

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    collector = TrajectoryCollector(root=tmp_path / "trajectories",
                                    session_id="s")
    for i in range(3):
        collector.append(Trajectory(user_request=f"q{i}", final_response="a",
                                    task_kind="user_request"))
    out = "\n".join(_experiment_health_lines(memory_dir))
    assert "⚠ NO arms stamped" in out
