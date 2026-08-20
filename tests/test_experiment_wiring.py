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


@pytest.mark.asyncio
async def test_introspect_shows_an_ENABLED_arm_with_no_traffic(tmp_path, monkeypatch):
    """Introspect review C1, end-to-end through the tool: an on-disk
    registry lists an enabled live spec that no trajectory carries — the
    exact state verify_depth sat in for three days while this report,
    the instrument that should have shown n=0, structurally could not.
    """
    import json
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, collector = _agent_with_collector(tmp_path)
    await _run(agent)          # corpus now carries risk_steer stamps

    reg = (tmp_path / "experiments.json")
    reg.write_text(json.dumps({"experiments": [
        {"name": "risk_steer", "arms": ["control", "treatment"],
         "traffic": 1.0, "enabled": True, "scope": "live"},
        {"name": "ghost_arm", "arms": ["control", "treatment"],
         "traffic": 1.0, "enabled": True, "scope": "live"},
    ]}))
    ex.reset_registry_cache()

    from ghost_agent.tools.introspect import tool_introspect
    out = await tool_introspect(action="experiments", context=agent.context)
    assert "risk_steer" in out
    assert "ghost_arm  (n=0)" in out, out[-800:]
    assert "enabled in the registry but NO enrolled turn" in out


@pytest.mark.asyncio
async def test_introspect_DENIES_bench_scoped_stale_live_stamps(tmp_path, monkeypatch):
    """The deny-scope filter had no pin — deleting `_deny_live` survived
    the whole suite. A spec re-scoped to bench must not render its stale
    live stamps in the live view."""
    import json
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, collector = _agent_with_collector(tmp_path)
    await _run(agent)          # stamps risk_steer as a LIVE arm

    reg = (tmp_path / "experiments.json")
    reg.write_text(json.dumps({"experiments": [
        {"name": "risk_steer", "arms": ["control", "treatment"],
         "traffic": 1.0, "enabled": True, "scope": "bench"},
    ]}))
    ex.reset_registry_cache()

    from ghost_agent.tools.introspect import tool_introspect
    out = await tool_introspect(action="experiments", context=agent.context)
    assert "■ risk_steer" not in out, (
        "a bench-scoped spec's stale live stamps rendered in the live view")


# ── Introspect review round 2 (2026-08-17) ─────────────────────────────────

def _write_registry(tmp_path, specs):
    import json
    (tmp_path / "experiments.json").write_text(json.dumps(
        {"experiments": specs}))
    ex.reset_registry_cache()


@pytest.mark.asyncio
async def test_BENCH_section_shows_an_enabled_unstamped_bench_arm(
        tmp_path, monkeypatch):
    """R2 MAJOR-2: the C1 zero-row fix was applied to the live view and
    NOT to the bench section INSIDE THE SAME FUNCTION, with the bench
    names already in hand. tts_bon is one drained budget away from
    verify_depth's exact inert state, invisibly."""
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, _ = _agent_with_collector(tmp_path)
    await _run(agent)
    _write_registry(tmp_path, [
        {"name": "risk_steer", "arms": ["control", "treatment"],
         "traffic": 1.0, "enabled": True, "scope": "live"},
        {"name": "bench_stamped", "arms": ["control", "treatment"],
         "traffic": 1.0, "enabled": True, "scope": "bench"},
        {"name": "bench_ghost", "arms": ["control", "treatment"],
         "traffic": 1.0, "enabled": True, "scope": "bench"},
    ])
    # A bench corpus with one stamped turn, via the admissibility reader.
    from ghost_agent.tools import introspect as I

    def _fake_bench_iter(reason, args):
        from ghost_agent.distill.collector import Trajectory
        return iter([Trajectory(user_request="b", final_response="x",
                                task_kind="bench",
                                extra={"experiments":
                                       {"bench_stamped": "control"}})])
    import ghost_agent.core.admissibility as adm
    monkeypatch.setattr(adm, "iter_bench_trajectories", _fake_bench_iter)

    from ghost_agent.tools.introspect import tool_introspect
    out = await tool_introspect(action="experiments", context=agent.context)
    assert "BENCH-SCOPED EXPERIMENTS" in out
    assert "bench_stamped" in out
    assert "bench_ghost  (n=0)" in out, out[-900:]


@pytest.mark.asyncio
async def test_a_CORRUPT_registry_is_ANNOUNCED_not_silently_defaulted(
        tmp_path, monkeypatch):
    """R2 MAJOR-3: the round-1 caveat fired on an exception load_registry
    can never raise, while the REAL failure — file exists, unparseable —
    silently substituted the code defaults: deny filter off, plus five
    false "enrollment broken" alarms for specs the operator never listed.
    """
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, _ = _agent_with_collector(tmp_path)
    await _run(agent)
    # ⚠ A DISTINGUISHING spec is load-bearing (R3 finding 2). The first
    # version of this test asserted `"(n=0)" not in out` against a corpus
    # where the setup turn had stamped EVERY default spec (traffic=1.0),
    # so `defaults - summary` was empty whether or not degradation nulled
    # `_expected` — the assertion passed with the fix reverted. The sixth
    # cannot-distinguish assertion this project has found; the cure is a
    # default spec the turn could not have stamped.
    ghost = ex.ExperimentSpec(
        name="default_ghost", arms=("control", "treatment"),
        traffic=1.0, enabled=True, description="R3 probe",
        scope=ex.SCOPE_LIVE)
    monkeypatch.setattr(ex, "DEFAULT_SPECS",
                        tuple(ex.DEFAULT_SPECS) + (ghost,))
    (tmp_path / "experiments.json").write_text("{not json at all")
    ex.reset_registry_cache()

    from ghost_agent.tools.introspect import tool_introspect
    out = await tool_introspect(action="experiments", context=agent.context)
    assert "UNREADABLE" in out, out[:400]
    # No false n=0 alarms from the substituted DEFAULT specs — and with
    # `default_ghost` unstamped by construction, this now goes red if
    # degradation stops nulling `_expected`.
    assert "(n=0)" not in out, (
        "default specs the operator never listed rendered as broken arms")


def test_load_registry_carries_its_degradation(tmp_path):
    import json
    p = tmp_path / "experiments.json"
    # Valid file: not degraded.
    p.write_text(json.dumps({"experiments": []}))
    ex.reset_registry_cache()
    assert ex.load_registry(p).degraded is False
    # Corrupt file: degraded.
    p.write_text("{broken")
    ex.reset_registry_cache()
    assert ex.load_registry(p).degraded is True
    # Missing file: defaults, NOT degraded (nothing was substituted).
    ex.reset_registry_cache()
    assert ex.load_registry(tmp_path / "absent.json").degraded is False


def test_the_kill_switch_reframes_the_zero_rows(tmp_path, monkeypatch):
    """R2 MINOR-5: with GHOST_EXPERIMENTS=0 nothing enrolls, so every
    enabled spec would alarm 'enrollment broken' forever — a correct
    observation with a wrong diagnosis attached."""
    monkeypatch.setenv(ex.ENV_KILL, "0")
    # Empty corpus AND a stamped corpus: both shapes must reframe.
    out = ex.render_report({}, expected_names=["ghost_arm"])
    assert "GHOST_EXPERIMENTS=0 is set" in out
    assert "ghost_arm" in out
    assert "enrollment/stamping is broken" not in out

    from ghost_agent.distill.collector import Trajectory
    stats = ex.summarize_trajectories(
        [Trajectory(user_request="q", final_response="a",
                    task_kind="user_request",
                    extra={ex.EXTRA_KEY: {"risk_steer": "control"}})])
    out2 = ex.render_report(stats, expected_names=["risk_steer", "ghost_arm"])
    assert "GHOST_EXPERIMENTS=0 is set" in out2
    assert "enrollment/stamping is broken" not in out2


def test_an_EMPTY_corpus_still_names_the_waiting_arms(monkeypatch):
    """R2 follow-up: the zero-row block was unreachable from render_report's
    own empty-corpus early returns — the identical early-return shape the
    same round fixed in learning_health. Day one of a fresh deploy is
    exactly when 'which arms should be accumulating' matters most."""
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    out = ex.render_report({}, expected_names=["verify_depth"])
    assert "verify_depth" in out
    assert "Enabled and waiting for traffic" in out
    # And with recorded-but-unstamped turns (the regression shape):
    out2 = ex.render_report({}, coverage={"user_turns": 7, "stamped": 0},
                            expected_names=["verify_depth"])
    assert "verify_depth" in out2


def test_the_CLI_report_shows_enabled_unstamped_arms(tmp_path, monkeypatch):
    """R2 MAJOR-1: the operator CLI (scripts/experiment_report.py) calls
    itself 'the same report' as introspect and kept enumerating only
    stamped arms after the introspect fix — this file's own R5 comment
    records the identical 'fixed one file over, re-created here' pattern.
    """
    import subprocess
    import sys as _sys
    home = tmp_path / "home"
    (home / "system").mkdir(parents=True)
    _write_registry_at = home / "system" / "experiments.json"
    _write_registry_at.write_text(json.dumps({"experiments": [
        {"name": "alpha_live", "arms": ["control", "treatment"],
         "traffic": 1.0, "enabled": True, "scope": "live"},
        {"name": "ghost_live", "arms": ["control", "treatment"],
         "traffic": 1.0, "enabled": True, "scope": "live"},
    ]}))
    from ghost_agent.distill.collector import Trajectory
    coll = TrajectoryCollector(root=home / "system" / "trajectories",
                               session_id="cli")
    coll.append(Trajectory(user_request="q", final_response="a",
                           task_kind="user_request",
                           extra={ex.EXTRA_KEY: {"alpha_live": "control"}}))
    env = {"GHOST_HOME": str(home), "PATH": "/usr/bin:/bin",
           "GHOST_API_KEY": "test-key"}
    out = subprocess.run(
        [_sys.executable, "scripts/experiment_report.py"],
        capture_output=True, text=True, env=env, timeout=120,
        cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr[-500:]
    assert "alpha_live" in out.stdout
    assert "ghost_live  (n=0)" in out.stdout, out.stdout[-700:]

    # And the machine-readable form: absent key must not read as healthy.
    outj = subprocess.run(
        [_sys.executable, "scripts/experiment_report.py", "--json"],
        capture_output=True, text=True, env=env, timeout=120,
        cwd=str(Path(__file__).resolve().parents[1]))
    data = json.loads(outj.stdout)
    assert data["enabled_unstamped"] == ["ghost_live"]
    assert data["registry_degraded"] is False


def test_the_CLI_json_carries_the_circular_caveat(tmp_path):
    """R4 C1: `experiment=` was threaded through three of the four
    `compare_arms` call sites. The missed one is this file's `--json`
    branch — so a machine consumer read `TREATMENT WORSE` on a metric the
    human report showed as `NO VERDICT`, on a bench arm already past the
    n>=30/arm floor. This file's own comments record the same "fixed one
    file over, re-created here" pattern twice."""
    import subprocess
    import sys as _sys
    home = tmp_path / "home"
    (home / "system").mkdir(parents=True)
    (home / "system" / "experiments.json").write_text(json.dumps({
        "experiments": [
            {"name": "verify_depth", "arms": ["control", "treatment"],
             "traffic": 1.0, "enabled": True, "scope": "live"}]}))
    from ghost_agent.distill.collector import Trajectory
    coll = TrajectoryCollector(root=home / "system" / "trajectories",
                               session_id="cli")
    for i in range(40):
        for arm, bad in (("control", i < 30), ("treatment", i < 4)):
            t = Trajectory(user_request="q", final_response="a",
                           task_kind="user_request",
                           extra={ex.EXTRA_KEY: {"verify_depth": arm}})
            t.outcome = "failed" if bad else "passed"
            coll.append(t)
    env = {"GHOST_HOME": str(home), "PATH": "/usr/bin:/bin",
           "GHOST_API_KEY": "test-key"}
    out = subprocess.run(
        [_sys.executable, "scripts/experiment_report.py", "--json"],
        capture_output=True, text=True, env=env, timeout=120,
        cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr[-400:]
    data = json.loads(out.stdout)
    # BOTH lists (review R5 F3): the fix threaded `experiment=` through two
    # calls in this branch and the first version of this pin checked one —
    # reproducing the very "three of four call sites" shape it exists to
    # prevent.
    for key in ("comparisons", "triggered_comparisons"):
        rows = data["verify_depth"][key]
        if not rows:
            continue
        fr = next((c for c in rows if c["metric"] == "failure_rate"), None)
        assert fr is not None, f"no failure_rate row in {key}"
        assert "CIRCULAR" in fr["confound"], (
            f"--json {key} dropped the per-experiment caveat: {fr}")
        assert fr["verdict"].startswith("NO VERDICT")


def test_learning_health_names_the_arms_when_NOTHING_is_stamped(tmp_path):
    """R2 MINOR-4: the zero-row block sat after the `not all_stats` early
    return — so the WORST state for it (nothing stamped at all) named no
    specs."""
    from ghost_agent.core.learning_health import _experiment_health_lines
    from ghost_agent.distill.collector import Trajectory
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    (tmp_path / "experiments.json").write_text(json.dumps({"experiments": [
        {"name": "ghost_live", "arms": ["control", "treatment"],
         "traffic": 1.0, "enabled": True, "scope": "live"},
    ]}))
    ex.reset_registry_cache()
    coll = TrajectoryCollector(root=tmp_path / "trajectories",
                               session_id="lh")
    for i in range(3):     # user turns exist, no stamps at all
        coll.append(Trajectory(user_request=f"q{i}", final_response="a",
                               task_kind="user_request"))
    out = "\n".join(_experiment_health_lines(memory_dir))
    assert "NO arms stamped" in out
    assert "ghost_live: n=0" in out, out


@pytest.mark.asyncio
async def test_BENCH_zero_rows_render_even_on_an_EMPTY_bench_corpus(
        tmp_path, monkeypatch):
    """R3 finding 1: the bench closure's `return ""` on an empty corpus
    skipped render_report entirely, so the R2 zero-row fix was unreachable
    in its WORST state — bench specs enabled, bench corpus never stamped
    (fresh home, or bench stamping wholly broken: the verify_depth shape).
    The section must render 'Enabled and waiting for traffic', not vanish."""
    monkeypatch.delenv(ex.ENV_KILL, raising=False)
    ex.reset_registry_cache()
    agent, _ = _agent_with_collector(tmp_path)
    await _run(agent)
    _write_registry(tmp_path, [
        {"name": "risk_steer", "arms": ["control", "treatment"],
         "traffic": 1.0, "enabled": True, "scope": "live"},
        {"name": "bench_ghost", "arms": ["control", "treatment"],
         "traffic": 1.0, "enabled": True, "scope": "bench"},
    ])
    import ghost_agent.core.admissibility as adm
    monkeypatch.setattr(adm, "iter_bench_trajectories",
                        lambda reason, args: iter([]))

    from ghost_agent.tools.introspect import tool_introspect
    out = await tool_introspect(action="experiments", context=agent.context)
    assert "BENCH-SCOPED EXPERIMENTS" in out, (
        "the section vanished on an empty bench corpus")
    assert "bench_ghost" in out, out[-600:]
    assert "waiting for traffic" in out


def test_learning_health_zero_rows_respect_the_kill_switch(tmp_path,
                                                           monkeypatch):
    """R3 finding 3: render_report got the GHOST_EXPERIMENTS=0 reframe in
    round 2 and this sibling did not — same round, same defect shape."""
    from ghost_agent.core.learning_health import _experiment_health_lines
    from ghost_agent.distill.collector import Trajectory
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True)
    (tmp_path / "experiments.json").write_text(json.dumps({"experiments": [
        {"name": "ghost_live", "arms": ["control", "treatment"],
         "traffic": 1.0, "enabled": True, "scope": "live"},
    ]}))
    ex.reset_registry_cache()
    coll = TrajectoryCollector(root=tmp_path / "trajectories",
                               session_id="lh")
    coll.append(Trajectory(user_request="q", final_response="a",
                           task_kind="user_request"))
    monkeypatch.setenv(ex.ENV_KILL, "0")
    out = "\n".join(_experiment_health_lines(memory_dir))
    assert "GHOST_EXPERIMENTS=0 is set" in out, out
    assert "enrollment broken" not in out


def test_the_CLI_json_mode_reports_a_MISSING_corpus_as_a_document(tmp_path):
    """R3 finding 5: the missing-root early return starved --json of the
    keys added so that absent != healthy — a pipe consumer got empty
    stdout and a parse error instead of a document naming the unknown."""
    import subprocess
    import sys as _sys
    home = tmp_path / "empty-home"
    (home / "system").mkdir(parents=True)      # no trajectories dir
    env = {"GHOST_HOME": str(home), "PATH": "/usr/bin:/bin",
           "GHOST_API_KEY": "test-key"}
    out = subprocess.run(
        [_sys.executable, "scripts/experiment_report.py", "--json"],
        capture_output=True, text=True, env=env, timeout=120,
        cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0
    data = json.loads(out.stdout)          # must PARSE
    assert "error" in data
    assert data["registry_degraded"] is None, (
        "unknown must be null, not a healthy-looking default")
