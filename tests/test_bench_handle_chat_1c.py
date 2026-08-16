"""§4BF 1c — the REAL handle_chat bench seams (R3 sweep MAJOR).

The solve-loop suite mocks GhostAgent entirely, so the production seams —
enrollment origin/eligible override, the calibration origin carve-out, the
join-handle stamps, task_kind/user_request overrides — executed only in
production: the R3 sweep proved a one-line reversion of the enrollment
override (or of the calibration carve-out) left all 12,786 tests green
while killing the flagship capability. These tests drive the REAL
``handle_chat`` / ``_record_calibration_safe`` on a bench-labeled context.
"""
import json
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core.calibration import CalibrationTracker
from ghost_agent.distill.collector import TrajectoryCollector


class _FakeBgTasks:
    def add_task(self, *a, **k):
        pass


def _bench_context(tmp_path):
    """A minimal REAL-enough context carrying every bench marker dream's
    bench block sets, plus a real bench collector, calibration tracker
    and experiments registry."""
    (tmp_path / "system" / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "system" / "experiments.json").write_text(json.dumps({
        "salt": "it",
        "experiments": [{"name": "tts_bon", "scope": "bench"}],
    }), encoding="utf-8")

    context = MagicMock(spec=GhostContext)
    context.llm_client = MagicMock()
    context.llm_client.vision_clients = None
    context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "All done — solved.",
                                 "tool_calls": []}}]})
    context.sandbox_dir = str(tmp_path)
    context.args = MagicMock()
    context.args.shell = "bash"
    context.args.max_context = 8000
    context.args.temperature = 0.5
    context.args.smart_memory = 0.0
    context.args.use_planning = False
    context.args.model = "test-model"
    context.args.perfect_it = False
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.memory_system = None
    # Production fidelity (R5 review): dream's isolate carries a
    # READ-ONLY skill memory, and `is_simulation` derives from it — a
    # None here ran every is_simulation-gated finalize block in the
    # OPPOSITE direction from a real bench turn, leaving the whole
    # skill-memory finalize surface dark in this harness.
    context.skill_memory = types.SimpleNamespace(
        is_read_only=True,
        last_playbook_triggers=["real lesson trigger"],
        get_recent_failures=lambda *a, **k: "No failures")
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    context.memory_dir = tmp_path / "system" / "memory"
    context.calibration_tracker = CalibrationTracker(
        tmp_path / "system" / "calibration")

    # The bench markers, exactly as dream.py's bench block sets them.
    bench_root = tmp_path / "system" / "bench" / "trajectories"
    context.trajectory_collector = TrajectoryCollector(
        root=bench_root, session_id="bench-mbpp")
    context.trajectory_task_kind = "bench"
    context.turn_origin_label = "bench"
    context.trajectory_extra_static = {"bench_bank": "mbpp",
                                       "bench_item": "mbpp-9"}
    context.trajectory_user_request_override = \
        "Write a python function that averages a list."
    context._last_bench_calib_req_id = ""
    context._last_bench_traj_id = ""
    return context, bench_root


def _bench_rows(bench_root):
    rows = []
    for day in sorted(p for p in bench_root.iterdir() if p.is_dir()):
        for f in sorted(day.glob("session-*.jsonl")):
            rows += [json.loads(l) for l in f.read_text().splitlines()
                     if l.strip()]
    return rows


@pytest.mark.asyncio
async def test_real_handle_chat_stamps_the_bench_population(
        tmp_path, monkeypatch):
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    from ghost_agent.core import experiments as ex
    ex.reset_registry_cache()
    context, bench_root = _bench_context(tmp_path)
    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"

    body = {"messages": [{"role": "user",
                          "content": "### SYNTHETIC TRAINING EXERCISE\n"
                                     "harness framing block..."}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]):
        await agent.handle_chat(body, _FakeBgTasks(),
                                request_id="bench-itest01")

    rows = _bench_rows(bench_root)
    assert rows, "no bench trajectory was written by the real finalize"
    traj = rows[-1]
    # Population stamps — the seams the R3 mutations killed silently.
    assert traj["task_kind"] == "bench"
    assert traj["extra"]["bench_bank"] == "mbpp"
    assert traj["extra"]["req_id"] == "bench-itest01"
    # user_request override: the CLEAN bank challenge, not the harness
    # framing block the message actually carried.
    assert traj["user_request"] == \
        "Write a python function that averages a list."
    assert "SYNTHETIC TRAINING EXERCISE" not in traj["user_request"]
    # R5: the harness skill memory carries a REAL turn's trigger — a
    # bench record must never stamp it as its own hydration.
    assert "hydrated_lessons" not in traj["extra"]
    # Enrollment: origin="bench" + forced eligibility reached the
    # bench-scoped spec and the arm STAMPED (reverting the eligible
    # override in handle_chat kills exactly this assert).
    assert traj["extra"]["experiments"].get("tts_bon") in (
        "control", "treatment")
    # The oracle join handle points at THIS trajectory.
    assert context._last_bench_traj_id == traj["id"]
    # And the bench collector's oracle write-back joins it.
    assert context.trajectory_collector.update_outcome(
        traj["id"], "passed", "", source="bench_validator")


@pytest.mark.asyncio
async def test_bench_turn_never_writes_the_selfhood_diary(
        tmp_path, monkeypatch):
    # R4 review CRIT: with a collector attached (1b), the finalize's
    # selfhood capture became reachable for bench turns and appended
    # first-person diary Experiences to the PRODUCTION autobiographical
    # store every idle night. The capture site is origin-gated now (and
    # dream's isolate also nulls self_model — suspenders).
    from ghost_agent.selfhood import SelfModel
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    from ghost_agent.core import experiments as ex
    ex.reset_registry_cache()
    context, _root = _bench_context(tmp_path)
    sm = SelfModel.__new__(SelfModel)     # real class → isinstance passes
    sm.enabled = True
    sm.capture_turn = MagicMock()
    context.self_model = sm
    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"

    body = {"messages": [{"role": "user", "content": "solve it"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]):
        await agent.handle_chat(body, _FakeBgTasks(),
                                request_id="bench-itest03")

    assert _bench_rows(_root), "trajectory should still be written"
    sm.capture_turn.assert_not_called()


@pytest.mark.asyncio
async def test_bench_turn_never_consumes_the_activity_digest(
        tmp_path, monkeypatch):
    # R5 review (behavioral pin for the R4 CRIT-B fix — the source-window
    # pin was PROVEN satisfiable by a comment): a bench finalize must not
    # advance the activity-digest watermark nor prepend the digest into
    # the solver reply. On a boot with no push transport this digest is
    # the ONLY delivery channel for notify events, including bench
    # experiment verdicts.
    from ghost_agent.core.autonomous_activity import (
        ActivityLog, SEVERITY_NOTIFY)
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    from ghost_agent.core import experiments as ex
    ex.reset_registry_cache()
    context, _root = _bench_context(tmp_path)
    alog = ActivityLog(tmp_path / "system" / "autonomous_activity.jsonl")
    alog.record("notify_test", "operator must see this",
                severity=SEVERITY_NOTIFY)
    context.activity_log = alog
    wm = tmp_path / "system" / "activity_digest.json"
    wm.write_text('{"offset": 0}', encoding="utf-8")
    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"

    body = {"messages": [{"role": "user", "content": "solve it"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]):
        final, _, _ = await agent.handle_chat(body, _FakeBgTasks(),
                                              request_id="bench-adg1")

    assert json.loads(wm.read_text())["offset"] == 0, \
        "bench finalize consumed the operator's digest watermark"
    assert "Background activity" not in (final or "")


@pytest.mark.asyncio
async def test_bench_turn_never_stashes_metacog_confidence(tmp_path):
    # R5 review MAJOR: the metacog bundle's _last_confidence is the gate
    # arbitrate_tool_calls reads on the NEXT turn — a confident bench
    # solve silently disarmed the arbiter across a real mutating-tool
    # dispatch. Bench is admitted to CALIBRATION only; the bundle stash
    # is real-turn-pure.
    context, _root = _bench_context(tmp_path)
    context.skill_memory = types.SimpleNamespace(is_read_only=True)
    reading = types.SimpleNamespace(
        composite=0.9, entropy_component=0.5, competence_component=0.9,
        uncertainty_pressure=0.0, entropy_observed=False,
        effort_component=0.5, effort_observed=False,
        below_threshold=False)
    _mc = MagicMock()
    _mc.confidence.score.return_value = reading
    context.metacog = _mc
    agent = GhostAgent(context)
    context._calib_pending = None      # force the compute-now path

    with patch("ghost_agent.core.agent.pretty_log"):
        await agent._record_calibration_safe(
            req_id="bench-mc1", tools_run=[], verifier_backfill=None,
            execution_failure_count=0, budget_exhausted=False,
            final_ai_content="done", user_request="avg")

    assert not _mc.record_confidence.called
    # The calibration row itself still landed (bench IS admitted there).
    rows = context.calibration_tracker._load_samples()
    assert rows and rows[-1].origin == "bench"

    # USER control: same shape, real origin → the bundle stash runs.
    context2, _ = _bench_context(tmp_path)
    context2.turn_origin_label = None
    context2.skill_memory = None
    context2.calibration_tracker = CalibrationTracker(
        tmp_path / "system" / "calibration-user")
    _mc2 = MagicMock()
    _mc2.confidence.score.return_value = reading
    context2.metacog = _mc2
    agent2 = GhostAgent(context2)
    context2._calib_pending = None
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent2._record_calibration_safe(
            req_id="real-mc1", tools_run=[], verifier_backfill=None,
            execution_failure_count=0, budget_exhausted=False,
            final_ai_content="done", user_request="avg")
    assert _mc2.record_confidence.called


@pytest.mark.asyncio
async def test_real_calibration_carveout_records_bench_origin(tmp_path):
    context, _root = _bench_context(tmp_path)
    # Bench rides read-only isolation — the carve-out, not the sim gate,
    # must decide. Use a real read-only marker like dream's isolate.
    context.skill_memory = types.SimpleNamespace(is_read_only=True)
    agent = GhostAgent(context)
    reading = types.SimpleNamespace(
        composite=0.7, entropy_component=0.5, competence_component=0.6,
        uncertainty_pressure=0.0, entropy_observed=False,
        effort_component=0.5, effort_observed=False)
    context._calib_pending = ("bench-itest02", reading)

    with patch("ghost_agent.core.agent.pretty_log"):
        await agent._record_calibration_safe(
            req_id="bench-itest02", tools_run=[],
            verifier_backfill=None, execution_failure_count=0,
            budget_exhausted=False, final_ai_content="done",
            user_request="avg list")

    rows = context.calibration_tracker._load_samples()
    assert rows, ("no calibration row — the bench carve-out in the "
                  "simulation gate is not admitting bench turns")
    assert rows[-1].origin == "bench"
    assert rows[-1].req_id == "bench-itest02"
    # Join handle set only AFTER the write, and joinable by the oracle.
    assert context._last_bench_calib_req_id == "bench-itest02"
    assert context.calibration_tracker.record_bench_validator_verdict(
        "bench-itest02", passed=False)

    # CONTROL: plain self-play (read-only, no bench label) stays excluded.
    context2, _ = _bench_context(tmp_path)
    context2.skill_memory = types.SimpleNamespace(is_read_only=True)
    context2.turn_origin_label = None
    context2.calibration_tracker = CalibrationTracker(
        tmp_path / "system" / "calibration2")
    agent2 = GhostAgent(context2)
    context2._calib_pending = ("sim-1", reading)
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent2._record_calibration_safe(
            req_id="sim-1", tools_run=[], verifier_backfill=None,
            execution_failure_count=0, budget_exhausted=False,
            final_ai_content="done", user_request="x")
    assert context2.calibration_tracker._load_samples() == []
