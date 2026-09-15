"""A late verdict that lands BEFORE the turn's own records — 2026-09-13.

The tool-free verdict branch (§4FN reply-shape / §4FY turn-state /
§4EQ arithmetic) has no `await`, so the verdict task spawned by
`_compute_verifier_verdict_gated` completes on the next loop tick and its
done-callback runs at finalize's next yield — BEFORE `_record_turn_trajectory`
writes the corpus row and BEFORE `_record_lesson_outcomes` writes the lesson
stash. Two consumers were dark for every zero-tool turn (47 of 121
human-labelled turns):

  * `_flush_stashed_lesson_outcome` found no stash, wrote nothing, and the
    stash written moments later was never drained — the lesson FAILURE arm;
  * `_backfill_trajectory_outcome` saw a cache miss and skipped the
    calibration re-label (PASSED: skipped the corpus write too).

The fix makes both order-independent with two bounded, context-keyed rings:
an in-flight trajectory's early verdict is parked and REPLAYED by the record;
an early lesson sign is parked and BOOKED by the stash write. Every pin here
drives the real methods in the wrong order and asserts on what the stores
received; each fails on the pre-fix tree.
"""
import asyncio
import types
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.agent import GhostAgent
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Trajectory


# ── the lesson stash ─────────────────────────────────────────────────────────

def _lesson_agent():
    calls = []
    sm = SimpleNamespace(record_surfaced_outcomes=lambda trig, ok: calls.append((tuple(trig), ok)))
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace(skill_memory=sm, memory_bus=None)
    return agent, calls


async def _record(agent, tid, triggers=("L1", "L2")):
    await agent._record_lesson_outcomes(
        surfaced_triggers=list(triggers), execution_failure_count=0,
        verifier_backfill=None, trajectory_id=tid)


async def test_a_verdict_that_lands_before_the_stash_is_booked_when_the_stash_is_written():
    agent, calls = _lesson_agent()
    agent._flush_stashed_lesson_outcome("t", False)     # verdict first…
    await asyncio.sleep(0.05)
    assert calls == []                                   # nothing to book yet
    await _record(agent, "t")                            # …then the turn's stash
    await asyncio.sleep(0.05)
    assert calls == [(("L1", "L2"), False)], calls
    # not left in the stash for a second verdict that never comes
    assert "t" not in (getattr(agent.context, "_surfaced_triggers_by_traj", None) or {})
    # the parked sign is consumed
    assert "t" not in agent.context._pending_lesson_outcome_by_traj


async def test_the_normal_order_is_unchanged():
    agent, calls = _lesson_agent()
    await _record(agent, "t")
    assert agent.context._surfaced_triggers_by_traj["t"] == ["L1", "L2"]
    agent._flush_stashed_lesson_outcome("t", True)
    await asyncio.sleep(0.2)
    assert calls == [(("L1", "L2"), True)]


async def test_an_early_booking_still_takes_a_later_opposite_sign_label():
    """The early booking goes through the same retained ring, so a human 👍
    after a machine REFUTE re-books (the R2 review's compensating write)."""
    agent, calls = _lesson_agent()
    agent._flush_stashed_lesson_outcome("t", False)
    await _record(agent, "t")
    await asyncio.sleep(0.05)
    agent._flush_stashed_lesson_outcome("t", True)       # human overrules
    await asyncio.sleep(0.2)
    assert calls == [(("L1", "L2"), False), (("L1", "L2"), True)], calls


async def test_a_second_flush_of_the_same_sign_does_not_park_again():
    agent, calls = _lesson_agent()
    agent._flush_stashed_lesson_outcome("t", False)
    await _record(agent, "t")
    await asyncio.sleep(0.05)
    agent._flush_stashed_lesson_outcome("t", False)      # same sign, already flushed
    assert "t" not in agent.context._pending_lesson_outcome_by_traj
    await asyncio.sleep(0.05)
    assert calls == [(("L1", "L2"), False)]


def test_the_parked_sign_ring_is_bounded():
    ctx = SimpleNamespace()
    for i in range(agent_mod._PENDING_LESSON_SIGN_MAX + 10):
        agent_mod._park_pending_lesson_sign(ctx, f"t{i}", True)
    assert len(ctx._pending_lesson_outcome_by_traj) == agent_mod._PENDING_LESSON_SIGN_MAX
    assert "t0" not in ctx._pending_lesson_outcome_by_traj
    assert agent_mod._take_pending_lesson_sign(ctx, "t5") is None
    assert agent_mod._take_pending_lesson_sign(ctx, "t200") is True


# ── the trajectory backfill ──────────────────────────────────────────────────

def _fake_backfill_agent(cached=None):
    writes, calib = [], []

    class _Rec:
        enabled = True

        def update_outcome(self, tid, outcome, reason="", source="", **kw):
            writes.append((tid, outcome))
            return True

        def has_human_label(self, tid):
            return False

    ctx = SimpleNamespace(
        trajectory_collector=_Rec(),
        _recent_trajectories_for_correction=OrderedDict(),
        calibration_tracker=SimpleNamespace(
            record_late_verdict_correction=lambda rid, v: calib.append((rid, v))),
        self_model=None,
        skill_memory=SimpleNamespace(is_read_only=False),
    )
    if cached is not None:
        ctx._recent_trajectories_for_correction["fp"] = cached
    fake = SimpleNamespace(
        context=ctx,
        _flush_stashed_lesson_outcome=lambda tid, ok: None,
        _drop_pending_corrections_for=lambda tid: None,
        _record_withheld_verdict=lambda *a, **k: None,
    )
    fake._backfill_trajectory_outcome = (
        lambda *a, **k: GhostAgent._backfill_trajectory_outcome(fake, *a, **k))
    return fake, writes, calib


async def test_an_in_flight_cache_miss_is_deferred_and_replayed_by_the_record():
    tid = "t" * 32
    fake, writes, calib = _fake_backfill_agent()
    agent_mod._mark_trajectory_in_flight(fake.context, tid)      # the turn started
    GhostAgent._backfill_trajectory_outcome(fake, tid, "failed", "verifier says no")
    await asyncio.sleep(0.2)
    assert writes == []                                          # deferred, not lost
    assert fake.context._pending_late_backfill[tid] == ("failed", "verifier says no")
    # the record lands: the row enters the cache, the replay runs
    fake.context._recent_trajectories_for_correction["fp"] = Trajectory(
        id=tid, session_id="s", outcome="unknown", extra={"req_id": "req-9"})
    GhostAgent._replay_deferred_late_backfill(fake, tid)
    await asyncio.sleep(0.3)
    assert writes == [(tid, "failed")]
    assert calib == [("req-9", 0.0)]                             # the re-label the miss skipped
    assert tid not in fake.context._pending_late_backfill
    assert tid not in fake.context._trajectories_in_flight


async def test_a_cache_miss_on_a_trajectory_not_in_flight_keeps_the_immediate_write():
    """Genuine eviction (an old turn): the FAILED must still land at once."""
    tid = "u" * 32
    fake, writes, calib = _fake_backfill_agent()
    GhostAgent._backfill_trajectory_outcome(fake, tid, "failed", "late")
    await asyncio.sleep(0.2)
    assert writes == [(tid, "failed")]
    assert calib == []
    assert not getattr(fake.context, "_pending_late_backfill", None)


async def test_a_cache_hit_is_never_deferred_even_while_in_flight():
    tid = "v" * 32
    cached = Trajectory(id=tid, session_id="s", outcome="unknown", extra={"req_id": "r1"})
    fake, writes, calib = _fake_backfill_agent(cached)
    agent_mod._mark_trajectory_in_flight(fake.context, tid)
    GhostAgent._backfill_trajectory_outcome(fake, tid, "failed", "x")
    await asyncio.sleep(0.2)
    assert writes == [(tid, "failed")] and calib == [("r1", 0.0)]


def test_replay_with_nothing_parked_only_clears_the_in_flight_mark():
    fake, writes, calib = _fake_backfill_agent()
    agent_mod._mark_trajectory_in_flight(fake.context, "w")
    GhostAgent._replay_deferred_late_backfill(fake, "w")
    assert "w" not in fake.context._trajectories_in_flight
    assert writes == []


def test_the_in_flight_ring_is_bounded_and_tolerates_a_mock_context():
    ctx = SimpleNamespace()
    for i in range(agent_mod._TRAJ_IN_FLIGHT_MAX + 5):
        agent_mod._mark_trajectory_in_flight(ctx, f"t{i}")
    assert len(ctx._trajectories_in_flight) == agent_mod._TRAJ_IN_FLIGHT_MAX
    assert not agent_mod._trajectory_is_in_flight(ctx, "t0")
    assert agent_mod._trajectory_is_in_flight(ctx, "t60")
    # a MagicMock context (the recorder tests) must read as "not in flight"
    assert agent_mod._trajectory_is_in_flight(MagicMock(), "t60") is False
    assert agent_mod._take_deferred_late_backfill(MagicMock(), "t60") is None


# ── end to end through the REAL recorder ─────────────────────────────────────

async def test_the_real_record_replays_a_verdict_that_beat_it(tmp_path, monkeypatch):
    """The whole ordering, on a real collector: mark → early FAILED verdict
    (deferred) → `_record_turn_trajectory` → the sidecar carries FAILED and
    the calibration re-label ran with the row's req_id."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    calib = []
    ctx = MagicMock()
    ctx.trajectory_collector = TrajectoryCollector(
        root=tmp_path / "system" / "trajectories", session_id="s")
    ctx.skill_memory = SimpleNamespace(is_read_only=False)
    del ctx.turn_origin_label
    ctx._recent_trajectories_for_correction = OrderedDict()
    ctx.calibration_tracker = SimpleNamespace(
        record_late_verdict_correction=lambda rid, v: calib.append((rid, v)))
    ctx.self_model = None
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    tid = "e" * 32
    agent._mark_trajectory_in_flight(tid)
    agent._backfill_trajectory_outcome(tid, "failed", "the reply violated the word cap")
    await asyncio.sleep(0.1)
    assert ctx.trajectory_collector.latest_correction(tid) is None    # deferred
    agent._record_turn_trajectory(
        messages=[{"role": "user", "content": "in ten words, why?"}],
        final_content="Because " + "words " * 30, req_id="req-e", model="m",
        trajectory_id=tid, user_request="in ten words, why?")
    for _ in range(30):
        await asyncio.sleep(0.02)
        if ctx.trajectory_collector.latest_correction(tid):
            break
    latest = ctx.trajectory_collector.latest_correction(tid)
    assert latest and latest.get("outcome") == "failed", latest
    assert calib == [("req-e", 0.0)]


# ── the loop wiring ──────────────────────────────────────────────────────────

async def test_handle_chat_marks_the_trajectory_in_flight_before_recording_it():
    """The id the loop allocates is marked in flight, and it is the SAME id
    the recorder later receives — the mark is what makes a verdict that
    lands between the two defer instead of missing the cache."""
    from unittest.mock import AsyncMock
    from tests.helpers import FakeBgTasks, make_context
    ctx = make_context()
    agent = GhostAgent(ctx)
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"role": "assistant", "content": "Forty-two.",
                                 "tool_calls": []}}]})
    marked, recorded = [], []
    orig_record = agent._record_turn_trajectory
    agent._mark_trajectory_in_flight = lambda tid: marked.append(tid)

    def _spy_record(**kw):
        recorded.append(kw.get("trajectory_id"))
        return orig_record(**kw)
    agent._record_turn_trajectory = _spy_record
    await agent.handle_chat({"messages": [{"role": "user", "content": "the answer?"}]},
                            FakeBgTasks())
    assert len(marked) == 1 and marked[0]
    assert recorded and recorded[0] == marked[0]


# ── R3 review of the fix: a deferral must not hold the lesson sign hostage ──

async def test_a_deferred_verdict_still_drains_an_existing_stash_and_the_replay_books_nothing_twice():
    """Non-stream order is lessons THEN record: a verdict landing between
    them finds the stash. If the record then raises (or a streamed reply
    is empty and skips it), the replay never runs — the stash must have
    been drained at deferral time, and a replay that DOES run must not
    book the sign a second time."""
    tid = "d" * 32
    fake, writes, calib = _fake_backfill_agent()
    booked = []
    fake.context.skill_memory = SimpleNamespace(
        is_read_only=False,
        record_surfaced_outcomes=lambda trig, ok: booked.append((tuple(trig), ok)))
    fake.context._surfaced_triggers_by_traj = OrderedDict({tid: ["L1"]})
    fake._FLUSHED_TRIG_RETAIN_MAX = 512
    fake._flush_stashed_lesson_outcome = (
        lambda t, ok: GhostAgent._flush_stashed_lesson_outcome(fake, t, ok))
    agent_mod._mark_trajectory_in_flight(fake.context, tid)
    GhostAgent._backfill_trajectory_outcome(fake, tid, "failed", "late")
    await asyncio.sleep(0.2)
    assert writes == []                                  # corpus write deferred…
    assert booked == [(("L1",), False)]                  # …the lesson sign is not
    fake.context._recent_trajectories_for_correction["fp"] = Trajectory(
        id=tid, session_id="s", outcome="unknown", extra={"req_id": "r-d"})
    GhostAgent._replay_deferred_late_backfill(fake, tid)
    await asyncio.sleep(0.3)
    assert writes == [(tid, "failed")]
    assert booked == [(("L1",), False)]                  # not booked twice


# ── the actual race, inside the real loop ────────────────────────────────────

async def test_a_tool_free_refute_inside_handle_chat_reaches_the_corpus_and_the_lesson_arm(
        tmp_path, monkeypatch, capsys):
    """No hand-sequencing: the §4FY word-cap refute is mechanical, the
    verdict task completes on the next loop tick, and its done-callback
    fires between finalize's awaits — before the lesson stash and the
    trajectory record. Pre-fix: `record_surfaced_outcomes` never called,
    no sidecar row with a FAILED outcome for the recorded trajectory."""
    from unittest.mock import AsyncMock
    from tests.helpers import FakeBgTasks, make_context
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    ctx.verifier = SimpleNamespace(llm_client=object())   # attached; never called
    ctx.trajectory_collector = TrajectoryCollector(
        root=tmp_path / "system" / "trajectories", session_id="s")
    ctx.skill_memory.is_read_only = False
    ctx.skill_memory.last_playbook_triggers = ["L1"]
    ctx.skill_memory._playbook_turn_key = ""
    ctx.skill_memory._bus_delivered_turn_key = ""
    ctx.memory_bus = None
    ctx._recent_trajectories_for_correction = OrderedDict()
    ctx.calibration_tracker = None
    ctx.self_model = None
    del ctx.turn_origin_label
    agent = GhostAgent(ctx)
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"role": "assistant", "tool_calls": [],
                                 "content": "The daytime sky looks blue because sunlight "
                                            "is scattered by the air molecules above us."}}]})
    recorded = []
    orig = agent._record_turn_trajectory

    def _spy(**kw):
        row = orig(**kw)
        recorded.append(kw.get("trajectory_id"))
        return row
    agent._record_turn_trajectory = _spy
    await agent.handle_chat(
        {"messages": [{"role": "user", "content":
                       "Answer in at most three words: what colour is the daytime sky?"}]},
        FakeBgTasks())
    tid = recorded[0]
    for _ in range(50):
        await asyncio.sleep(0.02)
        if ctx.trajectory_collector.latest_correction(tid):
            break
    latest = ctx.trajectory_collector.latest_correction(tid)
    assert latest and latest.get("outcome") == "failed", latest
    booked = [c.args for c in ctx.skill_memory.record_surfaced_outcomes.call_args_list]
    assert booked == [(["L1"], False)], booked
    out = capsys.readouterr().out
    # the race really happened in the loop: deferred first, replayed after
    assert "landed before the" in out and "replaying the deferred late failed" in out, out[-1500:]
    assert out.index("landed before the") < out.index("replaying the deferred late failed")
