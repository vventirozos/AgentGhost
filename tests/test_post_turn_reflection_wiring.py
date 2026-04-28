"""Tests for the user-correction → post-turn reflection wiring inside
GhostAgent.

These exercise ``_maybe_promote_prior_turn_via_user_correction`` and
``_stash_trajectory_for_correction_lookup`` end-to-end without
booting the full agent. The setup mirrors
``test_reflection_biological_tick.py``: ``GhostAgent.__new__`` to skip
``__init__``, hand-rolled context with just the attrs the helpers
read.

What we pin:

  * stash → cache lookup round-trips on the response fingerprint
  * a correction-shaped follow-up promotes the prior trajectory's
    in-memory outcome AND writes a sidecar correction record
  * a non-correction follow-up is a no-op
  * a correction-shaped follow-up with no stashed trajectory is a
    no-op (multi-process / cold-start safety)
  * a Reflector wired on the context gets its ``reflect_one``
    scheduled as an asyncio task with the right id
  * the helper never raises into the user turn
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.distill.collector import (
    TrajectoryCollector,
    CORRECTIONS_FILENAME,
)
from ghost_agent.distill.schema import Trajectory, Outcome


# ----------------------------------------------------- harness


def _bare_agent(ctx):
    a = GhostAgent.__new__(GhostAgent)
    a.context = ctx
    return a


def _ctx(*, collector=None, reflector=None, sink=None):
    ctx = SimpleNamespace()
    ctx.trajectory_collector = collector
    ctx.reflector = reflector
    ctx.reflection_sink = sink
    ctx.last_user_content = ""
    return ctx


def _t(tid: str, *, response: str = "answer", user_request: str = "ask"):
    return Trajectory(
        id=tid,
        user_request=user_request,
        final_response=response,
        outcome=Outcome.UNKNOWN.value,
    )


# ----------------------------------------------------- stash semantics


def test_stash_caches_trajectory_under_response_fingerprint():
    ctx = _ctx()
    a = _bare_agent(ctx)
    traj = _t("t1", response="The capital of France is Paris.")
    a._stash_trajectory_for_correction_lookup(traj)
    cache = ctx._recent_trajectories_for_correction
    assert len(cache) == 1
    fp = a._response_fingerprint("The capital of France is Paris.")
    assert fp in cache
    assert cache[fp] is traj


def test_stash_lru_caps_at_32_entries():
    ctx = _ctx()
    a = _bare_agent(ctx)
    for i in range(40):
        a._stash_trajectory_for_correction_lookup(
            _t(f"t{i}", response=f"response number {i}")
        )
    assert len(ctx._recent_trajectories_for_correction) == 32


def test_stash_skips_empty_response():
    ctx = _ctx()
    a = _bare_agent(ctx)
    a._stash_trajectory_for_correction_lookup(_t("t1", response=""))
    assert getattr(ctx, "_recent_trajectories_for_correction", None) in (None, {})


# ----------------------------------------------------- correction flow


def _build_correction_messages(prev_user, prev_assistant, current_user):
    return [
        {"role": "system", "content": "sysprompt"},
        {"role": "user", "content": prev_user},
        {"role": "assistant", "content": prev_assistant},
        {"role": "user", "content": current_user},
    ]


async def test_correction_promotes_in_memory_and_persists_sidecar(tmp_path):
    collector = TrajectoryCollector(root=tmp_path, session_id="s1")
    ctx = _ctx(collector=collector)
    a = _bare_agent(ctx)

    # Prior turn finished and was stashed.
    prev_assistant = "Here are the go files in your workspace: a.go b.go"
    traj = _t(
        "t1",
        user_request="list python files in workspace",
        response=prev_assistant,
    )
    collector.append(traj)
    a._stash_trajectory_for_correction_lookup(traj)

    # User corrects.
    msgs = _build_correction_messages(
        prev_user="list python files in workspace",
        prev_assistant=prev_assistant,
        current_user=(
            "no, list every PYTHON file in the workspace - python not go"
        ),
    )
    a._maybe_promote_prior_turn_via_user_correction(msgs, msgs[-1]["content"])

    # In-memory mutation
    assert traj.outcome == Outcome.FAILED.value
    assert traj.failure_reason

    # Sidecar record
    sidecar = tmp_path / CORRECTIONS_FILENAME
    assert sidecar.exists()
    rec = json.loads(sidecar.read_text().splitlines()[0])
    assert rec["trajectory_id"] == "t1"
    assert rec["outcome"] == Outcome.FAILED.value
    assert rec["source"] == "user_correction"


async def test_non_correction_followup_is_noop(tmp_path):
    collector = TrajectoryCollector(root=tmp_path, session_id="s1")
    ctx = _ctx(collector=collector)
    a = _bare_agent(ctx)

    prev_assistant = "Paris is the capital of France."
    traj = _t("t1", user_request="capital of france", response=prev_assistant)
    collector.append(traj)
    a._stash_trajectory_for_correction_lookup(traj)

    msgs = _build_correction_messages(
        prev_user="capital of france",
        prev_assistant=prev_assistant,
        current_user="great, what's the capital of germany?",
    )
    a._maybe_promote_prior_turn_via_user_correction(msgs, msgs[-1]["content"])

    assert traj.outcome == Outcome.UNKNOWN.value
    assert not (tmp_path / CORRECTIONS_FILENAME).exists()


async def test_no_stashed_traj_is_noop(tmp_path):
    """Cold start (process restart, message from a session whose
    prior turn we never stashed) — must not crash, must not write
    a sidecar."""
    collector = TrajectoryCollector(root=tmp_path, session_id="s1")
    ctx = _ctx(collector=collector)
    a = _bare_agent(ctx)
    msgs = _build_correction_messages(
        prev_user="list python files",
        prev_assistant="here's a list of go files",
        current_user="no, list python files",
    )
    a._maybe_promote_prior_turn_via_user_correction(msgs, msgs[-1]["content"])
    assert not (tmp_path / CORRECTIONS_FILENAME).exists()


async def test_no_collector_is_noop():
    ctx = _ctx(collector=None)
    a = _bare_agent(ctx)
    a._stash_trajectory_for_correction_lookup(
        _t("t1", response="paris")
    )
    msgs = _build_correction_messages(
        prev_user="capital of france",
        prev_assistant="paris",
        current_user="no, the answer is wrong - capital of france please",
    )
    a._maybe_promote_prior_turn_via_user_correction(msgs, msgs[-1]["content"])


async def test_correction_promotion_drops_cache_entry(tmp_path):
    """One promotion per stashed trajectory: a follow-up message that
    happens to also pattern-match correction shape must not double-
    promote (we'd be writing redundant sidecar entries and
    re-scheduling reflection on the same already-FAILED trajectory)."""
    collector = TrajectoryCollector(root=tmp_path, session_id="s1")
    ctx = _ctx(collector=collector)
    a = _bare_agent(ctx)

    prev_assistant = "Here is wrong answer about pandas dataframe filtering."
    traj = _t("t1", user_request="how to filter a pandas dataframe by date",
              response=prev_assistant)
    collector.append(traj)
    a._stash_trajectory_for_correction_lookup(traj)

    msgs = _build_correction_messages(
        prev_user="how to filter a pandas dataframe by date",
        prev_assistant=prev_assistant,
        current_user="no, how to filter a pandas dataframe by date "
                     "with timezone-aware boundaries please",
    )
    a._maybe_promote_prior_turn_via_user_correction(msgs, msgs[-1]["content"])
    assert traj.outcome == Outcome.FAILED.value
    # Cache entry was dropped — second call is a no-op.
    fp = a._response_fingerprint(prev_assistant)
    assert fp not in (ctx._recent_trajectories_for_correction or {})

    # And calling the helper again with the same shape must NOT add
    # another sidecar record.
    sidecar = tmp_path / CORRECTIONS_FILENAME
    line_count_before = len(sidecar.read_text().splitlines())
    a._maybe_promote_prior_turn_via_user_correction(msgs, msgs[-1]["content"])
    line_count_after = len(sidecar.read_text().splitlines())
    assert line_count_after == line_count_before


# ----------------------------------------------------- reflection scheduling


_VALID_CRITIQUE = (
    "DIAGNOSIS: assistant produced go-file list when asked for python\n"
    "REVISED PLAN:\n"
    "1. re-read the user's exact word 'python'\n"
    "2. filter on .py extension\n"
)


async def test_correction_schedules_reflect_one_via_create_task(tmp_path):
    """When a Reflector is wired, promotion must schedule
    reflect_one as an asyncio task on the running loop."""
    from ghost_agent.reflection.loop import Reflector

    collector = TrajectoryCollector(root=tmp_path, session_id="s1")

    async def critique(_p):
        return _VALID_CRITIQUE

    refl = Reflector(critique_fn=critique)
    sink_received = []

    def sink(t):
        sink_received.append(t)

    ctx = _ctx(collector=collector, reflector=refl, sink=sink)
    a = _bare_agent(ctx)

    prev_assistant = "Here are the go files: a.go b.go"
    traj = _t("t1",
              user_request="list python files in workspace",
              response=prev_assistant)
    collector.append(traj)
    a._stash_trajectory_for_correction_lookup(traj)

    msgs = _build_correction_messages(
        prev_user="list python files in workspace",
        prev_assistant=prev_assistant,
        current_user="no, list every python file in workspace please",
    )
    a._maybe_promote_prior_turn_via_user_correction(msgs, msgs[-1]["content"])

    # A task should have been scheduled.
    pending = ctx._pending_reflection_tasks
    assert pending and len(pending) == 1
    task = next(iter(pending))

    # Drain the task. The sink should receive the reflected trajectory.
    await task
    assert len(sink_received) == 1
    reflected = sink_received[0]
    assert reflected.task_kind == "reflection"
    assert reflected.extra["reflected_from"] == "t1"


async def test_correction_retracts_lessons_from_promoted_trajectory(tmp_path):
    """When a turn is promoted to FAILED via user-correction, any
    lessons the just-finished turn produced (Perfection-Protocol
    writes are the dominant case) must be scrubbed from the
    SkillMemory before retrieval can surface them on the next
    user query.

    Wires a real SkillMemory + collector + classifier path and
    asserts the playbook entry tagged with the prior turn's
    trajectory id is gone after the helper fires."""
    from ghost_agent.memory.skills import SkillMemory

    collector = TrajectoryCollector(root=tmp_path, session_id="s1")
    skill_mem = SkillMemory(tmp_path)
    ctx = _ctx(collector=collector)
    ctx.skill_memory = skill_mem
    ctx.memory_system = None
    a = _bare_agent(ctx)

    # Simulate the prior turn: trajectory recorded + opt-prot lesson
    # written tagged with that trajectory's id.
    prev_assistant = (
        "I incorrectly described django middleware in this response"
    )
    traj = _t(
        "T-poison",
        user_request="explain django middleware briefly",
        response=prev_assistant,
    )
    collector.append(traj)
    a._stash_trajectory_for_correction_lookup(traj)

    skill_mem.learn_lesson(
        task="Optimization Analysis: explain django middleware briefly",
        mistake="Sub-optimal pattern via Perfection Protocol",
        solution="some optimization advice based on the wrong answer",
        source_trajectory_id="T-poison",
        source="perfection_protocol",
    )
    skill_mem.learn_lesson(  # unrelated lesson — must survive
        task="how to fix asyncio race",
        mistake="x",
        solution="y",
        source_trajectory_id="T-other",
    )

    # User corrects.
    msgs = _build_correction_messages(
        prev_user="explain django middleware briefly",
        prev_assistant=prev_assistant,
        current_user=(
            "no, explain django middleware request response cycle briefly"
        ),
    )
    a._maybe_promote_prior_turn_via_user_correction(msgs, msgs[-1]["content"])

    # The opt-prot's poisoned lesson is gone; the unrelated one
    # survives.
    import json as _json
    pb = _json.loads((tmp_path / "skills_playbook.json").read_text())
    triggers = {p.get("trigger") for p in pb}
    assert "Optimization Analysis: explain django middleware briefly" not in triggers
    assert "how to fix asyncio race" in triggers


async def test_correction_skips_retraction_when_no_skill_memory(tmp_path):
    """Helper is robust to missing skill_memory — promotion still
    fires, retraction is silently skipped."""
    collector = TrajectoryCollector(root=tmp_path, session_id="s1")
    ctx = _ctx(collector=collector)
    ctx.skill_memory = None  # explicit
    ctx.memory_system = None
    a = _bare_agent(ctx)

    prev_assistant = "An incorrect answer about postgres indexes"
    traj = _t("t1", user_request="postgres indexes types", response=prev_assistant)
    collector.append(traj)
    a._stash_trajectory_for_correction_lookup(traj)

    msgs = _build_correction_messages(
        prev_user="postgres indexes types",
        prev_assistant=prev_assistant,
        current_user="no, postgres indexes types - btree gin gist explain",
    )
    # Must not raise.
    a._maybe_promote_prior_turn_via_user_correction(msgs, msgs[-1]["content"])
    assert traj.outcome == "failed"


async def test_done_callback_logs_reflection_result(tmp_path, caplog):
    """Reflection runs as a fire-and-forget task — without a
    done-callback, a critique timeout or parse error is silent. Pin
    that the wiring logs SOMETHING when the task completes (success
    or failure), so a tail of the agent log makes the post-turn
    reflection path visible."""
    import logging
    from ghost_agent.reflection.loop import Reflector

    collector = TrajectoryCollector(root=tmp_path, session_id="s1")

    # critique returns garbage — reflect_one will set
    # outcome.error="unparseable reflection response" and skip the
    # sink. The done-callback must still fire and log.
    async def critique(_p):
        return "this is not in the expected format"

    refl = Reflector(critique_fn=critique)
    ctx = _ctx(collector=collector, reflector=refl)
    a = _bare_agent(ctx)

    prev_assistant = "I described django incorrectly"
    traj = _t(
        "t1",
        user_request="explain django middleware briefly",
        response=prev_assistant,
    )
    collector.append(traj)
    a._stash_trajectory_for_correction_lookup(traj)

    msgs = _build_correction_messages(
        prev_user="explain django middleware briefly",
        prev_assistant=prev_assistant,
        current_user=(
            "no, explain django middleware request response cycle briefly"
        ),
    )
    with caplog.at_level(logging.WARNING, logger="GhostReflect"):
        a._maybe_promote_prior_turn_via_user_correction(msgs, msgs[-1]["content"])

        pending = ctx._pending_reflection_tasks
        assert pending and len(pending) == 1
        task = next(iter(pending))
        # Must drain the task so the done-callback fires.
        await task
        # Yield once so the done-callback gets to run before we
        # inspect side effects.
        await asyncio.sleep(0)

    # The done-callback ran without raising. We can't easily assert
    # on pretty_log output (it's a custom logger pipeline), but we
    # CAN assert the task completed without an unhandled exception.
    assert task.done()
    assert task.exception() is None


async def test_correction_without_reflector_still_promotes(tmp_path):
    collector = TrajectoryCollector(root=tmp_path, session_id="s1")
    ctx = _ctx(collector=collector, reflector=None)
    a = _bare_agent(ctx)

    prev_assistant = "An incorrect explanation of postgres index types"
    traj = _t("t1",
              user_request="what types of postgres indexes exist",
              response=prev_assistant)
    collector.append(traj)
    a._stash_trajectory_for_correction_lookup(traj)

    msgs = _build_correction_messages(
        prev_user="what types of postgres indexes exist",
        prev_assistant=prev_assistant,
        current_user="no, what types of postgres indexes exist - "
                     "be specific about btree, gin, gist",
    )
    a._maybe_promote_prior_turn_via_user_correction(msgs, msgs[-1]["content"])

    # Promotion still landed.
    assert traj.outcome == Outcome.FAILED.value
    # No reflection task was scheduled (no reflector).
    assert getattr(ctx, "_pending_reflection_tasks", None) in (None, set())


# ----------------------------------------------------- exception safety


async def test_helper_never_raises_on_malformed_messages():
    """Garbage in messages list must not propagate."""
    collector = MagicMock()
    collector.update_outcome = MagicMock()
    ctx = _ctx(collector=collector)
    a = _bare_agent(ctx)

    # Should swallow everything.
    a._maybe_promote_prior_turn_via_user_correction(None, None)
    a._maybe_promote_prior_turn_via_user_correction([], "")
    a._maybe_promote_prior_turn_via_user_correction(
        [{"role": "??"}, "not a dict", None],
        "no, that's wrong, list python files",
    )
    collector.update_outcome.assert_not_called()
