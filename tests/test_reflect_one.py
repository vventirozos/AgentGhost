"""Tests for Reflector.reflect_one — the single-trajectory reflection
entrypoint that powers the post-turn reflection scheduler.

What it must do:
  * call critique_fn exactly once for an unseen failed trajectory
  * pass the reflected trajectory to the sink when the critique parses
  * dedup against ``already_reflected`` *before* the LLM call
  * never raise, even when the sink raises
"""

from __future__ import annotations

import pytest

from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome
from ghost_agent.reflection.loop import Reflector


_VALID = (
    "DIAGNOSIS: tried to read from a path that wasn't there\n"
    "REVISED PLAN:\n"
    "1. search the workspace for the target file first\n"
    "2. then read it\n"
)


def _failed_traj(tid="tF"):
    return Trajectory(
        id=tid,
        user_request="parse a logfile",
        failure_reason="access.log not found",
        outcome=Outcome.FAILED.value,
        tool_calls=[
            ToolCall(name="execute", arguments={"code": "open('access.log')"},
                     result="FileNotFoundError"),
        ],
    )


# ------------------------------------------------------ happy path


async def test_reflect_one_calls_critique_and_invokes_sink():
    calls = []
    sink_received = []

    async def critique(prompt):
        calls.append(prompt)
        return _VALID

    def sink(t):
        sink_received.append(t)

    refl = Reflector(critique_fn=critique)
    out = await refl.reflect_one(_failed_traj(), sink=sink)

    assert len(calls) == 1
    assert out.ok
    assert out.diagnosis
    assert out.revised_plan
    assert len(sink_received) == 1
    reflected = sink_received[0]
    assert reflected.task_kind == "reflection"
    assert reflected.extra["reflected_from"] == "tF"


async def test_reflect_one_without_sink_is_still_ok():
    async def critique(_p):
        return _VALID
    refl = Reflector(critique_fn=critique)
    out = await refl.reflect_one(_failed_traj())
    assert out.ok
    assert out.reflected_trajectory is not None


# ------------------------------------------------------ dedup contract


async def test_reflect_one_short_circuits_when_id_already_in_set():
    """A trajectory id that's already been reflected (e.g., by a
    biological-tick run) must not pay the LLM cost a second time."""
    critique_calls = []

    async def critique(_p):
        critique_calls.append(1)
        return _VALID

    refl = Reflector(critique_fn=critique)
    already = {"tF"}
    out = await refl.reflect_one(_failed_traj("tF"), already_reflected=already)

    assert out.error == "already reflected"
    assert out.reflected_trajectory is None
    assert critique_calls == []  # never called


async def test_reflect_one_adds_id_to_set_before_awaiting():
    """Race-safety: the id must enter the dedup set BEFORE the
    critique await so a concurrent biological-tick can't double-fire
    on the same trajectory."""
    seen_during_critique = []
    already = set()

    async def critique(_p):
        # Snapshot the dedup set mid-critique.
        seen_during_critique.append(set(already))
        return _VALID

    refl = Reflector(critique_fn=critique)
    await refl.reflect_one(
        _failed_traj("tX"),
        already_reflected=already,
    )
    assert seen_during_critique  # critique ran
    assert "tX" in seen_during_critique[0], (
        "id was not registered in dedup set before the LLM call"
    )


# ------------------------------------------------------ error swallowing


async def test_reflect_one_swallows_sink_exception():
    async def critique(_p):
        return _VALID

    def bad_sink(_t):
        raise RuntimeError("disk full")

    refl = Reflector(critique_fn=critique)
    # Must not raise.
    out = await refl.reflect_one(_failed_traj(), sink=bad_sink)
    assert out.ok  # the inner reflection still succeeded


async def test_reflect_one_returns_error_on_unparseable_critique():
    async def critique(_p):
        return "this is not in the expected format"
    refl = Reflector(critique_fn=critique)
    out = await refl.reflect_one(_failed_traj())
    assert not out.ok
    assert "unparseable" in out.error.lower()


async def test_reflect_one_returns_error_on_critique_raise():
    async def critique(_p):
        raise RuntimeError("upstream down")
    refl = Reflector(critique_fn=critique)
    out = await refl.reflect_one(_failed_traj())
    assert not out.ok
    assert "RuntimeError" in out.error


# ------------------------------------------------------ recovery scaffolding


from ghost_agent.reflection.loop import _is_recovery_scaffold


def _scaffold_traj(prefix, tid="tS"):
    """A FAILED trajectory whose user prompt is system-generated recovery
    scaffolding (e.g. an AUTO-DIAGNOSTIC retry), exactly the kind that must
    NOT become a saved lesson."""
    return Trajectory(
        id=tid,
        user_request=prefix,
        failure_reason="browser selector '#loading-text' used 4x (>= 4 threshold)",
        outcome=Outcome.FAILED.value,
        tool_calls=[
            ToolCall(name="execute", arguments={"code": "scripts[1]"},
                     result="TypeError: Cannot read properties of undefined"),
        ],
    )


def test_is_recovery_scaffold_detects_markers():
    assert _is_recovery_scaffold(_scaffold_traj(
        "AUTO-DIAGNOSTIC: DIAGNOSTIC ERROR — analyze and fix: [eval]:6 ..."))
    assert _is_recovery_scaffold(_scaffold_traj(
        "SYSTEM ALERT: Your previous turn entered a self-repeating loop ..."))
    # Leading whitespace is tolerated (banners are matched after lstrip).
    assert _is_recovery_scaffold(_scaffold_traj(
        "\n  AUTO-DIAGNOSTIC: DIAGNOSTIC ERROR — foo"))


def test_is_recovery_scaffold_allows_real_tasks_and_learning_signal():
    assert not _is_recovery_scaffold(_failed_traj())  # ordinary user task
    # Self-play / judge-rejection turns ARE legitimate learning signal and
    # must remain reflectable — they are deliberately not in the marker set.
    assert not _is_recovery_scaffold(_scaffold_traj(
        "### SYNTHETIC TRAINING EXERCISE\nbuild a parser"))
    assert not _is_recovery_scaffold(_scaffold_traj(
        "SYSTEM JUDGE REJECTION: your last attempt was wrong"))


def test_is_reflectable_skips_recovery_scaffold_even_when_failed():
    refl = Reflector(critique_fn=lambda _p: _VALID)
    scaffold = _scaffold_traj("AUTO-DIAGNOSTIC: DIAGNOSTIC ERROR — fix it")
    assert scaffold.outcome == Outcome.FAILED.value
    assert refl._is_reflectable(scaffold) is False
    # A normal failed task is still reflectable (regression guard).
    assert refl._is_reflectable(_failed_traj()) is True


async def test_reflect_one_skips_scaffold_without_calling_critique_or_sink():
    calls = []
    sink_received = []

    async def critique(prompt):
        calls.append(prompt)
        return _VALID

    refl = Reflector(critique_fn=critique)
    out = await refl.reflect_one(
        _scaffold_traj("AUTO-DIAGNOSTIC: DIAGNOSTIC ERROR — fix it"),
        sink=sink_received.append,
    )
    assert not out.ok
    assert "recovery-scaffold" in out.error
    assert calls == []          # no LLM critique burned
    assert sink_received == []  # no lesson saved to the playbook
