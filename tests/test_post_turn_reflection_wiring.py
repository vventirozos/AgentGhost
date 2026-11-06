"""Tests for the user-correction promotion wiring inside GhostAgent.
(§4NM: the post-turn `reflect_one` scheduling that followed a promotion was
removed with the reflector; what remains is the promotion, its sidecar
record and the retraction of the turn's own lessons.)

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


def _ctx(*, collector=None):
    ctx = SimpleNamespace()
    ctx.trajectory_collector = collector
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


# ----------------------------------------------------- lesson retraction


async def test_correction_still_promotes(tmp_path):
    """A correction promotes the turn it corrects (§4TD: the lesson retraction
    beside it is removed with the lesson store)."""
    collector = TrajectoryCollector(root=tmp_path, session_id="s1")
    ctx = _ctx(collector=collector)
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


async def test_correction_promotes_with_no_post_turn_task(tmp_path):
    collector = TrajectoryCollector(root=tmp_path, session_id="s1")
    ctx = _ctx(collector=collector)
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

    # Promotion landed.
    assert traj.outcome == Outcome.FAILED.value


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
