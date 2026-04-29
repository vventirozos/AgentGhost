"""Tests for biological watchdog phase 2.7 (PRM retrain).

Mirrors the structure of test_reflection_biological_tick.py: a small
helper builds a mocked context shaped to trigger phase 2.7, then we
assert the cooldown-anchor / activity-clock / gating invariants.
"""

import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
from ghost_agent.prm.scorer import PRMScorer


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _balanced_corpus():
    """Returns a list of synthetic trajectories balanced enough for
    PRMTrainer to actually fit. ``PRMTrainer`` defaults: min_trajectories=5,
    min_samples=20, min_class_fraction=0.05."""
    passing = [
        Trajectory(
            user_request=f"good {i}",
            outcome=Outcome.PASSED.value,
            tool_calls=[
                ToolCall(name="scratchpad", arguments={"action": "store"})
                for _ in range(3)
            ],
            n_steps=3,
        )
        for i in range(8)
    ]
    failing = [
        Trajectory(
            user_request=f"bad {i}",
            outcome=Outcome.FAILED.value,
            tool_calls=[
                ToolCall(name="execute", arguments={"command": "x"}, error="boom")
                for _ in range(3)
            ],
            n_steps=3,
        )
        for i in range(8)
    ]
    return passing + failing


def _make_ctx(*, idle_secs: float, prm_scorer=None,
              collector=None, args=None, memory_dir=None,
              checkpoint_path=None):
    """Build a mocked context shaped so phases 1 / 2 / 2.5 / 2.6 short-
    circuit and we observe phase 2.7 in isolation."""
    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    # llm_client.foreground_tasks = 0 so the tick runs.
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    # Phase 1: journal None → short-circuits.
    ctx.journal = None
    # Phase 2: empty memory collection → dream skipped.
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    # Phase 2.5: reflector None → reflection skipped.
    ctx.reflector = None
    # last_activity_time positions us in the requested idle window.
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=idle_secs)

    if args is None:
        args = MagicMock()
        args.model = "test-model"
        # Use the static cooldown by default.
        args.prm_train_cooldown = None
    ctx.args = args

    ctx.frontier_tracker = None
    ctx.trajectory_collector = collector
    ctx.prm_scorer = prm_scorer
    ctx.memory_dir = memory_dir
    ctx._prm_checkpoint_path = checkpoint_path
    ctx.mcts_reasoner = None  # not under test here
    return ctx


async def _tick(ctx, *, suppress_other_phases: bool = True):
    """Run one tick. By default, pre-sets every non-PRM cooldown anchor
    to 'just fired' so phase 2.7 is observed in isolation.

    Phase 2.6 (skills_auto) shares ``iter_trajectories`` with phase 2.7,
    so without suppression a test that asserts on collector calls is
    flaky against the order phases fire."""
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    if suppress_other_phases:
        now = datetime.datetime.now()
        agent._last_dream_at = now
        agent._last_reflection_at = now
        agent._last_skills_auto_at = now
        agent._last_selfplay_at = now
    await agent._biological_tick()
    return agent


# ──────────────────────────────────────────────────────────────────────
# Cooldown constant ordering
# ──────────────────────────────────────────────────────────────────────

def test_prm_cooldown_constant_is_long_enough():
    """PRM retrain is more expensive than skills_auto and benefits less
    from immediate refire — its cooldown should be at least as long as
    skills_auto (currently 7200 s)."""
    assert GhostAgent._PRM_TRAIN_COOLDOWN >= GhostAgent._SKILLS_AUTO_COOLDOWN


# ──────────────────────────────────────────────────────────────────────
# Phase fires when wired
# ──────────────────────────────────────────────────────────────────────

async def test_phase_27_fires_when_collector_and_scorer_present(tmp_path: Path):
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        side_effect=lambda **kw: iter(_balanced_corpus())
    )
    scorer = PRMScorer()
    args = MagicMock()
    args.model = "test"
    args.prm_train_cooldown = None
    ctx = _make_ctx(
        idle_secs=1200,
        prm_scorer=scorer, collector=collector,
        args=args,
        checkpoint_path=tmp_path / "prm.json",
    )
    agent = await _tick(ctx)
    # Anchor was advanced
    assert agent._last_prm_train_at > datetime.datetime.min
    # Scorer hot-swap occurred.
    assert scorer.has_model is True


async def test_phase_27_skipped_when_collector_missing():
    scorer = PRMScorer()
    ctx = _make_ctx(idle_secs=1200, prm_scorer=scorer, collector=None)
    await _tick(ctx)
    assert scorer.has_model is False


async def test_phase_27_skipped_when_scorer_missing():
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter(_balanced_corpus()))
    ctx = _make_ctx(idle_secs=1200, prm_scorer=None, collector=collector)
    # Should run without raising.
    await _tick(ctx)


# ──────────────────────────────────────────────────────────────────────
# Idle-window gating (mirrors reflection / skills_auto)
# ──────────────────────────────────────────────────────────────────────

async def test_phase_27_does_not_fire_below_900s_idle():
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter(_balanced_corpus()))
    scorer = PRMScorer()
    ctx = _make_ctx(idle_secs=300, prm_scorer=scorer, collector=collector)
    await _tick(ctx)
    assert scorer.has_model is False


async def test_phase_27_does_not_fire_above_3600s_idle():
    """Above 3600 s, phase 3 (self-play) takes over. PRM retrain is
    scoped to the 900-3600 s window."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter(_balanced_corpus()))
    scorer = PRMScorer()
    ctx = _make_ctx(idle_secs=4000, prm_scorer=scorer, collector=collector)
    await _tick(ctx)
    assert scorer.has_model is False


# ──────────────────────────────────────────────────────────────────────
# Activity clock + cooldown anchor invariants
# ──────────────────────────────────────────────────────────────────────

async def test_phase_27_never_resets_activity_clock(tmp_path: Path):
    """Same rule as phases 1 / 2 / 2.5 / 2.6: must NOT touch
    ctx.last_activity_time. Otherwise phase 3 (self-play) would never
    fire."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter(_balanced_corpus()))
    scorer = PRMScorer()
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector,
        checkpoint_path=tmp_path / "prm.json",
    )
    activity_before = ctx.last_activity_time
    await _tick(ctx)
    assert ctx.last_activity_time == activity_before


async def test_phase_27_advances_anchor_on_exception(tmp_path: Path):
    """Critical invariant: an exception mid-fit must NOT leave the
    cooldown un-advanced. The collector raising should still result
    in the anchor moving forward — otherwise the failing fit refires
    every 60 s for the rest of the idle window."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        side_effect=RuntimeError("simulated read failure"),
    )
    scorer = PRMScorer()
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector,
        checkpoint_path=tmp_path / "prm.json",
    )
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._last_prm_train_at = datetime.datetime.min
    await agent._biological_tick()
    assert agent._last_prm_train_at > datetime.datetime.min


async def test_phase_27_respects_cooldown(tmp_path: Path):
    """Pre-set anchor to 'just fired' — phase must skip and not call
    iter_trajectories."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter(_balanced_corpus()))
    scorer = PRMScorer()
    args = MagicMock()
    args.prm_train_cooldown = None
    args.model = "test"
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector,
        args=args,
        checkpoint_path=tmp_path / "prm.json",
    )
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    # Suppress all phases except 2.7 so we can assert iter_trajectories
    # wasn't called *by phase 2.7*. (Phase 2.6 / skills_auto also
    # consumes iter_trajectories — without the suppression it fires
    # first and the assertion is flaky.)
    now = datetime.datetime.now()
    agent._last_dream_at = now
    agent._last_reflection_at = now
    agent._last_skills_auto_at = now
    agent._last_selfplay_at = now
    agent._last_prm_train_at = now  # cooldown not yet elapsed
    await agent._biological_tick()
    collector.iter_trajectories.assert_not_called()
    assert scorer.has_model is False


async def test_phase_27_honours_user_supplied_cooldown(tmp_path: Path):
    """``--prm-train-cooldown`` overrides the static class constant."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter(_balanced_corpus()))
    scorer = PRMScorer()
    args = MagicMock()
    args.prm_train_cooldown = 60  # 1 min — much shorter than the 3 h default
    args.model = "test"
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector,
        args=args,
        checkpoint_path=tmp_path / "prm.json",
    )
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    # Suppress other phases so iter_trajectories is observed only via
    # phase 2.7.
    now = datetime.datetime.now()
    agent._last_dream_at = now
    agent._last_reflection_at = now
    agent._last_skills_auto_at = now
    agent._last_selfplay_at = now
    # PRM anchor 5 minutes ago — within the static 3 h cooldown but
    # past the user's 1-min override.
    agent._last_prm_train_at = now - datetime.timedelta(seconds=300)
    await agent._biological_tick()
    # The user's shorter cooldown should let the phase fire.
    collector.iter_trajectories.assert_called_once()
    assert scorer.has_model is True


# ──────────────────────────────────────────────────────────────────────
# Bail behaviour — bad fits must NOT swap a stale model in
# ──────────────────────────────────────────────────────────────────────

async def test_phase_27_does_not_swap_when_trainer_bails(tmp_path: Path):
    """A trainer that bails (e.g., not enough trajectories) must NOT
    publish a model into the live scorer — otherwise the agent's plan
    scoring would be junk for the next 3 hours."""
    # One trajectory → far below trainer's min_trajectories floor.
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        return_value=iter([Trajectory(
            user_request="x",
            outcome=Outcome.PASSED.value,
            tool_calls=[ToolCall(name="a")],
        )])
    )
    scorer = PRMScorer()
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector,
        checkpoint_path=tmp_path / "prm.json",
    )
    await _tick(ctx)
    # Scorer remains un-trained.
    assert scorer.has_model is False


# ──────────────────────────────────────────────────────────────────────
# MCTS auto-plug-in on first successful fit
# ──────────────────────────────────────────────────────────────────────

async def test_phase_27_plugs_scorer_into_mcts_on_first_fit(tmp_path: Path):
    """Lifespan attached the scorer to context but couldn't yet plug
    it into MCTS because no model was loaded. Phase 2.7's first
    successful fit should bridge them."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        side_effect=lambda **kw: iter(_balanced_corpus())
    )
    scorer = PRMScorer()
    mcts = MagicMock()
    mcts.prm_scorer = None  # not yet plugged in
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector,
        checkpoint_path=tmp_path / "prm.json",
    )
    ctx.mcts_reasoner = mcts
    await _tick(ctx)
    assert scorer.has_model is True
    assert mcts.prm_scorer is scorer
