"""Tests for ``PRMScorer.uncertainty`` — the boundary-distance proxy
that frontier-aware self-play uses to weight clusters the PRM is least
sure about.

The contract:
    * untrained scorer → 1.0 (max uncertainty / no opinion)
    * trained scorer scoring p≈0.5 → ≈1.0 (most uncertain)
    * trained scorer scoring p≈0 or p≈1 → ≈0.0 (most confident)
    * NaN / inf propagation must not raise — clamps via _clamp_unit
    * Exceptions inside score() must surface as 1.0, never raise
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ghost_agent.prm import (
    ActionFeatures,
    PRMScorer,
    PRMTrainer,
    PlanState,
)
from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory


def _state(**kw):
    base = dict(
        user_request="x",
        steps_so_far=0,
        failures_so_far=0,
        pending_count=0,
        plan_depth=0,
        tools_used_this_turn=(),
        tools_failed_this_turn=(),
    )
    base.update(kw)
    return PlanState(**base)


def _action(**kw):
    base = dict(description="", tool_name="", tool_args={})
    base.update(kw)
    return ActionFeatures(**base)


def _passing(*, n=2, request="ok", tool="scratchpad"):
    return Trajectory(
        user_request=request,
        outcome=Outcome.PASSED.value,
        tool_calls=[ToolCall(name=tool) for _ in range(n)],
        n_steps=n,
    )


def _failing(*, n=2, request="bad", tool="execute"):
    return Trajectory(
        user_request=request,
        outcome=Outcome.FAILED.value,
        tool_calls=[ToolCall(name=tool, error="boom") for _ in range(n)],
        n_steps=n,
    )


def _balanced(n_pass=8, n_fail=8):
    return (
        [_passing(request=f"p{i}") for i in range(n_pass)]
        + [_failing(request=f"f{i}") for i in range(n_fail)]
    )


def _trained_scorer():
    trainer = PRMTrainer(min_samples=5, min_trajectories=2)
    trainer.run(_balanced())
    return PRMScorer(model=trainer.model)


class TestUncertaintyUntrained:
    def test_no_model_returns_maximum_uncertainty(self):
        scorer = PRMScorer()
        u = scorer.uncertainty(_state(), _action())
        assert u == pytest.approx(1.0)

    def test_no_model_is_consistent_across_states(self):
        scorer = PRMScorer()
        u1 = scorer.uncertainty(_state(user_request="a"), _action(tool_name="x"))
        u2 = scorer.uncertainty(_state(user_request="b"), _action(tool_name="y"))
        assert u1 == u2 == 1.0


class TestUncertaintyTrained:
    def test_returns_finite_in_unit_interval(self):
        scorer = _trained_scorer()
        u = scorer.uncertainty(_state(user_request="ok"), _action(tool_name="scratchpad"))
        assert 0.0 <= u <= 1.0

    def test_boundary_score_maps_to_maximum_uncertainty(self):
        """A score of exactly 0.5 should produce uncertainty 1.0."""
        scorer = PRMScorer()  # untrained → returns default 0.5
        # Patch to confirm the math, not the model.
        scorer._default_score = 0.5
        assert scorer.uncertainty(_state(), _action()) == pytest.approx(1.0)

    def test_extreme_score_maps_to_minimum_uncertainty(self):
        """Mock the score path to return 1.0; uncertainty should be 0.0."""
        scorer = PRMScorer()
        # Replace .score with a stub returning 1.0 — model bypassed entirely.
        scorer.score = MagicMock(return_value=1.0)
        assert scorer.uncertainty(_state(), _action()) == pytest.approx(0.0)

    def test_zero_score_maps_to_minimum_uncertainty(self):
        scorer = PRMScorer()
        scorer.score = MagicMock(return_value=0.0)
        assert scorer.uncertainty(_state(), _action()) == pytest.approx(0.0)

    def test_quarter_score_maps_to_half_uncertainty(self):
        """p=0.25 → uncertainty = 1 - 2·0.25 = 0.5"""
        scorer = PRMScorer()
        scorer.score = MagicMock(return_value=0.25)
        assert scorer.uncertainty(_state(), _action()) == pytest.approx(0.5)

    def test_three_quarter_score_maps_to_half_uncertainty(self):
        """Symmetric: p=0.75 → uncertainty = 1 - 2·0.25 = 0.5"""
        scorer = PRMScorer()
        scorer.score = MagicMock(return_value=0.75)
        assert scorer.uncertainty(_state(), _action()) == pytest.approx(0.5)


class TestUncertaintyRobustness:
    def test_score_exception_returns_max_uncertainty(self):
        """If score() raises, uncertainty MUST swallow and bias toward
        exploration — never propagate. Otherwise a single buggy state
        could break the whole self-play seed-picking pass.
        """
        scorer = PRMScorer()
        scorer.score = MagicMock(side_effect=RuntimeError("synthetic boom"))
        assert scorer.uncertainty(_state(), _action()) == pytest.approx(1.0)

    def test_nan_score_clamped_to_neutral_uncertainty(self):
        """_clamp_unit collapses NaN to 0.5 in score(); 0.5 → uncertainty 1.0."""
        scorer = PRMScorer()
        scorer.score = MagicMock(return_value=float("nan"))
        # NOTE: this exercises the case where score() somehow leaks NaN
        # past its own clamp (it shouldn't, but uncertainty must be
        # robust regardless). The 1 - 2·|nan - 0.5| evaluates to NaN,
        # which _clamp_unit then folds to 0.5.
        u = scorer.uncertainty(_state(), _action())
        assert 0.0 <= u <= 1.0
