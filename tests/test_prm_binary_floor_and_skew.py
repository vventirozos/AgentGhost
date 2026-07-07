"""PRM training-viability gate must match the label mode, and train↔serve
feature skew must be surfaced.

Two defects (PROJECT_JOURNAL §4B, prm):

1. **Binary-floor gates a continuous fit.** ``use_continuous_labels`` defaults
   to True (the model fits the discount-weighted soft values), but the
   viability gate bailed on the BINARY class balance. A mostly-failing corpus
   with a few passing steps has real continuous variance yet a binary positive
   fraction under the 5% floor → the old gate wrongly bailed "class imbalance"
   on a perfectly trainable set. The gate is now mode-aware: continuous mode
   floors on label VARIANCE, binary mode keeps the class-balance floor.

2. **Train↔serve feature skew is invisible.** Several turn-progress features
   vary across training samples (drawn mid-turn) but are always 0 at the live
   scoring site (turn start). The trainer now flags them so train accuracy is
   not read as deployed discrimination.
"""
from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
from ghost_agent.prm.features import PRM_FEATURE_NAMES
from ghost_agent.prm.trainer import (
    PRMTrainer,
    SERVE_TURN_START_INERT_FEATURES,
)


def _passing(request: str, n_steps: int = 3) -> Trajectory:
    return Trajectory(
        user_request=request,
        outcome=Outcome.PASSED.value,
        tool_calls=[ToolCall(name="scratchpad", arguments={"action": "store"})
                    for _ in range(n_steps)],
        n_steps=n_steps,
    )


def _failing(request: str, n_steps: int = 3) -> Trajectory:
    return Trajectory(
        user_request=request,
        outcome=Outcome.FAILED.value,
        tool_calls=[ToolCall(name="execute", arguments={"command": "x"}, error="boom")
                    for _ in range(n_steps)],
        n_steps=n_steps,
    )


def _mostly_failing_corpus():
    # 1 passing (3 binary-positive step samples) vs 30 failing (90 negatives)
    # → positive fraction ≈ 3.2%, under the 5% binary floor, but the continuous
    #   values span 0.0‥1.0 so a soft-target fit has a clean gradient.
    return [_passing("good 0")] + [_failing(f"bad {i}") for i in range(30)]


# ------------------------------------------------------------- the core fix


def test_continuous_mode_trains_despite_binary_single_class_floor():
    trainer = PRMTrainer(min_samples=5, min_trajectories=2)  # continuous default
    report = trainer.run(_mostly_failing_corpus())
    assert report.fit_succeeded is True, report.bail_reason
    assert "class imbalance" not in report.bail_reason
    assert trainer.model is not None


def test_binary_mode_still_bails_on_class_imbalance():
    trainer = PRMTrainer(min_samples=5, min_trajectories=2,
                         use_continuous_labels=False)
    report = trainer.run(_mostly_failing_corpus())
    assert report.fit_succeeded is False
    assert "class imbalance" in report.bail_reason


def test_continuous_mode_bails_when_no_success_examples():
    # All-failing corpus → no success-side samples → a value model can't learn
    # to discriminate. Continuous mode still bails (single-regime), reported as
    # a class imbalance — the more informative diagnosis than "near-constant".
    trainer = PRMTrainer(min_samples=5, min_trajectories=2)
    report = trainer.run([_failing(f"bad {i}") for i in range(10)])
    assert report.fit_succeeded is False
    assert "imbalance" in report.bail_reason


def test_continuous_mode_bails_when_no_failure_examples():
    # All-passing corpus → no failure-side samples → bail even though the
    # within-trajectory discount curve gives the soft values some variance.
    trainer = PRMTrainer(min_samples=5, min_trajectories=2)
    report = trainer.run([_passing(f"good {i}") for i in range(10)])
    assert report.fit_succeeded is False
    assert "imbalance" in report.bail_reason


# --------------------------------------------------------- serve/train skew


def test_serve_inert_features_are_all_real_feature_names():
    assert set(SERVE_TURN_START_INERT_FEATURES).issubset(set(PRM_FEATURE_NAMES))


def test_feature_skew_warning_surfaced_and_in_summary():
    # A balanced corpus fits; the failing trajectories make steps_so_far /
    # failures_so_far vary across steps → serve-inert features carry variance.
    corpus = ([_passing(f"good {i}") for i in range(8)]
              + [_failing(f"bad {i}") for i in range(8)])
    trainer = PRMTrainer(min_samples=5, min_trajectories=2)
    report = trainer.run(corpus)
    assert report.fit_succeeded is True, report.bail_reason
    assert report.feature_skew_warning
    assert "turn-start" in report.feature_skew_warning
    assert "⚠" in report.summary()
