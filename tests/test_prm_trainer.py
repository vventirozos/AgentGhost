"""Tests for prm.trainer.PRMTrainer end-to-end pipeline."""

from pathlib import Path

import pytest

from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
from ghost_agent.prm.scorer import PRMScorer
from ghost_agent.prm.trainer import PRMTrainer, TrainerReport


# ──────────────────────────────────────────────────────────────────────
# Synthetic trajectory factory
# ──────────────────────────────────────────────────────────────────────

def _passing_traj(*, request: str, n_steps: int = 3) -> Trajectory:
    """A 'good' trajectory: lightweight tools, no errors, terminal PASSED."""
    return Trajectory(
        user_request=request,
        outcome=Outcome.PASSED.value,
        tool_calls=[
            ToolCall(name="scratchpad", arguments={"action": "store"})
            for _ in range(n_steps)
        ],
        n_steps=n_steps,
    )


def _failing_traj(*, request: str, n_steps: int = 3) -> Trajectory:
    """A 'bad' trajectory: heavyweight tools, errored steps, terminal FAILED."""
    return Trajectory(
        user_request=request,
        outcome=Outcome.FAILED.value,
        tool_calls=[
            ToolCall(name="execute", arguments={"command": "x"}, error="boom")
            for _ in range(n_steps)
        ],
        n_steps=n_steps,
    )


def _balanced_corpus(*, n_passed: int = 8, n_failed: int = 8):
    return (
        [_passing_traj(request=f"good {i}") for i in range(n_passed)]
        + [_failing_traj(request=f"bad {i}") for i in range(n_failed)]
    )


# ──────────────────────────────────────────────────────────────────────
# Successful end-to-end fit
# ──────────────────────────────────────────────────────────────────────

def test_run_fits_and_saves(tmp_path: Path):
    trainer = PRMTrainer(min_samples=5, min_trajectories=2)
    save_path = tmp_path / "prm.json"
    report = trainer.run(_balanced_corpus(), save_path=save_path)

    assert isinstance(report, TrainerReport)
    assert report.fit_attempted is True
    assert report.fit_succeeded is True
    assert report.bail_reason == ""
    assert report.saved_to == str(save_path)
    assert save_path.exists()
    assert trainer.model is not None
    assert trainer.model.weights_ is not None


def test_run_without_save_path_fits_in_memory(tmp_path: Path):
    """The trainer must support an in-memory mode for tests / quick
    sanity checks."""
    trainer = PRMTrainer(min_samples=5, min_trajectories=2)
    report = trainer.run(_balanced_corpus())
    assert report.fit_succeeded is True
    assert report.saved_to == ""
    assert trainer.model is not None


def test_trained_model_predicts_in_unit_interval(tmp_path: Path):
    trainer = PRMTrainer(min_samples=5, min_trajectories=2)
    trainer.run(_balanced_corpus())
    from ghost_agent.prm.features import (
        ActionFeatures,
        PlanState,
        extract_step_features,
    )
    fv = extract_step_features(
        PlanState(user_request="x"),
        ActionFeatures(tool_name="scratchpad"),
    )
    p = trainer.model.predict_proba(fv)
    assert 0.0 <= p <= 1.0


def test_loaded_checkpoint_predicts_identically(tmp_path: Path):
    """A scorer loaded from the trainer's checkpoint must match the
    in-memory trainer.model bit-for-bit on identical inputs."""
    save_path = tmp_path / "prm.json"
    trainer = PRMTrainer(min_samples=5, min_trajectories=2, random_state=42)
    trainer.run(_balanced_corpus(), save_path=save_path)
    scorer = PRMScorer.load(save_path)

    from ghost_agent.prm.features import (
        ActionFeatures,
        PlanState,
        extract_step_features,
    )
    state = PlanState(user_request="hi")
    action = ActionFeatures(tool_name="scratchpad")
    fv = extract_step_features(state, action)

    p_trainer = trainer.model.predict_proba(fv)
    p_scorer = scorer.score(state, action)
    assert p_trainer == pytest.approx(p_scorer, abs=1e-12)


# ──────────────────────────────────────────────────────────────────────
# Bail behaviours — these are CRITICAL: a trainer that ships a bad
# model is worse than one that ships nothing.
# ──────────────────────────────────────────────────────────────────────

def test_bail_when_too_few_trajectories():
    trainer = PRMTrainer(min_trajectories=10, min_samples=1)
    report = trainer.run([_passing_traj(request="x")])
    assert report.fit_attempted is False
    assert "trajectories" in report.bail_reason
    assert trainer.model is None


def test_bail_when_too_few_samples():
    """One trajectory with 1 step → 1 sample → below default min_samples=20."""
    trainer = PRMTrainer(min_trajectories=1, min_samples=20)
    report = trainer.run([_passing_traj(request="x", n_steps=1)])
    assert report.fit_attempted is False
    assert "step samples" in report.bail_reason


def test_bail_when_class_imbalanced():
    """All-PASSED corpus → no negative samples → trainer must bail
    rather than produce a model that returns 1.0 everywhere."""
    corpus = [_passing_traj(request=f"x{i}") for i in range(20)]
    trainer = PRMTrainer(min_trajectories=2, min_samples=5)
    report = trainer.run(corpus)
    assert report.fit_attempted is False
    assert "imbalance" in report.bail_reason


def test_bail_skips_save_path():
    """A bailed trainer must NOT produce a checkpoint file at the save
    path — otherwise the next watchdog tick would load a non-existent
    or stale model."""
    trainer = PRMTrainer(min_trajectories=10, min_samples=1)
    save_path = Path("/tmp/should-not-exist-prm.json")
    if save_path.exists():
        save_path.unlink()
    report = trainer.run([_passing_traj(request="x")], save_path=save_path)
    assert report.fit_attempted is False
    assert not save_path.exists()


def test_unknown_outcomes_dont_count_toward_balance():
    """UNKNOWN trajectories produce zero samples — they shouldn't be
    able to satisfy the min_trajectories floor."""
    corpus = [
        Trajectory(
            user_request="x",
            outcome=Outcome.UNKNOWN.value,
            tool_calls=[ToolCall(name="a")],
        )
        for _ in range(20)
    ]
    trainer = PRMTrainer(min_trajectories=1, min_samples=5)
    report = trainer.run(corpus)
    # All samples are UNKNOWN → 0 samples → bail on min_samples.
    assert report.fit_attempted is False
    assert report.n_samples_total == 0


# ──────────────────────────────────────────────────────────────────────
# Report fields
# ──────────────────────────────────────────────────────────────────────

def test_report_summary_describes_state():
    trainer = PRMTrainer(min_samples=5, min_trajectories=2)
    report = trainer.run(_balanced_corpus())
    assert "samples" in report.summary()
    assert "trajectories" in report.summary()


def test_report_summary_for_bail():
    trainer = PRMTrainer(min_trajectories=99)
    report = trainer.run([_passing_traj(request="x")])
    assert "no fit attempted" in report.summary()
