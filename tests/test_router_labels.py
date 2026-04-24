"""Tests for router.labels."""

from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome
from ghost_agent.router.labels import (
    derive_label, label_trajectories, LabelSpec, class_balance,
)


def _traj(**kwargs) -> Trajectory:
    return Trajectory(**kwargs)


def test_failed_trajectory_is_hard():
    t = _traj(outcome=Outcome.FAILED.value)
    assert derive_label(t) == "hard"


def test_tiny_trajectory_is_easy():
    t = _traj(outcome=Outcome.PASSED.value, n_steps=1)
    assert derive_label(t) == "easy"


def test_heavyweight_tool_flips_to_hard():
    t = _traj(
        outcome=Outcome.PASSED.value,
        n_steps=1,
        tool_calls=[ToolCall(name="browser")],
    )
    assert derive_label(t) == "hard"


def test_many_steps_is_hard():
    t = _traj(outcome=Outcome.PASSED.value, n_steps=6)
    assert derive_label(t) == "hard"


def test_many_tool_calls_is_hard():
    t = _traj(
        outcome=Outcome.PASSED.value,
        n_steps=2,
        tool_calls=[ToolCall(name="file_system") for _ in range(5)],
    )
    assert derive_label(t) == "hard"


def test_middle_ground_is_none():
    t = _traj(
        outcome=Outcome.PASSED.value,
        n_steps=3,
        tool_calls=[ToolCall(name="file_system"), ToolCall(name="recall")],
    )
    # 3 steps > easy_max (2) but < hard_min (4); 2 calls > easy_max (1) but < hard_min (4)
    assert derive_label(t) is None


def test_label_trajectories_filters_ambiguous():
    trajs = [
        _traj(outcome=Outcome.FAILED.value),                           # hard
        _traj(outcome=Outcome.PASSED.value, n_steps=1),                # easy
        _traj(outcome=Outcome.PASSED.value, n_steps=3,                 # ambiguous
              tool_calls=[ToolCall(name="file_system"),
                          ToolCall(name="recall")]),
    ]
    labeled = label_trajectories(trajs)
    assert len(labeled) == 2  # ambiguous dropped
    labels = [label for _t, label in labeled]
    assert sorted(labels) == ["easy", "hard"]


def test_custom_label_spec_shifts_boundary():
    spec = LabelSpec(easy_max_steps=10, easy_max_tool_calls=10,
                     hard_min_steps=100, hard_min_tool_calls=100)
    t = _traj(outcome=Outcome.PASSED.value, n_steps=5,
              tool_calls=[ToolCall(name="file_system") for _ in range(5)])
    # With loose spec, 5 steps + 5 non-heavy calls is easy
    assert derive_label(t, spec) == "easy"


def test_class_balance_reports_ratios():
    balance = class_balance(["easy", "easy", "hard"])
    assert balance == {"easy": 2, "hard": 1, "total": 3, "hard_ratio": 1 / 3}


def test_class_balance_empty():
    balance = class_balance([])
    assert balance["total"] == 0
    assert balance["hard_ratio"] == 0.0
