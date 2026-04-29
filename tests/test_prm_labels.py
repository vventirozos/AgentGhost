"""Tests for prm.labels — Monte Carlo value backprop."""

import pytest

from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
from ghost_agent.prm.labels import (
    StepLabelSpec,
    StepSample,
    class_balance,
    derive_step_labels,
    iter_step_samples,
    label_step_value_binary,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _traj(*, outcome: str, tool_calls=None, request: str = "do something") -> Trajectory:
    return Trajectory(
        user_request=request,
        outcome=outcome,
        tool_calls=list(tool_calls or []),
        n_steps=len(tool_calls or []),
    )


def _call(name: str, **kwargs) -> ToolCall:
    return ToolCall(name=name, **kwargs)


# ──────────────────────────────────────────────────────────────────────
# derive_step_labels — value math
# ──────────────────────────────────────────────────────────────────────

def test_passed_with_default_discount_yields_increasing_values():
    """γ=0.9 with N=4 → values [0.729, 0.81, 0.9, 1.0]. Monotone non-
    decreasing because we credit-assign more to steps closer to the win."""
    t = _traj(
        outcome=Outcome.PASSED.value,
        tool_calls=[_call("a"), _call("b"), _call("c"), _call("d")],
    )
    values = derive_step_labels(t, StepLabelSpec(discount_factor=0.9))
    assert len(values) == 4
    assert values == sorted(values)
    assert values[-1] == pytest.approx(1.0)
    assert values[0] == pytest.approx(0.9 ** 3)


def test_failed_yields_zeros():
    t = _traj(
        outcome=Outcome.FAILED.value,
        tool_calls=[_call("a"), _call("b")],
    )
    values = derive_step_labels(t)
    assert values == [0.0, 0.0]


def test_unknown_yields_empty_list():
    t = _traj(
        outcome=Outcome.UNKNOWN.value,
        tool_calls=[_call("a"), _call("b")],
    )
    assert derive_step_labels(t) == []


def test_empty_tool_calls_yields_empty_list():
    t = _traj(outcome=Outcome.PASSED.value, tool_calls=[])
    assert derive_step_labels(t) == []


def test_min_steps_floor_filters_short_trajectories():
    t = _traj(outcome=Outcome.PASSED.value, tool_calls=[_call("a")])
    # min_steps=2 filters this out
    assert derive_step_labels(t, StepLabelSpec(min_steps=2)) == []
    # min_steps=1 keeps it
    assert derive_step_labels(t, StepLabelSpec(min_steps=1)) == [pytest.approx(1.0)]


def test_include_failed_false_skips_failed_trajectories():
    t = _traj(outcome=Outcome.FAILED.value, tool_calls=[_call("a")])
    assert derive_step_labels(t, StepLabelSpec(include_failed=False)) == []
    assert derive_step_labels(t, StepLabelSpec(include_failed=True)) == [0.0]


def test_discount_factor_zero_only_credits_terminal_step():
    """γ=0 → γ^0=1 for the terminal step, γ^1=0 for any earlier step."""
    t = _traj(
        outcome=Outcome.PASSED.value,
        tool_calls=[_call("a"), _call("b"), _call("c")],
    )
    values = derive_step_labels(t, StepLabelSpec(discount_factor=0.0))
    assert values == [0.0, 0.0, 1.0]


def test_discount_factor_one_credits_every_step_equally():
    t = _traj(
        outcome=Outcome.PASSED.value,
        tool_calls=[_call("a"), _call("b"), _call("c")],
    )
    values = derive_step_labels(t, StepLabelSpec(discount_factor=1.0))
    assert all(v == pytest.approx(1.0) for v in values)


def test_discount_factor_outside_range_clamped():
    """Out-of-range γ would push labels outside [0, 1]. Clamping is
    documented behaviour — must not silently produce garbage."""
    t = _traj(
        outcome=Outcome.PASSED.value,
        tool_calls=[_call("a"), _call("b")],
    )
    too_low = derive_step_labels(t, StepLabelSpec(discount_factor=-0.5))
    too_high = derive_step_labels(t, StepLabelSpec(discount_factor=2.0))
    assert all(0.0 <= v <= 1.0 for v in too_low)
    assert all(0.0 <= v <= 1.0 for v in too_high)


# ──────────────────────────────────────────────────────────────────────
# label_step_value_binary
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,threshold,expected", [
    (1.0, 0.5, 1),
    (0.0, 0.5, 0),
    (0.5, 0.5, 1),    # at-threshold goes positive
    (0.49, 0.5, 0),
    (0.81, 0.5, 1),
    (0.0, 0.0, 1),    # 0 >= 0 → positive when threshold is 0
])
def test_label_step_value_binary(value, threshold, expected):
    assert label_step_value_binary(value, threshold) == expected


# ──────────────────────────────────────────────────────────────────────
# iter_step_samples — full pipeline
# ──────────────────────────────────────────────────────────────────────

def test_iter_step_samples_for_passed_trajectory():
    t = _traj(
        outcome=Outcome.PASSED.value,
        tool_calls=[_call("file_system"), _call("execute")],
        request="parse the log",
    )
    samples = list(iter_step_samples([t]))
    assert len(samples) == 2
    assert samples[0].step_index == 0
    assert samples[1].step_index == 1
    assert samples[0].terminal_outcome == Outcome.PASSED.value
    # Last step gets full credit; first gets discounted.
    assert samples[1].value > samples[0].value


def test_iter_step_samples_skips_unknown_trajectories():
    passed = _traj(outcome=Outcome.PASSED.value, tool_calls=[_call("a")])
    unknown = _traj(outcome=Outcome.UNKNOWN.value, tool_calls=[_call("b")])
    failed = _traj(outcome=Outcome.FAILED.value, tool_calls=[_call("c")])
    samples = list(iter_step_samples([passed, unknown, failed]))
    # 1 passed + 0 unknown + 1 failed = 2 (since include_failed=True default)
    assert len(samples) == 2


def test_iter_step_samples_state_reflects_prefix_only():
    """A step's state must reflect what was known *before* the step
    fired — leaking later steps would let the model post-hoc infer
    the answer instead of learning to predict it."""
    t = _traj(
        outcome=Outcome.PASSED.value,
        tool_calls=[
            _call("file_system"),
            _call("execute"),
            _call("vision"),
        ],
    )
    samples = list(iter_step_samples([t]))
    # Step 0: nothing used yet.
    assert samples[0].state.steps_so_far == 0
    assert samples[0].state.tools_used_this_turn == ()
    # Step 1: only step 0 visible.
    assert samples[1].state.steps_so_far == 1
    assert samples[1].state.tools_used_this_turn == ("file_system",)
    # Step 2: both prior steps visible.
    assert samples[2].state.steps_so_far == 2
    assert samples[2].state.tools_used_this_turn == ("file_system", "execute")


def test_iter_step_samples_failures_so_far_counts_only_errored_prior_calls():
    t = _traj(
        outcome=Outcome.PASSED.value,
        tool_calls=[
            _call("a", error="boom"),
            _call("b"),
            _call("c", error="kaboom"),
            _call("d"),
        ],
    )
    samples = list(iter_step_samples([t]))
    assert samples[0].state.failures_so_far == 0
    assert samples[1].state.failures_so_far == 1
    assert samples[2].state.failures_so_far == 1
    assert samples[3].state.failures_so_far == 2


def test_iter_step_samples_action_extraction():
    t = _traj(
        outcome=Outcome.PASSED.value,
        tool_calls=[
            ToolCall(
                name="execute",
                arguments={"command": "ls -la"},
                result="ok",
            ),
        ],
    )
    samples = list(iter_step_samples([t]))
    assert samples[0].action.tool_name == "execute"
    assert samples[0].action.tool_args == {"command": "ls -la"}


def test_iter_step_samples_preserves_trajectory_id():
    t = Trajectory(
        id="traj-xyz",
        user_request="x",
        outcome=Outcome.PASSED.value,
        tool_calls=[_call("a")],
    )
    samples = list(iter_step_samples([t]))
    assert samples[0].trajectory_id == "traj-xyz"


def test_iter_step_samples_lazy():
    """The iterator must be lazy — a generator, not a list build —
    so callers can stream a multi-GB corpus through fit()."""
    import types
    result = iter_step_samples([])
    assert isinstance(result, types.GeneratorType)


# ──────────────────────────────────────────────────────────────────────
# class_balance summary
# ──────────────────────────────────────────────────────────────────────

def test_class_balance_summary_counts():
    samples = [
        StepSample(state=None, action=None, value=1.0, binary=1),
        StepSample(state=None, action=None, value=0.95, binary=1),
        StepSample(state=None, action=None, value=0.0, binary=0),
    ]
    bal = class_balance(samples)
    assert bal["positive"] == 2
    assert bal["negative"] == 1
    assert bal["total"] == 3
    assert bal["positive_ratio"] == pytest.approx(2 / 3)


def test_class_balance_handles_empty():
    bal = class_balance([])
    assert bal["total"] == 0
    assert bal["positive_ratio"] == 0.0
