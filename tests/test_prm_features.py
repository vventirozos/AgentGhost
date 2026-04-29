"""Tests for prm.features step-feature extraction."""

import math

import pytest

from ghost_agent.prm.features import (
    PRM_FEATURE_NAMES,
    ActionFeatures,
    FeatureVector,
    PlanState,
    extract_step_features,
    feature_vector_to_list,
)


def _state(**overrides):
    """Tiny helper so each test reads only what it cares about."""
    base = dict(
        user_request="default request",
        steps_so_far=0,
        failures_so_far=0,
        pending_count=0,
        plan_depth=0,
        tools_used_this_turn=(),
        tools_failed_this_turn=(),
    )
    base.update(overrides)
    return PlanState(**base)


def _action(**overrides):
    base = dict(description="", tool_name="", tool_args={})
    base.update(overrides)
    return ActionFeatures(**base)


# ──────────────────────────────────────────────────────────────────────
# Vector shape / determinism
# ──────────────────────────────────────────────────────────────────────

def test_extracted_vector_length_matches_feature_names():
    fv = extract_step_features(_state(), _action())
    assert len(fv.values) == len(PRM_FEATURE_NAMES)


def test_extracted_vector_keys_match_feature_names():
    fv = extract_step_features(_state(), _action())
    assert set(fv.by_name.keys()) == set(PRM_FEATURE_NAMES)


def test_extracted_vector_order_matches_feature_names():
    """The frozen ordering is what lets the model load weights against
    arbitrary input order. Bit-for-bit reproducibility required."""
    fv = extract_step_features(_state(user_request="hi"), _action(tool_name="execute"))
    expected = tuple(fv.by_name[name] for name in PRM_FEATURE_NAMES)
    assert fv.values == expected


def test_same_inputs_yield_same_vector():
    s = _state(user_request="hello world")
    a = _action(tool_name="file_system", description="read x")
    fv1 = extract_step_features(s, a)
    fv2 = extract_step_features(s, a)
    assert fv1.values == fv2.values


def test_feature_vector_to_list_matches_values():
    fv = extract_step_features(_state(), _action())
    assert feature_vector_to_list(fv) == list(fv.values)


# ──────────────────────────────────────────────────────────────────────
# Request-shape features
# ──────────────────────────────────────────────────────────────────────

def test_request_length_log1p_grows_with_input():
    short_fv = extract_step_features(_state(user_request="hi"), _action())
    long_fv = extract_step_features(_state(user_request="x" * 500), _action())
    assert long_fv.by_name["request_length_log1p"] > short_fv.by_name["request_length_log1p"]


def test_request_has_code_fence_detected():
    fv = extract_step_features(
        _state(user_request="```python\nprint('hi')\n```"), _action()
    )
    assert fv.by_name["request_has_code_fence"] == 1.0


def test_request_has_url_detected():
    fv = extract_step_features(
        _state(user_request="please scrape https://example.com"), _action()
    )
    assert fv.by_name["request_has_url"] == 1.0


def test_request_imperative_count_log1p_for_action_request():
    fv = extract_step_features(
        _state(user_request="write a script and run it"), _action()
    )
    assert fv.by_name["request_imperative_count_log1p"] > 0.0


def test_request_question_words_ratio_positive_for_question():
    fv = extract_step_features(
        _state(user_request="why does this happen?"), _action()
    )
    assert fv.by_name["request_question_words_ratio"] > 0.0
    assert fv.by_name["request_has_question_mark"] == 1.0


# ──────────────────────────────────────────────────────────────────────
# Plan-progress features
# ──────────────────────────────────────────────────────────────────────

def test_plan_steps_so_far_log1p_grows_monotonically():
    a = _action()
    fv0 = extract_step_features(_state(steps_so_far=0), a)
    fv5 = extract_step_features(_state(steps_so_far=5), a)
    fv50 = extract_step_features(_state(steps_so_far=50), a)
    assert fv0.by_name["plan_steps_so_far_log1p"] < fv5.by_name["plan_steps_so_far_log1p"]
    assert fv5.by_name["plan_steps_so_far_log1p"] < fv50.by_name["plan_steps_so_far_log1p"]


def test_plan_has_any_failure_flag():
    no_fail = extract_step_features(_state(failures_so_far=0), _action())
    with_fail = extract_step_features(_state(failures_so_far=2), _action())
    assert no_fail.by_name["plan_has_any_failure"] == 0.0
    assert with_fail.by_name["plan_has_any_failure"] == 1.0


# ──────────────────────────────────────────────────────────────────────
# Action-shape features
# ──────────────────────────────────────────────────────────────────────

def test_action_args_count_log1p():
    fv = extract_step_features(
        _state(),
        _action(tool_args={"a": 1, "b": 2, "c": 3}),
    )
    assert fv.by_name["action_args_count_log1p"] > 0.0


def test_action_has_url_in_args():
    fv = extract_step_features(
        _state(),
        _action(tool_args={"url": "https://example.com"}),
    )
    assert fv.by_name["action_has_url_in_args"] == 1.0


def test_action_has_filepath_in_args():
    fv = extract_step_features(
        _state(),
        _action(tool_args={"path": "src/main.py"}),
    )
    assert fv.by_name["action_has_filepath_in_args"] == 1.0


def test_action_args_total_length_handles_non_string_values():
    """Ints, lists, None must not crash the length accumulator."""
    fv = extract_step_features(
        _state(),
        _action(tool_args={"n": 42, "items": [1, 2, 3], "missing": None}),
    )
    # Just survives + produces a finite float.
    assert math.isfinite(fv.by_name["action_args_total_length_log1p"])


# ──────────────────────────────────────────────────────────────────────
# Tool-bucket features (each tool falls into exactly one bucket)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tool,expected_flag", [
    ("browser", "tool_is_heavyweight"),
    ("execute", "tool_is_heavyweight"),
    ("vision", "tool_is_heavyweight"),
    ("file_system", "tool_is_lightweight"),
    ("scratchpad", "tool_is_lightweight"),
    ("web_search", "tool_is_external"),
    ("knowledge_base", "tool_is_external"),
    ("skill_memory", "tool_is_memory"),
    ("episodic_memory", "tool_is_memory"),
])
def test_tool_bucket_assignment(tool, expected_flag):
    fv = extract_step_features(_state(), _action(tool_name=tool))
    assert fv.by_name[expected_flag] == 1.0
    other_flags = {
        "tool_is_heavyweight",
        "tool_is_lightweight",
        "tool_is_external",
        "tool_is_memory",
        "tool_is_unknown",
    } - {expected_flag}
    for f in other_flags:
        assert fv.by_name[f] == 0.0, f"unexpected flag {f}=1 for tool={tool}"


def test_tool_unknown_bucket_for_unrecognised_tool():
    fv = extract_step_features(_state(), _action(tool_name="some_random_tool_xyz"))
    assert fv.by_name["tool_is_unknown"] == 1.0


# ──────────────────────────────────────────────────────────────────────
# Cross features
# ──────────────────────────────────────────────────────────────────────

def test_tool_already_used_this_turn():
    fv = extract_step_features(
        _state(tools_used_this_turn=("execute", "file_system")),
        _action(tool_name="execute"),
    )
    assert fv.by_name["tool_already_used_this_turn"] == 1.0


def test_tool_failed_this_turn():
    fv = extract_step_features(
        _state(tools_failed_this_turn=("browser",)),
        _action(tool_name="browser"),
    )
    assert fv.by_name["tool_failed_this_turn"] == 1.0


def test_tool_not_yet_used_this_turn():
    fv = extract_step_features(
        _state(tools_used_this_turn=("file_system",)),
        _action(tool_name="execute"),
    )
    assert fv.by_name["tool_already_used_this_turn"] == 0.0


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────

def test_empty_inputs_dont_crash():
    fv = extract_step_features(_state(), _action())
    assert all(math.isfinite(v) for v in fv.values)


def test_huge_request_doesnt_overflow():
    fv = extract_step_features(
        _state(user_request="x" * 1_000_000), _action(),
    )
    # Log scaling keeps things finite.
    assert all(math.isfinite(v) for v in fv.values)


def test_missing_keys_are_caught_loudly(monkeypatch):
    """If somebody forgets to add a feature to a category but updates
    PRM_FEATURE_NAMES, extraction must fail loudly rather than yield
    a vector silently misaligned with the model's weights."""
    import ghost_agent.prm.features as F
    new_names = F.PRM_FEATURE_NAMES + ("nonexistent_feature",)
    monkeypatch.setattr(F, "PRM_FEATURE_NAMES", new_names)
    with pytest.raises(RuntimeError, match="missing keys"):
        F.extract_step_features(_state(), _action())
