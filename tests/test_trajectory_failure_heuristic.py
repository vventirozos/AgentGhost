"""Tests for distill.outcome_heuristics — the chat-trajectory FAILED
promotion logic.

Context
-------
``Reflector.run`` only iterates trajectories with
``outcome == FAILED.value``. Chat trajectories ship with UNKNOWN (no
validator). Without this heuristic, interactive-session failures
(the 2026-04-26 webOS loop, for example) never produce reflection
lessons even though they're some of the richest failure data the
agent generates.

These tests pin (a) the four firing signals, (b) the conservative
no-op behavior on healthy trajectories, and (c) the contract that
PASSED / FAILED are never overruled.
"""

from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome
from ghost_agent.distill.outcome_heuristics import (
    apply_chat_outcome_heuristics,
    classify_chat_outcome,
)


# ---------------------------------------------------------------- helpers


def _traj(**kw) -> Trajectory:
    """Build a minimal Trajectory for tests. Defaults to UNKNOWN."""
    kw.setdefault("outcome", Outcome.UNKNOWN.value)
    return Trajectory(**kw)


# ---------------------------------------------------------------- happy-path: no promotion


def test_healthy_trajectory_stays_unknown():
    """A normal turn with one successful tool call must NOT be
    promoted. False positives flood the lesson store."""
    t = _traj(
        user_request="show me the readme",
        final_response="Here's the README content...",
        tool_calls=[ToolCall(name="file_system", arguments={"operation": "read", "path": "README.md"}, result="# Project\n...")],
    )
    v = classify_chat_outcome(t)
    assert v.promoted is False
    assert v.outcome == Outcome.UNKNOWN.value


def test_two_distinct_browser_clicks_does_not_fire():
    """Two clicks on different selectors is a normal multi-step
    flow, not a stuck loop."""
    t = _traj(
        tool_calls=[
            ToolCall(name="browser", arguments={"selector": "#a"}, result="ok"),
            ToolCall(name="browser", arguments={"selector": "#b"}, result="ok"),
        ],
        final_response="done",
    )
    assert classify_chat_outcome(t).promoted is False


def test_one_tool_error_does_not_fire():
    """A single error is normal — the agent should retry."""
    t = _traj(
        tool_calls=[
            ToolCall(name="execute", result="Error: command not found"),
            ToolCall(name="execute", result="ok"),
        ],
    )
    assert classify_chat_outcome(t).promoted is False


# ---------------------------------------------------------------- signal 1: abort markers


def test_attempt_aborted_cross_turn_loop_promotes_to_failed():
    t = _traj(
        final_response=(
            "[ATTEMPT_ABORTED_CROSS_TURN_LOOP] The solver opened three "
            "consecutive turns with near-identical reasoning."
        ),
    )
    v = classify_chat_outcome(t)
    assert v.promoted
    assert v.outcome == Outcome.FAILED.value
    assert "ATTEMPT_ABORTED_CROSS_TURN_LOOP" in v.reason


def test_attempt_aborted_thinking_loop_promotes_to_failed():
    t = _traj(
        final_response="prefix [ATTEMPT_ABORTED_THINKING_LOOP] suffix",
    )
    v = classify_chat_outcome(t)
    assert v.promoted
    assert "ATTEMPT_ABORTED_THINKING_LOOP" in v.reason


def test_unrelated_text_with_brackets_does_not_match():
    """Defensive: the regex must require the ATTEMPT_ABORTED_ prefix
    so a free-form reply that happens to mention "[ABORTED]" or
    similar doesn't trigger a false positive."""
    t = _traj(final_response="The mission was [ABORTED] mid-flight.")
    assert classify_chat_outcome(t).promoted is False


# ---------------------------------------------------------------- signal 2: selector thrash


def test_same_browser_selector_four_times_promotes():
    """The webOS incident shape: same selector clicked over and
    over within one turn (an interact sequence with the same
    target)."""
    t = _traj(
        tool_calls=[ToolCall(
            name="browser",
            arguments={
                "operation": "interact",
                "actions": [
                    {"action": "click", "selector": "#start-btn"},
                    {"action": "click", "selector": "#start-btn"},
                    {"action": "click", "selector": "#start-btn"},
                    {"action": "click", "selector": "#start-btn"},
                ],
            },
            result="ok",
        )],
    )
    v = classify_chat_outcome(t)
    assert v.promoted
    assert v.outcome == Outcome.FAILED.value
    assert "#start-btn" in v.reason


def test_three_repeats_below_default_threshold():
    """Threshold default is 4 — three clicks on the same target
    should NOT fire so a legitimate 'click button → screenshot'
    multi-step flow doesn't get marked failed."""
    t = _traj(
        tool_calls=[ToolCall(
            name="browser",
            arguments={"actions": [
                {"action": "click", "selector": "#x"},
                {"action": "click", "selector": "#x"},
                {"action": "click", "selector": "#x"},
            ]},
        )],
    )
    assert classify_chat_outcome(t).promoted is False


def test_repeated_selector_threshold_is_configurable():
    """The repeated_selector_threshold knob is exposed for tests
    and tuning — production callers should use the default."""
    t = _traj(
        tool_calls=[ToolCall(
            name="browser",
            arguments={"actions": [
                {"action": "click", "selector": "#x"},
                {"action": "click", "selector": "#x"},
            ]},
        )],
    )
    v = classify_chat_outcome(t, repeated_selector_threshold=2)
    assert v.promoted


def test_atomic_browser_calls_count_toward_selector_thrash():
    """Atomic browser calls (selector at top level) are counted
    alongside interact sub-actions — same agent behaviour."""
    t = _traj(
        tool_calls=[
            ToolCall(name="browser", arguments={"selector": "#start-btn"}),
            ToolCall(name="browser", arguments={"selector": "#start-btn"}),
            ToolCall(name="browser", arguments={"selector": "#start-btn"}),
            ToolCall(name="browser", arguments={"selector": "#start-btn"}),
        ],
    )
    assert classify_chat_outcome(t).promoted


# ---------------------------------------------------------------- signal 3: repeated tool errors


def test_three_identical_tool_errors_promote():
    t = _traj(
        tool_calls=[
            ToolCall(name="execute", result="Error: ModuleNotFoundError: No module named 'foo'"),
            ToolCall(name="execute", result="Error: ModuleNotFoundError: No module named 'foo'"),
            ToolCall(name="execute", result="Error: ModuleNotFoundError: No module named 'foo'"),
        ],
    )
    v = classify_chat_outcome(t)
    assert v.promoted
    assert "execute" in v.reason


def test_two_identical_errors_do_not_promote():
    t = _traj(
        tool_calls=[
            ToolCall(name="execute", result="Error: bang"),
            ToolCall(name="execute", result="Error: bang"),
        ],
    )
    assert classify_chat_outcome(t).promoted is False


def test_three_different_errors_do_not_promote():
    """Different errors mean the agent IS exploring — that's not
    stuck, that's progress."""
    t = _traj(
        tool_calls=[
            ToolCall(name="execute", result="Error: A"),
            ToolCall(name="execute", result="Error: B"),
            ToolCall(name="execute", result="Error: C"),
        ],
    )
    assert classify_chat_outcome(t).promoted is False


def test_error_normalization_handles_whitespace_and_prefix():
    """Two tool errors that differ only in prefix / whitespace
    must hash to the same key so the count fires."""
    t = _traj(
        tool_calls=[
            ToolCall(name="execute", result="Error:  command not found"),
            ToolCall(name="execute", result="error: command not found"),
            ToolCall(name="execute", result="failed: command not found"),
        ],
    )
    assert classify_chat_outcome(t).promoted


# ---------------------------------------------------------------- signal 4: aborted browser sequence


def test_browser_sequence_aborted_promotes():
    """A goto failure that aborts the rest of the interact sequence
    leaves a clear "SEQUENCE ABORTED" banner in the result text;
    that's a signal the agent's URL was wrong and the rest of the
    plan never had a chance."""
    t = _traj(
        tool_calls=[ToolCall(
            name="browser",
            arguments={"operation": "interact", "url": "file:///bad.html"},
            result=(
                "⚠ SEQUENCE ABORTED: goto_failed. The first goto failed; "
                "subsequent actions were skipped."
            ),
        )],
    )
    v = classify_chat_outcome(t)
    assert v.promoted
    assert "abort" in v.reason.lower()


# ---------------------------------------------------------------- existing labels are respected


def test_existing_passed_outcome_not_overruled():
    t = _traj(
        outcome=Outcome.PASSED.value,
        final_response="[ATTEMPT_ABORTED_CROSS_TURN_LOOP]",
    )
    v = classify_chat_outcome(t)
    assert v.outcome == Outcome.PASSED.value
    assert v.promoted is False  # we only PROMOTE from UNKNOWN


def test_existing_failed_outcome_not_overruled():
    t = _traj(
        outcome=Outcome.FAILED.value,
        failure_reason="validator said no",
    )
    v = classify_chat_outcome(t)
    assert v.outcome == Outcome.FAILED.value
    assert v.promoted is False  # already FAILED — nothing to promote


# ---------------------------------------------------------------- mutating helper


def test_apply_helper_mutates_in_place():
    t = _traj(final_response="[ATTEMPT_ABORTED_CROSS_TURN_LOOP]")
    changed = apply_chat_outcome_heuristics(t)
    assert changed is True
    assert t.outcome == Outcome.FAILED.value
    assert t.failure_reason  # populated


def test_apply_helper_returns_false_when_no_promotion():
    t = _traj(user_request="hi", final_response="hello")
    changed = apply_chat_outcome_heuristics(t)
    assert changed is False
    assert t.outcome == Outcome.UNKNOWN.value
    assert t.failure_reason == ""


def test_apply_helper_does_not_overwrite_existing_failure_reason():
    """If the recorder has already set failure_reason for some
    other reason, classification should not clobber it."""
    t = _traj(
        final_response="[ATTEMPT_ABORTED_THINKING_LOOP]",
        failure_reason="pre-existing reason",
    )
    apply_chat_outcome_heuristics(t)
    assert t.outcome == Outcome.FAILED.value
    assert t.failure_reason == "pre-existing reason"


# ---------------------------------------------------------------- integration: reflector picks up promoted trajectories


def test_reflector_iterates_promoted_trajectories():
    """End-to-end shape of fix 4: a promoted UNKNOWN→FAILED
    trajectory must satisfy the Reflector's input filter."""
    from ghost_agent.reflection.loop import Reflector
    import asyncio

    t = _traj(
        user_request="build a thing",
        final_response="[ATTEMPT_ABORTED_CROSS_TURN_LOOP]",
    )
    apply_chat_outcome_heuristics(t)
    assert t.outcome == Outcome.FAILED.value

    # Stub critique returns a parseable reflection so we know the
    # trajectory was actually picked up by the loop.
    def fake_critique(prompt: str) -> str:
        return (
            "DIAGNOSIS: agent went in a loop\n"
            "1. step out of the loop\n"
            "2. plan smaller\n"
        )

    refl = Reflector(critique_fn=fake_critique, per_call_timeout_s=1.0, max_failures=1)

    async def go():
        return await refl.run(failed_source=[t])

    report = asyncio.run(go())
    assert report.seen_failures == 1
    assert report.reflected_ok == 1, (
        "promoted FAILED trajectory must be visible to the Reflector — "
        "this is the whole point of the heuristic"
    )


# ---------------------------------------------------------------- defensive: malformed trajectories


def test_classification_handles_empty_tool_calls():
    t = _traj(tool_calls=[], final_response="ok")
    v = classify_chat_outcome(t)
    assert v.promoted is False


def test_classification_handles_missing_arguments():
    t = _traj(tool_calls=[ToolCall(name="browser", arguments={})])
    v = classify_chat_outcome(t)
    assert v.promoted is False


def test_classification_handles_non_dict_action_in_actions_list():
    """Defensive — a malformed actions list entry must not crash."""
    t = _traj(tool_calls=[ToolCall(
        name="browser",
        arguments={"actions": [None, "not-a-dict", {"action": "click", "selector": "#x"}]},
    )])
    # Just must not raise — outcome doesn't matter.
    classify_chat_outcome(t)


# --------------------------------------------- #27d: structured ToolCall.error


def test_structured_error_flag_drives_repeated_error_signal():
    """When ToolCall.error is populated (now done on the chat path too), the
    repeated-identical-error signal fires even if the result TEXT wouldn't trip
    the sniff — e.g. a native-tools corruption shape."""
    # Result text is deliberately NOT sniff-detectable (no 'error:'/traceback).
    weird = "unexpected internal state 0xdeadbeef quux"
    t = _traj(tool_calls=[
        ToolCall(name="introspect", result=weird, error="introspect :: bad args"),
        ToolCall(name="introspect", result=weird, error="introspect :: bad args"),
        ToolCall(name="introspect", result=weird, error="introspect :: bad args"),
    ])
    c = classify_chat_outcome(t)
    assert c.outcome == Outcome.FAILED.value
    assert "introspect" in c.reason


def test_text_sniff_still_works_without_flag():
    """Legacy trajectories with no ToolCall.error still classify via the text
    fallback."""
    err = "Error: something broke badly"
    t = _traj(tool_calls=[
        ToolCall(name="execute", result=err),
        ToolCall(name="execute", result=err),
        ToolCall(name="execute", result=err),
    ])
    assert classify_chat_outcome(t).outcome == Outcome.FAILED.value


def test_tool_call_failed_prefers_flag():
    from ghost_agent.distill.outcome_heuristics import _tool_call_failed
    # Flag set, clean text → still failed.
    assert _tool_call_failed(ToolCall(name="x", result="all good", error="boom")) is True
    # No flag, clean text → not failed.
    assert _tool_call_failed(ToolCall(name="x", result="all good")) is False
    # No flag, error-ish text → failed via fallback.
    assert _tool_call_failed(ToolCall(name="x", result="Traceback (most recent call last)")) is True


def test_recorder_populates_error_flag_on_chat_path():
    """The trajectory recorder must set ToolCall.error for a failing chat tool
    result (integration guard on the agent-side wiring)."""
    import inspect
    from ghost_agent.core.agent import GhostAgent
    src = inspect.getsource(GhostAgent._record_turn_trajectory)
    assert "obj.error = _normalize_tool_error" in src
    assert "_looks_like_tool_error(obj.result)" in src
