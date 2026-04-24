"""Tests for reflection.prompts."""

from ghost_agent.distill.schema import Trajectory, ToolCall, Outcome
from ghost_agent.reflection.prompts import (
    REFLECTION_PROMPT_TEMPLATE,
    build_reflection_prompt,
    parse_reflection_output,
    _summarize_tool_calls,
    _truncate,
)


def test_prompt_template_has_placeholders():
    assert "{user_request}" in REFLECTION_PROMPT_TEMPLATE
    assert "{failure_reason}" in REFLECTION_PROMPT_TEMPLATE
    assert "{tried_summary}" in REFLECTION_PROMPT_TEMPLATE


def test_build_reflection_prompt_includes_trajectory_content():
    t = Trajectory(
        user_request="solve a hard puzzle",
        failure_reason="validator exit 1: output mismatch",
        tool_calls=[ToolCall(name="execute", arguments={"code": "print(1)"},
                             result="1", error="")],
    )
    p = build_reflection_prompt(t)
    assert "solve a hard puzzle" in p
    assert "validator exit 1" in p
    assert "execute" in p
    assert "print(1)" in p


def test_build_prompt_handles_missing_failure_reason():
    t = Trajectory(user_request="req", failure_reason="")
    p = build_reflection_prompt(t)
    assert "(validator reported no reason)" in p


def test_build_prompt_handles_no_tool_calls():
    t = Trajectory(user_request="req", failure_reason="bad")
    p = build_reflection_prompt(t)
    assert "(no tool calls on this attempt)" in p


def test_build_prompt_truncates_long_inputs():
    big = "X" * 5000
    t = Trajectory(user_request=big, failure_reason=big)
    p = build_reflection_prompt(t)
    # We cap user_request default at 1200 chars; should include truncation marker
    assert "[truncated" in p


def test_truncate_preserves_short_text():
    assert _truncate("short", 100) == "short"
    assert _truncate("", 10) == ""
    assert _truncate(None, 10) == ""


def test_summarize_tool_calls_empty():
    assert _summarize_tool_calls([]) == "(no tool calls on this attempt)"


def test_summarize_tool_calls_multiple():
    calls = [
        ToolCall(name="file_system", arguments={"action": "read"}, result="contents"),
        ToolCall(name="execute", arguments={"code": "print(1)"},
                 result="1", error="no error"),
    ]
    s = _summarize_tool_calls(calls)
    assert "file_system" in s
    assert "execute" in s
    assert "1. " in s  # numbered
    assert "2. " in s


# -----------------------------------------------------------------
# parse_reflection_output
# -----------------------------------------------------------------

def test_parse_valid_output():
    text = """DIAGNOSIS: the script used python2 syntax
REVISED PLAN:
1. switch to python3 syntax
2. re-run the validator
"""
    d, plan = parse_reflection_output(text)
    assert "python2 syntax" in d
    assert plan == ["switch to python3 syntax", "re-run the validator"]


def test_parse_output_with_extra_prose_before_sections():
    text = """Let me analyze this carefully.
DIAGNOSIS: missing import statement
REVISED PLAN:
1. add `import os` at top
2. re-run
"""
    d, plan = parse_reflection_output(text)
    assert "missing import" in d
    assert len(plan) == 2


def test_parse_empty_string_returns_empty():
    d, plan = parse_reflection_output("")
    assert d == ""
    assert plan == []


def test_parse_missing_plan_returns_diagnosis():
    text = "DIAGNOSIS: something broke"
    d, plan = parse_reflection_output(text)
    assert "something broke" in d
    assert plan == []


def test_parse_missing_diagnosis_returns_plan():
    text = """REVISED PLAN:
1. first thing
2. second thing
"""
    d, plan = parse_reflection_output(text)
    assert d == ""
    assert plan == ["first thing", "second thing"]


def test_parse_handles_bullet_markers():
    text = """DIAGNOSIS: x
REVISED PLAN:
- alpha
- beta
"""
    d, plan = parse_reflection_output(text)
    assert plan == ["alpha", "beta"]


def test_parse_handles_asterisk_bullets():
    text = """DIAGNOSIS: x
REVISED PLAN:
* foo
* bar
"""
    d, plan = parse_reflection_output(text)
    assert plan == ["foo", "bar"]


def test_parse_stops_plan_at_blank_line():
    text = """DIAGNOSIS: x
REVISED PLAN:
1. first
2. second

(end of response)
"""
    d, plan = parse_reflection_output(text)
    assert plan == ["first", "second"]


def test_parse_case_insensitive_headers():
    text = """diagnosis: lowercase worked
revised plan:
1. yes
"""
    d, plan = parse_reflection_output(text)
    assert "lowercase worked" in d
    assert plan == ["yes"]
