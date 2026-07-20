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


def test_parse_word_plan_in_diagnosis_does_not_hijack_plan_section():
    """`DIAGNOSIS: The plan failed because:` contains the bare word
    "plan" — it must NOT open the plan section, or the diagnosis
    bullets get captured as the "revised plan" and the blank-line
    break stops before the real REVISED PLAN."""
    text = ("DIAGNOSIS: The plan failed because:\n"
            "- the file was missing\n"
            "\n"
            "REVISED PLAN:\n"
            "1. Use browser\n"
            "2. ...")
    d, plan = parse_reflection_output(text)
    assert "plan failed" in d.lower()
    assert plan[0] == "Use browser"
    assert "the file was missing" not in " ".join(plan)


def test_parse_plan_header_with_inline_first_step():
    """The header line may carry the first step after the colon."""
    text = """DIAGNOSIS: x
REVISED PLAN: 1. alpha
2. beta
"""
    d, plan = parse_reflection_output(text)
    assert plan == ["alpha", "beta"]


def test_parse_midline_plan_prose_alone_is_not_a_section():
    """Prose mentioning "plan" with no real header must not fabricate
    a plan out of unrelated bullets."""
    text = ("DIAGNOSIS: The plan failed because:\n"
            "- the file was missing\n")
    d, plan = parse_reflection_output(text)
    assert "plan failed" in d.lower()
    assert plan == []
