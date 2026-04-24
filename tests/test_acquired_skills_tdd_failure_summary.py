"""Regression tests for the TDD-failure one-line summary helper.

Before: ``pretty_log("TEST FAILED", f"Skill '{name}' failed its TDD test.", …)``
printed only the generic line, so the operator reading the trace had
to grep the agent log to find out WHY. The LLM itself saw the full
``execution_result`` in the tool return value (which is why the
self-correction loop works), but the human-visible trace line was
generic — not diagnosable from 20 feet away.

After: ``_summarise_tdd_failure`` extracts a short, specific cause
string that gets appended to the pretty_log. The helper must:

  * Pick the traceback's final ``<Err>: <msg>`` line when the body
    is a Python traceback (the most actionable summary).
  * Surface the sanitizer's ``Syntax Error Detected: …`` line.
  * Distinguish the "no stdout" case with a dedicated sentence.
  * Fall back to the first non-empty body line when nothing else
    matches.
  * Never raise — it's pure log-surface polish; a defensive bug here
    must not crash skill creation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ghost_agent.tools.acquired_skills import _summarise_tdd_failure


# ---------------------------------------------------------------------------
# Python traceback → last-line extraction
# ---------------------------------------------------------------------------


class TestTracebackExtraction:
    def test_picks_valueerror_last_line(self):
        """The 2026-04-24 generate_secure_password incident shape.
        The user's Turn-1 TDD failure was ValueError because the
        __main__ block did int(sys.argv[1]) on a JSON test_payload."""
        body = (
            "--- EXECUTION RESULT ---\n"
            "EXIT CODE: 1\n"
            "STDOUT/STDERR:\n"
            "Traceback (most recent call last):\n"
            "  File \"test_skill.py\", line 15, in <module>\n"
            "    length = int(sys.argv[1])\n"
            "ValueError: invalid literal for int() with base 10: '{\"length\": 16}'\n"
        )
        summary = _summarise_tdd_failure(body)
        assert summary.startswith("ValueError:")
        assert "invalid literal for int()" in summary
        # Full line surfaces the offending payload.
        assert '{"length": 16}' in summary

    def test_picks_keyerror_last_line(self):
        body = (
            "--- EXECUTION RESULT ---\n"
            "EXIT CODE: 1\n"
            "STDOUT/STDERR:\n"
            "Traceback (most recent call last):\n"
            "  File \"test_skill.py\", line 9, in <module>\n"
            "    main(args['missing'])\n"
            "KeyError: 'missing'\n"
        )
        assert _summarise_tdd_failure(body) == "KeyError: 'missing'"

    def test_picks_importerror(self):
        body = (
            "EXIT CODE: 1\n"
            "STDOUT/STDERR:\n"
            "Traceback (most recent call last):\n"
            "  File \"test_skill.py\", line 1, in <module>\n"
            "    import missing_dep\n"
            "ModuleNotFoundError: No module named 'missing_dep'\n"
        )
        summary = _summarise_tdd_failure(body)
        assert summary.startswith("ModuleNotFoundError:")
        assert "missing_dep" in summary


# ---------------------------------------------------------------------------
# Sanitizer surface errors
# ---------------------------------------------------------------------------


class TestSanitizerSurface:
    def test_syntax_error_detected_line_surfaces(self):
        body = (
            "--- EXECUTION RESULT ---\n"
            "EXIT CODE: 1\n"
            "STDOUT/STDERR:\n"
            "Syntax Error Detected: SyntaxError: invalid syntax (<unknown>, line 1)\n"
            "Please fix the code and try again.\n"
        )
        summary = _summarise_tdd_failure(body)
        assert summary.startswith("Syntax Error Detected:")
        assert "invalid syntax" in summary

    def test_system_error_surface(self):
        body = (
            "--- EXECUTION RESULT ---\n"
            "EXIT CODE: 1\n"
            "STDOUT/STDERR:\n"
            "SYSTEM ERROR: FORBIDDEN IMPORT DETECTED -> 'browser'\n"
            "CRITICAL: 'browser' is a Native JSON Tool...\n"
        )
        summary = _summarise_tdd_failure(body)
        assert summary.startswith("SYSTEM ERROR:")
        assert "FORBIDDEN IMPORT" in summary


# ---------------------------------------------------------------------------
# No-stdout sentinel
# ---------------------------------------------------------------------------


class TestNoStdoutSentinel:
    def test_silent_success_identified(self):
        """The "script ran but printed nothing" case has a dedicated
        sentence so the operator can distinguish it at a glance from
        a real runtime error. This is the shape the if-name-equals
        -main-forgotten failure takes."""
        body = (
            "--- EXECUTION RESULT ---\n"
            "EXIT CODE: 0\n"
            "STDOUT/STDERR:\n"
            "(Process executed successfully, but no output was printed to stdout.)\n"
        )
        summary = _summarise_tdd_failure(body)
        assert "printed nothing" in summary.lower() or "no output" in summary.lower()


# ---------------------------------------------------------------------------
# Fallback + defensive behaviour
# ---------------------------------------------------------------------------


class TestFallbackAndDefensive:
    def test_first_line_fallback_when_no_patterns_match(self):
        body = (
            "--- EXECUTION RESULT ---\n"
            "EXIT CODE: 1\n"
            "STDOUT/STDERR:\n"
            "Custom error message without an exception format\n"
            "Another line\n"
        )
        summary = _summarise_tdd_failure(body)
        assert "Custom error message" in summary

    def test_empty_string_returns_placeholder(self):
        summary = _summarise_tdd_failure("")
        assert summary == "unknown cause"

    def test_none_input_returns_placeholder(self):
        summary = _summarise_tdd_failure(None)
        assert summary == "unknown cause"

    def test_long_summary_is_truncated_to_200_chars(self):
        """A giant stderr line can't dominate the trace row."""
        giant = "ValueError: " + ("x" * 500)
        body = (
            "STDOUT/STDERR:\n"
            f"{giant}\n"
        )
        summary = _summarise_tdd_failure(body)
        assert len(summary) <= 200
        assert summary.startswith("ValueError:")

    def test_body_with_only_blank_lines_returns_placeholder(self):
        body = "STDOUT/STDERR:\n\n\n   \n"
        summary = _summarise_tdd_failure(body)
        assert summary == "no diagnostic output"

    def test_no_header_marker_still_extracts(self):
        """When _format_error didn't wrap the output (edge case), the
        helper still picks the last traceback line."""
        body = (
            "Traceback (most recent call last):\n"
            "  File \"x.py\", line 1, in <module>\n"
            "TypeError: unhashable type: 'list'\n"
        )
        assert _summarise_tdd_failure(body) == "TypeError: unhashable type: 'list'"

    def test_never_raises_on_weird_input(self):
        """Defensive contract: pure log-polish must never break skill
        creation. Hand it garbage shapes and confirm no exception."""
        for weird in [b"bytes", 42, object(), ["a", "b"], {"k": "v"}]:
            # Should not raise — _summarise_tdd_failure catches all.
            _summarise_tdd_failure(weird)  # noqa: no assertion needed


# ---------------------------------------------------------------------------
# Integration: the pretty_log line uses the helper
# ---------------------------------------------------------------------------


def test_tool_create_skill_wires_cause_into_pretty_log():
    """Grep-assert that `tool_create_skill` actually calls the helper
    and interpolates the result into the pretty_log message. A future
    refactor that dropped the interpolation would silently regress
    the operator-visible trace."""
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "ghost_agent" / "tools" / "acquired_skills.py"
    ).read_text()
    # The helper is called and its result is interpolated into the
    # pretty_log message.
    assert "_summarise_tdd_failure(execution_result)" in src
    assert "failed its TDD test — {" in src or "failed its TDD test — {_cause}" in src
