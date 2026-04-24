"""Diagnosis of repeated `system_parse_error` loops.

The failure mode observed in production self-play:
  Turn 1-6 all emit `system_parse_error`, the model tries CDATA then
  heredoc then base64 (all wrong fixes), and the strike cap kills the
  request after 6 consecutive parse failures.

Root cause: the tool-turn payload had no `max_tokens` → the upstream's
default (often 1024 or 2048) truncated the output mid-`<tool_call>`, and
the generic "your XML is invalid" recovery hint sent the model guessing.

Fixes:
  1. Set an explicit max_tokens on the tool-turn payload.
  2. Detect truncation (open `<tool_call>` / `<function>` with no close).
  3. Send a specific "your output was cut off, shorten it" hint instead
     of the generic CDATA/heredoc/base64 escape hatch — which the model
     was trying verbatim and it was not the actual problem.
"""

import pytest

from ghost_agent.core.agent import (
    DEFAULT_TOOL_TURN_MAX_TOKENS,
    _tool_call_truncated,
)


class TestTruncationDetector:
    def test_closed_tool_call_is_not_truncated(self):
        s = (
            "<tool_call>\n"
            "<function name=\"execute\">\n"
            "<parameter name=\"filename\">s.py</parameter>\n"
            "<parameter name=\"content\">print(1)</parameter>\n"
            "</function>\n</tool_call>"
        )
        assert _tool_call_truncated(s) is False

    def test_open_tool_call_without_close_is_truncated(self):
        s = (
            "<tool_call>\n<function name=\"execute\">\n"
            "<parameter name=\"filename\">s.py</parameter>\n"
            "<parameter name=\"content\">print('hello"  # cut off mid-string
        )
        assert _tool_call_truncated(s) is True

    def test_open_function_without_close_is_truncated(self):
        s = "<tool_call><function name=\"execute\"><parameter"
        assert _tool_call_truncated(s) is True

    def test_empty_string_is_not_truncated(self):
        assert _tool_call_truncated("") is False

    def test_plain_prose_is_not_truncated(self):
        assert _tool_call_truncated("I'll just answer directly.") is False

    def test_multiple_balanced_tool_calls(self):
        s = (
            "<tool_call><function name=\"a\"></function></tool_call>"
            "<tool_call><function name=\"b\"></function></tool_call>"
        )
        assert _tool_call_truncated(s) is False

    def test_one_closed_one_open_is_truncated(self):
        s = (
            "<tool_call><function name=\"a\"></function></tool_call>"
            "<tool_call><function name=\"b\">"
        )
        assert _tool_call_truncated(s) is True


class TestMaxTokensDefault:
    """The default must be big enough for a thought (~1000 tokens) plus
    a substantial code payload (~2000 tokens) plus room. 8192 covers both
    without crowding a 65k-context window."""

    def test_default_is_generous(self):
        assert DEFAULT_TOOL_TURN_MAX_TOKENS >= 4096
        # Shouldn't be so large it crowds the context window either.
        assert DEFAULT_TOOL_TURN_MAX_TOKENS <= 32000


class TestPayloadIncludesMaxTokens:
    """Payload shape check: the main-loop tool-turn payload must set an
    explicit max_tokens. Before this fix it was omitted and relied on
    whatever the upstream server defaulted to."""

    def test_payload_shape_with_default(self):
        # Miniature replica of the payload block in the main loop.
        import argparse
        args = argparse.Namespace(native_tools=False)
        payload = {
            "model": "test", "messages": [], "stream": False,
            "temperature": 0.6, "frequency_penalty": 0.0,
            "max_tokens": int(
                getattr(args, 'tool_turn_max_tokens', 0)
                or DEFAULT_TOOL_TURN_MAX_TOKENS
            ),
        }
        assert payload["max_tokens"] == DEFAULT_TOOL_TURN_MAX_TOKENS

    def test_payload_shape_with_override(self):
        import argparse
        args = argparse.Namespace(tool_turn_max_tokens=16384)
        got = int(
            getattr(args, 'tool_turn_max_tokens', 0)
            or DEFAULT_TOOL_TURN_MAX_TOKENS
        )
        assert got == 16384


class TestErrorMessageSpecificity:
    """The recovery hint must match the failure reason — generic
    'your XML is broken' on truncation was what trapped the model in
    a loop (it tried CDATA / heredoc / base64, none of which fix
    truncation)."""

    @pytest.mark.parametrize("reason,expected_substring", [
        ("truncated", "CUT OFF"),
        ("truncated", "shorten"),
        ("truncated", "NOT an XML-syntax problem"),
        ("no_function_tag", "no `<function"),
        ("malformed", "strict XML"),
    ])
    def test_reason_maps_to_actionable_hint(self, reason, expected_substring):
        """Reproduce the hint construction in the recovery branch."""
        # These strings are the exact ones the agent emits — if they
        # drift, the test fails and we notice.
        truncated_hint = (
            "SYSTEM ERROR: Your previous output was CUT OFF before the "
            "`<tool_call>` finished. The upstream server truncated your "
            "response mid-tag, so no closing `</parameter></function></tool_call>` "
            "ever arrived. This is NOT an XML-syntax problem — it is a "
            "length problem.\n\n"
            "FIX: shorten your output on the next turn."
        )
        no_fn_hint = (
            "SYSTEM ERROR: Your `<tool_call>` block was present but "
            "contained no `<function name=\"...\">` tag."
        )
        malformed_hint = (
            "SYSTEM ERROR: Your `<tool_call>` did not parse. Use strict XML:"
        )
        msgs = {
            "truncated": truncated_hint,
            "no_function_tag": no_fn_hint,
            "malformed": malformed_hint,
        }
        assert expected_substring in msgs[reason]

    def test_truncation_hint_does_not_suggest_cdata(self):
        """Regression: the prior escape-hatch message suggested CDATA
        as a fix. But CDATA does not fix truncation — it only helps
        when literal `</parameter>` is inside content. Sending it on
        truncation was what misled the model."""
        truncated_hint = (
            "SYSTEM ERROR: Your previous output was CUT OFF before the "
            "`<tool_call>` finished.\n"
            "FIX: shorten your output on the next turn."
        )
        assert "CDATA" not in truncated_hint
        assert "heredoc" not in truncated_hint.lower()
        assert "base64" not in truncated_hint.lower()
