"""Unit tests for ghost_agent.core.metacog_log — the structured log
helper that backs every uplift log line.

Contract:
  * `metacog <subsystem>` title prefix
  * key=value content, space-separated, insertion-ordered
  * floats round to 2dp
  * booleans render yes/no
  * None renders as -
  * strings with whitespace auto-quoted
  * never raises on weird inputs
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from ghost_agent.core.metacog_log import (
    LEVEL_DEBUG,
    LEVEL_ERROR,
    LEVEL_INFO,
    LEVEL_WARN,
    Subsystem,
    _fmt_value,
    _format_fields,
    emit,
)


# ──────────────────────────────────────────────────────────────────────
# _fmt_value
# ──────────────────────────────────────────────────────────────────────

class TestFmtValue:
    def test_none_renders_as_dash(self):
        assert _fmt_value(None) == "-"

    def test_bool_renders_yes_no(self):
        assert _fmt_value(True) == "yes"
        assert _fmt_value(False) == "no"

    def test_float_two_decimals(self):
        assert _fmt_value(0.123456) == "0.12"
        assert _fmt_value(0.0) == "0.00"
        assert _fmt_value(1.0) == "1.00"

    def test_float_nan_renders_as_nan(self):
        assert _fmt_value(float("nan")) == "nan"

    def test_int_unchanged(self):
        assert _fmt_value(42) == "42"

    def test_string_unquoted_when_no_whitespace(self):
        assert _fmt_value("execute") == "execute"
        assert _fmt_value("postgres_admin") == "postgres_admin"

    def test_string_quoted_with_spaces(self):
        assert _fmt_value("diverged plans") == '"diverged plans"'

    def test_string_with_quotes_escaped(self):
        assert _fmt_value('a "quoted" thing') == '"a \\"quoted\\" thing"'

    def test_empty_string_quoted(self):
        assert _fmt_value("") == '""'

    def test_newline_quoted(self):
        assert "\\n" not in _fmt_value("a\nb")  # don't literalise \n
        # But it IS quoted
        assert _fmt_value("a\nb").startswith('"')


# ──────────────────────────────────────────────────────────────────────
# _format_fields
# ──────────────────────────────────────────────────────────────────────

class TestFormatFields:
    def test_ordered_pairs(self):
        out = _format_fields({"a": 1, "b": "x", "c": 0.5})
        assert out == "a=1 b=x c=0.50"

    def test_empty_dict(self):
        assert _format_fields({}) == ""

    def test_mixed_types(self):
        out = _format_fields({
            "tool": "execute", "action": "ask_user",
            "sim": 0.42, "below_t": True,
            "reason": "diverged plans", "task": None,
        })
        # Each pair is verified by spot check
        assert "tool=execute" in out
        assert "action=ask_user" in out
        assert "sim=0.42" in out
        assert "below_t=yes" in out
        assert 'reason="diverged plans"' in out
        assert "task=-" in out


# ──────────────────────────────────────────────────────────────────────
# emit
# ──────────────────────────────────────────────────────────────────────

class TestEmit:
    def test_emit_calls_pretty_log_with_correct_title(self):
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            emit(Subsystem.ARBITER,
                 tool="execute", action="ask_user", sim=0.42)
            assert pl.called
            args, kwargs = pl.call_args
            # Title is the first positional arg
            assert args[0] == "Metacog Arbiter"
            # Content is the second positional arg
            assert "tool=execute" in args[1]
            assert "action=ask_user" in args[1]
            assert "sim=0.42" in args[1]
            assert kwargs["level"] == "INFO"

    def test_emit_respects_level(self):
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            emit(Subsystem.HOST, level=LEVEL_ERROR, severity="critical")
            args, kwargs = pl.call_args
            assert kwargs["level"] == "ERROR"

    def test_emit_falls_back_to_logger_when_pretty_log_unavailable(self, caplog):
        with patch("ghost_agent.utils.logging.pretty_log",
                   side_effect=ImportError("no")):
            with caplog.at_level(logging.INFO, logger="GhostAgent"):
                emit(Subsystem.BOOT, level=LEVEL_INFO, enabled=True)
        # Fallback path emitted SOMETHING through stdlib logger
        assert any("metacog boot" in r.message.lower()
                   for r in caplog.records)

    def test_emit_never_raises_on_weird_field_value(self):
        # An object whose str() raises should not bubble up
        class Bad:
            def __str__(self):
                raise RuntimeError("nope")

        # Should not raise
        emit(Subsystem.CONF, bad=Bad())

    def test_subsystem_labels_stable(self):
        # If this test breaks, every operator grep on the old labels
        # breaks too — treat as a contract.
        assert Subsystem.BOOT == "boot"
        assert Subsystem.CONF == "conf"
        assert Subsystem.ARBITER == "arbiter"
        assert Subsystem.VALID == "valid"
        assert Subsystem.HOST == "host"
        assert Subsystem.REPLAN == "replan"
        assert Subsystem.GATE == "gate"
        assert Subsystem.SUMMARY == "summary"

    def test_title_capitalised_for_pretty_log(self):
        """pretty_log lowercases titles itself, so we pass 'Metacog X'
        and let it normalise. This test just pins the format."""
        with patch("ghost_agent.utils.logging.pretty_log") as pl:
            emit(Subsystem.REPLAN, action="revised", task="t42")
            assert pl.call_args[0][0] == "Metacog Replan"
