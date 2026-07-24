"""Durable-mirror logging (2026-07-24).

pretty_log writes an aligned, TRUNCATED line to the operator's stdout stream
AND its FULL, untruncated content to a durable file sink (the "GhostStream"
mirror logger) so `$GHOST_HOME/system/ghost-agent.log` becomes a complete,
plain-text, restart-surviving record of what the agent did — the log you can
grep or hand to a reader to reconstruct a turn.

These tests pin: (1) the mirror captures full content the stream truncates,
(2) BEGIN/END request frames are mirrored, (3) the mirror is file-only so it
never double-prints on the operator's stdout stream.
"""
from __future__ import annotations

import logging

from ghost_agent.utils.logging import (
    setup_logging, pretty_log, Icons, request_id_context, _PrettyLogHandler,
)


def test_mirror_captures_full_content_the_stream_truncates(tmp_path):
    log_file = tmp_path / "g.log"
    setup_logging(str(log_file), daemon=False)
    long = "A" * 200 + " TAIL_MARKER"   # INFO stream truncates at 60; tail is lost there
    tok = request_id_context.set("abcdef12")
    try:
        pretty_log("shell command", long, icon=Icons.TOOL_SHELL, level="INFO")
    finally:
        request_id_context.reset(tok)
    text = log_file.read_text()
    assert "shell command" in text
    # The whole thing survives in the durable mirror, tail included.
    assert "A" * 200 in text
    assert "TAIL_MARKER" in text


def test_begin_and_end_frames_are_mirrored(tmp_path):
    log_file = tmp_path / "g.log"
    setup_logging(str(log_file), daemon=False)
    tok = request_id_context.set("req12345")
    try:
        pretty_log("start", special_marker="BEGIN")
        pretty_log("end", special_marker="END")
    finally:
        request_id_context.reset(tok)
    text = log_file.read_text()
    assert "request started" in text
    assert "request finished" in text
    assert "req12345" in text  # request id correlates the turn's lines


def test_mirror_logger_is_file_only_no_double_stdout(tmp_path):
    setup_logging(str(tmp_path / "g.log"), daemon=False)
    ml = logging.getLogger("GhostStream")
    # propagate=False + no console/pretty handler => a mirror line lands only in
    # the file, never a second time on the operator's stdout stream.
    assert ml.propagate is False
    assert not any(isinstance(h, _PrettyLogHandler) for h in ml.handlers)
    assert any(isinstance(h, logging.FileHandler) for h in ml.handlers)


def test_warning_level_preserved_in_mirror(tmp_path):
    log_file = tmp_path / "g.log"
    setup_logging(str(log_file), daemon=False)
    tok = request_id_context.set("warnreq1")
    try:
        pretty_log("verifier", "REFUTED: the build did not compile",
                   icon=Icons.WARN, level="WARNING")
    finally:
        request_id_context.reset(tok)
    text = log_file.read_text()
    # The mirror line carries WARNING level so `grep WARNING` finds it durably.
    assert "REFUTED: the build did not compile" in text
    assert "WARNING" in text
