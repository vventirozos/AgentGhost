"""Tests for the logging-framework overhaul (Tier 0+1):

* setup_logging now configures ALL first-party loggers (not just
  GhostAgent) — selfhood/workspace/optim/distill/reflection were
  previously orphaned (no handlers → logs discarded).
* A _PrettyLogHandler routes stdlib WARNING+ records through pretty_log
  so failures appear in the monitored console stream with an icon/tag.
* WARNING/ERROR content gets a larger truncation budget than INFO.
* Duplicate icon glyphs were de-collided + new icons added.
"""

import io
import logging
from contextlib import redirect_stdout

import pytest

from ghost_agent.utils import logging as glog
from ghost_agent.utils.logging import (
    setup_logging, pretty_log, Icons, _GHOST_LOGGERS, request_id_context,
)


@pytest.fixture(autouse=True)
def _restore_logging():
    """Snapshot + restore each Ghost* logger's handlers/level so a test
    calling setup_logging (with a tmp log file) can't leak a FileHandler
    pointing at a deleted file into the rest of the suite."""
    saved = {n: (list(logging.getLogger(n).handlers), logging.getLogger(n).level)
             for n in _GHOST_LOGGERS}
    yield
    for n in _GHOST_LOGGERS:
        lg = logging.getLogger(n)
        orig_handlers, orig_level = saved[n]
        for h in list(lg.handlers):
            if h not in orig_handlers:
                try:
                    h.close()
                except Exception:
                    pass
                lg.removeHandler(h)
        for h in orig_handlers:
            if h not in lg.handlers:
                lg.addHandler(h)
        lg.setLevel(orig_level)


# -----------------------------------------------------------------
# Tier 0 — every subsystem logger is configured
# -----------------------------------------------------------------

def test_all_ghost_loggers_get_handlers(tmp_path):
    setup_logging(str(tmp_path / "g.log"), daemon=False)
    for name in _GHOST_LOGGERS:
        lg = logging.getLogger(name)
        assert lg.handlers, f"{name} has no handlers"
        assert any(isinstance(h, logging.FileHandler) for h in lg.handlers), \
            f"{name} has no FileHandler"


def test_orphan_logger_warning_reaches_file(tmp_path):
    log_file = tmp_path / "g.log"
    setup_logging(str(log_file), daemon=True)  # file-only
    logging.getLogger("GhostSelfhood").warning("selfhood probe XYZ")
    for h in logging.getLogger("GhostSelfhood").handlers:
        h.flush()
    text = log_file.read_text()
    assert "selfhood probe XYZ" in text          # previously discarded
    assert "GhostSelfhood" in text                # logger name now in the file format


# -----------------------------------------------------------------
# Tier 1 — WARNING+ routed through pretty_log to the console stream
# -----------------------------------------------------------------

def test_pretty_handler_routes_warning_to_stream(tmp_path):
    setup_logging(str(tmp_path / "g.log"), daemon=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        logging.getLogger("GhostAgent").error("disk on fire")
    out = buf.getvalue()
    assert "disk on fire" in out
    assert Icons.FAIL in out          # ❌ rendered, not a bare plain line
    assert "agent" in out.lower()     # subsystem-derived title


def test_pretty_handler_skips_info_and_debug(tmp_path):
    setup_logging(str(tmp_path / "g.log"), daemon=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        logging.getLogger("GhostAgent").info("just fyi")
        logging.getLogger("GhostAgent").debug("noise")
    # INFO/DEBUG stdlib logs do NOT pollute the pretty console stream.
    assert "just fyi" not in buf.getvalue()
    assert "noise" not in buf.getvalue()


def test_no_recursion_on_error_log_in_debug_mode(tmp_path):
    # In debug mode pretty_log mirrors to logger.debug; routing must not recurse.
    setup_logging(str(tmp_path / "g.log"), debug=True, daemon=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        logging.getLogger("GhostAgent").error("recursion probe")
    # Reaching here at all means no infinite loop; it appears exactly once.
    assert buf.getvalue().count("recursion probe") == 1


# -----------------------------------------------------------------
# Tier 1 — failure content survives, INFO stays tight
# -----------------------------------------------------------------

def _emit(level, content):
    token = request_id_context.set("ab12cd34")
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            pretty_log("probe", content, icon=Icons.WARN, level=level)
        return buf.getvalue()
    finally:
        request_id_context.reset(token)


def test_warning_content_not_truncated_at_60():
    out = _emit("ERROR", "boom " * 40)   # 200 chars
    assert out.count("boom") > 30        # survives well past the 60-char INFO cap


def test_info_content_truncated_at_60():
    out = _emit("INFO", "tick " * 40)
    assert out.count("tick") < 20        # INFO stays tight


# -----------------------------------------------------------------
# Tier 1 — icons de-collided + new icons present
# -----------------------------------------------------------------

def test_icon_glyphs_decollided():
    assert Icons.MEM_LIBRARY != Icons.MEM_INGEST
    assert Icons.MEM_SPLIT != Icons.CUT
    assert Icons.NODE_EDGE != Icons.SYSTEM_BOOT


def test_new_icons_present_and_distinct():
    new = {
        "TOOL_BROWSER", "IMAGE_GEN", "REPORT_PDF",
        "NODE_WORKER", "NODE_EDGE", "SELF_STATE", "SKILL_GRADUATE",
    }
    glyphs = []
    for name in new:
        g = getattr(Icons, name, None)
        assert g, f"Icons.{name} missing"
        glyphs.append(g)
    # The browser glyph in particular must differ from the shell glyph it used to borrow.
    assert Icons.TOOL_BROWSER != Icons.TOOL_SHELL


# -----------------------------------------------------------------
# guardrails preserved
# -----------------------------------------------------------------

def test_daemon_mode_no_console_handler(tmp_path):
    setup_logging(str(tmp_path / "g.log"), daemon=True)
    handlers = logging.getLogger("GhostAgent").handlers
    assert not any(isinstance(h, glog._PrettyLogHandler) for h in handlers)
    assert sum(isinstance(h, logging.FileHandler) for h in handlers) == 1


def test_setup_logging_idempotent_for_all_loggers(tmp_path):
    setup_logging(str(tmp_path / "g.log"), daemon=False)
    counts = {n: len(logging.getLogger(n).handlers) for n in _GHOST_LOGGERS}
    for _ in range(4):
        setup_logging(str(tmp_path / "g.log"), daemon=False)
    for n in _GHOST_LOGGERS:
        assert len(logging.getLogger(n).handlers) == counts[n], f"{n} accumulated handlers"
