"""The stream-guard seam module (IMPROVEMENTS.md #5).

First module of the guard seam: the pure streaming guards moved out of the
11k-line agent.py into their own testable module. agent.py re-exports them so
all existing references keep working. New stream guards should land HERE.
"""
from ghost_agent.core import stream_guards as SG
from ghost_agent.core import agent as A


def test_reexported_from_agent_are_identical():
    # Behavior-preserving move: the names in agent.py ARE the module's objects.
    assert A._detect_thinking_loop is SG._detect_thinking_loop
    assert A._detect_tool_call_loop is SG._detect_tool_call_loop
    assert A._tail_has_stop_marker is SG._tail_has_stop_marker
    assert A.THINKING_LOOP_WINDOW == SG.THINKING_LOOP_WINDOW
    assert A.TOOL_CALL_LOOP_PROBE_EVERY == SG.TOOL_CALL_LOOP_PROBE_EVERY


def test_detect_thinking_loop_fires_on_repetition():
    assert SG._detect_thinking_loop(("The answer is 42. " * 40)) is True
    assert SG._detect_thinking_loop("a single diverse non-repeating sentence.") is False
    assert SG._detect_thinking_loop("") is False


def test_detect_tool_call_loop_on_unclosed_opens():
    assert SG._detect_tool_call_loop("<tool_call>" * 50) is True
    # Balanced opens/closes → not a collapse.
    assert SG._detect_tool_call_loop("<tool_call></tool_call>" * 3) is False
    assert SG._detect_tool_call_loop("") is False


def test_tail_marker_is_bounded():
    huge = "x" * 500_000
    assert SG._tail_has_stop_marker(huge + "</think", "k") is True
    assert SG._tail_has_stop_marker(huge + "plain text", "t") is False


def test_guards_live_in_the_module_not_inline():
    """Guard against regression: the guard definitions must live in
    stream_guards.py; agent.py imports them (the seam)."""
    import inspect
    from pathlib import Path
    agent_src = Path(inspect.getfile(A)).read_text()
    assert "from .stream_guards import" in agent_src
    # The old inline `def _detect_thinking_loop` must be gone from agent.py.
    assert "def _detect_thinking_loop(buf" not in agent_src
