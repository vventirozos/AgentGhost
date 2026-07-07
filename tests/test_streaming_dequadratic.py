"""De-quadratic streaming guards (IMPROVEMENTS.md #12).

The streaming display did O(buffer) work per token: two full-buffer `.lower()`
marker scans, an ungated full-buffer tool-call regex, a per-token n-gram loop
detector once thinking passed 32K, and a per-chunk full-content scrub sub —
each O(n) per token → O(n²) over a long stream, plus GB of transient string
allocation on the event loop. These are now bounded (tail scans + probe cadence
+ skip-when-clean) with the same detection behavior.
"""
import re

import pytest

from ghost_agent.core import agent as A


# ------------------------------------------------- tail stop-marker helper


def test_tail_marker_detects_think_close():
    buf = "some reasoning ... </think>"
    assert A._tail_has_stop_marker(buf, ">") is True


def test_tail_marker_detects_tool_call_open():
    buf = "text <tool_call"
    assert A._tail_has_stop_marker(buf, "l") is True


def test_tail_marker_false_when_absent():
    buf = "just a long ordinary thought with no markers " * 100
    assert A._tail_has_stop_marker(buf, "s") is False


def test_tail_marker_only_scans_recent_tail():
    # A marker far in the past (already latched earlier) is NOT re-detected by
    # a tail scan — which is fine, the caller latches stop_printing on first
    # sight. What matters: the scan cost is bounded, and a marker completing in
    # the CURRENT token is caught.
    huge = "x" * 500_000
    # Marker split so its final char arrives in this token.
    buf = huge + "</think"
    assert A._tail_has_stop_marker(buf, "k") is True
    # No marker in the tail → False regardless of buffer size.
    assert A._tail_has_stop_marker(huge + "plain", "n") is False


def test_tail_marker_equivalent_to_full_scan_at_boundary():
    """For any single-token growth, the tail check agrees with a full lower()
    scan about whether a marker JUST completed."""
    base = "reasoning " * 50
    for marker in ("</think>", "<tool_call>"):
        buf = base + marker
        token = marker[-1]
        assert A._tail_has_stop_marker(buf, token) is True


# ------------------------------------------------- probe cadence constants


def test_tool_call_probe_constant_is_used():
    """TOOL_CALL_LOOP_PROBE_EVERY was defined-but-never-consulted; the stream
    loop must now reference it to gate the tool-call collapse probe."""
    import inspect
    src = inspect.getsource(A.GhostAgent.handle_chat)
    assert "next_tool_probe" in src
    assert "TOOL_CALL_LOOP_PROBE_EVERY" in src


def test_thinking_loop_probe_gated_by_cadence():
    """The n-gram detector must be gated on next_loop_probe (not run per
    token in the 32-64K window)."""
    import inspect
    src = inspect.getsource(A.GhostAgent.handle_chat)
    # The old per-token boundary branch is gone; detection rides the cadence.
    assert "len(guard_buf) >= next_loop_probe" in src
    assert "len(guard_buf) > base_cap and len(guard_buf) <= extended_cap" not in src


def test_detectors_still_catch_loops():
    """Behavior preserved: the pure detector functions are unchanged and still
    fire on a genuine repetition."""
    loop = ("The answer is 42. " * 40)
    assert A._detect_thinking_loop(loop) is True
    spam = "<tool_call>" * 50
    assert A._detect_tool_call_loop(spam) is True
    assert A._detect_thinking_loop("a diverse non-repeating sentence here.") is False
