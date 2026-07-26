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


# --- paragraph-repeat detector (2026-07-25, req f59a793d) --------------------
# The exact-tail n-gram probe needs the CURRENT 200-char tail to have occurred
# 3x, so a paraphrase loop that interleaves varied filler between verbatim
# planning paragraphs evaded it for ~19.5K chars (~3 minutes) live. Whole-line
# repetition in the THINKING channel is a much earlier signature.


def test_paragraph_loop_reexported_from_agent():
    assert A._detect_paragraph_loop is SG._detect_paragraph_loop
    assert A.PARAGRAPH_LOOP_MIN_LINE == SG.PARAGRAPH_LOOP_MIN_LINE
    assert A.PARAGRAPH_LOOP_THRESHOLD == SG.PARAGRAPH_LOOP_THRESHOLD


def test_paragraph_loop_fires_on_verbatim_repeats_with_varied_filler():
    para = "Let me write this now. I'll be comprehensive but efficient."
    buf = "".join(
        f"Considering angle {i} of the data model restructure in detail.\n"
        + para + "\n"
        for i in range(60)               # well past threshold AND the floor
    ) + "And now let me actually"        # trailing mid-line fragment
    assert len(buf) >= SG.PARAGRAPH_LOOP_MIN_BUF
    assert SG._detect_paragraph_loop(buf) is True
    # The generic n-gram probe does NOT fire on this shape — that gap is
    # exactly why the paragraph detector exists. (If this ever starts
    # passing, the window sizes changed and this test should be revisited.)
    assert SG._detect_thinking_loop(buf) is False


def test_paragraph_loop_floor_protects_small_buffers():
    # The live false-positive storm (2026-07-25 second deploy) aborted a
    # spec think at 3,019 chars. Below the floor NOTHING fires, no matter
    # how repetitive.
    para = "This exact planning sentence repeats a suspicious number of times."
    buf = (para + "\n") * 20
    buf = buf[:SG.PARAGRAPH_LOOP_MIN_BUF - 100]
    assert SG._detect_paragraph_loop(buf) is False


def _padded(body, target=None):
    """Prefix diverse filler so the buffer clears the evaluation floor."""
    target = target or (SG.PARAGRAPH_LOOP_MIN_BUF + 500)
    filler = []
    i = 0
    while sum(len(f) for f in filler) + len(body) < target:
        filler.append(
            f"Background thought {i}: an unrelated distinct planning "
            f"sentence about component number {i}.\n")
        i += 1
    return "".join(filler) + body


def test_paragraph_loop_novel_interleaved_line_does_not_mask_repeats():
    para = "This exact planning sentence repeats a suspicious number of times."
    body = "".join(
        para + "\n" + f"unique separator line {i} long enough to be scanned\n"
        for i in range(SG.PARAGRAPH_LOOP_THRESHOLD)
    ) + "tail fragment"
    # Newest completed long line is a UNIQUE separator; the scan window
    # (PARAGRAPH_LOOP_SCAN_LINES > 1) still reaches the repeated line.
    assert SG._detect_paragraph_loop(_padded(body)) is True


def test_paragraph_loop_below_threshold_is_clean():
    para = "This exact planning sentence repeats a suspicious number of times."
    body = "".join(
        para + "\n" + f"unique separator line {i} long enough to be scanned\n"
        for i in range(SG.PARAGRAPH_LOOP_THRESHOLD - 1)
    ) + "tail fragment"
    assert SG._detect_paragraph_loop(_padded(body)) is False


def test_paragraph_loop_ignores_short_line_repeats():
    # Short interjections legitimately repeat in reasoning ("Yes.", "Hmm.");
    # under MIN_LINE they are never counted.
    body = ("Yes.\n" * 40) + "One final distinct line that is long enough here.\n" + "t"
    assert SG._detect_paragraph_loop(_padded(body)) is False


def test_paragraph_loop_diverse_prose_is_clean():
    buf = "\n".join(
        f"Step {i}: a distinct reasoning sentence about part {i} of the plan."
        for i in range(150)
    )
    assert len(buf) >= SG.PARAGRAPH_LOOP_MIN_BUF
    assert SG._detect_paragraph_loop(buf) is False
    assert SG._detect_paragraph_loop("") is False
    assert SG._detect_paragraph_loop("short") is False
