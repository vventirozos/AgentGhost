"""Streaming sanity guards — pure functions over the generation buffer.

This is the first module of the guard SEAM (IMPROVEMENTS.md #5). `core/agent.py`
grew to 11k+ lines because every hardening session appended another inline guard
to the streaming turn loop, each parametrised by yet another module constant at
the top of the file. These guards are pure functions of the buffer (no agent
state), so they belong in their own testable module — new stream guards should
land HERE and be referenced from the loop, not inlined into `handle_chat`.

`agent.py` re-exports these names for backward compatibility, so existing
references (and tests) keep working unchanged.
"""
import re

# --- thinking-loop (n-gram repetition) detector -----------------------------
THINKING_LOOP_PROBE_EVERY = 500     # run the repetition probe every N chars
THINKING_LOOP_WINDOW = 200          # length of the n-gram we look for
THINKING_LOOP_THRESHOLD = 3         # window appearing >= N times = loop
# Conservative (500/200/3): needs ~600 chars of genuine repetition before firing.
# The aggressive (300/150/2) was tuned to kill a weak model's enumeration loops
# fast at an "acceptable false-positive risk"; a strong model rarely produces
# those, so that cost (aborting legitimate reasoning that restates a constraint
# / loop invariant / repeated code pattern) now outweighs the benefit. The
# tool-call-collapse probe + the 200K char ceiling remain as fast backstops.

# --- tool-call generation-collapse detector ---------------------------------
# Qwen has been observed emitting 8000+ consecutive `<tool_call>` tokens with
# zero `</tool_call>` / `<function>` / `<parameter>`, burning 300+ s of decoder
# time before hitting max_tokens. The n-gram detector catches this eventually
# (after ~600 chars); this specialised probe fails fast after ~10 unclosed opens.
TOOL_CALL_LOOP_THRESHOLD = 10       # unclosed `<tool_call>` openings = collapse
TOOL_CALL_LOOP_PROBE_EVERY = 200    # run the probe every N chars of new content

# --- stream stop markers ----------------------------------------------------
_STREAM_STOP_MARKERS = ("</think", "<tool_call")


def _detect_thinking_loop(buf: str) -> bool:
    """True if the tail of `buf` repeats itself enough to be a loop.

    Checks two n-gram sizes (the tight 200-char window for fast repeats,
    plus a 400-char window as a backstop for slightly-paraphrased runs
    where each paragraph is just long enough to dodge the 200-char probe)."""
    if len(buf) < THINKING_LOOP_WINDOW * THINKING_LOOP_THRESHOLD:
        return False
    tail = buf[-THINKING_LOOP_WINDOW:]
    if buf.count(tail) >= THINKING_LOOP_THRESHOLD:
        return True
    wide_window = THINKING_LOOP_WINDOW * 2
    if len(buf) >= wide_window * THINKING_LOOP_THRESHOLD:
        wide_tail = buf[-wide_window:]
        if buf.count(wide_tail) >= THINKING_LOOP_THRESHOLD:
            return True
    return False


def _tail_has_stop_marker(buf: str, new_token: str) -> bool:
    """True if ``</think`` or ``<tool_call`` appears in the RECENT tail of buf.

    The streaming display latches ``stop_printing`` the first time either
    marker appears, then never checks again — but the old check lowercased the
    ENTIRE accumulated buffer on every chunk (O(n) per token → O(n²) over a
    long thinking stream, plus GB of transient string allocation on the event
    loop). A marker can straddle at most one chunk boundary, so scanning a tail
    of ``len(new_token) + 16`` chars is sufficient and O(1) per token."""
    window = len(new_token) + 16
    tail = buf[-window:].lower()
    return any(m in tail for m in _STREAM_STOP_MARKERS)


def _detect_tool_call_loop(buf: str) -> bool:
    """True if the content buffer has accumulated too many unclosed
    `<tool_call>` openings — a decoder-collapse signature where the
    model is stuck emitting opening tags but never closing them.

    The healthy case is N opens + N closes (≥0 complete tool calls) or a
    single open waiting for its close. Anything where opens -
    closes > THRESHOLD is a run of openings with no progress, and we
    should kill the stream rather than let it run to max_tokens."""
    if not buf:
        return False
    opens = len(re.findall(r'<tool_call\b', buf, re.IGNORECASE))
    closes = len(re.findall(r'</tool_call\b', buf, re.IGNORECASE))
    return (opens - closes) > TOOL_CALL_LOOP_THRESHOLD
