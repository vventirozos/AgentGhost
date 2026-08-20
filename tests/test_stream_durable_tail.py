"""A raise mid-final-stream must not skip the durable tail.

§4BV R7 lens B, fixed 2026-08-19. The final-generation drain in
`stream_wrapper` iterated `stream_chat_completion` with no enclosing handler
(AST-proven), so any raise — `_do_stream_chat_completion` still re-raises on
HTTP 4xx/5xx and on its generic path — skipped episode, hydration judge,
trajectory, work_log, verifier spawn and lesson outcomes. The client saw
routes.py's error event; the turn vanished from every persistent record.

The fix converts a raise into the two frames the loop body already handles:
an `{"error": …}` frame (parsed into `stream_aborted` + the truncation
marker on the durable content) and a `data: [DONE]` (captured and released
AFTER the tail, per §4AT-A). It was deliberately NOT fixed inside llm.py:
`handle_chat`'s internal drain is wrapped in a try whose handlers depend on
the raise — one runs the emergency context prune on a 400 mentioning
"context" — so the conversion must happen at exactly one call site.
"""

import ast
import asyncio
import inspect
import json

import pytest

from ghost_agent.core.agent import _stream_or_abort_frames


async def _gen(chunks, exc=None):
    for c in chunks:
        yield c
    if exc is not None:
        raise exc


def _drain(it):
    out = []

    async def go():
        async for c in it:
            out.append(c)

    asyncio.run(go())
    return out


def test_a_clean_stream_passes_through_byte_identical():
    chunks = [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
              b"data: [DONE]\n\n"]
    assert _drain(_stream_or_abort_frames(_gen(chunks))) == chunks


def test_a_raise_becomes_the_abort_frame_plus_DONE():
    """The conversion. The loop body downstream parses the error frame into
    `stream_aborted` and captures the [DONE] to release after the tail —
    so these two frames are what make the durable tail run."""
    chunks = [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n']
    out = _drain(_stream_or_abort_frames(
        _gen(chunks, exc=RuntimeError("upstream fell over"))))
    assert out[0] == chunks[0], "the partial content must still be delivered"
    frame = out[1].decode()
    assert frame.startswith("data: ")
    body = json.loads(frame[6:].strip())
    assert "error" in body and "choices" not in body, (
        f"the conversion frame {body} is not the shape the loop's abort "
        f"handler parses — stream_aborted will never be set")
    assert "RuntimeError" in body["error"]
    assert out[-1] == b"data: [DONE]\n\n", (
        "no [DONE] emitted — the §4AT-A held-sentinel release has nothing "
        "to hold, and a client waiting on the end-of-stream marker hangs")


def test_cancellation_still_propagates():
    """A cancelled turn must still cancel — CancelledError is BaseException
    and the guard must not convert it into a tidy abort frame."""
    with pytest.raises(asyncio.CancelledError):
        _drain(_stream_or_abort_frames(
            _gen([b"data: x\n\n"], exc=asyncio.CancelledError())))


def test_exactly_one_call_site_is_guarded_and_it_is_the_right_one():
    """⚠ BOTH DIRECTIONS. The final-generation drain must go through the
    guard (or the durable tail is skippable again); `handle_chat`'s internal
    drain must NOT (its enclosing try's handlers depend on the raise — one
    runs the emergency context prune on 400+"context", and wrapping it
    silently deletes that recovery)."""
    from ghost_agent.core import agent as agent_mod

    tree = ast.parse(inspect.getsource(agent_mod))
    guarded, bare = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFor):
            continue
        src = ast.unparse(node.iter)
        if "stream_chat_completion" not in src:
            continue
        (guarded if "_stream_or_abort_frames" in src else bare).append(node)

    assert len(guarded) == 1, (
        f"{len(guarded)} guarded stream drains — the final-generation site "
        f"must be exactly one, or a raise skips the durable tail again")
    assert len(bare) == 1, (
        f"{len(bare)} unguarded stream drains — handle_chat's drain must "
        f"stay bare: its exception handlers run the emergency context prune")

    # the bare one must genuinely sit inside a try with handlers
    class _Finder(ast.NodeVisitor):
        def __init__(self, target):
            self.target, self.stack, self.hit = target, [], None

        def generic_visit(self, node):
            self.stack.append(node)
            if node is self.target:
                self.hit = [n for n in self.stack if isinstance(n, ast.Try)]
            super().generic_visit(node)
            self.stack.pop()

    f = _Finder(bare[0])
    f.visit(tree)
    assert f.hit, (
        "handle_chat's stream drain is no longer inside a try — its raise "
        "now skips ITS tail too; if this was deliberate, wrap it in the "
        "guard instead of leaving it bare")
