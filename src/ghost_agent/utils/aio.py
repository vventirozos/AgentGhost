# src/ghost_agent/utils/aio.py
"""Cancellation-safe ``wait_for`` for the Python < 3.12 runtime (§4JS).

WHY. The 2026-09-22 deploy: ``kill -TERM`` sat 15 minutes in the lifespan
shutdown's ``await bio``. The biological watchdog had been cancelled while
an idle self-play turn was streaming an LLM response — and the cancellation
vanished: no aborted-turn record, the sim turn finished, a SECOND sim ran,
the watchdog went back to its 60 s sleep, and the process never exited
(SIGUSR2 task dump: the watchdog alive at its sleep line). The stream
reader takes every chunk through ``asyncio.wait_for(chunk_iter.__anext__(),
timeout)``, and the 3.10 stdlib ``wait_for`` has this in its cancel path::

    except exceptions.CancelledError:
        if fut.done():
            return fut.result()     # the caller's cancellation is DROPPED

— bpo-42130 / gh-86296, fixed in 3.12 by ``asyncio.timeouts``. When the
outer task is cancelled in the same loop iteration the inner future
completes, ``wait_for`` returns the result and the task carries on as if
never cancelled. A per-chunk inner future completes every few milliseconds,
so a cancel landing mid-stream is *likely* to be lost, not merely possible.
(The runtime is 3.10.21; a stdlib fix is two major versions away.)

WHAT. ``wait_for(aw, timeout)`` with ``asyncio.wait_for``'s contract —
the awaitable's result, ``asyncio.TimeoutError`` on the deadline with the
inner cancelled first — built on ``asyncio.wait``, whose cancel path has no
``fut.done()`` shortcut: a cancellation of the caller always propagates,
and the inner future is cancelled on the way out. Use it wherever the
awaited thing completes often inside a loop (an iterator step, a queue
get, a readline); a single long await keeps the stdlib form — its race
window is one loop iteration per call.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Optional

__all__ = ["wait_for"]

#: After a timeout the inner future is cancelled; wait this long for that
#: cancellation to land so the task is not left "destroyed but pending".
_INNER_CANCEL_GRACE_S = 1.0


async def wait_for(aw: Awaitable[Any], timeout: Optional[float]) -> Any:
    """``asyncio.wait_for`` that never swallows the caller's cancellation.

    Same contract: returns the awaitable's result; raises
    ``asyncio.TimeoutError`` once ``timeout`` seconds pass, after cancelling
    the inner future; ``timeout=None`` waits indefinitely. Differs only
    where the stdlib is wrong: a cancellation of the calling task raises
    ``CancelledError`` in the caller even when the inner future completed
    in the same loop iteration.
    """
    if timeout is None:
        return await aw
    fut = asyncio.ensure_future(aw)
    try:
        done, _pending = await asyncio.wait({fut}, timeout=timeout)
    except asyncio.CancelledError:
        fut.cancel()
        raise
    if fut in done:
        return fut.result()
    fut.cancel()
    # Let the inner cancellation land. `asyncio.wait` again, not `await
    # fut`: awaiting the cancelled future would raise ITS CancelledError,
    # indistinguishable from ours; and a cancellation of the caller that
    # arrives here still propagates out of the wait.
    await asyncio.wait({fut}, timeout=_INNER_CANCEL_GRACE_S)
    raise asyncio.TimeoutError()
