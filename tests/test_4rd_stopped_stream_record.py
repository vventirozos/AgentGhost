"""§4RD — a streamed final closed before its tail still books the turn.

Req e2c7c7f4 (2026-10-24): 46 minutes of a WebOS build, then Stop during the
deadline report. The web proxy cancels its upstream reader FIRST, so the
agent's drain was closed at a `yield` (GeneratorExit) and everything after
it — the "final reply" line, the trajectory row, the turn outcome — never
ran; only the finally's stop-state save did. No ledger knew the turn existed.

World where each test fails: the finally stops booking a closed drain (no
row), or books one ALSO when the tail already recorded (two rows).
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import agent as agmod
from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.turns import get_turn_registry
from tests.helpers import force_final_generation
from tests.test_streaming_tail_cancellable import _ctx, _non_stream_reply, _BODY


@pytest.fixture(autouse=True)
def _ff(monkeypatch):
    force_final_generation(monkeypatch)


seen = []


def _agent(stream_fn, rows, monkeypatch):
    ctx = _ctx()
    ctx.llm_client = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_non_stream_reply)
    ctx.llm_client.stream_chat_completion = stream_fn
    a = GhostAgent(ctx)

    def spy(*args, **kw):
        from ghost_agent.utils.logging import request_id_context
        rows.append(kw.get("final_content"))
        seen.append({"rid": request_id_context.get(), "elapsed": kw.get("elapsed_s"),
                     "project_id": kw.get("project_id"), "lock": kw.get("pressure_lockdown"),
                     "tid": kw.get("trajectory_id")})
        return None
    a._record_turn_trajectory = spy
    monkeypatch.setattr(agmod, "_persist_stream_stop_state", lambda *aa, **kk: True)
    return a


PARTS = ["Here is where things stand: the shell, dock and window manager ",
         "are written; apps.js still fails on ", "the calculator's $ helper. ",
         "The files are in webos/."]


def _fake(gate):
    async def fake_stream(*a, **kw):
        for i, t in enumerate(PARTS):
            if i == 2:
                await gate.wait()
            yield f'data: {json.dumps({"choices": [{"delta": {"content": t}}]})}\n\n'.encode()
        yield b"data: [DONE]\n\n"
    return fake_stream


async def _run(monkeypatch, *, stop: bool, keep_reading: bool, foreign_close: bool = False):
    rows, gate = [], asyncio.Event()
    a = _agent(_fake(gate), rows, monkeypatch)
    reg = get_turn_registry(a)
    gen, _, rid = await a.handle_chat(dict(_BODY), MagicMock(), request_id="probe-4rd-stop")
    got = []
    async for c in gen:
        got.append(c)
        if len(got) == 1:
            if stop:
                reg.cancel(rid)        # web Stop → /api/turn/cancel (cooperative)
            gate.set()
        elif len(got) >= 2 and not keep_reading:
            break                      # the proxy hung up: the next send fails
    if foreign_close:
        # the close runs in whatever context does it (a GC finalizer task):
        # another request's id is live there
        from ghost_agent.utils.logging import request_id_context
        request_id_context.set("someone-else")
    await gen.aclose()
    return rows


@pytest.mark.asyncio
async def test_stop_then_the_proxy_hangs_up_books_one_aborted_row(monkeypatch):
    rows = await _run(monkeypatch, stop=True, keep_reading=False)
    assert len(rows) == 1, rows
    assert "[ATTEMPT_ABORTED_TURN]" in rows[0]
    assert "stopped while it streamed" in rows[0]


@pytest.mark.asyncio
async def test_a_bare_disconnect_books_one_aborted_row(monkeypatch):
    # r1 M1: cut from outside is an INTERRUPTION (§4MS), not a failed attempt
    rows = await _run(monkeypatch, stop=False, keep_reading=False)
    assert len(rows) == 1, rows
    assert "[TURN_INTERRUPTED]" in rows[0] and "[ATTEMPT_ABORTED_TURN]" not in rows[0]
    assert "client closed the stream" in rows[0]
    assert "Here is where things stand" in rows[0]      # the text that did stream


@pytest.mark.asyncio
async def test_stop_with_a_reader_that_keeps_reading_books_once(monkeypatch):
    # the tail runs (truncation marker), the finally must not add a second row
    rows = await _run(monkeypatch, stop=True, keep_reading=True)
    assert len(rows) == 1, rows
    assert "[ATTEMPT_ABORTED_TURN]" not in rows[0]
    assert "cancelled by the user mid-answer" in rows[0]


@pytest.mark.asyncio
async def test_a_full_drain_books_once_and_not_as_aborted(monkeypatch):
    rows = await _run(monkeypatch, stop=False, keep_reading=True)
    assert len(rows) == 1, rows
    assert "[ATTEMPT_ABORTED_TURN]" not in rows[0]


@pytest.mark.asyncio
async def test_the_closed_drain_records_with_the_requests_snapshots(monkeypatch):
    # r1 M2: the finally runs outside the request's context; the row must
    # still carry the request id, a real duration and the drain's project
    seen.clear()
    await _run(monkeypatch, stop=False, keep_reading=False, foreign_close=True)
    assert len(seen) == 1, seen
    assert seen[0]["rid"] == "probe-4rd-stop"
    assert seen[0]["elapsed"] is not None and seen[0]["elapsed"] >= 0.0
    assert seen[0]["project_id"] is not None
    assert seen[0]["lock"] is not None


@pytest.mark.asyncio
async def test_a_hang_up_at_the_deadline_is_the_agents_failure(monkeypatch):
    # r2 M: the drain runs after the request clock closed — the deadline is
    # read while the request is open, so a cut at it stays a FAILED attempt (§4MS r1)
    from ghost_agent.utils import logging as glog
    monkeypatch.setattr(glog, "request_deadline_s", lambda *a, **k: 0.001)
    rows = await _run(monkeypatch, stop=False, keep_reading=False)
    assert len(rows) == 1, rows
    assert "[ATTEMPT_ABORTED_TURN]" in rows[0] and "[TURN_INTERRUPTED]" not in rows[0]


@pytest.mark.asyncio
async def test_the_aborted_row_keeps_the_turns_trajectory_id(monkeypatch):
    seen.clear()
    await _run(monkeypatch, stop=False, keep_reading=False)
    assert seen and seen[0]["tid"], seen
