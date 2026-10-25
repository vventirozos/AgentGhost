"""§4RD — a Stop on the streamed final: the agent first, and nothing is lost.

Req e2c7c7f4 (2026-10-24): the web proxy tore its relay down BEFORE it told
the agent to cancel, so the agent's drain was closed at a yield — no record,
no outcome, and the session history kept nothing of the 158 s report.

World where each test fails: the proxy cuts the relay before the agent's
cancel again (order test), stops waiting for a cancelled stream to end
(grace test), or the route drops the streamed text of a reply the client
closed (session test).
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _task(**kw):
    t = {"done": False, "cancelled": False, "new_data_event": asyncio.Event(),
         "buffer": [], "buffer_size": 0}
    t.update(kw)
    return t


@pytest.mark.asyncio
async def test_the_agent_is_cancelled_before_the_relay_is_cut(monkeypatch):
    monkeypatch.setenv("GHOST_API_KEY", "x")
    from interface import server
    order = []

    async def relay():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            order.append("relay cut")
            raise

    bg = asyncio.ensure_future(relay())
    await asyncio.sleep(0)

    async def post(*a, **k):
        order.append("agent cancel")
        return MagicMock(status_code=200, json=lambda: {"cancelled": True})
    monkeypatch.setattr(server, "STOP_DRAIN_GRACE_S", 0.05)
    server.active_chat_tasks["4rd-order"] = _task(request_id="abcd1234", background_task=bg)
    try:
        with patch.object(server, "_get_http_client", return_value=MagicMock(post=post)):
            out = await server.chat_cancel_proxy("4rd-order")
        await asyncio.sleep(0)
        assert order == ["agent cancel", "relay cut"], order
        assert out["agent"]["cancelled"] is True
        assert server.active_chat_tasks["4rd-order"]["cancelled"] is True
    finally:
        server.active_chat_tasks.pop("4rd-order", None)
        bg.cancel()


@pytest.mark.asyncio
async def test_a_stream_that_ends_inside_the_grace_is_not_cut(monkeypatch):
    monkeypatch.setenv("GHOST_API_KEY", "x")
    from interface import server
    ended = {"cut": False}

    async def relay():
        try:
            await asyncio.sleep(0.05)          # the agent's stream ends on its own
        except asyncio.CancelledError:
            ended["cut"] = True
            raise

    bg = asyncio.ensure_future(relay())

    async def post(*a, **k):
        return MagicMock(status_code=200, json=lambda: {"cancelled": True})
    server.active_chat_tasks["4rd-grace"] = _task(request_id="abcd1234", background_task=bg)
    try:
        with patch.object(server, "_get_http_client", return_value=MagicMock(post=post)):
            await server.chat_cancel_proxy("4rd-grace")
        assert bg.done() and not bg.cancelled() and not ended["cut"]
    finally:
        server.active_chat_tasks.pop("4rd-grace", None)


@pytest.mark.asyncio
async def test_a_refused_agent_cancel_cuts_the_relay_at_once(monkeypatch):
    monkeypatch.setenv("GHOST_API_KEY", "x")
    from interface import server

    async def relay():
        await asyncio.sleep(30)
    bg = asyncio.ensure_future(relay())
    await asyncio.sleep(0)

    async def post(*a, **k):
        return MagicMock(status_code=404, json=lambda: {"cancelled": False})
    server.active_chat_tasks["4rd-refused"] = _task(request_id="abcd1234", background_task=bg)
    try:
        with patch.object(server, "_get_http_client", return_value=MagicMock(post=post)):
            await asyncio.wait_for(server.chat_cancel_proxy("4rd-refused"), 1.0)
        await asyncio.sleep(0)
        assert bg.cancelled()
    finally:
        server.active_chat_tasks.pop("4rd-refused", None)


@pytest.mark.asyncio
async def test_a_reply_closed_mid_stream_is_kept_in_the_session_marked_cut():
    from tests.test_feedback_stream_id_restamp import _make_request
    from ghost_agent.api.routes import chat_proxy, STREAM_CUT_NOTE

    def frame(text):
        return ("data: " + json.dumps({"id": "chatcmpl-r", "choices": [{"index": 0, "delta": {"content": text}}]})
                + "\n\n").encode()

    async def streamed():
        yield frame("Here is where things stand: ")
        yield frame("the shell and dock work.")
        await asyncio.sleep(30)                 # the rest never comes: the client goes
        yield frame(" never")

    async def fake_handle_chat(*a, **k):
        return (streamed(), 1, "r")
    agent = MagicMock()
    agent.handle_chat = fake_handle_chat
    agent.context.args.model = "m"
    store = MagicMock()
    store.get.return_value = None
    req = _make_request({"stream": True, "session_id": "s1", "messages": [{"role": "user", "content": "hi"}]})
    req.app = MagicMock(); req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent), \
            patch("ghost_agent.core.sessions.get_session_store", return_value=store):
        resp = await chat_proxy(req, MagicMock())
        it = resp.body_iterator
        got = 0
        async for chunk in it:
            if b"the shell and dock" in chunk:
                break
            got += 1
        await it.aclose()                        # the hang-up
        for _ in range(50):
            if store.append_turn.called:
                break
            await asyncio.sleep(0.02)
    assert store.append_turn.called
    stored = store.append_turn.call_args.args[2]
    assert stored.startswith("Here is where things stand: the shell and dock work.")
    assert stored.endswith(STREAM_CUT_NOTE)


@pytest.mark.asyncio
async def test_a_finished_stream_is_stored_once_without_the_cut_note():
    from tests.test_feedback_stream_id_restamp import _make_request
    from ghost_agent.api.routes import chat_proxy, STREAM_CUT_NOTE

    async def streamed():
        yield ("data: " + json.dumps({"id": "c", "choices": [{"index": 0, "delta": {"content": "Done."}}]})
               + "\n\n").encode()
        yield b"data: [DONE]\n\n"

    async def fake_handle_chat(*a, **k):
        return (streamed(), 1, "r")
    agent = MagicMock(); agent.handle_chat = fake_handle_chat; agent.context.args.model = "m"
    store = MagicMock(); store.get.return_value = None
    req = _make_request({"stream": True, "session_id": "s1", "messages": [{"role": "user", "content": "hi"}]})
    req.app = MagicMock(); req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent), \
            patch("ghost_agent.core.sessions.get_session_store", return_value=store):
        resp = await chat_proxy(req, MagicMock())
        async for _ in resp.body_iterator:
            pass
        await asyncio.sleep(0.05)
    assert store.append_turn.call_count == 1
    assert STREAM_CUT_NOTE not in store.append_turn.call_args.args[2]
