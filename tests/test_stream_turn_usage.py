"""The streamed reply carries the turn's token totals (2026-10-27).

One ``ghost.usage`` frame, just before the stream's first ``[DONE]``, for
the CLI's token counter. Owner turns only; absent when the totals are
unknown (an absent frame is honest, a zero is not)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ghost_agent.api.routes import _turn_usage_frame, _with_turn_usage  # noqa: E402
from tests.test_feedback_stream_id_restamp import (  # noqa: E402
    LLAMA_FRAMES, REQ, _make_request, _sse_delta_text)

USAGE = {"tokens_in": 31234, "tokens_out": 840, "cached_tokens": 28000, "calls": 3}


def _agent(usage=USAGE, frames=LLAMA_FRAMES):
    async def _streamed_final():
        for f in frames:
            yield f

    async def fake_handle_chat(*args, request_id=None, **kwargs):
        # as the real one: `req_id = request_id or <new id>`
        return (_streamed_final(), 1, request_id or REQ)

    agent = MagicMock()
    agent.handle_chat = fake_handle_chat
    agent.context.args.model = "test-model"
    seen = []
    agent.context.llm_client.usage_for = lambda rid: (seen.append(rid), dict(usage or {}))[1]
    return agent, seen


async def _drive(agent, headers=None):
    from ghost_agent.api.routes import chat_proxy
    req = _make_request({"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    req.headers.update(headers or {})
    req.app = MagicMock()
    req.app.state.agent = agent
    with patch("ghost_agent.api.routes.get_agent", return_value=agent):
        resp = await chat_proxy(req, MagicMock())
    out = b""
    async for chunk in resp.body_iterator:
        out += chunk if isinstance(chunk, (bytes, bytearray)) else str(chunk).encode()
    return out


def _frames(body: bytes):
    return [f.strip() for f in body.decode().split("\n\n") if f.strip()]


def _usage_frames(body: bytes):
    out = []
    for f in _frames(body):
        if f.startswith("data: {"):
            d = json.loads(f[6:])
            if "usage" in (d.get("ghost") or {}):
                out.append(d)
    return out


class TestRoute:

    @pytest.mark.asyncio
    async def test_one_usage_frame_right_before_done(self):
        agent, seen = _agent()
        body = await _drive(agent)
        frames = _frames(body)
        assert frames[-1] == "data: [DONE]"
        last = json.loads(frames[-2][6:])
        assert last["ghost"]["usage"] == {"prompt_tokens": 31234, "completion_tokens": 840,
                                          "cached_tokens": 28000, "calls": 3}
        assert last["choices"] == [] and last["id"].startswith("chatcmpl-")
        assert len(_usage_frames(body)) == 1
        assert _sse_delta_text(body) == "Hey Vasilis!"      # the reply text is untouched
        assert seen == [last["id"][len("chatcmpl-"):]]   # the turn's own id

    @pytest.mark.asyncio
    async def test_without_a_header_or_tap_the_id_is_read_off_the_frames(self):
        """`X-Ghost-Stream: final` (piped CLI): no tap, so the route mints
        no id and the agent's own is read off the restamped frames."""
        agent, seen = _agent()
        await _drive(agent, {"X-Ghost-Stream": "final"})
        assert seen == [REQ]

    @pytest.mark.asyncio
    async def test_the_header_request_id_is_used(self):
        agent, seen = _agent()
        await _drive(agent, {"X-Request-ID": "abc12345"})
        assert seen == ["abc12345"]

    @pytest.mark.asyncio
    async def test_unknown_usage_sends_no_frame(self):
        agent, _ = _agent(usage={})
        body = await _drive(agent)
        assert _usage_frames(body) == []
        assert body.rstrip().endswith(b"data: [DONE]")

    @pytest.mark.asyncio
    async def test_a_magicmock_client_sends_no_frame(self):
        agent, _ = _agent()
        del agent.context.llm_client.usage_for          # back to a MagicMock attribute
        assert _usage_frames(await _drive(agent)) == []

    @pytest.mark.asyncio
    async def test_a_member_turn_gets_no_frame(self):
        agent, seen = _agent()
        body = await _drive(agent, {"X-Ghost-Requester": "member"})
        assert _usage_frames(body) == [] and seen == []


class TestWrapper:

    @staticmethod
    async def _run(chunks, usage=USAGE, request_id="r1"):
        agent, _ = _agent(usage=usage)

        async def gen():
            for c in chunks:
                yield c
        return [c async for c in _with_turn_usage(gen(), agent, request_id)]

    @pytest.mark.asyncio
    async def test_done_inside_a_multi_frame_chunk_is_split(self):
        out = await self._run([b'data: {"a":1}\n\ndata: [DONE]\n\n'])
        assert out[0] == b'data: {"a":1}\n\n'
        assert b'"usage"' in out[1]
        assert out[2] == b"data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_only_the_first_done_gets_a_frame(self):
        out = await self._run([b"data: [DONE]\n\n", b"data: [DONE]\n\n"])
        assert sum(b'"usage"' in c for c in out) == 1 and len(out) == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cut", range(0, 16))
    async def test_a_reply_that_says_data_done_keeps_its_text(self, cut):
        """Review r1 M1: the sentinel's TEXT inside a content delta (a reply
        about SSE) matched a substring search — the frame was split, its
        text lost, and the real [DONE] got no usage frame. Every 15-char
        slice offset of the tap's real framing."""
        from ghost_agent.core.reply_tap import ReplyTap
        reply = "x" * cut + "Streams end at:\ndata: [DONE]\n\nThat is the sentinel."
        agent, _ = _agent()

        async def body():
            yield b": processing request...\n\n"
            for f in ReplyTap("abc12345", "m", 1).finish(reply):
                yield f
        out = b"".join([c async for c in _with_turn_usage(body(), agent, "abc12345")])
        text, usage = "", None
        for line in out.decode().split("\n"):
            if not line.startswith("data: "):
                continue
            if line[6:] == "[DONE]":
                break
            o = json.loads(line[6:])                 # every line parses
            usage = (o.get("ghost") or {}).get("usage") or usage
            text += ((o.get("choices") or [{}])[0].get("delta", {}).get("content", "")) if o.get("choices") else ""
        assert text == reply and usage is not None
        assert out.rstrip().endswith(b"data: [DONE]")

    @pytest.mark.asyncio
    async def test_a_uniquified_id_is_the_one_read(self):
        """Review r1 m2: a colliding header id is filed as `id#2`; usage is
        booked under that, and the frame must carry it (ClockworkPi reads
        the request id off the LAST frame)."""
        seen = []
        agent, _ = _agent()
        agent.context.llm_client.usage_for = lambda rid: (seen.append(rid), dict(USAGE))[1]

        async def body():
            # the tap's live frames carry the header id until `bind` (review r2)
            yield b'data: {"id": "chatcmpl-abc12345", "choices": []}\n\n'
            yield b'data: {"id": "chatcmpl-abc12345#2", "choices": []}\n\n'
            yield b"data: [DONE]\n\n"
        out = [c async for c in _with_turn_usage(body(), agent, "abc12345")]
        assert seen == ["abc12345#2"]
        assert b'"chatcmpl-abc12345#2"' in out[2]

    @pytest.mark.asyncio
    async def test_unmetered_calls_are_passed_on(self):
        out = await self._run([b"data: [DONE]\n\n"], usage=dict(
            USAGE, unmetered_calls=1, tokens_out_estimated=12))
        u = json.loads(out[0].decode()[6:])["ghost"]["usage"]
        assert u["unmetered_calls"] == 1 and u["tokens_out_estimated"] == 12

    @pytest.mark.asyncio
    async def test_closing_the_wrapper_closes_the_body(self):
        """An explicit close reaches the body's own finally. (Starlette does
        not close on a disconnect — review r1 m4 — so this is the explicit-
        consumer contract, not a disconnect guarantee.)"""
        closed = []

        async def body():
            try:
                yield b"data: {}\n\n"
                yield b"data: {}\n\n"
            finally:
                closed.append(1)
        agent, _ = _agent()
        w = _with_turn_usage(body(), agent, "r1")
        await w.__anext__()
        await w.aclose()
        assert closed == [1]

    def test_frame_is_none_without_a_request_id(self):
        agent, _ = _agent()
        assert _turn_usage_frame(agent, "") is None
