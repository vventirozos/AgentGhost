"""Streamed build-spec generation with mid-stream loop abort (2026-07-25).

Regression target (req 92a968fc): the spec call was non-streaming, so a
thinking loop could not be detected until the full 16384-token budget was
burned (~4 minutes on the live box, twice in one request) — the existing
think-hog guard then fired on the corpse. The spec call now streams via
``llm.stream_chat_completion`` and, while content is still empty, runs the
exact-n-gram detector (`_detect_thinking_loop`) plus a 30K reasoning
ceiling; an abort falls straight into the existing one-shot no-think
retry. The paragraph-repeat detector is deliberately NOT on this path —
its first deploy aborted the thinking of every live coding leaf
(2026-07-25) because spec planning legitimately restates the task.

Clients without ``stream_chat_completion`` (older doubles, alternate
backends) degrade to the single-shot ``chat_completion`` — covered by the
whole pre-existing test_coding_executor.py suite, whose fakes are exactly
that shape.
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.coding_executor import (
    _stream_spec_completion,
    build_coding_task,
)

SPEC_OK = json.dumps({
    "files": [{"path": "parser.py", "content": "def parse(p):\n    return []\n"}],
    "verify": "",
    "summary": "wrote parser.py",
    "ledger": "",
})

LOOP_PARA = ("Let me write this now. I'll keep the same general structure "
             "but update the session-related parts.")


def _sse(delta: dict) -> bytes:
    return ("data: " + json.dumps(
        {"choices": [{"delta": delta}]})).encode("utf-8")


def _loop_chunks(n=120):
    """A reasoning stream stuck in VERBATIM repetition (the exact-n-gram
    shape) that would keep going for a long time — the guard must not
    consume it all. Varied-filler paraphrase loops are deliberately NOT
    aborted on this path any more: the paragraph detector false-fired on
    every live coding leaf (2026-07-25 second deploy) and was removed."""
    out = [_sse({"reasoning_content": LOOP_PARA + "\n"})
           for _ in range(n)]
    out.append(b"data: [DONE]")
    return out


def _diverse_chunks(total_chars, chunk=500):
    """Non-repeating reasoning of ~total_chars, no content."""
    out, i, made = [], 0, 0
    while made < total_chars:
        text = f"[{i}] distinct planning thought number {i} " * 8 + "\n"
        out.append(_sse({"reasoning_content": text}))
        made += len(text)
        i += 1
    out.append(b"data: [DONE]")
    return out


class StreamingFakeLLM:
    """Yields canned SSE chunks; records how many were consumed and any
    non-streaming retry calls."""

    def __init__(self, chunks, retry_content=SPEC_OK):
        self.chunks = chunks
        self.consumed = 0
        self.stream_calls = 0
        self.chat_calls = 0
        self.chat_payloads = []
        self.retry_content = retry_content

    async def stream_chat_completion(self, payload, is_background=False):
        self.stream_calls += 1
        for c in self.chunks:
            self.consumed += 1
            yield c

    async def chat_completion(self, payload, is_background=False, **_kw):
        self.chat_calls += 1
        self.chat_payloads.append(payload)
        return {"choices": [{"message": {
            "content": self.retry_content, "reasoning_content": ""}}]}


def _ctx(llm):
    return SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m"))


class FakeRunner:
    def __init__(self):
        self.calls = []

    async def __call__(self, name, args):
        self.calls.append((name, args))
        return "Successfully wrote file." if name == "file_system" else "OK"


# ---------------------------------------------------------- accumulator unit


@pytest.mark.asyncio
async def test_stream_accumulates_content_and_reasoning():
    llm = StreamingFakeLLM([
        _sse({"reasoning_content": "thinking about it...\n"}),
        _sse({"content": SPEC_OK[:20]}),
        _sse({"content": SPEC_OK[20:]}),
        b"data: [DONE]",
    ])
    content, reasoning, aborted = await _stream_spec_completion(
        llm, {"model": "m", "messages": []}, False)
    assert content == SPEC_OK
    assert "thinking" in reasoning
    assert aborted is None


@pytest.mark.asyncio
async def test_loop_stream_aborts_early():
    chunks = _loop_chunks(n=120)
    llm = StreamingFakeLLM(chunks)
    content, reasoning, aborted = await _stream_spec_completion(
        llm, {"model": "m", "messages": []}, False)
    assert aborted == "loop"
    assert content == ""
    # The guard consumed only a fraction of the runaway stream — the whole
    # point: not paying for the full generation.
    assert llm.consumed < len(chunks) // 2


@pytest.mark.asyncio
async def test_loop_probe_never_fires_once_content_started():
    # Once content bytes exist the probes stop entirely: generated
    # code/data repeats lines legitimately and the spec is on its way —
    # even a blatant verbatim run after the JSON must not abort.
    chunks = [_sse({"reasoning_content": f"distinct thought {i}.\n"})
              for i in range(3)]
    chunks += [_sse({"content": SPEC_OK})]
    chunks += [_sse({"reasoning_content": LOOP_PARA + "\n"})] * 60
    chunks += [b"data: [DONE]"]
    llm = StreamingFakeLLM(chunks)
    content, _, aborted = await _stream_spec_completion(
        llm, {"model": "m", "messages": []}, False)
    assert aborted is None
    assert content == SPEC_OK


@pytest.mark.asyncio
async def test_reasoning_ceiling_aborts_non_looping_runaway():
    # Diverse (loop-free) reasoning that just keeps going: the exact
    # n-gram probe never fires, the 30K ceiling does — this is the
    # budget-burn shape (40-75K chars observed live).
    from ghost_agent.core.coding_executor import SPEC_REASONING_ABORT_CHARS
    chunks = _diverse_chunks(SPEC_REASONING_ABORT_CHARS + 15_000)
    llm = StreamingFakeLLM(chunks)
    content, reasoning, aborted = await _stream_spec_completion(
        llm, {"model": "m", "messages": []}, False)
    assert aborted == "ceiling"
    assert content == ""
    assert len(reasoning) < SPEC_REASONING_ABORT_CHARS + 2_000
    assert llm.consumed < len(chunks)


@pytest.mark.asyncio
async def test_normal_sized_planning_is_never_aborted():
    # ~12K of diverse planning then a clean spec — the live false-positive
    # regression shape (first deploy aborted EVERY leaf): must run clean.
    from ghost_agent.core.coding_executor import SPEC_REASONING_ABORT_CHARS
    chunks = _diverse_chunks(12_000)[:-1]          # drop [DONE]
    chunks += [_sse({"content": SPEC_OK}), b"data: [DONE]"]
    llm = StreamingFakeLLM(chunks)
    content, _, aborted = await _stream_spec_completion(
        llm, {"model": "m", "messages": []}, False)
    assert aborted is None
    assert content == SPEC_OK


@pytest.mark.asyncio
async def test_error_event_stops_cleanly():
    llm = StreamingFakeLLM([
        _sse({"reasoning_content": "partial think"}),
        b'data: {"error": "Upstream stalled (mid-stream)."}',
        b"data: [DONE]",
    ])
    content, reasoning, aborted = await _stream_spec_completion(
        llm, {"model": "m", "messages": []}, False)
    assert content == "" and "partial think" in reasoning
    assert aborted is None


@pytest.mark.asyncio
async def test_nonstreaming_client_falls_back_to_chat_completion():
    class PlainLLM:
        def __init__(self):
            self.calls = 0

        async def chat_completion(self, payload, is_background=False, **_kw):
            self.calls += 1
            assert payload.get("stream") is False
            return {"choices": [{"message": {
                "content": "C", "reasoning_content": "R"}}]}

    llm = PlainLLM()
    content, reasoning, aborted = await _stream_spec_completion(
        llm, {"model": "m", "messages": []}, False)
    assert (content, reasoning, aborted) == ("C", "R", None)
    assert llm.calls == 1


# ------------------------------------------------------- end-to-end behaviour


@pytest.mark.asyncio
async def test_loop_abort_falls_into_nothink_retry_and_builds():
    llm = StreamingFakeLLM(_loop_chunks(n=120))
    runner = FakeRunner()
    res = await build_coding_task(_ctx(llm), "build the parser",
                                  tool_runner=runner)
    assert res.ok and res.files == ["parser.py"]
    # Stream aborted → exactly one non-streaming no-think retry.
    assert llm.chat_calls == 1
    retry = llm.chat_payloads[0]
    assert retry.get("chat_template_kwargs") == {"enable_thinking": False}
    assert retry["messages"][-1]["content"].rstrip().endswith("/no_think")


@pytest.mark.asyncio
async def test_clean_stream_never_calls_nothink_retry():
    llm = StreamingFakeLLM([
        _sse({"reasoning_content": "planning briefly.\n"}),
        _sse({"content": SPEC_OK}),
        b"data: [DONE]",
    ])
    runner = FakeRunner()
    res = await build_coding_task(_ctx(llm), "build the parser",
                                  tool_runner=runner)
    assert res.ok
    assert llm.stream_calls == 1
    assert llm.chat_calls == 0
