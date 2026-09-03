"""§4EC — module-level stream-scrub helpers: survivors of the §R re-verification
of §4BZ (2026-09-02). These helpers are pure functions, driven directly."""
from ghost_agent.core.agent import _emit_safe_end


class TestEmitSafeEndBacktickMention:
    """`_emit_safe_end(view, emitted)` holds back a trailing tag fragment
    (`<tool…` with no `>` yet) UNLESS the `<` is a backtick-quoted mention
    (`` `<tool_call>` `` in prose). Mutants: `and`→`or` on the mention test
    (a mention is held like a real tag: the client stalls until the next
    `>`), and `p > 0`→`p <= 0` (the mention check reads the wrong neighbour).
    The differential fuzzer's alphabet never composed backtick+`<`+no-`>`."""

    def test_a_backtick_quoted_fragment_is_not_held(self):
        view = "Use `<tool"
        assert _emit_safe_end(view, 0) == len(view)

    def test_a_bare_fragment_is_held(self):
        view = "Use <tool"
        assert _emit_safe_end(view, 0) == view.index("<")

    def test_the_backtick_must_be_the_immediate_neighbour(self):
        view = "Use ` <tool"          # a space between: not a code-span mention
        assert _emit_safe_end(view, 0) == view.index("<")

    def test_fragment_at_position_zero_is_held(self):
        assert _emit_safe_end("<tool", 0) == 0

    def test_fragment_at_position_zero_is_held_even_if_the_view_ends_in_a_backtick(self):
        # `p <= 0 and view[p-1]` would read view[-1] (the trailing backtick) and release
        assert _emit_safe_end("<tool`", 0) == 0

    def test_fragment_at_position_one_after_a_backtick_is_released(self):
        view = "`<tool"
        assert _emit_safe_end(view, 0) == len(view)


# ── the streamed final generation, driven (§4BZ B-1/B-2/B-4/B-5 wiring) ─────
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.agent import StreamState
from tests.test_finalize_stream_pins import make_stream_agent, sse, _make_stream_state
from tests.test_finalize_stream_r1_fixes import _drive, _client_text


def _durable(a):
    return a._record_calibration_safe.call_args.kwargs


class TestStreamWiring:
    _ROW = "ROW 0 0 0 0 0 0 0 0\n"   # 20-char periodic unit (the R1 fixture)

    @pytest.mark.asyncio
    async def test_watchdog_break_lands_in_the_durable_text_not_the_client(self):
        """L25115 `full_content += break_text`: the synthetic break must be in the
        DURABLE content (the next turn replans on it) while the client gets only
        the clean interrupt notice."""
        a = make_stream_agent(); a._record_calibration_safe = AsyncMock()
        chunks = await _drive(a, ["<think>\nlooping now:\n"] + [self._ROW] * 60)
        assert "SYSTEM OVERRIDE" not in _client_text(chunks)
        assert "SYSTEM OVERRIDE" in _durable(a)["final_ai_content"]

    @pytest.mark.asyncio
    async def test_a_clean_stream_is_neither_truncated_nor_marked(self):
        """L25160/25164 `_cancel_cut and full_content` → `or`: every completed
        stream would be stamped truncated with a cancel note."""
        a = make_stream_agent(); a._record_calibration_safe = AsyncMock()
        await _drive(a, ["A complete ", "answer."])
        kw = _durable(a)
        assert kw["truncated"] is False and "TRUNCATED" not in kw["final_ai_content"]

    @pytest.mark.asyncio
    async def test_an_upstream_error_chunk_marks_the_reply_as_aborted(self):
        """L24911-24918 + L25164: an `{"error": …}` SSE chunk aborts the stream
        and the durable text carries the upstream-abort marker."""
        a = make_stream_agent(); a._record_calibration_safe = AsyncMock()

        async def final_stream(payload, use_coding=False):
            yield sse({"content": "partial answer "})
            yield b'data: {"error": {"message": "upstream exploded"}}\n\n'
            yield b"data: [DONE]\n\n"
        a.context.llm_client.stream_chat_completion = final_stream
        reg = MagicMock(); reg.is_cancelled.return_value = False
        gen, _, _ = a._stream_final_generation(_make_stream_state(reg))
        got = [c async for c in gen]
        kw = _durable(a)
        assert kw["truncated"] is True
        assert "upstream aborted" in kw["final_ai_content"]
        assert "partial answer" in kw["final_ai_content"]

    @pytest.mark.asyncio
    async def test_a_short_think_loop_is_not_severed(self):
        """L25082 `len(tail) == 400`: the guard reads a FULL 400-char tail; five
        repeats inside a shorter reasoning block are not enough evidence.
        16 rows = 328 chars: `tail.count(last_60)` IS 5 here, so only the
        400-char gate keeps the stream alive (R2 reviewer: 8 rows never reached
        the count and the gate mutant survived)."""
        a = make_stream_agent(); a._record_calibration_safe = AsyncMock()
        deltas = ["<think>\n"] + [self._ROW] * 16 + ["</think>\nDone."]   # 328 chars
        chunks = await _drive(a, deltas)
        assert "Done." in _client_text(chunks)
        assert "SYSTEM OVERRIDE" not in _durable(a)["final_ai_content"]

    @pytest.mark.asyncio
    async def test_a_stream_prefix_is_emitted_exactly_once(self):
        """L24777-24783 + L24843-24845: the prefix goes out as the first chunk
        and seeds the scrub view / emitted length, so the scrub never re-emits it."""
        a = make_stream_agent(); a._record_calibration_safe = AsyncMock()

        async def final_stream(payload, use_coding=False):
            yield sse({"content": "body text"})
            yield b"data: [DONE]\n\n"
        a.context.llm_client.stream_chat_completion = final_stream
        reg = MagicMock(); reg.is_cancelled.return_value = False
        ss = _make_stream_state(reg)
        ss = StreamState(**{**ss.__dict__, "stream_prefix": "PREFIX> "})
        gen, _, _ = a._stream_final_generation(ss)
        text = _client_text([c async for c in gen])
        assert text.count("PREFIX>") == 1 and text.endswith("body text"), text
        assert _durable(a)["final_ai_content"].startswith("PREFIX> ")


@pytest.mark.asyncio
async def test_the_drain_unregisters_the_turn_exactly_once():
    """L25974 `_turn_reg.unregister(req_id, _active_turn)` deleted: the turn
    would stay registered after a streamed reply — the global turn lock is
    never released, and the Stop button has nothing to cancel."""
    a = make_stream_agent(); a._record_calibration_safe = AsyncMock()

    async def final_stream(payload, use_coding=False):
        yield sse({"content": "done"})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock(); reg.is_cancelled.return_value = False
    gen, _, _ = a._stream_final_generation(_make_stream_state(reg))
    [c async for c in gen]
    assert reg.unregister.call_count == 1
