"""§ finalize/stream slice R1 (2026-08-19) — pins for the round's FIXES.

Lens A/B found nine reproduced MAJORs; these pin the fixed behaviors by
driving the real functions/generators (no source-string assertions). The
pre-existing guard pins live in test_finalize_stream_pins.py (lens C).
"""

import json

import pytest

from ghost_agent.core.agent import (
    GhostAgent, _scrub_task_status_runs, _truncate_prompt_bleed)
from ghost_agent.core import stream_guards as SG
from ghost_agent.utils.constraints import (
    enforce_start_with, parse_start_with_phrase)
from unittest.mock import AsyncMock, MagicMock

from tests.test_finalize_stream_pins import (
    _fs, _make_stream_state, make_fin_agent, make_stream_agent, sse,
    StreamScript, run_chat)


# ── A-F3: prompt-bleed strong/weak + task-status runs ────────────────────────

class TestPromptBleedTruncation:
    def test_lone_readme_tools_heading_is_kept(self):
        text = "# My Project\n\nIntro.\n\n# Tools\n\n- hammer\n\n# License\nMIT"
        assert _truncate_prompt_bleed(text) == text

    def test_lone_schema_example_is_kept(self):
        text = ('Define it like this:\n```json\n{"type": "function", '
                '"function": {"name": "x"}}\n```\nThat is all.')
        assert _truncate_prompt_bleed(text) == text

    def test_strong_marker_truncates(self):
        text = "Answer head.\n<tools>\nleaked tool list"
        assert _truncate_prompt_bleed(text) == "Answer head.\n"

    def test_two_weak_markers_corroborate(self):
        text = ('head\n# Tools\nstuff {"type": "function" more')
        assert _truncate_prompt_bleed(text) == "head\n"

    def test_strong_present_cuts_at_earliest_marker(self):
        text = ('head\n# Tools\nmid\nCRITICAL INSTRUCTION: leak')
        assert _truncate_prompt_bleed(text) == "head\n"


class TestTaskStatusRunScrub:
    def test_isolated_status_lines_are_kept(self):
        text = ("Current state:\n- Draft the intro (DONE)\n"
                "- Ship the report (PENDING)\nMore prose.")
        assert _scrub_task_status_runs(text) == text

    def test_regurgitated_tree_run_is_stripped(self):
        # R2 M-4: a stripped run must be BOTH >=3 lines AND task-shaped
        # (task_NN id / emoji / bracket lead) — the real regurgitation shape.
        tree = "\n".join(f"[task_{i}] deploy step (IN_PROGRESS)"
                         for i in range(4))
        text = f"head\n{tree}\ntail"
        assert _scrub_task_status_runs(text) == "head\ntail"

    def test_plain_status_answer_run_is_kept(self):
        # R2 M-4: a legitimate 4-item "where do the workstreams stand" answer
        # (no task ids/markers) was being deleted wholesale.
        text = ("Current standing:\n- Ship the report (DONE)\n"
                "- Fix the parser (IN_PROGRESS)\n- Update docs (PENDING)\n"
                "- Deploy to nova (BLOCKED)\nMore prose.")
        assert _scrub_task_status_runs(text) == text

    def test_two_line_run_is_kept(self):
        text = "a (DONE)\nb (FAILED)\nrest"
        assert _scrub_task_status_runs(text) == text


# ── A-F5: start-with phrase parsing ──────────────────────────────────────────

class TestStartWithPhraseParsing:
    def test_closed_quote_with_trailing_clause(self):
        # the phrasing class the hoist was built for — used to yield the
        # whole tail and render the hoist inert.
        phrase = parse_start_with_phrase(
            ['Start your reply with "BLUF:" and then summarise the incident'])
        assert phrase == "BLUF:"

    def test_plain_quoted_phrase_still_parses(self):
        assert parse_start_with_phrase(
            ['Begin your response with "BLUF:"']) == "BLUF:"

    def test_hoist_fires_on_trailing_clause_phrasing(self):
        reply = "Some narration first.\n\nBLUF: the incident is contained.\n\nDetails follow here at length."
        out, dropped = enforce_start_with(
            reply,
            ['Start your reply with "BLUF:" and then summarise the incident'])
        assert out.startswith("BLUF:")
        assert dropped > 0


# ── B-4: paragraph guard counts standalone lines ─────────────────────────────

class TestParagraphGuardLineAnchored:
    ERR = ("AssertionError: expected exit code 0 but got 1 "
           "(ModuleNotFoundError: No module named 'ghost_agent')")

    def _buf(self, standalone: int, embedded: int) -> str:
        parts = []
        for i in range(40):
            parts.append(f"Diverse debugging step {i}: "
                         f"inspected frame variable set number {i}.")
            if i < embedded:
                parts.append(f"Earlier we saw that {self.ERR} while retrying.")
        parts += [self.ERR] * standalone
        parts.append("still typing")
        return "\n".join(parts).ljust(SG.PARAGRAPH_LOOP_MIN_BUF + 1, "x")

    def test_embedded_quotes_do_not_fire(self):
        # 2 standalone + 4 embedded used to fire (substring count = 6).
        assert SG._detect_paragraph_loop(self._buf(2, 4)) is False

    def test_standalone_repeats_still_fire(self):
        assert SG._detect_paragraph_loop(
            self._buf(SG.PARAGRAPH_LOOP_THRESHOLD, 0)) is True


# ── stream-driven pins (B-1, B-2, B-5) ───────────────────────────────────────

def _client_text(chunks) -> str:
    out = []
    for c in chunks:
        try:
            s = c.decode("utf-8")
        except Exception:
            continue
        for line in s.split("\n\n"):
            line = line.strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except Exception:
                continue
            for ch in d.get("choices", []):
                t = (ch.get("delta") or {}).get("content")
                if t:
                    out.append(t)
    return "".join(out)


async def _drive(a, deltas, cancelled_after=None):
    async def final_stream(payload, use_coding=False):
        for d in deltas:
            yield sse({"content": d})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock()
    if cancelled_after is None:
        reg.is_cancelled.return_value = False
    else:
        calls = {"n": 0}

        def _c(_rid):
            calls["n"] += 1
            return calls["n"] > cancelled_after
        reg.is_cancelled.side_effect = _c
    gen, _, _ = a._stream_final_generation(_make_stream_state(reg))
    got = []
    async for chunk in gen:
        got.append(chunk)
    return got


class TestScrubDesyncFixed:
    @pytest.mark.asyncio
    async def test_split_tag_neither_leaks_nor_swallows(self):
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        chunks = await _drive(a, [
            "Hello. ", "<tool", "_call>", "<function name=\"x\">",
            "</function></tool_call>", " The answer is 42."])
        text = _client_text(chunks)
        assert "<tool" not in text          # no partial-tag leak
        assert "The answer is 42." in text  # no post-block swallow
        assert text.startswith("Hello. ")

    @pytest.mark.asyncio
    async def test_backtick_quoted_mention_is_preserved(self):
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        chunks = await _drive(a, [
            "To call a tool you emit a `<tool_call>` block. Hope that helps!"])
        text = _client_text(chunks)
        assert "`<tool_call>`" in text
        assert "Hope that helps!" in text


class TestWatchdogScoped:
    _ROW = "ROW 0 0 0 0 0 0 0 0\n"   # 20-char periodic unit

    @pytest.mark.asyncio
    async def test_periodic_plain_answer_is_not_severed(self):
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        deltas = ["The zero matrix you asked for:\n"] + [self._ROW] * 40
        chunks = await _drive(a, deltas)
        text = _client_text(chunks)
        assert text.count("ROW 0") >= 40          # nothing severed
        assert "SYSTEM OVERRIDE" not in text
        assert "stream interrupted" not in text

    @pytest.mark.asyncio
    async def test_think_loop_still_severed_and_client_stays_clean(self):
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        deltas = ["<think>\nlooping now:\n"] + [self._ROW] * 60
        chunks = await _drive(a, deltas)
        text = _client_text(chunks)
        assert text.count("ROW 0") < 60           # severed mid-loop
        # the client must NEVER receive the raw synthetic tool XML
        assert "SYSTEM OVERRIDE" not in text
        assert "<tool_call" not in text


class TestCancelCutIsTruncated:
    @pytest.mark.asyncio
    async def test_cancel_marks_truncated_and_durable_note(self):
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        await _drive(a, ["A long answer that gets ", "cut right here",
                         " and this part never sends"], cancelled_after=1)
        assert a._record_calibration_safe.called
        kw = a._record_calibration_safe.call_args.kwargs
        assert kw["truncated"] is True
        assert "cancelled by the user" in kw["final_ai_content"]


# ── B-3: the content-channel n-gram probe is think-scoped ────────────────────

class TestContentChannelProbeScope:
    @pytest.mark.asyncio
    async def test_repetitive_tool_body_is_not_aborted(self):
        # a file write of identical fixture rows used to abort as a
        # "thinking loop" on no-think turns (content discarded + fake strike).
        a = make_stream_agent()
        row = '  {"user_id": 0, "name": "placeholder", "score": 0.0},\n'
        body = ("<tool_call>\n<function name=\"file_system\">\n"
                "<parameter name=\"content\">[\n" + row * 40)
        chunks = [sse({"content": body[i:i + 80]})
                  for i in range(0, len(body), 80)]
        script = StreamScript(chunks)
        await run_chat(a, script)
        assert script.drained[0] is True, "repetitive tool body was aborted"

    @pytest.mark.asyncio
    async def test_inline_think_loop_still_aborts(self):
        # the positive arm: an OPEN inline <think> block that loops must
        # still be caught by the content-channel fallback.
        a = make_stream_agent()
        unit = "We must recheck the vowels count function once more. "
        chunks = [sse({"content": "<think>\n"})]
        chunks += [sse({"content": unit * 3}) for _ in range(12)]
        chunks.append(sse({"content": "THINKLOOP_COMPLETED_TEXT"}))
        script = StreamScript(chunks)
        out = await run_chat(a, script)
        assert script.drained[0] is False, "inline think loop was not aborted"
        assert "THINKLOOP_COMPLETED_TEXT" not in str(out)


# ── finalize-driven pins (A-F1, A-F2, A-F4) ──────────────────────────────────

class TestFallbackHeaderHonesty:
    @pytest.mark.asyncio
    async def test_errored_tool_gets_failed_head(self):
        a = make_fin_agent()
        tools = [{"name": "execute", "content":
                  "Error: Command failed\ngcc: fatal error: no input files"}]
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content="", tools_run_this_turn=tools,
            execution_failure_count=1, last_was_failure=True))
        assert "Process finished successfully." not in out
        assert "FAILED" in out

    @pytest.mark.asyncio
    async def test_clean_tool_keeps_success_head(self):
        a = make_fin_agent()
        tools = [{"name": "execute", "content":
                  "STDOUT/STDERR:\nall 3 tests passed\nEXIT CODE: 0"}]
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content="", tools_run_this_turn=tools))
        assert "Process finished successfully." in out


class TestHeadInsertRespectsStartWith:
    @pytest.mark.asyncio
    async def test_deferred_correction_lands_below_mandated_head(self):
        a = make_fin_agent()
        a._take_active_correction = MagicMock(
            return_value="⚠ CORRECTION: earlier verdict revised.\n\n---\n\n")
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content="BLUF: all clear.\n\nDetails follow here.",
            last_user_content='Begin your response with "BLUF:"'))
        assert out.startswith("BLUF: all clear.")
        assert "⚠ CORRECTION" in out

    @pytest.mark.asyncio
    async def test_without_constraint_correction_still_prepends(self):
        a = make_fin_agent()
        a._take_active_correction = MagicMock(
            return_value="⚠ CORRECTION: earlier verdict revised.\n\n---\n\n")
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content="Plain answer.",
            last_user_content="just answer"))
        assert out.startswith("⚠ CORRECTION")


class TestStaleVerdictRecompute:
    @pytest.mark.asyncio
    async def test_moved_text_recomputes(self):
        # the cached verdict fingerprints DIFFERENT text than the delivered
        # reply → finalize must recompute instead of stamping it.
        a = make_fin_agent()
        a.context.verifier = None
        a._compute_verifier_verdict_gated = AsyncMock(
            return_value=(None, {"name": "execute", "content": "x"}))
        vr = MagicMock()
        tools = [{"name": "execute", "content": "SUCCESS: ok"}]
        await a._finalize_and_return(_fs(
            final_ai_content="delivered text",
            tools_run_this_turn=tools,
            _verdict_is_fresh=True,
            _verifier_verdict_cache=(vr, tools[0], hash("judged text"))))
        assert a._compute_verifier_verdict_gated.called

    @pytest.mark.asyncio
    async def test_unmoved_text_reuses_cache(self):
        a = make_fin_agent()
        a.context.verifier = None
        a._compute_verifier_verdict_gated = AsyncMock(
            return_value=(None, {"name": "execute", "content": "x"}))
        vr = MagicMock()
        tools = [{"name": "execute", "content": "SUCCESS: ok"}]
        await a._finalize_and_return(_fs(
            final_ai_content="delivered text",
            tools_run_this_turn=tools,
            _verdict_is_fresh=True,
            _verifier_verdict_cache=(vr, tools[0], hash("delivered text"))))
        assert not a._compute_verifier_verdict_gated.called
