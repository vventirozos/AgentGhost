"""§ finalize/stream slice R3 (2026-08-19) — pins for the round's fixes.

R3 attacked R2's fixes and the class held a seventh time: the think-gate's
quote-skip swallowed a REAL closer ending in a quoted string (false sever
back), the 64-char release valve recreated the swallow it prevented, the
banner check keyed on the tool NAME (acquired skills/delegate false-
successed), one task-id in prose poisoned a legit run, and the fence walk
was blind to ~~~ fences. Plus the R1-rooted quadratic scrub re-sub, now
incremental. All pins drive real code.
"""

import time

import pytest

from ghost_agent.core.agent import (
    _emit_safe_end, _frag_is_forming_tag, _inline_think_open,
    _scrub_task_status_runs, _scrub_tail_is_open)
from unittest.mock import AsyncMock, MagicMock

from tests.test_finalize_stream_pins import (
    _fs, make_fin_agent, make_stream_agent, sse, StreamScript, run_chat)
from tests.test_finalize_stream_r1_fixes import _client_text, _drive


# ── R3 A-D1: asymmetric think gate ───────────────────────────────────────────

class TestThinkGateAsymmetry:
    def test_quote_terminated_real_closer_counts(self):
        buf = '<think>the file is "notes.md"</think>\nAnswer follows.'
        assert _inline_think_open(buf) is False

    def test_apostrophe_preceded_real_opener_counts(self):
        assert _inline_think_open("users'<think>\nnow reasoning") is True

    def test_backtick_mention_still_skipped(self):
        assert _inline_think_open("I never emit the `<think` token") is False

    @pytest.mark.asyncio
    async def test_quoted_string_before_closer_not_severed(self):
        # the R3 reproduction: quote-ending think block + periodic answer.
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        row = "ROW 0 0 0 0 0 0 0 0\n"
        deltas = ['<think>quote: "done"</think>\n'] + [row] * 40
        chunks = await _drive(a, deltas)
        text = _client_text(chunks)
        assert text.count("ROW 0") >= 40
        assert "stream interrupted" not in text


# ── R3 A-D2: tag-name boundary replaces the 64-char valve ───────────────────

class TestFormingTagBoundary:
    def test_long_attribute_opener_is_held(self):
        frag = 'function name="deep_research" extra="' + "x" * 40
        assert _frag_is_forming_tag(frag.lower()) is True

    def test_longer_word_prose_released_immediately(self):
        assert _frag_is_forming_tag("tool_threshold else zero") is False
        assert _frag_is_forming_tag("functional programming rocks") is False

    def test_partial_names_still_held(self):
        assert _frag_is_forming_tag("tool_c") is True
        assert _frag_is_forming_tag("fu") is True

    @pytest.mark.asyncio
    async def test_long_attr_opener_no_leak_no_swallow(self):
        # the R3 reproduction: >64-char opener completing later.
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        opener = 'Answer: <function name="deep_research" extra="' + "x" * 40
        chunks = await _drive(a, [
            opener, '">body</function>', " The result is 42."])
        text = _client_text(chunks)
        assert "<function" not in text
        assert "The result is 42." in text
        assert text.startswith("Answer: ")


# ── R3 A-D3: incremental scrub view ──────────────────────────────────────────

class TestIncrementalScrub:
    def test_tail_open_detector(self):
        assert _scrub_tail_is_open("x <tool_call>eating this") is True
        assert _scrub_tail_is_open(
            "x <tool_call>done</tool_call>") is False
        assert _scrub_tail_is_open("no tags at all") is False

    @pytest.mark.asyncio
    async def test_freeze_and_resume_byte_exact(self):
        # open tag ('>' recompute) → '>'-less body (frozen) → close ('>')
        # → post text: the client must see exactly pre+post.
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        chunks = await _drive(a, [
            "pre ", "<tool_call>", "body one ", "body two ",
            "</tool_call>", " post text"])
        assert _client_text(chunks) == "pre  post text"

    @pytest.mark.asyncio
    async def test_unclosed_opener_spam_is_linear_enough(self):
        # 300 '>'-less spam chunks used to re-sub the whole buffer each time
        # (quadratic-with-backtracking). Bound the wall clock generously.
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        t0 = time.monotonic()
        await _drive(a, ["<tool "] * 300 + ["and done>"])
        assert time.monotonic() - t0 < 10.0


# ── R3 B-D1: banner shape, not tool name ─────────────────────────────────────

class TestBannerShapeNotName:
    @pytest.mark.asyncio
    async def test_acquired_skill_failure_is_failed(self):
        a = make_fin_agent()
        tools = [{"name": "analyze_logs", "content":
                  "--- EXECUTION RESULT ---\nEXIT CODE: 1\n"
                  "STDOUT/STDERR:\nTraceback (most recent call last): boom"}]
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content="", tools_run_this_turn=tools))
        assert "Process finished successfully." not in out
        assert "FAILED" in out

    @pytest.mark.asyncio
    async def test_delegate_sandbox_job_failure_is_failed(self):
        a = make_fin_agent()
        tools = [{"name": "delegate", "content":
                  "[sandbox job 3 finished — EXIT CODE: 1] tail of log"}]
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content="", tools_run_this_turn=tools))
        assert "Process finished successfully." not in out

    @pytest.mark.asyncio
    async def test_acquired_skill_success_is_success(self):
        a = make_fin_agent()
        tools = [{"name": "analyze_logs", "content":
                  "--- EXECUTION RESULT ---\nEXIT CODE: 0\n"
                  "STDOUT/STDERR:\nall good"}]
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content="", tools_run_this_turn=tools))
        assert "Process finished successfully." in out


# ── R3 B-D2: majority-shaped runs ────────────────────────────────────────────

class TestMajorityShapedRuns:
    def test_one_id_in_prose_does_not_poison_run(self):
        text = ("Current standing:\n- Ship the report (DONE)\n"
                "- Fix the task_7 regression (IN_PROGRESS)\n"
                "- Update docs (PENDING)\nMore prose.")
        assert _scrub_task_status_runs(text) == text

    def test_markdown_link_lead_is_not_task_shaped(self):
        text = ("[README](docs/readme.md) refreshed (DONE)\n"
                "[CHANGELOG](docs/log.md) updated (DONE)\n"
                "[LICENSE](LICENSE) checked (DONE)\nAll set.")
        assert _scrub_task_status_runs(text) == text

    def test_majority_shaped_run_still_stripped(self):
        tree = "\n".join(f"[task_{i}] step (IN_PROGRESS)" for i in range(4))
        text = f"head\n{tree}\ntail"
        assert _scrub_task_status_runs(text) == "head\ntail"


# ── R3 B-D3: dual-dialect fences ─────────────────────────────────────────────

class TestFenceDialects:
    @pytest.mark.asyncio
    async def test_tilde_fence_not_corrupted(self):
        a = make_fin_agent()
        a._take_active_correction = MagicMock(
            return_value="NOTE_BLOCK_XYZ\n\n---\n\n")
        reply = ("BLUF: here is the README template.\n~~~markdown\n# Title\n\n"
                 "body text\n~~~\n\nDone.")
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content=reply,
            last_user_content='Begin your response with "BLUF:"'))
        assert "NOTE_BLOCK_XYZ" in out
        import re as _re
        before = out[:out.find("NOTE_BLOCK_XYZ")]
        assert len(_re.findall(r'(?m)^\s*(?:`{3,}|~{3,})', before)) % 2 == 0

    @pytest.mark.asyncio
    async def test_unclosed_fence_at_eos_prepends(self):
        # constraint yields to content integrity: plain prepend, fence intact.
        a = make_fin_agent()
        a._take_active_correction = MagicMock(
            return_value="NOTE_BLOCK_XYZ\n\n---\n\n")
        reply = "BLUF: partial dump.\n```python\ndef f():\n\n    pass"
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content=reply,
            last_user_content='Begin your response with "BLUF:"'))
        assert out.startswith("NOTE_BLOCK_XYZ")
        assert "def f():\n\n    pass" in out
