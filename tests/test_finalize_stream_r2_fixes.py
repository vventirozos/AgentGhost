"""§ finalize/stream slice R2 (2026-08-19) — pins for the round's fixes.

R2 attacked R1's fixes and the class held a sixth time (both lenses): the
hold-back guarded the LAST '<' (a proxy — an earlier forming tag was
released), the think-gates were re-armed by quoted mentions and never
disarmed by `</thinking>`, the A-F4 fingerprint was bypassed by the SYNC
stamp's 2-tuple, `_head_insert` landed inside code fences, the A-F1 prefix
sniff branded log dumps FAILED, the weak-marker corroboration missed
proximity, and the status-run rule deleted legitimate answers. All pins
drive real code.
"""

import json
import types

import pytest

from ghost_agent.core.agent import (
    GhostAgent, _inline_think_open, _emit_safe_end,
    _scrub_task_status_runs, _truncate_prompt_bleed)
from unittest.mock import AsyncMock, MagicMock

from tests.test_finalize_stream_pins import (
    _fs, _make_stream_state, make_fin_agent, make_stream_agent, sse,
    StreamScript, run_chat)
from tests.test_finalize_stream_r1_fixes import _client_text, _drive


# ── R2 C1/M4: hold-back from the FIRST unresolved tag fragment ───────────────

class TestHoldBackFirstUnresolved:
    def test_second_lt_does_not_release_forming_tag(self):
        # lens A C1 reproduction: client must equal the scrubbed durable view.
        view1 = "abc<function or <t"
        # first '<' opens a tag-prefixed unresolved fragment → hold from it
        assert _emit_safe_end(view1, 0) == 3

    def test_non_tag_first_lt_released_tag_second_held(self):
        view = "a < b <tool"
        assert _emit_safe_end(view, 0) == 6  # " b " emits; "<tool" held

    def test_long_prose_fragment_released(self):
        view = "if x <tool_threshold else zero " + "y" * 60
        assert _emit_safe_end(view, 0) == len(view)

    @pytest.mark.asyncio
    async def test_two_lt_stream_no_leak_no_swallow(self):
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        chunks = await _drive(a, [
            "abc", "<function or ", "<t",
            "his> syntax</function> real answer XYZ"])
        text = _client_text(chunks)
        assert "<function" not in text
        assert "real answer XYZ" in text
        assert text.startswith("abc")

    @pytest.mark.asyncio
    async def test_dangling_opener_suppressed_at_end(self):
        # R2 M4: stream ends mid-tag → the client never sees the raw opener.
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        chunks = await _drive(a, ["Answer done. ", "<tool_call"])
        text = _client_text(chunks)
        assert "<tool_call" not in text
        assert "Answer done." in text


# ── R2 M1/M2/M3: mention-aware, closer-prefix think gates ───────────────────

class TestInlineThinkGate:
    def test_quoted_mention_does_not_open(self):
        assert _inline_think_open("the `<think` marker is used here") is False

    def test_thinking_variant_closes(self):
        assert _inline_think_open(
            "<thinking>\nreason\n</thinking>\nanswer") is False

    def test_real_open_block_detected(self):
        assert _inline_think_open("<think>\nstill reasoning") is True

    @pytest.mark.asyncio
    async def test_quoted_think_mention_plus_periodic_not_severed(self):
        # lens A M1: a backticked `<think` mention re-armed the watchdog.
        a = make_stream_agent()
        a._record_calibration_safe = AsyncMock()
        row = "ROW 0 0 0 0 0 0 0 0\n"
        deltas = ["I never use the `<think` marker here.\n"] + [row] * 40
        chunks = await _drive(a, deltas)
        text = _client_text(chunks)
        assert text.count("ROW 0") >= 40
        assert "stream interrupted" not in text

    @pytest.mark.asyncio
    async def test_closed_thinking_variant_plus_tool_body_not_aborted(self):
        # lens A M2: </thinking> never disarmed the probe gate.
        a = make_stream_agent()
        row = '  {"user_id": 0, "name": "placeholder", "score": 0.0},\n'
        body = ("<thinking>plan the write</thinking>\n"
                "<tool_call>\n<function name=\"file_system\">\n"
                "<parameter name=\"content\">[\n" + row * 40)
        chunks = [sse({"content": body[i:i + 80]})
                  for i in range(0, len(body), 80)]
        script = StreamScript(chunks)
        await run_chat(a, script)
        assert script.drained[0] is True, "closed <thinking> turn was aborted"


# ── R2 C-1: fingerprint gate covers the sync stamp shape ────────────────────

class TestVerdictGateSyncShape:
    @pytest.mark.asyncio
    async def test_real_verdict_without_fingerprint_recomputes(self):
        # the SYNC branch stamped 2-tuples; a real verdict with a missing
        # fingerprint must mean "unknown → recompute", never "trusted".
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
            _verifier_verdict_cache=(vr, tools[0])))   # legacy 2-tuple
        assert a._compute_verifier_verdict_gated.called

    @pytest.mark.asyncio
    async def test_none_verdict_two_tuple_still_reused(self):
        # the deliberate bookkeeping-skip shape (None, None) keeps its quiet
        # reuse path.
        a = make_fin_agent()
        a.context.verifier = None
        a._compute_verifier_verdict_gated = AsyncMock(
            return_value=(None, None))
        tools = [{"name": "execute", "content": "SUCCESS: ok"}]
        await a._finalize_and_return(_fs(
            final_ai_content="delivered text",
            tools_run_this_turn=tools,
            _verdict_is_fresh=True,
            _verifier_verdict_cache=(None, None)))
        assert not a._compute_verifier_verdict_gated.called


# ── R2 M-1: fence-aware _head_insert ─────────────────────────────────────────

class TestHeadInsertFenceAware:
    @pytest.mark.asyncio
    async def test_block_never_lands_inside_a_code_fence(self):
        a = make_fin_agent()
        a._take_active_correction = MagicMock(
            return_value="NOTE_BLOCK_XYZ\n\n---\n\n")
        reply = ("BLUF: patch below fixes it.\n```python\ndef f():\n\n"
                 "    return 1\n```\n\nDone.")
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content=reply,
            last_user_content='Begin your response with "BLUF:"'))
        assert out.startswith("BLUF:")
        assert "NOTE_BLOCK_XYZ" in out
        # the note must sit OUTSIDE the fence: an even number of ``` before it
        assert out[:out.find("NOTE_BLOCK_XYZ")].count("```") % 2 == 0


# ── R2 M-2: fallback header classification ───────────────────────────────────

class TestFallbackHeaderClassification:
    @pytest.mark.asyncio
    async def test_error_count_log_read_is_not_branded_failed(self):
        a = make_fin_agent()
        tools = [{"name": "file_system",
                  "content": "ERROR count: 0 across 14 services\nall clean"}]
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content="", tools_run_this_turn=tools))
        assert "FAILED" not in out
        assert "Process finished successfully." in out

    @pytest.mark.asyncio
    async def test_execute_exit_zero_outranks_errorish_text(self):
        a = make_fin_agent()
        tools = [{"name": "execute", "content":
                  "STDOUT/STDERR:\nError: retrying x... recovered\n"
                  "EXIT CODE: 0"}]
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content="", tools_run_this_turn=tools))
        assert "Process finished successfully." in out

    @pytest.mark.asyncio
    async def test_exit_code_as_data_in_non_execute_is_ignored(self):
        a = make_fin_agent()
        tools = [{"name": "file_system",
                  "content": "ci-log.txt saved earlier says EXIT CODE: 3"}]
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content="", tools_run_this_turn=tools))
        assert "FAILED" not in out


# ── R2 M-3 / A-F3a / A-F3b: bleed refinements ────────────────────────────────

class TestBleedRefinements:
    def test_far_apart_weak_markers_are_kept(self):
        text = ("# My Project\n\n# Tools\n\n- hammer\n\n" + "prose " * 150 +
                '\nexample: {"type": "function", "function": {}}')
        assert _truncate_prompt_bleed(text) == text

    def test_adjacent_weak_markers_truncate(self):
        text = ('head\n# Tools\n{"type": "function", "function": {}}')
        assert _truncate_prompt_bleed(text) == "head\n"

    def test_quoted_strong_marker_is_kept(self):
        text = ('My prompt has sections like "CRITICAL INSTRUCTION:" and '
                "more — here is the full explanation of each.")
        assert _truncate_prompt_bleed(text) == text

    def test_native_pointer_sentence_truncates(self):
        text = ("head\n(Tool schemas are advertised via the native "
                "tools channel)\nleaked list")
        assert _truncate_prompt_bleed(text) == "head\n"


# ── R2 M-5: A-F6 pins (previously silently revertible) ──────────────────────

class TestRanInfoErrorShapes:
    @pytest.mark.asyncio
    async def test_critical_tool_error_does_not_resolve_unknowns(self):
        a = make_fin_agent()
        unknown = types.SimpleNamespace(
            resolved=False, resolution="search the web for it", impact=3,
            text="what port does nova use")
        tracker = MagicMock()
        tracker.unknowns = [unknown]
        tracker.should_ask_user.return_value = None
        tracker.get_risk_summary.return_value = ""
        a.context.uncertainty_tracker = tracker
        tools = [{"name": "web_search",
                  "content": "Critical Tool Error: all circuits failed"}]
        await a._finalize_and_return(_fs(
            final_ai_content="answer", tools_run_this_turn=tools))
        assert not tracker.resolve_unknown.called

    @pytest.mark.asyncio
    async def test_clean_info_tool_resolves_unknowns(self):
        a = make_fin_agent()
        unknown = types.SimpleNamespace(
            resolved=False, resolution="search the web for it", impact=3,
            text="what port does nova use")
        tracker = MagicMock()
        tracker.unknowns = [unknown]
        tracker.should_ask_user.return_value = None
        tracker.get_risk_summary.return_value = ""
        a.context.uncertainty_tracker = tracker
        tools = [{"name": "web_search", "content": "nova listens on 8088"}]
        await a._finalize_and_return(_fs(
            final_ai_content="answer", tools_run_this_turn=tools))
        assert tracker.resolve_unknown.called


class TestCopiedCountsAsWrite:
    @pytest.mark.asyncio
    async def test_three_copies_fire_promotion_nudge(self):
        from tests.test_finalize_stream_pins import FakeScratchpad
        a = make_fin_agent()
        a.context.scratchpad = FakeScratchpad()
        a.context.current_project_id = None
        msgs = []
        for i in range(9):
            msgs.append({"role": "user",
                         "content": f"please copy asset {i} into place now"})
            msgs.append({"role": "assistant", "content": f"done {i}"})
        copies = [{"name": "file_system",
                   "content": f"SUCCESS: Copied 'a{i}.png' -> 'b{i}.png'"}
                  for i in range(3)]
        out, _, _ = await a._finalize_and_return(_fs(
            final_ai_content="A fine answer.", messages=msgs,
            tools_run_this_turn=copies))
        assert "tracked project" in out
