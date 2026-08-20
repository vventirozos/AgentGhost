"""§ context slice R2 (2026-08-19) — pins for the round's fixes.

R2 found the tenth "defect inside the previous fix" instance: the R1
XML-safety branch's close-before-open wedge GREW a message 1.33x/iteration
(2.2e9 chars in 20s on the request path), the sum-proxy selector starved the
cutter and mauled the goal, the request-end disarm skipped STREAMED turns
(the web UI's only path), the goal duplicated inside the recent window, and
first-open..last-close destroyed prose between blocks.
"""

import asyncio
import json
import time

import pytest

from ghost_agent.core.agent import GhostAgent
from unittest.mock import AsyncMock, MagicMock


def _agent():
    a = GhostAgent.__new__(GhostAgent)
    a.context = MagicMock()
    a.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "condensed summary."}}]})
    return a


# ── R2 C1: the close-before-open wedge is dead ───────────────────────────────

class TestNoGrowthWedge:
    def test_close_before_unclosed_open_shrinks_and_terminates(self):
        c = ("echoed log: </tool_call> some prose here " + "x" * 3000
             + " then an unclosed <tool_call" + "y" * 3000)
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "goal"},
                {"role": "assistant", "content": c},
                {"role": "user", "content": "now"}]
        before = len(c)
        t0 = time.monotonic()
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=800)
        took = time.monotonic() - t0
        assert took < 2.0, "wedge: cap took too long"
        after = len(out[2]["content"])
        assert after < before, "cut did not shrink the message"

    def test_every_cut_strictly_shrinks_total(self):
        # generic anti-wedge invariant on a mixed nasty history
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "goal"},
                {"role": "assistant",
                 "content": "</tool_call>" + "a" * 9000 + "<tool_call"},
                {"role": "tool", "name": "x",
                 "content": "<tool_call>" + "b" * 9000},
                {"role": "user", "content": "now"}]
        before = sum(len(str(m.get("content"))) for m in msgs)
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=800)
        after = sum(len(str(m.get("content"))) for m in out)
        assert after <= before


# ── R2 M2: the selector matches the cutter ───────────────────────────────────

class TestSelectorMatchesCutter:
    def test_uncuttable_many_parts_do_not_starve_the_cutter(self):
        # a 100-part message (no part >4000) must not block cutting the
        # 100KB string sibling, and the goal must survive untouched.
        parts = [{"type": "text", "text": "p" * 3900} for _ in range(25)]
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "THE_GOAL keep me"},
                {"role": "tool", "name": "x", "content": parts},
                {"role": "assistant", "content": "s" * 100000},
                {"role": "user", "content": "NEWEST keep me too"}]
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=9000)
        assert out[1]["content"] == "THE_GOAL keep me"
        assert out[4]["content"] == "NEWEST keep me too"
        assert "dropped by context budget" in out[3]["content"]

    def test_sum_of_small_parts_is_cuttable_now(self):
        parts = [{"type": "text", "text": "p" * 3900} for _ in range(25)]
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "goal"},
                {"role": "tool", "name": "x", "content": parts},
                {"role": "user", "content": "now"}]
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=2000)
        kept = out[2]["content"]
        total = sum(len(p.get("text", "")) for p in kept
                    if isinstance(p, dict))
        assert total < 25 * 3900
        assert any("dropped by context budget" in p.get("text", "")
                   for p in kept if isinstance(p, dict))


# ── R2 M5: prose between blocks survives ─────────────────────────────────────

class TestProseBetweenBlocksSurvives:
    def test_finding_between_two_blocks_kept(self):
        c = ("<tool_call><function name=\"a\">" + "A" * 6000
             + "</function></tool_call>"
             + " IMPORTANT FINDING: the root cause is X "
             + "<tool_call><function name=\"b\">small</function></tool_call>"
             + " tail prose")
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "goal"},
                {"role": "assistant", "content": c},
                {"role": "user", "content": "now"}]
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=800)
        got = out[2]["content"]
        assert "IMPORTANT FINDING: the root cause is X" in got
        assert "tool_call block" in got  # the big block was removed


# ── R2 M4: the goal is never duplicated ──────────────────────────────────────

class TestGoalNotDuplicated:
    def test_goal_inside_recent_window_appears_once(self):
        # § context R3 MAJOR-2: >=6 NON-system messages so the MAIN branch
        # (which owns _goal_head) runs — five routed down the <=5 truncation
        # branch and the pin was vacuous against the unconditional prepend.
        msgs = [{"role": "system", "content": "sys"},
                {"role": "tool", "name": "fs",
                 "content": "seed " + "z" * 9000},
                {"role": "user", "content": "UNIQUE_GOAL_TOKEN do it"},
                {"role": "assistant", "content": "a1"},
                {"role": "tool", "name": "x", "content": "t1"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "and then?"},
                {"role": "assistant", "content": "a3"}]
        a = _agent()
        out = asyncio.run(a._prune_context(msgs, max_tokens=100))
        text = "\n".join(str(m.get("content")) for m in out)
        assert text.count("UNIQUE_GOAL_TOKEN") == 1


# ── R2 M3: streamed turns disarm the read budget ─────────────────────────────

class TestStreamedDisarm:
    @pytest.mark.asyncio
    async def test_stream_then_unregister_clears_budget(self):
        # § context R3 MAJOR-2 (pin rewritten twice): the governor harness
        # never reaches the streamed-final generator (its return is plain
        # text), so the drain disarm must be pinned on the REAL
        # _stream_final_generation, whose return wraps stream_wrapper in
        # _stream_then_unregister. The budget is armed AFTER the generator
        # exists, so only the drain's finally can clear it.
        from ghost_agent.tools.file_system import ReadBudget
        from tests.test_finalize_stream_pins import (
            make_stream_agent, _make_stream_state, sse)

        a2 = make_stream_agent()

        async def final_stream(payload, use_coding=False):
            yield sse({"content": "hello"})
            yield b"data: [DONE]\n\n"

        a2.context.llm_client.stream_chat_completion = final_stream
        a2._record_calibration_safe = AsyncMock()
        reg = MagicMock()
        reg.is_cancelled.return_value = False
        gen, _, _ = a2._stream_final_generation(_make_stream_state(reg))
        a2.context._read_budget = ReadBudget(0)   # armed mid-"request"
        async for _ in gen:
            pass
        assert a2.context._read_budget is None, (
            "streamed drain left the read budget armed")


# ── R3 additions ─────────────────────────────────────────────────────────────

class TestSegmentScanPinned:
    def test_prose_around_small_blocks_is_cut(self):
        # § context R3 V3: the tag-free segment scan was wholly unpinned —
        # a huge prose message whose blocks are all SMALL must still shrink
        # via a prose-segment cut (never inside a block).
        c = ("P" * 9000
             + "<tool_call><function name=\"a\">tiny</function></tool_call>"
             + "Q" * 500)
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "goal"},
                {"role": "assistant", "content": c},
                {"role": "user", "content": "now"}]
        from ghost_agent.core.agent import GhostAgent
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=800)
        got = out[2]["content"]
        assert len(got) < len(c)
        assert "dropped by context budget" in got
        # the intact block survives whole
        assert "<tool_call><function name=\"a\">tiny</function></tool_call>" in got


class TestMalformedPartsNeverRaise:
    def test_non_str_text_parts_are_tolerated(self):
        # § context R3 MINOR: a non-str "text" part raised TypeError inside
        # the cap and bricked every over-budget turn.
        from ghost_agent.core.agent import GhostAgent
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "goal"},
                {"role": "tool", "name": "x",
                 "content": [{"type": "text", "text": 12345},
                             {"type": "text", "text": None},
                             {"type": "text", "text": "ok " * 3000}]},
                {"role": "user", "content": "now"}]
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=800)
        assert "dropped by context budget" in str(out[2]["content"])

    def test_sum_shape_malformed_part_reaches_the_sum_branch(self):
        # § context R4: the previous fixture's 9000-char part meant the
        # best-pick site always cut first — the `_txt_total` and
        # kept/dropped guards were silently revertible. This fixture has NO
        # part >4000 (sum branch only) plus a malformed part.
        from ghost_agent.core.agent import GhostAgent
        parts = ([{"type": "text", "text": "p" * 3900} for _ in range(3)]
                 + [{"type": "text", "text": 12345}])
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "goal"},
                {"role": "tool", "name": "x", "content": parts},
                {"role": "user", "content": "now"}]
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=800)
        kept = out[2]["content"]
        assert any("dropped by context budget" in str(p.get("text", ""))
                   for p in kept if isinstance(p, dict))

    def test_counter_visible_nonstr_part_is_cuttable(self):
        # § context R4 parity: a list-valued "text" (JSON-reachable) costs
        # its serialized size in the counter — the cutter must be able to
        # shrink it instead of last-resort-mauling the goal.
        from ghost_agent.core.agent import GhostAgent
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "THE_GOAL keep me"},
                {"role": "tool", "name": "x",
                 "content": [{"type": "text", "text": ["A" * 4000] * 25}]},
                {"role": "user", "content": "NEWEST keep me too"}]
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=2000)
        assert out[1]["content"] == "THE_GOAL keep me"
        assert out[3]["content"] == "NEWEST keep me too"
        part = out[2]["content"][0]
        assert isinstance(part.get("text"), str)
        assert "non-text part" in part["text"]


class TestCancelPathDisarm:
    @pytest.mark.asyncio
    async def test_cancelled_turn_disarms_the_budget(self):
        # § context R3 MAJOR-1: the TurnCancelled return (the Stop button —
        # correlated with exactly the lockdown turns) left the budget armed.
        from ghost_agent.tools.file_system import ReadBudget
        from tests.test_context_governor_pins import (
            _ctx, _wire_stream, _FakeBgTasks)
        import tempfile, pathlib
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            ctx = _ctx(pathlib.Path(td), max_context=240000)
            agent = GhostAgent(ctx)
            agent.thinking_budget_override = "selfplay"
            _wire_stream(ctx, [
                '<tool_call>{"name": "nosuch_tool", "arguments": {}}</tool_call>',
                "FINAL: done."])
            import ghost_agent.core.turns as turns_mod
            reg = turns_mod.get_turn_registry(agent)
            body = {"messages": [{"role": "user", "content": "long job"}]}
            _orig = reg.is_cancelled
            calls = {"n": 0}

            def _cancel_second(rid):
                calls["n"] += 1
                # arm a leftover budget mid-request, then cancel
                if calls["n"] == 2:
                    ctx._read_budget = ReadBudget(0)
                    return True
                return _orig(rid)

            with patch.object(reg, "is_cancelled", side_effect=_cancel_second), \
                 patch("ghost_agent.core.agent.pretty_log"), \
                 patch("ghost_agent.core.agent.get_active_tool_definitions",
                       return_value=[]):
                out = await agent.handle_chat(body, _FakeBgTasks(),
                                              request_id="pin-cancel")
            assert calls["n"] >= 2, "cancel check never reached arming point"
            assert "cancelled" in str(out).lower()
            assert ctx._read_budget is None, (
                "cancelled turn left the read budget armed")


class TestCrashExitDisarm:
    @pytest.mark.asyncio
    async def test_mid_request_exception_disarms_the_budget(self):
        # § context R4 M1d: the universal disarm's value beyond cancel is the
        # CRASH exit — an exception between arming and any point disarm must
        # still cross the outer finally.
        from ghost_agent.tools.file_system import ReadBudget
        from tests.test_context_governor_pins import (
            _ctx, _wire_stream, _FakeBgTasks)
        import tempfile, pathlib
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            ctx = pathlib.Path(td)
            ctx = _ctx(ctx, max_context=240000)
            agent = GhostAgent(ctx)
            agent.thinking_budget_override = "selfplay"
            _wire_stream(ctx, ["FINAL: done."])
            body = {"messages": [{"role": "user", "content": "boom"}]}

            async def _boom(fs_state):
                ctx._read_budget = ReadBudget(0)   # armed mid-request
                raise RuntimeError("mid-request crash")

            with patch.object(agent, "_finalize_and_return",
                              side_effect=_boom), \
                 patch("ghost_agent.core.agent.pretty_log"), \
                 patch("ghost_agent.core.agent.get_active_tool_definitions",
                       return_value=[]):
                try:
                    await agent.handle_chat(body, _FakeBgTasks(),
                                            request_id="pin-crash")
                except RuntimeError:
                    pass
            assert ctx._read_budget is None, (
                "crash exit left the read budget armed")
