"""§ context slice R1 (2026-08-19) — pins for the round's fixes.

Lens A: the tail-cap's counter/cutter type mismatch (7.5x-budget payloads),
the marker spliced inside tool_call XML, the anchor cap vaporizing the
newest findings, the third "goal" site, and the protect-first-user proxy.
Lens B: the ReadBudget arm-without-disarm, the first-read exemption, the
uncharged read_chunked bypass, the injection list-repr destruction, and the
lockdown-blind sampler. All pins drive real code.
"""

import asyncio
import json

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.context_manager import ContextManager
from ghost_agent.tools.file_system import ReadBudget, read_byte_budget
from unittest.mock import AsyncMock, MagicMock


def _agent():
    a = GhostAgent.__new__(GhostAgent)
    a.context = MagicMock()
    a.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "condensed summary."}}]})
    return a


# ── A-F1: the cutter cuts everything the counter counts ─────────────────────

class TestCutterCoversCounter:
    def test_list_text_part_is_capped(self):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "goal"},
                {"role": "assistant",
                 "content": [{"type": "text", "text": "L" * 60000},
                             {"type": "image_url",
                              "image_url": {"url": "data:image/x;base64,AA"}}]}]
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=2000)
        part = out[2]["content"][0]["text"]
        assert len(part) < 60000
        assert "dropped by context budget" in part
        # image part untouched
        assert out[2]["content"][1]["type"] == "image_url"

    def test_native_tool_calls_args_are_capped_as_valid_json(self):
        big = json.dumps({"operation": "write", "path": "x",
                          "content": "B" * 60000})
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "goal"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "t", "type": "function",
                                 "function": {"name": "file_system",
                                              "arguments": big}}]}]
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=2000)
        args = out[2]["tool_calls"][0]["function"]["arguments"]
        assert len(args) < 5000
        parsed = json.loads(args)          # MUST stay valid JSON
        assert "_dropped_by_context_budget" in parsed


# ── A-F2: never splice the marker inside tool_call XML ───────────────────────

class TestNoSpliceInsideToolCallXml:
    def test_oversized_block_is_removed_whole(self):
        c = ("prose before <tool_call><function name=\"x\">"
             + "A" * 12000 + "</function></tool_call> prose after")
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "goal"},
                {"role": "assistant", "content": c}]
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=1000)
        got = out[2]["content"]
        # the marker never sits between intact tags
        assert not ("<tool_call" in got and "</tool_call>" in got
                    and "dropped by context budget" in got.split(
                        "<tool_call")[1].split("</tool_call>")[0])
        assert "tool_call block" in got and "dropped" in got
        assert "prose before" in got and "prose after" in got

    def test_unclosed_block_is_removed_whole(self):
        c = "prose " + "<tool_call><function name=\"x\">" + "A" * 12000
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "goal"},
                {"role": "assistant", "content": c}]
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=1000)
        got = out[2]["content"]
        assert "unclosed tool_call" in got
        assert "A" * 100 not in got


# ── A-F5a: newest user protected; honest last resort ─────────────────────────

class TestNewestUserProtected:
    def test_other_candidates_cut_first(self):
        # the NEWEST user message is deliberately the LARGEST — without the
        # exemption, largest-first picking would cut IT first (the mutant),
        # so this fixture discriminates the exemption itself.
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "filler " + "q" * 12000},
                {"role": "user",
                 "content": "PASTE " + "p" * 15000
                            + " CRITICAL_CONSTRAINT_MID " + "p" * 15000}]
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=11000)
        newest = out[3]["content"]
        assert "CRITICAL_CONSTRAINT_MID" in newest
        assert "dropped by context budget" not in newest
        assert "dropped by context budget" in out[2]["content"]

    def test_last_resort_uses_paste_note(self):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "small"},
                {"role": "user", "content": "PASTE " + "p" * 28000}]
        out = GhostAgent._cap_oversized_tail(msgs, max_tokens=2000)
        newest = out[3]["content"]
        assert "pasted content exceeded" in newest
        assert "start_line" not in newest


# ── A-F3 / A-F4: anchors keep the newest; goal is the first USER ─────────────

class TestAnchorsAndGoal:
    def _msgs(self, n_anchors):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "the real goal"}]
        for i in range(n_anchors):
            msgs.append({"role": "assistant", "content": f"step {i}"})
            msgs.append({"role": "tool", "name": "execute",
                         "content": f"Error: SECRET_{i} failed badly "
                                    + "x" * 300})
        msgs += [{"role": "assistant", "content": "recent a"},
                 {"role": "user", "content": "recent q"},
                 {"role": "assistant", "content": "recent b"},
                 {"role": "user", "content": "recent q2"},
                 {"role": "assistant", "content": "recent c"},
                 {"role": "user", "content": "now do it"}]
        return msgs

    def test_newest_anchors_kept_and_overflow_summarized(self):
        a = _agent()
        out = asyncio.run(a._prune_context(self._msgs(6), max_tokens=100))
        text = "\n".join(str(m.get("content")) for m in out)
        # newest anchors survive verbatim
        assert "SECRET_5" in text and "SECRET_4" in text
        # the overflow went to the SUMMARIZER, not the void
        ca = a.context.llm_client.chat_completion.call_args
        sent = json.dumps([list(ca.args), dict(ca.kwargs)], default=str)
        assert "SECRET_0" in sent or "SECRET_0" in text

    def test_tool_seeded_history_keeps_user_goal(self):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "tool", "name": "fs", "content": "seed result"},
                {"role": "user", "content": "REAL_GOAL_MARKER do the thing"}]
        for i in range(10):
            msgs.append({"role": "assistant", "content": f"work {i} "
                                                         + "y" * 200})
        a = _agent()
        out = asyncio.run(a._prune_context(msgs, max_tokens=100))
        text = "\n".join(str(m.get("content")) for m in out)
        assert "REAL_GOAL_MARKER" in text


# ── A-F5b: CM L3 protects the newest user message ────────────────────────────

class TestCmNewestUserSacrosanct:
    def test_current_request_not_compressed(self):
        # drive the internal ladder application directly at L3 — the live
        # tail shape [..., user_current, assistant, tool] with keep_full
        # putting the current user in the OLD segment.
        cm = ContextManager(max_tokens=1000)
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "old goal " + "g" * 1200},
                {"role": "assistant", "content": "a" * 2500},
                {"role": "user", "content": "CURRENT req "
                                            + "c" * 800
                                            + " HARD_CONSTRAINT_AT_900 "
                                            + "c" * 200},
                {"role": "assistant", "content": "recent"},
                {"role": "tool", "name": "fs", "content": "recent tool"}]
        out = cm._apply_compression(msgs, 3)
        text = "\n".join(str(m.get("content")) for m in out)
        assert "HARD_CONSTRAINT_AT_900" in text
        # the goal stays sacrosanct too (regression guard for C-MAJOR-1)
        assert "old goal" in text and "g" * 1200 in text


# ── B2b: no first-read exemption ─────────────────────────────────────────────

class TestFirstReadHonorsBudget:
    @pytest.mark.asyncio
    async def test_first_read_bigger_than_remaining_is_refused(self, tmp_path):
        from ghost_agent.tools.file_system import tool_read_file
        big = tmp_path / "big.txt"
        big.write_text("z" * 50000)
        out = await tool_read_file(
            "big.txt", sandbox_dir=tmp_path, max_context=240000,
            read_budget=ReadBudget(3000))
        assert out.startswith("Error")
        assert "read budget" in out or "context ceiling" in out


# ── B3: chunked reads are budgeted ───────────────────────────────────────────

class TestChunkedReadsBudgeted:
    @pytest.mark.asyncio
    async def test_chunked_refused_under_lockdown(self, tmp_path):
        from ghost_agent.tools.file_system import tool_file_system
        f = tmp_path / "doc.txt"
        f.write_text("line\n" * 20000)
        out = await tool_file_system(
            operation="read_chunked", path="doc.txt", sandbox_dir=tmp_path,
            max_context=240000, read_budget=ReadBudget(0))
        assert str(out).startswith("Error")
        assert "context ceiling" in str(out)

    @pytest.mark.asyncio
    async def test_chunked_read_charges_the_budget(self, tmp_path):
        from ghost_agent.tools.file_system import tool_file_system
        f = tmp_path / "doc.txt"
        f.write_text("line\n" * 20000)
        rb = ReadBudget(500000)
        out = await tool_file_system(
            operation="read_chunked", path="doc.txt", sandbox_dir=tmp_path,
            max_context=240000, read_budget=rb)
        assert not str(out).startswith("Error")
        assert rb.spent > 0


# ── B4: injection preserves structured list content ──────────────────────────

class TestInjectionListContent:
    def _msgs(self):
        return [{"role": "system", "content": "sys"},
                {"role": "user", "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64,QUJD"}}]}]

    def test_legacy_mode_keeps_image_part(self):
        out = GhostAgent._compose_injection(
            self._msgs(), "STABLE_BLOCK", "DYNAMIC_STATE", pin=False)
        c = out[-1]["content"]
        assert isinstance(c, list)
        assert any(p.get("type") == "image_url" for p in c)
        assert "base64" not in json.dumps(
            [p for p in c if p.get("type") == "text"][0]["text"])[:200]

    def test_pin_mode_keeps_image_part(self):
        out = GhostAgent._compose_injection(
            self._msgs(), "STABLE_BLOCK", "DYNAMIC_STATE", pin=True)
        first_user = next(m for m in out if m["role"] == "user")
        c = first_user["content"]
        assert isinstance(c, list)
        assert c[0]["type"] == "text" and "STABLE_BLOCK" in c[0]["text"]
        assert any(p.get("type") == "image_url" for p in c)


# ── B6: sampler respects a zero budget ───────────────────────────────────────

class TestSamplerRespectsLockdown:
    @pytest.mark.asyncio
    async def test_data_shaped_file_refused_under_lockdown(self, tmp_path):
        from ghost_agent.tools.file_system import tool_read_file
        gen = tmp_path / "table.h"
        gen.write_text(("0x1F, " * 60 + "\n") * 800)   # ~100KB data table
        out = await tool_read_file(
            "table.h", sandbox_dir=tmp_path, max_context=240000,
            read_budget=ReadBudget(0))
        assert str(out).startswith("Error")
        assert "SAMPLE ONLY" not in str(out)
