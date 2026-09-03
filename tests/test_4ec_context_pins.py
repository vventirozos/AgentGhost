"""§4EC — context-slice survivors of the §R re-verification of §4CA (2026-09-02):
`_cap_oversized_tail` (cutter / selector / sum branch / native-args stub, driven
as the static method it is) and `ContextManager`'s level, role and summary
rules (never pinned; §4CA touched only the newest-user rule)."""
import json

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.context_manager import ContextManager


def _msgs(*extra):
    return [{"role": "system", "content": "sys"},
            {"role": "user", "content": "goal: do the thing"},
            *extra,
            {"role": "user", "content": "newest question"}]


def _texts(m):
    return [p["text"] for p in m["content"] if isinstance(p, dict) and isinstance(p.get("text"), str)]


class TestTailCapCutter:
    def test_many_small_parts_are_cut_to_a_4k_head_plus_marker(self):
        """L5528-5545: no single part > 4000, sum > 8000 → keep ~4000 chars of
        head parts, drop the rest behind ONE marker part."""
        parts = [{"type": "text", "text": f"p{i}:" + "x" * 1500} for i in range(8)]   # 12,032 chars
        m = {"role": "assistant", "content": parts}
        GhostAgent._cap_oversized_tail(_msgs(m), max_tokens=800)
        kept = _texts(m)
        assert kept[-1].startswith("[...") and "dropped" in kept[-1]
        assert 2 <= len(kept) - 1 <= 3, kept          # ~4000 chars of head parts survive
        assert kept[0].startswith("p0:")

    def test_oversized_native_arguments_get_a_600_char_head_stub(self):
        """L5546-5561: the LARGEST arguments string over 4000 is replaced by a
        JSON stub keeping a 600-char head; a small one is untouched."""
        m = {"role": "assistant", "content": "",
             "tool_calls": [{"id": "a", "function": {"name": "t", "arguments": "s" * 100}},
                            {"id": "b", "function": {"name": "t", "arguments": "B" * 5000}},
                            {"id": "c", "function": {"name": "t", "arguments": "C" * 4500}}]}
        GhostAgent._cap_oversized_tail(_msgs(m), max_tokens=800)
        stub = json.loads(m["tool_calls"][1]["function"]["arguments"])
        assert stub["head"] == "B" * 600 and "5,000 chars" in stub["_dropped_by_context_budget"]
        assert m["tool_calls"][0]["function"]["arguments"] == "s" * 100
        # the second-largest is cut on a LATER iteration, never before the largest
        assert m["tool_calls"][2]["function"]["arguments"] in ("C" * 4500,) or "head" in m["tool_calls"][2]["function"]["arguments"]

    def test_a_4001_char_argument_is_cut_and_a_4000_char_one_is_not(self):
        m = {"role": "assistant", "content": "",
             "tool_calls": [{"id": "a", "function": {"name": "t", "arguments": "A" * 4000}},
                            {"id": "b", "function": {"name": "t", "arguments": "B" * 4001}}]}
        GhostAgent._cap_oversized_tail(_msgs(m), max_tokens=100)
        assert m["tool_calls"][0]["function"]["arguments"] == "A" * 4000
        assert "head" in m["tool_calls"][1]["function"]["arguments"]

    def test_the_largest_cuttable_message_is_cut_first(self):
        """L5605: selector picks max `_max_cuttable`; with a budget that one cut
        satisfies, the smaller candidate must be untouched."""
        small = {"role": "assistant", "content": "s" * 5000}
        big = {"role": "assistant", "content": "b" * 9000}
        GhostAgent._cap_oversized_tail(_msgs(small, big), max_tokens=3000)
        assert small["content"] == "s" * 5000
        assert len(big["content"]) < 9000

    def test_max_cuttable_reads_parts_and_native_arguments_like_the_cutter(self):
        """L5564-5592 mirrors the cutter: one part > 4000 → that part; else the
        SUM when > 8000; native arguments > 4000 count; a str under 4000 is 0."""
        one_big = {"role": "assistant", "content": [{"type": "text", "text": "x" * 6000}, {"type": "text", "text": "y" * 100}]}
        many_small = {"role": "assistant", "content": [{"type": "text", "text": "x" * 3000}] * 3}
        under = {"role": "assistant", "content": [{"type": "text", "text": "x" * 3000}] * 2}
        args = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "t", "arguments": "a" * 7000}}]}
        short = {"role": "assistant", "content": "s" * 3999}
        # drive through the cap: whoever the selector cuts first reveals its ranking
        GhostAgent._cap_oversized_tail(_msgs(under, many_small, one_big, args, short), max_tokens=2500)
        assert short["content"] == "s" * 3999                  # 0 → never a candidate
        assert _texts(under) == ["x" * 3000, "x" * 3000]       # sum 6000 ≤ 8000 → never a candidate
        assert len(_texts(one_big)[0]) < 6000                  # 6000 > 4000 → cut
        assert "head" in args["tool_calls"][0]["function"]["arguments"]   # 7000 → cut
        assert _texts(many_small)[-1].startswith("[...")       # sum 9000 > 8000 → sum branch


class TestContextManagerLevels:
    @pytest.mark.parametrize("ratio,level", [(0.59, 0), (0.60, 1), (0.74, 1), (0.75, 2), (0.84, 2), (0.85, 3), (0.94, 3), (0.95, 4)])
    def test_level_by_ratio(self, ratio, level):
        cm = ContextManager(max_tokens=1000, token_estimator=lambda msgs: int(ratio * 1000))
        cm.compress_if_needed([{"role": "user", "content": "x"}])
        assert cm.compression_level == level

    def test_max_level_caps_the_level(self):
        cm = ContextManager(max_tokens=1000, token_estimator=lambda msgs: 990)
        cm.compress_if_needed([{"role": "user", "content": "x"}], max_level=3)
        assert cm.compression_level == 3


class TestCompressMessageRules:
    def _cm(self):
        return ContextManager(max_tokens=1000)

    def test_tool_output_is_summarised_from_level_1(self):
        body = "\n".join(["header"] * 3 + [f"line {i} " + "." * 40 for i in range(20)] + ["error: boom"] + ["tail"] * 3)   # > 500 chars
        msg = {"role": "tool", "name": "execute", "content": body}
        out = self._cm()._compress_message(msg, 1)
        assert "lines compressed" in out["content"] and "error: boom" in out["content"]
        assert self._cm()._compress_message(msg, 0) is msg

    def test_long_assistant_prose_is_truncated_from_level_2_unless_it_carries_a_tool_call(self):
        cm = self._cm()
        prose = {"role": "assistant", "content": "p" * 3000}
        assert cm._compress_message(prose, 1) is prose
        assert len(cm._compress_message(prose, 2)["content"]) < 3000
        call = {"role": "assistant", "content": "<tool_call>" + "p" * 3000 + "</tool_call>"}
        assert cm._compress_message(call, 2) is call
        short = {"role": "assistant", "content": "p" * 2000}
        assert cm._compress_message(short, 2) is short          # not > 2000

    def test_long_user_text_is_truncated_only_from_level_3(self):
        cm = self._cm()
        user = {"role": "user", "content": "u" * 1001}
        assert cm._compress_message(user, 2) is user
        out = cm._compress_message(user, 3)
        assert out["content"].startswith("u" * 500) and "truncated" in out["content"]
        assert cm._compress_message({"role": "user", "content": "u" * 1000}, 3)["content"] == "u" * 1000

    def test_non_string_content_is_never_touched(self):
        msg = {"role": "tool", "content": [{"type": "text", "text": "x" * 5000}]}
        assert self._cm()._compress_message(msg, 3) is msg


class TestSummarizeToolOutput:
    def _cm(self):
        return ContextManager(max_tokens=1000)

    def test_short_or_few_line_output_is_kept(self):
        cm = self._cm()
        short = {"role": "tool", "content": "x" * 500}
        assert cm._summarize_tool_output(short) is short
        few = {"role": "tool", "content": "\n".join(["y" * 100] * 10)}      # > 500 chars, 10 lines
        assert cm._summarize_tool_output(few) is few

    def test_shape_is_head3_keyword_lines_marker_tail3(self):
        cm = self._cm()
        lines = [f"h{i}" for i in range(3)] + [f"m{i} " + "." * 60 for i in range(20)] + ["WARNING: careful"] + [f"t{i}" for i in range(3)]
        out = cm._summarize_tool_output({"role": "tool", "name": "x", "content": "\n".join(lines)})["content"].split("\n")
        assert out[:3] == ["h0", "h1", "h2"] and out[-3:] == ["t0", "t1", "t2"]
        assert "WARNING: careful" in out and f"[... {len(lines) - 6} lines compressed]" in out
        assert not any(l.startswith("m1 ") for l in out)

    def test_cache_hit_returns_the_stored_summary(self):
        cm = self._cm()
        body = "\n".join([f"h{i}" for i in range(3)] + ["." * 60] * 20 + ["t"] * 3)
        msg = {"role": "tool", "name": "x", "content": body}
        first = cm._summarize_tool_output(msg)["content"]
        key = next(iter(cm._summaries_cache))
        cm._summaries_cache[key] = "CACHED SUMMARY"
        assert cm._summarize_tool_output(msg)["content"] == "CACHED SUMMARY"

    def test_a_summary_that_saves_under_20_percent_is_not_used(self):
        cm = self._cm()
        # every middle line carries a keyword → nothing is dropped → summary ≥ 80% → original kept
        body = "\n".join([f"h{i}" for i in range(3)] + ["result " + "." * 40] * 20 + ["t"] * 3)
        msg = {"role": "tool", "name": "x", "content": body}
        assert cm._summarize_tool_output(msg) is msg


class TestContextManagerEdges:
    def test_recovery_below_60_percent_resets_the_level(self):
        """L83 `self._compression_level = 0` deleted: a request that recovers
        would stay compressed at the old level forever."""
        est = {"v": 990}
        cm = ContextManager(max_tokens=1000, token_estimator=lambda msgs: est["v"])
        cm.compress_if_needed([{"role": "user", "content": "x"}])
        assert cm.compression_level == 4
        est["v"] = 100
        cm.compress_if_needed([{"role": "user", "content": "x"}])
        assert cm.compression_level == 0

    def test_a_short_conversation_is_never_compressed_at_level_1(self):
        """L139 `len(conv_msgs) <= keep_full`: six or fewer conversation turns
        stay at full fidelity."""
        cm = ContextManager(max_tokens=1000, token_estimator=lambda msgs: 700)   # ratio 0.7 → L1
        long_tool = {"role": "tool", "name": "x", "content": "\n".join(["r " + "." * 60] * 20)}
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, long_tool,
                {"role": "assistant", "content": "a"}]
        assert cm.compress_if_needed(msgs) == msgs
        assert msgs[2]["content"] == long_tool["content"]

    def test_an_assistant_message_with_only_a_closing_tool_call_tag_is_kept(self):
        cm = ContextManager(max_tokens=1000)
        msg = {"role": "assistant", "content": "p" * 3000 + "</tool_call>"}
        assert cm._compress_message(msg, 2) is msg

    def test_a_short_tool_output_is_kept_however_many_lines_it_has(self):
        cm = ContextManager(max_tokens=1000)
        msg = {"role": "tool", "name": "x", "content": "\n".join(["l" * 20] * 12)}   # 251 chars, 12 lines
        assert cm._summarize_tool_output(msg) is msg

    def test_the_summary_cache_is_bounded_and_lru(self):
        cm = ContextManager(max_tokens=1000); cm._summaries_cache_max = 2
        def big(tag):
            return {"role": "tool", "name": "x", "content": "\n".join([f"{tag}{i}" for i in range(3)] + ["." * 60] * 20 + ["t"] * 3)}
        cm._summarize_tool_output(big("a")); cm._summarize_tool_output(big("b"))
        ka, kb = list(cm._summaries_cache)
        cm._summarize_tool_output(big("a"))                 # hit → bump a to the tail
        cm._summarize_tool_output(big("c"))                 # insert c → evict the oldest, which is now b
        assert len(cm._summaries_cache) == 2
        assert ka in cm._summaries_cache and kb not in cm._summaries_cache

    def test_emergency_prune_keeps_system_last_user_and_last_tool_only(self):
        big = "z" * 1500
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "u1"}, {"role": "tool", "name": "t", "content": "t1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"}, {"role": "tool", "name": "t", "content": big},
                {"role": "assistant", "content": "a2"}]
        out = ContextManager._emergency_prune(msgs)
        assert [m["role"] for m in out] == ["system", "user", "tool"]
        assert out[1]["content"] == "u2"
        assert out[2]["content"].startswith("z" * 1000) and "EMERGENCY TRUNCATED" in out[2]["content"]
        assert len(out[2]["content"]) < 1500

    def test_emergency_prune_finds_the_last_user_after_the_last_tool(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"},
                {"role": "tool", "name": "t", "content": "t1"}, {"role": "user", "content": "u2"}]
        out = ContextManager._emergency_prune(msgs)
        assert [m["content"] for m in out] == ["s", "u2", "t1"]

    def test_emergency_prune_without_a_tool_keeps_system_and_last_user(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a"}]
        assert [m["content"] for m in ContextManager._emergency_prune(msgs)] == ["s", "u1"]

    def test_level_4_routes_to_the_emergency_prune(self):
        cm = ContextManager(max_tokens=1000, token_estimator=lambda msgs: 990)
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u1"},
                {"role": "tool", "name": "t", "content": "t1"}, {"role": "assistant", "content": "a"},
                {"role": "user", "content": "u2"}]
        assert [m["content"] for m in cm.compress_if_needed(msgs)] == ["s", "u2", "t1"]


# ── the cutter's <tool_call> branch (§4CA R1 F2 / R2 M5 / R3 V3) ─────────────
class TestTailCapToolCallBlocks:
    def _one(self, content, max_tokens=800):
        m = {"role": "assistant", "content": content}
        GhostAgent._cap_oversized_tail(_msgs(m), max_tokens=max_tokens)
        return m["content"]

    def test_an_oversized_closed_block_is_removed_whole(self):
        out = self._one("intro\n<tool_call>" + "x" * 5000 + "</tool_call>\nafter")
        assert "<tool_call>" not in out and "</tool_call>" not in out
        assert "tool_call block (5,023 chars) dropped" in out and out.startswith("intro") and out.endswith("after")

    def test_an_unclosed_block_over_2000_chars_is_dropped_from_its_opener(self):
        out = self._one("intro\n<tool_call>" + "x" * 4500)
        assert "unclosed tool_call" in out and out.startswith("intro") and "xxxx" not in out

    def test_a_short_unclosed_block_is_left_and_the_prose_is_cut_instead(self):
        prose = "p" * 6000
        out = self._one(prose + "\n<tool_call>" + "x" * 500)
        assert "<tool_call>" in out and out.endswith("x" * 500)          # the block stays whole
        assert len(out) < len(prose) + 520                                # the prose segment was cut

    def test_only_prose_segments_over_4000_are_cut_and_the_largest_first(self):
        small = "s" * 3000; big = "b" * 9000; blk = "<tool_call>" + "y" * 100 + "</tool_call>"
        out = self._one(small + blk + big + blk + "tail")
        assert small in out and blk in out and out.endswith("tail")
        assert "b" * 9000 not in out and "dropped by context budget" in out

    def test_a_message_with_blocks_but_no_cuttable_segment_is_refused_not_mangled(self):
        blk = "<tool_call>" + "y" * 100 + "</tool_call>"
        content = ("s" * 3000 + blk) * 2        # every prose segment ≤ 4000, every block ≤ 4000
        newest = "n" * 9000
        m = {"role": "assistant", "content": content}
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "goal"}, m,
                {"role": "user", "content": newest}]
        GhostAgent._cap_oversized_tail(msgs, max_tokens=1500)
        assert m["content"] == content                                     # refused, untouched
        assert len(msgs[-1]["content"]) < 9000                             # the last resort took the cut


class TestTailCapLastResorts:
    def test_the_newest_user_is_cut_last_with_the_paste_note(self):
        newest = {"role": "user", "content": "n" * 9000}
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "goal"},
                {"role": "assistant", "content": "short"}, newest]
        GhostAgent._cap_oversized_tail(msgs, max_tokens=1500)
        assert len(newest["content"]) < 9000 and "pasted" in newest["content"].lower()

    def test_the_goal_is_cut_only_after_the_newest_user(self):
        goal = {"role": "user", "content": "g" * 9000}
        newest = {"role": "user", "content": "n" * 9000}
        msgs = [{"role": "system", "content": "sys"}, goal, {"role": "assistant", "content": "a"}, newest]
        GhostAgent._cap_oversized_tail(msgs, max_tokens=1500)
        assert len(newest["content"]) < 9000 and len(goal["content"]) < 9000

    def test_a_non_str_text_part_is_cuttable_with_an_honest_stub(self):
        part = {"type": "text", "text": ["z"] * 3000}          # JSON-reachable, counted by the counter
        m = {"role": "assistant", "content": [part]}
        GhostAgent._cap_oversized_tail(_msgs(m), max_tokens=800)
        assert isinstance(part["text"], str) and "non-text part" in part["text"]


# ── _compose_injection placement (§4CA B4 neighbourhood) ─────────────────────
class TestInjectionPlacement:
    STABLE, DYN = "STABLE-CTX", "DYN-STATE"

    def test_pin_mode_single_user_message(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "ask"}]
        out = GhostAgent._compose_injection(msgs, self.STABLE, self.DYN, True)
        assert out[1]["content"].startswith("<session_context>") and "[USER INSTRUCTION]" in out[1]["content"] and out[1]["content"].endswith("ask")
        assert out[-1]["role"] == "user" and out[-1]["content"].startswith("<system_state_update>") and len(out) == 3

    def test_pin_mode_last_message_is_a_later_user(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "first"},
                {"role": "assistant", "content": "a"}, {"role": "user", "content": "latest"}]
        out = GhostAgent._compose_injection(msgs, self.STABLE, self.DYN, True)
        assert out[1]["content"].startswith("<session_context>") and len(out) == 4
        assert out[3]["content"].startswith("<system_state_update>") and out[3]["content"].endswith("latest")

    def test_pin_mode_last_message_is_a_tool_result(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "first"},
                {"role": "tool", "content": "r"}]
        out = GhostAgent._compose_injection(msgs, self.STABLE, self.DYN, True)
        assert len(out) == 4 and out[3]["role"] == "user" and out[3]["content"].startswith("<system_state_update>")

    def test_pin_mode_without_any_user_message_inserts_after_system(self):
        msgs = [{"role": "system", "content": "s"}]
        out = GhostAgent._compose_injection(msgs, self.STABLE, self.DYN, True)
        assert [m["role"] for m in out] == ["system", "user", "user"]
        assert out[1]["content"].startswith("<session_context>") and out[2]["content"].startswith("<system_state_update>")

    def test_legacy_mode_prefixes_the_last_user_or_appends(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "ask"}]
        out = GhostAgent._compose_injection(msgs, self.STABLE, self.DYN, False)
        assert len(out) == 2 and out[1]["content"].startswith("<system_state_update>") and "[USER INSTRUCTION]" in out[1]["content"] and out[1]["content"].endswith("ask")
        msgs2 = [{"role": "system", "content": "s"}, {"role": "user", "content": "ask"}, {"role": "tool", "content": "r"}]
        out2 = GhostAgent._compose_injection(msgs2, self.STABLE, self.DYN, False)
        assert len(out2) == 4 and out2[-1]["role"] == "user" and out2[-1]["content"].startswith("<system_state_update>")


# ── _prune_context summarisation path (§4CA R1 F3/F4 neighbourhood) ──────────
import asyncio
import json as _json
from tests.test_context_r1_fixes import _agent as _prune_agent


class TestPruneSummarisationPath:
    def _msgs(self, middle):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "the real goal"}]
        msgs += middle
        msgs += [{"role": "assistant", "content": "recent a"}, {"role": "user", "content": "recent q"},
                 {"role": "assistant", "content": "recent b"}, {"role": "user", "content": "recent q2"},
                 {"role": "assistant", "content": "recent c"}, {"role": "user", "content": "now do it"}]
        return msgs

    def test_image_parts_in_truncated_messages_reach_the_summariser_as_a_placeholder(self):
        """L5683-5690: a native image part must not be f-stringed into the
        summariser prompt as its base64 — it becomes the placeholder text."""
        middle = [{"role": "user", "content": [{"type": "text", "text": "look at this"},
                                               {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDREVGR0hJSktMTU5PUA=="}}]}]
        middle += [{"role": "assistant", "content": f"work {i} " + "y" * 200} for i in range(12)]
        a = _prune_agent()
        asyncio.run(a._prune_context(self._msgs(middle), max_tokens=100))
        ca = a.context.llm_client.chat_completion.call_args
        sent = _json.dumps([list(ca.args), dict(ca.kwargs)], default=str)
        assert "Image attached and passed to vision node" in sent
        assert "QUJDREVGR0hJSktMTU5PUA" not in sent

    def test_an_assistant_finding_in_the_middle_is_kept_as_an_anchor(self):
        """L5750: an assistant message carrying an anchor keyword survives the
        prune verbatim-prefixed with [ANCHORED]; plain chatter does not."""
        middle = [{"role": "assistant", "content": "traceback MARKER_FINDING the parser drops empty names " + "z" * 200}]
        middle += [{"role": "assistant", "content": f"chatter {i} MARKER_CHATTER " + "y" * 200} for i in range(12)]
        a = _prune_agent()
        out = asyncio.run(a._prune_context(self._msgs(middle), max_tokens=100))
        text = "\n".join(str(m.get("content")) for m in out)
        assert "MARKER_FINDING" in text and "[ANCHORED]" in text
        assert text.count("MARKER_CHATTER") < 12
