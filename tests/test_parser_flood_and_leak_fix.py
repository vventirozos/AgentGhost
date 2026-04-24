"""Tests for the self-play parser hardening and XML-leak fixes.

Four edits are covered:

1. (agent.py) `_MAX_TOOL_CALL_BLOCKS` caps how many `<tool_call>` openings
   the XML parser attempts per response. A degenerate Qwen stream with
   thousands of malformed blocks was observed amplifying into 4000+
   `system_parse_error` entries in a single turn.

2. (agent.py) In-loop strike cap. The outer turn-loop cap at
   `execution_failure_count >= 6` only fires at turn boundaries. Without
   an in-loop break, one over-stuffed `tool_calls` list drained the whole
   loop before the cap fired next turn.

3. (agent.py) Widened UI scrub regex. The previous regex caught only
   `<tool_call>` and `<tool>` shapes, so a bare `<function name="...">`
   block (no outer wrapper) leaked verbatim into the user-facing reply.
   The new pattern also handles `<function>` and uses a backreference so
   that an inner `</function>` inside `<tool_call>...</tool_call>` no
   longer terminates the outer match early.

4. (dream.py) Synthetic-challenge `<think>` scrub. The old regex
   `<think>.*?(?:</think>|$)` consumed everything from an unclosed
   `<think>` to end-of-string — including the required
   `<challenge_prompt>` / `<validation_script>` blocks — whenever the
   model forgot to emit `</think>` or nested the XML blocks inside a
   closed `<think>` wrapper. The fix strips only closed `<think>` blocks
   and adds a raw-content fallback for extraction.
"""

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_SRC = REPO_ROOT / "src" / "ghost_agent" / "core" / "agent.py"
DREAM_SRC = REPO_ROOT / "src" / "ghost_agent" / "core" / "dream.py"


# ---------------------------------------------------------------------------
# Edit 3 — Widened UI scrub regex (core/agent.py, ~line 2969)
# ---------------------------------------------------------------------------

# Mirror of the scrub regex. The source-level guard test below ensures the
# two stay in sync.
SCRUB_PATTERN = re.compile(
    r'<(tool_call|tool|function)\b[^>]*>.*?(?:</\1\b[^>]*>|$)',
    flags=re.DOTALL | re.IGNORECASE,
)


def _scrub(s: str) -> str:
    return SCRUB_PATTERN.sub('', s).strip()


class TestUIScrubWidenedRegex:
    def test_empty_tool_call_with_empty_function_is_fully_stripped(self):
        """The exact shape the user reported as leaking verbatim."""
        s = "<tool_call>\n<function name=\"self_play\">\n</function>\n</tool_call>"
        assert _scrub(s) == ""

    def test_bare_function_without_outer_tool_call_is_stripped(self):
        """The old scrub only caught `<tool_call>` / `<tool>`, so a bare
        `<function>` block (same shape the parser accepts via the
        function-name-as-attribute path) leaked into the reply."""
        s = '<function name="self_play">body</function>'
        assert _scrub(s) == ""

    def test_backreference_prevents_inner_function_from_terminating_outer(self):
        s = "<tool_call><function></function></tool_call>XYZ"
        # Without `\1` the inner `</function>` would match the close
        # alternative and leave an orphan `</tool_call>XYZ` behind.
        assert _scrub(s) == "XYZ"

    def test_unclosed_tool_call_runs_to_end_of_string(self):
        s = "prose before <tool_call>unclosed body runs off"
        assert _scrub(s) == "prose before"

    def test_prose_around_balanced_block_is_preserved(self):
        s = ("Sure, running it.\n"
             "<tool_call><function name=\"x\"></function></tool_call>\n"
             "All done!")
        out = _scrub(s)
        assert "Sure, running it." in out
        assert "All done!" in out
        assert "tool_call" not in out
        assert "function" not in out

    def test_plain_prose_is_unchanged(self):
        s = "Plain reply with no XML."
        assert _scrub(s) == s

    def test_case_insensitive(self):
        s = '<TOOL_CALL><FUNCTION name="x"></FUNCTION></TOOL_CALL>'
        assert _scrub(s) == ""

    def test_function_with_attributes_and_whitespace(self):
        s = '<function   name="self_play"   extra="attr" >body</function>'
        assert _scrub(s) == ""

    def test_source_contains_widened_regex(self):
        """Guard against a refactor silently reverting to the old
        `(?:tool_call|tool)` pattern or dropping the backreference."""
        src = AGENT_SRC.read_text()
        assert r"<(tool_call|tool|function)\b" in src
        assert r"</\1\b" in src

    def test_end_of_handle_chat_scrub_matches_mid_flow_scrub(self):
        """Symmetry guard: the last-resort scrub on `final_ai_content`
        (end of handle_chat) must be at least as strict as the mid-flow
        scrub on `ui_content`. Otherwise a bare `<function>...</function>`
        that bypasses the mid-flow branch (e.g. via the perfect-it
        follow-up or a non-has_tool_tag path) leaks verbatim — exactly
        the shape the user reported as showing up as a literal XML
        reply after a second `run self play` invocation."""
        src = AGENT_SRC.read_text()
        # Both occurrences of the widened pattern must be present —
        # one for ui_content, one for final_ai_content. Both use `\Z`
        # (absolute EOS) so a trailing `\n` after a tool_call doesn't
        # escape the scrub.
        count = src.count(r"<(tool_call|tool|function)\b[^>]*>.*?(?:</\1\b[^>]*>|\Z)")
        assert count >= 2, (
            f"Expected the widened scrub pattern at both the mid-flow "
            f"and end-of-handle_chat sites; found {count}"
        )
        # Belt-and-suspenders: the OLD narrow pattern
        # `<(?:tool_call|tool)\b...` must not appear in agent.py anymore
        # — any remaining use would be a regression hotspot.
        assert r"<(?:tool_call|tool)\b.*?>" not in src, (
            "Old narrow scrub pattern still present. It misses bare "
            "<function> blocks and nested </function> cases — replace "
            "with the widened `(tool_call|tool|function)` + backreference."
        )
        # And the OLD `|$)` form on the widened pattern must also be
        # gone — it was the source of the trailing-newline leak.
        assert r"<(tool_call|tool|function)\b[^>]*>.*?(?:</\1\b[^>]*>|$)" not in src, (
            "Widened scrub pattern is still using `$` as the missing-"
            "close alternative. In non-MULTILINE mode `$` matches just "
            "before a trailing `\\n`, letting the newline escape. Use "
            "`\\Z` (absolute EOS) instead."
        )


class TestEphemeralTerminalDirectiveRetired:
    """The `[EPHEMERAL_TERMINAL_DIRECTIVE]` injection + sweep pair was
    retired once terminal tools started bypassing the summary LLM turn
    entirely (DIRECT-FROM-TOOL SUMMARY). With no LLM summary call there
    is no need to coach the model with a text-only directive, and with
    no directive being appended there is no need to sweep one out.

    These tests pin the retirement so a future refactor that
    reintroduces the directive without also re-adding the sweep (or
    vice versa) fails loudly."""

    def test_injection_site_is_gone(self):
        src = AGENT_SRC.read_text()
        # The literal injection line must not reappear.
        assert 'messages.append({\n                                    "role": "user",\n                                    "content": (\n                                        "[EPHEMERAL_TERMINAL_DIRECTIVE]' not in src

    def test_sweep_loop_is_gone(self):
        src = AGENT_SRC.read_text()
        # The executable sweep list-comprehension must be gone. A
        # surviving comment referring to the retired mechanism is
        # fine (and helpful for archaeology).
        assert "EPHEMERAL DIRECTIVE SWEEP" not in src
        assert "_ephemeral_before = len(messages)" not in src


class TestStreamingScrubOnFinalGeneration:
    """The non-streaming path's scrubs cannot protect the client from
    raw upstream chunks that go through the streaming path. When
    `is_final_generation` is set (post-terminal-tool summary turn, or
    planner-says-no-action turn), any `<tool_call>` / `<function>` /
    `<tool_response>` in the stream must be stripped BEFORE the bytes
    reach the client. Trace: user reported
    `<tool_call><function name=\"self_play\"></function></tool_call>`
    showing up as the assistant's literal reply on a second
    `run self play` — that text was yielded raw through
    `agent.py:~2182` by the pre-fix code."""

    def test_source_uses_scrubbed_streaming_on_final_generation(self):
        src = AGENT_SRC.read_text()
        # The gating flag must key on `is_final_generation` — the same
        # signal force_final_response / planner=no-action set.
        assert "_stream_scrub_active = bool(is_final_generation)" in src
        # The pattern must cover all four leak shapes (tool_call, tool,
        # function, tool_response) with the backreference, AND use `\Z`
        # (absolute end of string) as the missing-close-tag alternative
        # so trailing newlines don't escape the scrub. The `$`
        # alternative — which matches both EOS and "just before final
        # `\n`" — allowed a single `\n` to slip through, which was what
        # the user observed as a "blank" reply.
        assert (
            r"<(tool_call|tool|function|tool_response)\b[^>]*>.*?"
        ) in src
        assert r"(?:</\1\b[^>]*>|\Z)" in src
        # Belt-and-suspenders: the OLD `$` alternative must NOT appear
        # alongside the stream-scrub pattern any more.
        assert "</\\1\\b[^>]*>|$)" not in src.replace("|\\Z)", "")

    def test_source_emits_scrubbed_delta_only(self):
        """The scrub branch must emit only the NEW portion of the
        scrubbed view so that, as a tool_call block gets absorbed by
        the scrub, nothing ships to the client."""
        src = AGENT_SRC.read_text()
        assert "_scrubbed_view = _stream_scrub_pattern.sub('', full_content)" in src
        assert "_to_emit = _scrubbed_view[_scrubbed_emitted_len:]" in src
        assert "_scrubbed_emitted_len = len(_scrubbed_view)" in src

    def test_source_bypasses_raw_yield_on_scrub_path(self):
        """When the scrub path is active for a content chunk, the code
        must NOT also `yield chunk` (the raw upstream bytes) — that
        would double-emit, with the raw form reaching the client."""
        src = AGENT_SRC.read_text()
        # The non-scrub `yield chunk` sits in the `else` branch
        # matching `_stream_scrub_active and _is_content_chunk`.
        assert "if _stream_scrub_active and _is_content_chunk:" in src
        # Ordering check: the synthetic-emit block must come before
        # the `else: ... yield chunk` pass-through.
        scrub_branch = src.find("if _stream_scrub_active and _is_content_chunk:")
        else_branch = src.find(
            "# Non-scrub path (mid-tool-loop stream,"
        )
        assert 0 < scrub_branch < else_branch

    def test_scrub_stream_emits_only_clean_text_for_leaked_xml(self):
        """Behavioral: simulate a streaming response that leaks a full
        tool_call block embedded in prose. The scrubbed-view strategy
        (compute `scrubbed = pattern.sub('', full_content)` on every
        new chunk, emit only the growing delta) must leave the client
        with only the prose, never the XML."""
        import re as _re

        pattern = _re.compile(
            r'<(tool_call|tool|function|tool_response)\b[^>]*>.*?'
            r'(?:</\1\b[^>]*>|$)',
            flags=_re.DOTALL | _re.IGNORECASE,
        )

        # Simulate the exact shape the user saw leak, split across
        # chunks as a streaming upstream would deliver it.
        simulated_chunks = [
            "The self-play session ",
            "finished successfully. ",
            "<tool_call>\n",           # leak starts — scrubbed view stops growing
            "<function name=\"self_play\">\n",
            "</function>\n",
            "</tool_call>",            # leak ends — the whole block drops out
            "\nReady for next command!",
        ]

        full = ""
        emitted_parts = []
        emitted_len = 0
        for ch in simulated_chunks:
            full += ch
            scrubbed = pattern.sub('', full)
            new = scrubbed[emitted_len:]
            if new:
                emitted_parts.append(new)
                emitted_len = len(scrubbed)

        emitted = "".join(emitted_parts)
        # User should see clean prose, no XML.
        assert "<tool_call" not in emitted
        assert "<function" not in emitted
        assert "</tool_call" not in emitted
        assert "</function" not in emitted
        # The legitimate prose on either side must survive.
        assert "finished successfully" in emitted
        assert "Ready for next command" in emitted

    def test_scrub_stream_handles_unclosed_tool_call(self):
        """Pathological case: the model starts `<tool_call>` but the
        stream ends before the close tag arrives. The scrub's `|$`
        alternative absorbs the unclosed tail, so the client sees
        only the clean prefix — never the partial XML."""
        import re as _re
        pattern = _re.compile(
            r'<(tool_call|tool|function|tool_response)\b[^>]*>.*?'
            r'(?:</\1\b[^>]*>|$)',
            flags=_re.DOTALL | _re.IGNORECASE,
        )

        simulated_chunks = [
            "All done. ",
            "<tool_call>",                  # leak starts
            "<function name=\"x\">",
            # stream ends — no close tag, no closing }
        ]

        full = ""
        emitted_len = 0
        emitted = ""
        for ch in simulated_chunks:
            full += ch
            scrubbed = pattern.sub('', full)
            new = scrubbed[emitted_len:]
            emitted += new
            emitted_len = len(scrubbed)

        # Unclosed tool_call at end of stream → whole tail absorbed by
        # the `|$` alternative. User sees only the prefix.
        assert emitted == "All done. "

    def test_scrub_stream_is_disabled_when_not_final_generation(self):
        """Regression guard — mid-tool-loop streams must still yield
        raw chunks so the downstream tool-call parser sees the real
        XML. Scrub only activates when the turn is a final-generation
        text-only summary."""
        src = AGENT_SRC.read_text()
        # The boolean MUST be bound to is_final_generation. A bug
        # that makes it always-on would silently prevent the agent
        # from ever executing a tool via the streaming path.
        assert "_stream_scrub_active = bool(is_final_generation)" in src
        # And the else branch must still say "yield chunk" for the
        # non-scrub path.
        assert (
            "# Non-scrub path (mid-tool-loop stream,"
            in src
        )

    def test_scrub_empty_output_fallback_wired(self):
        """When the scrub swallows EVERY content chunk (i.e. the
        upstream response was pure <tool_call> XML on a turn the
        planner routed as text-only), the client would otherwise see
        a blank reply. The wrapper must emit a synthetic fallback
        message so the user gets SOMETHING actionable.

        This was the user's third-`self play` symptom: empty reply
        after the previous streaming scrub landed. The fallback
        converts it to 'I prepared a tool call but ... please
        rephrase' — visible and useful."""
        src = AGENT_SRC.read_text()
        # Dedicated section heading so the fallback is greppable.
        assert "SCRUBBED-STREAM EMPTY-OUTPUT FALLBACK" in src
        # The guard must check that (a) the scrub was active, (b) the
        # upstream emitted content, (c) nothing past the prefix reached
        # the client.
        assert "_stream_scrub_active" in src
        assert "_scrubbed_emitted_len <= len(stream_prefix)" in src
        # The fallback must mention the intended tool when it can
        # extract the name — that's the most useful signal for the
        # user to rephrase with.
        assert "Intended tool:" in src
        # And the fallback SSE chunk must be yielded.
        assert "_fallback_chunk = {" in src

    def test_fallback_extracts_intended_tool_name(self):
        """Behavioral: simulate the condition that triggers the
        fallback (full_content contains a tool_call, nothing after
        scrubbing) and verify that the intended-tool extractor
        pulls the function name out correctly."""
        import re as _re
        # Replica of the extraction regex used in the source.
        pat = _re.compile(
            r'<function(?:\s+name=|=)\s*["\']?([a-zA-Z0-9_]+)',
            _re.IGNORECASE,
        )
        # Shapes the model actually emits.
        cases = [
            ('<tool_call>\n<function name="self_play">\n</function>\n</tool_call>', "self_play"),
            ('<function name=\'deep_research\'>x</function>', "deep_research"),
            ('<tool_call><function name=execute></function></tool_call>', "execute"),
        ]
        for content, expected in cases:
            m = pat.search(content)
            assert m, f"no match in {content!r}"
            assert m.group(1) == expected, f"{content!r} -> {m.group(1)!r}, wanted {expected!r}"

    def test_fallback_not_fired_when_normal_text_was_emitted(self):
        """Regression guard: if the stream produced real text content
        (e.g., a healthy summary), the fallback branch must NOT fire
        — otherwise we'd append the fallback message to a perfectly
        good reply."""
        # Mirror the source's condition gate as a pure predicate and
        # exercise it against a healthy-stream trace.
        def _would_fire(full_content: str, stream_prefix: str, emitted_len: int) -> bool:
            return (
                True  # _stream_scrub_active — assumed for this test
                and bool(full_content.strip())
                and len(full_content.strip()) > len(stream_prefix.strip())
                and emitted_len <= len(stream_prefix)
            )

        # Healthy case: model emitted a summary, scrub didn't strip it,
        # plenty of content reached the client.
        healthy_full = "Self-play completed. I tackled a SQLite challenge and passed."
        healthy_prefix = ""
        healthy_emitted = len(healthy_full)
        assert _would_fire(healthy_full, healthy_prefix, healthy_emitted) is False

        # Pathological case: upstream emitted a pure tool_call, scrub
        # ate everything, emitted_len stayed at the prefix length.
        bad_full = "<tool_call>\n<function name=\"self_play\">\n</function>\n</tool_call>"
        bad_prefix = ""
        bad_emitted = 0
        assert _would_fire(bad_full, bad_prefix, bad_emitted) is True

        # Empty upstream (no content at all): fallback should NOT fire
        # — there's no tool_call to surface.
        empty_full = ""
        assert _would_fire(empty_full, "", 0) is False


# ---------------------------------------------------------------------------
# Edit 1 — Parser block cap (core/agent.py, ~line 2568)
# ---------------------------------------------------------------------------


class TestParserBlockCap:
    """Bounds the blocks list regardless of how many `<tool_call>`
    openings the model emitted. The constant (5) is implementation-
    defined; this test encodes the behavior we require — a degenerate
    response does not produce thousands of parse attempts."""

    MAX_TOOL_CALL_BLOCKS = 5

    def _split_and_cap(self, parse_target: str):
        blocks = re.split(r'<tool_call.*?>', parse_target, flags=re.IGNORECASE)
        if len(blocks) > self.MAX_TOOL_CALL_BLOCKS + 1:
            blocks = blocks[: self.MAX_TOOL_CALL_BLOCKS + 1]
        return blocks

    def test_degenerate_response_with_thousands_of_openings_is_capped(self):
        # 4056 is the observed real-world pathological case.
        parse_target = "<tool_call>garbage" * 4056
        blocks = self._split_and_cap(parse_target)
        assert len(blocks) == self.MAX_TOOL_CALL_BLOCKS + 1
        # And therefore at most 5 tool_call bodies are parsed.
        assert len(blocks) - 1 == self.MAX_TOOL_CALL_BLOCKS

    def test_five_openings_is_not_truncated(self):
        parse_target = "<tool_call>x" * self.MAX_TOOL_CALL_BLOCKS
        blocks = self._split_and_cap(parse_target)
        # re.split with N matches → N+1 elements; cap trips only at len > 6.
        assert len(blocks) == self.MAX_TOOL_CALL_BLOCKS + 1

    def test_six_openings_is_truncated_to_six_elements(self):
        parse_target = "<tool_call>x" * (self.MAX_TOOL_CALL_BLOCKS + 1)
        blocks = self._split_and_cap(parse_target)
        assert len(blocks) == self.MAX_TOOL_CALL_BLOCKS + 1

    def test_three_openings_well_under_cap_is_full(self):
        parse_target = ("<tool_call>one</tool_call>"
                        "<tool_call>two</tool_call>"
                        "<tool_call>three</tool_call>")
        blocks = self._split_and_cap(parse_target)
        assert len(blocks) == 4  # prefix + 3 splits

    def test_no_tool_call_openings_yields_single_block(self):
        parse_target = "just prose, no tool calls here"
        blocks = self._split_and_cap(parse_target)
        assert len(blocks) == 1
        assert blocks[0] == parse_target

    def test_source_enforces_the_cap(self):
        src = AGENT_SRC.read_text()
        assert "_MAX_TOOL_CALL_BLOCKS" in src
        assert "blocks = blocks[: _MAX_TOOL_CALL_BLOCKS + 1]" in src


# ---------------------------------------------------------------------------
# Edit 2 — In-loop strike cap (core/agent.py, ~line 3153)
# ---------------------------------------------------------------------------


class TestInLoopStrikeCap:
    """Replicates the per-tool loop's break-on-cap behavior. Each tool is
    treated as a `system_parse_error` (so it increments the counter), which
    is the worst case the in-loop cap was added to defend against."""

    STRIKE_CAP = 6

    def _simulate_tool_loop(self, n_tools: int):
        execution_failure_count = 0
        processed = 0
        tool_calls = list(range(n_tools))
        for _tc_idx, _tool in enumerate(tool_calls):
            # Exact structure of the in-loop cap check in agent.py.
            if execution_failure_count >= self.STRIKE_CAP:
                break
            execution_failure_count += 1
            processed += 1
        return processed, execution_failure_count

    def test_4056_failures_cap_at_six(self):
        processed, count = self._simulate_tool_loop(4056)
        assert processed == self.STRIKE_CAP
        assert count == self.STRIKE_CAP

    def test_three_failures_all_processed(self):
        processed, count = self._simulate_tool_loop(3)
        assert processed == 3
        assert count == 3

    def test_exactly_cap_failures_all_processed(self):
        processed, count = self._simulate_tool_loop(self.STRIKE_CAP)
        assert processed == self.STRIKE_CAP
        assert count == self.STRIKE_CAP

    def test_one_over_cap_skips_the_seventh(self):
        processed, count = self._simulate_tool_loop(self.STRIKE_CAP + 1)
        # The 7th iteration's cap check fires before its increment.
        assert processed == self.STRIKE_CAP
        assert count == self.STRIKE_CAP

    def test_zero_tools_is_noop(self):
        processed, count = self._simulate_tool_loop(0)
        assert processed == 0
        assert count == 0

    def test_source_contains_in_loop_cap(self):
        src = AGENT_SRC.read_text()
        assert "for _tc_idx, tool in enumerate(tool_calls):" in src
        assert "if execution_failure_count >= 6:" in src
        assert "Strike cap hit mid-loop" in src


# ---------------------------------------------------------------------------
# Edit 5 — Truncation-error dedupe (core/agent.py, ~line 2946)
#
# When upstream cuts the response off mid-tool_call, re.split on
# `<tool_call>` produces N fragments that ALL fail parsing for the same
# reason ("truncated"). Before this edit, each fragment appended its own
# `system_parse_error` — so a single truncation event counted as up to 5
# strikes (because of the block-cap) on the execution counter, firing the
# truncation-specific recovery hint 5 times and burning through the
# model's strike budget in one turn.
#
# The dedupe emits at most one truncation-reason `system_parse_error`
# per response, while still letting genuinely distinct failures
# (no_function_tag, malformed) emit their own errors.
# ---------------------------------------------------------------------------


class TestTruncationErrorDedupe:
    """Simulate the block-iteration dedupe logic. Each element in the
    `parse_reasons` list represents the failure reason a single fragment
    would produce; the simulation counts how many `system_parse_error`
    entries are emitted after dedupe."""

    def _simulate_block_loop(self, parse_reasons):
        emitted_truncation_error = False
        emitted = []
        for reason in parse_reasons:
            if reason is None:
                # Block parsed successfully — no system_parse_error.
                continue
            if reason == "truncated" and emitted_truncation_error:
                continue
            if reason == "truncated":
                emitted_truncation_error = True
            emitted.append(reason)
        return emitted

    def test_five_truncated_fragments_collapse_to_one(self):
        # This is the exact shape the user's log captured:
        #   consecutive=1..5, reason=truncated.
        emitted = self._simulate_block_loop(["truncated"] * 5)
        assert emitted == ["truncated"]

    def test_mixed_failures_are_not_deduped(self):
        # Only truncated entries dedupe — no_function_tag and malformed
        # are independent failure modes.
        emitted = self._simulate_block_loop([
            "truncated", "truncated", "no_function_tag", "malformed", "truncated",
        ])
        # One truncated + the two distinct failures = 3 entries.
        assert emitted == ["truncated", "no_function_tag", "malformed"]

    def test_valid_call_before_truncation_is_preserved(self):
        # First fragment parses cleanly; later fragments are truncation
        # junk. We still want the valid call executed and ONE truncation
        # error surfaced.
        emitted = self._simulate_block_loop([None, "truncated", "truncated"])
        assert emitted == ["truncated"]

    def test_all_blocks_clean_emits_no_errors(self):
        emitted = self._simulate_block_loop([None, None, None])
        assert emitted == []

    def test_single_truncation_still_emits_one(self):
        # Regression guard: dedupe must not suppress the legitimate
        # single-fragment case.
        emitted = self._simulate_block_loop(["truncated"])
        assert emitted == ["truncated"]

    def test_no_function_tag_still_emits_every_time(self):
        # These represent genuinely different broken blocks and should
        # NOT be deduped — a future extension might want to dedupe them
        # too, but for now we only claim the property for "truncated".
        emitted = self._simulate_block_loop(["no_function_tag", "no_function_tag"])
        assert emitted == ["no_function_tag", "no_function_tag"]

    def test_source_wires_the_dedupe(self):
        src = AGENT_SRC.read_text()
        assert "emitted_truncation_error = False" in src
        assert 'parse_failure_reason == "truncated" and emitted_truncation_error' in src
        assert "emitted_truncation_error = True" in src


# ---------------------------------------------------------------------------
# Edit 4 — dream.py <think> scrub + raw-content fallback
# ---------------------------------------------------------------------------

# New (safe) and old (destructive) scrub regexes — we compare both to prove
# the fix changes behavior only in the buggy cases.
NEW_THINK_SCRUB = re.compile(r'<think>.*?</think>', flags=re.DOTALL | re.IGNORECASE)
OLD_THINK_SCRUB = re.compile(r'<think>.*?(?:</think>|$)', flags=re.DOTALL | re.IGNORECASE)


def _extract_xml_block(tag: str, text: str) -> str:
    """Mirror of dream.py's `_extract_xml_block` after the reliability
    improvements: (a) close tag tolerates trailing whitespace, (b) the
    missing-close fallback stops at the next top-level block opener so
    a body with no close tag doesn't silently swallow later sections."""
    close_tag = rf'</{tag}\s*>'
    m = re.search(rf'<{tag}[^>]*>(.*){close_tag}', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    next_block_re = r'<(?:challenge_prompt|setup_script|validation_script)\b'
    m = re.search(rf'<{tag}[^>]*>(.*?)(?={next_block_re}|$)', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def _extract_with_fallback(tag: str, scrubbed: str, raw: str) -> str:
    hit = _extract_xml_block(tag, scrubbed)
    return hit if hit else _extract_xml_block(tag, raw)


class TestDreamThinkScrubFix:
    def test_closed_think_is_stripped(self):
        s = "<think>reasoning</think>\n<challenge_prompt>X</challenge_prompt>"
        out = NEW_THINK_SCRUB.sub('', s)
        assert "<think>" not in out
        assert "reasoning" not in out
        assert "<challenge_prompt>" in out

    def test_unclosed_think_does_not_eat_following_blocks(self):
        """THE critical fix. Old regex consumed everything from `<think>`
        to end-of-string — this is the dominant challenge-generation
        failure mode near the max_tokens boundary."""
        s = ("<think>reasoning without close\n"
             "<challenge_prompt>X</challenge_prompt>\n"
             "<validation_script>print(1)</validation_script>")

        new_out = NEW_THINK_SCRUB.sub('', s)
        old_out = OLD_THINK_SCRUB.sub('', s)

        # NEW: XML blocks survive.
        assert "<challenge_prompt>" in new_out
        assert "<validation_script>" in new_out
        # OLD: blocks were eaten (the bug we fixed — kept as a regression
        # oracle so reverting the scrub trips this test loudly).
        assert "<challenge_prompt>" not in old_out
        assert "<validation_script>" not in old_out

    def test_extraction_via_raw_fallback_when_blocks_nested_in_think(self):
        """Qwen builds sometimes place the required XML blocks INSIDE
        `<think>...</think>` because the prompt tells them to think in
        `<think>` tags. The scrub removes them, but the raw copy still
        has them."""
        raw = ("<think>I will produce:\n"
               "<challenge_prompt>CHALLENGE</challenge_prompt>\n"
               "<validation_script>VALIDATOR</validation_script>\n"
               "</think>")
        scrubbed = NEW_THINK_SCRUB.sub('', raw)

        # Scrubbed copy is emptied by the think-removal.
        assert _extract_xml_block("challenge_prompt", scrubbed) == ""
        assert _extract_xml_block("validation_script", scrubbed) == ""

        # Raw-fallback recovers both.
        assert _extract_with_fallback("challenge_prompt", scrubbed, raw) == "CHALLENGE"
        assert _extract_with_fallback("validation_script", scrubbed, raw) == "VALIDATOR"

    def test_happy_path_extraction_unchanged(self):
        raw = ("<think>plan</think>\n"
               "<challenge_prompt>CH</challenge_prompt>\n"
               "<validation_script>VS</validation_script>")
        scrubbed = NEW_THINK_SCRUB.sub('', raw)
        assert _extract_with_fallback("challenge_prompt", scrubbed, raw) == "CH"
        assert _extract_with_fallback("validation_script", scrubbed, raw) == "VS"

    def test_no_think_tag_works(self):
        raw = ("<challenge_prompt>CH</challenge_prompt>\n"
               "<validation_script>VS</validation_script>")
        scrubbed = NEW_THINK_SCRUB.sub('', raw)
        assert _extract_with_fallback("challenge_prompt", scrubbed, raw) == "CH"
        assert _extract_with_fallback("validation_script", scrubbed, raw) == "VS"

    def test_truncated_mid_validator_still_extracts(self):
        raw = ("<think>plan</think>\n"
               "<challenge_prompt>CH</challenge_prompt>\n"
               "<validation_script>print(1")  # upstream cut off mid-script
        scrubbed = NEW_THINK_SCRUB.sub('', raw)
        assert _extract_with_fallback("challenge_prompt", scrubbed, raw) == "CH"
        # validation_script uses the `(.*?)$` fallback in _extract_xml_block.
        assert "print(1" in _extract_with_fallback("validation_script", scrubbed, raw)

    def test_source_no_longer_has_the_destructive_fallback(self):
        src = DREAM_SRC.read_text()
        # Scan non-comment lines only — the explanatory docstring legitimately
        # quotes the old pattern, so we check call sites, not mentions.
        code_lines = [
            line for line in src.splitlines()
            if line.lstrip() and not line.lstrip().startswith("#")
        ]
        code = "\n".join(code_lines)

        # The destructive call-site must be gone.
        assert r"re.sub(r'<think>.*?(?:</think>|$)'" not in code
        assert r'"<think>.*?(?:</think>|$)"' not in code

        # The safe scrub and the raw-content fallback must be wired.
        assert r"re.sub(r'<think>.*?</think>'" in code
        assert "raw_content_text" in code
        assert "_extract_with_fallback" in code


# ---------------------------------------------------------------------------
# Edit 6 — Self-play challenge-extraction reliability (core/dream.py)
#
# Three linked fixes in the generation loop:
#   (a) max_tokens raised from 8192 → 16384 so verbose challenges don't
#       starve the <validation_script> block.
#   (b) Close-tag regex tolerates trailing whitespace (`</tag >`) and the
#       missing-close fallback stops at the next top-level block opener
#       instead of running to end-of-string.
#   (c) Rejection feedback names the specific missing block instead of
#       sending a generic "emit valid XML" hint.
# ---------------------------------------------------------------------------


class TestChallengeExtractorRobustness:
    """The shape the user hit in the log: challenge_prompt parsed fine,
    but the response was cut off before the validator block. These tests
    also exercise close-tag whitespace tolerance and the safer
    missing-close fallback."""

    def test_close_tag_with_trailing_whitespace_is_accepted(self):
        s = "<challenge_prompt>task desc</challenge_prompt >"
        assert _extract_xml_block("challenge_prompt", s) == "task desc"

    def test_close_tag_with_trailing_newline_is_accepted(self):
        s = "<challenge_prompt>task desc</challenge_prompt\n>"
        assert _extract_xml_block("challenge_prompt", s) == "task desc"

    def test_missing_close_fallback_stops_at_next_block(self):
        """The old fallback ran the body to end-of-string, which silently
        absorbed the next section's content. The new fallback stops at the
        next top-level block opener."""
        s = ("<challenge_prompt>challenge body\n"
             "<setup_script>SETUP</setup_script>\n"
             "<validation_script>VAL</validation_script>")
        # No </challenge_prompt> close — should stop at <setup_script>, not
        # swallow both following blocks.
        ch = _extract_xml_block("challenge_prompt", s)
        assert ch == "challenge body"
        assert "<setup_script>" not in ch
        assert "VAL" not in ch

    def test_missing_close_falls_through_to_end_when_no_next_block(self):
        """When there's nothing after the unclosed tag, fallback captures
        body-to-end-of-string as before (preserves the truncated-validator
        recovery path)."""
        s = "<challenge_prompt>task desc, never closed"
        assert _extract_xml_block("challenge_prompt", s) == "task desc, never closed"

    def test_all_three_blocks_with_whitespace_variants_extract(self):
        """Real responses contain a mix of tight and whitespace-padded
        close tags; all three must still extract cleanly."""
        s = ("<challenge_prompt>CH</challenge_prompt>\n"
             "<setup_script>SETUP</setup_script >\n"
             "<validation_script>VAL</validation_script\n>\n")
        assert _extract_xml_block("challenge_prompt", s) == "CH"
        assert _extract_xml_block("setup_script", s) == "SETUP"
        assert _extract_xml_block("validation_script", s) == "VAL"

    def test_source_raises_max_tokens(self):
        src = DREAM_SRC.read_text()
        assert '"max_tokens": 16384' in src
        assert '"max_tokens": 8192' not in src

    def test_source_uses_whitespace_tolerant_close_tag(self):
        src = DREAM_SRC.read_text()
        # Scan non-comment lines only.
        code = "\n".join(
            line for line in src.splitlines()
            if line.lstrip() and not line.lstrip().startswith("#")
        )
        assert r"</{tag}\s*>" in code

    def test_source_logs_both_head_and_tail_on_failure(self):
        src = DREAM_SRC.read_text()
        # Both head and tail previews must be present so the failure log
        # can distinguish "never wrote the block" from "malformed close
        # tag near the end".
        assert "content_text[:400]" in src
        assert "content_text[-400:]" in src

    def test_source_emits_targeted_rejection_feedback(self):
        src = DREAM_SRC.read_text()
        # Each of the three missing-block branches must be distinguishable
        # — this is what tells the model WHICH block to fix on retry.
        assert "missing or truncated before the" in src
        assert "missing or malformed around the" in src


# ---------------------------------------------------------------------------
# Edit 7 — Worker tool_turn max_tokens & truncation diagnostics (core/agent.py)
#
# Production self-play workers were emitting responses in the ~12k-token
# range (verbose <think> preamble + full inline solution script via
# `execute`) and hitting the 8k ceiling mid-<parameter name="content">.
# Raising the cap to 16384 removes the most common truncation trigger.
#
# The truncation detector's warning was also expanded to surface head +
# tail + tool_call/function/parameter open/close counts so the next
# truncation (if any) can be diagnosed from the log alone.
#
# The post-stream "thought" telemetry was rewritten because the old
# `{thinking_token_count} tokens · {chars} chars` conflated the
# reasoning-channel token count with total chars across both channels,
# producing the misleading 341-chars-per-token ratio the user flagged.
# ---------------------------------------------------------------------------


class TestToolTurnMaxTokensRaised:
    def test_default_is_16384(self):
        """Guard the precise value — a drift back below ~12k will
        silently reintroduce the observed truncation class."""
        from ghost_agent.core.agent import DEFAULT_TOOL_TURN_MAX_TOKENS
        assert DEFAULT_TOOL_TURN_MAX_TOKENS == 16384

    def test_default_fits_observed_worker_response_size(self):
        """The worker was emitting ~48k chars ≈ ~12k tokens. The cap
        must be comfortably larger so a real solution script doesn't
        get severed mid-parameter."""
        from ghost_agent.core.agent import DEFAULT_TOOL_TURN_MAX_TOKENS
        observed_worker_tokens = 48000 // 4  # rough 4-chars-per-token
        assert DEFAULT_TOOL_TURN_MAX_TOKENS > observed_worker_tokens

    def test_default_leaves_room_in_65k_context(self):
        """Must also not crowd out surrounding messages + tool results
        in a 65k window."""
        from ghost_agent.core.agent import DEFAULT_TOOL_TURN_MAX_TOKENS
        assert DEFAULT_TOOL_TURN_MAX_TOKENS <= 24576  # ~37% of 65k

    def test_existing_generous_and_not_huge_bounds_still_hold(self):
        """The existing test_default_is_generous bounds (>=4096, <=32000)
        must continue to pass — this is a sanity check."""
        from ghost_agent.core.agent import DEFAULT_TOOL_TURN_MAX_TOKENS
        assert DEFAULT_TOOL_TURN_MAX_TOKENS >= 4096
        assert DEFAULT_TOOL_TURN_MAX_TOKENS <= 32000


class TestTruncationDiagnostics:
    """The expanded `logger.warning(...)` + `pretty_log(...)` emitted
    when `_tool_call_truncated` fires. Source-level guards only — the
    log path runs deep inside `handle_chat` and isn't worth booting a
    full agent for in a unit test."""

    def test_truncation_warning_includes_response_length(self):
        src = AGENT_SRC.read_text()
        # The len(...) must appear in the warning format arguments.
        assert "Parse target truncated. len=%d" in src

    def test_truncation_warning_counts_all_three_tag_families(self):
        src = AGENT_SRC.read_text()
        # All three families (tool_call, function, parameter) must be
        # counted — previous diagnostics covered only the first two and
        # missed the "huge <parameter> body got severed" case.
        assert "_tc_opens" in src and "_tc_closes" in src
        assert "_fn_opens" in src and "_fn_closes" in src
        assert "_param_opens" in src and "_param_closes" in src

    def test_truncation_warning_emits_head_and_tail(self):
        src = AGENT_SRC.read_text()
        # Head AND tail — not just tail like the old version.
        assert "parse_target[:300]" in src
        assert "parse_target[-600:]" in src

    def test_pretty_log_surfaces_truncation_in_live_stream(self):
        src = AGENT_SRC.read_text()
        # A visible pretty_log line so the user sees it in the live
        # trace, not only in file-based logs.
        assert '"Upstream Truncation"' in src


class TestThoughtTelemetryFormat:
    """The `pretty_log("thought", ...)` summary that runs post-stream."""

    def test_reasoning_and_content_are_reported_separately(self):
        src = AGENT_SRC.read_text()
        # The misleading merged form must be gone.
        assert "{thinking_token_count} tokens · {chars} chars" not in src
        # The separated form must be present.
        assert "reasoning: {thinking_token_count} tokens / {reasoning_chars} chars" in src
        assert "content: {content_chars} chars" in src
