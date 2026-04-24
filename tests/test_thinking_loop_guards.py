"""Tests for the streaming-loop guards, atomic logging, and prompt rules
added to defend against runaway thinking loops (the vowels_count incident).
"""
import io
import threading
from contextlib import redirect_stdout

import pytest

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.agent import (
    MAX_THINKING_CHARS,
    THINKING_LOOP_PROBE_EVERY,
    THINKING_LOOP_THRESHOLD,
    THINKING_LOOP_WINDOW,
    _detect_thinking_loop,
    _render_assistant_with_tool_calls,
)
from ghost_agent.core.prompts import PLANNING_SYSTEM_PROMPT, SPECIALIST_SYSTEM_PROMPT
from ghost_agent.utils import logging as glog
from ghost_agent.utils.logging import (
    atomic_print,
    pretty_log,
    request_id_context,
    _format_delta,
    _req_color,
    _req_tag,
)


# ---------------------------------------------------------------------------
# _detect_thinking_loop
# ---------------------------------------------------------------------------

class TestDetectThinkingLoop:
    def test_empty_buffer_is_not_a_loop(self):
        assert _detect_thinking_loop("") is False

    def test_short_buffer_is_not_a_loop(self):
        # Below the minimum required length (window * threshold).
        buf = "a" * (THINKING_LOOP_WINDOW * THINKING_LOOP_THRESHOLD - 1)
        assert _detect_thinking_loop(buf) is False

    def test_unique_long_text_is_not_a_loop(self):
        # Pseudo-unique content: each chunk is distinct, no repeats.
        buf = "".join(f"sentence number {i} with unique words. " for i in range(2000))
        assert _detect_thinking_loop(buf) is False

    def test_exact_repetition_is_a_loop(self):
        # The vowels_count pathology: the same paragraph emitted N times.
        paragraph = "Let me check the function again. " * 20  # ~660 chars
        buf = paragraph * THINKING_LOOP_THRESHOLD
        assert _detect_thinking_loop(buf) is True

    def test_threshold_minus_one_is_not_a_loop(self):
        # Just under the threshold should not trip.
        paragraph = "x" * THINKING_LOOP_WINDOW
        buf = paragraph * (THINKING_LOOP_THRESHOLD - 1)
        assert _detect_thinking_loop(buf) is False

    def test_loop_buried_in_long_buffer(self):
        prefix = "intro that does not repeat. " * 50
        loop_chunk = "DEBUG STATE A=1 B=2 C=3 retry. " * 20
        buf = prefix + loop_chunk * THINKING_LOOP_THRESHOLD
        assert _detect_thinking_loop(buf) is True

    def test_constants_are_sane(self):
        assert MAX_THINKING_CHARS > 0
        assert THINKING_LOOP_PROBE_EVERY > 0
        assert THINKING_LOOP_WINDOW > 0
        assert THINKING_LOOP_THRESHOLD >= 2


# ---------------------------------------------------------------------------
# atomic_print + per-request log isolation
# ---------------------------------------------------------------------------

class TestAtomicPrint:
    def test_atomic_print_writes_one_line_with_newline(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            atomic_print("hello world")
        assert buf.getvalue() == "hello world\n"

    def test_atomic_print_uses_module_lock(self):
        # The lock must be a real threading.Lock instance, not a placeholder.
        assert isinstance(glog._STDOUT_LOCK, type(threading.Lock()))

    def test_concurrent_atomic_prints_do_not_interleave(self):
        # Each thread writes 200 lines; with the lock, every output line must
        # be one of the two complete strings (never spliced together).
        line_a = "A" * 80
        line_b = "B" * 80
        buf = io.StringIO()

        def writer(line, n):
            for _ in range(n):
                atomic_print(line)

        with redirect_stdout(buf):
            t1 = threading.Thread(target=writer, args=(line_a, 200))
            t2 = threading.Thread(target=writer, args=(line_b, 200))
            t1.start(); t2.start()
            t1.join(); t2.join()

        out_lines = buf.getvalue().splitlines()
        assert len(out_lines) == 400
        for line in out_lines:
            assert line in (line_a, line_b), f"interleaved line detected: {line!r}"

    def test_pretty_log_uses_atomic_print_with_tag_and_title(self):
        # Redirected StringIO is not a TTY, so the formatter emits no ANSI
        # escapes — the assertions can match on plain text.
        token = request_id_context.set("re12345678")
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                pretty_log("Test Title", "payload", icon="X")
            out = buf.getvalue()
        finally:
            request_id_context.reset(token)
        # 2-char tag derived from the first two characters of the request id.
        assert "RE" in out
        # Title is lowercased in the new layout.
        assert "test title" in out
        assert "payload" in out
        assert out.endswith("\n")
        assert out.count("\n") == 1  # single atomic line
        # No ANSI escape leaked into a non-TTY sink.
        assert "\033[" not in out

    def test_request_tag_is_deterministic(self):
        assert _req_tag("a8a93a27") == "A8"
        assert _req_tag("eb30aa56") == "EB"
        assert _req_tag("SYSTEM") == "**"

    def test_request_color_disabled_for_non_tty(self):
        # In tests, stdout is a captured sink — _USE_COLOR must be False so
        # log files and CI output stay clean of escape sequences.
        assert _req_color("anything") == ""

    def test_format_delta_returns_blank_for_unknown_request(self):
        # A request with no BEGIN marker has no start time → blank delta.
        assert _format_delta("never-started").strip() == ""

    def test_begin_end_lifecycle_records_delta(self):
        token = request_id_context.set("ff00ff00")
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                pretty_log("X", special_marker="BEGIN")
                # Between BEGIN and END the delta should be non-empty.
                mid = _format_delta("ff00ff00").strip()
                pretty_log("X", special_marker="END")
                # After END the state is cleared.
                post = _format_delta("ff00ff00").strip()
        finally:
            request_id_context.reset(token)
        assert mid != ""
        assert mid.startswith("+")
        assert post == ""

    def test_pretty_log_begin_marker_emits_box_frame(self):
        token = request_id_context.set("aabbccdd")
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                pretty_log("X", special_marker="BEGIN")
                pretty_log("X", special_marker="END")
            out = buf.getvalue()
        finally:
            request_id_context.reset(token)
        # Box-drawing top + bottom edges flank the request.
        assert "┌─" in out
        assert "└─" in out
        assert "request started" in out
        assert "request finished" in out
        assert "AA" in out  # 2-char tag from the first two id chars
        assert "aabbccdd" in out  # short id printed in BEGIN frame
        # Two complete atomic lines.
        assert out.count("\n") == 2


# ---------------------------------------------------------------------------
# Prompt rules: assertion-distrust / suspect-the-test guidance
# ---------------------------------------------------------------------------

class TestSelfCorrectionPrompts:
    def test_specialist_prompt_has_suspect_test_rule(self):
        text = SPECIALIST_SYSTEM_PROMPT.upper()
        assert "SUSPECT THE TEST FIRST" in text
        # Mentions the actual failure mode it's defending against.
        assert "SPEC" in text and "ASSERTION" in text

    def test_specialist_prompt_is_numbered_after_completion(self):
        # SUSPECT THE TEST FIRST was originally rule 10, but got pushed to
        # 11 when the END-TO-END DEMO rule was inserted at position 10 to
        # address the verifier's REFUTED(90%) on demos that only
        # exercised a leaf helper. The invariant this test protects is
        # that SUSPECT THE TEST FIRST still lives AFTER COMPLETION in the
        # numbered list (so the debug heuristic can't fire on clean runs).
        idx_completion = SPECIALIST_SYSTEM_PROMPT.find("9. COMPLETION")
        idx_suspect = SPECIALIST_SYSTEM_PROMPT.find("SUSPECT THE TEST FIRST")
        assert idx_completion != -1 and idx_suspect != -1
        assert idx_suspect > idx_completion

    def test_planning_prompt_has_assertion_distrust_rule(self):
        text = PLANNING_SYSTEM_PROMPT.upper()
        assert "ASSERTION DISTRUST" in text
        assert "SPEC" in text

    def test_planning_prompt_appended_after_tool_knowledge(self):
        idx8 = PLANNING_SYSTEM_PROMPT.find("8. TOOL KNOWLEDGE")
        idx9 = PLANNING_SYSTEM_PROMPT.find("9. ASSERTION DISTRUST")
        assert idx8 != -1 and idx9 != -1
        assert idx9 > idx8


# ---------------------------------------------------------------------------
# _render_assistant_with_tool_calls — parallel tool-call regression
# ---------------------------------------------------------------------------

class TestRenderAssistantWithToolCalls:
    def _make_call(self, name: str, **kwargs):
        import json as _json
        return {
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": _json.dumps(kwargs)},
        }

    def test_empty_tool_calls_returns_content_verbatim(self):
        assert _render_assistant_with_tool_calls("hello", []) == "hello"
        assert _render_assistant_with_tool_calls(None, []) == ""

    def test_single_tool_call_renders_xml(self):
        call = self._make_call("update_profile", category="root", key="city", value="Athens")
        out = _render_assistant_with_tool_calls("", [call])
        assert "<tool_call>" in out
        assert '<function name="update_profile">' in out
        assert "<parameter name=\"category\">" in out
        assert "Athens" in out

    def test_four_parallel_tool_calls_all_render(self):
        # The exact production pathology: 4 parallel update_profile calls.
        calls = [
            self._make_call("update_profile", category="relationships", key="son_1_name", value="Thodoris"),
            self._make_call("update_profile", category="relationships", key="son_1_age", value="9 years old"),
            self._make_call("update_profile", category="relationships", key="son_2_name", value="Leonidas"),
            self._make_call("update_profile", category="relationships", key="son_2_age", value="1 month old"),
        ]
        out = _render_assistant_with_tool_calls("", calls)
        # All four tool_call blocks must be present (the regression: only the
        # first survived, calls 2-4 were silently dropped by the per-iteration
        # `<tool_call> in ast_content` check).
        assert out.count("<tool_call>") == 4
        assert out.count("</tool_call>") == 4
        # Each unique value must appear in the rendered text.
        for needle in ("Thodoris", "9 years old", "Leonidas", "1 month old"):
            assert needle in out
        # Each unique key must appear.
        for needle in ("son_1_name", "son_1_age", "son_2_name", "son_2_age"):
            assert needle in out

    def test_already_inline_tool_calls_are_not_double_rendered(self):
        # If the model emitted tool_calls as text (rare path: streamed via
        # delta.content rather than delta.tool_calls), the inline XML is
        # already in the assistant content. We must NOT append again.
        existing = "thinking done.\n<tool_call><function name=\"x\"></function></tool_call>"
        call = self._make_call("update_profile", key="k", value="v")
        out = _render_assistant_with_tool_calls(existing, [call])
        assert out.count("<tool_call>") == 1
        # update_profile from the synthetic call must NOT have been added.
        assert "update_profile" not in out

    def test_dict_arguments_serialise_as_json(self):
        # Non-string parameter values must round-trip through json.dumps so
        # the next turn can re-parse them. The legacy `str(v)` path emitted
        # Python repr (`{'k': 'v'}`) which mis-parses as JSON.
        call = self._make_call("execute", env={"FOO": "bar"}, dry_run=True, retries=None)
        out = _render_assistant_with_tool_calls("", [call])
        assert '{"FOO": "bar"}' in out
        assert "true" in out  # bool → JSON
        assert "null" in out  # None → JSON

    def test_tool_call_count_preserved_for_arbitrary_n(self):
        calls = [self._make_call("f", i=i) for i in range(7)]
        out = _render_assistant_with_tool_calls("", calls)
        assert out.count("<tool_call>") == 7


# ---------------------------------------------------------------------------
# Module wiring: streaming-guard symbols are exported on the agent module
# ---------------------------------------------------------------------------

class TestAgentModuleWiring:
    def test_atomic_print_is_imported_into_agent_module(self):
        # The streaming loop calls atomic_print directly — if the import is
        # ever dropped, this catches it before runtime.
        assert hasattr(agent_mod, "atomic_print")
        assert agent_mod.atomic_print is atomic_print

    def test_loop_detector_is_exported(self):
        assert callable(agent_mod._detect_thinking_loop)

    def test_max_thinking_chars_reasonable(self):
        # Should be large enough for a real chain of thought but small enough
        # to kill the runaway-loop pathology in seconds, not minutes.
        assert 4_000 <= MAX_THINKING_CHARS <= 200_000

    def test_render_helper_is_exported(self):
        assert callable(agent_mod._render_assistant_with_tool_calls)

    def test_post_stream_summary_uses_thought_label(self):
        # The streaming loop emits the summary line with title="thought"
        # (past tense) so it doesn't visually collide with the live
        # "thinking" lines emitted in verbose mode. Grep the source.
        from pathlib import Path
        source = Path(agent_mod.__file__).read_text()
        # The summary call site must use the past-tense label.
        assert 'pretty_log(\n                                "thought"' in source
        assert "thought" in source


# ---------------------------------------------------------------------------
# Regression check against the actual incident transcript
# ---------------------------------------------------------------------------

class TestVowelsCountRegression:
    def test_vowels_count_repeated_paragraph_trips_detector(self):
        # This is the exact shape of the runaway thinking from the incident:
        # the model kept restating "Let me check the code again" verbatim.
        paragraph = (
            "Let me check the code again. The function has if char.lower() "
            "in vowels: count += 1. For 'rhythm', the y is at index 1 and "
            "the last character is m at index 4. So the condition for y is "
            "false. But the function is returning 2. "
        )
        buf = paragraph * 10
        assert _detect_thinking_loop(buf) is True

    def test_distinct_reasoning_paragraphs_do_not_trip_detector(self):
        # Healthy reasoning: monotonically progressing, no paragraph repeats.
        # Each step is unique because of its index, so no 400-char window can
        # appear more than once.
        steps = [
            f"Step {i}: examine sub-case {i*7 % 13} of the algorithm and "
            f"verify branch {i*3 % 17} against expectation {i*5 % 19}; the "
            f"intermediate accumulator should hold value {i*11 % 23} here."
            for i in range(200)
        ]
        buf = " ".join(steps)
        assert _detect_thinking_loop(buf) is False
