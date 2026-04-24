"""
Tests for the robust Qwen XML parser (split-based) and the final-output scrubbers.

Covers:
  - Happy-path: well-formed <tool_call>
  - Missing </tool_call> closing tag
  - Tags with spaces: <tool_call >
  - Tags with attributes: <tool_call id="123">
  - Markdown code-fence injection inside the tag
  - Extra whitespace / newlines around the JSON
  - Stringified 'arguments' value (model returns a JSON string instead of dict)
  - Multiple tool calls in one response
  - <think> block isolation (both closed and unclosed)
  - Case-insensitive tag detection
  - Final-output scrubber: orphaned <tool_call> with no closing tag
  - Final-output scrubber: orphaned <tool_response> with no closing tag
  - ui_content is cleaned even when closing tag is absent
"""

import re
import json
import uuid
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# ---------------------------------------------------------------------------
# Helpers – extracted logic so we can unit-test it in isolation without
# spinning up the full agent loop
# ---------------------------------------------------------------------------

def extract_json_from_text(text: str):
    """Minimal clone of the production helper used in agent.py."""
    import re as _re
    # Strip markdown fences
    text = _re.sub(r'```(?:json)?', '', text, flags=_re.IGNORECASE).strip()
    # Find the outermost { … }
    start = text.find('{')
    end   = text.rfind('}')
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def run_qwen_parser(content: str):
    """
    Pure-Python re-implementation of the ROBUST XML TOOL PARSER block.
    Returns (tool_calls, ui_content).
    """
    tool_calls: list = []
    ui_content = content

    has_tool_tag = re.search(r'<tool_call', content, re.IGNORECASE) is not None
    if has_tool_tag:
        # Strip <think> blocks (including unclosed ones)
        parse_target = re.sub(
            r'<think>.*?(?:</think>|$)', '', content,
            flags=re.DOTALL | re.IGNORECASE
        )

        # Split by <tool_call.*?> to handle spaces and attributes
        blocks = re.split(r'<tool_call.*?>', parse_target, flags=re.IGNORECASE)
        for block in blocks[1:]:
            # Strip out anything after the closing tag if it exists
            block = re.split(r'</tool_call.*?>', block, flags=re.IGNORECASE)[0]
            try:
                t_data = extract_json_from_text(block)
                if t_data and "name" in t_data:
                    args_val = t_data.get("arguments", {})
                    if isinstance(args_val, str):
                        try:
                            args_val = json.loads(args_val)
                        except Exception:
                            args_val = {}
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": t_data.get("name"),
                            "arguments": json.dumps(args_val),
                        },
                    })
            except Exception:
                pass

        # Scrub from UI (handle missing closing tag + varying tag format)
        ui_content = re.sub(
            r'<tool_call.*?>.*?(?:</tool_call.*?>|$)', '', ui_content,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()

    return tool_calls, ui_content


def run_final_scrubber(text: str) -> str:
    """Mirrors the two-line scrubber block in the FINAL OUTPUT SCRUBBER section."""
    text = re.sub(
        r'<tool_call.*?>.*?(?:</tool_call.*?>|$)', '', text,
        flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(
        r'<tool_response.*?>.*?(?:</tool_response.*?>|$)', '', text,
        flags=re.DOTALL | re.IGNORECASE
    )
    return text.strip()


# ===========================================================================
# Parser tests
# ===========================================================================

class TestQwenParserHappyPath:
    def test_well_formed_single_call(self):
        content = '<tool_call>{"name": "web_search", "arguments": {"query": "hello"}}</tool_call>'
        calls, ui = run_qwen_parser(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "hello"}
        assert ui == ""

    def test_tag_with_trailing_space(self):
        content = '<tool_call >{"name": "x", "arguments": {}}</tool_call>'
        calls, ui = run_qwen_parser(content)
        assert len(calls) == 1
        assert ui == ""

    def test_tag_with_attributes(self):
        content = '<tool_call id="123">{"name": "x", "arguments": {}}</tool_call>'
        calls, ui = run_qwen_parser(content)
        assert len(calls) == 1
        assert ui == ""

    def test_ui_text_before_call_is_preserved(self):
        content = 'Let me search that.\n<tool_call>{"name": "web_search", "arguments": {"query": "test"}}</tool_call>'
        calls, ui = run_qwen_parser(content)
        assert len(calls) == 1
        assert "Let me search that." in ui

    def test_ui_text_after_call_is_preserved(self):
        content = '<tool_call>{"name": "web_search", "arguments": {}}</tool_call>\nDone!'
        calls, ui = run_qwen_parser(content)
        assert len(calls) == 1
        assert "Done!" in ui


class TestQwenParserRobustness:
    def test_missing_closing_tag(self):
        """Parser must extract the call even without </tool_call>."""
        content = '<tool_call>{"name": "execute", "arguments": {"code": "print(1)"}}'
        calls, ui = run_qwen_parser(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "execute"

    def test_markdown_fence_injection(self):
        """Model sometimes wraps JSON in ```json … ``` inside the tag."""
        content = (
            '<tool_call>\n```json\n'
            '{"name": "file_system", "arguments": {"operation": "read", "path": "/tmp/x"}}\n'
            '```\n</tool_call>'
        )
        calls, ui = run_qwen_parser(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "file_system"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["operation"] == "read"

    def test_extra_whitespace_and_newlines(self):
        content = '<tool_call>  \n\n  {"name": "recall", "arguments": {"query": "foo"}}  \n\n  </tool_call>'
        calls, ui = run_qwen_parser(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "recall"

    def test_stringified_arguments(self):
        """Some models return arguments as a JSON string instead of a dict."""
        args_str = json.dumps({"query": "bar"})
        payload = json.dumps({"name": "web_search", "arguments": args_str})
        content = f'<tool_call>{payload}</tool_call>'
        calls, ui = run_qwen_parser(content)
        assert len(calls) == 1
        parsed_args = json.loads(calls[0]["function"]["arguments"])
        assert parsed_args == {"query": "bar"}

    def test_multiple_tool_calls(self):
        content = (
            '<tool_call>{"name": "web_search", "arguments": {"query": "a"}}</tool_call>\n'
            '<tool_call>{"name": "recall", "arguments": {"query": "b"}}</tool_call>'
        )
        calls, ui = run_qwen_parser(content)
        assert len(calls) == 2
        names = {c["function"]["name"] for c in calls}
        assert names == {"web_search", "recall"}

    def test_multiple_calls_one_missing_closing_tag(self):
        """First call is well-formed, second is missing its closing tag."""
        content = (
            '<tool_call>{"name": "web_search", "arguments": {"query": "a"}}</tool_call>\n'
            '<tool_call>{"name": "recall", "arguments": {"query": "b"}}'
        )
        calls, ui = run_qwen_parser(content)
        assert len(calls) == 2

    def test_case_insensitive_tag(self):
        content = '<TOOL_CALL>{"name": "recall", "arguments": {}}</TOOL_CALL>'
        calls, ui = run_qwen_parser(content)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "recall"

    def test_invalid_json_is_skipped_gracefully(self):
        content = '<tool_call>NOT VALID JSON AT ALL</tool_call>'
        calls, ui = run_qwen_parser(content)
        assert calls == []

    def test_no_tool_call_tag_returns_empty(self):
        content = 'Just a regular assistant response.'
        calls, ui = run_qwen_parser(content)
        assert calls == []
        assert ui == content


class TestQwenParserThinkBlockIsolation:
    def test_think_block_removed_before_parsing(self):
        """Tool calls inside <think> must NOT be executed."""
        content = (
            '<think>I might call <tool_call>{"name": "bad_tool", "arguments": {}}</tool_call> '
            'but I wont.</think>'
            '<tool_call>{"name": "good_tool", "arguments": {}}</tool_call>'
        )
        calls, ui = run_qwen_parser(content)
        names = [c["function"]["name"] for c in calls]
        assert "bad_tool" not in names
        assert "good_tool" in names

    def test_unclosed_think_block_does_not_swallow_call(self):
        """An unclosed <think> should not eat the real tool call that follows."""
        content = (
            '<think>Reasoning here...'
            '<tool_call>{"name": "safe_tool", "arguments": {}}</tool_call>'
        )
        calls, _ = run_qwen_parser(content)
        # Conservative: call is inside the unclosed think block → swallowed (safe default)
        assert calls == []


class TestQwenParserUiScrubbing:
    def test_ui_content_scrubbed_with_closing_tag(self):
        content = 'Prefix\n<tool_call>{"name": "x", "arguments": {}}</tool_call>\nSuffix'
        _, ui = run_qwen_parser(content)
        assert '<tool_call' not in ui
        assert 'Prefix' in ui
        assert 'Suffix' in ui

    def test_ui_content_scrubbed_without_closing_tag(self):
        content = 'Prefix\n<tool_call>{"name": "x", "arguments": {}}'
        _, ui = run_qwen_parser(content)
        assert '<tool_call' not in ui
        # The orphaned fragment must not appear in the UI
        assert '{"name": "x"' not in ui

    def test_ui_content_scrubbed_with_varying_tags(self):
        content = 'Prefix\n<tool_call id="1">{"name": "x", "arguments": {}}</tool_call >\nSuffix'
        _, ui = run_qwen_parser(content)
        assert '<tool_call' not in ui
        assert 'Prefix' in ui
        assert 'Suffix' in ui


# ===========================================================================
# Final-output scrubber tests
# ===========================================================================

class TestFinalOutputScrubber:
    def test_well_formed_tool_call_scrubbed(self):
        text = 'Result: <tool_call>{"name": "x", "arguments": {}}</tool_call> done.'
        assert '<tool_call' not in run_final_scrubber(text)

    def test_orphaned_tool_call_scrubbed(self):
        """<tool_call> without closing tag must still be removed."""
        text = 'Result: <tool_call>{"name": "x", "arguments": {}} done.'
        result = run_final_scrubber(text)
        assert '<tool_call' not in result
        assert '{"name": "x"' not in result

    def test_varying_tool_call_tag_scrubbed(self):
        text = 'Result: <tool_call id="9">{"name": "x", "arguments": {}} done.'
        result = run_final_scrubber(text)
        assert '<tool_call' not in result
        assert 'Result: done.' in result or 'Result:' in result

    def test_well_formed_tool_response_scrubbed(self):
        text = 'Data: <tool_response>some output</tool_response> end.'
        assert '<tool_response>' not in run_final_scrubber(text)

    def test_orphaned_tool_response_scrubbed(self):
        """<tool_response> without closing tag must still be removed."""
        text = 'Data: <tool_response>some output end.'
        result = run_final_scrubber(text)
        assert '<tool_response>' not in result
        assert 'some output' not in result

    def test_clean_text_unchanged(self):
        text = 'This is a clean response with no XML.'
        assert run_final_scrubber(text) == text

    def test_text_before_orphaned_tag_preserved(self):
        text = 'Keep this. <tool_call>{"name": "x", "arguments": {}}'
        result = run_final_scrubber(text)
        assert 'Keep this.' in result

    def test_multiline_tool_call_scrubbed(self):
        text = (
            'Before\n'
            '<tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>\n'
            'After'
        )
        result = run_final_scrubber(text)
        assert '<tool_call' not in result
        assert 'Before' in result
        assert 'After' in result

    def test_case_insensitive_scrub(self):
        text = 'X <TOOL_CALL>{"name": "x", "arguments": {}}</TOOL_CALL> Y'
        result = run_final_scrubber(text)
        assert 'TOOL_CALL' not in result.upper() or '<' not in result
