"""Direct unit tests for GhostAgent._parse_assistant_tool_calls.

The XML tool-call parser lived inline in handle_chat for its whole life —
~640 lines of hot-path logic exercised only indirectly through integration
tests. The 2026-07-08 extraction (step 1 of the handle_chat decomposition)
made it a method with a crisp contract:

    (tool_calls, ui_content, parse_failure_reason) =
        agent._parse_assistant_tool_calls(content, msg)

These tests pin the contract across the emission shapes the upstream
actually produces. The agent is built via __new__ — the method's only
self-dependency is `available_tools` (hasattr-guarded).
"""
import json

import pytest

from ghost_agent.core.agent import GhostAgent


def make_agent(tool_names=("web_search", "execute", "file_system", "introspect")):
    agent = GhostAgent.__new__(GhostAgent)
    agent.available_tools = {n: (lambda **kw: None) for n in tool_names}
    return agent


def parse(content, msg=None, agent=None):
    agent = agent or make_agent()
    return agent._parse_assistant_tool_calls(content, msg if msg is not None else {})


def first_call(tool_calls):
    fn = tool_calls[0]["function"]
    args = fn.get("arguments")
    if isinstance(args, str):
        args = json.loads(args)
    return fn["name"], args


class TestXmlShapes:
    def test_canonical_xml_tool_call(self):
        content = (
            "I'll search for that.\n"
            "<tool_call>\n<function name=\"web_search\">\n"
            "<parameter name=\"query\">postgresql 19 features</parameter>\n"
            "</function>\n</tool_call>"
        )
        tool_calls, ui_content, reason = parse(content)
        name, args = first_call(tool_calls)
        assert name == "web_search"
        assert args["query"] == "postgresql 19 features"
        assert reason == ""
        # Tool XML scrubbed from the user-facing content, prose kept.
        assert "<tool_call" not in ui_content
        assert "I'll search for that." in ui_content

    def test_bare_function_tag_is_healed(self):
        content = (
            "<function name=\"web_search\">\n"
            "<parameter name=\"query\">tor search</parameter>\n"
            "</function>"
        )
        tool_calls, _, _ = parse(content)
        name, args = first_call(tool_calls)
        assert name == "web_search"
        assert args["query"] == "tor search"

    def test_sloppy_attribute_syntax_is_healed(self):
        # Qwen 3.6+ shapes: <function = "x">, <parameter name = "y">
        content = (
            "<tool_call>\n<function = \"execute\">\n"
            "<parameter name = \"command\">ls -la</parameter>\n"
            "</function>\n</tool_call>"
        )
        tool_calls, _, _ = parse(content)
        name, args = first_call(tool_calls)
        assert name == "execute"
        assert args["command"] == "ls -la"

    def test_tool_call_inside_think_is_not_executed(self):
        content = (
            "<think>maybe I should <tool_call><function name=\"execute\">"
            "</function></tool_call></think>\nJust an answer, no tools."
        )
        tool_calls, _, _ = parse(content)
        assert tool_calls == []

    def test_plain_prose_yields_no_calls_and_no_failure(self):
        tool_calls, ui_content, reason = parse("The answer is 42.")
        assert tool_calls == []
        assert reason == ""
        assert "42" in ui_content

    def test_truncated_tool_call_sets_reason_and_still_recovers(self):
        # Opened, never closed — upstream token cap severed the stream. The
        # parser flags the truncation (so the recovery branch can tell the
        # model "your output was cut off, shorten it") AND still best-effort
        # recovers the dangling call via the unclosed-parameter repair.
        content = (
            "<tool_call>\n<function name=\"execute\">\n"
            "<parameter name=\"content\">print('hi'"
        )
        tool_calls, _, reason = parse(content)
        assert reason == "truncated"
        name, args = first_call(tool_calls)
        assert name == "execute"
        assert args["content"].startswith("print('hi'")


class TestFallbacks:
    def test_native_tool_calls_pass_through(self):
        native = [{
            "id": "call_1", "type": "function",
            "function": {"name": "web_search",
                         "arguments": json.dumps({"query": "x"})},
        }]
        tool_calls, _, _ = parse("plain text", msg={"tool_calls": native})
        name, args = first_call(tool_calls)
        assert name == "web_search"
        assert args["query"] == "x"

    def test_raw_json_tool_call_recovered(self):
        content = json.dumps(
            {"name": "web_search", "arguments": {"query": "raw json"}})
        tool_calls, _, _ = parse(content)
        name, args = first_call(tool_calls)
        assert name == "web_search"
        assert args["query"] == "raw json"

    def test_reasoning_content_field_merged_for_history(self):
        # reasoning_content arrives as a separate field on some servers;
        # the parser treats it as a think block (not user-facing).
        msg = {"reasoning_content": None}
        tool_calls, ui_content, reason = parse("hello", msg=msg)
        assert tool_calls == []
        assert "hello" in ui_content
