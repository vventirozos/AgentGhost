"""Turn-loop review, Round 3 (§ turn-loop R3, 2026-08-19) — behavioral pins.

R3 attacked R2's own fixes and again found the SAME class ("guarded a proxy,
not the thing"): the usable-native gate re-opened its parent regression for
no-arg tools and mis-classified malformed args; the raw-JSON heal corrupted an
argument value; the collapse allowlist trusted a pre-heal action. All pins
drive the REAL methods and were mutation-verified.
"""

import argparse
import json

import pytest

from ghost_agent.core.agent import GhostAgent, TurnState, _PURE_TRIGGER_TOOLS
from ghost_agent.core.strikes import StrikeLedger
from unittest.mock import AsyncMock, MagicMock


def _parser_agent(names=("dream_mode", "web_search", "recall", "execute",
                         "file_system")):
    agent = GhostAgent.__new__(GhostAgent)
    agent.available_tools = {n: (lambda **kw: None) for n in names}
    return agent


def _names(tcs):
    return [(t.get("function") or {}).get("name") for t in tcs]


def _args0(tcs):
    x = tcs[0]["function"]["arguments"]
    return json.loads(x) if isinstance(x, str) else x


_XML = ('<tool_call><function name="web_search">'
        '<parameter name="query">x</parameter></function></tool_call>')
_XML_RECALL = ('<tool_call><function name="recall">'
               '<parameter name="query">real</parameter></function></tool_call>')


class TestUsableNativeGateHardened:
    """F1a/F1b/F1c: the gate must distinguish a COMPLETE no-arg native call
    from a DEGENERATE one, and validate args are a real dict."""

    def test_pure_trigger_set_is_derived_and_excludes_execute(self):
        assert "dream_mode" in _PURE_TRIGGER_TOOLS
        assert "self_play" in _PURE_TRIGGER_TOOLS
        assert "stop_self_play" in _PURE_TRIGGER_TOOLS
        # execute has params (just none required) — a complete call needs args,
        # so an empty execute must NOT be treated as a trigger.
        assert "execute" not in _PURE_TRIGGER_TOOLS

    def test_no_arg_trigger_wins_over_echoed_tag_in_prose(self):
        # F1a: dream_mode (no-arg) + prose that ECHOES a real tool tag → the
        # native trigger must win; the echo must NOT be executed.
        agent = _parser_agent()
        content = f"entering dream mode. (a search looks like {_XML})"
        msg = {"content": content, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "dream_mode", "arguments": "{}"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, msg)
        assert _names(tcs) == ["dream_mode"]

    def test_truncated_native_args_string_yields_to_xml(self):
        # F1b: a truncated/unparseable native args string is degenerate.
        agent = _parser_agent()
        msg = {"content": _XML_RECALL, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "web_search", "arguments": '{"query":"trunc'}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_XML_RECALL, msg)
        assert _names(tcs) == ["recall"]

    def test_list_native_args_yields_to_xml(self):
        # F1c: args that parse to a list (not a dict) are degenerate.
        agent = _parser_agent()
        msg = {"content": _XML_RECALL, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "web_search", "arguments": "[1,2,3]"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_XML_RECALL, msg)
        assert _names(tcs) == ["recall"]

    def test_empty_args_execute_still_yields_to_rich_xml(self):
        # Q1 preserved: execute is not a trigger, so empty native execute must
        # still yield to the rich XML execute call.
        agent = _parser_agent()
        xml = ('<tool_call><function name="execute">'
               '<parameter name="content">print("real")</parameter>'
               '</function></tool_call>')
        msg = {"content": xml, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "execute", "arguments": "{}"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(xml, msg)
        assert _names(tcs) == ["execute"]
        assert _args0(tcs).get("content") == 'print("real")'

    # Lens C-3 — the two previously-unpinned _has_args branches.
    def test_empty_dict_native_args_yields_to_xml(self):
        # dict branch: {} native args + rich XML → XML wins.
        agent = _parser_agent()
        msg = {"content": _XML_RECALL, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "web_search", "arguments": {}}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_XML_RECALL, msg)
        assert _names(tcs) == ["recall"]

    def test_none_native_args_yields_to_xml(self):
        # None branch: arguments=None + rich XML → XML wins (web_search is not
        # a trigger, so empty is degenerate).
        agent = _parser_agent()
        msg = {"content": _XML_RECALL, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "web_search", "arguments": None}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_XML_RECALL, msg)
        assert _names(tcs) == ["recall"]


class TestRawJsonArgNotCorrupted:
    """Lens C-1: the bare-function heal must not inject `<tool_call>` into a
    raw-JSON argument value."""

    def test_execute_command_with_real_function_tag_literal_uncorrupted(self):
        agent = _parser_agent()
        cmd = "grep -n '<function name=' x.py"
        content = json.dumps({"name": "execute",
                              "arguments": {"command": cmd}})
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, {})
        assert _names(tcs) == ["execute"]
        assert _args0(tcs)["command"] == cmd

    def test_file_write_content_with_function_tag_literal_uncorrupted(self):
        agent = _parser_agent()
        body = 'tag = "<function=deploy>"'
        content = json.dumps({"name": "file_system",
                              "arguments": {"operation": "write",
                                            "path": "p.py", "content": body}})
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, {})
        assert _names(tcs) == ["file_system"]
        assert _args0(tcs)["content"] == body


class TestMultiTagBareFunctionHeal:
    """Lens C-3: two bare real function tags (no wrapper) must BOTH parse."""

    def test_two_bare_function_tags_both_parse(self):
        agent = _parser_agent()
        content = ('<function name="web_search"><parameter name="query">one'
                   '</parameter></function>\n'
                   '<function name="recall"><parameter name="query">two'
                   '</parameter></function>')
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, {})
        assert _names(tcs) == ["web_search", "recall"]


# ── dispatch harness (F2a/F2b) ───────────────────────────────────────────────
def _dispatch_agent():
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args = argparse.Namespace(smart_memory=0.0, enable_preflight_guard=True)
    agent = GhostAgent(ctx)
    agent.available_tools = {}
    agent.disabled_tools = set()
    return agent


def _ts(**over):
    fields = dict(
        _constraint_steer_pending=None, _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False, _request_sys3_prev_justification="",
        consecutive_parse_errors=0, current_plan_json="",
        execution_failure_count=0, final_ai_content="", fname="",
        force_final_response=False, force_stop=False, forget_was_called=False,
        last_was_failure=True, preflight_blocks_this_request=0,
        request_sandbox_state="", transient_failure_count=0, tool_calls=[],
        msg={"role": "assistant", "content": ""}, ui_content="",
        parse_failure_reason="", model="test-model",
        last_user_content="do the thing", char_budget=4000,
        strikes=StrikeLedger(), task_tree=MagicMock(), _user_batch_intent=None,
        _request_constraints=[], repeated_action_steered=set(), messages=[],
        seen_tools=set(), executed_idempotent=set(), raw_tools_called=set(),
        tool_usage={}, tools_run_this_turn=[], request_state=MagicMock(),
    )
    fields.update(over)
    return TurnState(**fields)


def _call(name, args, cid):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


async def _run_dup(agent_name, args):
    agent = _dispatch_agent()
    calls = {"n": 0}

    async def t(**kwargs):
        calls["n"] += 1
        return "ok"

    agent.available_tools = {agent_name: t}
    ts = _ts(tool_calls=[_call(agent_name, args, "a0"),
                         _call(agent_name, args, "a1")])
    await agent._dispatch_and_process_tool_batch(ts)
    return calls["n"]


class TestMutatingAliasesDoNotCollapse:
    """F2a/F2b: a mutation classified by an ALIASED/omitted arg must not
    collapse."""

    @pytest.mark.asyncio
    async def test_kb_transcribe_alias_not_collapsed(self):
        # transcribe → ingest_document (a write) is healed inside the tool.
        assert await _run_dup("knowledge_base",
                              {"action": "transcribe", "filename": "t.mp4"}) == 2

    @pytest.mark.asyncio
    async def test_kb_ingest_alias_not_collapsed(self):
        assert await _run_dup("knowledge_base",
                              {"action": "ingest", "filename": "a.pdf"}) == 2

    @pytest.mark.asyncio
    async def test_kb_query_still_collapses(self):
        # regression: a real read still collapses.
        assert await _run_dup("knowledge_base",
                              {"action": "query", "filename": "d",
                               "question": "q"}) == 1

    @pytest.mark.asyncio
    async def test_file_system_copy_not_collapsed(self):
        assert await _run_dup("file_system",
                              {"operation": "copy", "path": "a",
                               "dest": "b"}) == 2
