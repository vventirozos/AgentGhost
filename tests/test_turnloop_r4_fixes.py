"""Turn-loop review, Round 4 (§ turn-loop R4, 2026-08-19) — behavioral pins.

R4's converging findings: the static `_PURE_TRIGGER_TOOLS` was blind to
RUNTIME-registered tools (a dynamic no-arg macro's native call lost to an
echoed tag — the recurring proxy class, fixed via `_STATIC_TOOL_NAMES`), one
unadvertised kb action dodged `is_mutating`, and four R3-diff sites were
unpinned. All pins drive the REAL methods; each mutation-verified.
"""

import argparse
import json

import pytest

from ghost_agent.core.agent import (GhostAgent, TurnState,
                                    _PURE_TRIGGER_TOOLS, _STATIC_TOOL_NAMES)
from ghost_agent.core.strikes import StrikeLedger
from unittest.mock import AsyncMock, MagicMock


def _parser_agent(names=("dream_mode", "web_search", "recall", "execute",
                         "morning_briefing")):
    agent = GhostAgent.__new__(GhostAgent)
    agent.available_tools = {n: (lambda **kw: None) for n in names}
    return agent


def _names(tcs):
    return [(t.get("function") or {}).get("name") for t in tcs]


def _args0(tcs):
    x = tcs[0]["function"]["arguments"]
    return json.loads(x) if isinstance(x, str) else x


_ECHO = ('I will call the <function name="web_search">'
         '<parameter name="query">weather</parameter></function> helper.')
_XML_RECALL = ('<tool_call><function name="recall">'
               '<parameter name="query">real</parameter></function></tool_call>')


class TestDynamicNoArgToolsUsable:
    """MAJOR (lens A+C): a runtime-registered tool (composed macro / acquired
    skill) is never in the static trigger set, but its empty-args native call
    must still win over an echoed tag — dispatching it errors recoverably at
    worst; executing an echo (possibly mutating) is unrecoverable."""

    def test_static_names_set_is_populated(self):
        assert "execute" in _STATIC_TOOL_NAMES
        assert "web_search" in _STATIC_TOOL_NAMES
        # a runtime macro name is, by construction, not static
        assert "morning_briefing" not in _STATIC_TOOL_NAMES

    def test_dynamic_no_arg_macro_wins_over_echoed_tag(self):
        agent = _parser_agent()
        msg = {"content": _ECHO, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "morning_briefing", "arguments": "{}"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_ECHO, msg)
        assert _names(tcs) == ["morning_briefing"]

    def test_static_non_trigger_empty_args_still_yields_to_xml(self):
        # control: web_search is STATIC and takes params — empty native args
        # must still yield to the rich XML call (R3 semantics preserved).
        agent = _parser_agent()
        msg = {"content": _XML_RECALL, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "web_search", "arguments": "{}"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_XML_RECALL, msg)
        assert _names(tcs) == ["recall"]


class TestUsableNativeGateEdgePins:
    """Lens C-c: the R3-diff sites that survived neutering."""

    def test_null_string_args_trigger_still_usable(self):
        # the '"null"' fast-path must classify as empty (→ trigger usable),
        # not as "has args" nor as unparseable.
        agent = _parser_agent()
        msg = {"content": _ECHO, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "dream_mode", "arguments": "null"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_ECHO, msg)
        assert _names(tcs) == ["dream_mode"]

    def test_list_literal_string_args_trigger_still_usable(self):
        agent = _parser_agent()
        msg = {"content": _ECHO, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "dream_mode", "arguments": "[]"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_ECHO, msg)
        assert _names(tcs) == ["dream_mode"]

    def test_scalar_args_are_degenerate(self):
        # arguments=42 (non-str, non-dict, non-None) → degenerate → XML wins.
        agent = _parser_agent()
        msg = {"content": _XML_RECALL, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "web_search", "arguments": 42}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(_XML_RECALL, msg)
        assert _names(tcs) == ["recall"]

    def test_padded_raw_json_is_not_healed(self):
        # `_looks_raw_json` strips leading whitespace: a padded raw-JSON call
        # whose command contains a real fn-tag literal must recover unmangled.
        agent = _parser_agent(("execute",))
        cmd = "grep -n '<function name=' x.py"
        content = "\n  " + json.dumps({"name": "execute",
                                       "arguments": {"command": cmd}})
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, {})
        assert _names(tcs) == ["execute"]
        assert _args0(tcs)["command"] == cmd


# ── dispatch harness ─────────────────────────────────────────────────────────
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


async def _run_dup(tool_name, args):
    agent = _dispatch_agent()
    calls = {"n": 0}

    async def t(**kwargs):
        calls["n"] += 1
        return "ok"

    agent.available_tools = {tool_name: t}
    tc = [{"id": f"c{i}", "type": "function",
           "function": {"name": tool_name, "arguments": json.dumps(args)}}
          for i in range(2)]
    ts = _ts(tool_calls=tc)
    await agent._dispatch_and_process_tool_batch(ts)
    return calls["n"]


class TestIsMutatingCompletenessPins:
    """Lens B MINOR + lens C-c case-insensitivity: aliased / unadvertised /
    case-variant kb mutations must not collapse."""

    @pytest.mark.asyncio
    async def test_kb_update_profile_action_not_collapsed(self):
        # unadvertised pass-through to tool_update_profile (a write).
        assert await _run_dup("knowledge_base",
                              {"action": "update_profile", "category": "root",
                               "key": "city", "value": "Athens"}) == 2

    @pytest.mark.asyncio
    async def test_kb_uppercase_transcribe_not_collapsed(self):
        assert await _run_dup("knowledge_base",
                              {"action": "TRANSCRIBE",
                               "filename": "t.mp4"}) == 2

    @pytest.mark.asyncio
    async def test_kb_padded_ingest_not_collapsed(self):
        assert await _run_dup("knowledge_base",
                              {"action": " ingest \n",
                               "filename": "a.pdf"}) == 2
