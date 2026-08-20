"""Turn-loop review, Round 2 (§ turn-loop R2, 2026-08-19) — behavioral pins.

R2 attacked R1's own fixes and found the SAME defect class in both ("guarded a
proxy, not the thing"), plus a shared root cause: `has_tool_tag` was a bare
substring match. These pins drive the REAL methods; each was mutation-verified
(fails on the pre-fix / neutered code, passes on the shipped code).

Findings pinned:
  * ROOT — `has_tool_tag` now requires a REAL tool tag, so a prose mention of
    `<function>` and a raw-JSON tool call whose args contain `<function`/
    `<tool_call>` are no longer misrouted into the XML healer (lens C-1, C-3).
  * MAJOR (lens A) — usable-native gate: a degenerate native call (empty args,
    or a name the agent doesn't expose) must NOT shadow a rich XML call.
  * CRITICAL (lens B) — batch-collapse allowlist: runtime-registered
    macro/skill names (unknowable to a denylist) must NOT collapse.
  * MAJOR (lens C-2) — three previously-unpinned live guards: mid-loop strike
    cap, arg unescaping, truncation-error dedupe.
  * MINOR (lens C-4) — the pre-flight guard's enable flag actually gates it.
"""

import argparse
import json

import pytest

from ghost_agent.core.agent import GhostAgent, TurnState
from ghost_agent.core.strikes import StrikeLedger
from unittest.mock import AsyncMock, MagicMock


# ── parser harness ───────────────────────────────────────────────────────────
def _parser_agent(names=("execute", "web_search", "recall", "file_system",
                         "introspect")):
    agent = GhostAgent.__new__(GhostAgent)
    agent.available_tools = {n: (lambda **kw: None) for n in names}
    return agent


def _names(tcs):
    return [(t.get("function") or {}).get("name") for t in tcs]


def _args0(tcs):
    x = tcs[0]["function"]["arguments"]
    return json.loads(x) if isinstance(x, str) else x


class TestRealTagRoutingGate:
    """ROOT: has_tool_tag requires a real `<function name=`/`<function=` tag or
    a `<tool_call>` wrapper — not a bare substring."""

    def test_raw_json_call_with_function_literal_in_args_recovers(self):
        # lens C-1 (R3-corrected): the literal MUST be a real `<function name=`
        # tag so it engages `_FN_TAG_RE` + the bare-function heal — the earlier
        # `<function` (no name=) matched nothing and pinned nothing (vacuous).
        # The command must be recovered UNCORRUPTED (the heal must not inject
        # `<tool_call>` into the raw-JSON argument value).
        agent = _parser_agent()
        cmd = "grep -n '<function name=' x.py"
        content = json.dumps({"name": "execute", "arguments": {"command": cmd}})
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, {})
        assert _names(tcs) == ["execute"]
        assert _args0(tcs)["command"] == cmd  # exact, uncorrupted

    def test_raw_json_write_with_tool_call_literal_in_args_recovers(self):
        # lens C-1b: a file write whose content contains `<tool_call>` literal.
        agent = _parser_agent()
        content = json.dumps({"name": "file_system",
                              "arguments": {"operation": "write", "path": "p.py",
                                            "content": "x = '<tool_call>'"}})
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, {})
        assert _names(tcs) == ["file_system"]
        assert _args0(tcs)["content"] == "x = '<tool_call>'"

    def test_prose_reply_mentioning_function_is_not_corrupted(self):
        # lens C-3/A2: a plain prose reply that explains `<function>` to the
        # user (no native calls) must stay prose — no system_parse_error, no
        # truncated reply.
        agent = _parser_agent()
        content = "In XML you write <function> to call a tool. Hope that helps!"
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, {})
        assert tcs == []
        assert "system_parse_error" not in _names(tcs)
        assert ui == content  # preserved verbatim

    def test_real_bare_function_tag_still_heals_and_parses(self):
        # regression: a REAL bare <function name=...> (no wrapper) must still
        # be wrapped + parsed.
        agent = _parser_agent()
        content = ('<function name="web_search">\n'
                   '<parameter name="query">z</parameter>\n</function>')
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, {})
        assert _names(tcs) == ["web_search"]
        assert "<function" not in ui

    def test_canonical_xml_still_parses(self):
        agent = _parser_agent()
        content = ('I search.\n<tool_call>\n<function name="web_search">\n'
                   '<parameter name="query">z</parameter>\n</function>\n</tool_call>')
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, {})
        assert _names(tcs) == ["web_search"]
        assert "I search." in ui


class TestUsableNativeGate:
    """MAJOR (lens A): a degenerate native call must not shadow a rich XML
    call sitting in the content."""

    def test_empty_args_native_yields_to_rich_xml(self):
        # Q1: native execute {} + content rich execute XML → XML wins.
        agent = _parser_agent()
        content = ('<tool_call><function name="execute">'
                   '<parameter name="content">print("real intended code")'
                   '</parameter></function></tool_call>')
        msg = {"content": content, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "execute", "arguments": "{}"}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, msg)
        assert _names(tcs) == ["execute"]
        assert _args0(tcs).get("content") == 'print("real intended code")'

    def test_unavailable_native_name_yields_to_rich_xml(self):
        # Q1b: native name not in available_tools + content rich web_search.
        agent = _parser_agent()
        content = ('<tool_call><function name="web_search">'
                   '<parameter name="query">real query</parameter>'
                   '</function></tool_call>')
        msg = {"content": content, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "nonexistent_tool", "arguments": '{"x": 1}'}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, msg)
        assert _names(tcs) == ["web_search"]

    def test_usable_native_still_wins_over_real_xml(self):
        # A usable native call (available name + real args) still takes
        # precedence over a competing real XML call (native-tools authority).
        agent = _parser_agent()
        content = ('<tool_call><function name="web_search">'
                   '<parameter name="query">from-xml</parameter>'
                   '</function></tool_call>')
        msg = {"content": content, "tool_calls": [{
            "id": "n", "type": "function",
            "function": {"name": "recall", "arguments": '{"query": "from-native"}'}}]}
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, msg)
        assert _names(tcs) == ["recall"]
        # and the real XML call text is scrubbed from the user reply
        assert "<function" not in ui


# ── dispatch harness ─────────────────────────────────────────────────────────
def _dispatch_agent(preflight=True):
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args = argparse.Namespace(smart_memory=0.0,
                                  enable_preflight_guard=preflight)
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


class TestCollapseAllowlistDynamicTools:
    """CRITICAL (lens B): runtime-registered side-effecting tools (macros,
    acquired skills) are unknown to any static set — the allowlist makes them
    collapse-unsafe by default."""

    @pytest.mark.asyncio
    async def test_dynamic_macro_duplicates_not_collapsed(self):
        agent = _dispatch_agent()
        calls = {"n": 0}

        async def macro(**kwargs):
            calls["n"] += 1
            return "ran"

        # a name no static denylist could ever contain
        agent.available_tools = {"deploy_prod_macro": macro}
        args = {"env": "prod"}
        ts = _ts(tool_calls=[_call("deploy_prod_macro", args, "d0"),
                             _call("deploy_prod_macro", args, "d1")])
        await agent._dispatch_and_process_tool_batch(ts)
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_known_read_still_collapses(self):
        agent = _dispatch_agent()
        calls = {"n": 0}

        async def rd(**kwargs):
            calls["n"] += 1
            return f"r{calls['n']}"

        agent.available_tools = {"recall": rd}
        args = {"query": "same"}
        ts = _ts(tool_calls=[_call("recall", args, "r0"),
                             _call("recall", args, "r1"),
                             _call("recall", args, "r2")])
        await agent._dispatch_and_process_tool_batch(ts)
        assert calls["n"] == 1  # collapsed
        assert len([m for m in ts.messages if m.get("role") == "tool"]) == 3


class TestPreviouslyUnpinnedGuards:
    """MAJOR (lens C-2): three live guards that were deletable-green."""

    @pytest.mark.asyncio
    async def test_arg_unescaping_at_dispatch(self):
        agent = _dispatch_agent()
        box = {}

        async def cap(**kwargs):
            box.update(kwargs)
            return "ok"

        agent.available_tools = {"web_search": cap}
        ts = _ts(tool_calls=[_call("web_search",
                                   {"query": "a &amp; b &lt;x&gt;"}, "w0")])
        await agent._dispatch_and_process_tool_batch(ts)
        assert box["query"] == "a & b <x>"

    @pytest.mark.asyncio
    async def test_mid_loop_strike_cap_stops_processing(self):
        # A degenerate response carrying many system_parse_error entries must
        # not drain the whole list: the mid-loop cap breaks once
        # execution_failure_count hits 6, so the 7th/8th get no tool message.
        agent = _dispatch_agent()
        tc = [{"id": f"e{i}", "type": "function",
               "function": {"name": "system_parse_error", "arguments": "{}"}}
              for i in range(8)]
        ts = _ts(tool_calls=tc)
        await agent._dispatch_and_process_tool_batch(ts)
        tool_msgs = [m for m in ts.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 6  # capped at 6; last two skipped
        assert ts.execution_failure_count == 6

    def test_truncation_error_deduped_to_one(self):
        # several truncated <tool_call> fragments → exactly ONE synthetic error
        # (not one strike per fragment).
        agent = _parser_agent()
        content = '<tool_call>\n<function name="execute"\n' * 3
        tcs, ui, reason = agent._parse_assistant_tool_calls(content, {})
        assert reason == "truncated"
        assert _names(tcs).count("system_parse_error") == 1


class TestPreflightGuardEnableFlagGates:
    """MINOR (lens C-4): the enable flag actually gates the guard (both ways),
    using a real Namespace so the default is exercised, not a MagicMock."""

    @pytest.mark.asyncio
    async def test_guard_off_does_not_block(self):
        agent = _dispatch_agent(preflight=False)
        assert agent._preflight_guard_enabled is False
        calls = {"n": 0}

        async def failing(**kwargs):
            calls["n"] += 1
            return "Error: boom on z.txt"

        agent.available_tools = {"file_system": failing}
        args = {"operation": "read", "path": "z.txt"}
        for _ in range(3):  # would block on the 3rd if the guard were ON
            ts = _ts(tool_calls=[_call("file_system", args, "z")])
            await agent._dispatch_and_process_tool_batch(ts)
        assert calls["n"] == 3  # never blocked — guard disabled

    @pytest.mark.asyncio
    async def test_guard_on_blocks_third(self):
        agent = _dispatch_agent(preflight=True)
        assert agent._preflight_guard_enabled is True
        calls = {"n": 0}

        async def failing(**kwargs):
            calls["n"] += 1
            return "Error: boom on z.txt"

        agent.available_tools = {"file_system": failing}
        args = {"operation": "read", "path": "z.txt"}
        for _ in range(3):
            ts = _ts(tool_calls=[_call("file_system", args, "z")])
            await agent._dispatch_and_process_tool_batch(ts)
        assert calls["n"] == 2  # third blocked
