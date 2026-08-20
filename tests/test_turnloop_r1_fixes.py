"""Turn-loop review, Round 1 (§ turn-loop R1, 2026-08-19) — behavioral pins.

These drive the REAL methods (`_parse_assistant_tool_calls`,
`_dispatch_and_process_tool_batch`) rather than matching source strings, so a
refactor that neuters the guarded behavior FAILS here (the "token pins vs
executed pins" lesson). Each test was mutation-verified: it fails on the
pre-fix / neutered code and passes on the shipped code.

Pinned findings:
  * CRITICAL — native `tool_calls` precedence at the routing gate: a native
    reply whose content preamble merely MENTIONS tool XML (`<function`/
    `<tool_call>`) must NOT be routed into the XML healer and drop the real
    native calls. (lens B)
  * MAJOR M2 — batch-dedup collapse denylist gap: byte-identical
    side-effecting calls (postgres_admin / jobs / manage_composed_skills) in
    one batch must NOT collapse to a single execution. (lens A)
  * MAJOR M1 / MD7,MD8,MD9 — the live pre-flight repeat-failure guard's
    block / record / world-changed-reset wiring, previously pinned only by
    `inspect.getsource` substring checks that survive behavior removal.
    (lens A + lens C)
"""

import json

import pytest

from ghost_agent.core.agent import GhostAgent, TurnState
from ghost_agent.core.strikes import StrikeLedger
from unittest.mock import AsyncMock, MagicMock


# ── parser harness (mirrors test_parse_assistant_tool_calls.py) ──────────────
def _parser_agent(tool_names=("web_search", "execute", "file_system",
                              "introspect", "recall")):
    agent = GhostAgent.__new__(GhostAgent)
    agent.available_tools = {n: (lambda **kw: None) for n in tool_names}
    return agent


def _names(tool_calls):
    return [(t.get("function") or {}).get("name") for t in tool_calls]


class TestNativePrecedenceAtRoutingGate:
    """CRITICAL: native tool_calls win over a content preamble that only
    MENTIONS tool XML. Pre-fix these dropped the native call and stamped a
    system_parse_error (reason='truncated')."""

    def test_native_call_survives_function_mention_in_content(self):
        # R2: a prose `<function>` mention is NOT a real tag (has_tool_tag now
        # requires the name=/= dialect), so the native call is honored AND the
        # prose is preserved verbatim (the R1 scrub-of-prose was over-eager —
        # "<function>" here is the model's own words, not a tool-call artifact).
        agent = _parser_agent()
        content = "I'll use the <function> calling convention to look this up."
        msg = {
            "content": content,
            "tool_calls": [{
                "id": "call_native_1", "type": "function",
                "function": {"name": "recall",
                             "arguments": '{"query": "postgres tuning"}'},
            }],
        }
        tool_calls, ui, reason = agent._parse_assistant_tool_calls(content, msg)
        assert _names(tool_calls) == ["recall"]
        assert "system_parse_error" not in _names(tool_calls)
        args = tool_calls[0]["function"]["arguments"]
        args = json.loads(args) if isinstance(args, str) else args
        assert args["query"] == "postgres tuning"
        # prose preserved — the mention is legitimate text, not scrubbed
        assert "<function>" in ui

    def test_native_call_survives_tool_call_mention_in_content(self):
        # A bare `<tool_call>` remains a strong wrapper marker (kept, so a
        # severed/truncated real call still routes to the XML truncation hint),
        # but the native call must still win via the usable-native gate.
        agent = _parser_agent()
        content = "Let me check. The <tool_call> protocol wraps calls."
        msg = {
            "content": content,
            "tool_calls": [{
                "id": "call_native_2", "type": "function",
                "function": {"name": "web_search",
                             "arguments": '{"query": "y"}'},
            }],
        }
        tool_calls, ui, reason = agent._parse_assistant_tool_calls(content, msg)
        assert _names(tool_calls) == ["web_search"]
        assert "system_parse_error" not in _names(tool_calls)

    def test_xml_only_path_still_parses_when_no_native_calls(self):
        # regression guard: the XML healer must remain the fallback when the
        # server did NOT populate native tool_calls.
        agent = _parser_agent()
        content = (
            "I'll search.\n<tool_call>\n<function name=\"web_search\">\n"
            "<parameter name=\"query\">z</parameter>\n</function>\n</tool_call>"
        )
        tool_calls, ui, reason = agent._parse_assistant_tool_calls(content, {})
        assert _names(tool_calls) == ["web_search"]
        assert reason == ""
        assert "<tool_call" not in ui
        assert "I'll search." in ui

    def test_native_only_content_untouched(self):
        agent = _parser_agent()
        content = "Sure, let me look."
        msg = {"content": content, "tool_calls": [{
            "id": "c", "type": "function",
            "function": {"name": "recall", "arguments": '{"query": "q"}'}}]}
        tool_calls, ui, reason = agent._parse_assistant_tool_calls(content, msg)
        assert _names(tool_calls) == ["recall"]
        assert ui == "Sure, let me look."


# ── dispatch harness (mirrors test_dispatch_pipeline_extraction.py) ──────────
def _dispatch_agent():
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)
    agent.available_tools = {}
    agent.disabled_tools = set()
    return agent


def _ts(**over):
    fields = dict(
        _constraint_steer_pending=None,
        _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False,
        _request_sys3_prev_justification="",
        consecutive_parse_errors=0,
        current_plan_json="",
        execution_failure_count=0,
        final_ai_content="",
        fname="",
        force_final_response=False,
        force_stop=False,
        forget_was_called=False,
        last_was_failure=True,
        preflight_blocks_this_request=0,
        request_sandbox_state="",
        transient_failure_count=0,
        tool_calls=[],
        msg={"role": "assistant", "content": ""},
        ui_content="",
        parse_failure_reason="",
        model="test-model",
        last_user_content="do the thing",
        char_budget=4000,
        strikes=StrikeLedger(),
        task_tree=MagicMock(),
        _user_batch_intent=None,
        _request_constraints=[],
        repeated_action_steered=set(),
        messages=[],
        seen_tools=set(),
        executed_idempotent=set(),
        raw_tools_called=set(),
        tool_usage={},
        tools_run_this_turn=[],
        request_state=MagicMock(),
    )
    fields.update(over)
    return TurnState(**fields)


def _call(name, args):
    return {"id": f"id_{name}_{hash(json.dumps(args)) & 0xffff}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class TestBatchCollapseSideEffectingTools:
    """M2: byte-identical side-effecting calls in one batch must each run —
    dropping a real write is silent state drift (the design's stated bias:
    'executing a redundant read is merely wasteful, so err toward not
    collapsing')."""

    @pytest.mark.asyncio
    async def test_duplicate_postgres_admin_writes_not_collapsed(self):
        agent = _dispatch_agent()
        calls = {"n": 0}

        async def pg(**kwargs):
            calls["n"] += 1
            return "INSERT 0 1"

        agent.available_tools = {"postgres_admin": pg}
        args = {"sql": "INSERT INTO t(x) VALUES (1)"}
        tc = [_call("postgres_admin", args), _call("postgres_admin", args)]
        # identical ids would be de-duped by id; force distinct ids, same args
        tc[0]["id"], tc[1]["id"] = "p0", "p1"
        ts = _ts(tool_calls=tc)
        await agent._dispatch_and_process_tool_batch(ts)
        assert calls["n"] == 2  # both INSERTs executed, not collapsed

    @pytest.mark.asyncio
    async def test_duplicate_jobs_actions_not_collapsed(self):
        agent = _dispatch_agent()
        calls = {"n": 0}

        async def jobs(**kwargs):
            calls["n"] += 1
            return "job started"

        agent.available_tools = {"jobs": jobs}
        args = {"action": "start", "name": "worker"}
        tc = [_call("jobs", args), _call("jobs", args)]
        tc[0]["id"], tc[1]["id"] = "j0", "j1"
        ts = _ts(tool_calls=tc)
        await agent._dispatch_and_process_tool_batch(ts)
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_duplicate_composed_skill_runs_not_collapsed(self):
        agent = _dispatch_agent()
        calls = {"n": 0}

        async def macro(**kwargs):
            calls["n"] += 1
            return "ran macro"

        agent.available_tools = {"manage_composed_skills": macro}
        args = {"action": "run", "name": "deploy"}
        tc = [_call("manage_composed_skills", args),
              _call("manage_composed_skills", args)]
        tc[0]["id"], tc[1]["id"] = "s0", "s1"
        ts = _ts(tool_calls=tc)
        await agent._dispatch_and_process_tool_batch(ts)
        assert calls["n"] == 2


class TestPreflightGuardIsDrivenPinned:
    """M1 / MD7-9: the live pre-flight repeat-failure guard, pinned through a
    REAL dispatch (not a getsource substring). repeat_threshold=2 → the 3rd
    identical failing call is intercepted pre-dispatch."""

    @pytest.mark.asyncio
    async def test_third_identical_failure_is_blocked_predispatch(self):
        agent = _dispatch_agent()
        calls = {"n": 0}

        async def failing(**kwargs):
            calls["n"] += 1
            return "Error: boom on a.txt"

        agent.available_tools = {"file_system": failing}
        args = {"operation": "read", "path": "a.txt"}

        for _ in range(2):  # two recorded failures arm the guard
            ts = _ts(tool_calls=[_call("file_system", args)])
            await agent._dispatch_and_process_tool_batch(ts)
        assert calls["n"] == 2

        # third identical re-issue: BLOCKED before dispatch (MD7 block wiring)
        ts3 = _ts(tool_calls=[_call("file_system", args)])
        await agent._dispatch_and_process_tool_batch(ts3)
        assert calls["n"] == 2  # not dispatched a third time
        assert ts3.preflight_blocks_this_request >= 1
        tool_msgs = [m for m in ts3.messages if m.get("role") == "tool"]
        assert any("pre-flight guard" in (m.get("content") or "").lower()
                   for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_record_wiring_arms_would_repeat(self):
        # MD9: record() must fire on a failed dispatch. After two identical
        # failures, the guard's own would_repeat() reports the block.
        agent = _dispatch_agent()

        async def failing(**kwargs):
            return "Error: still broken"

        agent.available_tools = {"file_system": failing}
        args = {"operation": "read", "path": "b.txt"}
        for _ in range(2):
            ts = _ts(tool_calls=[_call("file_system", args)])
            await agent._dispatch_and_process_tool_batch(ts)

        from ghost_agent.core.agent import guard_key_target, primary_target_from_args
        key = guard_key_target(primary_target_from_args(args),
                               "")  # a_hash fallback unused when target present
        assert agent._failure_guard.would_repeat("file_system", key, "read")

    @pytest.mark.asyncio
    async def test_successful_mutation_clears_the_guard(self):
        # MD8: a SUCCESSFUL state-mutating call must clear the armed guard so a
        # previously-blocked call is allowed again (world-changed reset).
        agent = _dispatch_agent()
        reads = {"n": 0}

        async def fs(**kwargs):
            if kwargs.get("operation") == "write":
                return "written ok"
            reads["n"] += 1
            return "Error: boom on c.txt"

        agent.available_tools = {"file_system": fs}
        read_args = {"operation": "read", "path": "c.txt"}
        for _ in range(2):  # arm the guard on the read
            ts = _ts(tool_calls=[_call("file_system", read_args)])
            await agent._dispatch_and_process_tool_batch(ts)
        assert reads["n"] == 2

        # a successful write mutates the world → clears recorded failures
        ts_w = _ts(tool_calls=[_call("file_system",
                                     {"operation": "write", "path": "d.txt",
                                      "content": "x"})])
        await agent._dispatch_and_process_tool_batch(ts_w)

        # the read is now allowed through again (guard was cleared)
        ts_r = _ts(tool_calls=[_call("file_system", read_args)])
        await agent._dispatch_and_process_tool_batch(ts_r)
        assert reads["n"] == 3  # dispatched again, not blocked


class TestParserInstrumentHardening:
    """Lens C flagged these live parser behaviors as defended only by
    re-implementations / value-blind source-string matches (SURVIVED
    mutation). Pin them behaviorally against the real parser."""

    def test_degenerate_flood_is_capped(self):
        # A degenerate generation looping on a malformed <tool_call> shape must
        # not drain the strike counter: the parser caps blocks at
        # _MAX_TOOL_CALL_BLOCKS (=5). Value-blind source checks let the cap be
        # widened to 100000 undetected.
        agent = _parser_agent()
        flood = "<tool_call>garbage no function tag here</tool_call>" * 20
        tool_calls, _, _ = agent._parse_assistant_tool_calls(flood, {})
        assert len(tool_calls) <= 5
        assert all(n == "system_parse_error" for n in _names(tool_calls))

    def test_native_repair_is_wired_at_the_call_site(self):
        # The repair HELPER is unit-tested elsewhere; this pins that the parser
        # METHOD actually invokes it. Feed the verbatim live corruption
        # (session e260ae3e) as a native tool_calls array.
        agent = _parser_agent(("introspect", "list_lessons", "self_state",
                               "execute"))
        corrupt = ("summary</parameter>\n</function>\n</tool_call>\n<tool_call>\n"
                   "<function=list_lessons>\n<parameter=scope>\nall")
        msg = {"content": "", "tool_calls": [{
            "id": "call_x", "type": "function",
            "function": {"name": "introspect",
                         "arguments": json.dumps({"action": corrupt,
                                                  "limit": 10})}}]}
        tool_calls, _, _ = agent._parse_assistant_tool_calls("", msg)
        names = _names(tool_calls)
        # primary introspect recovered to the intended action…
        assert names[0] == "introspect"
        a0 = tool_calls[0]["function"]["arguments"]
        a0 = json.loads(a0) if isinstance(a0, str) else a0
        assert a0["action"] == "summary"
        # …and the leaked second call is recovered, not dropped.
        assert "list_lessons" in names
