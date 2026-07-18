"""#5 decomposition step 2 — `_dispatch_and_process_tool_batch` + TurnState.

First isolated coverage of the tool guard/dispatch/result pipeline, which
lived inline in handle_chat its whole life. The method was extracted VERBATIM
against `TurnState`; these tests pin the extraction contract:

  * the boundary protocol (False = continue the turn loop, True = break),
  * the TurnState repack in `finally` (state survives a raising tool path),
  * the guard behavior for empty batches and unknown tools.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent, TurnState


def _make_agent():
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)
    agent.available_tools = {}
    agent.disabled_tools = set()
    return agent


def _make_ts(**over):
    fields = dict(
        # mutated scalars
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
        last_was_failure=True,   # region sets it False early — observable
        preflight_blocks_this_request=0,
        request_sandbox_state="",
        transient_failure_count=0,
        # read-only / containers
        tool_calls=[],
        msg={"role": "assistant", "content": ""},
        ui_content="",
        parse_failure_reason="",
        model="test-model",
        last_user_content="do the thing",
        char_budget=4000,
        strikes=MagicMock(),
        task_tree=MagicMock(),
        _user_batch_intent=None,
        _request_constraints=[],
        repeated_action_steered=False,
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


# ── contract shape ───────────────────────────────────────────────────────────

def test_turnstate_mutated_fields_are_real_fields():
    field_names = set(TurnState.__dataclass_fields__)
    assert set(TurnState.MUTATED_FIELDS) <= field_names
    # every mutated field the extraction repacks must exist on the class
    assert "execution_failure_count" in TurnState.MUTATED_FIELDS
    assert "_request_sys3_fired_once" in TurnState.MUTATED_FIELDS  # the
    # cross-iteration SYSTEM-3 latch — the reason TurnState exists at all


def test_handle_chat_delegates_to_the_method():
    src = inspect.getsource(GhostAgent.handle_chat)
    assert "_dispatch_and_process_tool_batch(_ts)" in src
    # the pipeline body moved out — its landmarks must be gone from handle_chat
    assert "pending_idempotent = set()" not in src
    assert "SYSTEM 3 PIVOT" not in src


def test_method_carries_the_pipeline_landmarks():
    src = inspect.getsource(GhostAgent._dispatch_and_process_tool_batch)
    for marker in ("pending_idempotent = set()", "SYSTEM 3 PIVOT",
                   "Loop Breaker", "unescape_xml_values"):
        assert marker in src, f"pipeline landmark missing: {marker}"
    # the two rewritten control-flow statements
    assert "return True  # was `break`" in src
    assert "return False  # was `continue`" in src


# ── behavior ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_batch_continues_turn_loop():
    agent = _make_agent()
    ts = _make_ts(ui_content="hello there")
    out = await agent._dispatch_and_process_tool_batch(ts)
    assert out is False                      # loop-body tail → next turn
    assert ts.messages == [ts.msg]           # assistant msg appended
    assert ts.last_was_failure is False      # reset early in the region
    assert "hello there" in ts.final_ai_content  # ui preamble flushed


@pytest.mark.asyncio
async def test_unknown_tool_is_guarded_not_raised():
    agent = _make_agent()
    ts = _make_ts(tool_calls=[
        {"id": "t1", "type": "function",
         "function": {"name": "no_such_tool", "arguments": "{}"}},
    ])
    out = await agent._dispatch_and_process_tool_batch(ts)
    assert out is False
    assert "no_such_tool" in ts.raw_tools_called
    assert ts.tool_usage.get("no_such_tool") == 1
    # the guard answered INSIDE the batch (a steer/tool message), so the
    # conversation grew beyond just the assistant msg
    assert len(ts.messages) > 1


@pytest.mark.asyncio
async def test_finally_repacks_state_when_pipeline_raises():
    agent = _make_agent()
    ts = _make_ts(final_ai_content="A", ui_content="B", messages=None)
    with pytest.raises(AttributeError):
        # messages=None → .append raises AFTER the ui flush mutated
        # final_ai_content; the finally-repack must still deliver it
        await agent._dispatch_and_process_tool_batch(ts)
    assert ts.final_ai_content == "A\n\nB"


# ── batch-local duplicate collapse (2026-07-18) ─────────────────────────────
# One response can carry a runaway burst of byte-identical read-only calls
# (2026-07-17 22:14: 144 identical file_system reads in a single batch). The
# dispatch now executes such duplicates ONCE and fans the result out; every
# tool_call_id still gets a reply, and the no-progress ledger still counts
# each repeat. Mutating duplicates are never collapsed.

from ghost_agent.core.strikes import StrikeLedger


@pytest.mark.asyncio
async def test_duplicate_readonly_calls_execute_once_but_all_answered():
    agent = _make_agent()
    calls = {"n": 0}

    async def probe(**kwargs):
        calls["n"] += 1
        return f"probe-result-{calls['n']}"

    agent.available_tools = {"probe_tool": probe}
    tc = [
        {"id": f"t{i}", "type": "function",
         "function": {"name": "probe_tool", "arguments": '{"q": "same"}'}}
        for i in range(3)
    ]
    ts = _make_ts(tool_calls=tc, strikes=StrikeLedger(),
                  repeated_action_steered=set())
    await agent._dispatch_and_process_tool_batch(ts)

    assert calls["n"] == 1  # executed exactly once
    tool_msgs = [m for m in ts.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 3  # every call id answered
    assert len({m["content"] for m in tool_msgs}) == 1  # shared result


@pytest.mark.asyncio
async def test_distinct_readonly_calls_not_collapsed():
    agent = _make_agent()
    calls = {"n": 0}

    async def probe(**kwargs):
        calls["n"] += 1
        return f"probe-result-{calls['n']}"

    agent.available_tools = {"probe_tool": probe}
    tc = [
        {"id": "t0", "type": "function",
         "function": {"name": "probe_tool", "arguments": '{"q": "alpha"}'}},
        {"id": "t1", "type": "function",
         "function": {"name": "probe_tool", "arguments": '{"q": "beta"}'}},
    ]
    ts = _make_ts(tool_calls=tc, strikes=StrikeLedger(),
                  repeated_action_steered=set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_duplicate_mutating_calls_not_collapsed():
    agent = _make_agent()
    calls = {"n": 0}

    async def fs(**kwargs):
        calls["n"] += 1
        return "written"

    agent.available_tools = {"file_system": fs}
    tc = [
        {"id": f"m{i}", "type": "function",
         "function": {"name": "file_system",
                      "arguments": '{"operation": "write", "path": "a.txt", "content": "x"}'}}
        for i in range(2)
    ]
    ts = _make_ts(tool_calls=tc, strikes=StrikeLedger(),
                  repeated_action_steered=set())
    await agent._dispatch_and_process_tool_batch(ts)
    assert calls["n"] == 2  # a repeated write may be intentional — run both


@pytest.mark.asyncio
async def test_successful_file_write_resets_noprogress_ledger():
    """World-changed reset: a file mutation clears accumulated no-progress
    observations, so a post-fix re-observation is verification, not a loop."""
    agent = _make_agent()

    async def probe(**kwargs):
        return "same page content"

    async def fs(**kwargs):
        return "SUCCESS: wrote a.txt"

    agent.available_tools = {"probe_tool": probe, "file_system": fs}
    strikes = StrikeLedger()

    # accumulate one observation of the page
    ts1 = _make_ts(tool_calls=[
        {"id": "t0", "type": "function",
         "function": {"name": "probe_tool", "arguments": '{"q": "page"}'}},
    ], strikes=strikes, repeated_action_steered=set())
    await agent._dispatch_and_process_tool_batch(ts1)
    assert strikes.action_sigs  # observation counted

    # a successful write must clear the observation ledger
    ts2 = _make_ts(tool_calls=[
        {"id": "m0", "type": "function",
         "function": {"name": "file_system",
                      "arguments": '{"operation": "write", "path": "a.txt", "content": "x"}'}},
    ], strikes=strikes, repeated_action_steered=set())
    await agent._dispatch_and_process_tool_batch(ts2)
    assert strikes.action_sigs == {}
