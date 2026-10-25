"""
The System 3 "Crisis Pivot" — since §4RI (2026-10-25, request 43fe95a2) a
steer INSIDE the conversation, not two side LLM calls.

Why: on a long coding request the conversation's KV state outgrows
--cache-ram (llama: "prompt state size 5123 MiB exceeds cache size limit
4096 MiB, skipping"), so the generator/evaluator side calls displaced the
slot and the next turn re-processed ~110k tokens (~17 min) — for a planner
that saw a 1.5 KB summary. The main model holds the full context; it plans.

Covers:
  1. pivot_steer_message: the three approaches, the tool list, the error, the
     second-pivot wording, the episodic hint.
  2. handle_chat: the pivot fires at execution_failure_count == 4, injects
     the SYSTEM 3 PIVOT message, and makes NO extra LLM call.
  3. handle_chat: the first pivot decays the counter so the hard limit is not
     hit in the same breath.
  4. handle_chat: not triggered on the first failure.
  5. handle_chat: failures that continue still reach the hard limit.
  6. _recovery_episodes_hint: a memory lookup, empty for a member, no LLM.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext, pivot_steer_message
from ghost_agent.utils.logging import requester_role_context


class MockArgs:
    temperature = 0.7
    max_context = 8000
    smart_memory = 0.0
    perfect_it = False


def _make_agent(llm_responses=None):
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MockArgs()
    ctx.llm_client = MagicMock()
    ctx.llm_client.swarm_clients = None
    ctx.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "ok", "tool_calls": []}}]}
    )
    if llm_responses is not None:
        ctx.llm_client.chat_completion = AsyncMock(side_effect=llm_responses)
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string = MagicMock(return_value="")
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_playbook_context = MagicMock(return_value="")
    ctx.memory_system = MagicMock()
    ctx.memory_system.search = MagicMock(return_value="")
    ctx.episodic_memory = None
    ctx.cached_sandbox_state = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all = MagicMock(return_value="None.")
    return GhostAgent(ctx)


def _llm_ok(content="ok"):
    return {"choices": [{"message": {"content": content, "tool_calls": []}}]}


def _execute_tool_call():
    return {"choices": [{"message": {"content": None, "tool_calls": [{
        "id": "call_exec", "type": "function",
        "function": {"name": "execute", "arguments": json.dumps({"content": "bad code(", "language": "python"})}}]}}]}


EXEC_FAIL_RESULT = "EXIT CODE: 1\nSTDOUT/STDERR: SyntaxError: invalid syntax"


# ── 1. the steer text ───────────────────────────────────────────────────────

def test_the_steer_lays_out_three_approaches_over_the_real_tools():
    msg = pivot_steer_message(1, 4, "EXIT CODE: 1\nSTDOUT/STDERR: SyntaxError: invalid syntax",
                              ["web_search", "browser", "file_system"])
    assert msg.startswith("SYSTEM 3 PIVOT #1: 4 tool calls in a row have failed — the last: EXIT CODE: 1 STDOUT/STDERR: SyntaxError")
    assert "A: the most standard" in msg and "B: a defensive" in msg and "C: a creative" in msg
    assert "ONLY these tools: browser, file_system, web_search" in msg
    assert "Never invent a tool" in msg and "unavailable is not retried" in msg     # §4PC
    assert "then ACT on it now" in msg
    assert "first pivot" not in msg


def test_the_second_pivot_names_the_approach_that_failed_after_the_first():
    msg = pivot_steer_message(2, 5, "timeout", ["execute"], prev_error="the first error " * 60)
    assert "SYSTEM 3 PIVOT #2" in msg
    assert "after the first pivot failed too" in msg and "different from both" in msg
    assert len(msg) < 2500                                     # the previous error is capped


def test_the_steer_carries_the_recovery_hint_and_caps_the_error():
    msg = pivot_steer_message(1, 4, "x" * 5000, [], recovery_hint="### PAST RECOVERY STRATEGIES\n- [ok] a → b")
    assert msg.endswith("- [ok] a → b")
    assert "(none listed" in msg
    assert msg.index("x" * 600) and ("x" * 601) not in msg


# ── 2–5. handle_chat ────────────────────────────────────────────────────────

async def _run(agent, body):
    with patch("ghost_agent.tools.file_system.tool_list_files", new=AsyncMock(return_value="empty")), \
            patch("ghost_agent.core.agent.pretty_log"):
        return await agent.handle_chat(body, background_tasks=MagicMock())


@pytest.mark.asyncio
async def test_the_pivot_is_a_message_in_the_conversation_and_no_extra_llm_call():
    """Fails in the world where the pivot was two side calls: the fifth
    chat_completion would have been the strategy generator, not the model's
    next turn."""
    exec_call = _execute_tool_call()
    seen = []

    async def mock_llm(payload, **kwargs):
        seen.append([str(m.get("content", "")) for m in payload.get("messages", [])])
        if any("SYSTEM 3 PIVOT" in c for c in seen[-1]):
            return _llm_ok("Fixed with approach B.")
        return exec_call

    agent = _make_agent()
    agent.context.llm_client.chat_completion = mock_llm
    agent.available_tools["execute"] = AsyncMock(return_value=EXEC_FAIL_RESULT)
    body = {"messages": [{"role": "user", "content": "run my script"}], "model": "test-model"}
    out, _, _ = await _run(agent, body)
    pivots = [c for turn in seen for c in turn if "SYSTEM 3 PIVOT #1" in c]
    assert pivots, "no SYSTEM 3 PIVOT message reached the model"
    assert "three DIFFERENT ways forward" in pivots[0] and "execute" in pivots[0]
    assert "SyntaxError" in pivots[0]                          # the last error, in the steer
    # every payload is a turn of THIS conversation (carries the user's request):
    # no side call with a strategy prompt was made
    assert all(any("run my script" in c for c in turn) for turn in seen), "a side call left the conversation"
    assert all(not any("### PROPOSED STRATEGIES" in c or "System 3 strategist" in c for c in turn) for turn in seen)
    assert "Fixed with approach B." in out


@pytest.mark.asyncio
async def test_the_first_pivot_decays_the_counter():
    """After the pivot the model gets a fresh attempt: four failures, a pivot,
    then a success — the hard limit must not fire."""
    exec_call = _execute_tool_call()
    agent = _make_agent(llm_responses=[exec_call, exec_call, exec_call, exec_call, _llm_ok("All done!")])
    agent.available_tools["execute"] = AsyncMock(return_value=EXEC_FAIL_RESULT)
    body = {"messages": [{"role": "user", "content": "run script"}], "model": "test-model"}
    out, _, _ = await _run(agent, body)
    assert "failed too many times" not in out.lower() and "All done!" in out


@pytest.mark.asyncio
async def test_not_triggered_on_the_first_failure():
    exec_call = _execute_tool_call()
    seen = []

    async def mock_llm(payload, **kwargs):
        seen.append([str(m.get("content", "")) for m in payload.get("messages", [])])
        return exec_call if len(seen) == 1 else _llm_ok("Recovered!")

    agent = _make_agent()
    agent.context.llm_client.chat_completion = mock_llm
    agent.available_tools["execute"] = AsyncMock(side_effect=[EXEC_FAIL_RESULT, "EXIT CODE: 0\nSTDOUT/STDERR: ok"])
    body = {"messages": [{"role": "user", "content": "run script"}], "model": "test-model"}
    await _run(agent, body)
    assert not any("SYSTEM 3 PIVOT" in c for turn in seen for c in turn)


@pytest.mark.asyncio
async def test_failures_that_continue_still_reach_the_hard_limit():
    exec_call = _execute_tool_call()
    seen = []

    async def mock_llm(payload, **kwargs):
        seen.append([str(m.get("content", "")) for m in payload.get("messages", [])])
        return exec_call if len(seen) <= 8 else _llm_ok("Giving up.")

    agent = _make_agent()
    agent.context.llm_client.chat_completion = mock_llm
    agent.context.args.max_context = 999999
    agent.available_tools["execute"] = AsyncMock(return_value=EXEC_FAIL_RESULT)
    body = {"messages": [{"role": "user", "content": "run forever"}], "model": "test-model"}
    await _run(agent, body)
    flat = [c for turn in seen for c in turn] + [str(m.get("content", "")) for m in body.get("messages", [])]
    assert any("SYSTEM 3 PIVOT #1" in c for c in flat)
    assert any("failed too many times" in c.lower() for c in flat), "the hard limit never fired"


# ── 6. the episodic hint ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_recovery_hint_is_a_memory_lookup_and_empty_for_a_member():
    agent = _make_agent()
    epi = MagicMock()
    epi.search_recoveries = MagicMock(return_value=[
        {"recovery_evidence": "ok", "trigger": "pip failed", "outcome": "used the vendored wheel"}])
    agent.context.episodic_memory = epi
    with patch("ghost_agent.core.agent.pretty_log"):
        hint = await agent._recovery_episodes_hint("pip install failed")
    assert "PAST RECOVERY STRATEGIES" in hint and "used the vendored wheel" in hint
    agent.context.llm_client.chat_completion.assert_not_awaited()
    tok = requester_role_context.set("member")
    try:
        assert await agent._recovery_episodes_hint("pip install failed") == ""
    finally:
        requester_role_context.reset(tok)
