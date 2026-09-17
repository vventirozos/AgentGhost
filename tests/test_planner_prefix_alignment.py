"""The planner rides the main request's KV instead of evicting it (2026-09-15, §4HC).

Measured on the server log: with the planner OFF (req 6a7882f5) main turns
prefilled 3–8k tokens each, 88% reuse; with it ON (req c16679f1) every
main turn re-prefilled 24k→44k tokens (~397k, ~510 s) because the
planner's prompt — its own system prompt plus the transcript re-serialised
— shared ~1k template tokens with the main prompt on the single slot, and
its context (55–81k) was larger than the main's.

From turn 2 the planner now sends the previous main request's exact
rendered messages plus one trailing instruction. The pins drive
`handle_chat` end to end and read the payloads the LLM client received.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core import experiments as EXP
from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.prompts import PLANNING_SYSTEM_PROMPT


@pytest.fixture
def agent(mock_context):
    return GhostAgent(mock_context)


def _plan(thought="plan", nxt="task_1"):
    return json.dumps({
        "thought": thought,
        "tree_update": {"id": "root", "description": "root", "status": "IN_PROGRESS",
                        "subtasks": [{"id": "task_1", "description": "list files",
                                      "status": "IN_PROGRESS", "subtasks": []}]},
        "next_action_id": nxt,
        "required_tool": "file_system",
    })


def _drive(agent, monkeypatch, *, turns=2):
    """Run a planner-ON, two-tool-turn conversation; return the payloads
    the LLM client saw, tagged by task_label."""
    agent.context.args.use_planning = True
    monkeypatch.setattr(EXP, "arm_for", lambda ctx, name, req_id="": EXP.TREATMENT)
    seen = []
    state = {"main_calls": 0}

    async def fake(payload, *a, **kw):
        label = kw.get("task_label") or ""
        seen.append((label, payload))
        if label == "planner":
            return {"choices": [{"message": {"content": _plan()},
                                 "finish_reason": "stop"}]}
        state["main_calls"] += 1
        if state["main_calls"] < turns:
            return {"choices": [{"message": {"content": None, "tool_calls": [{
                "id": f"call_{state['main_calls']}",
                "function": {"name": "file_system",
                             "arguments": '{"operation": "list"}'}}]}}]}
        return {"choices": [{"message": {"content": "Done.", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=fake)
    agent.available_tools["file_system"] = AsyncMock(return_value="file1.txt\nfile2.txt")
    return seen


@pytest.mark.asyncio
async def test_turn_two_planner_prompt_is_the_previous_main_prompt_plus_a_tail(agent, monkeypatch):
    """FAILS IF: the planner keeps its own system prompt after turn 1.

    The world it fails in is the shipped one: planner #2's first message
    was PLANNING_SYSTEM_PROMPT, the main request's head was nowhere in it.
    """
    seen = _drive(agent, monkeypatch)
    body = {"messages": [{"role": "user", "content": "Write a python script that lists the workspace files, run it, and summarise the output."}],
            "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    planners = [p for l, p in seen if l == "planner"]
    mains = [p for l, p in seen if l != "planner"]
    assert len(planners) >= 2 and len(mains) >= 2, (len(planners), len(mains))
    p2 = planners[1]["messages"]
    m1 = mains[0]["messages"]
    # Every message of the previous MAIN request, verbatim, then exactly one tail.
    assert p2[:-1] == m1
    assert p2[-1]["role"] == "user"
    assert PLANNING_SYSTEM_PROMPT[:40] in p2[-1]["content"]
    assert "NEW SINCE YOUR LAST PLAN" in p2[-1]["content"]


@pytest.mark.asyncio
async def test_turn_one_planner_keeps_the_legacy_shape(agent, monkeypatch):
    """FAILS IF: alignment is attempted with no head to align to."""
    seen = _drive(agent, monkeypatch)
    body = {"messages": [{"role": "user", "content": "Write a python script that lists the workspace files, run it, and summarise the output."}],
            "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    p1 = [p for l, p in seen if l == "planner"][0]["messages"]
    assert p1[0]["role"] == "system"
    assert p1[0]["content"].endswith(PLANNING_SYSTEM_PROMPT) or PLANNING_SYSTEM_PROMPT in p1[0]["content"]
    assert "### RECENT CONVERSATION" in p1[-1]["content"]


@pytest.mark.asyncio
async def test_the_tail_carries_what_changed_since_the_last_main_call(agent, monkeypatch):
    """FAILS IF: the delta transcript is dropped — the planner would plan
    against a conversation that ends before the tool result it needs."""
    seen = _drive(agent, monkeypatch)
    body = {"messages": [{"role": "user", "content": "Write a python script that lists the workspace files, run it, and summarise the output."}],
            "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    p2 = [p for l, p in seen if l == "planner"][1]["messages"]
    tail = p2[-1]["content"]
    # The transcript's role prefix exists ONLY in the delta (the legacy
    # transient's "Last Tool Output" left the aligned tail in §4HN).
    assert "TOOL (file_system):" in tail
    assert "file1.txt" in tail


@pytest.mark.asyncio
async def test_native_tools_ride_the_aligned_planner_with_tool_choice_none(agent, monkeypatch):
    """FAILS IF: the aligned planner drops `tools` (the template renders them
    into the HEAD — dropping them breaks the very prefix this exists for)
    or leaves tool_choice open (a plan call must not call tools)."""
    agent.context.args.native_tools = True
    seen = _drive(agent, monkeypatch)
    body = {"messages": [{"role": "user", "content": "Write a python script that lists the workspace files, run it, and summarise the output."}],
            "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    planners = [p for l, p in seen if l == "planner"]
    mains = [p for l, p in seen if l != "planner"]
    if not mains[0].get("tools"):
        pytest.skip("this harness did not attach native tools on the main call")
    assert planners[1].get("tools") == mains[0]["tools"]
    assert planners[1].get("tool_choice") == "none"
    # §4HC round 2: the aligned reply is SCHEMA-constrained, not json_object.
    assert planners[1].get("response_format", {}).get("type") == "json_schema"


@pytest.mark.asyncio
async def test_the_head_is_per_request(agent, monkeypatch):
    """FAILS IF: a head from one request leaks into the next — a stale head
    shares nothing with the new conversation and costs a full prefill."""
    seen = _drive(agent, monkeypatch)
    body = {"messages": [{"role": "user", "content": "Write a python script that lists the workspace files, run it, and summarise the output."}],
            "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    seen.clear()
    body2 = {"messages": [{"role": "user", "content": "Write a second python script that counts them and run it."}],
             "model": "Qwen-Test"}
    await agent.handle_chat(body2, background_tasks=MagicMock())
    p1 = [p for l, p in seen if l == "planner"][0]["messages"]
    assert p1[0]["role"] == "system" and "RECENT CONVERSATION" in p1[-1]["content"]


# ---------------------------------------------------------------- §4HC round-2 (req 552a1ffd)

@pytest.mark.asyncio
async def test_the_aligned_reply_is_schema_constrained(agent, monkeypatch):
    """FAILS IF: the aligned call ships `json_object` (or nothing).

    Live: four of five aligned planner replies were XML `<tool_call>`
    blocks — the main persona answered the planning instruction and
    `json_object` did not hold. Reproduced offline; a schema requiring the
    plan's keys fixed it on the same payload.
    """
    seen = _drive(agent, monkeypatch)
    body = {"messages": [{"role": "user", "content": "Write a python script that lists the workspace files, run it, and summarise the output."}],
            "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    planners = [p for l, p in seen if l == "planner"]
    rf = planners[1].get("response_format") or {}
    assert rf.get("type") == "json_schema"
    schema = rf["json_schema"]["schema"]
    assert set(schema["required"]) == {"thought", "tree_update", "next_action_id", "required_tool"}
    # The legacy turn-1 call is untouched.
    assert (planners[0].get("response_format") or {}).get("type") == "json_object"


@pytest.mark.asyncio
async def test_the_delta_transcript_is_capped_per_message(agent, monkeypatch):
    """FAILS IF: the tail carries tool results at the transcript's default
    18k chars each — live tails ran 37–50k chars, ~11k tokens prefilled
    per planner call, for a gist the transient block already summarises.
    """
    from ghost_agent.core import agent as A
    seen = _drive(agent, monkeypatch)
    # A production-sized context: the transcript's DEFAULT cap is derived
    # from max_context (0.07×), and under the mock's 8k it is 560 chars —
    # smaller than the planner cap, so an uncapped mutant survived.
    agent.context.args.max_context = 262144
    big = "X" * 20000
    agent.available_tools["file_system"] = AsyncMock(return_value=big)
    body = {"messages": [{"role": "user", "content": "Write a python script that lists the workspace files, run it, and summarise the output."}],
            "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    tail = [p for l, p in seen if l == "planner"][1]["messages"][-1]["content"]
    # The delta block must not carry the whole 20k result.
    # §4HN: the aligned tail no longer carries "### CURRENT SITUATION"; the
    # delta ends where the tool list begins.
    delta = tail.split("### NEW SINCE YOUR LAST PLAN", 1)[1].split("### AVAILABLE NATIVE TOOLS", 1)[0]
    assert delta.count("X") <= A._PLANNER_DELTA_CHARS_PER_MSG + 50
    assert "TOOL (file_system):" in delta


def test_recent_transcript_default_cap_is_unchanged(agent):
    """FAILS IF: adding the optional cap changed the legacy transcript."""
    agent.context.args.max_context = 262144
    msgs = [{"role": "tool", "name": "t", "content": "Y" * 30000}]
    out = agent._get_recent_transcript(msgs)
    assert out.count("Y") == max(500, int(262144 * 3.5 * 0.02))
    assert agent._get_recent_transcript(msgs, char_limit=100).count("Y") == 200  # floor


@pytest.mark.asyncio
async def test_a_tool_call_shaped_planner_reply_is_named_in_the_log(agent, monkeypatch, capsys):
    """FAILS IF: an XML tool call answering the planner is logged only as
    'No thought provided.' — the symptom with no cause."""
    agent.context.args.use_planning = True
    monkeypatch.setattr(EXP, "arm_for", lambda ctx, name, req_id="": EXP.TREATMENT)
    state = {"main": 0}

    async def fake(payload, *a, **kw):
        if (kw.get("task_label") or "") == "planner":
            return {"choices": [{"message": {"content": "<tool_call>\n<function=web_search>\n</function>\n</tool_call>"},
                                 "finish_reason": "stop"}]}
        state["main"] += 1
        return {"choices": [{"message": {"content": "Done.", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=fake)
    body = {"messages": [{"role": "user", "content": "Write a python script that lists the workspace files, run it, and summarise the output."}],
            "model": "Qwen-Test"}
    await agent.handle_chat(body, background_tasks=MagicMock())
    out = capsys.readouterr().out
    assert "reply was a tool call, not a plan" in out
