"""The planner's CRITICAL INSTRUCTION follows the turn's mode (§4HF).

`is_final_generation` is `force_final_response OR required_tool == "none"`,
but the dynamic-state instruction keyed on `next_action_id` alone. Live
(req 69fb588e, turn 10) the plan said "move to task_5 (compile and deliver
the forensic report)" with required_tool "none": the turn was STREAMED as
the final answer while the state told the model to "Execute the tool(s)
required for the FOCUS TASK" — and it did, a search call, in the final
stream. Driven through `handle_chat`; each pin names the world it fails in.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core import experiments as EXP
from ghost_agent.core.agent import GhostAgent

# Task-shaped, like the planner pins' request: a conversational-looking
# turn never reaches the planner at all.
_REQ = {"messages": [{"role": "user", "content":
                      "Write a python script that lists the workspace files, run it, and summarise the output."}],
        "model": "Qwen-Test"}
_TREE = {"id": "root", "description": "investigate", "status": "IN_PROGRESS",
         "children": [{"id": "task_5", "description": "compile and deliver the report", "status": "READY"}]}


@pytest.fixture
def agent(mock_context):
    return GhostAgent(mock_context)


def _drive(agent, monkeypatch, plan):
    agent.context.args.use_planning = True
    monkeypatch.setattr(EXP, "arm_for", lambda ctx, name, req_id="": EXP.TREATMENT)
    seen = []

    async def fake(payload, *a, **kw):
        label = kw.get("task_label") or ""
        seen.append((label, payload))
        if label == "planner":
            return {"choices": [{"message": {"content": json.dumps(plan)}, "finish_reason": "stop"}]}
        return {"choices": [{"message": {"content": "The sender is not established.", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=fake)
    agent.available_tools["web_search"] = AsyncMock(return_value="results")
    return seen


def _state_text(payload):
    return "\n".join(str(m.get("content") or "") for m in payload.get("messages", []))


@pytest.mark.asyncio
async def test_required_tool_none_with_a_focus_task_is_a_text_only_turn(agent, monkeypatch):
    """FAILS IF: the instruction keys on next_action_id alone — the live
    world."""
    seen = _drive(agent, monkeypatch, {"thought": "I have enough to finalize.", "tree_update": _TREE,
                                       "next_action_id": "task_5", "required_tool": "none"})
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    assert any(l == "planner" for l, _ in seen), "the planner never ran"
    mains = [p for l, p in seen if l != "planner"]
    assert mains, "no main turn ran"
    text = _state_text(mains[0])
    assert "DO NOT USE TOOLS this turn" in text
    assert "task_5 is the delivery itself" in text
    assert "Execute the tool(s) required for the FOCUS TASK" not in text


@pytest.mark.asyncio
async def test_a_focus_task_with_a_named_tool_still_executes(agent, monkeypatch):
    """FAILS IF: the reconciliation swallows every planned turn — a task
    with a named tool must keep its execute instruction."""
    seen = _drive(agent, monkeypatch, {"thought": "search first", "tree_update": _TREE,
                                       "next_action_id": "task_5", "required_tool": "web_search"})
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    mains = [p for l, p in seen if l != "planner"]
    text = _state_text(mains[0])
    assert "Execute the tool(s) required for the FOCUS TASK" in text
    assert "DO NOT USE TOOLS this turn" not in text


@pytest.mark.asyncio
async def test_next_action_none_keeps_its_wording(agent, monkeypatch):
    """FAILS IF: the historical next_action_id=none route changes shape —
    no focus note is appended when there is no focus task."""
    seen = _drive(agent, monkeypatch, {"thought": "answer directly", "tree_update": _TREE,
                                       "next_action_id": "none", "required_tool": "none"})
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    mains = [p for l, p in seen if l != "planner"]
    text = _state_text(mains[0])
    assert "DO NOT USE TOOLS this turn" in text
    assert "is the delivery itself" not in text


@pytest.mark.asyncio
async def test_the_plan_names_itself_as_not_a_project(agent, monkeypatch):
    """§4HK — FAILS IF: the rendered plan carries no such note. Rerun 2 of
    the operator's loop (1e9b6f34) spent a turn on manage_projects(task_update)
    for the planner's own task ids ("no active project") and then told the
    user the project context was lost."""
    from ghost_agent.core.agent import _PLAN_IS_NOT_A_PROJECT_NOTE
    seen = _drive(agent, monkeypatch, {"thought": "search first", "tree_update": _TREE,
                                       "next_action_id": "task_5", "required_tool": "web_search"})
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    mains = [p for l, p in seen if l != "planner"]
    text = _state_text(mains[0])
    assert "FOCUS TASK: task_5" in text
    assert _PLAN_IS_NOT_A_PROJECT_NOTE in text
    assert "manage_projects" in _PLAN_IS_NOT_A_PROJECT_NOTE and "NOT a tracked project" in _PLAN_IS_NOT_A_PROJECT_NOTE


@pytest.mark.asyncio
async def test_the_note_rides_only_with_a_plan(agent, monkeypatch):
    """FAILS IF: the note is emitted unconditionally — a turn with no plan
    rendered would tell the model about a tree it cannot see."""
    from ghost_agent.core.agent import _PLAN_IS_NOT_A_PROJECT_NOTE
    agent.context.args.use_planning = False
    seen = []

    async def fake(payload, *a, **kw):
        seen.append(payload)
        return {"choices": [{"message": {"content": "The sender is not established.", "tool_calls": []}}]}
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=fake)
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    assert seen and _PLAN_IS_NOT_A_PROJECT_NOTE not in _state_text(seen[0])
