"""The aligned planner tail carries only what the prefix lacks (§4HN).

Measured on de95699d: 26–42k chars per plan (7–14k tokens re-prefilled,
13–33 s each turn). Of that, the user request, the last two tool outputs
(up to 84k chars each) and the scrapbook/sandbox blocks were duplicates of
the shared prefix — the prefix IS the previous main request, which holds the
first user message, every tool message and the dynamic state verbatim. The
turn-1 legacy shape (no prefix) keeps the full transient. Each pin names
the world it fails in.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.core import experiments as EXP
from ghost_agent.core.agent import GhostAgent

_REQ = {"messages": [{"role": "user", "content":
                      "Write a python script that lists the workspace files, run it, and summarise the output."}],
        "model": "Qwen-Test"}
_TREE = {"id": "root", "description": "list the workspace and summarise", "status": "IN_PROGRESS",
         "children": [{"id": "task_1", "description": "list the files", "status": "READY"},
                      {"id": "task_2", "description": "summarise", "status": "PENDING"}]}


@pytest.fixture
def agent(mock_context):
    return GhostAgent(mock_context)


def _drive(agent, monkeypatch, tool_result="f.txt"):
    agent.context.args.use_planning = True
    monkeypatch.setattr(EXP, "arm_for", lambda ctx, name, req_id="": EXP.TREATMENT)
    seen = []
    state = {"main": 0}

    async def fake(payload, *a, **kw):
        label = kw.get("task_label") or ""
        seen.append((label, payload))
        if label == "planner":
            plan = {"thought": "go", "tree_update": _TREE, "next_action_id": "task_1", "required_tool": "file_system"}
            return {"choices": [{"message": {"content": json.dumps(plan)}, "finish_reason": "stop"}]}
        state["main"] += 1
        if state["main"] < 3:
            return {"choices": [{"message": {"content": None, "tool_calls": [{
                "id": f"c{state['main']}",
                "function": {"name": "file_system", "arguments": '{"operation": "list"}'}}]}}]}
        return {"choices": [{"message": {"content": "Done.", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=fake)
    agent.available_tools["file_system"] = AsyncMock(return_value=tool_result)
    return seen


def _tails(seen):
    planners = [p for l, p in seen if l == "planner"]
    return planners[0]["messages"][-1]["content"], planners[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_the_aligned_tail_drops_what_the_prefix_already_holds(agent, monkeypatch):
    """FAILS IF: turn 2+ still ships the full transient — the live world
    (the request, the last tool outputs and the scrapbook rode every tail)."""
    seen = _drive(agent, monkeypatch, tool_result="RESULT-MARKER " + "r" * 500)
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    t1, t2 = _tails(seen)
    for dup in ("### CURRENT SITUATION", "User Request:", "Last Tool Output:", "SCRAPBOOK:", "SANDBOX STATE:"):
        assert dup not in t2, dup
    # the request text itself is not repeated in the tail
    assert "lists the workspace files, run it, and summarise" not in t2
    # …and the result appears exactly once — as the delta gist, not again as Last Tool Output
    assert t2.count("RESULT-MARKER") == 1


@pytest.mark.asyncio
async def test_the_aligned_tail_keeps_what_the_planner_needs(agent, monkeypatch):
    """FAILS IF: the cut takes the plan, the tool list, the temporal anchor
    or the delta with it."""
    seen = _drive(agent, monkeypatch)
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    _, t2 = _tails(seen)
    assert "### NEW SINCE YOUR LAST PLAN" in t2 and "TOOL (file_system):" in t2
    assert "### AVAILABLE NATIVE TOOLS" in t2 and "file_system" in t2
    assert "TURN 2" in t2 and "NEVER revert a 'DONE' task" in t2
    assert "### CURRENT PLAN (JSON)" in t2 and '"id":"task_1"' in t2
    assert "in the conversation above" in t2


@pytest.mark.asyncio
async def test_the_plan_json_is_compact_on_the_aligned_tail(agent, monkeypatch):
    """FAILS IF: the plan is still pretty-printed (indent=2) on turn 2+ —
    every task costs a dozen lines of whitespace per plan."""
    seen = _drive(agent, monkeypatch)
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    t1, t2 = _tails(seen)
    plan2 = t2.split("### CURRENT PLAN (JSON)", 1)[1]
    assert '{"id":"root"' in plan2 and '\n  "id"' not in plan2
    assert json.loads(plan2.strip().splitlines()[0])["id"] == "root"


@pytest.mark.asyncio
async def test_turn_one_keeps_the_legacy_transient(agent, monkeypatch):
    """FAILS IF: the cut reaches the legacy shape — turn 1 has no prefix,
    so the request and the last tool output must still ride the tail."""
    seen = _drive(agent, monkeypatch)
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    t1, _ = _tails(seen)
    assert "### CURRENT SITUATION" in t1 and "User Request:" in t1 and "Last Tool Output:" in t1


@pytest.mark.asyncio
async def test_the_aligned_tail_is_materially_shorter(agent, monkeypatch):
    """FAILS IF: the cut is cosmetic. With a 6k tool result the aligned tail
    is the planner prompt + a ≤2.5k delta gist + ~2k of fixed blocks; the old
    shape added the same 6k result again as Last Tool Output (and the
    request, and the scrapbook), so it cannot fit under this bound."""
    from ghost_agent.core import agent as A
    from ghost_agent.core.prompts import PLANNING_SYSTEM_PROMPT
    seen = _drive(agent, monkeypatch, tool_result="z" * 6000)
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    _, t2 = _tails(seen)
    assert t2.count("z" * 100) <= A._PLANNER_DELTA_CHARS_PER_MSG // 100
    assert len(t2) < len(PLANNING_SYSTEM_PROMPT) + A._PLANNER_DELTA_CHARS_PER_MSG + 3500, len(t2)


@pytest.mark.asyncio
async def test_the_log_line_reports_the_composition(agent, monkeypatch):
    """FAILS IF: the operator cannot see what the tail is made of — the cut
    is verified live from this line."""
    seen = _drive(agent, monkeypatch)
    with patch("ghost_agent.core.agent.pretty_log") as pl:
        await agent.handle_chat(_REQ, background_tasks=MagicMock())
    lines = [c.args[1] for c in pl.call_args_list if c.args and c.args[0] == "Planner Prefix"]
    assert lines, "no Planner Prefix line"
    assert all(("prompt " in ln and "delta " in ln and "plan " in ln and "tools " in ln) for ln in lines)
