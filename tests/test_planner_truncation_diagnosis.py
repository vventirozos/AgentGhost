"""A brace count is not a finish reason (2026-09-15, §4HC).

`salvage_truncated_plan` logged "TRUNCATED at max_tokens" for every planner
reply that failed to parse with unbalanced braces. Live (req c16679f1) a
3,647-char reply with one unescaped quote inside a string — nowhere near
the 8,192-token cap, whose real hits run 25–37k chars — was reported as a
token-cap cut. Corpus: 4 of 84 such labels could not have been cap hits.
The salvage is unchanged; the diagnosis now follows the response's
`finish_reason`, and only a REAL cap hit arms the next-plan steer.
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core import experiments as EXP
from ghost_agent.core.agent import GhostAgent, salvage_truncated_plan

_CUT = '{"thought": "plan", "tree_update": {"id": "root", "subtasks": [{"id": "t1", "desc'
# Unbalanced AND finish_reason=stop: the model simply stopped mid-object. A
# merely mis-quoted reply parses under the tolerant extractor and never
# reaches the salvage path at all (the battery's second survivor).
_MALFORMED = '{"thought": "he said "no" here", "tree_update": {"id": "root"'


@pytest.fixture
def agent(mock_context):
    return GhostAgent(mock_context)


def test_a_length_finish_is_reported_as_a_cap_hit(caplog):
    """FAILS IF: the cap-hit wording is lost for genuine truncations."""
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        salvage_truncated_plan(_CUT, "length")
    assert "TRUNCATED at max_tokens" in caplog.text


def test_a_stop_finish_is_reported_as_malformed_not_a_cap_hit(caplog):
    """FAILS IF: the diagnosis still comes from the brace count.

    The world it fails in is the shipped one: this reply is 84 chars long.
    """
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        salvage_truncated_plan(_MALFORMED, "stop")
    assert "MALFORMED" in caplog.text
    assert "finish_reason=stop" in caplog.text
    assert "TRUNCATED at max_tokens" not in caplog.text


def test_the_salvage_itself_is_the_same_either_way():
    """FAILS IF: the diagnosis change altered what is recovered."""
    a = salvage_truncated_plan(_CUT, "length")
    b = salvage_truncated_plan(_CUT, "stop")
    assert a == b


def test_default_keeps_the_historical_wording(caplog):
    """FAILS IF: an existing caller that passes no finish_reason changes
    meaning — the default is the old behaviour."""
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        salvage_truncated_plan(_CUT)
    assert "TRUNCATED at max_tokens" in caplog.text


def _drive(agent, monkeypatch, planner_replies):
    agent.context.args.use_planning = True
    monkeypatch.setattr(EXP, "arm_for", lambda ctx, name, req_id="": EXP.TREATMENT)
    seen = []
    replies = list(planner_replies)
    state = {"main": 0}

    async def fake(payload, *a, **kw):
        label = kw.get("task_label") or ""
        seen.append((label, payload))
        if label == "planner":
            content, finish = replies.pop(0) if replies else (json.dumps({
                "thought": "ok", "tree_update": {}, "next_action_id": "none"}), "stop")
            return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
        state["main"] += 1
        if state["main"] < 3:
            return {"choices": [{"message": {"content": None, "tool_calls": [{
                "id": f"c{state['main']}",
                "function": {"name": "file_system", "arguments": '{"operation": "list"}'}}]}}]}
        return {"choices": [{"message": {"content": "Done.", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=fake)
    agent.available_tools["file_system"] = AsyncMock(return_value="f.txt")
    return seen


_REQ = {"messages": [{"role": "user", "content":
                      "Write a python script that lists the workspace files, run it, and summarise the output."}],
        "model": "Qwen-Test"}


@pytest.mark.asyncio
async def test_a_real_cap_hit_arms_the_changed_tasks_only_steer(agent, monkeypatch):
    """FAILS IF: the steer never reaches the next planner tail.

    Turn-1 planner is cut at the cap; the turn-2 planner (aligned, so the
    steer rides its tail) must be told to emit changed tasks only.
    """
    seen = _drive(agent, monkeypatch, [(_CUT, "length")])
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    planners = [p for l, p in seen if l == "planner"]
    assert len(planners) >= 2
    assert "CUT AT THE TOKEN CAP" in planners[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_a_malformed_reply_does_not_arm_the_steer(agent, monkeypatch):
    """FAILS IF: the steer keys on the brace count — the very confusion the
    diagnosis fix removes. A malformed reply is not a size problem."""
    seen = _drive(agent, monkeypatch, [(_MALFORMED, "stop")])
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    planners = [p for l, p in seen if l == "planner"]
    assert len(planners) >= 2
    assert "CUT AT THE TOKEN CAP" not in planners[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_the_steer_is_one_turn_only(agent, monkeypatch):
    """FAILS IF: the flag is never cleared — every later plan would be told
    to emit deltas after a single early cap hit."""
    ok = json.dumps({"thought": "fine", "tree_update": {}, "next_action_id": "t1"})
    seen = _drive(agent, monkeypatch, [(_CUT, "length"), (ok, "stop"), (ok, "stop")])
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    planners = [p for l, p in seen if l == "planner"]
    assert len(planners) >= 3
    assert "CUT AT THE TOKEN CAP" in planners[1]["messages"][-1]["content"]
    assert "CUT AT THE TOKEN CAP" not in planners[2]["messages"][-1]["content"]
