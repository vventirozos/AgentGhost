"""The one-task-per-turn latch yields to a verifier auto-repair re-entry —
2026-09-13.

Once a `manage_projects` call closes a task DONE, the loop re-asserts
`force_final_response = True` on EVERY iteration so the turn converges. The
verifier gate's auto-repair sets `force_final_response = False` and
`continue`s with a directive that DEMANDS tools ("actually RUN it", or
gather the evidence a REFUTED claim lacks) — and the next iteration's latch
re-forced the final: `tool_choice: none`, every emitted call dropped with a
"not applied" note, the repair round burned for nothing. Reachable on every
non-streamed client (Slack, CLI, bench, self-play).

The decision is now `_latch_forces_final(task_closed, repair_reentry_active)`,
pinned as a table, and the loop is driven end to end with a scripted model:
the repair turn's tool call must actually run. Control: without a repair the
latch still drops a late tool call — both worlds are checked.
"""
import json
import os
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core.agent import GhostAgent, _latch_forces_final
from tests.helpers import FakeBgTasks, make_context


@pytest.mark.parametrize("closed,repair,expect", [
    (False, False, False),
    (False, True, False),
    (True, False, True),
    (True, True, False),      # the fix: a running repair keeps its tools
])
def test_latch_decision_table(closed, repair, expect):
    assert _latch_forces_final(closed, repair) is expect


def _resp(content, tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content,
                                     "tool_calls": tool_calls or []}}]}


def _tc(cid, name, args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


DONE_READBACK = json.dumps({"updated": [{"id": "t1", "status": "DONE",
                                         "result_summary": "wrote app.py"}],
                            "count": 1})


def _agent(monkeypatch, *, write_first: bool):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")         # production mode
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")  # pure predicate, no await
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    fs = AsyncMock(return_value="SUCCESS: Wrote 5000 chars to 'app.py'.")  # ≥ UNVERIFIED_WRITE_MIN_CHARS: the latch is the subject here
    mp = AsyncMock(return_value=DONE_READBACK)
    ex = AsyncMock(return_value="ok\nEXIT CODE: 0")
    agent.available_tools = {"file_system": fs, "manage_projects": mp, "execute": ex}
    first = [_tc("c1", "manage_projects", {"action": "update", "id": "t1", "status": "DONE"})]
    if write_first:
        first.insert(0, _tc("c0", "file_system", {"operation": "write", "path": "app.py",
                                                  "content": "print('hello world')"}))
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("Working.", first),                                   # closes the task
        _resp("Task 1 is done: app.py written."),                   # final → gate
        _resp("Running it.", [_tc("c2", "execute", {"command": "python app.py"})]),
        _resp("Verified: app.py runs and prints hello world."),     # final
        _resp("(unreachable)"),
    ])
    return agent, ex


async def test_the_repair_turn_can_run_its_tool_after_a_task_closed(monkeypatch):
    """Task closed (latch armed) + unverified write → the async gate forces
    the "actually RUN it" re-entry → the execute call must RUN. Pre-fix
    the latch re-forced the final and the call was dropped."""
    agent, ex = _agent(monkeypatch, write_first=True)
    body = {"messages": [{"role": "user", "content": "start task 1"}]}
    out, _, _ = await agent.handle_chat(body, FakeBgTasks())
    ex.assert_awaited_once()
    assert "Verified" in out


async def test_without_a_repair_the_latch_still_drops_a_late_tool_call(monkeypatch):
    """Control (both worlds agree): no unverified write → no repair → the
    latch keeps forcing the final and the execute call is dropped."""
    agent, ex = _agent(monkeypatch, write_first=False)
    body = {"messages": [{"role": "user", "content": "start task 1"}]}
    await agent.handle_chat(body, FakeBgTasks())
    ex.assert_not_awaited()



@pytest.mark.parametrize("pinned", ["1", "0"])
async def test_after_the_repair_ran_the_latch_turn_is_told_it_is_final(monkeypatch, pinned):
    """R4 review: once the repair's tool ran, the one-task latch re-forced the
    final — but AFTER the schema-side decision of the same iteration, so that
    turn got the full tool header with no final directive while every call it
    made was dropped. The schema decision now honours the latch."""
    monkeypatch.setenv("GHOST_PIN_TOOL_SCHEMAS", pinned)
    agent, ex = _agent(monkeypatch, write_first=True)
    body = {"messages": [{"role": "user", "content": "start task 1"}]}
    await agent.handle_chat(body, FakeBgTasks())
    ex.assert_awaited_once()
    calls = agent.context.llm_client.chat_completion.call_args_list
    assert len(calls) >= 4
    after_repair = calls[3]                      # the turn after the repair's execute ran
    payload = after_repair.args[0] if after_repair.args and isinstance(after_repair.args[0], dict) else after_repair.kwargs
    text = "\n".join(str(m.get("content") or "") for m in payload.get("messages", []) if isinstance(m, dict))
    assert "DO NOT emit any" in text or "DO NOT USE TOOLS" in text, text[-600:]
