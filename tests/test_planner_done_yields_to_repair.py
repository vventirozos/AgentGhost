"""The planner's DONE plan yields to a verifier auto-repair re-entry —
2026-09-24 (slack-23a1fa85).

§4JS made the DONE plan a forced final instead of a loop stop so the
verifier's in-loop repair could fire on the planning arm. It fired — and the
next iteration's planner saw the root still DONE and re-armed the forced
final: the repair directive ("actually RUN it") went into a `tool_choice:
none` turn, the model's browser call was dropped, and the reply then said it
had "navigated to it in the browser to check". Same defect the one-task
latch had on 2026-09-13, one path over. The DONE plan now asks the same
rule (`_latch_forces_final(_plan_signals_done, _repair_reentry_active)`).

Driven end to end with a scripted model on the planning arm: the repair
turn's tool call must RUN. Control: no repair → the DONE plan still keeps a
late tool call off (both worlds are checked).
"""
import json
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core.agent import GhostAgent
from tests.helpers import FakeBgTasks, make_context


def _force_planning_arm(monkeypatch):
    import importlib
    for _modname in ("ghost_agent.core.experiments", "src.ghost_agent.core.experiments"):
        try:
            _exp = importlib.import_module(_modname)
        except ImportError:
            continue
        _real = _exp.arm_for
        monkeypatch.setattr(_exp, "arm_for",
                            lambda ctx_, name, req_id="", _e=_exp, _r=_real: (
                                _e.TREATMENT if name == "use_planning" else _r(ctx_, name, req_id)))


def _resp(content, tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content,
                                     "tool_calls": tool_calls or []}}]}


def _tc(cid, name, args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _plan(status, thought, tool="none"):
    tree = {"id": "root", "description": "Write and verify app.py", "status": status, "children": []}
    return _resp("```json\n" + json.dumps({"thought": thought, "tree_update": tree,
                                            "next_action_id": "root", "required_tool": tool}) + "\n```")


def _agent(monkeypatch, *, write_first: bool):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")         # production mode
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")  # pure predicate, no await
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    _force_planning_arm(monkeypatch)
    ctx = make_context()
    ctx.args.use_planning = True
    agent = GhostAgent(ctx)
    fs = AsyncMock(return_value="SUCCESS: Wrote 5000 chars to 'app.py'.")   # ≥ UNVERIFIED_WRITE_MIN_CHARS
    ex = AsyncMock(return_value="ok\nEXIT CODE: 0")
    search = AsyncMock(return_value="### 1. Result\nnothing to see\n")
    agent.available_tools = {"file_system": fs, "execute": ex, "web_search": search}
    first_tool = (_tc("c0", "file_system", {"operation": "write", "path": "app.py",
                                             "content": "print('hello world')"})
                  if write_first else _tc("c0", "web_search", {"query": "hello"}))
    plans = iter([_plan("IN_PROGRESS", "Do the first step.", "file_system" if write_first else "web_search"),
                  _plan("DONE", "Done; answer now."),
                  _plan("DONE", "Still done; deliver."),
                  _plan("DONE", "Still done; deliver."),
                  _plan("DONE", "(spare)")])
    mains = iter([
        _resp("Working.", [first_tool]),
        _resp("Done: app.py written." if write_first else "Done: nothing found."),   # final → gate
        _resp("Running it.", [_tc("c2", "execute", {"command": "python app.py"})]),  # repair / late call
        _resp("Verified: app.py runs and prints hello world."),
        _resp("(unreachable)"),
    ])

    async def _llm(*args, **kwargs):
        return next(plans) if kwargs.get("task_label") == "planner" else next(mains)

    ctx.llm_client.chat_completion = AsyncMock(side_effect=_llm)
    return agent, ex


@pytest.mark.parametrize("native_tools", [False, True])
async def test_the_repair_turn_runs_its_tool_under_a_done_plan(monkeypatch, native_tools):
    """Unverified write + DONE plan → the gate's repair re-entry → the
    execute call must RUN. Pre-fix the DONE plan re-forced the final and
    the call was dropped."""
    agent, ex = _agent(monkeypatch, write_first=True)
    agent.context.args.native_tools = native_tools
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "write app.py that prints hello world and verify it runs"}]},
        FakeBgTasks())
    ex.assert_awaited_once()
    assert "Verified" in out
    # R2 review: the repair turn's PROMPT must not carry the final-generation
    # directive either ("DO NOT emit any <tool_call>" next to "actually RUN it")
    mains = [c for c in agent.context.llm_client.chat_completion.call_args_list
             if c.kwargs.get("task_label") != "planner"]
    repair_call = mains[2]
    payload = repair_call.args[0] if repair_call.args and isinstance(repair_call.args[0], dict) else repair_call.kwargs
    text = "\n".join(str(m.get("content") or "") for m in payload.get("messages", []) if isinstance(m, dict))
    assert "DO NOT emit any" not in text and "DO NOT USE TOOLS" not in text
    if native_tools:
        assert payload.get("tool_choice") == "auto", payload.get("tool_choice")   # R3: pinned only where the key exists


async def test_without_a_repair_the_done_plan_still_drops_a_late_tool_call(monkeypatch):
    """Control (R4 pins review: the old version never emitted the call and
    every plan paired DONE with required_tool=none, which forces the final by
    itself). Here the DONE plan names `execute`, so ONLY the DONE branch can
    force the final; the model emits the call and it must be dropped."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    _force_planning_arm(monkeypatch)
    ctx = make_context()
    ctx.args.use_planning = True
    agent = GhostAgent(ctx)
    ex = AsyncMock(return_value="ok\nEXIT CODE: 0")
    search = AsyncMock(return_value="### 1. Result\nnothing to see\n")
    agent.available_tools = {"execute": ex, "web_search": search}
    plans = iter([_plan("IN_PROGRESS", "Search first.", "web_search")]
                 + [_plan("DONE", f"Done, deliver ({i}).", "execute") for i in range(6)])
    mains = iter([
        _resp("Working.", [_tc("c0", "web_search", {"query": "hello"})]),
        _resp("Running it.", [_tc("c2", "execute", {"command": "echo hi"})]),   # late call under a DONE plan
        _resp("Nothing found."), _resp("Nothing found."), _resp("(unreachable)"),
    ])

    async def _llm(*args, **kwargs):
        return next(plans) if kwargs.get("task_label") == "planner" else next(mains)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_llm)
    await agent.handle_chat({"messages": [{"role": "user", "content": "search for hello and tell me"}]},
                            FakeBgTasks())
    search.assert_awaited_once()
    ex.assert_not_awaited()


async def test_after_the_repair_ran_its_tools_the_done_plan_converges_again(monkeypatch):
    """R2 review: the flag used to stay set for the rest of the request, so
    a model that kept calling tools after its repair was never asked to
    finish. Script: write → DONE → unverified → repair runs execute → the
    model tries a SECOND execute under a DONE plan → dropped, final ships."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    _force_planning_arm(monkeypatch)
    ctx = make_context()
    ctx.args.use_planning = True
    agent = GhostAgent(ctx)
    fs = AsyncMock(return_value="SUCCESS: Wrote 5000 chars to 'app.py'.")
    ex = AsyncMock(return_value="ok\nEXIT CODE: 0")
    agent.available_tools = {"file_system": fs, "execute": ex}
    plans = iter([_plan("IN_PROGRESS", "Write it.", "file_system")] + [_plan("DONE", "Done; deliver.")] * 6)
    mains = iter([
        _resp("Working.", [_tc("c0", "file_system", {"operation": "write", "path": "app.py", "content": "print(1)"})]),
        _resp("Done: app.py written."),                                                  # gate → repair
        _resp("Running it.", [_tc("c2", "execute", {"command": "python app.py"})]),     # the repair's tool
        _resp("Let me run it once more.", [_tc("c3", "execute", {"command": "python app.py"})]),  # must be DROPPED
        _resp("Verified: app.py runs."),
        _resp("Verified: app.py runs (retry)."),
        _resp("(unreachable)"),
    ])

    async def _llm(*args, **kwargs):
        return next(plans) if kwargs.get("task_label") == "planner" else next(mains)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_llm)
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "write app.py and verify it runs"}]}, FakeBgTasks())
    assert ex.await_count == 1, ex.await_count                  # the second call was dropped
    # the turn converged on the honest dropped-call note (the accumulated
    # "Working. Running it." counts as an answer, so no answer-now retry —
    # pinned as the world that exists, not an OR)
    assert "pending execute" in out and "Verified" not in out
    mains_calls = [c for c in ctx.llm_client.chat_completion.call_args_list
                   if c.kwargs.get("task_label") != "planner"]
    assert len(mains_calls) == 4, len(mains_calls)            # write, answer, repair run, the dropped one
    last = mains_calls[-1]
    payload = last.args[0] if last.args and isinstance(last.args[0], dict) else last.kwargs
    text = "\n".join(str(m.get("content") or "") for m in payload.get("messages", []) if isinstance(m, dict))
    assert "DO NOT" in text                                   # the converge directive reached that turn


async def test_a_rejected_call_does_not_count_as_the_repairs_tools(monkeypatch):
    """R3 review: the flag cleared on ANY appended row, including the loop's
    synthetic REJECTED rows. Script: write → DONE → unverified → the repair
    turn calls a DISABLED tool (a synthetic `tool_disabled` row, no role or
    origin involved) → the flag must stay set, so the NEXT turn's real
    execute call still runs under the DONE plan."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    _force_planning_arm(monkeypatch)
    ctx = make_context()
    ctx.args.use_planning = True
    agent = GhostAgent(ctx)
    fs = AsyncMock(return_value="SUCCESS: Wrote 5000 chars to 'app.py'.")
    ex = AsyncMock(return_value="ok\nEXIT CODE: 0")
    learn = AsyncMock(return_value="(never reached)")
    agent.available_tools = {"file_system": fs, "execute": ex, "learn_skill": learn}
    agent.disabled_tools = {"learn_skill"}
    plans = iter([_plan("IN_PROGRESS", "Write it.", "file_system")] + [_plan("DONE", "Done; deliver.")] * 6)
    mains = iter([
        _resp("Working.", [_tc("c0", "file_system", {"operation": "write", "path": "app.py", "content": "print(1)"})]),
        _resp("Done: app.py written."),                                                  # gate → repair
        # a DISABLED tool is REJECTED before dispatch (a synthetic row; an
        # unknown tool would rebuild the tool table and drop the mocks)
        _resp("Noting.", [_tc("c2", "learn_skill", {"lesson": "x"})]),
        _resp("Running it.", [_tc("c3", "execute", {"command": "python app.py"})]),     # must RUN
        _resp("Verified: app.py runs."),
        _resp("(unreachable)"),
    ])

    async def _llm(*args, **kwargs):
        return next(plans) if kwargs.get("task_label") == "planner" else next(mains)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_llm)
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "write app.py and verify it runs"}]}, FakeBgTasks())
    learn.assert_not_awaited()
    ex.assert_awaited_once()
    assert "Verified" in out


async def test_a_repair_that_only_re_emits_rejected_calls_still_converges(monkeypatch):
    """R4 review: synthetic rows never cleared the flag, so a model that kept
    re-emitting a REJECTED call had tools and an ignored DONE plan until the
    turn budget ran out. After three rows of any kind the converge signals
    re-arm."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    _force_planning_arm(monkeypatch)
    ctx = make_context()
    ctx.args.use_planning = True
    agent = GhostAgent(ctx)
    fs = AsyncMock(return_value="SUCCESS: Wrote 5000 chars to 'app.py'.")
    learn = AsyncMock(return_value="(never reached)")
    agent.available_tools = {"file_system": fs, "learn_skill": learn}
    plans = iter([_plan("IN_PROGRESS", "Write it.", "file_system")] + [_plan("DONE", "Done; deliver.")] * 30)
    rejected = _resp("Noting.", [_tc("cx", "learn_skill", {"lesson": "x"})])
    mains = iter([
        _resp("Working.", [_tc("c0", "file_system", {"operation": "write", "path": "app.py", "content": "print(1)"})]),
        _resp("Done: app.py written."),
    ] + [rejected] * 30)
    calls = []

    async def _llm(*args, **kwargs):
        calls.append(kwargs.get("task_label"))
        return next(plans) if kwargs.get("task_label") == "planner" else next(mains)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_llm)
    await agent.handle_chat(
        {"messages": [{"role": "user", "content": "write app.py and verify it runs"}]}, FakeBgTasks(),
        request_id="probe-rejectloop")
    mains_used = sum(1 for c in calls if c != "planner")
    assert mains_used <= 8, mains_used             # bounded well below the 40-turn budget



async def test_the_row_bound_is_measured_from_the_repair_not_from_zero(monkeypatch):
    """R6 pins review: the 3-row bound reads `_repair_reentry_rows_at`
    through five state copies; a lost copy reads 0 and the bound fires at
    once on any request that already had ≥3 rows. Script: 3 real calls
    before the repair, the repair's first call REJECTED, its second a real
    `execute` — which must run."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    _force_planning_arm(monkeypatch)
    ctx = make_context()
    ctx.args.use_planning = True
    agent = GhostAgent(ctx)
    fs = AsyncMock(return_value="SUCCESS: Wrote 5000 chars to 'app.py'.")
    ex = AsyncMock(return_value="ok\nEXIT CODE: 0")
    search = AsyncMock(return_value="### 1. r\nx\n")
    learn = AsyncMock(return_value="(never reached)")
    agent.available_tools = {"file_system": fs, "execute": ex, "web_search": search, "learn_skill": learn}
    agent.disabled_tools = {"learn_skill"}
    plans = iter([_plan("IN_PROGRESS", "Look first.", "web_search"), _plan("IN_PROGRESS", "Write.", "file_system")]
                 + [_plan("DONE", f"Done ({i}).") for i in range(8)])
    mains = iter([
        _resp("Looking.", [_tc("s0", "web_search", {"query": "a"}), _tc("s1", "web_search", {"query": "b"})]),
        _resp("Writing.", [_tc("c0", "file_system", {"operation": "write", "path": "app.py", "content": "print(1)"})]),
        _resp("Done: app.py written."),                                                # gate → repair
        _resp("Noting.", [_tc("c2", "learn_skill", {"lesson": "x"})]),                 # REJECTED
        _resp("Running it.", [_tc("c3", "execute", {"command": "python app.py"})]),   # must RUN
        _resp("Verified: app.py runs."), _resp("(unreachable)"),
    ])

    async def _llm(*args, **kwargs):
        return next(plans) if kwargs.get("task_label") == "planner" else next(mains)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_llm)
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "look around, write app.py and verify it runs"}]}, FakeBgTasks())
    ex.assert_awaited_once()
