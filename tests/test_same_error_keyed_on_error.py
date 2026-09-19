"""§4IK — the same-error count is keyed on the error, not the command head.

THE LIVE FAILURE (probe ifs04495…, 2026-09-18, ninth IFS/Oxford run). The
identical `RuntimeError: SpecError: [pl]` ended four `python3 …` runs and
three `for f in …; do python3 …; done` runs. Keyed on `<head> (same
error)`, that was two signatures — `python3 (same error)` ×4 and `for
(same error)` ×3 — so the steer fired twice and the five-run REPORT tier
never did; the request ran to the reserved turn.

Now the ledger key is `SAME_ERROR_TARGET_SUFFIX` alone (the error
fingerprint carries the identity); the heads seen are kept as an
annotation (`strikes.exec_error_heads`) for the log and the steer.

World where each pin fails: the site keys on the head again, the heads
stop being recorded, the hard stop no longer accumulates across heads, or
the steer/report text loses the error line.
"""
import ast
import inspect
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import GhostAgent, TurnState
from ghost_agent.core.strikes import (EXECUTE_SAME_ERROR_HARD_STOP, EXECUTE_SAME_ERROR_STEER,
                                      SAME_ERROR_TARGET_SUFFIX, StrikeLedger)

ERR = "EXIT CODE: 0\nERR dict nxacc16: RuntimeError: SpecError: [pl]  (/src/eckit/spec/Spec.cc:34 _get_t)\n"


def _make_agent():
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)

    async def ex(**kw):
        return ERR
    agent.available_tools = {"execute": ex}
    agent.disabled_tools = set()
    agent.context.current_project_id = None
    agent.context._script_iter = {}
    agent.context._futility_steer_done = False
    agent.context._futility_report_done = False
    agent.context._breaker_forced_final = False
    return agent


def _ts(command, strikes, steered):
    return TurnState(
        _constraint_steer_pending=None, _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False, _request_sys3_prev_justification="",
        consecutive_parse_errors=0, current_plan_json="",
        execution_failure_count=0, final_ai_content="", fname="",
        force_final_response=False, force_stop=False, forget_was_called=False,
        last_was_failure=True, preflight_blocks_this_request=0,
        request_sandbox_state="", transient_failure_count=0,
        tool_calls=[{"id": "c", "type": "function", "function": {"name": "execute", "arguments": json.dumps({"command": command})}}],
        msg={"role": "assistant", "content": ""}, ui_content="",
        parse_failure_reason="", model="test-model",
        last_user_content="build the grid", char_budget=4000,
        strikes=strikes, task_tree=MagicMock(),
        _user_batch_intent=None, _request_constraints=[],
        repeated_action_steered=steered, messages=[], seen_tools=set(),
        executed_idempotent=set(), raw_tools_called=set(), tool_usage={},
        tools_run_this_turn=[], request_state=MagicMock(),
    )


COMMANDS = ["cd /workspace && python3 probe.py", "python3 probe2.py 2>&1 | grep ERR",
            "for f in probe3.py; do python3 $f; done", "cd /workspace && python3 probe4.py",
            "for f in probe5.py probe6.py; do python3 $f; done"]   # live, tools go off after the 5th


@pytest.mark.asyncio
async def test_same_error_accumulates_across_command_heads():
    agent = _make_agent()
    strikes, steered = StrikeLedger(), set()
    events = []
    for i, cmd in enumerate(COMMANDS):
        ts = _ts(cmd, strikes, steered)
        await agent._dispatch_and_process_tool_batch(ts)
        alerts = [m["content"] for m in ts.messages if m.get("role") == "user" and "SYSTEM ALERT" in str(m.get("content"))]
        events.append((i, bool(alerts), ts.force_final_response, alerts[-1] if alerts else ""))
    steer_at = [i for i, a, f, _ in events if a and not f]
    report_at = [i for i, a, f, _ in events if a and f]
    assert steer_at == [EXECUTE_SAME_ERROR_STEER - 1]           # 3rd run: heads python3, python3, for
    assert report_at == [EXECUTE_SAME_ERROR_HARD_STOP - 1]       # 5th run, heads mixed
    steer_text = events[steer_at[0]][3]
    assert "SpecError: [pl]" in steer_text and "`python3`" in steer_text and "`for`" in steer_text
    report_text = events[report_at[0]][3]
    assert "SpecError: [pl]" in report_text and "STOP trying variations" in report_text
    # one signature for the whole class, keyed on the error
    sigs = [k for k in strikes.action_sigs if k.startswith("execute|")]
    assert sigs == [f"execute|{SAME_ERROR_TARGET_SUFFIX}|" + sigs[0].rsplit("|", 1)[1]]
    assert strikes.exec_error_heads[sigs[0]] == ["python3", "for"]


def test_site_keys_on_the_suffix_not_the_head():
    tree = ast.parse(inspect.getsource(ag))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
              and n.name == "_dispatch_and_process_tool_batch")
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", "") == "note_action"
             and c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "execute"]
    assert len(calls) == 1
    target = calls[0].args[1]
    assert isinstance(target, ast.Attribute) and target.attr == "SAME_ERROR_TARGET_SUFFIX"
    assert not any(isinstance(n, ast.JoinedStr) for n in ast.walk(target))
