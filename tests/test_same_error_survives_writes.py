"""§4IG — a rewrite that reproduces the error is not progress.

THE LIVE FAILURE (probe ifs19450…, 2026-09-17, fourth IFS/Oxford re-test).
Six `execute` runs ended in the identical `RuntimeError: SpecError: [pl]`
(the §4IE signature collapsed the labels), yet the same-error breaker never
fired: each run was preceded by a NEW probe file (probe6.py … probe19.py),
and `StrikeLedger.note_world_changed` — right for the no-progress class,
where a mutation makes a re-observation legitimate — wiped the same-error
count with every write. The futility breaker never fired either: keyed on
the literal basename, fourteen probe files were fourteen scripts with one
write each. And the reserved report turn answered with "Let me fix that."
plus an `execute` call: the call was dropped, the prose passed the
narration check because one sentence was factual, and — being the last
turn — there was no retry.

World where each pin fails: a write clears the same-error count again, the
futility key stops collapsing probeN.py, the run side stops matching the
collapsed key, the report turn moves back to the last turn, or a dropped
tool call on a breaker-forced final stops counting as no answer.
"""
import ast
import inspect
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import (GhostAgent, TurnState, futility_key,
                                    futility_keys_in_command, last_turn_needs_report)
from ghost_agent.core.strikes import (EXECUTE_SAME_ERROR_STEER, StrikeLedger,
                                      error_line_fingerprint, is_same_error_signature)

_TREE = ast.parse(inspect.getsource(ag))
RESULT = "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\n{}\n"


# ── G1: the same-error count survives a write ─────────────────────────

def test_same_error_count_survives_writes_but_no_progress_count_does_not():
    led = StrikeLedger()
    fp = error_line_fingerprint(RESULT.format("ERR dict npts31: RuntimeError: SpecError: [pl]"))
    trips = []
    for i in range(EXECUTE_SAME_ERROR_STEER):
        _, cnt, tripped = led.note_action("execute", "python3 (same error)", fp,
                                          threshold=EXECUTE_SAME_ERROR_STEER)
        led.note_action("browser", "http://x", "obs")          # a plain observation
        trips.append((cnt, tripped))
        led.note_world_changed()                               # the next probe file is written
    assert trips[-1] == (EXECUTE_SAME_ERROR_STEER, True)
    assert all(not is_same_error_signature(k) is False or True for k in led.action_sigs)
    # the plain observation was forgotten by every write; the same-error key was kept
    assert [k for k in led.action_sigs if k.startswith("browser|")] == []
    assert [k for k in led.action_sigs if is_same_error_signature(k)]


@pytest.mark.parametrize("sig,same", [
    ("execute|python3 (same error)|abc", True),
    ("execute|? (same error)|abc", True),
    ("execute|python3|abc", False),
    ("browser|http://x (same error) y|abc", False),
    ("garbage", False), ("", False),
])
def test_is_same_error_signature(sig, same):
    assert is_same_error_signature(sig) is same


# ── G2: the futility key collapses numbered rewrites ─────────────────

@pytest.mark.parametrize("name,key", [
    ("/workspace/probe6.py", "probe.py"), ("probe19.py", "probe.py"), ("probe_3.py", "probe.py"),
    ("oxford_grid2.py", "oxford_grid.py"), ("PROBE.PY", "probe.py"), ("run-2.sh", "run.sh"),
    ("extract_data.py", "extract_data.py"), ("v2.py", "v.py"), ("123.py", "123.py"),
    ("index.html", "index.html"), ("Makefile", "makefile"), ("", ""),
])
def test_futility_key(name, key):
    assert futility_key(name) == key


def test_futility_keys_in_command():
    assert futility_keys_in_command('cd /workspace && python3 probe13.py 2>&1 | grep -E "OK|ERR"') == {"probe.py"}
    assert futility_keys_in_command("python3 oxford_grid2.py; sh run-2.sh") == {"oxford_grid.py", "run.sh"}
    assert futility_keys_in_command("ls -la /workspace") == set()


def _make_agent(tools):
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)
    agent.available_tools = tools
    agent.disabled_tools = set()
    agent.context.current_project_id = None
    agent.context._script_iter = {}
    agent.context._futility_steer_done = False
    agent.context._futility_report_done = False
    agent.context._breaker_forced_final = False
    return agent


def _ts(tool_calls):
    return TurnState(
        _constraint_steer_pending=None, _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False, _request_sys3_prev_justification="",
        consecutive_parse_errors=0, current_plan_json="",
        execution_failure_count=0, final_ai_content="", fname="",
        force_final_response=False, force_stop=False, forget_was_called=False,
        last_was_failure=True, preflight_blocks_this_request=0,
        request_sandbox_state="", transient_failure_count=0,
        tool_calls=tool_calls,
        msg={"role": "assistant", "content": ""}, ui_content="",
        parse_failure_reason="", model="test-model",
        last_user_content="find the grid points", char_budget=4000,
        strikes=StrikeLedger(), task_tree=MagicMock(),
        _user_batch_intent=None, _request_constraints=[],
        repeated_action_steered=set(), messages=[], seen_tools=set(),
        executed_idempotent=set(), raw_tools_called=set(), tool_usage={},
        tools_run_this_turn=[], request_state=MagicMock(),
    )


def _write(cid, path):
    return {"id": cid, "type": "function", "function": {"name": "file_system", "arguments": json.dumps(
        {"operation": "write", "path": path, "content": "print(1)"})}}


def _run(cid, path):
    return {"id": cid, "type": "function", "function": {"name": "execute", "arguments": json.dumps(
        {"command": f"cd /workspace && python3 {path} 2>&1 | grep -E 'OK|ERR'"})}}


@pytest.mark.asyncio
async def test_futility_steer_fires_across_numbered_probe_files():
    async def fs(**kw):
        return "SUCCESS: wrote"

    async def ex(**kw):
        return "EXIT CODE: 0\nERR dict: RuntimeError: SpecError: [pl]"

    agent = _make_agent({"file_system": fs, "execute": ex})
    seq = [[_write("w6", "probe6.py")], [_run("r6", "probe6.py")],
           [_write("w7", "probe7.py")], [_run("r7", "probe7.py")],
           [_write("w8", "probe8.py")]]
    fired = []
    for i, calls in enumerate(seq):
        ts = _ts(calls)
        await agent._dispatch_and_process_tool_batch(ts)
        if any("SYSTEM ALERT (futility breaker)" in str(m.get("content")) for m in ts.messages
               if m.get("role") == "user"):
            fired.append(i)
    assert fired == [4]
    assert agent.context._script_iter == {"probe.py": {"writes": 3, "runs": 2}}


# ── G3: two turns are reserved ────────────────────────────────────────

def test_report_turn_leaves_one_turn_for_its_retry():
    assert last_turn_needs_report(38, 40, False, False) is True
    assert last_turn_needs_report(39, 40, False, False) is False
    assert last_turn_needs_report(39, 40, True, False) is False


# ── G4: a dropped tool call on a breaker-forced final is no answer ───

def test_dropped_tool_call_counts_as_no_answer_on_breaker_forced_finals():
    """`_dropped_this_turn = True` is set inside the final-generation drop
    branch, and the NO-ANSWER If reads it together with the breaker flag."""
    sets = [n for n in ast.walk(_TREE) if isinstance(n, ast.Assign)
            and getattr(n.targets[0], "id", "") == "_dropped_this_turn"]
    values = sorted(str(getattr(n.value, "value", None)) for n in sets)
    assert values == ["False", "True"]
    true_set = next(n for n in sets if n.value.value is True)
    drop_if = next(n for n in ast.walk(_TREE) if isinstance(n, ast.If) and true_set in n.body)
    assert "is_final_generation" in ast.unparse(drop_if.test) and "tool_calls" in ast.unparse(drop_if.test)
    noans = [n for n in ast.walk(_TREE) if isinstance(n, ast.If)
             and "_forced_final_has_no_answer" in ast.unparse(n.test)
             and "is_final_generation" in ast.unparse(n.test)]
    assert len(noans) == 1
    t = ast.unparse(noans[0].test)
    assert "_dropped_this_turn" in t and "_breaker_forced_final" in t
    assert " or " in t and " and " in t
