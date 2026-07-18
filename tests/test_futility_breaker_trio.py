"""Struggle-mitigation trio (2026-07-18, xrick coding-session postmortem).

Request 5b9fcc8f rewrote extract_data.py 5x and reran it 5x (every run
exit 0, goal counts never met), failed FOUR consecutive repairs of the
same `if c == '\\':` unterminated-string bug, and finally died when an
n-gram thinking kill landed ON turn 40 — the grounding retry had no next
iteration and working narration shipped as the "answer".

Covers:
  * edit-run futility breaker (3 writes + 2 runs of one script → one
    strategy-shift steer per request)
  * error-keyed fix recipes on the write-time syntax gate
  * turn-budget exhaustion banner (for/else on the turn loop)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import ast
import inspect
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.agent import GhostAgent, TurnState
from ghost_agent.core.strikes import StrikeLedger


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
    return agent


def _ts(tool_calls):
    fields = dict(
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
        last_user_content="fix the extractor", char_budget=4000,
        strikes=StrikeLedger(), task_tree=MagicMock(),
        _user_batch_intent=None, _request_constraints=[],
        repeated_action_steered=set(), messages=[], seen_tools=set(),
        executed_idempotent=set(), raw_tools_called=set(), tool_usage={},
        tools_run_this_turn=[], request_state=MagicMock(),
    )
    return TurnState(**fields)


def _write_call(cid, path):
    return {"id": cid, "type": "function",
            "function": {"name": "file_system",
                         "arguments": json.dumps(
                             {"operation": "write", "path": path,
                              "content": "print(1)"})}}


def _run_call(cid, path):
    return {"id": cid, "type": "function",
            "function": {"name": "execute",
                         "arguments": json.dumps(
                             {"command": f"python3 {path}"})}}


def _steers(ts):
    return [m for m in ts.messages
            if m.get("role") == "user"
            and "SYSTEM ALERT (futility breaker)" in str(m.get("content"))]


# ── futility breaker ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_futility_breaker_fires_on_third_write_second_run():
    async def fs(**kw):
        return "SUCCESS: wrote"

    async def ex(**kw):
        return "EXIT CODE: 0\nran fine"

    agent = _make_agent({"file_system": fs, "execute": ex})
    seq = [
        [_write_call("w1", "extract_data.py")],
        [_run_call("r1", "extract_data.py")],
        [_write_call("w2", "extract_data.py")],
        [_run_call("r2", "extract_data.py")],
        [_write_call("w3", "extract_data.py")],   # 3 writes + 2 runs → trip
    ]
    fired_at = []
    for i, calls in enumerate(seq):
        ts = _ts(calls)
        await agent._dispatch_and_process_tool_batch(ts)
        if _steers(ts):
            fired_at.append(i)
    assert fired_at == [4]
    assert agent.context._futility_steer_done is True


@pytest.mark.asyncio
async def test_futility_steer_content_and_once_per_request():
    async def fs(**kw):
        return "SUCCESS: wrote"

    async def ex(**kw):
        return "EXIT CODE: 0"

    agent = _make_agent({"file_system": fs, "execute": ex})
    for calls in ([_write_call("a", "s.py")], [_run_call("b", "s.py")],
                  [_write_call("c", "s.py")], [_run_call("d", "s.py")]):
        await agent._dispatch_and_process_tool_batch(_ts(calls))
    ts = _ts([_write_call("e", "s.py")])
    await agent._dispatch_and_process_tool_batch(ts)
    (steer,) = _steers(ts)
    content = steer["content"]
    assert "RECORD" in content and "ledger" in content
    assert "SHRINK" in content and "smallest unit" in content
    assert "switch class" in content
    # 4th write → no second steer
    ts2 = _ts([_write_call("f", "s.py")])
    await agent._dispatch_and_process_tool_batch(ts2)
    assert _steers(ts2) == []


@pytest.mark.asyncio
async def test_futility_breaker_ignores_non_code_files():
    async def fs(**kw):
        return "SUCCESS: wrote"

    async def ex(**kw):
        return "EXIT CODE: 0"

    agent = _make_agent({"file_system": fs, "execute": ex})
    for cid in ("1", "2", "3", "4", "5"):
        ts = _ts([_write_call(cid, "index.html")])
        await agent._dispatch_and_process_tool_batch(ts)
        assert _steers(ts) == []


# ── syntax-gate recipes ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_backslash_trap_gets_recipe(tmp_path):
    from ghost_agent.tools.file_system import _syntax_feedback
    bad = tmp_path / "parser.py"
    bad.write_text("def f(text, i):\n    if text[i] == '\\':\n        return 1\n")
    out = await _syntax_feedback(bad, "parser.py")
    assert "SYNTAX CHECK FAILED" in out
    assert "RECIPE:" in out
    assert "chr(92)" in out


@pytest.mark.asyncio
async def test_clean_file_no_feedback(tmp_path):
    from ghost_agent.tools.file_system import _syntax_feedback
    ok = tmp_path / "fine.py"
    ok.write_text("def f():\n    return 1\n")
    assert await _syntax_feedback(ok, "fine.py") == ""


@pytest.mark.asyncio
async def test_unrelated_syntax_error_gets_no_recipe(tmp_path):
    from ghost_agent.tools.file_system import _syntax_feedback
    bad = tmp_path / "b.py"
    bad.write_text("def f(:\n    pass\n")
    out = await _syntax_feedback(bad, "b.py")
    assert "SYNTAX CHECK FAILED" in out
    assert "RECIPE:" not in out


# ── turn-budget exhaustion banner ────────────────────────────────────

def test_turn_loop_has_else_clause_with_banner():
    import ghost_agent.core.agent as agent_mod
    src = inspect.getsource(agent_mod)
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            it = ast.get_source_segment(src, node.iter) or ""
            if "effective_max_turns" in it:
                assert node.orelse, "turn loop lost its else clause"
                seg = "\n".join(
                    ast.get_source_segment(src, n) or "" for n in node.orelse)
                assert "[TURN BUDGET EXHAUSTED]" in seg
                assert "NOT a finished result" in seg
                found = True
    assert found
