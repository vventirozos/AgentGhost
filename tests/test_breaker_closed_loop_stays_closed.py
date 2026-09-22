"""§4ID — a loop a breaker closed stays closed.

THE LIVE FAILURE (probe ifs17585…, 2026-09-17, the IFS/Oxford grid ask).
The futility breaker's report tier fired at turn 30 ("rewritten 6x / rerun
5x — forcing a blocker report"), the forced final missed twice and the
honest fallback was about to ship — then the verifier gate saw that the
last tool was the very file write that tripped the breaker, called it an
"unverified mutation (untested write)", and re-opened the loop for a repair
round: turns 33–39 rewrote and reran the same probe.py four more times
until the reserved turn 40. The untested write IS the churn the breaker
stopped, not a deliverable to test.

Also: §4IC never reset `_futility_report_done` per request, so after one
report the tier could never fire again in the process's lifetime.

World where each pin fails: a breaker site stops raising the flag, the
gate stops consulting it, a per-request reset is dropped, the report
alert stops forbidding predicted outputs, or the futility steer stops
telling the model to look the API up before guessing again.
"""
import ast
import inspect
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import GhostAgent, TurnState, blocker_report_alert
from ghost_agent.core.strikes import StrikeLedger

FLAG = "_breaker_forced_final"


_MODULE_TREE = ast.parse(inspect.getsource(ag))


def _tree(obj):
    """The function's AST, located by name in the MODULE tree (dedenting
    `handle_chat`'s own source breaks on its multi-line strings)."""
    for n in ast.walk(_MODULE_TREE):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == obj.__name__:
            return n
    raise AssertionError(obj.__name__)


def _assigns_flag(body_nodes, value):
    """True when one of `body_nodes` (walked) assigns self.context.FLAG = value."""
    for n in body_nodes:
        for a in ast.walk(n):
            if (isinstance(a, ast.Assign) and len(a.targets) == 1
                    and isinstance(a.targets[0], ast.Attribute)
                    and a.targets[0].attr == FLAG
                    and isinstance(a.value, ast.Constant) and a.value.value is value):
                return True
    return False


def _enclosing_bodies_with(tree, pred):
    """Every statement list (If/For/With/Try bodies, function bodies) that
    contains a statement satisfying `pred` DIRECTLY."""
    out = []
    for n in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(n, field, None)
            if isinstance(stmts, list) and any(pred(s) for s in stmts):
                out.append(stmts)
    return out


def _is_assign_to(stmt, attr_or_name, value=None):
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return False
    t = stmt.targets[0]
    name = t.attr if isinstance(t, ast.Attribute) else getattr(t, "id", None)
    if name != attr_or_name:
        return False
    return value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is value)


# ── the three breaker sites raise the flag in the SAME body ───────────

def test_futility_report_tier_raises_the_flag():
    tree = _tree(GhostAgent._dispatch_and_process_tool_batch)
    bodies = _enclosing_bodies_with(tree, lambda s: _is_assign_to(s, "_futility_report_done", True))
    assert len(bodies) == 1
    assert _assigns_flag(bodies[0], True)


def test_edit_churn_report_tier_raises_the_flag():
    tree = _tree(GhostAgent._dispatch_and_process_tool_batch)
    def is_stop_log(s):
        return (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                and getattr(s.value.func, "id", "") == "pretty_log"
                and s.value.args and isinstance(s.value.args[0], ast.Constant)
                and s.value.args[0].value == "Edit Churn Stop")
    bodies = _enclosing_bodies_with(tree, is_stop_log)
    assert len(bodies) == 1
    assert _assigns_flag(bodies[0], True)


def test_reserved_report_turn_raises_the_flag():
    tree = _tree(GhostAgent.handle_chat)
    bodies = _enclosing_bodies_with(tree, lambda s: _is_assign_to(s, "_report_turn_forced", True))
    # two report sites since 2026-09-21: the reserved last turn and the
    # client-deadline report (§4JP) — each must arm the flag
    assert len(bodies) == 2
    assert all(_assigns_flag(b, True) for b in bodies)


# ── the gate consults it, before the repair branch ────────────────────

def test_unverified_repair_is_gated_on_the_flag():
    src = inspect.getsource(ag)
    tree = ast.parse(src)
    crit_ifs = []          # If nodes whose body assigns _crit = "unverified mutation …"
    guard_ifs = []         # If nodes whose test reads the flag under `not`/as a bare getattr
    for n in ast.walk(tree):
        if not isinstance(n, ast.If):
            continue
        for s in n.body:
            if (_is_assign_to(s, "_crit") and isinstance(s.value, ast.Constant)
                    and str(s.value.value).startswith("unverified mutation")):
                crit_ifs.append(n)
        if FLAG in ast.unparse(n.test):
            guard_ifs.append(n)
    assert len(crit_ifs) == 1, "the untested-write repair branch moved"
    assert guard_ifs, "no branch consults the breaker-closed flag"
    # The guard is an elif SIBLING evaluated BEFORE the repair branch: the
    # repair If must be the `orelse` of the guard If.
    crit = crit_ifs[0]
    assert any(g.orelse and g.orelse[0] is crit for g in guard_ifs), \
        "the breaker guard must be tested before the untested-write repair"
    g = next(g for g in guard_ifs if g.orelse and g.orelse[0] is crit)
    # …and that guard sets no repair and both read `_unverified`.
    assert "_unverified" in ast.unparse(g.test)
    # POLARITY: the guard fires when the flag is SET. An inverted test
    # (`not getattr(…)`) would skip the repair for every ordinary request
    # and hand the breaker-closed loop straight to the repair branch.
    flag_reads = [n for n in ast.walk(g.test) if isinstance(n, ast.Call)
                  and any(isinstance(a, ast.Constant) and a.value == FLAG for a in n.args)]
    assert len(flag_reads) == 1
    negated = [n for n in ast.walk(g.test) if isinstance(n, ast.UnaryOp)
               and isinstance(n.op, ast.Not) and flag_reads[0] in list(ast.walk(n))]
    assert not negated, "the breaker guard reads the flag negated"
    assert isinstance(g.test, ast.BoolOp) and isinstance(g.test.op, ast.And)
    assert any(_is_assign_to(s, "_do_repair", False) for s in g.body)
    assert not any(_is_assign_to(s, "_do_repair", True) for s in g.body)


# ── per-request resets ────────────────────────────────────────────────

@pytest.mark.parametrize("attr", ["_futility_steer_done", "_futility_report_done", FLAG])
def test_request_start_resets(attr):
    tree = _tree(GhostAgent.handle_chat)
    hits = [s for s in ast.walk(tree) if _is_assign_to(s, attr, False)
            and isinstance(s.targets[0], ast.Attribute)]
    assert len(hits) == 1, attr


# ── the report alert forbids predicted outputs ────────────────────────

def test_report_alert_forbids_predicted_outputs():
    text = blocker_report_alert("futility", "x")
    assert "did not observe" in text or "not observe" in text
    assert "UNTESTED" in text
    assert "complete solution" in text          # the exact phrase the live reply used


# ── the futility steer says LOOK IT UP before another guess ───────────

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
        {"command": f"python3 {path}"})}}


@pytest.mark.asyncio
async def test_futility_steer_tells_the_model_to_look_the_api_up():
    async def fs(**kw):
        return "SUCCESS: wrote"

    async def ex(**kw):
        return "EXIT CODE: 0\nGrid: cannot build grid without 'type'"

    agent = _make_agent({"file_system": fs, "execute": ex})
    for calls in ([_write("a", "probe.py")], [_run("b", "probe.py")],
                  [_write("c", "probe.py")], [_run("d", "probe.py")]):
        await agent._dispatch_and_process_tool_batch(_ts(calls))
    ts = _ts([_write("e", "probe.py")])
    await agent._dispatch_and_process_tool_batch(ts)
    steers = [m for m in ts.messages if m.get("role") == "user"
              and "SYSTEM ALERT (futility breaker)" in str(m.get("content"))]
    (steer,) = steers
    c = steer["content"]
    assert "STOP guessing" in c and "look it up" in c
    assert "`search`" in c                      # the tool by its registered name
    assert "help()" in c
    assert "compute the result another way" in c


@pytest.mark.asyncio
async def test_futility_report_tier_sets_the_flag_live():
    """Behavioural twin of the AST pin: drive 6 writes + 4 runs and read the flag."""
    async def fs(**kw):
        return "SUCCESS: wrote"

    async def ex(**kw):
        return "EXIT CODE: 0"

    agent = _make_agent({"file_system": fs, "execute": ex})
    seq = []
    for i in range(6):
        seq.append([_write(f"w{i}", "probe.py")])
        if i < 4:
            seq.append([_run(f"r{i}", "probe.py")])
    forced_at = []
    for i, calls in enumerate(seq):
        ts = _ts(calls)
        await agent._dispatch_and_process_tool_batch(ts)
        if ts.force_final_response:
            forced_at.append(i)
    assert forced_at, "report tier never fired"
    assert agent.context._breaker_forced_final is True
    assert agent.context._futility_report_done is True
