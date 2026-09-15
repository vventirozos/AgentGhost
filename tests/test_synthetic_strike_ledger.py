"""Every synthetic rejection that counts as a strike reaches the ledger —
2026-09-13.

The dispatch loop mints rejections of its own (unknown or disabled tool,
unparseable call, bad JSON arguments, blocked empty write, participant
constraint, invocation error). Each did `execution_failure_count += 1`, but
only ONE (the invocation error, fixed earlier) also told the `StrikeLedger`.
The batch tail decays the counter whenever every DISPATCHED call succeeded
and the ledger holds no failure — so a batch of `[hallucinated_tool, read]`
netted to zero strikes, every turn, and the 6-strike cap and the
same-failure loop breaker never fired on the model's most persistent
mistake. One closure, `_strike_synthetic`, now does all three things at
every site; the AST enumeration below fails the moment a site bypasses it.

Worlds where these pins fail: the pre-fix tree (counter decays back to 0;
`decay_frozen` never set), and any tree where a new synthetic site
increments the counter without the closure.
"""
import ast
import inspect
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.agent import GhostAgent, TurnState
from ghost_agent.core.strikes import StrikeLedger


# ── R1 enumeration: the class, from the AST ──────────────────────────────────

def _is_synthetic_append(stmt) -> bool:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    f = stmt.value.func
    if not (isinstance(f, ast.Attribute) and f.attr == "append"
            and isinstance(f.value, ast.Name) and f.value.id == "tools_run_this_turn"):
        return False
    for arg in stmt.value.args:
        if isinstance(arg, ast.Dict):
            for k in arg.keys:
                if isinstance(k, ast.Constant) and k.value == "_synthetic":
                    return True
    return False


def _increments_counter(stmt) -> bool:
    return (isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "execution_failure_count")


def _calls_strike_closure(stmt) -> bool:
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "_strike_synthetic")


def synthetic_strike_sites(source: str):
    """(lineno, increments, ledgered) for every statement list inside
    `_dispatch_and_process_tool_batch` that appends a `_synthetic` row.
    A site that increments the strike counter must call the closure in
    the SAME statement list."""
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "_dispatch_and_process_tool_batch")
    out = []
    for node in ast.walk(fn):
        for field in ("body", "orelse", "finalbody", "handlers"):
            stmts = getattr(node, field, None)
            if not isinstance(stmts, list) or not stmts or not isinstance(stmts[0], ast.stmt):
                continue
            appends = [s for s in stmts if _is_synthetic_append(s)]
            if not appends:
                continue
            out.append((appends[0].lineno,
                        any(_increments_counter(s) for s in stmts),
                        any(_calls_strike_closure(s) for s in stmts)))
    return sorted(out)


def test_every_synthetic_site_that_counts_a_strike_tells_the_ledger():
    src = inspect.getsource(agent_mod)
    sites = synthetic_strike_sites(src)
    striking = [s for s in sites if s[1]]
    assert len(sites) >= 10, sites            # the enumeration sees the class
    assert len(striking) >= 7, striking       # …and its striking members
    unledgered = [s for s in striking if not s[2]]
    assert unledgered == [], f"synthetic strike sites without the ledger: {unledgered}"
    # deliberate NON-strikes (idempotency note, preflight block, imagine
    # note) exist and stay exempt — the rule is about the counter
    assert any(not s[1] for s in sites), sites


def test_the_enumeration_fires_when_one_site_drops_the_closure():
    """R6/R7-2: the instrument can fail. Remove one closure call from the
    source and the checker must report exactly that site."""
    src = inspect.getsource(agent_mod)
    needle = '_strike_synthetic(fname, "unknown_tool")'
    assert src.count(needle) == 1
    broken = src.replace(needle, "pass")
    bad = [s for s in synthetic_strike_sites(broken) if s[1] and not s[2]]
    assert len(bad) == 1, bad


# ── behaviour, through the REAL dispatch loop with a REAL ledger ─────────────

def _make_agent(tools, disabled=()):
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    agent = GhostAgent(ctx)
    agent.available_tools = dict(tools)
    agent.disabled_tools = set(disabled)
    # an unknown name makes the loop rebuild the tool map from the live
    # registry (a skill created mid-session) — which would wipe the fakes
    agent._rebuild_available_tools = lambda: None
    return agent


def _make_ts(tool_calls, strikes, failures=0):
    return TurnState(
        _constraint_steer_pending=None, _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False, _request_sys3_prev_justification="",
        consecutive_parse_errors=0, current_plan_json="",
        execution_failure_count=failures, final_ai_content="", fname="",
        force_final_response=False, force_stop=False, forget_was_called=False,
        last_was_failure=False, preflight_blocks_this_request=0,
        request_sandbox_state="", transient_failure_count=0,
        tool_calls=tool_calls, msg={"role": "assistant", "content": ""},
        ui_content="", parse_failure_reason="", model="test-model",
        last_user_content="do the thing", char_budget=4000,
        strikes=strikes, task_tree=MagicMock(), _user_batch_intent=None,
        _request_constraints=[], repeated_action_steered=set(), messages=[],
        seen_tools=set(), executed_idempotent=set(), raw_tools_called=set(),
        tool_usage={}, tools_run_this_turn=[], request_state=MagicMock(),
    )


def _call(cid, name, args):
    raw = args if isinstance(args, str) else json.dumps(args)
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": raw}}


async def _reader(**kw):
    return "fine: 3 files"


async def _fs(**kw):
    return "SUCCESS: Wrote 5 chars to 'x.py'."


@pytest.mark.asyncio
@pytest.mark.parametrize("label,tools,disabled,bad_call", [
    ("unknown tool", {"reader": _reader}, (), _call("c1", "no_such_tool", {})),
    ("disabled tool", {"reader": _reader, "gone": _reader}, ("gone",), _call("c1", "gone", {})),
    ("bad JSON arguments", {"reader": _reader}, (), _call("c1", "reader", "{not json")),
    ("blocked empty write", {"reader": _reader, "file_system": _fs}, (),
     _call("c1", "file_system", {"operation": "write", "path": "temp.py", "content": ""})),
])
async def test_a_synthetic_strike_survives_a_clean_read_in_the_same_batch(
        label, tools, disabled, bad_call):
    """Pre-fix: +1 for the rejection, −1 at the batch tail (every
    DISPATCHED call succeeded, ledger empty) → 0. The strike must stand."""
    agent = _make_agent(tools, disabled)
    ts = _make_ts([bad_call, _call("c2", "reader", {"q": "1"})], StrikeLedger())
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.execution_failure_count == 1, label
    assert ts.strikes.failure_sigs, f"{label}: the ledger saw nothing"
    assert ts.strikes.consecutive_clean_successes <= 1, label


@pytest.mark.asyncio
async def test_the_same_synthetic_rejection_three_times_freezes_decay():
    """The same-failure loop breaker needs the signature: three identical
    hallucinated calls across batches must freeze the ledger."""
    agent = _make_agent({"reader": _reader})
    strikes = StrikeLedger()
    failures = 0
    for i in range(3):
        ts = _make_ts([_call(f"c{i}", "no_such_tool", {}),
                       _call(f"r{i}", "reader", {"q": str(i)})], strikes, failures)
        await agent._dispatch_and_process_tool_batch(ts)
        failures = ts.execution_failure_count
    assert failures == 3
    assert strikes.decay_frozen is True


@pytest.mark.asyncio
async def test_a_clean_batch_still_decays_a_real_strike():
    """Regression guard for the fix's own blast radius: with no synthetic
    rejection in the batch the decay is unchanged."""
    agent = _make_agent({"reader": _reader})
    ts = _make_ts([_call("c1", "reader", {"q": "1"})], StrikeLedger(), failures=2)
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.execution_failure_count == 1


# ── R3 review of the fix: the two sites without a behavioural pin, and the
#    operator line the ledger entry earns ───────────────────────────────────

def _raises_on_call(**kw):
    raise TypeError("unexpected keyword argument 'q'")


@pytest.mark.asyncio
@pytest.mark.parametrize("label,tools,bad_call", [
    ("parse error", {"reader": _reader}, _call("c1", "system_parse_error", {})),
    ("invocation error", {"reader": _reader, "picky": _raises_on_call}, _call("c1", "picky", {"q": "1"})),
])
async def test_the_remaining_synthetic_strikes_survive_a_clean_read(label, tools, bad_call):
    agent = _make_agent(tools)
    ts = _make_ts([bad_call, _call("c2", "reader", {"q": "1"})], StrikeLedger())
    await agent._dispatch_and_process_tool_batch(ts)
    assert ts.execution_failure_count == 1, label
    assert ts.strikes.failure_sigs, label


@pytest.mark.asyncio
async def test_the_third_identical_synthetic_rejection_prints_the_loop_breaker_line(capsys):
    agent = _make_agent({"reader": _reader})
    strikes = StrikeLedger()
    failures = 0
    for i in range(3):
        ts = _make_ts([_call(f"c{i}", "no_such_tool", {}),
                       _call(f"r{i}", "reader", {"q": str(i)})], strikes, failures)
        await agent._dispatch_and_process_tool_batch(ts)
        failures = ts.execution_failure_count
    out = capsys.readouterr().out
    assert out.count("Same synthetic rejection ×3") == 1, out
