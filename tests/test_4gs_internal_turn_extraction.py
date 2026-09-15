"""§4GS — decomposition step 4b: `handle_chat` → `_run_internal_turn`.

The 2026-07-23 attempt stopped here deliberately: the transform was solved,
the INPUT/REPACK sets were not. Every AST heuristic missed a different class
of loop-carried state, and the dangerous miss is SILENT — a steering flag or
counter written in the region and read on the NEXT turn, across the loop
back-edge, goes stale with no crash to catch it.

`scripts/liveness_4b.py` computes those sets from BYTECODE: live-in at the
region's entry instruction is the input set, live-out across every exit
(the back-edge included, because on a CFG it is just another successor) is
the repack set. Measured on this region: 76 inputs, 21 repack, 9 cells of
which 6 are region-only and `lc` / `raw_tools_called` are read-only.

These pins hold the extraction's contract:
  * the boundary protocol — "continue" / "break" bind to the TURN loop,
    "proceed" falls through to the step-2 dispatch that follows;
  * the method needs nothing from its caller's frame beyond `rs` (bytecode
    liveness, re-derived here, so moving code across the boundary FAILS
    instead of going stale);
  * the repack is symmetric — what the method writes back is exactly what
    the call site copies out;
  * and loop-carried state really does survive the round trip.
"""
import ast
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.agent import GhostAgent, InternalTurnState


# ── the boundary protocol ─────────────────────────────────────────────────

def test_the_method_returns_the_three_way_flow_and_nothing_else():
    tree = ast.parse(inspect.getsource(agent_mod))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "_run_internal_turn")
    returned = set()

    class R(ast.NodeVisitor):
        # ⚠ NESTED FUNCTIONS ARE NOT THE METHOD. The region carries its own
        # closures (`_emit_thinking`, `_flush_thinking`, …) and their bare
        # `return`s are theirs, not the boundary protocol's.
        def visit_FunctionDef(self, n):
            pass
        visit_AsyncFunctionDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

        def visit_Return(self, n):
            assert isinstance(n.value, ast.Constant), ast.dump(n)[:90]
            returned.add(n.value.value)
    r = R()
    for st in fn.body:
        r.visit(st)
    assert returned == {"continue", "break", "proceed"}, sorted(returned)


def test_no_turn_loop_control_flow_survived_inside_the_method():
    """A `continue`/`break` left behind would bind to some INNER loop and
    silently mean something else — the shape the transform exists to
    remove."""
    tree = ast.parse(inspect.getsource(agent_mod))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "_run_internal_turn")

    class V(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0
            self.loose = []

        def visit_For(self, n):
            self.depth += 1
            self.generic_visit(n)
            self.depth -= 1
        visit_AsyncFor = visit_For
        visit_While = visit_For

        def visit_FunctionDef(self, n):
            pass
        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Continue(self, n):
            if self.depth == 0:
                self.loose.append(n.lineno)

        def visit_Break(self, n):
            if self.depth == 0:
                self.loose.append(n.lineno)
    v = V()
    for st in fn.body:
        v.visit(st)
    assert v.loose == [], v.loose


def test_the_call_site_switches_on_all_three_outcomes():
    tree = ast.parse(inspect.getsource(agent_mod))
    hc = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_chat")
    called = [n for n in ast.walk(hc)
              if isinstance(n, ast.Call)
              and getattr(n.func, "attr", "") == "_run_internal_turn"]
    assert len(called) == 1, len(called)
    flows = {c.comparators[0].value for c in ast.walk(hc)
             if isinstance(c, ast.Compare)
             and getattr(c.left, "id", "") == "_flow"
             and isinstance(c.comparators[0], ast.Constant)}
    assert flows == {"continue", "break"}, sorted(flows)


# ── the repack is symmetric ───────────────────────────────────────────────

def _finally_targets(fn_name, tree):
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == fn_name)
    out = set()
    for n in ast.walk(fn):
        if not isinstance(n, ast.Try):
            continue
        for stmt in n.finalbody:
            for a in ast.walk(stmt):
                if isinstance(a, ast.Assign) and len(a.targets) == 1:
                    t, v = a.targets[0], a.value
                    if (isinstance(t, ast.Attribute)
                            and getattr(t.value, "id", "") == "rs"):
                        out.add(("method", t.attr))
                    if (isinstance(v, ast.Attribute)
                            and getattr(v.value, "id", "") == "_its"
                            and isinstance(t, ast.Name)):
                        out.add(("caller", t.id))
    return out


def test_what_the_method_writes_back_is_what_the_caller_copies_out():
    tree = ast.parse(inspect.getsource(agent_mod))
    method = {n for k, n in _finally_targets("_run_internal_turn", tree) if k == "method"}
    caller = {n for k, n in _finally_targets("handle_chat", tree) if k == "caller"}
    assert method, "the method's finally repacks nothing"
    assert method == caller, (
        f"asymmetric repack — method only: {sorted(method - caller)}; "
        f"caller only: {sorted(caller - method)}")
    # …and every repacked name is a field, or the copy-out reads a ghost
    fields = set(InternalTurnState.__dataclass_fields__)
    assert method <= fields, sorted(method - fields)


def test_every_field_is_unpacked_at_the_top_so_the_finally_cannot_raise():
    """A repack name unbound when the method raises makes the `finally`
    throw a NameError over the real exception. Read the prologue from the
    AST: the statements BEFORE the method's `try` must bind every field."""
    tree = ast.parse(inspect.getsource(agent_mod))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "_run_internal_turn")
    prologue = []
    for st in fn.body:
        if isinstance(st, ast.Try):
            break
        prologue.append(st)
    bound = {st.targets[0].id for st in prologue
             if isinstance(st, ast.Assign) and len(st.targets) == 1
             and isinstance(st.targets[0], ast.Name)
             and isinstance(st.value, ast.Attribute)
             and getattr(st.value.value, "id", "") == "rs"
             and st.value.attr == st.targets[0].id}
    missing = set(InternalTurnState.__dataclass_fields__) - bound
    assert not missing, sorted(missing)


# ── the analysis itself, re-derived (self-calibrating) ───────────────────

def test_the_method_needs_nothing_from_its_callers_frame():
    """Bytecode liveness at the method's entry: everything it reads must
    come from `rs` (or be `self`). If someone moves code in that reads a
    handle_chat local the state does not carry, this fails — which is the
    check the 2026-07-23 attempt did not have."""
    from liveness_4b import liveness
    code = GhostAgent._run_internal_turn.__code__
    _by, order, _succ, live_in, _lo = liveness(code)
    entry_live = live_in[order[0]]
    assert entry_live <= {"self", "rs"}, sorted(entry_live)


def test_the_symtable_has_no_free_names_in_the_method():
    import symtable
    src = inspect.getsource(agent_mod)
    st = symtable.symtable(src, "agent.py", "exec")

    def find(table, name):
        for c in table.get_children():
            if c.get_name() == name:
                return c
            r = find(c, name)
            if r:
                return r
        return None
    m = find(find(st, "GhostAgent"), "_run_internal_turn")
    assert m is not None
    assert sorted(m.get_frees()) == [], sorted(m.get_frees())


# ── the silent failure the memory named: state across the back-edge ──────

def _scripted_agent(reply_fn):
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.agent import GhostContext
    ctx = MagicMock(spec=GhostContext)
    ctx.llm_client = MagicMock()
    ctx.llm_client.vision_clients = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.args = MagicMock()
    ctx.args.shell = "bash"
    ctx.args.max_context = 8000
    ctx.args.temperature = 0.5
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    ctx.args.model = "qwen3.6"
    ctx.args.perfect_it = False
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.memory_system = None
    ctx.skill_memory = None
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = ""
    agent = GhostAgent(ctx)
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=reply_fn)
    agent.available_tools = {}
    agent.disabled_tools = set()
    return agent


@pytest.mark.asyncio
async def test_the_failure_counter_survives_the_loop_back_edge():
    """THE pin for the failure the 2026-07-23 note called silent.

    `execution_failure_count` is written inside the region and read on the
    NEXT turn; the strike cap at six aborts the loop. If the repack misses
    it — the exact shape a "read-after-region" scan cannot see — the counter
    resets every turn, the cap never fires, and the loop burns every one of
    its turns instead. No crash, no red test, just forty LLM calls where
    there should be seven.
    """
    from unittest.mock import patch
    calls = {"n": 0}

    async def always_calls_a_missing_tool(payload, **kw):
        calls["n"] += 1
        return {"choices": [{"message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{calls['n']}", "type": "function",
                            "function": {"name": "nope", "arguments": "{}"}}],
        }}]}

    agent = _scripted_agent(always_calls_a_missing_tool)

    class Bg:
        def add_task(self, *a, **k):
            pass

    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "nope"}}]):
        final, _, _ = await agent.handle_chat(
            {"messages": [{"role": "user", "content": "do the thing"}]}, Bg())

    assert calls["n"] <= 10, (
        f"{calls['n']} LLM calls — the strike cap never fired, so the "
        "failure counter is not surviving the turn-loop back edge")
    assert "hard limit after repeated failures" in final, final[:200]
