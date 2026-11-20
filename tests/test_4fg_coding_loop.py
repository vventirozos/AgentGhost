"""§4FG pins: the agentic coding-leaf executor (core/coding_loop.py).

The sub-loop itself (a GhostAgent on an isolated context) is replaced by a
fake in these pins; what is pinned is everything around it — the contract
parser, the workspace diff, the gates, the fresh-attempt-with-witness
policy, the isolation shape, and the dispatch seam every caller inherits.
"""
import os
from pathlib import Path
from types import SimpleNamespace


from ghost_agent.core import coding_loop as cl








def _ctx(tmp_path, constraints=None):
    class Store:
        def get_project(self, pid):
            return {"id": pid, "workspace_dir": str(tmp_path), "metadata": {}}
    return SimpleNamespace(current_project_id="p1", project_store=Store(),
                           llm_client=object(), args=SimpleNamespace(model="m"))


def _runner_ok(calls):
    async def run(name, args):
        calls.append((name, args))
        if name == "execute":
            return "3 passed\nEXIT CODE: 0"
        return "ok"
    return run










def test_leaf_context_keeps_the_project_and_wraps_memory_read_only():
    """World where it fails: the isolation nulls the project (paths would
    resolve to the sandbox root, the §4EI class) or leaves memory writable
    (a leaf would teach)."""
    from ghost_agent.core.isolation import is_simulated_context
    base = SimpleNamespace(sandbox_dir="/tmp/sb", current_project_id=None,
                           memory_system=object(), graph_memory=None,
                           args=SimpleNamespace(perfect_it=True, smart_memory=0.5, native_tools=False),
                           workspace_model=object(), journal=object(), llm_client=object())
    # the base is UNBOUND (as the advancer's context is); the leaf pins the
    # project explicitly
    iso = cl.build_leaf_context(base, leaf_id="L1", project_id="p1")
    assert iso.current_project_id == "p1" and iso.sandbox_dir == "/tmp/sb"
    assert base.current_project_id is None           # the live context is untouched
    assert is_simulated_context(iso) and not is_simulated_context(base)   # §4TD: the mark, not a wrapper
    assert iso.trajectory_task_kind == "leaf" and iso.turn_origin_label == "leaf"
    assert iso.trajectory_extra_static == {"leaf_id": "L1"}
    assert iso.workspace_model is None and iso.journal is None
    assert iso.args.perfect_it is False and iso.args.native_tools is True
    # §4KB round 5: `git` joined this set and was REMOVED. Four review rounds
    # produced nine sandbox escapes and every one came from the git tool.
    # Asserted as the EXACT set, not a superset: the pin exists to catch a
    # tool silently entering the leaf — which is how `git` got here.
    assert iso._subagent_allowed_tools == frozenset({"file_system", "execute", "lab", "run"})   # §4SX D4 + D5
    assert base.args.perfect_it is True          # the live args are untouched






def test_advancer_forwards_the_project_id_to_the_executor():
    """AST pin over `advance_once`: the `coding_executor(...)` call must pass
    `project_id=project_id`. World where it fails: the kwarg is dropped and
    every caller silently falls back to the spec executor."""
    import ast, inspect
    from ghost_agent.core import project_advancer as pa
    tree = ast.parse(inspect.getsource(pa))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "coding_executor"]
    assert calls, "advance_once must call coding_executor"
    for c in calls:
        kws = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kws.get("project_id") == "project_id", kws


