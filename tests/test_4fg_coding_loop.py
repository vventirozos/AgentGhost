"""§4FG pins: the agentic coding-leaf executor (core/coding_loop.py).

The sub-loop itself (a GhostAgent on an isolated context) is replaced by a
fake in these pins; what is pinned is everything around it — the contract
parser, the workspace diff, the gates, the fresh-attempt-with-witness
policy, the isolation shape, and the dispatch seam every caller inherits.
"""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import coding_loop as cl


def test_parse_leaf_reply_takes_the_last_contract_lines_and_none_means_empty():
    """World where it fails: the parser takes the FIRST VERIFY (the model
    quoting the contract while thinking), or treats 'none' as a command."""
    txt = ("The contract asks me to finish with two lines, like:\n"
           "VERIFY: echo placeholder\nSUMMARY: placeholder\n...work...\n"
           "VERIFY: python -m pytest -q tests/test_fib.py\nSUMMARY: fib + tests, 3 passed")
    assert cl.parse_leaf_reply(txt) == ("python -m pytest -q tests/test_fib.py", "fib + tests, 3 passed")
    assert cl.parse_leaf_reply("done\nVERIFY: none\nSUMMARY: docs only") == ("", "docs only")
    assert cl.parse_leaf_reply("no contract at all") == ("", "")
    assert cl.parse_leaf_reply("VERIFY: `make test`") == ("make test", "")


def test_snapshot_diff_reports_created_and_changed_files_only(tmp_path):
    """World where it fails: files are taken from the model's claims, or a
    deleted file is reported as written."""
    (tmp_path / "a.py").write_text("1")
    (tmp_path / "keep.txt").write_text("k")
    (tmp_path / "gone.txt").write_text("g")
    (tmp_path / ".hidden").write_text("h")
    before = cl.snapshot_workspace(tmp_path)
    (tmp_path / "a.py").write_text("2")
    (tmp_path / "new" ).mkdir()
    (tmp_path / "new" / "b.py").write_text("b")
    (tmp_path / "gone.txt").unlink()
    (tmp_path / ".hidden").write_text("changed")
    after = cl.snapshot_workspace(tmp_path)
    assert cl.diff_snapshots(before, after) == ["a.py", "new/b.py"]
    assert cl.snapshot_workspace(None) == {}


def test_executor_kind_env_beats_metadata_and_defaults_to_spec(monkeypatch):
    class Store:
        def get_project(self, pid):
            return {"metadata": {"executor": "agentic"}} if pid == "p1" else {"metadata": {}}
    ctx = SimpleNamespace(current_project_id="p1", project_store=Store())
    monkeypatch.delenv("GHOST_CODING_EXECUTOR", raising=False)
    assert cl.executor_kind(ctx) == "agentic"
    # The advancer's context carries NO binding (idle ticks, the HTTP route):
    # the explicit project id must decide. World where it fails: the seam
    # reads only the context and an "agentic" project runs the spec executor
    # — measured on the first leaf pilot (spec vs spec).
    unbound = SimpleNamespace(current_project_id=None, project_store=Store())
    assert cl.executor_kind(unbound) == "spec"
    assert cl.executor_kind(unbound, project_id="p1") == "agentic"
    assert cl.executor_kind(unbound, project_id="other") == "spec"
    monkeypatch.setenv("GHOST_CODING_EXECUTOR", "spec")
    assert cl.executor_kind(ctx) == "spec"
    monkeypatch.setenv("GHOST_CODING_EXECUTOR", "agentic")
    assert cl.executor_kind(SimpleNamespace()) == "agentic"
    monkeypatch.delenv("GHOST_CODING_EXECUTOR", raising=False)
    assert cl.executor_kind(SimpleNamespace()) == "spec"


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


def test_success_path_uses_the_workspace_diff_and_runs_the_verify(monkeypatch, tmp_path):
    """World where it fails: files come from the reply, the verify command
    is not run through the tool runner, or the ledger note lies about the
    verify."""
    ctx = _ctx(tmp_path)
    calls = []

    async def fake_turn(context, *, leaf_id, prompt, is_background, **kw):
        (tmp_path / "fib.py").write_text("def fib(n): return n")
        (tmp_path / "tests").mkdir(exist_ok=True)
        (tmp_path / "tests" / "test_fib.py").write_text("def test(): assert True")
        return "built it\nVERIFY: python -m pytest -q tests/test_fib.py\nSUMMARY: fib with a test"
    monkeypatch.setattr(cl, "run_leaf_turn", fake_turn)

    async def no_smoke(tool_runner, written):
        return None
    monkeypatch.setattr("ghost_agent.core.build_gates.smoke_gate", no_smoke)
    res = asyncio.run(cl.build_coding_task_agentic(ctx, "add fib", tool_runner=_runner_ok(calls)))
    assert res.ok is True
    assert res.files == ["fib.py", "tests/test_fib.py"]
    assert res.summary == "fib with a test"
    assert "verify=yes" in res.ledger_note and "files=2" in res.ledger_note
    assert ("execute", {"command": "python -m pytest -q tests/test_fib.py"}) in calls


def test_failure_starts_a_fresh_attempt_carrying_the_witness_then_gives_up(monkeypatch, tmp_path):
    """World where it fails: the second attempt does not see the failing
    evidence (a blind retry), or the loop keeps going past max_attempts."""
    ctx = _ctx(tmp_path)
    prompts = []

    async def fake_turn(context, *, leaf_id, prompt, is_background, **kw):
        prompts.append(prompt)
        (tmp_path / "x.py").write_text(f"v{len(prompts)}")
        return "VERIFY: python -m pytest -q\nSUMMARY: try"
    monkeypatch.setattr(cl, "run_leaf_turn", fake_turn)

    async def failing_runner(name, args):
        return "FAILED tests/test_x.py::test_a - AssertionError\nEXIT CODE: 1"
    res = asyncio.run(cl.build_coding_task_agentic(ctx, "fix x", tool_runner=failing_runner,
                                                   max_attempts=2))
    assert res.ok is False
    assert len(prompts) == 2
    assert "PREVIOUS ATTEMPT FAILED" in prompts[1] and "AssertionError" in prompts[1]
    assert "PREVIOUS ATTEMPT FAILED" not in prompts[0]
    assert "attempt 1" in res.detail and "attempt 2" in res.detail


def test_no_files_and_no_verify_is_a_failed_attempt_not_a_pass(monkeypatch, tmp_path):
    """World where it fails: an attempt that changed nothing and ran nothing
    is marked DONE (the phantom-completion class the spec executor also
    guards against)."""
    ctx = _ctx(tmp_path)

    async def fake_turn(context, *, leaf_id, prompt, is_background, **kw):
        return "I looked around. SUMMARY: nothing to do"
    monkeypatch.setattr(cl, "run_leaf_turn", fake_turn)
    res = asyncio.run(cl.build_coding_task_agentic(ctx, "do x", tool_runner=_runner_ok([]),
                                                   max_attempts=1))
    assert res.ok is False and "no VERIFY" in res.detail


def test_constraint_gate_runs_on_written_files_and_can_fail_the_leaf(monkeypatch, tmp_path):
    ctx = _ctx(tmp_path)

    async def fake_turn(context, *, leaf_id, prompt, is_background, **kw):
        (tmp_path / "engine.py").write_text("minimax()")
        return "VERIFY: none\nSUMMARY: engine"
    monkeypatch.setattr(cl, "run_leaf_turn", fake_turn)
    seen = {}

    async def gate(context, constraints, files, *, is_background=True, **kw):
        seen["files"] = dict(files); seen["constraints"] = list(constraints)
        return False, "wrote a coded engine"
    monkeypatch.setattr("ghost_agent.core.build_gates.constraint_gate", gate)
    res = asyncio.run(cl.build_coding_task_agentic(ctx, "x", tool_runner=_runner_ok([]),
                                                   constraints=["YOU play, not a coded AI"],
                                                   max_attempts=1))
    assert res.ok is False and "constraint gate" in res.detail
    assert seen["files"] == {"engine.py": "minimax()"}


def test_leaf_context_keeps_the_project_and_wraps_memory_read_only():
    """World where it fails: the isolation nulls the project (paths would
    resolve to the sandbox root, the §4EI class) or leaves memory writable
    (a leaf would teach)."""
    from ghost_agent.memory.readonly import ReadOnlySkillMemory
    base = SimpleNamespace(sandbox_dir="/tmp/sb", current_project_id=None,
                           memory_system=object(), skill_memory=object(), graph_memory=None,
                           args=SimpleNamespace(perfect_it=True, smart_memory=0.5, native_tools=False),
                           workspace_model=object(), journal=object(), llm_client=object())
    # the base is UNBOUND (as the advancer's context is); the leaf pins the
    # project explicitly
    iso = cl.build_leaf_context(base, leaf_id="L1", project_id="p1")
    assert iso.current_project_id == "p1" and iso.sandbox_dir == "/tmp/sb"
    assert base.current_project_id is None           # the live context is untouched
    assert isinstance(iso.skill_memory, ReadOnlySkillMemory)
    assert iso.trajectory_task_kind == "leaf" and iso.turn_origin_label == "leaf"
    assert iso.trajectory_extra_static == {"leaf_id": "L1"}
    assert iso.workspace_model is None and iso.journal is None
    assert iso.args.perfect_it is False and iso.args.native_tools is True
    assert iso._subagent_allowed_tools == frozenset({"file_system", "execute"})
    assert base.args.perfect_it is True          # the live args are untouched


def test_dispatch_seam_routes_every_caller_through_build_coding_task(monkeypatch, tmp_path):
    """Executed: with the env set, the legacy entry point returns the agentic
    result without running the spec path. World where it fails: a caller
    binds the agentic executor directly (a second seam) or the seam sits
    after the spec path started."""
    from ghost_agent.core import coding_executor as ce
    ctx = _ctx(tmp_path)
    marker = ce.CodingResult(True, "agentic ran", ["a.py"], "note")

    async def fake_agentic(context, description, **kw):
        assert kw["tool_runner"] is runner and kw["constraints"] == ["c"]
        assert kw["project_id"] == "p9"          # forwarded, not read from the context
        return marker

    async def runner(name, args):
        return "ok"
    monkeypatch.setattr(cl, "build_coding_task_agentic", fake_agentic)
    monkeypatch.setenv("GHOST_CODING_EXECUTOR", "agentic")
    res = asyncio.run(ce.build_coding_task(ctx, "task", tool_runner=runner, constraints=["c"],
                                           project_id="p9"))
    assert res is marker
    import inspect
    src = inspect.getsource(ce.build_coding_task)
    assert src.index("executor_kind(context") < src.index("_generate_build_spec")


def test_metadata_selection_works_through_the_seam_on_an_unbound_context(monkeypatch, tmp_path):
    """The exact world the first pilot lived in: no env flag, the project's
    metadata says agentic, the advancer's context is unbound. World where it
    fails: the seam ignores the forwarded project id and runs the spec path."""
    from ghost_agent.core import coding_executor as ce
    class Store:
        def get_project(self, pid):
            return {"id": pid, "workspace_dir": str(tmp_path),
                    "metadata": {"executor": "agentic"} if pid == "pa" else {}}
    ctx = SimpleNamespace(current_project_id=None, project_store=Store(),
                          llm_client=object(), args=SimpleNamespace(model="m"))
    monkeypatch.delenv("GHOST_CODING_EXECUTOR", raising=False)
    calls = []

    async def fake_agentic(context, description, **kw):
        calls.append(kw.get("project_id"))
        return ce.CodingResult(True, "agentic", [], "")

    async def runner(name, args):
        return "ok"
    monkeypatch.setattr(cl, "build_coding_task_agentic", fake_agentic)
    res = asyncio.run(ce.build_coding_task(ctx, "task", tool_runner=runner, project_id="pa"))
    assert res.ok and calls == ["pa"]


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
