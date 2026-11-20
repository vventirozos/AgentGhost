"""§4SD — the advancer's side of an out-of-time build step: the unattended
step runs under its deadline, an out-of-time result puts the task back to
READY (PENDING), and an owner's batch stops at it instead of re-picking it.

(The coding executor's `generate` file calls these pins grew up with were
removed with the executor, §4UY C18.)
"""
import asyncio
import time
from types import SimpleNamespace

import ghost_agent.core.coding_executor as CE


def test_the_unattended_step_runs_under_its_deadline():
    # world: the step's cap never reaches the leaf (the wrapper is unused)
    import ast
    import pathlib
    tree = ast.parse((pathlib.Path(__file__).resolve().parents[1]
                      / "src/ghost_agent/core/project_advancer.py").read_text(encoding="utf-8"))
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "_advance_unattended_body")
    waits = [c for c in ast.walk(body) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
             and c.func.id == "_wait_for"]
    assert len(waits) == 1
    inner = waits[0].args[0]
    assert isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "_with_leaf_deadline"


def test_the_advancer_puts_an_out_of_time_task_back_to_ready():
    # world (review r11): an out-of-time leaf was marked FAILED, which no
    # unattended tick retries; a timed-out step goes back to READY
    from ghost_agent.core import project_advancer as PA
    calls = []
    store = SimpleNamespace(
        update_task=lambda tid, **kw: calls.append(("update", tid, kw)),
        # §4SU Phase 3 r2: the claim is released through the store (a claimed
        # hold goes back to the hold, anything else to PENDING)
        release_claim=lambda tid, **kw: calls.append(("update", tid, {"status": "PENDING", **kw})),
        log_event=lambda *a, **k: calls.append(("event", a[2])),
    )
    plan = SimpleNamespace(update_status=lambda *a, **k: calls.append(("status", a, k)))
    nxt = SimpleNamespace(id="t1", description="build it")
    import unittest.mock as um
    with um.patch.object(PA, "_increment_budget", lambda *a: None), \
         um.patch.object(PA, "_work_log_step", lambda *a, **k: None), \
         um.patch("ghost_agent.core.project_safety.record_runtime", lambda *a, **k: None):
        res = PA._finalize_coding(SimpleNamespace(), store, plan, "p1", nxt,
                                  CE.CodingResult(False, "out of time: …", files=["a.js"], out_of_time=True),
                                  time.time())
    assert ("update", "t1") == calls[0][:2] and calls[0][2]["status"] == "PENDING"
    assert not any(c[0] == "status" for c in calls)
    assert "back to READY" in res.summary


def test_an_owner_batch_stops_at_an_out_of_time_task(tmp_path):
    # world (review r12): the READY task was picked again at once, 40 times
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.memory.scratchpad import Scratchpad
    from ghost_agent.core import project_advancer as PA
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    ctx = SimpleNamespace(project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                          graph_memory=None, current_project_id=None)
    pid = store.create_project("P", kind="CODING")
    store.add_task(pid, "Implement the engine")
    store.add_task(pid, "Implement the UI")
    calls = []

    async def builder(_c, desc, **_k):
        calls.append(desc)
        return CE.CodingResult(False, "out of time: …", files=["a.js"], out_of_time=True)

    async def cls(_d):
        return "coding"

    async def runner(*_a):
        return "OK"
    res = asyncio.run(PA.advance_many(ctx, pid, max_tasks=None, owner_requested=True, tool_runner=runner,
                                      llm_classifier=cls, coding_executor=builder, stop_on_fail=False))
    assert len(calls) == 1 and res.stop_reason == "out_of_time"
    assert res.advanced[-1]["status"] == "PENDING"
