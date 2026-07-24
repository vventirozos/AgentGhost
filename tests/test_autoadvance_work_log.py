"""Autoadvance → work_log mirror (2026-07-24).

Until now only INTERACTIVE turns wrote work_log (the finalize chain), so a
project mostly built by autoadvance had a near-empty journal — live:
2 work_log rows vs 8 autoadvance_step events on 6a471d630e81 — and the
RECENT WORK LOG briefing + file_history couldn't see what the idle loop did.
`_finalize_coding` (DONE and FAILED arms) now mirrors each step into
work_log, which also makes autoadvanced files visible to `file_history`.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.core.planning import ProjectPlan
from ghost_agent.core.coding_executor import CodingResult
from ghost_agent.core.project_advancer import _finalize_coding


@pytest.fixture
def store(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return ProjectStore(tmp_path / "memory", sandbox_root=sandbox)


def _setup(store):
    pid = store.create_project("App", kind="CODING", goal="build")
    tid = store.add_task(pid, "server.js: build the API server")
    plan = ProjectPlan(store, pid)
    nxt = plan.next_ready_leaf()
    ctx = SimpleNamespace(project_store=store)
    return pid, plan, nxt, ctx


def test_coding_success_mirrors_work_log_and_file_history(store):
    import time
    pid, plan, nxt, ctx = _setup(store)
    cres = CodingResult(ok=True, summary="Express API with /entries CRUD",
                        files=["server.js"], ledger_note="server.js hosts API")
    res = _finalize_coding(ctx, store, plan, pid, nxt, cres, time.time())
    assert res.ok
    logs = store.recent_work_logs(pid)
    assert len(logs) == 1
    p = logs[0]["payload"]
    assert p["outcome"] == "completed"
    assert p["files"] == ["server.js"]
    assert p["request"].startswith("[autoadvance]")
    assert "Express API" in p["note"]
    # file_history sees the autoadvanced file (the P2 goal).
    hist = store.file_history(pid, "server.js")
    assert any(h["type"] == "work_log" for h in hist)
    # Manifest was seeded from the build summary too (P1 feeder).
    assert "server.js" in store.get_file_manifest(pid)


def test_coding_failure_mirrors_work_log(store):
    import time
    pid, plan, nxt, ctx = _setup(store)
    cres = CodingResult(ok=False, summary="SyntaxError at line 3", files=[])
    _finalize_coding(ctx, store, plan, pid, nxt, cres, time.time())
    logs = store.recent_work_logs(pid)
    assert len(logs) == 1
    p = logs[0]["payload"]
    assert p["outcome"] == "had_failures"
    assert "FAILED" in p["note"]
