"""§4LZ lens C — state integrity: behaviour pins."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ── C1: a damaged store is set aside, never overwritten by the next save ──

def test_a_damaged_macro_store_survives_the_next_save(tmp_path):
    from ghost_agent.tools.composed_skills import ComposedSkillRegistry
    f = tmp_path / "composed_skills.json"
    f.write_text('{"m1": {"name": "m1", "steps": []')          # truncated
    reg = ComposedSkillRegistry(tmp_path)
    reg.save()
    kept = list(tmp_path.glob("composed_skills.json.corrupt-*"))
    assert kept and kept[0].read_text().startswith('{"m1"')


def test_a_damaged_skill_registry_survives_the_next_save(tmp_path):
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager
    mgr = AcquiredSkillManager(tmp_path, memory_system=None)
    mgr.save_skill("one", "d", {}, "def run(a):\n    return 1\n")
    mgr.registry_path.write_text('{"one": {"name": "one"')    # truncated
    mgr.save_skill("two", "d", {}, "def run(a):\n    return 2\n")
    kept = list(mgr.registry_path.parent.glob("skills_registry.json.corrupt-*"))
    assert kept and '"one"' in kept[0].read_text()


def test_a_damaged_task_store_survives(tmp_path, monkeypatch):
    import ghost_agent.tools.tasks as T
    f = tmp_path / "tasks.json"
    f.write_text('{"tasks": {"job_a": {')
    monkeypatch.setattr(T, "task_store_path", str(f))
    assert T._load_task_store() == {}
    assert list(tmp_path.glob("tasks.json.corrupt-*"))


# ── C2: concurrent metadata writers never lose each other's updates ──

def _store(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    return ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")


def test_runtime_counters_add_up_under_interleaving(tmp_path, monkeypatch):
    from ghost_agent.core.project_safety import record_runtime
    store = _store(tmp_path)
    pid = store.create_project("App")
    real_get = store.get_project
    fired = {"done": False}

    def get_then_interleave(p):
        got = real_get(p)
        if not fired["done"]:
            fired["done"] = True
            record_runtime(store, pid, 5.0, tool_calls=1)      # lands between read and write
        return got
    monkeypatch.setattr(store, "get_project", get_then_interleave)
    record_runtime(store, pid, 10.0, tool_calls=1)
    meta = real_get(pid)["metadata"]
    assert meta["runtime_used_seconds"] == 15.0 and meta["tool_call_used"] == 2


def test_a_budget_tick_keeps_keys_written_meanwhile(tmp_path):
    from ghost_agent.core.project_advancer import _increment_budget, _stamp_autoadvanced
    store = _store(tmp_path)
    pid = store.create_project("App")
    store.update_project(pid, metadata={"research_index": [{"slug": "a"}]})
    _increment_budget(store, pid)
    _increment_budget(store, pid)
    _stamp_autoadvanced(store, pid)
    meta = store.get_project(pid)["metadata"]
    assert meta["steps_used"] == 2 and meta["research_index"] == [{"slug": "a"}]


# ── C3: the API's delete/archive stops the project's services first ──

@pytest.mark.parametrize("hard", [True, False])
async def test_the_api_delete_stops_services(tmp_path, monkeypatch, hard):
    import ghost_agent.api.projects_routes as A
    import ghost_agent.tools.projects as P
    store = _store(tmp_path)
    pid = store.create_project("Shipped")
    calls = []
    monkeypatch.setattr(P, "_stop_project_services", lambda ctx, p, purge=False: calls.append((p, purge)) or 1)
    ctx = SimpleNamespace(project_store=store, current_project_id=None)
    monkeypatch.setattr(A, "_store", lambda req: store)
    monkeypatch.setattr(A, "_context", lambda req: ctx)
    await A.delete_project(pid, MagicMock(), hard=hard)
    assert calls == [(pid, hard)]
