"""Project/task ids must resolve the same regardless of how they were
transmitted. IDs are generated as lowercase hex, but an LLM echoing one
back in a tool call mangles the case of opaque hex tokens
(9b5bd5cd812b -> 9B5Bd5Cd812B), which used to surface as
"project not found". The store + tool now canonicalise every id."""

from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore, _canon_id
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import tool_manage_projects


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _mangle(pid: str) -> str:
    """Mimic the LLM uppercasing alternate chars of a hex id."""
    return "".join(c.upper() if i % 2 else c for i, c in enumerate(pid))


# ── _canon_id ────────────────────────────────────────────────────────

def test_canon_id():
    assert _canon_id("9B5Bd5Cd812B") == "9b5bd5cd812b"
    assert _canon_id("  AbC123  ") == "abc123"
    assert _canon_id(None) == ""
    assert _canon_id("abc123") == "abc123"  # idempotent


def test_generated_ids_are_canonical(store):
    pid = store.create_project("P")
    assert pid == _canon_id(pid)
    assert len(pid) == 12


# ── store lookups are case / whitespace insensitive ──────────────────

def test_get_project_resolves_mangled_case(store):
    pid = store.create_project("My Project")
    assert store.get_project(_mangle(pid)) is not None
    assert store.get_project(pid.upper())["id"] == pid
    assert store.get_project(f"  {pid}  ")["id"] == pid


def test_update_and_delete_project_mangled(store):
    pid = store.create_project("P")
    assert store.update_project(_mangle(pid), title="X") is True
    assert store.get_project(pid)["title"] == "X"
    assert store.delete_project(pid.upper()) is True


def test_task_lookups_mangled(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "do it")
    assert store.get_task(_mangle(tid))["id"] == tid
    assert len(store.list_tasks(_mangle(pid))) >= 1
    store.log_event(_mangle(pid), tid, "autoadvance_step", {})
    assert len(store.list_events(pid.upper())) >= 1


# ── tool boundary (the exact reported failure path) ──────────────────

async def test_tool_switch_resolves_mangled_id(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    ctx = SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        current_project_id=None,
    )
    pid = store.create_project("My Project")
    out = await tool_manage_projects(ctx, action="switch", project_id=_mangle(pid))
    assert "not found" not in out.lower()
    # current_project_id is stored in canonical (lowercase) form
    assert ctx.current_project_id == pid
