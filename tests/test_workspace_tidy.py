"""Recurring workspace tidy (2026-07-18).

The DONE sweep fires once, on the transition — but verification and
post-completion debugging keep producing screenshots and scaffolding
AFTER it (live case: the game project rolled DONE at 21:41 and had six
unswept screenshots by the next morning), which the operator deleted by
hand. `tidy_project_workspace` is the recurring counterpart: safe on any
status, deletes only categorical debris + unregistered, unreferenced,
age-gated media; never touches source files or the keep-set.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
import time
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.core.workspace_cleanup import (
    tidy_project_workspace, TIDY_MIN_AGE_HOURS,
)


@pytest.fixture
def store(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return ProjectStore(tmp_path / "memory", sandbox_root=sandbox)


@pytest.fixture
def pid(store):
    return store.create_project("Game", kind="CODING", goal="Build the game")


def _ws(store, pid):
    d = store.sandbox_root / "projects" / pid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _age(path, hours):
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


def test_tidy_deletes_old_unregistered_screenshots_keeps_source(store, pid):
    ws = _ws(store, pid)
    (ws / "index.html").write_text("<html>game</html>")
    (ws / "game.js").write_text("loop()")
    (ws / "debug_start.png").write_bytes(b"p" * 100)
    (ws / "game_screenshot.png").write_bytes(b"p" * 100)
    _age(ws / "debug_start.png", 30)
    _age(ws / "game_screenshot.png", 30)

    s = tidy_project_workspace(store, pid)
    assert sorted(s["deleted"]) == ["debug_start.png", "game_screenshot.png"]
    assert (ws / "index.html").exists()
    assert (ws / "game.js").exists()
    assert s["freed_bytes"] == 200


def test_tidy_age_gate_protects_fresh_screenshots(store, pid):
    ws = _ws(store, pid)
    (ws / "fresh.png").write_bytes(b"p")
    _age(ws / "fresh.png", 1)  # 1h old < 24h gate
    s = tidy_project_workspace(store, pid)
    assert s["deleted"] == []
    assert (ws / "fresh.png").exists()


def test_tidy_explicit_call_with_zero_age_gate(store, pid):
    ws = _ws(store, pid)
    (ws / "fresh.png").write_bytes(b"p")
    s = tidy_project_workspace(store, pid, min_age_hours=0.0)
    assert s["deleted"] == ["fresh.png"]


def test_tidy_keeps_registered_and_referenced_media(store, pid):
    ws = _ws(store, pid)
    tid = store.add_task(pid, "Build")
    # registered deliverable
    (ws / "logo.png").write_bytes(b"p" * 10)
    store.add_artifact(tid, "file", "logo.png")
    # unregistered but referenced by source → asset, not screenshot
    (ws / "sprites.png").write_bytes(b"p" * 10)
    (ws / "index.html").write_text('<img src="sprites.png">')
    # unregistered, unreferenced, old → scratch
    (ws / "shot.png").write_bytes(b"p" * 10)
    for f in ("logo.png", "sprites.png", "shot.png"):
        _age(ws / f, 48)

    s = tidy_project_workspace(store, pid)
    assert s["deleted"] == ["shot.png"]
    assert (ws / "logo.png").exists()
    assert (ws / "sprites.png").exists()
    assert s["kept_referenced"] == ["sprites.png"]


def test_tidy_removes_old_debris(store, pid):
    ws = _ws(store, pid)
    cache = ws / "__pycache__"
    cache.mkdir()
    (cache / "m.pyc").write_bytes(b"c")
    (ws / ".browser_runner.py").write_text("x")
    _age(cache / "m.pyc", 30)
    _age(ws / ".browser_runner.py", 30)
    s = tidy_project_workspace(store, pid)
    assert "__pycache__/m.pyc" in s["deleted"]
    assert ".browser_runner.py" in s["deleted"]
    assert "__pycache__" in s["dirs_removed"]


def test_tidy_age_gate_applies_to_debris_too(store, pid):
    """H8 regression (2026-07-20): the age gate covered only media, so a
    recurring tidy on an ACTIVE project deleted brand-new debris-shaped
    files out from under in-flight work. Fresh debris must survive the
    default tidy exactly like fresh screenshots do."""
    ws = _ws(store, pid)
    cache = ws / "__pycache__"
    cache.mkdir()
    (cache / "m.pyc").write_bytes(b"c")
    (ws / ".hidden_scratch").write_text("in-flight tool state")
    s = tidy_project_workspace(store, pid)
    assert s["deleted"] == []
    assert (cache / "m.pyc").exists()
    assert (ws / ".hidden_scratch").exists()


def test_tidy_never_touches_git_tree_or_config_dotfiles(store, pid):
    """H8 regression (2026-07-20): tidy deleted `.git/` contents and
    dot-config files (`.eslintrc.json`, `.env.local`) on ACTIVE projects.
    The `.git/` subtree is excluded outright — even past the age gate,
    even at min_age_hours=0 — and keep-dotfile VARIANTS are not debris."""
    ws = _ws(store, pid)
    git = ws / ".git"
    (git / "objects" / "ab").mkdir(parents=True)
    (git / "refs" / "tags").mkdir(parents=True)
    (git / "config").write_text("[core]")
    (git / "objects" / "ab" / "cdef").write_bytes(b"blob")
    (ws / ".eslintrc.json").write_text("{}")
    (ws / ".env.local").write_text("KEY=1")
    for rel in (".git/config", ".git/objects/ab/cdef"):
        _age(ws / rel, 999)

    s = tidy_project_workspace(store, pid, min_age_hours=0.0)

    assert s["deleted"] == []
    assert (git / "config").exists()
    assert (git / "objects" / "ab" / "cdef").exists()
    assert (git / "refs" / "tags").is_dir()  # empty git dir never pruned
    assert (ws / ".eslintrc.json").exists()
    assert (ws / ".env.local").exists()


def test_tidy_path_escape_id_deletes_nothing(store):
    """C1 regression (2026-07-20): a `..`-bearing project_id resolved the
    walk root OUTSIDE the sandbox (reproduced with `../..` → sandbox
    parent). Defense-in-depth behind the tool boundary: tidy must refuse
    to walk and delete nothing."""
    parent = store.sandbox_root.parent
    decoy = parent / "decoy_scratch.tmp"  # debris-shaped, would be swept
    decoy.write_text("outside the sandbox")
    _age(decoy, 999)

    s = tidy_project_workspace(store, "../..", min_age_hours=0.0)

    assert s["status"] == "skipped: path escape"
    assert s["deleted"] == []
    assert decoy.exists()


def test_tidy_never_deletes_source_even_when_unregistered_and_old(store, pid):
    ws = _ws(store, pid)
    (ws / "helper_patch.py").write_text("print('debug helper')")
    _age(ws / "helper_patch.py", 999)
    s = tidy_project_workspace(store, pid)
    assert s["deleted"] == []
    assert (ws / "helper_patch.py").exists()


def test_tidy_dry_run_touches_nothing(store, pid):
    ws = _ws(store, pid)
    (ws / "shot.png").write_bytes(b"p")
    _age(ws / "shot.png", 48)
    s = tidy_project_workspace(store, pid, dry_run=True)
    assert s["deleted"] == ["shot.png"]
    assert (ws / "shot.png").exists()


def test_tidy_logs_one_event_with_counts(store, pid):
    ws = _ws(store, pid)
    (ws / "a.png").write_bytes(b"p")
    (ws / "b.png").write_bytes(b"p")
    for f in ("a.png", "b.png"):
        _age(ws / f, 48)
    tidy_project_workspace(store, pid)
    ev = store.list_events(pid, limit=5, event_type="workspace_tidy")
    assert len(ev) == 1
    assert ev[0]["payload"]["deleted_count"] == 2


def test_tidy_works_on_done_projects(store, pid):
    ws = _ws(store, pid)
    store.update_project(pid, status="DONE")
    (ws / "post_done_shot.png").write_bytes(b"p")
    _age(ws / "post_done_shot.png", 48)
    s = tidy_project_workspace(store, pid)
    assert s["deleted"] == ["post_done_shot.png"]


def test_tidy_missing_dir_is_skipped(store, pid):
    s = tidy_project_workspace(store, pid)
    # workspace dir may have been created by the store on project create;
    # remove it to model a never-touched project
    import shutil
    ws = store.sandbox_root / "projects" / pid
    if ws.exists():
        shutil.rmtree(ws)
    s = tidy_project_workspace(store, pid)
    assert s["status"].startswith("skipped")
    assert s["deleted"] == []


# ---------------------------------------------------------------- wiring

@pytest.mark.asyncio
async def test_cleanup_action_runs_tidy_without_age_gate(store, pid, tmp_path):
    from ghost_agent.tools.projects import tool_manage_projects
    from ghost_agent.memory.scratchpad import Scratchpad
    ws = _ws(store, pid)
    (ws / "fresh_shot.png").write_bytes(b"p" * 7)
    context = SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None, contradiction_log=None,
        current_project_id=pid, llm_client=None,
    )
    out = json.loads(await tool_manage_projects(context, action="cleanup"))
    assert out["deleted"] == ["fresh_shot.png"]
    assert not (ws / "fresh_shot.png").exists()


def test_idle_phase_wired_into_watchdog():
    import inspect
    import ghost_agent.core.agent as agent_mod
    src = inspect.getsource(agent_mod.GhostAgent)
    assert "tidy_project_workspace" in src
    assert "_WORKSPACE_TIDY_COOLDOWN" in src
    assert "_last_workspace_tidy_at" in src
