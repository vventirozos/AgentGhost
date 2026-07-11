"""Regressions from the 2026-07-11 ghost-agent.log audit.

Three bugs, all observed live in one session, all of which DEADLOCKED real
project work:

1. The constraint judgment gate audited the WHOLE project on every task
   close, so one violating file written by an early task blocked the DONE
   transition of every OTHER task, forever.
2. `add_task` never reopened a DONE project, so tasks added to a finished
   project were unreachable by autoadvance ("all tasks complete" while N sat
   PENDING).
3. `manage_projects` was missing from READWRITE_LOOP_TOOLS, so a no-progress
   READ loop force-stopped the turn into a text-only response — barring the
   pending write and (twice) getting the whole reply eaten by the stream
   scrub.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.core.strikes import (
    READWRITE_LOOP_TOOLS, is_readwrite_loop_exempt,
)
from ghost_agent.tools.projects import _files_for_task


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


# ══════════════════════════════════════════════════════════════════════
# 1. Constraint gate must audit only THIS task's files
# ══════════════════════════════════════════════════════════════════════

class TestConstraintGateScope:
    def _project_with_files(self, store, tmp_path):
        pid = store.create_project("Meta", kind="GENERAL")
        base = tmp_path / "sb" / "projects" / pid
        base.mkdir(parents=True, exist_ok=True)
        (base / "context_boundary.md").write_text(
            'The system prompt says "be concise."')   # task 1's violation
        (base / "output_generation.md").write_text(
            "A clean analysis with no verbatim quotes.")
        t1 = store.add_task(pid, "Context Boundary")
        t2 = store.add_task(pid, "Output Generation")
        store.register_file_artifact(t1, "context_boundary.md")
        store.register_file_artifact(t2, "output_generation.md")
        return pid, t1, t2

    def test_only_the_tasks_own_files_are_audited(self, store, tmp_path):
        """THE deadlock: closing task 2 must NOT be judged on task 1's file."""
        pid, t1, t2 = self._project_with_files(store, tmp_path)
        files = _files_for_task(store, pid, t2)
        assert set(files) == {"output_generation.md"}
        assert "context_boundary.md" not in files      # the bug
        assert "be concise" not in "".join(files.values())

    def test_offending_task_still_sees_its_own_file(self, store, tmp_path):
        pid, t1, t2 = self._project_with_files(store, tmp_path)
        files = _files_for_task(store, pid, t1)
        assert set(files) == {"context_boundary.md"}
        assert "be concise" in files["context_boundary.md"]

    def test_deliverables_arg_is_used(self, store, tmp_path):
        pid = store.create_project("P")
        base = tmp_path / "sb" / "projects" / pid
        base.mkdir(parents=True, exist_ok=True)
        (base / "fresh.md").write_text("brand new content")
        tid = store.add_task(pid, "T")   # nothing registered yet
        files = _files_for_task(store, pid, tid, deliverables=["fresh.md"])
        assert files == {"fresh.md": "brand new content"}

    def test_no_attributable_files_yields_empty(self, store, tmp_path):
        """Empty ⇒ the caller SKIPS the gate. Auditing someone else's
        artifact is precisely the bug."""
        pid = store.create_project("P")
        tid = store.add_task(pid, "T")
        assert _files_for_task(store, pid, tid) == {}

    def test_path_traversal_contained(self, store, tmp_path):
        pid = store.create_project("P")
        tid = store.add_task(pid, "T")
        secret = tmp_path / "sb" / "outside.md"
        secret.parent.mkdir(parents=True, exist_ok=True)
        secret.write_text("SECRET")
        files = _files_for_task(store, pid, tid,
                                deliverables=["../../outside.md"])
        assert files == {}

    def test_missing_file_skipped_not_raised(self, store, tmp_path):
        pid = store.create_project("P")
        tid = store.add_task(pid, "T")
        assert _files_for_task(store, pid, tid,
                               deliverables=["ghost.md"]) == {}

    def test_call_site_no_longer_gathers_whole_project(self):
        from pathlib import Path as _P
        src = (_P(__file__).resolve().parents[1] / "src" / "ghost_agent"
               / "tools" / "projects.py").read_text()
        # The judgment gate must not CALL the project-wide collector (the
        # name still appears in the comment explaining why it was removed).
        assert "_gather_project_files(store" not in src
        assert "import _gather_project_files" not in src
        assert "_files_for_task(store, project_id, tid" in src


# ══════════════════════════════════════════════════════════════════════
# 2. Adding a task to a DONE project must REOPEN it
# ══════════════════════════════════════════════════════════════════════

class TestReopenOnAddTask:
    def test_done_project_reopens(self, store):
        pid = store.create_project("Meta")
        t1 = store.add_task(pid, "only task")
        store.update_task(t1, status="DONE")
        # (Project may or may not auto-roll to DONE; force the end state.)
        store.update_project(pid, status="DONE")
        assert store.get_project(pid)["status"] == "DONE"

        store.add_task(pid, "a new task the user just asked for")
        assert store.get_project(pid)["status"] == "ACTIVE", (
            "adding work to a finished project must un-finish it — else "
            "advance_once refuses ('project is DONE, not ACTIVE') and the "
            "new task is unreachable forever")

    def test_reopen_is_logged(self, store):
        pid = store.create_project("Meta")
        store.update_project(pid, status="DONE")
        store.add_task(pid, "new work")
        kinds = [e.get("type") for e in store.list_events(pid, limit=20)]
        assert "project_reopened" in kinds

    def test_active_project_untouched(self, store):
        pid = store.create_project("Meta")
        store.add_task(pid, "t")
        assert store.get_project(pid)["status"] == "ACTIVE"
        events = [e.get("type") for e in store.list_events(pid, limit=20)]
        assert "project_reopened" not in events   # no spurious event

    def test_archived_project_not_resurrected(self, store):
        """ARCHIVED is a deliberate end-state (cleanup already swept it)."""
        pid = store.create_project("Old")
        store.update_project(pid, status="ARCHIVED")
        store.add_task(pid, "stray task")
        assert store.get_project(pid)["status"] == "ARCHIVED"

    def test_adding_a_done_task_does_not_reopen(self, store):
        # Back-filling an already-complete task is not new work.
        pid = store.create_project("Meta")
        store.update_project(pid, status="DONE")
        store.add_task(pid, "historical record", status="DONE")
        assert store.get_project(pid)["status"] == "DONE"

    def test_reopened_project_is_advanceable(self, store):
        """The end-to-end property: after reopening, advance_once's own
        ACTIVE gate passes."""
        pid = store.create_project("Meta")
        store.update_project(pid, status="DONE")
        store.add_task(pid, "Input Processing Analysis")
        proj = store.get_project(pid)
        assert proj["status"] == "ACTIVE"          # the gate advance_once reads
        pending = [t for t in store.list_tasks(pid)
                   if t["status"] == "PENDING"]
        assert len(pending) == 1


# ══════════════════════════════════════════════════════════════════════
# 3. manage_projects must get the SOFT steer, not a force-stop
# ══════════════════════════════════════════════════════════════════════

class TestManageProjectsLoopExempt:
    def test_manage_projects_is_readwrite_exempt(self):
        assert "manage_projects" in READWRITE_LOOP_TOOLS
        assert is_readwrite_loop_exempt("manage_projects") is True

    def test_peer_readwrite_tools_still_exempt(self):
        for t in ("file_system", "manage_tasks", "manage_composed_skills",
                  "knowledge_base", "update_profile"):
            assert is_readwrite_loop_exempt(t) is True

    def test_pure_read_tools_still_force_stop(self):
        # These have no write action through the same tool — the hard
        # force-stop is correct for them.
        for t in ("web_search", "recall", "browser", "introspect"):
            assert is_readwrite_loop_exempt(t) is False
