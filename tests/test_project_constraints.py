"""Tests for the explicit-constraint plumbing through the project stack:
store column round-trip + migration, plan hydration, tool capture at
create/decompose, the constraint DONE-gate on task_update, briefing
render, and the agent's verifier constraint note."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
import sqlite3
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.core.planning import ProjectPlan, TaskStatus
from ghost_agent.core.prompts import build_project_briefing
from ghost_agent.tools.projects import tool_manage_projects

CHESS_MSG = (
    "create a new project where you will build a full chess game that we "
    "can play against each other, don't come up with some random AI for "
    "this, it's gonna be a a turn by turn game where YOU will play "
    "against me."
)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(tmp_path, store):
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None,
        workspace_model=None,
        current_project_id=None,
        last_user_content="",
    )


# ---------------------------------------------------------------- store

class TestStoreConstraints:
    def test_add_task_roundtrip(self, store):
        pid = store.create_project("Data Pipeline")
        tid = store.add_task(pid, "write the loader",
                             constraints=["don't use pandas"])
        task = store.get_task(tid)
        assert task["constraints"] == ["don't use pandas"]

    def test_default_empty(self, store):
        pid = store.create_project("P")
        tid = store.add_task(pid, "t")
        assert store.get_task(tid)["constraints"] == []

    def test_update_task_constraints(self, store):
        pid = store.create_project("P")
        tid = store.add_task(pid, "t")
        store.update_task(tid, constraints=["no regex"])
        assert store.get_task(tid)["constraints"] == ["no regex"]

    def test_migration_adds_column_to_legacy_db(self, tmp_path):
        # Build a DB lacking constraints_json, then reopen through the store.
        legacy_dir = tmp_path / "legacy"
        legacy_dir.mkdir()
        db = legacy_dir / "projects.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE tasks (id TEXT PRIMARY KEY, project_id TEXT, "
            "parent_id TEXT, description TEXT, status TEXT, "
            "dependency_type TEXT, alternatives_json TEXT, "
            "postconditions_json TEXT, depends_on_json TEXT, "
            "result_summary TEXT, failure_reason TEXT, "
            "revision_count INTEGER, actual_tool_used TEXT, "
            "estimated_cost REAL, actual_cost REAL, depth INTEGER, "
            "position INTEGER, created_at REAL, updated_at REAL)")
        conn.commit()
        conn.close()
        store = ProjectStore(legacy_dir)
        cols = {r[1] for r in sqlite3.connect(db).execute(
            "PRAGMA table_info(tasks)").fetchall()}
        assert "constraints_json" in cols

    def test_plan_hydrates_constraints(self, store):
        pid = store.create_project("P")
        plan = ProjectPlan(store, pid)
        tid = plan.add_task("t", constraints=["YOU will play against me"])
        fresh = ProjectPlan(store, pid)
        assert fresh.tree.nodes[tid].constraints == ["YOU will play against me"]


# ---------------------------------------------------------------- tool

class TestToolCapture:
    async def test_create_captures_message_constraints(self, context, store):
        context.last_user_content = CHESS_MSG
        res = json.loads(await tool_manage_projects(
            context, action="create", title="Chess Game",
            subtasks=["Build the chess board state protocol"]))
        pid = res["created"]
        proj = store.get_project(pid)
        cons = proj["metadata"]["constraints"]
        assert any("random AI" in c for c in cons)
        assert any("YOU will play against me" in c for c in cons)
        # ... and the constraints ride on the created task(s) too.
        tasks = store.list_tasks(pid)
        assert tasks and all(t["constraints"] for t in tasks)
        assert "EXPLICIT USER CONSTRAINTS" in res["agent_instruction"]

    async def test_create_without_constraints_is_clean(self, context, store):
        context.last_user_content = "start a project to organize my notes"
        res = json.loads(await tool_manage_projects(
            context, action="create", title="Notes"))
        proj = store.get_project(res["created"])
        assert "constraints" not in proj["metadata"]

    async def test_decompose_stamps_constraints(self, context, store):
        context.last_user_content = "start a project for the ETL rewrite"
        res = json.loads(await tool_manage_projects(
            context, action="create", title="ETL Rewrite"))
        pid = res["created"]
        context.last_user_content = "plan it out, but don't touch the legacy schema"
        res2 = json.loads(await tool_manage_projects(
            context, action="task_decompose", project_id=pid,
            subtasks=["extract module", "load module"]))
        for tid in res2["created"]:
            assert any("legacy schema" in c
                       for c in store.get_task(tid)["constraints"])

    async def test_reuse_path_merges_fresh_constraints(self, context, store):
        context.last_user_content = "start the chess game project"
        res = json.loads(await tool_manage_projects(
            context, action="create", title="Chess Game"))
        pid = res["created"]
        # Same title again, now WITH a correction in the message.
        context.last_user_content = (
            "create the chess game project, don't come up with some random "
            "AI, YOU will play against me")
        res2 = json.loads(await tool_manage_projects(
            context, action="create", title="Chess Game"))
        assert res2["refused"] is True
        assert "CORRECTION DETECTED" in res2["agent_instruction"]
        merged = store.get_project(pid)["metadata"]["constraints"]
        assert any("random" in c for c in merged)


# ---------------------------------------------------------------- gate

class TestConstraintDoneGate:
    async def test_done_without_evidence_is_gated(self, context, store):
        pid = store.create_project(
            "Data Pipeline", metadata={"constraints": ["don't use pandas"]})
        tid = store.add_task(pid, "write the summary text")
        res = json.loads(await tool_manage_projects(
            context, action="task_update", project_id=pid,
            task_id=tid, status="DONE"))
        assert res["gated_constraints"] == [tid]
        assert "don't use pandas" in res["agent_instruction_constraints"]
        assert store.get_task(tid)["status"] != "DONE"

    async def test_done_with_evidence_passes(self, context, store):
        pid = store.create_project(
            "Data Pipeline", metadata={"constraints": ["don't use pandas"]})
        tid = store.add_task(pid, "write the summary text")
        res = json.loads(await tool_manage_projects(
            context, action="task_update", project_id=pid,
            task_id=tid, status="DONE",
            result="loader written with csv stdlib only — no pandas import"))
        assert not res.get("gated_constraints")
        assert store.get_task(tid)["status"] == "DONE"

    async def test_task_level_constraints_gate_too(self, context, store):
        pid = store.create_project("P")
        tid = store.add_task(pid, "write the notes file",
                             constraints=["no external services"])
        res = json.loads(await tool_manage_projects(
            context, action="task_update", project_id=pid,
            task_id=tid, status="DONE"))
        assert res["gated_constraints"] == [tid]

    async def test_unconstrained_task_not_gated(self, context, store):
        pid = store.create_project("P")
        tid = store.add_task(pid, "write the notes file")
        res = json.loads(await tool_manage_projects(
            context, action="task_update", project_id=pid,
            task_id=tid, status="DONE"))
        assert not res.get("gated_constraints")
        assert store.get_task(tid)["status"] == "DONE"


# ---------------------------------------------------------------- briefing

class TestBriefingRender:
    def test_constraints_rendered_every_briefing(self, store):
        pid = store.create_project(
            "Chess Game",
            metadata={"constraints": ["YOU will play against me"]})
        briefing = build_project_briefing(store, pid)
        assert "EXPLICIT USER CONSTRAINTS" in briefing
        assert "! YOU will play against me" in briefing

    def test_no_constraints_no_section(self, store):
        pid = store.create_project("P")
        assert "EXPLICIT USER CONSTRAINTS" not in build_project_briefing(store, pid)


# ---------------------------------------------------------------- agent note

class TestActiveConstraintNote:
    def test_note_from_active_project(self, store):
        from ghost_agent.core.agent import GhostAgent
        agent = object.__new__(GhostAgent)
        pid = store.create_project(
            "Chess Game", metadata={"constraints": ["no random AI"]})
        agent.context = SimpleNamespace(project_store=store,
                                        current_project_id=pid)
        note = agent._active_constraint_note()
        assert "no random AI" in note
        assert note.endswith("USER REQUEST: ")

    def test_empty_without_project(self):
        from ghost_agent.core.agent import GhostAgent
        agent = object.__new__(GhostAgent)
        agent.context = SimpleNamespace(project_store=None,
                                        current_project_id=None)
        assert agent._active_constraint_note() == ""
