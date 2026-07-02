"""Tests for delete-then-recreate correction detection: the tombstone
table that survives the hard-delete cascade, title-similarity lookup,
and the create-path instruction that forces a re-plan from the current
message."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
import time
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import tool_manage_projects


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
        request_start_project_id=None,
        last_user_content="",
    )


class TestTombstones:
    def test_hard_delete_writes_tombstone(self, store):
        pid = store.create_project("Chess Game", goal="play chess")
        store.delete_project(pid, hard=True)
        ts = store.find_deleted_similar("Chess Game")
        assert ts is not None
        assert ts["id"] == pid
        assert ts["title"] == "Chess Game"
        assert ts["deleted_at"] > 0

    def test_soft_delete_archive_writes_no_tombstone(self, store):
        pid = store.create_project("Notes App")
        store.delete_project(pid, hard=False)
        assert store.find_deleted_similar("Notes App") is None

    def test_similarity_tolerates_filler_words(self, store):
        pid = store.create_project("Chess Game")
        store.delete_project(pid, hard=True)
        # "project"/"new"/"game" are stopwords for title identity.
        assert store.find_deleted_similar("new chess game project") is not None

    def test_unrelated_title_no_match(self, store):
        pid = store.create_project("Chess Game")
        store.delete_project(pid, hard=True)
        assert store.find_deleted_similar("Snake Game") is None

    def test_window_expiry(self, store):
        pid = store.create_project("Chess Game")
        store.delete_project(pid, hard=True)
        assert store.find_deleted_similar("Chess Game", within_secs=0.0) is None


class TestCreateAfterDelete:
    async def test_recreate_flags_correction(self, context, store):
        context.last_user_content = "create a chess game project"
        res = json.loads(await tool_manage_projects(
            context, action="create", title="Chess Game"))
        pid = res["created"]
        # Simulate the user's rejection: delete via the store (the tool
        # path has an eligibility gate that needs request-scoped state).
        store.delete_project(pid, hard=True)
        context.current_project_id = None
        context.last_user_content = (
            "create a new project where you will build a full chess game, "
            "don't come up with some random AI for this, YOU will play "
            "against me")
        res2 = json.loads(await tool_manage_projects(
            context, action="create", title="Chess Game"))
        assert "CORRECTION CONTEXT" in res2["agent_instruction"]
        assert "DELETED" in res2["agent_instruction"]
        proj = store.get_project(res2["created"])
        assert proj["metadata"]["correction_of"] == pid
        assert any("random AI" in c
                   for c in proj["metadata"]["constraints"])

    async def test_fresh_create_has_no_correction_note(self, context, store):
        context.last_user_content = "create a project for my garden plan"
        res = json.loads(await tool_manage_projects(
            context, action="create", title="Garden Plan"))
        assert "CORRECTION CONTEXT" not in res["agent_instruction"]
        assert "correction_of" not in store.get_project(res["created"])["metadata"]
