"""Tests for cross-project retrieval + RELATED WORK briefing (feature 3B).

Builds real ProjectStore + GraphMemory, links concepts for several projects,
then asserts that find_related_projects ranks by shared-concept overlap and
that the project briefing surfaces a RELATED WORK block — scoped so an
unrelated project shows nothing.
"""

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.graph import GraphMemory
from ghost_agent.core.project_concepts import (
    link_project_concepts, find_related_projects, render_related_work,
)
from ghost_agent.core.prompts import build_project_briefing


@pytest.fixture
def wired(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    gm = GraphMemory(tmp_path / "mem")

    def mk(title, goal):
        pid = store.create_project(title, goal=goal)
        link_project_concepts(gm, store.get_project(pid))
        return pid

    pet = mk("PetAI", "a GRU that models pet behaviour with torch")
    genesis = mk("Genesis", "recurrent networks and torch for world simulation")
    webapp = mk("WebApp", "a fastapi crud service with sqlalchemy")
    return store, gm, {"pet": pet, "genesis": genesis, "webapp": webapp}


def test_related_projects_ranked_by_overlap(wired):
    store, gm, ids = wired
    related = find_related_projects(gm, store, ids["pet"])
    # Genesis shares both torch + recurrent-net; WebApp shares nothing.
    assert related, "expected at least one related project"
    assert related[0]["project_id"] == ids["genesis"]
    assert set(related[0]["shared"]) == {"library:torch", "technique:recurrent-net"}
    assert all(r["project_id"] != ids["webapp"] for r in related)


def test_unrelated_project_has_no_related_work(wired):
    store, gm, ids = wired
    # WebApp shares no concepts with the two ML projects.
    assert find_related_projects(gm, store, ids["webapp"]) == []


def test_briefing_surfaces_related_work(wired):
    store, gm, ids = wired
    b = build_project_briefing(store, ids["pet"], graph_memory=gm)
    assert "RELATED WORK (" in b
    assert "Genesis" in b
    assert "recurrent-net" in b


def test_briefing_without_graph_has_no_related_work(wired):
    store, gm, ids = wired
    # Omitting graph_memory (API/tool callers) → no RELATED WORK, no crash.
    b = build_project_briefing(store, ids["pet"])
    assert "RELATED WORK (" not in b


def test_briefing_related_work_scoped_to_self(wired):
    store, gm, ids = wired
    # WebApp's briefing must not pull in the ML projects.
    b = build_project_briefing(store, ids["webapp"], graph_memory=gm)
    assert "RELATED WORK (" not in b


def test_render_related_work_empty():
    assert render_related_work([]) == ""


def test_find_related_unknown_project(wired):
    store, gm, _ = wired
    assert find_related_projects(gm, store, "nonexistent") == []


def test_find_related_no_graph(wired):
    store, _, ids = wired
    assert find_related_projects(None, store, ids["pet"]) == []
