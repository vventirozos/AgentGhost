"""Hard-deleting a project removes its knowledge-graph edges.

Live req 2cb40b10 (2026-10-06): the user deleted two projects and re-asked
the same question. Recall still returned `user RESUMES project 7b62e5e533d1`
and `project 7b62e5e533d1 TESTED 4-layer recursive cascade`, so the model
read "the user runs this experiment as a project" and created a third one.
`GraphMemory.forget_project` + the tool's delete branch now remove every edge
that names the deleted project (archived first, like every graph delete).
"""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from ghost_agent.memory.graph import GraphMemory
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.readonly import ReadOnlyGraphMemory
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import tool_manage_projects


def _rows(gm):
    with sqlite3.connect(gm.db_path) as conn:
        return {(s, p, o) for s, p, o in
                conn.execute("SELECT subject, predicate, object FROM triplets")}


def _seed_live_shape(gm, pid="7b62e5e533d1"):
    """The edges the live graph held for project 7b62e5e533d1, both the
    tool's `project:<id>` shape and the extractor's `project <id>` shape."""
    gm.add_triplets([
        {"subject": "project", "predicate": "HAS_ID", "object": pid},
        {"subject": "user", "predicate": "RESUMES", "object": f"project {pid}"},
        {"subject": f"project {pid}", "predicate": "TESTED", "object": "4-layer recursive cascade"},
        {"subject": f"project {pid}", "predicate": "USES_TECHNIQUE", "object": "technique:bayesian"},
        {"subject": f"project:{pid}", "predicate": "HAS_TITLE", "object": "ai self awareness exploration"},
        {"subject": f"project:{pid}", "predicate": "HAS_TASK", "object": "task:aaaa11112222"},
        {"subject": "task:aaaa11112222", "predicate": "HAS_DESCRIPTION", "object": "layer 1"},
        {"subject": "project", "predicate": "HAS_TITLE", "object": "AI self awareness exploration"},
    ], raw=True)


# Unrelated facts that must survive every delete below.
_KEEP = [
    {"subject": "user", "predicate": "EXPLORES", "object": "self-awareness"},
    {"subject": "project:e4e240b630f6", "predicate": "HAS_TITLE", "object": "webos"},
    {"subject": "project:e4e240b630f6", "predicate": "USES_TECHNIQUE", "object": "technique:bayesian"},
    {"subject": "project", "predicate": "HAS_ID", "object": "e4e240b630f6"},
    # a different id that CONTAINS the deleted one is not it
    {"subject": "project 7b62e5e533d1ff", "predicate": "HAS_GOAL", "object": "x"},
]


@pytest.fixture
def gm(tmp_path):
    g = GraphMemory(tmp_path)
    _seed_live_shape(g)
    g.add_triplets(_KEEP, raw=True)
    return g


def test_forget_project_removes_every_edge_naming_the_project(gm):
    n = gm.forget_project("7b62e5e533d1", title="AI self awareness exploration")
    left = _rows(gm)
    assert n == 8
    assert not any("7b62e5e533d1" == s.split()[-1].split(":")[-1] or o == "7b62e5e533d1"
                   for s, _, o in left)
    assert ("task:aaaa11112222", "HAS_DESCRIPTION", "layer 1") not in left
    assert not any(p == "HAS_TITLE" and s == "project" for s, p, _ in left)
    # positive twin: the unrelated facts, the other project and the SHARED
    # concept node's other edge are all intact
    for t in _KEEP:
        assert (t["subject"], t["predicate"], t["object"]) in left
    # and the in-memory mirror agrees with sqlite
    assert not gm.nx_graph.has_node("project 7b62e5e533d1")
    assert gm.nx_graph.has_node("technique:bayesian")


def test_forget_project_archives_before_delete(gm, tmp_path):
    gm.forget_project("7b62e5e533d1")
    arch = (tmp_path / GraphMemory._ARCHIVE_FILENAME).read_text().splitlines()
    recs = [json.loads(l) for l in arch]
    assert {r["reason"] for r in recs} == {"forget_project"}
    assert {"subject": "user", "predicate": "RESUMES", "object": "project 7b62e5e533d1"}.items() \
        <= next(r for r in recs if r["predicate"] == "RESUMES").items()


def test_title_edge_kept_when_another_project_shares_the_title(gm):
    gm.forget_project("7b62e5e533d1", title="AI self awareness exploration",
                      forget_title=False)
    assert ("project", "HAS_TITLE", "ai self awareness exploration") in _rows(gm) \
        or ("project", "HAS_TITLE", "AI self awareness exploration") in _rows(gm)


def test_forget_project_removes_expired_rows_too(gm):
    with sqlite3.connect(gm.db_path) as conn:
        conn.execute("INSERT INTO triplets (subject, predicate, object, valid_until) "
                     "VALUES ('project 7b62e5e533d1', 'HAS_STATUS', 'archived', 1.0)")
    gm.forget_project("7b62e5e533d1")
    assert not any("7b62e5e533d1" in s for s, _, _ in _rows(gm)
                   if not s.endswith("ff"))


@pytest.mark.parametrize("bad", ["", "ab", "project", "7b62 e5e533d1", None])
def test_forget_project_refuses_non_id_input(gm, bad):
    before = _rows(gm)
    assert gm.forget_project(bad) == 0
    assert _rows(gm) == before


def test_readonly_proxy_blocks_forget_project(gm):
    # explicit: the proxy's generic forget_* fallback would pass this alone
    assert "forget_project" in ReadOnlyGraphMemory._MUTATORS
    ro = ReadOnlyGraphMemory(gm)
    before = _rows(gm)
    ro.forget_project("7b62e5e533d1")
    assert _rows(gm) == before


# ---------------------------------------------------------------- tool wiring

@pytest.fixture
def context(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=GraphMemory(tmp_path),
        current_project_id=None,
        request_start_project_id=None,
        last_user_content="",
        conversation_key="conv-1",
    )


async def _create(context, title):
    res = json.loads(await tool_manage_projects(
        context, action="create", title=title, kind="GENERAL", goal="g"))
    return res["created"]


async def test_tool_delete_unlinks_the_project_from_the_graph(context):
    gm = context.graph_memory
    pid = await _create(context, "4-Layer Recursive Cascade Test")
    await tool_manage_projects(context, action="task_decompose",
                               subtasks=["Layer 1", "Layer 2"])
    gm.add_triplets([{"subject": "user", "predicate": "RESUMES",
                      "object": f"project {pid}"}], raw=True)
    assert any(pid in s for s, _, _ in _rows(gm))      # the tool linked it

    res = json.loads(await tool_manage_projects(context, action="delete",
                                                project_id=pid))
    assert res.get("deleted") is True
    left = _rows(gm)
    assert not any(pid in s or pid in o for s, _, o in left)
    assert not any(s.startswith("task:") for s, _, _ in left)


async def test_tool_delete_keeps_the_twin_projects_title_edge(context):
    """Two projects with the same title (live: 9c69d96a88f0 and
    e6f85dfb94f8): deleting one keeps `project HAS_TITLE <title>`."""
    gm = context.graph_memory
    a = await _create(context, "Cascade Test")
    store = context.project_store
    b = store.create_project(title="Cascade Test", kind="GENERAL", goal="g")
    gm.add_triplets([{"subject": "project", "predicate": "HAS_TITLE",
                      "object": "cascade test"}], raw=True)
    await tool_manage_projects(context, action="delete", project_id=a)
    left = _rows(gm)
    assert ("project", "HAS_TITLE", "cascade test") in left
    assert not any(a in s for s, _, _ in left)
    assert store.get_project(b)


async def test_archive_does_not_unlink(context):
    """Archive is resumable — its graph edges stay."""
    pid = await _create(context, "Keep Me")
    await tool_manage_projects(context, action="archive", project_id=pid)
    assert any(f"project:{pid}" == s for s, _, _ in _rows(context.graph_memory))


async def test_tool_delete_succeeds_without_graph(context):
    context.graph_memory = None
    pid = await _create(context, "No Graph")
    res = json.loads(await tool_manage_projects(context, action="delete",
                                                project_id=pid))
    assert res.get("deleted") is True


# ------------------------------------------------------------- HTTP API wiring

@pytest.fixture
def api(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ghost_agent.api.projects_routes import projects_router
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    ctx = SimpleNamespace(args=SimpleNamespace(api_key=""), project_store=store,
                          scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                          graph_memory=GraphMemory(tmp_path), current_project_id=None)
    app = FastAPI()
    app.state.agent = SimpleNamespace(context=ctx)
    app.include_router(projects_router)
    return TestClient(app), store, ctx.graph_memory


@pytest.mark.parametrize("hard,unlinked", [("true", True), ("false", False)])
def test_api_delete_unlinks_only_on_hard(api, hard, unlinked):
    """`DELETE /api/projects/{pid}` is the second deliverer: hard removes the
    graph edges like the tool; the default soft delete (archive) keeps them."""
    tc, store, gm = api
    pid = store.create_project(title="Cascade", kind="GENERAL", goal="g")
    gm.add_triplets([{"subject": "user", "predicate": "RESUMES",
                      "object": f"project {pid}"}], raw=True)
    assert tc.delete(f"/api/projects/{pid}", params={"hard": hard}).status_code == 204
    assert any(pid in o for _, _, o in _rows(gm)) is not unlinked


# ------------------------------------------------- review round 1 (2026-10-06)

def test_title_named_node_is_forgotten_the_live_survivor(tmp_path):
    """Review #1: the extractor's main shape names the project by TITLE.
    Live, `ai self awareness exploration TESTED self-awareness emergence`
    survived the id-only version — the same steering edge as 2cb40b10."""
    g = GraphMemory(tmp_path)
    g.add_triplets([
        {"subject": "ai self awareness exploration", "predicate": "TESTED",
         "object": "self-awareness emergence"},
        {"subject": "AI-Self Awareness  Exploration", "predicate": "HAS_STATUS", "object": "done"},
        {"subject": "user", "predicate": "WORKS_ON", "object": "ai self awareness exploration"},
        # an owner life fact on the title node is kept
        {"subject": "user", "predicate": "OWNS", "object": "ai self awareness exploration"},
        # a node that merely CONTAINS the title is not the title
        {"subject": "ai self awareness exploration notes", "predicate": "HAS", "object": "x"},
        {"subject": "user", "predicate": "EXPLORES", "object": "self-awareness"},
    ], raw=True)
    g.forget_project("7b62e5e533d1", title="AI self awareness exploration")
    left = _rows(g)
    assert not any(p in ("TESTED", "HAS_STATUS", "WORKS_ON") for _, p, _ in left)
    assert any(p == "OWNS" for _, p, _ in left)
    assert ("ai self awareness exploration notes", "HAS", "x") in left
    assert ("user", "EXPLORES", "self-awareness") in left


def test_one_word_title_node_is_not_forgotten(tmp_path):
    """A one-word title is also a topic ("Chess"): its node is kept; the
    generic `project HAS_TITLE chess` edge still goes."""
    g = GraphMemory(tmp_path)
    g.add_triplets([
        {"subject": "user", "predicate": "PLAYS", "object": "chess"},
        {"subject": "chess", "predicate": "HAS_OPENING", "object": "sicilian"},
        {"subject": "project", "predicate": "HAS_TITLE", "object": "chess"},
    ], raw=True)
    g.forget_project("aaaabbbbcccc", title="Chess")
    left = _rows(g)
    assert ("user", "PLAYS", "chess") in left and ("chess", "HAS_OPENING", "sicilian") in left
    assert ("project", "HAS_TITLE", "chess") not in left


def test_shared_title_keeps_the_title_node_too(tmp_path):
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": "cascade test run", "predicate": "TESTED", "object": "x"}], raw=True)
    g.forget_project("aaaabbbbcccc", title="Cascade Test Run", forget_title=False)
    assert ("cascade test run", "TESTED", "x") in _rows(g)


def test_archive_failure_soft_expires_instead_of_skipping(gm, monkeypatch):
    """Review #4: a failed archive used to delete nothing and report 0, so the
    project was gone and its edges kept steering. Now: soft-expire."""
    monkeypatch.setattr(gm, "_archive_rows", lambda *a, **k: False)
    assert gm.forget_project("7b62e5e533d1") > 0
    with sqlite3.connect(gm.db_path) as conn:
        live = conn.execute("SELECT subject, object FROM triplets WHERE valid_until IS NULL").fetchall()
        expired = conn.execute("SELECT COUNT(*) FROM triplets WHERE valid_until IS NOT NULL").fetchone()[0]
    assert expired > 0
    assert not any("7b62e5e533d1" == s.split()[-1] for s, _ in live)
    assert not gm.nx_graph.has_node("project 7b62e5e533d1")


async def test_accent_twin_title_is_shared(context):
    """Review #3: the caller compared with .lower(), forget_project with
    _fold, so deleting "Café Planner" removed live "Cafe Planner"'s title
    edge. Both now use GraphMemory.project_title_key."""
    gm = context.graph_memory
    a = await _create(context, "Café Planner")
    context.project_store.create_project(title="Cafe Planner", kind="GENERAL", goal="g")
    gm.add_triplets([{"subject": "project", "predicate": "HAS_TITLE", "object": "cafe planner"},
                     {"subject": "cafe planner", "predicate": "RUNS_ON", "object": "8100"}], raw=True)
    await tool_manage_projects(context, action="delete", project_id=a)
    left = _rows(gm)
    assert ("project", "HAS_TITLE", "cafe planner") in left
    assert ("cafe planner", "RUNS_ON", "8100") in left
