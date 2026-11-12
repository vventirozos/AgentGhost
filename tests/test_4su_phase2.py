"""§4SU Phase 2 — a task's declared kind (core/task_kind.py) routes it; the
wording classifiers stay the default for an undeclared task."""
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import ghost_agent.core.project_advancer as PA
from ghost_agent.core import task_close as TC
from ghost_agent.core.coding_executor import CodingResult
from ghost_agent.core.planning import ProjectPlan, TaskStatus
from ghost_agent.core.task_kind import normalize_kind, with_deliverable, TASK_KINDS
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import tool_manage_projects
from tests.test_4qa_project_task_integrity import _ctx as _tool_ctx
from tests.test_4su_phase0 import owner_turn  # noqa: F401  (fixture)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")


def _actx(store):
    return SimpleNamespace(project_store=store, llm_client=None, scratchpad=None,
                           workspace_model=None, current_project_id=None)


# ───────────────────────────── the vocabulary ─────────────────────────────

@pytest.mark.parametrize("raw,kind", [("build", "build"), ("Code", "build"), ("document", "write"),
                                      ("experiment", "model_run"), ("model-run", "model_run"),
                                      ("human", "owner"), ("", ""), ("none", ""), (None, "")])
def test_kinds_normalise(raw, kind):
    assert normalize_kind(raw) == kind


def test_an_unknown_kind_is_refused_not_dropped():
    with pytest.raises(ValueError):
        normalize_kind("bulid")


def test_a_deliverable_is_appended_once():
    assert with_deliverable("Write the report", "out/report.md") == "Write the report → out/report.md"
    assert with_deliverable("Write it → a.md", "b.md") == "Write it → a.md"
    assert with_deliverable("Write it", "") == "Write it"


# ───────────────────────────── storage ─────────────────────────────

def test_the_kind_is_stored_hydrated_and_updated(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Build the parser", kind="code")
    assert store.get_task(tid)["kind"] == "build"
    assert ProjectPlan(store, pid).tree.nodes[tid].kind == "build"
    store.update_task(tid, kind="research")
    assert store.get_task(tid)["kind"] == "research"
    with pytest.raises(ValueError):
        store.update_task(tid, kind="nonsense")
    with pytest.raises(ValueError):
        store.add_task(pid, "x", kind="nonsense")


def test_an_old_database_gains_the_column(tmp_path):
    s = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = s.create_project("P")
    tid = s.add_task(pid, "old task")
    db = next(Path(tmp_path / "m").rglob("*.db"))
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE tasks DROP COLUMN kind")
    con.commit()
    con.close()
    s2 = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    assert s2.get_task(tid)["kind"] == ""
    s2.update_task(tid, kind="write")
    assert s2.get_task(tid)["kind"] == "write"


def test_the_plan_shows_the_kind(store):
    pid = store.create_project("P")
    root = store.add_task(pid, "Ship")
    store.add_task(pid, "Write the notes → notes.md", parent_id=root, kind="write")
    assert "<write> Write the notes → notes.md" in ProjectPlan(store, pid).render()


# ───────────────────────────── the tool ─────────────────────────────

async def test_object_subtasks_declare_kind_and_deliverable(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _tool_ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="Site", goal="Ship the site", subtasks=[
        {"description": "Build the page", "kind": "build", "deliverable": "index.html"},
        {"description": "Ask the owner for the domain", "kind": "owner"},
        "Research hosting options",
    ])
    pid = ctx.current_project_id
    rows = {t["description"]: t["kind"] for t in store.list_tasks(pid)}
    assert rows == {"Build the page → index.html": "build", "Ask the owner for the domain": "owner",
                    "Research hosting options": ""}


async def test_decompose_takes_objects_as_json_text(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _tool_ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="Study", goal="Study prompts")
    pid = ctx.current_project_id
    out = json.loads(str(await tool_manage_projects(ctx, action="task_decompose", subtasks=json.dumps([
        {"description": "Probe the three prompts", "kind": "model_run", "deliverable": "results/runs.md"},
        {"description": "Write up the findings", "kind": "write", "deliverable": "report.md"}]))))
    assert [t["kind"] for t in out["tasks"]] == ["model_run", "write"]
    assert {t["kind"] for t in store.list_tasks(pid)} == {"model_run", "write"}


async def test_an_unknown_kind_in_a_plan_is_an_error(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _tool_ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="Study", goal="g")
    out = str(await tool_manage_projects(ctx, action="task_decompose",
                                         subtasks=[{"description": "x", "kind": "bulid"}]))
    assert "unknown task kind" in out
    assert not store.list_tasks(ctx.current_project_id)


async def test_stringified_objects_are_read_and_a_broken_list_is_refused(tmp_path, owner_turn):
    # r1 MINOR: each item a JSON string; a list that does not parse
    owner_turn("a1b2c3d4")
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _tool_ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="P", goal="g")
    pid = ctx.current_project_id
    await tool_manage_projects(ctx, action="task_decompose", subtasks=[
        '{"description": "Build the page", "kind": "build"}', "Research hosting"])
    assert {t["description"]: t["kind"] for t in store.list_tasks(pid)} == {
        "Build the page": "build", "Research hosting": ""}
    out = str(await tool_manage_projects(ctx, action="task_decompose",
                                         subtasks='[{"description": "x", "kind": "build"},]'))
    assert "does not parse" in out and len(store.list_tasks(pid)) == 2


async def test_task_add_and_task_update_set_the_kind(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _tool_ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="P", goal="g")
    out = json.loads(str(await tool_manage_projects(ctx, action="task_add", description="Probe it",
                                                    task_kind="model_run")))
    tid = out.get("task_id") or out.get("id")
    assert store.get_task(tid)["kind"] == "model_run"
    await tool_manage_projects(ctx, action="task_update", task_id=tid, task_kind="none")
    assert store.get_task(tid)["kind"] == ""


# ───────────────────────────── the router ─────────────────────────────

def _exe(calls):
    async def exe(ctx, desc, **kw):
        calls.append(desc)
        return CodingResult(True, "built", files=[])
    return exe


async def _advance(store, pid, *, owner=True, project_kind=None, calls=None, searches=None, classifier=None):
    calls = calls if calls is not None else []
    searches = searches if searches is not None else []

    async def runner(name, args):
        searches.append(name)
        return "Result 1: a long relevant page about the topic. " * 20

    cls = classifier or AsyncMock(return_value="research")
    res = await PA.advance_once(_actx(store), pid, tool_runner=runner, coding_executor=_exe(calls),
                                llm_classifier=cls, owner_requested=owner)
    return res, calls, searches, cls


async def test_a_declared_build_is_built_whatever_its_words(store):
    pid = store.create_project("P")
    store.update_project(pid, status="ACTIVE")
    store.add_task(pid, "Research and compare the parsing libraries", kind="build")
    res, calls, searches, cls = await _advance(store, pid)
    assert calls and "web_search" not in searches
    cls.assert_not_called()


async def test_a_declared_research_task_searches_in_a_coding_project(store):
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    store.add_task(pid, "Implement nothing: survey the parser landscape", kind="research")
    res, calls, searches, _ = await _advance(store, pid)
    assert not calls and "web_search" in searches


async def test_declared_research_promising_a_code_file_is_built(store):
    pid = store.create_project("P")
    store.update_project(pid, status="ACTIVE")
    store.add_task(pid, "Survey the parsers and write the adapter → src/adapter.py", kind="research")
    res, calls, searches, _ = await _advance(store, pid)
    assert calls


@pytest.mark.parametrize("project_kind", ["GENERAL", "CODING"])
async def test_a_declared_model_run_holds_in_every_project_kind(store, project_kind):
    # the audit: a CODING project skipped the experiment hold
    pid = store.create_project("P", kind=project_kind)
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Collect the answers to the three prompts", kind="model_run")
    res, calls, _, _ = await _advance(store, pid, owner=False)
    assert not calls and store.get_task(tid)["status"] == "NEEDS_USER"
    assert store.latest_task_event(pid, tid, ("autoadvance_needs_user",))["payload"]["hold"] == "model_experiment"


async def test_a_declared_model_run_is_left_to_the_owner_s_turn(store):
    pid = store.create_project("P", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Collect the answers to the three prompts", kind="model_run")
    res, calls, _, _ = await _advance(store, pid, owner=True)
    assert not calls and res.classification == "blocked" and store.get_task(tid)["status"] == "PENDING"


async def test_a_declared_owner_task_waits_for_the_owner(store):
    pid = store.create_project("P")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Buy the domain name", kind="owner")
    res, calls, searches, _ = await _advance(store, pid)
    assert not calls and not searches and store.get_task(tid)["status"] == "NEEDS_USER"


async def test_an_undeclared_task_is_routed_by_the_classifier_as_before(store):
    pid = store.create_project("P")
    store.update_project(pid, status="ACTIVE")
    store.add_task(pid, "ambiguous task title")
    res, calls, searches, cls = await _advance(store, pid, classifier=AsyncMock(return_value="coding"))
    cls.assert_called_once()
    assert calls


async def test_a_kind_never_removes_the_experiment_hold(store):
    # r1 MAJOR 2: a kind may ADD the hold (model_run), never remove it
    pid = store.create_project("P", goal="Evaluate Qwen on GSM8K")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Run the 50 GSM8K prompts on Qwen and record its answers", kind="build")
    res, calls, _, _ = await _advance(store, pid, owner=False)
    assert not calls and store.get_task(tid)["status"] == "NEEDS_USER"


async def test_a_declared_introspective_model_run_stays_held(store):
    pid = store.create_project("P")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Evaluate whether I truly choose my own responses", kind="model_run")
    res, calls, _, _ = await _advance(store, pid, owner=False)
    assert not calls and store.get_task(tid)["status"] == "NEEDS_USER"


# ───────────────────────────── the close check ─────────────────────────────

def test_a_declared_model_run_needs_a_transcript_whatever_its_words(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Collect the answers", kind="model_run")
    v = TC.check_close(store, pid, ProjectPlan(store, pid).tree, tid, actor="model", result="collected")
    assert v.first.gate == "model_run" and "declared a model_run" in v.first.reason


def test_a_retype_does_not_skip_the_experiment_gate(store):
    # r1 MAJOR 2: task_update task_kind="build" after a refusal closed it
    pid = store.create_project("P", goal="Evaluate Qwen on GSM8K")
    tid = store.add_task(pid, "Run the 50 GSM8K prompts on Qwen and record its answers")
    tree = lambda: ProjectPlan(store, pid).tree
    assert TC.check_close(store, pid, tree(), tid, actor="model", result="done").first.gate == "model_run"
    store.update_task(tid, kind="build")
    assert TC.check_close(store, pid, tree(), tid, actor="model", result="done").first.gate == "model_run"


def test_a_declared_build_that_writes_up_a_model_run_is_still_refused(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Write the summary → results/l1.md", kind="build")
    ws = Path(store.ensure_workspace(pid))
    (ws / "results").mkdir()
    (ws / "results/l1.md").write_text("# Run\n\n## Transcript\n\nModel response: the sky is blue (Qwen said).\n")
    v = TC.check_close(store, pid, ProjectPlan(store, pid).tree, tid, actor="model", result="wrote it",
                       files=["results/l1.md"])
    assert v.first and v.first.gate == "model_run"


# ───────────────────────────── the web console ─────────────────────────────

async def test_the_console_creates_and_edits_the_kind(store, monkeypatch):
    from ghost_agent.api import projects_routes as R
    pid = store.create_project("P")
    monkeypatch.setattr(R, "_store", lambda r: store)
    out = await R.add_task(pid, R.TaskCreate(description="Probe it", kind="experiment"), SimpleNamespace(headers={}))
    tid = out["id"]
    assert store.get_task(tid)["kind"] == "model_run"
    await R.update_task(pid, tid, R.TaskUpdate(kind="owner"), SimpleNamespace(headers={}))
    assert store.get_task(tid)["kind"] == "owner"
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        await R.add_task(pid, R.TaskCreate(description="x", kind="nonsense"), SimpleNamespace(headers={}))


def test_every_kind_has_a_route():
    from ghost_agent.core.task_kind import ROUTE_LABEL
    assert set(ROUTE_LABEL) == set(TASK_KINDS)


def test_the_briefing_shows_a_declared_kind(store):
    from ghost_agent.core.prompts import build_project_briefing
    pid = store.create_project("P")
    store.add_task(pid, "Probe the three prompts", kind="model_run")
    store.add_task(pid, "Plain task")
    b = build_project_briefing(store, pid)
    assert "<model_run> Probe the three prompts" in b
    assert "] Plain task" in b


async def test_the_owner_s_run_builds_a_gate_word_form_task(store):
    pid = store.create_project("P", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    store.add_task(pid, "Build the approval form", kind="build")
    res, calls, _, _ = await _advance(store, pid, owner=True)
    assert calls


def test_plain_text_subtasks_with_brackets_stay_text():
    # r2 MINOR/NIT: only a list of OBJECTS is read as JSON
    from ghost_agent.tools.projects import _split_declared_subtasks as split
    assert split("[draft] write {name} template") == ("[draft] write {name} template", {})
    assert split("[ ] Set up {config}\n[ ] Write tests")[1] == {}
    assert split(['{"a": 1}', "x"]) == (['{"a": 1}', "x"], {})


@pytest.mark.parametrize("text", ["{name} template for the {site}", "{intro}\nWrite the body\n{outro}",
                                  '{"a": 1}'])
def test_brace_text_stays_text(text):
    # r4 MAJOR/MINOR: only a subtask-shaped JSON object is read as one
    from ghost_agent.tools.projects import _split_declared_subtasks as split
    assert split(text) == (text, {})
    assert split([text]) == ([text], {})


def test_a_bare_object_is_read_and_a_shapeless_one_refused():
    # r3 MINOR: one object not in a list; a string object judged as the dict form
    from ghost_agent.tools.projects import _split_declared_subtasks as split
    assert split('{"description": "A", "kind": "build"}') == (["A"], {"A": "build"})
    with pytest.raises(ValueError):
        split(['{"name": "Build parser", "kind": "build"}'])


async def test_decompose_under_a_task_keeps_the_kinds(tmp_path, owner_turn):
    # battery P3: the under-a-task path (plan.decompose kinds=)
    owner_turn("a1b2c3d4")
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _tool_ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title="P", goal="g")
    pid = ctx.current_project_id
    root = store.add_task(pid, "Ship the study")
    await tool_manage_projects(ctx, action="task_decompose", task_id=root, subtasks=[
        {"description": "Probe the prompts", "kind": "model_run"}, "Write it up"])
    kids = {t["description"]: t["kind"] for t in store.list_tasks(pid) if t.get("parent_id") == root}
    assert kids == {"Probe the prompts": "model_run", "Write it up": ""}


async def test_the_owner_s_run_keeps_an_introspective_model_run_a_model_run(store):
    # battery A12: never released to a brief
    pid = store.create_project("P")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Evaluate whether I truly choose my own responses", kind="model_run")
    res, calls, searches, _ = await _advance(store, pid, owner=True)
    assert not calls and not searches and res.classification == "blocked"
    assert store.get_task(tid)["status"] != "DONE"
