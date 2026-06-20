"""Unit tests for the manage_projects tool handler."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import (
    tool_manage_projects, MANAGE_PROJECTS_TOOL_DEF, _ACTIONS,
)


@pytest.fixture
def context(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    scratchpad = Scratchpad(persist_path=tmp_path / "sp.db")
    return SimpleNamespace(
        project_store=store,
        scratchpad=scratchpad,
        current_project_id=None,
    )


def _parse(s: str):
    """Parse a JSON response or raise so tests surface malformed output."""
    return json.loads(s)


# --------------------------------------------------------------------- basics

async def test_tool_def_has_required_action(context):
    assert MANAGE_PROJECTS_TOOL_DEF["function"]["name"] == "manage_projects"
    params = MANAGE_PROJECTS_TOOL_DEF["function"]["parameters"]
    assert "action" in params["required"]


async def test_rejects_unknown_action(context):
    res = await tool_manage_projects(context, action="nope")
    assert res.startswith("ERROR")


async def test_rejects_when_store_missing(tmp_path):
    ctx = SimpleNamespace(project_store=None, scratchpad=None,
                          current_project_id=None)
    res = await tool_manage_projects(ctx, action="list")
    assert "project_store" in res


# --------------------------------------------------------------------- create / switch / exit

async def test_create_enters_project_mode_and_persists_to_scratchpad(context):
    res = await tool_manage_projects(context, action="create",
                                     title="T1", kind="CODING", goal="Ship")
    data = _parse(res)
    assert "created" in data
    assert context.current_project_id == data["created"]
    # Scratchpad sentinel was written
    assert context.scratchpad.get("__current_project__") == data["created"]


async def test_create_requires_title(context):
    res = await tool_manage_projects(context, action="create")
    assert res.startswith("ERROR")


async def test_switch_returns_workspace_note(context):
    """Switching into a project tells the model its working directory moved
    to projects/<id>/ (so it doesn't fumble file paths and burn strikes)."""
    pid = _parse(await tool_manage_projects(context, action="create", title="WS"))["created"]
    res = _parse(await tool_manage_projects(context, action="switch", project_id=pid))
    assert res["workspace"] == f"projects/{pid}"
    assert f"projects/{pid}" in res["note"]
    # the note steers the model to bare filenames (now "BARE name")
    assert "bare name" in res["note"].lower()


async def test_create_with_same_title_reuses_id_regardless_of_age(context):
    """The Qwen retry-storm symptom: same title called repeatedly creates
    duplicates. Now reuses the existing id forever (until archived)."""
    a = _parse(await tool_manage_projects(context, action="create",
                                          title="Build a parser"))
    b = _parse(await tool_manage_projects(context, action="create",
                                          title="Build a parser"))
    assert a["created"] == b["created"]
    assert b.get("duplicate_of") == a["created"]
    assert "STOP" in b.get("note", "").upper() or "stop" in b.get("note", "")
    # Only one row in the store
    assert len(context.project_store.list_projects()) == 1


async def test_duplicate_check_is_case_and_whitespace_insensitive(context):
    a = _parse(await tool_manage_projects(context, action="create",
                                          title="Build a Parser"))
    b = _parse(await tool_manage_projects(context, action="create",
                                          title="  build a parser  "))
    assert a["created"] == b["created"]


async def test_duplicate_check_ignores_archived(context):
    a = _parse(await tool_manage_projects(context, action="create", title="Foo"))
    # Archive it, then create again — should be allowed (legitimate restart)
    await tool_manage_projects(context, action="archive",
                               project_id=a["created"])
    b = _parse(await tool_manage_projects(context, action="create", title="Foo"))
    assert a["created"] != b["created"]
    # archive is reversible: the original row still exists, marked ARCHIVED
    got = _parse(await tool_manage_projects(context, action="get",
                                            project_id=a["created"]))
    assert got["status"] == "ARCHIVED"


async def test_duplicate_check_can_be_disabled_via_zero_window(context, monkeypatch):
    """Setting the window to 0 disables the guard for tests / opt-out."""
    from ghost_agent.tools import projects as projects_module
    monkeypatch.setattr(projects_module,
                        "_DUPLICATE_CREATE_WINDOW_SECONDS", 0)
    a = _parse(await tool_manage_projects(context, action="create", title="Foo"))
    b = _parse(await tool_manage_projects(context, action="create", title="Foo"))
    assert a["created"] != b["created"]


async def test_duplicate_check_holds_after_long_age(context, monkeypatch):
    """Regression for the 2026-04-19 trace: project lasted 8 minutes;
    the old 5-min window expired and a duplicate slipped through.
    Now the guard is existence-based, no expiry."""
    a = _parse(await tool_manage_projects(context, action="create",
                                          title="Long Project"))
    # Backdate the project's created_at by an hour to simulate an old
    # active project. The guard must STILL refuse a new create.
    import time as _time
    with context.project_store._connect() as conn:
        conn.execute(
            "UPDATE projects SET created_at = ? WHERE id = ?",
            (_time.time() - 3600, a["created"]),
        )
        conn.commit()
    b = _parse(await tool_manage_projects(context, action="create",
                                          title="Long Project"))
    assert b["created"] == a["created"]
    assert b.get("duplicate_of") == a["created"]


async def test_duplicate_note_tells_user_how_to_force_new(context):
    """The user-visible note must explain the escape hatch (archive
    the old project) so the model has a concrete recovery action
    instead of looping."""
    _parse(await tool_manage_projects(context, action="create", title="X"))
    res = _parse(await tool_manage_projects(context, action="create", title="X"))
    assert "archive" in res["note"].lower()
    assert "action=archive" in res["note"]  # the dedicated soft-delete action


async def test_duplicate_response_has_refused_flag(context):
    """2026-04-19 trace E8: model kept interpreting the successful
    'created' response as permission to proceed with new work, burning
    turns. Response now carries refused=True so the model can't mis-
    read it as a happy-path create."""
    _parse(await tool_manage_projects(context, action="create", title="X"))
    res = _parse(await tool_manage_projects(context, action="create", title="X"))
    assert res["refused"] is True
    assert res["action_taken"] == "reused_existing_project"
    assert "agent_instruction" in res
    assert "STOP" in res["agent_instruction"]
    # created= kept for back-compat but should equal the existing id
    assert res["created"] == res["existing_project_id"]


async def test_duplicate_retry_count_increments(context):
    a = _parse(await tool_manage_projects(context, action="create", title="X"))
    b = _parse(await tool_manage_projects(context, action="create", title="X"))
    c = _parse(await tool_manage_projects(context, action="create", title="X"))
    assert b["retry_count"] == 1
    assert c["retry_count"] == 2
    # Metadata is persisted so we can inspect the loop later
    proj = context.project_store.get_project(a["created"])
    assert proj["metadata"]["duplicate_create_retries"] == 2


async def test_duplicate_retry_escalates_after_threshold(context):
    """On retry >= 3 the wording changes to a much louder alarm and
    explicitly tells the model it is in a loop."""
    _parse(await tool_manage_projects(context, action="create", title="X"))
    for _ in range(3):
        res = _parse(await tool_manage_projects(
            context, action="create", title="X",
        ))
    # 3rd refused call ⇒ retry_count=3 ⇒ escalated message
    assert res["retry_count"] == 3
    instr = res["agent_instruction"]
    assert "LOOP" in instr
    assert "retry #3" in instr
    assert "bug" in instr  # tells the model that re-reading is a bug


async def test_first_create_does_not_carry_refused(context):
    """Sanity: the very first create call is a success, not a refusal."""
    res = _parse(await tool_manage_projects(context, action="create", title="New"))
    assert "refused" not in res
    assert res["created"]
    # A successful create DOES carry an agent_instruction now (the pacing /
    # per-file decomposition guidance) — but it must be the success guidance,
    # not the refusal "STOP calling create" message.
    assert "STOP calling create" not in res.get("agent_instruction", "")
    assert "task_decompose" in res.get("agent_instruction", "")


async def test_switch_changes_active_project(context):
    r1 = _parse(await tool_manage_projects(context, action="create", title="A"))
    r2 = _parse(await tool_manage_projects(context, action="create", title="B"))
    res = await tool_manage_projects(context, action="switch",
                                     project_id=r1["created"])
    data = _parse(res)
    assert data["switched_to"] == r1["created"]
    assert context.current_project_id == r1["created"]


async def test_switch_rejects_unknown_project(context):
    res = await tool_manage_projects(context, action="switch",
                                     project_id="nope")
    assert res.startswith("ERROR")


async def test_exit_clears_current_and_scratchpad(context):
    _parse(await tool_manage_projects(context, action="create", title="A"))
    assert context.current_project_id is not None
    res = await tool_manage_projects(context, action="exit")
    assert "exited" in _parse(res)
    assert context.current_project_id is None
    assert context.scratchpad.get("__current_project__") is None


# --------------------------------------------------------------------- list / get / update / delete

async def test_list_returns_all_projects_with_current(context):
    p1 = _parse(await tool_manage_projects(context, action="create", title="A"))
    p2 = _parse(await tool_manage_projects(context, action="create", title="B"))
    res = _parse(await tool_manage_projects(context, action="list"))
    titles = {p["title"] for p in res["projects"]}
    assert titles == {"A", "B"}
    assert res["current"] == p2["created"]


async def test_list_filters_by_status(context):
    a = _parse(await tool_manage_projects(context, action="create", title="A"))
    _parse(await tool_manage_projects(context, action="create", title="B"))
    # Archive A (still exists, just not ACTIVE)
    await tool_manage_projects(context, action="archive", project_id=a["created"])
    res = _parse(await tool_manage_projects(context, action="list",
                                            status_filter="ACTIVE"))
    assert all(p["id"] != a["created"] for p in res["projects"])


async def test_get_returns_project(context):
    p = _parse(await tool_manage_projects(context, action="create", title="A"))
    res = _parse(await tool_manage_projects(context, action="get",
                                            project_id=p["created"]))
    assert res["title"] == "A"


async def test_update_changes_fields(context):
    p = _parse(await tool_manage_projects(context, action="create", title="A"))
    res = _parse(await tool_manage_projects(context, action="update",
                                            project_id=p["created"],
                                            goal="new goal",
                                            status="PAUSED"))
    assert res["updated"] is True
    got = _parse(await tool_manage_projects(context, action="get",
                                            project_id=p["created"]))
    assert got["goal"] == "new goal"
    assert got["status"] == "PAUSED"


async def test_delete_is_permanent_and_exits_if_current(context):
    """`delete` is a HARD, irreversible removal: the row is gone (get →
    error/None), and if it was the current project, project mode is exited."""
    p = _parse(await tool_manage_projects(context, action="create", title="A"))
    pid = p["created"]
    assert context.current_project_id == pid
    res = _parse(await tool_manage_projects(context, action="delete", project_id=pid))
    assert res.get("deleted") is True and res.get("hard") is True
    assert context.current_project_id is None
    # row is actually gone — not merely archived
    assert context.project_store.get_project(pid) is None
    assert all(p2["id"] != pid for p2 in context.project_store.list_projects())


async def test_delete_removes_workspace_files(context, tmp_path):
    """`delete` also wipes the project's on-disk workspace."""
    p = _parse(await tool_manage_projects(context, action="create", title="WS"))
    pid = p["created"]
    ws = context.project_store.sandbox_root / "projects" / pid
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "scratch.txt").write_text("data")
    await tool_manage_projects(context, action="delete", project_id=pid)
    assert not ws.exists()  # files removed


async def test_archive_is_reversible_and_keeps_row(context):
    """`archive` is the soft path: row survives as ARCHIVED, resumable."""
    p = _parse(await tool_manage_projects(context, action="create", title="A"))
    pid = p["created"]
    res = _parse(await tool_manage_projects(context, action="archive", project_id=pid))
    assert res.get("archived") is True
    assert context.current_project_id is None  # exited current
    got = _parse(await tool_manage_projects(context, action="get", project_id=pid))
    assert got["status"] == "ARCHIVED"  # still there


async def test_archive_keeps_workspace_files(context):
    p = _parse(await tool_manage_projects(context, action="create", title="WS2"))
    pid = p["created"]
    ws = context.project_store.sandbox_root / "projects" / pid
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "keep.txt").write_text("data")
    await tool_manage_projects(context, action="archive", project_id=pid)
    assert (ws / "keep.txt").exists()  # files preserved on archive


# --------------------------------------------------------------------- tasks

async def test_task_add_and_list(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    r = _parse(await tool_manage_projects(context, action="task_add",
                                          description="root"))
    tid = r["task_id"]
    child = _parse(await tool_manage_projects(context, action="task_add",
                                              description="child",
                                              parent_id=tid))
    res = _parse(await tool_manage_projects(context, action="task_list"))
    descs = [t["description"] for t in res["tasks"]]
    assert descs == ["root", "child"]


async def test_task_add_requires_project(context):
    res = await tool_manage_projects(context, action="task_add",
                                     description="x")
    assert res.startswith("ERROR")


async def test_task_add_requires_description(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    res = await tool_manage_projects(context, action="task_add")
    assert res.startswith("ERROR")


async def test_task_add_response_warns_about_pending_status(context):
    """2026-04-19 trace 94 regression: model marked tasks DONE right
    after adding them because it conflated 'added' with 'completed'.
    The add response must tell the model otherwise."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    r = _parse(await tool_manage_projects(
        context, action="task_add", description="Research parsers",
    ))
    assert r["task_id"]
    assert r["status"] == "PENDING"
    assert "agent_instruction" in r
    assert "Added != Done" in r["agent_instruction"]
    assert "STOP" in r["agent_instruction"]


async def test_task_add_refuses_duplicate_sibling(context):
    """Regression: model loops adding same task when re-reading the
    user's bullet list. Refuse same description at same level."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    a = _parse(await tool_manage_projects(
        context, action="task_add", description="Research parsers",
    ))
    b = _parse(await tool_manage_projects(
        context, action="task_add", description="Research parsers",
    ))
    assert b.get("refused") is True
    assert b["existing_task_id"] == a["task_id"]
    assert b["task_id"] == a["task_id"]  # back-compat
    assert "NEXT item" in b["agent_instruction"]
    # Only one row in the store
    assert len(context.project_store.list_tasks(
        context.current_project_id,
    )) == 1


async def test_task_add_duplicate_check_is_case_insensitive(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    a = _parse(await tool_manage_projects(
        context, action="task_add", description="Design THE Parser",
    ))
    b = _parse(await tool_manage_projects(
        context, action="task_add", description="design the parser  ",
    ))
    assert b.get("refused") is True
    assert b["existing_task_id"] == a["task_id"]


async def test_task_add_allows_same_description_under_different_parent(context):
    """Legit case: a subtree can re-use a description as a subtask of
    a different parent. E.g. 'write tests' under both modules."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    a = _parse(await tool_manage_projects(
        context, action="task_add", description="module a",
    ))
    b = _parse(await tool_manage_projects(
        context, action="task_add", description="module b",
    ))
    # Same description under a vs b — both allowed
    ca = _parse(await tool_manage_projects(
        context, action="task_add", description="write tests",
        parent_id=a["task_id"],
    ))
    cb = _parse(await tool_manage_projects(
        context, action="task_add", description="write tests",
        parent_id=b["task_id"],
    ))
    assert ca["task_id"] != cb["task_id"]
    assert not ca.get("refused")
    assert not cb.get("refused")


async def test_task_add_allows_same_description_after_completion(context):
    """Legit case: a task was done; the user asks to add a new one
    with the same description to redo it. A DONE sibling doesn't
    block the add."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    a = _parse(await tool_manage_projects(
        context, action="task_add", description="Benchmark",
    ))
    await tool_manage_projects(
        context, action="task_update", task_id=a["task_id"], status="DONE",
    )
    b = _parse(await tool_manage_projects(
        context, action="task_add", description="Benchmark",
    ))
    assert not b.get("refused")
    assert b["task_id"] != a["task_id"]


async def test_tool_description_prefers_decompose_for_lists(context):
    """If this directive drops out, Qwen will go back to N-turn task_add
    loops for user-provided lists."""
    from ghost_agent.tools.projects import MANAGE_PROJECTS_TOOL_DEF
    desc = MANAGE_PROJECTS_TOOL_DEF["function"]["description"]
    assert "2+ tasks" in desc or "2+ task" in desc
    assert "task_decompose" in desc
    assert "Added" in desc and "Done" in desc


async def test_task_update_marks_done(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    r = _parse(await tool_manage_projects(context, action="task_add",
                                          description="root"))
    res = _parse(await tool_manage_projects(
        context, action="task_update", task_id=r["task_id"],
        status="DONE", result="shipped",
    ))
    # Response shape changed when task_ids[] was added — `updated` is now
    # a list of slim {id,status,result_summary} entries rather than a bool.
    assert res["count"] == 1
    assert res["updated"][0]["status"] == "DONE"
    assert res["updated"][0]["result_summary"] == "shipped"


async def test_task_update_rejects_bad_status(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    r = _parse(await tool_manage_projects(context, action="task_add",
                                          description="root"))
    res = await tool_manage_projects(
        context, action="task_update", task_id=r["task_id"],
        status="BOGUS",
    )
    assert res.startswith("ERROR")


async def test_task_update_bulk_marks_many_done(context):
    """The trace symptom: model loops one task_update per turn for 6
    siblings. With task_ids=[…] this becomes one call.
    """
    _parse(await tool_manage_projects(context, action="create", title="P"))
    ids = [
        _parse(await tool_manage_projects(
            context, action="task_add", description=f"t{i}",
        ))["task_id"]
        for i in range(4)
    ]
    res = _parse(await tool_manage_projects(
        context, action="task_update",
        task_ids=ids, status="DONE",
    ))
    assert res["count"] == 4
    assert {u["status"] for u in res["updated"]} == {"DONE"}
    for tid in ids:
        assert context.project_store.get_task(tid)["status"] == "DONE"


async def test_task_update_bulk_reports_missing_ids_but_continues(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    real = _parse(await tool_manage_projects(
        context, action="task_add", description="real",
    ))["task_id"]
    res = _parse(await tool_manage_projects(
        context, action="task_update",
        task_ids=[real, "ghost-id"], status="DONE",
    ))
    assert res["count"] == 1
    assert res["missing"] == ["ghost-id"]
    assert context.project_store.get_task(real)["status"] == "DONE"


async def test_task_update_bulk_combines_task_id_and_task_ids(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    a = _parse(await tool_manage_projects(
        context, action="task_add", description="a",
    ))["task_id"]
    b = _parse(await tool_manage_projects(
        context, action="task_add", description="b",
    ))["task_id"]
    res = _parse(await tool_manage_projects(
        context, action="task_update",
        task_id=a, task_ids=[b], status="DONE",
    ))
    assert res["count"] == 2


async def test_task_update_bulk_drops_description_for_safety(context):
    """A description rewrite in a bulk call would broadcast one new
    description across many tasks — almost always a mistake. Single-id
    calls still allow it.
    """
    _parse(await tool_manage_projects(context, action="create", title="P"))
    a = _parse(await tool_manage_projects(
        context, action="task_add", description="a",
    ))["task_id"]
    b = _parse(await tool_manage_projects(
        context, action="task_add", description="b",
    ))["task_id"]
    await tool_manage_projects(
        context, action="task_update",
        task_ids=[a, b], status="IN_PROGRESS",
        description="overwritten",
    )
    assert context.project_store.get_task(a)["description"] == "a"
    assert context.project_store.get_task(b)["description"] == "b"


async def test_task_update_single_still_accepts_description(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    a = _parse(await tool_manage_projects(
        context, action="task_add", description="a",
    ))["task_id"]
    await tool_manage_projects(
        context, action="task_update",
        task_id=a, description="renamed",
    )
    assert context.project_store.get_task(a)["description"] == "renamed"


async def test_task_decompose_creates_subtasks(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    r = _parse(await tool_manage_projects(context, action="task_add",
                                          description="root"))
    res = _parse(await tool_manage_projects(
        context, action="task_decompose", task_id=r["task_id"],
        subtasks=["a", "b", "c"],
    ))
    assert len(res["created"]) == 3


async def test_task_decompose_requires_subtasks(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    r = _parse(await tool_manage_projects(context, action="task_add",
                                          description="root"))
    res = await tool_manage_projects(context, action="task_decompose",
                                     task_id=r["task_id"])
    assert res.startswith("ERROR")


async def test_task_decompose_without_task_id_creates_top_level(context):
    """The natural 'fresh project, fan it out' case must work without
    requiring the LLM to first add a root task. This was the regression
    that caused Qwen to loop on `task_id is required` and create
    duplicate projects.
    """
    p = _parse(await tool_manage_projects(context, action="create", title="P"))
    res = _parse(await tool_manage_projects(
        context, action="task_decompose",
        subtasks=["research", "design", "implement"],
    ))
    assert len(res["created"]) == 3
    assert res["parent_id"] is None
    # All three are top-level (parent_id None)
    tasks = context.project_store.list_tasks(p["created"])
    assert {t["description"] for t in tasks} == {"research", "design", "implement"}
    assert all(t["parent_id"] is None for t in tasks)


async def test_task_decompose_with_project_id_as_task_id_is_treated_as_top_level(context):
    """Qwen's second guess was passing project_id in the task_id field.
    Treat that as the same shape as 'no task_id' instead of returning
    'unknown task' so the LLM doesn't loop."""
    p = _parse(await tool_manage_projects(context, action="create", title="P"))
    res = _parse(await tool_manage_projects(
        context, action="task_decompose",
        task_id=p["created"],  # the project id, not a task id
        subtasks=["a", "b"],
    ))
    assert len(res["created"]) == 2
    assert res["parent_id"] is None
    tasks = context.project_store.list_tasks(p["created"])
    assert all(t["parent_id"] is None for t in tasks)


async def test_task_decompose_with_unknown_task_id_returns_clear_error(context):
    """A genuinely bad task_id (not the project id, not in the tree)
    should still error — silent fallback would be a footgun.
    """
    _parse(await tool_manage_projects(context, action="create", title="P"))
    res = await tool_manage_projects(
        context, action="task_decompose",
        task_id="totally-fake-id",
        subtasks=["x"],
    )
    assert res.startswith("ERROR")
    assert "unknown task" in res


# --------------------------------------------------------------------- list coercion (regression)

async def test_subtasks_string_with_newlines_is_split(context):
    """The trace bug: model passed a multi-line string instead of a
    list. Iterating it as ``for desc in subtasks`` produced one task
    per character. Now it splits on newlines."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    res = _parse(await tool_manage_projects(
        context, action="task_decompose",
        subtasks="research the parsers\ndesign the streaming reader\nimplement",
    ))
    assert len(res["created"]) == 3


async def test_subtasks_string_with_commas_is_NOT_split(context):
    """Regression: 'Implement aggregation (mean, p50, p95, p99)' must
    survive as ONE task. Comma fallback was too aggressive — newlines
    are the only string delimiter we trust now.
    """
    _parse(await tool_manage_projects(context, action="create", title="P"))
    res = _parse(await tool_manage_projects(
        context, action="task_decompose",
        subtasks="Implement aggregation (mean, p50, p95, p99)",
    ))
    assert len(res["created"]) == 1
    tid = res["created"][0]
    t = context.project_store.get_task(tid)
    assert t["description"] == "Implement aggregation (mean, p50, p95, p99)"


async def test_subtasks_numbered_string_without_newlines_is_split(context):
    """Regression from the 2026-04-19 trace: XML parser flattened
    newlines and the model's numbered list collapsed into a single
    string with '1. …2. …3. …' markers. That must split into 5 tasks,
    not 1."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    blob = (
        "1. Design the log parsing strategy "
        "2. Define the Nginx log format "
        "3. Implement the parser script "
        "4. Create sample data for testing "
        "5. Test the parser and validate stats"
    )
    res = _parse(await tool_manage_projects(
        context, action="task_decompose", subtasks=blob,
    ))
    assert len(res["created"]) == 5
    descs = [context.project_store.get_task(t)["description"]
             for t in res["created"]]
    assert descs[0].startswith("Design the log parsing strategy")
    assert descs[4].startswith("Test the parser")


async def test_subtasks_bulleted_string_is_split(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    blob = "- alpha - beta - gamma"
    res = _parse(await tool_manage_projects(
        context, action="task_decompose", subtasks=blob,
    ))
    assert len(res["created"]) == 3


async def test_paren_numbered_markers_are_split(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    blob = "1) research 2) design 3) implement"
    res = _parse(await tool_manage_projects(
        context, action="task_decompose", subtasks=blob,
    ))
    assert len(res["created"]) == 3


async def test_single_list_marker_in_prose_is_NOT_split(context):
    """Exactly one marker like 'Step 1. Plan the migration' must NOT
    fire the list splitter — that's prose, not a list."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    res = _parse(await tool_manage_projects(
        context, action="task_decompose",
        subtasks="Step 1. Plan the migration before touching anything",
    ))
    assert len(res["created"]) == 1


async def test_mixed_newlines_and_numbers_prefers_newlines(context):
    """When both newlines and numbers are present, newlines win —
    they're the explicit delimiter the schema description asks for."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    blob = "1. first step\n2. second step\n3. third step"
    res = _parse(await tool_manage_projects(
        context, action="task_decompose", subtasks=blob,
    ))
    assert len(res["created"]) == 3
    descs = [context.project_store.get_task(t)["description"]
             for t in res["created"]]
    # The numeric prefix should survive when splitting on newlines
    # (since numbered prefixes are fine inside a description);
    # only the fallback marker-split strips them.
    assert descs[0] == "1. first step"


# --------------------------------------------------------------------- JSON-array-string coercion (2026-04-19 regression)

async def test_subtasks_json_array_string_is_parsed(context):
    """Exact failure mode from the trace: model sends the 'array<string>'
    param as a single JSON-encoded string. Must parse it like a list."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    res = _parse(await tool_manage_projects(
        context, action="task_decompose",
        subtasks='["Design schema", "Create generator", "Implement parser", "Aggregate stats", "Test pipeline"]',
    ))
    assert len(res["created"]) == 5
    descs = [context.project_store.get_task(t)["description"]
             for t in res["created"]]
    assert descs[0] == "Design schema"
    assert descs[-1] == "Test pipeline"


async def test_subtasks_malformed_json_array_falls_through(context):
    """If the string starts with [ but isn't valid JSON, fall back to
    the marker/newline path instead of crashing."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    # No valid JSON here — fall through, no markers → single task
    res = _parse(await tool_manage_projects(
        context, action="task_decompose",
        subtasks="[this is not valid json",
    ))
    assert len(res["created"]) == 1


async def test_task_id_as_json_array_string_of_one_is_extracted(context):
    """Trace turn 5: model passed task_id='[\"eed65da9a1e4\"]'. Must
    be treated as the singular form."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    tid = _parse(await tool_manage_projects(
        context, action="task_add", description="root",
    ))["task_id"]
    res = _parse(await tool_manage_projects(
        context, action="task_update",
        task_id=f'["{tid}"]',  # the bug shape
        status="DONE",
    ))
    assert res["count"] == 1
    assert res["updated"][0]["id"] == tid
    assert context.project_store.get_task(tid)["status"] == "DONE"


async def test_task_id_as_json_array_of_many_routes_to_task_ids(context):
    """If the model smuggled multiple ids through the singular field,
    treat them as a batch instead of erroring."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    a = _parse(await tool_manage_projects(
        context, action="task_add", description="a",
    ))["task_id"]
    b = _parse(await tool_manage_projects(
        context, action="task_add", description="b",
    ))["task_id"]
    res = _parse(await tool_manage_projects(
        context, action="task_update",
        task_id=f'["{a}", "{b}"]',
        status="DONE",
    ))
    assert res["count"] == 2
    assert context.project_store.get_task(a)["status"] == "DONE"
    assert context.project_store.get_task(b)["status"] == "DONE"


async def test_task_id_plain_string_still_works(context):
    """Don't regress the happy path."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    tid = _parse(await tool_manage_projects(
        context, action="task_add", description="x",
    ))["task_id"]
    res = _parse(await tool_manage_projects(
        context, action="task_update", task_id=tid, status="DONE",
    ))
    assert res["count"] == 1
    assert res["updated"][0]["id"] == tid


async def test_alternatives_json_array_string_also_coerced(context):
    """The coercion path applies to every array<string> param, not just
    subtasks — make sure alternatives/postconditions get the same fix."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    r = _parse(await tool_manage_projects(
        context, action="task_add",
        description="root",
        alternatives='["alt1", "alt2"]',
        postconditions='["pc1", "pc2"]',
    ))
    t = context.project_store.get_task(r["task_id"])
    assert t["alternatives"] == ["alt1", "alt2"]
    assert t["postconditions"] == ["pc1", "pc2"]


async def test_subtasks_single_phrase_with_one_comma_kept_whole_when_no_newlines(context):
    """A description like 'Implement, test, ship' should NOT be split
    when the model intended it as one task. Heuristic: only newline
    OR (comma AND >1 segments) triggers split — a single comma in a
    bare phrase is preserved by passing it inside a list."""
    _parse(await tool_manage_projects(context, action="create", title="P"))
    # Single-element list — the safe canonical form
    res = _parse(await tool_manage_projects(
        context, action="task_decompose",
        subtasks=["Implement, test, ship"],
    ))
    assert len(res["created"]) == 1


async def test_subtasks_list_with_blanks_is_filtered(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    res = _parse(await tool_manage_projects(
        context, action="task_decompose",
        subtasks=["", "real", "  ", "also real"],
    ))
    assert len(res["created"]) == 2


async def test_alternatives_and_postconditions_also_coerced(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    r = _parse(await tool_manage_projects(
        context, action="task_add",
        description="root",
        alternatives="alt one\nalt two",
        postconditions="cond a\ncond b",
    ))
    t = context.project_store.get_task(r["task_id"])
    assert t["alternatives"] == ["alt one", "alt two"]
    assert t["postconditions"] == ["cond a", "cond b"]


# --------------------------------------------------------------------- task_list slimming

async def test_task_list_returns_slim_rows_by_default(context):
    p = _parse(await tool_manage_projects(context, action="create", title="P"))
    for i in range(5):
        await tool_manage_projects(context, action="task_add",
                                   description=f"t{i}")
    res = _parse(await tool_manage_projects(context, action="task_list"))
    assert res["count"] == 5
    sample = res["tasks"][0]
    # Internal/heavy columns are NOT in the slim row
    assert "alternatives" not in sample
    assert "postconditions" not in sample
    assert "metadata_json" not in sample
    assert "created_at" not in sample
    # Useful columns ARE in the slim row
    assert "id" in sample and "description" in sample and "status" in sample


async def test_task_list_verbose_returns_full_rows(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    await tool_manage_projects(context, action="task_add", description="t1")
    res = _parse(await tool_manage_projects(
        context, action="task_list",
        metadata={"verbose": True},
    ))
    sample = res["tasks"][0]
    assert "alternatives" in sample
    assert "postconditions" in sample
    assert "created_at" in sample


async def test_task_next_returns_leaf(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    r = _parse(await tool_manage_projects(context, action="task_add",
                                          description="root"))
    c = _parse(await tool_manage_projects(context, action="task_add",
                                          description="c", parent_id=r["task_id"]))
    res = _parse(await tool_manage_projects(context, action="task_next"))
    assert res["next"]["id"] == c["task_id"]


async def test_task_next_returns_none_when_empty(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    res = _parse(await tool_manage_projects(context, action="task_next"))
    assert res["next"] is None


# --------------------------------------------------------------------- resume / status

async def test_resume_logs_event_and_sets_current(context):
    p = _parse(await tool_manage_projects(context, action="create", title="P"))
    _parse(await tool_manage_projects(context, action="exit"))
    res = _parse(await tool_manage_projects(context, action="resume",
                                            project_id=p["created"]))
    assert res["project"]["id"] == p["created"]
    assert context.current_project_id == p["created"]
    evs = context.project_store.list_events(p["created"],
                                            event_type="project_resumed")
    assert evs


async def test_status_free_chat_when_no_current(context):
    res = _parse(await tool_manage_projects(context, action="status"))
    assert res["mode"] == "free_chat"


async def test_status_project_mode_has_briefing(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    res = _parse(await tool_manage_projects(context, action="status"))
    assert res["mode"] == "project"
    assert "briefing" in res


# --------------------------------------------------------------------- artifacts / events

async def test_artifact_add_and_event_log(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    r = _parse(await tool_manage_projects(context, action="task_add",
                                          description="root"))
    aid = _parse(await tool_manage_projects(
        context, action="artifact_add",
        task_id=r["task_id"], artifact_kind="note", payload="hello",
    ))
    assert aid["artifact_id"]
    events = _parse(await tool_manage_projects(context, action="event_log",
                                               limit=50))
    types = [e["type"] for e in events["events"]]
    assert "artifact_added" in types
    assert "task_added" in types


async def test_artifact_rejects_bad_kind(context):
    _parse(await tool_manage_projects(context, action="create", title="P"))
    r = _parse(await tool_manage_projects(context, action="task_add",
                                          description="root"))
    res = await tool_manage_projects(context, action="artifact_add",
                                     task_id=r["task_id"],
                                     artifact_kind="bogus", payload="x")
    assert res.startswith("ERROR")


async def test_event_log_filter_by_type(context):
    p = _parse(await tool_manage_projects(context, action="create", title="P"))
    res = _parse(await tool_manage_projects(context, action="event_log",
                                            event_type="project_created"))
    assert all(e["type"] == "project_created" for e in res["events"])


# --------------------------------------------------------------------- promotion (suggestion-only)

async def test_promote_from_context_creates_project_and_root(context):
    res = _parse(await tool_manage_projects(
        context, action="promote_from_context",
        title="Emergent effort", goal="Get it done",
        subtasks=["step1", "step2"],
        context_summary="past chat summary",
    ))
    pid = res["promoted"]
    assert context.current_project_id == pid
    tasks = context.project_store.list_tasks(pid)
    # 1 root (goal) + 2 subtasks
    assert len(tasks) == 3
    # metadata flag is set
    proj = context.project_store.get_project(pid)
    assert proj["metadata"].get("promoted_from_context") is True
    # context_summary captured as event
    evs = context.project_store.list_events(pid, event_type="context_snapshot")
    assert evs and evs[0]["payload"]["summary"] == "past chat summary"


async def test_promote_requires_title(context):
    res = await tool_manage_projects(context, action="promote_from_context")
    assert res.startswith("ERROR")


# --------------------------------------------------------------------- registry wiring

async def test_tool_is_registered(tmp_path):
    from ghost_agent.tools.registry import TOOL_DEFINITIONS, get_available_tools

    names = [d["function"]["name"] for d in TOOL_DEFINITIONS]
    assert "manage_projects" in names

    # Build a minimal context and assert a handler is wired
    from types import SimpleNamespace
    ctx = SimpleNamespace(
        sandbox_dir=tmp_path,
        tor_proxy=None,
        args=SimpleNamespace(anonymous=True, max_context=4096, model="x",
                             temperature=0.1, default_db=None),
        llm_client=SimpleNamespace(image_gen_clients=None),
        memory_system=None, profile_memory=None, graph_memory=None,
        skill_memory=None, scratchpad=None, sandbox_manager=None,
        scheduler=None, memory_bus=None,
        project_store=ProjectStore(tmp_path / "mem"),
        current_project_id=None,
    )
    tools = get_available_tools(ctx)
    assert "manage_projects" in tools


async def test_all_advertised_actions_are_implemented(context):
    """Every action in `_ACTIONS` must be reachable without unknown-action error."""
    # We hit each at least once and ensure the handler accepted the
    # action string (even if it returned an ERROR for missing args).
    for act in _ACTIONS:
        res = await tool_manage_projects(context, action=act)
        assert not res.startswith("ERROR: unknown action"), f"action {act!r} not implemented"
