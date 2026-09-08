"""Hard delete is NOT gated on activation or on the user's wording.

Live req 1e593552 (2026-09-06): the user said "i got everything i needed …
actually delete it" about the project the conversation had just finished.
That project had auto-rolled to DONE and was no longer the active project,
and the message named it neither by id nor by its exact stored title. The
2026-06-12 delete-eligibility gate (`_delete_eligibility_error`, keyed on a
`request_start_project_id` snapshot stamped in handle_chat) therefore
refused the delete — five times in a row, across a `switch` to the project
that could not change the request-start snapshot — until the strike ledger
gave up. The gate was removed (journal §4FC): a project that resolves by id
or title is deletable whether or not it is activated.

These pins distinguish "gate removed" from "gate restored": every context
below carries the attributes the gate used to read (`request_start_project_id`
is None — exactly what handle_chat stamped when nothing was active — and a
`last_user_content` that names nothing), so a restored gate refuses and the
pins fail. Each absence assertion ("no REFUSED") is paired with the positive
twin: the row is gone, the workspace is gone, the tombstone is written.
"""

import ast
import json
import pathlib
import sys
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
    # The production shape the gate keyed on: handle_chat stamped
    # `request_start_project_id = current_project_id` (None here — nothing
    # was active when the message arrived) and `last_user_content`.
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None,
        current_project_id=None,
        request_start_project_id=None,
        last_user_content="",
        conversation_key="conv-1",
    )


async def _create(context, title):
    res = await tool_manage_projects(context, action="create",
                                     title=title, kind="CODING", goal="x")
    return json.loads(res)["created"]


def _materialise_workspace(store, pid):
    ws = pathlib.Path(store.get_project(pid)["workspace_dir"])
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "data.json").write_text("{}")
    return ws


async def test_req_1e593552_shape_non_active_unnamed_project_is_deleted(
        context, store):
    """The live refusal, replayed: the project is not active, the message
    names neither its id nor its stored title, and the request-start
    snapshot is None. Under the gate this was REFUSED five times; now it
    deletes — row, workspace and all — and writes the tombstone."""
    pid = await _create(context, "Elden Ring Build Tracker")
    ws = _materialise_workspace(store, pid)
    # The conversation finished the project: it rolled to DONE and left
    # project mode, so nothing is active when the next message arrives.
    store.update_project(pid, status="DONE")
    context.current_project_id = None
    context.request_start_project_id = None
    context.last_user_content = (
        "i got everything i needed. so the elden tracker project, "
        "actually delete it, don't archive it")
    assert pid.lower() not in context.last_user_content.lower()
    assert "elden ring build tracker" not in context.last_user_content.lower()

    res = await tool_manage_projects(context, action="delete", project_id=pid)

    assert not res.startswith("ERROR"), res
    assert "REFUSED" not in res
    body = json.loads(res)
    assert body["deleted"] is True and body["project_id"] == pid
    assert store.get_project(pid) is None
    assert not ws.exists()
    ts = store.find_deleted_similar("Elden Ring Build Tracker")
    assert ts is not None and ts["id"] == pid


async def test_delete_by_title_of_a_non_active_project(context, store):
    """Same world, resolved by title instead of id (the two spellings the
    model uses); title resolution must not re-introduce a gate."""
    pid = await _create(context, "Side Project")
    context.current_project_id = None
    context.request_start_project_id = None
    context.last_user_content = "actually delete it"
    res = await tool_manage_projects(context, action="delete",
                                     title="Side Project")
    assert json.loads(res)["deleted"] is True
    assert store.get_project(pid) is None


async def test_project_created_mid_request_is_deletable(context, store):
    """The churn scenario the gate was written for (delete the start
    project, create a new one, delete THAT). The gate refused the second
    delete; the operator's rule is that any resolvable project is
    deletable, so it now succeeds. Both deletes are observable."""
    start_pid = await _create(context, "The Infinite Archive")
    context.request_start_project_id = start_pid
    context.last_user_content = (
        "i don't really like it, delete it an make something else")
    res = await tool_manage_projects(context, action="delete",
                                     project_id=start_pid)
    assert json.loads(res)["deleted"] is True

    new_pid = await _create(context, "The Algorithmic Garden")
    assert store.get_project(new_pid) is not None
    res = await tool_manage_projects(context, action="delete",
                                     project_id=new_pid)
    assert not res.startswith("ERROR"), res
    assert "REFUSED" not in res and "BUILD" not in res
    assert json.loads(res)["deleted"] is True
    assert store.get_project(new_pid) is None


async def test_delete_after_switch_clears_the_activation(context, store):
    """What the live agent tried on retry: switch to the project, then
    delete. The delete succeeds and leaves project mode (current cleared),
    so the next turn does not resolve against a deleted id."""
    pid = await _create(context, "Switched Then Deleted")
    context.request_start_project_id = None
    context.last_user_content = "delete it"
    res = await tool_manage_projects(context, action="switch", project_id=pid)
    assert not res.startswith("ERROR"), res
    assert context.current_project_id == pid
    res = await tool_manage_projects(context, action="delete", project_id=pid)
    assert json.loads(res)["deleted"] is True
    assert store.get_project(pid) is None
    assert getattr(context, "current_project_id", None) is None


async def test_delete_of_a_missing_project_still_fails_loudly(context, store):
    """Control: removing the gate must not have removed the other refusal
    on this path — an unresolvable id is still a loud error, nothing is
    deleted, and the message says so."""
    context.request_start_project_id = None
    context.last_user_content = "delete it"
    res = await tool_manage_projects(context, action="delete",
                                     project_id="000000000000")
    assert res.startswith("ERROR") and "NOTHING was deleted" in res


async def test_archive_is_unchanged(context, store):
    """Control: archive (soft, reversible) behaves exactly as before."""
    pid = await _create(context, "Side Quest")
    context.request_start_project_id = None
    context.last_user_content = "delete it"
    res = await tool_manage_projects(context, action="archive", project_id=pid)
    assert json.loads(res)["archived"] is True
    assert (store.get_project(pid) or {}).get("status") == "ARCHIVED"


# ── enumeration: the snapshot the gate keyed on is gone from the tree ──

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


def _attribute_sites(name):
    hits = []
    for path in sorted(_SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == name:
                hits.append(f"{path.relative_to(_SRC.parent)}:{node.lineno}")
            elif isinstance(node, ast.Constant) and node.value == name:
                # getattr/hasattr/setattr(context, "<name>") spellings
                hits.append(f"{path.relative_to(_SRC.parent)}:{node.lineno}")
    return hits


def test_no_code_reads_or_writes_the_request_start_snapshot():
    """The gate's only input was `context.request_start_project_id`, stamped
    once per request in handle_chat and read nowhere else. Both ends were
    removed together; a site that re-stamps or re-reads it (attribute or
    getattr/hasattr string) is the gate coming back under another name."""
    assert _attribute_sites("request_start_project_id") == []


def test_the_enumeration_can_see_an_attribute_site(tmp_path, monkeypatch):
    """The instrument can fail: point it at a tree that stamps the
    attribute both ways and it reports both lines."""
    fake = tmp_path / "src" / "ghost_agent"
    fake.mkdir(parents=True)
    (fake / "m.py").write_text(
        "def f(ctx):\n"
        "    ctx.request_start_project_id = 1\n"
        "    return getattr(ctx, 'request_start_project_id', None)\n")
    monkeypatch.setattr(sys.modules[__name__], "_SRC", fake)
    hits = _attribute_sites("request_start_project_id")
    assert hits == ["ghost_agent/m.py:2", "ghost_agent/m.py:3"]
