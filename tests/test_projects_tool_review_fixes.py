"""Regression tests for the 2026-07-20 three-stack review fixes in
tools/projects.py (tool-boundary half; workspace_cleanup carries its own
containment check separately).

Covers:
  - C1: action=cleanup with a traversal project_id is rejected by store
    validation before tidy runs (no deletion outside the sandbox)
  - task_update with `result` but no `status` persists the result_summary
    (never a success-shaped no-op) and the gate-recovery text says to
    re-pass status=done
  - bulk task_update: a fileless sibling is NOT refused on the first
    task's constraint-audit violation (no files → skip the judgment gate)
  - reconcile_conversation with a missing/evicted sentinel fails CLOSED:
    the stale process-global project is parked unless the message names it
  - metadata that parses to a non-dict (e.g. '["x"]') is rejected
  - get/switch/resume resolve an explicit `title`
  - a case-mangled parent_id is canonicalized (sibling-duplicate guard)
"""

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
    tool_manage_projects,
    conversation_fingerprint,
    reconcile_conversation,
)


@pytest.fixture
def context(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    scratchpad = Scratchpad(persist_path=tmp_path / "sp.db")
    return SimpleNamespace(
        project_store=store,
        scratchpad=scratchpad,
        graph_memory=None,
        current_project_id=None,
    )


def _parse(s: str):
    return json.loads(s)


async def _create(context, title="P", **kw):
    res = await tool_manage_projects(context, action="create", title=title, **kw)
    return _parse(res)["created"]


# ------------------------------------------------------------ C1: cleanup

async def test_cleanup_traversal_id_rejected_and_deletes_nothing(context, tmp_path):
    """`cleanup` was the one destructive action that skipped
    `_resolve_project_ref` — a raw `project_id="../.."` resolved to the
    sandbox PARENT and the tidy deleted files there."""
    await _create(context, title="Tidy Me")
    # Files at the sandbox parent (= what <sb>/projects/../.. resolves to):
    # both are in tidy's deletion categories (media debris / dotfile).
    victim_png = tmp_path / "screenshot.png"
    victim_png.write_bytes(b"p" * 16)
    victim_dot = tmp_path / ".secret"
    victim_dot.write_text("keep me")

    res = await tool_manage_projects(context, action="cleanup",
                                     project_id="../..")
    assert res.startswith("ERROR")
    assert "not found" in res
    assert victim_png.exists()
    assert victim_dot.exists()


async def test_cleanup_valid_id_still_tidies(context):
    store = context.project_store
    pid = await _create(context, title="Tidy Valid")
    ws = Path(store.sandbox_root) / "projects" / pid
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "shot.png").write_bytes(b"p" * 8)
    res = _parse(await tool_manage_projects(context, action="cleanup",
                                            project_id=pid))
    assert res["deleted"] == ["shot.png"]
    assert not (ws / "shot.png").exists()


# --------------------------------------- task_update: result without status

async def test_result_without_status_persists_not_silent_noop(context):
    store = context.project_store
    await _create(context, title="Evidence")
    tid = _parse(await tool_manage_projects(
        context, action="task_add", description="write the report"))["task_id"]

    res = _parse(await tool_manage_projects(
        context, action="task_update", task_id=tid,
        result="report written, 3 sections"))
    # Something was actually WRITTEN — never {"count": 1} with no effect.
    assert store.get_task(tid)["result_summary"] == "report written, 3 sections"
    # Status untouched (a result alone does not close a task) and the
    # response says so, steering the model to re-pass status=done.
    assert store.get_task(tid)["status"] == "PENDING"
    assert res["result_recorded_status_unchanged"] == [tid]
    assert "status=done" in res["agent_instruction_result_only"]


async def test_gate_recovery_loop_closes_with_result_then_done(context):
    """The observed spin: DONE refused by the visual gate → model re-calls
    with `result` only (as the old instruction said) → no-op → gate refuses
    again, forever. Now: the result persists, the instruction says to
    re-pass status=done, and the follow-up DONE clears the gate."""
    store = context.project_store
    await _create(context, title="Game")
    tid = _parse(await tool_manage_projects(
        context, action="task_add",
        description="build the webgl hud overlay"))["task_id"]

    # 1) DONE with no evidence → gated, and the recovery text now includes
    #    the status=done re-pass.
    g = _parse(await tool_manage_projects(
        context, action="task_update", task_id=tid, status="DONE"))
    assert g["gated_unverified"] == [tid]
    assert "status=done" in g["agent_instruction"]

    # 2) result-only call persists the evidence.
    _parse(await tool_manage_projects(
        context, action="task_update", task_id=tid,
        result="screenshot verified: HUD renders over the canvas"))
    assert store.get_task(tid)["result_summary"]

    # 3) re-pass status=done → the stored summary clears the gate.
    d = _parse(await tool_manage_projects(
        context, action="task_update", task_id=tid, status="DONE"))
    assert d["count"] == 1
    assert store.get_task(tid)["status"] == "DONE"


async def test_task_update_with_nothing_to_do_is_not_reported_written(context):
    store = context.project_store
    await _create(context, title="Noop")
    tid = _parse(await tool_manage_projects(
        context, action="task_add", description="a task"))["task_id"]
    res = _parse(await tool_manage_projects(
        context, action="task_update", task_id=tid, result="   "))
    # Blank result + no status: nothing written, and no "result recorded"
    # claim either.
    assert "result_recorded_status_unchanged" not in res
    assert store.get_task(tid)["result_summary"] in ("", None)


# ------------------------------------ bulk task_update: fileless sibling

async def test_bulk_fileless_sibling_not_refused_on_first_tasks_audit(
        context, monkeypatch):
    """The one-audit-per-call cache leaked the FIRST task's violation onto
    every later task — including a fileless sibling whose contract is
    'no files → skip the judgment gate'."""
    import ghost_agent.core.build_gates as bg

    async def _always_violates(ctx, constraints, files, is_background=False):
        return (False, "audit: a.txt imports libX")

    monkeypatch.setattr(bg, "constraint_gate", _always_violates)

    store = context.project_store
    pid = await _create(context, title="Gated Build",
                        constraints=["never use libX"],
                        subtasks=["Build part A", "Build part B"])
    tasks = store.list_tasks(pid)
    a = next(t["id"] for t in tasks if "part A" in t["description"])
    b = next(t["id"] for t in tasks if "part B" in t["description"])

    # Task A has an attributable file (registered artifact); B has none.
    ws = Path(store.sandbox_root) / "projects" / pid
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "a.txt").write_text("import libX")
    store.add_artifact(a, "file", "a.txt")

    res = _parse(await tool_manage_projects(
        context, action="task_update", task_ids=[a, b], status="DONE",
        result="both parts complete per the constraint"))

    # A (audited, violating) is refused; fileless B goes DONE.
    assert res["constraint_violations"] == [a]
    assert store.get_task(a)["status"] != "DONE"
    assert store.get_task(b)["status"] == "DONE"
    # The violation recovery text also names the status=done re-pass.
    assert "status=done" in res["agent_instruction_violation"]


# ------------------------------------------- reconcile fails CLOSED

CONV_A = conversation_fingerprint(
    [{"role": "user", "content": "let's build a pet ai"}])
CONV_B = conversation_fingerprint(
    [{"role": "user", "content": "unrelated postgres question"}])


async def test_reconcile_missing_sentinel_parks_stale_project(context):
    """LRU eviction of the binding sentinels must not leave the previous
    conversation's project active for the next request (fail CLOSED)."""
    reconcile_conversation(context, CONV_A)
    pid = await _create(context, title="PetAI")
    # Simulate the 50-entry/24h scratchpad eviction of both sentinels.
    context.scratchpad.delete("__current_project__")
    context.scratchpad.delete("__current_project_conv__")
    context.current_project_id = pid          # stale process-global
    context.last_user_content = "what's the weather like"

    reconcile_conversation(context, CONV_B)
    assert context.current_project_id is None


async def test_reconcile_missing_sentinel_keeps_explicitly_named_project(context):
    reconcile_conversation(context, CONV_A)
    pid = await _create(context, title="PetAI")
    context.scratchpad.delete("__current_project__")
    context.scratchpad.delete("__current_project_conv__")
    context.current_project_id = pid
    context.last_user_content = "proceed with task 3 of the PetAI project"

    reconcile_conversation(context, CONV_B)
    # Escape hatch: named → stays active AND is re-bound for this request.
    assert context.current_project_id == pid
    assert context.scratchpad.get("__current_project__") == pid
    assert context.scratchpad.get("__current_project_conv__") == CONV_B


async def test_reconcile_no_sentinel_no_current_is_noop(context):
    context.last_user_content = ""
    reconcile_conversation(context, CONV_B)   # must not raise
    assert context.current_project_id is None


# ---------------------------------------------- metadata must be a dict

async def test_metadata_array_string_rejected(context):
    store = context.project_store
    pid = await _create(context, title="Meta")
    res = await tool_manage_projects(context, action="update",
                                     project_id=pid, metadata='["x"]')
    assert res.startswith("ERROR")
    assert "object" in res
    assert isinstance(store.get_project(pid)["metadata"], dict)


async def test_metadata_real_list_rejected(context):
    pid = await _create(context, title="Meta2")
    res = await tool_manage_projects(context, action="update",
                                     project_id=pid, metadata=["x", "y"])
    assert res.startswith("ERROR")


async def test_metadata_dict_string_still_accepted(context):
    store = context.project_store
    pid = await _create(context, title="Meta3")
    res = await tool_manage_projects(context, action="update",
                                     project_id=pid,
                                     metadata='{"steps_note": "ok"}')
    assert not res.startswith("ERROR")
    assert store.get_project(pid)["metadata"].get("steps_note") == "ok"


# ------------------------------------- title resolves on get/switch/resume

async def test_get_resolves_explicit_title(context):
    falcon = await _create(context, title="Falcon")
    await _create(context, title="Owl")       # current is now Owl
    res = _parse(await tool_manage_projects(context, action="get",
                                            title="falcon"))
    assert res["id"] == falcon


async def test_switch_resolves_explicit_title(context):
    falcon = await _create(context, title="Falcon")
    await _create(context, title="Owl")
    res = _parse(await tool_manage_projects(context, action="switch",
                                            title="Falcon"))
    assert res["switched_to"] == falcon
    assert context.current_project_id == falcon


async def test_resume_resolves_explicit_title_and_unarchives(context):
    store = context.project_store
    falcon = await _create(context, title="Falcon")
    await tool_manage_projects(context, action="archive", project_id=falcon)
    await _create(context, title="Owl")       # a different active project
    res = _parse(await tool_manage_projects(context, action="resume",
                                            title="Falcon"))
    assert res["project"]["id"] == falcon
    assert store.get_project(falcon)["status"] == "ACTIVE"
    assert context.current_project_id == falcon


async def test_resume_unknown_title_errors(context):
    await _create(context, title="Only One")
    res = await tool_manage_projects(context, action="resume",
                                     title="Nope Never")
    assert res.startswith("ERROR")


# ---------------------------------------------- parent_id canonicalized

async def test_mangled_parent_id_still_hits_sibling_duplicate_guard(context):
    store = context.project_store
    await _create(context, title="Tree")
    parent = _parse(await tool_manage_projects(
        context, action="task_add", description="parent shell"))["task_id"]
    child = _parse(await tool_manage_projects(
        context, action="task_add", description="build the child",
        parent_id=parent))["task_id"]

    # Same description under a case-mangled echo of the parent id: the
    # guard compares parent ids, so a raw mangled id used to slip past it
    # and create a duplicate sibling.
    res = _parse(await tool_manage_projects(
        context, action="task_add", description="build the child",
        parent_id=parent.upper()))
    assert res.get("refused") is True
    assert res["existing_task_id"] == child
    # And no duplicate row landed.
    siblings = [t for t in store.list_tasks(context.current_project_id)
                if t.get("parent_id") == parent
                and t["description"] == "build the child"]
    assert len(siblings) == 1
