"""§4SU Phase 4 — the model-facing surface: a smaller tool definition, a
`replan` action (a new plan for the SAME project), `list` returns display
lines, a plan / preview hands the turn to the user (the trailing-promise
guard lets "reply yes and I'll start" stand), a finished project's briefing,
the WAITING / RUNNING status aliases, and one `refused` list instead of the
legacy per-gate reply keys."""
import json
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import tool_manage_projects, MANAGE_PROJECTS_TOOL_DEF
from tests.test_4qa_project_task_integrity import _ctx
from tests.test_4su_phase0 import owner_turn  # noqa: F401  (fixture)


async def _proj(tmp_path, title="Garden planner", goal="Plan the garden"):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _ctx(store, tmp_path)
    await tool_manage_projects(ctx, action="create", title=title, goal=goal)
    return ctx, store, ctx.current_project_id


def _j(out):
    s = str(getattr(out, "text", None) or out)
    return json.loads(s[s.index("{"):])


# ───────────────────────────── the tool definition ─────────────────────────────

def test_the_definition_is_smaller_and_shows_five_states():
    fn = MANAGE_PROJECTS_TOOL_DEF["function"]
    props = fn["parameters"]["properties"]
    assert len(json.dumps(MANAGE_PROJECTS_TOOL_DEF)) < 7000
    assert "replan" in props["action"]["enum"]
    assert set(props["status"]["enum"]) == {"PENDING", "DONE", "FAILED", "WAITING"}
    for gone in ("not_model_data", "dependency_type", "alternatives", "max_topics",
                 "artifact_kind", "event_type", "query"):
        assert gone not in props


def test_every_visible_action_is_handled():
    from ghost_agent.tools.projects import _VISIBLE_ACTIONS, _ACTIONS
    assert set(_VISIBLE_ACTIONS) <= set(_ACTIONS)


# ───────────────────────────── replan ─────────────────────────────

async def test_replan_keeps_done_work_and_replaces_the_open_plan(tmp_path):
    ctx, store, pid = await _proj(tmp_path)
    done = store.add_task(pid, "Measure the beds")
    store.update_task(done, status="DONE", result_summary="4 beds, 2x1 m")
    store.add_task(pid, "Buy seeds")
    parent = store.add_task(pid, "Old phase")
    store.add_task(pid, "old child", parent_id=parent)
    out = _j(await tool_manage_projects(ctx, action="replan",
                                        subtasks=["Pick vegetables", "Draw the layout"]))
    assert out["replanned"] == {"removed_open_tasks": 3, "kept_done_tasks": 1}
    assert next(iter(out)) == "plan_shown"
    left = {t["description"]: t["status"] for t in store.list_tasks(pid)}
    assert left == {"Measure the beds": "DONE", "Pick vegetables": "PENDING", "Draw the layout": "PENDING"}
    assert "REPLANNED" in store.get_ledger(pid)
    ev = [e for e in store.list_events(pid, limit=50) if e.get("type") == "project_replanned"]
    assert ev and ev[0]["payload"]["added"] == 2


async def test_replan_keeps_an_open_parent_with_done_work_under_it(tmp_path):
    ctx, store, pid = await _proj(tmp_path)
    parent = store.add_task(pid, "Phase 1")
    kid = store.add_task(pid, "built part", parent_id=parent)
    store.update_task(kid, status="DONE", result_summary="ok")
    await tool_manage_projects(ctx, action="replan", subtasks=["New step"])
    descs = {t["description"] for t in store.list_tasks(pid)}
    assert {"Phase 1", "built part", "New step"} <= descs


async def test_replan_refuses_while_a_task_runs_and_changes_nothing(tmp_path):
    ctx, store, pid = await _proj(tmp_path)
    t = store.add_task(pid, "Running")
    store.update_task(t, status="IN_PROGRESS")
    out = str(await tool_manage_projects(ctx, action="replan", subtasks=["x"]))
    assert "NOT replanned" in out and "running" in out
    assert [r["id"] for r in store.list_tasks(pid)] == [t]


async def test_a_blank_new_plan_deletes_nothing(tmp_path):
    ctx, store, pid = await _proj(tmp_path)
    keep = store.add_task(pid, "Buy seeds")
    out = str(await tool_manage_projects(ctx, action="replan", subtasks=["  ", ""]))
    assert "nothing changed" in out.lower()
    assert store.get_task(keep) is not None


async def test_replan_of_a_single_file_goal_is_a_normal_replan(tmp_path):
    # §4UI: the goal-text single-file refusal is gone — the model's plan stands
    ctx, store, pid = await _proj(tmp_path, title="Tetris clone",
                                  goal="a single-file HTML tetris game in one index.html")
    store.add_task(pid, "Build it")
    out = str(await tool_manage_projects(ctx, action="replan", subtasks=["Board", "Pieces"]))
    assert "NOT replanned" not in out and "SINGLE-FILE" not in out
    descs = [t["description"] for t in store.list_tasks(pid)]
    assert "Board" in descs and "Pieces" in descs


async def test_replan_never_finishes_the_project_midway(tmp_path):
    # r1 C1/C2: deleting the last open task first rolled the project up to
    # DONE — the workspace sweep ran and the owner's constraints were retired
    ctx, store, pid = await _proj(tmp_path)
    store.update_project(pid, metadata={"constraints": ["use metric units only"]})
    done = store.add_task(pid, "Measure")
    store.add_task(pid, "Buy seeds")
    store.update_task(done, status="DONE", result_summary="ok")
    assert store.get_project(pid)["status"] == "ACTIVE"
    fired = []
    store.on_project_done = lambda *a, **k: fired.append(a)
    await tool_manage_projects(ctx, action="replan", subtasks=["Pick vegetables"])
    assert not fired
    assert store.get_project(pid)["status"] == "ACTIVE"
    assert store.get_project(pid)["metadata"].get("constraints") == ["use metric units only"]
    assert not [e for e in store.list_events(pid, limit=50) if e.get("type") == "project_auto_rollup"]


async def test_a_new_plan_like_the_old_one_is_laid_down_whole(tmp_path):
    # r1 M1: the dedupe ran against the tasks being replaced
    ctx, store, pid = await _proj(tmp_path)
    store.add_task(pid, "Buy seeds")
    out = _j(await tool_manage_projects(ctx, action="replan", subtasks=["Buy seeds", "Water plan"]))
    assert out["replanned"]["removed_open_tasks"] == 1
    assert sorted(t["description"] for t in store.list_tasks(pid)) == ["Buy seeds", "Water plan"]


async def test_a_kept_parent_left_with_done_children_is_closed(tmp_path):
    # r1 M2: nothing ever closed it, the project stayed ACTIVE with no work
    ctx, store, pid = await _proj(tmp_path)
    ph = store.add_task(pid, "Phase 1")
    k = store.add_task(pid, "built part", parent_id=ph)
    store.update_task(k, status="DONE", result_summary="built part, checked")
    store.add_task(pid, "open part", parent_id=ph)
    await tool_manage_projects(ctx, action="replan", subtasks=["New step"])
    assert store.get_task(ph)["status"] == "DONE"


async def test_replan_refuses_an_archived_project(tmp_path):
    ctx, store, pid = await _proj(tmp_path)
    keep = store.add_task(pid, "Buy seeds")
    store.update_project(pid, status="ARCHIVED")
    out = str(await tool_manage_projects(ctx, action="replan", project_id=pid, subtasks=["x"]))
    assert "ARCHIVED" in out and store.get_task(keep) is not None


async def test_a_stale_in_progress_task_does_not_block_replan(tmp_path):
    import sqlite3
    from pathlib import Path
    ctx, store, pid = await _proj(tmp_path)
    t = store.add_task(pid, "Left running")
    store.update_task(t, status="IN_PROGRESS")
    db = next(Path(tmp_path / "m").rglob("*.db"))
    con = sqlite3.connect(db)
    con.execute("UPDATE tasks SET updated_at = updated_at - 3600 WHERE id = ?", (t,))
    con.commit()
    con.close()
    out = _j(await tool_manage_projects(ctx, action="replan", subtasks=["x"]))
    assert out["replanned"]["removed_open_tasks"] == 1 and store.get_task(t) is None


async def test_replan_needs_the_new_plan(tmp_path):
    ctx, store, pid = await _proj(tmp_path)
    keep = store.add_task(pid, "Buy seeds")
    out = str(await tool_manage_projects(ctx, action="replan"))
    assert "required" in out and store.get_task(keep) is not None


# ───────────────────────────── list ─────────────────────────────

async def test_list_returns_display_lines_not_rows(tmp_path):
    ctx, store, pid = await _proj(tmp_path)
    out = _j(await tool_manage_projects(ctx, action="list"))
    assert "projects" not in out
    assert out["display"].startswith(pid) and "<- ACTIVE" in out["display"]
    assert out["count"] == 1
    full = _j(await tool_manage_projects(ctx, action="list", metadata={"verbose": True}))
    assert full["projects"][0]["id"] == pid


# ───────────────────────────── the turn ends on a plan / a preview ─────────────────────────────

async def test_create_and_decompose_hand_the_turn_to_the_user(tmp_path):
    from ghost_agent.core.agent import _awaits_owner_go_ahead
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    ctx = _ctx(store, tmp_path)
    created = await tool_manage_projects(ctx, action="create", title="Garden", goal="g")
    assert _awaits_owner_go_ahead({"name": "manage_projects", "content": created})
    dec = await tool_manage_projects(ctx, action="task_decompose", subtasks=["a", "b"])
    assert _awaits_owner_go_ahead({"name": "manage_projects", "content": dec})
    listed = await tool_manage_projects(ctx, action="list")
    assert not _awaits_owner_go_ahead({"name": "manage_projects", "content": listed})


def test_only_the_tool_s_own_first_key_counts():
    from ghost_agent.core.agent import _awaits_owner_go_ahead
    assert _awaits_owner_go_ahead({"name": "manage_projects",
                                   "content": 'OK: {"plan_shown": true, "x": 1}'})
    # §4UK: lifecycle previews are gone — their keys no longer count
    assert not _awaits_owner_go_ahead({"name": "manage_projects",
                                       "content": 'OK: {"confirmation_needed": true, "x": 1}'})
    # the marker quoted inside a field is not the payload's first key
    assert not _awaits_owner_go_ahead({"name": "manage_projects",
                                       "content": '{"note": "{\\"plan_shown\\": true"}'})
    # a later key is not the marker either
    assert not _awaits_owner_go_ahead({"name": "manage_projects",
                                       "content": '{"created": [], "plan_shown": true}'})
    assert not _awaits_owner_go_ahead({"name": "file_system",
                                       "content": '{"plan_shown": true}'})
    assert not _awaits_owner_go_ahead({"name": "manage_projects",
                                       "content": SimpleNamespace(reason_code="confirm_dead_end", text="")})


def test_the_guard_exemption_reads_the_latest_row(monkeypatch):
    from ghost_agent.core.agent import _awaits_new_message_confirm
    plan = {"name": "manage_projects", "content": '{"plan_shown": true}'}
    other = {"name": "web_search", "content": "results"}
    assert _awaits_new_message_confirm([other, plan])
    assert not _awaits_new_message_confirm([plan, other])


# ───────────────────────────── the briefing of a finished project ─────────────────────────────

@pytest.mark.parametrize("status,word", [("DONE", "FINISHED"), ("ARCHIVED", "ARCHIVED")])
def test_a_finished_project_s_briefing_says_how_to_reopen(tmp_path, status, word):
    from ghost_agent.core.prompts import build_project_briefing
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("Garden")
    t = store.add_task(pid, "Plan")
    store.update_task(t, status="DONE", result_summary="planned")
    store.update_project(pid, status=status)
    out = build_project_briefing(store, pid)
    assert f"This project is {word}" in out
    assert "ALREADY ACTIVE" not in out and "ONE TASK AT A TIME" not in out
    assert ("resume it first" in out) == (status == "ARCHIVED")


def test_an_active_project_keeps_the_pacing_rule(tmp_path):
    from ghost_agent.core.prompts import build_project_briefing
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("Garden")
    store.add_task(pid, "Plan")
    out = build_project_briefing(store, pid)
    assert "ALREADY ACTIVE" in out and "FINISHED" not in out


async def test_a_task_added_to_a_finished_project_reopens_it(tmp_path):
    # what the finished briefing promises
    ctx, store, pid = await _proj(tmp_path)
    store.update_project(pid, status="DONE")
    await tool_manage_projects(ctx, action="task_add", description="One more bed")
    assert store.get_project(pid)["status"] == "ACTIVE"


# ───────────────────────────── statuses + replies ─────────────────────────────

async def test_waiting_and_running_are_status_aliases(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _proj(tmp_path)
    t = store.add_task(pid, "Ask the neighbour")
    await tool_manage_projects(ctx, action="task_update", task_id=t, status="WAITING")
    row = store.get_task(t)
    assert row["status"] == "NEEDS_USER" and row["wait_on"] == "owner"


async def test_a_refused_close_has_one_refused_list_and_no_legacy_keys(tmp_path, owner_turn):
    owner_turn("a1b2c3d4")
    ctx, store, pid = await _proj(tmp_path)
    t = store.add_task(pid, "Write the plan → plan.md")
    out = _j(await tool_manage_projects(ctx, action="task_update", task_id=t,
                                        status="DONE", result="wrote it"))
    assert out["refused"] and out["refused"][0]["gate"] == "deliverable"
    assert "status=done AND the result" in out["agent_instruction_refused"]
    for legacy in ("undelivered", "waiting_on_dependencies", "unrun_experiments", "still_failing",
                   "constraint_violations", "gated_constraints", "gated_unverified",
                   "agent_instruction_undelivered"):
        assert legacy not in out


def test_a_refused_payload_with_nothing_updated_is_a_rejection():
    from ghost_agent.tools.projects import _ok
    assert _ok({"updated": [], "count": 0, "refused": [{"task_id": "t", "gate": "x"}]}).is_failure
    assert not _ok({"updated": [{"id": "t"}], "count": 1,
                    "refused": [{"task_id": "u", "gate": "x"}]}).is_failure


# ───────────────────────────── review r3 ─────────────────────────────

async def test_replan_takes_no_task_id(tmp_path):
    ctx, store, pid = await _proj(tmp_path)
    keep = store.add_task(pid, "Buy seeds")
    out = str(await tool_manage_projects(ctx, action="replan", task_id=keep, subtasks=["x"]))
    assert "takes no task_id" in out and store.get_task(keep) is not None


async def test_replan_waits_for_a_live_unattended_step(tmp_path, monkeypatch):
    import ghost_agent.core.project_advancer as PA
    ctx, store, pid = await _proj(tmp_path)
    t = store.add_task(pid, "Building")
    store.update_task(t, status="IN_PROGRESS")
    import sqlite3
    from pathlib import Path
    db = next(Path(tmp_path / "m").rglob("*.db"))
    con = sqlite3.connect(db)
    con.execute("UPDATE tasks SET updated_at = updated_at - 3600 WHERE id = ?", (t,))
    con.commit()
    con.close()
    monkeypatch.setattr(PA, "unattended_step_live", lambda s, p: "20 min")
    out = str(await tool_manage_projects(ctx, action="replan", subtasks=["x"]))
    assert "NOT replanned" in out and store.get_task(t) is not None


async def test_a_replan_that_fails_midway_takes_its_new_tasks_back(tmp_path, monkeypatch):
    from ghost_agent.core import planning as PL
    ctx, store, pid = await _proj(tmp_path)
    keep = store.add_task(pid, "Buy seeds")
    real = PL.ProjectPlan.add_task
    n = {"i": 0}

    def flaky(self, desc, *a, **k):
        n["i"] += 1
        if n["i"] == 2:
            raise ValueError("bad task")
        return real(self, desc, *a, **k)
    monkeypatch.setattr(PL.ProjectPlan, "add_task", flaky)
    out = str(await tool_manage_projects(ctx, action="replan", subtasks=["one", "two"]))
    assert "NOT replanned" in out
    assert [t["description"] for t in store.list_tasks(pid)] == ["Buy seeds"]
