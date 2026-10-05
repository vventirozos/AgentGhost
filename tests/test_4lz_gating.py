"""§4LZ lens A — who may do what: behaviour pins."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.utils.logging import request_id_context, request_kind


@pytest.mark.parametrize("rid,kind", [
    ("probe-1", "probe"), ("sched-nightly", "background"), ("sub-w", "background"),
    ("sim-abc", "background"), ("job-123", "job"), ("bench-x", "test"), ("replay-y", "test"),
    ("SYSTEM", "system"), ("", "system"), ("39c394ca", "owner"),
])
def test_one_classification_for_every_request(rid, kind):
    assert request_kind(rid) == kind


# ── F1: self-play cannot touch production macros or page the owner ──

def test_self_play_denies_what_replay_denies_for_state_and_outward_tools():
    from ghost_agent.core.dream import SELF_PLAY_FORBIDDEN_TOOLS
    for t in ("manage_composed_skills", "notify_operator", "jobs", "rotate_secrets", "knowledge_base"):
        assert t in SELF_PLAY_FORBIDDEN_TOOLS


async def test_a_self_play_turn_cannot_change_the_macro_store(tmp_path):
    from ghost_agent.tools.composed_skills import _registry_from_context, tool_manage_composed_skills
    ctx = type("C", (), {"memory_dir": tmp_path, "sandbox_dir": tmp_path})()
    _registry_from_context(ctx).compile_from_pattern("keep", [{"tool": "web_search", "params": {"query": "$q"}}], "t")
    tok = request_id_context.set("sim-0001")
    try:
        await tool_manage_composed_skills(context=ctx, action="delete", name="keep")
    finally:
        request_id_context.reset(tok)
    assert "keep" in _registry_from_context(ctx).skills


@pytest.mark.parametrize("rid,refused", [("sched-x", True), ("sim-1", True), ("bench-2", True),
                                         ("probe-3", True), ("39c394ca", False), ("job-4", False)])
def test_the_skill_store_takes_writes_only_from_the_owner(rid, refused):
    from ghost_agent.tools.acquired_skills import _not_an_owner_write
    tok = request_id_context.set(rid)
    try:
        assert _not_an_owner_write() is refused
    finally:
        request_id_context.reset(tok)


# ── F2: a job wakes with its starter's class; a probe's job never wakes ──

def _ctx():
    return SimpleNamespace(llm_client=MagicMock(), args=SimpleNamespace(model="m"),
                           sandbox_manager=None, activity_log=None, agent=MagicMock())


@pytest.mark.parametrize("starter,wakes,prefix", [
    ("probe-9", False, None), ("sched-daily", True, "sub-job-"), ("sim-1", True, "sub-job-"),
    ("39c394ca", True, "job-"), ("", True, "job-"),
])
async def test_a_job_wakes_as_its_starter(monkeypatch, starter, wakes, prefix):
    import ghost_agent.main as MAIN
    import ghost_agent.tools.tasks as T
    monkeypatch.setattr(T, "should_defer_scheduled_task", lambda llm: False)
    seen = []

    async def fake_fg(ctx, body, rid):
        seen.append(rid)
        return "done", None, None
    monkeypatch.setattr(MAIN, "_handle_chat_foreground", fake_fg)
    MAIN._RESUMED_JOBS.clear()
    MAIN._resume_times.clear()
    entry = {"id": f"job-{abs(hash(starter)) % 10**8:08d}", "state": "done", "exit_code": 0,
             "command": "sleep 1", "started_by": starter}
    await MAIN._resume_after_job(_ctx(), entry)
    assert bool(seen) is wakes
    if wakes:
        assert seen[0].startswith(prefix)
        assert request_kind(seen[0]) == ("background" if prefix == "sub-job-" else "job")


# ── F3: a probe never pages the owner or spends the quota ──

async def test_a_probe_notification_is_a_dry_run(monkeypatch):
    import ghost_agent.tools.notify_tool as N
    log = MagicMock()
    monkeypatch.setattr(N, "get_activity_log", lambda ctx: log)
    tok = request_id_context.set("probe-n")
    try:
        out = await N.tool_notify_operator(message="disk almost full", context=MagicMock())
    finally:
        request_id_context.reset(tok)
    assert "not sent" in out.lower() and "disk almost full" in out
    log.record.assert_not_called()


# ── F4: a probe's fork is marked, and the owner never inherits it ──

async def test_a_probe_fork_is_marked_and_skipped_by_the_owner(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.tools.projects import tool_manage_projects
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    ctx = SimpleNamespace(project_store=store, scratchpad=None, graph_memory=None,
                          workspace_model=None, current_project_id=None, last_user_content="")
    parent = store.create_project("Coach")
    store.update_project(parent, status="DONE")
    with store._lock, store._connect() as conn:      # RELEASED is earned via release; set directly here
        conn.execute("UPDATE projects SET status='RELEASED' WHERE id=?", (parent,))
    tok = request_id_context.set("probe-f")
    try:
        await tool_manage_projects(ctx, action="create_version", project_id=parent, description="x")
    finally:
        request_id_context.reset(tok)
    kids = store.list_children(parent)
    assert kids and (kids[0].get("metadata") or {}).get("probe_created")
    tok = request_id_context.set("req-owner")
    try:
        res = json.loads(await tool_manage_projects(ctx, action="create_version", project_id=parent,
                                                     description="owner change"))
    finally:
        request_id_context.reset(tok)
    owner_kids = [k for k in store.list_children(parent) if not (k.get("metadata") or {}).get("probe_created")]
    assert owner_kids, res
