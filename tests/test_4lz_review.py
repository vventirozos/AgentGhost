"""§4LZ fresh-reader findings: behaviour pins."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from ghost_agent.utils.logging import request_id_context


@pytest.fixture(autouse=True)
def _fresh_notify_budget(monkeypatch):
    import ghost_agent.tools.notify_tool as nt
    monkeypatch.setattr(nt, "_sent_timestamps", [])


def _records(tmp_path):
    p = tmp_path / "activity.jsonl"
    return [l for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


# ── MAJOR 1: a probe's notify ask never reaches the owner by the backstop ──

@pytest.mark.parametrize("rid,fires", [("probe-abc123", False), ("50855398", True)])
def test_the_finish_line_backstop_never_pages_for_a_probe(tmp_path, rid, fires):
    from ghost_agent.core.agent import _notify_promise_backstop
    from ghost_agent.core.autonomous_activity import ActivityLog
    from ghost_agent.tools.outcome import ToolOutcome
    ctx = SimpleNamespace(activity_log=ActivityLog(tmp_path / "activity.jsonl"))
    tok = request_id_context.set(rid)
    try:
        fired = _notify_promise_backstop(
            ctx, last_user_content="build it and notify me in slack when you're done",
            tools_run=[{"name": "notify_operator",
                        "content": ToolOutcome.ok("PROBE — not sent. Would have sent: done")}],
            final_content="Built it.", req_id=rid, had_failures=False)
    finally:
        request_id_context.reset(tok)
    assert fired is fires
    assert len(_records(tmp_path)) == (1 if fires else 0)


def test_a_probe_cannot_park_a_notify_promise_on_a_project(tmp_path):
    from ghost_agent.core.notify_promise import capture_promise, peek_promise
    from ghost_agent.memory.projects import ProjectStore
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("App")
    tok = request_id_context.set("probe-p")
    try:
        assert capture_promise(store, pid, "notify me when done", req_id="probe-p") is False
    finally:
        request_id_context.reset(tok)
    assert peek_promise(store, pid) is None
    assert capture_promise(store, pid, "notify me when done", req_id="owner1") is True


# ── MAJOR 2: the search/memory tools' own failure heads stay failures ──

@pytest.mark.parametrize("text", [
    "Ingest Error: the YouTube route failed", "Disk Error: failed to stat x",
    "Web Error: unreachable", "Embedding Error: x", "Search failed: timeout",
    "[error] boom", "Exception: x", "Traceback (most recent call last):\n  x",
])
def test_a_tool_written_failure_head_is_a_failure(text):
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    assert _looks_like_tool_error(text, "knowledge_base")
    assert _looks_like_tool_error(text, "web_search")


@pytest.mark.parametrize("text", [
    "### 1. Exceptional performance\nError: quoted in a snippet",
    "Python Exception handling: a guide", "Found 3 documents. Error: none",
])
def test_content_that_mentions_errors_is_not_a_failure(text):
    from ghost_agent.distill.outcome_heuristics import _looks_like_tool_error
    assert not _looks_like_tool_error(text, "web_search")


# ── MAJOR 3: a restated create keeps the new constraints' origin ──

async def test_a_restated_create_stamps_the_new_constraints_origin(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.tools.projects import tool_manage_projects
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    ctx = SimpleNamespace(project_store=store, scratchpad=None, graph_memory=None,
                          workspace_model=None, current_project_id=None, last_user_content="")
    tok = request_id_context.set("owner-1")
    try:
        await tool_manage_projects(ctx, action="create", title="Chess", goal="a chess game",
                                   constraints=["no AI opponent"])
    finally:
        request_id_context.reset(tok)
    tok = request_id_context.set("sched-retry")
    try:
        await tool_manage_projects(ctx, action="create", title="Chess", goal="a chess game",
                                   constraints=["no AI opponent", "dark theme"])
    finally:
        request_id_context.reset(tok)
    (proj,) = store.list_projects()
    meta = proj["metadata"]
    assert "dark theme" in meta["constraints"]
    assert (meta.get("constraint_origins") or {}).get("dark theme") == "auto"


# ── MINOR 4: the owner's fork retires a probe's leftover fork ──

async def test_the_owner_fork_archives_a_probe_fork(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.tools.projects import tool_manage_projects
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    ctx = SimpleNamespace(project_store=store, scratchpad=None, graph_memory=None,
                          workspace_model=None, current_project_id=None, last_user_content="")
    parent = store.create_project("Coach")
    with store._lock, store._connect() as conn:
        conn.execute("UPDATE projects SET status='RELEASED' WHERE id=?", (parent,))
    for rid in ("probe-f", "req-owner"):
        tok = request_id_context.set(rid)
        try:
            await tool_manage_projects(ctx, action="create_version", project_id=parent, description="x")
        finally:
            request_id_context.reset(tok)
    live = [k for k in store.list_children(parent) if str(k.get("status")).upper() != "ARCHIVED"]
    assert len(live) == 1 and not (live[0].get("metadata") or {}).get("probe_created")


# ── MINOR 6: the JSON store helpers ──

def test_a_store_saved_meanwhile_is_not_set_aside(tmp_path):
    from ghost_agent.utils.json_store import preserve_corrupt
    f = tmp_path / "s.json"
    f.write_text('{"good": 1}')                    # another writer already repaired it
    preserve_corrupt(f, ValueError("old read"), "s")
    assert json.loads(f.read_text()) == {"good": 1}
    assert not list(tmp_path.glob("s.json.corrupt-*"))


def test_two_set_asides_keep_both_damaged_files(tmp_path):
    from ghost_agent.utils.json_store import preserve_corrupt
    f = tmp_path / "s.json"
    for body in ('{"a": ', '{"b": '):
        f.write_text(body)
        preserve_corrupt(f, ValueError("x"), "s")
    kept = sorted(p.read_text() for p in tmp_path.glob("s.json.corrupt-*"))
    assert kept == ['{"a": ', '{"b": ']


def test_an_atomic_write_leaves_no_temp_behind(tmp_path):
    from ghost_agent.utils.json_store import write_json_atomic
    write_json_atomic(tmp_path / "s.json", {"x": 1})
    assert os.listdir(tmp_path) == ["s.json"]


# ── MINOR 8: a job a test replay started wakes as background ──

async def test_a_replay_started_job_wakes_as_background(monkeypatch):
    from unittest.mock import MagicMock
    import ghost_agent.main as MAIN
    import ghost_agent.tools.tasks as T
    from ghost_agent.utils.logging import request_kind
    monkeypatch.setattr(T, "should_defer_scheduled_task", lambda llm: False)
    seen = []

    async def fake_fg(ctx, body, rid):
        seen.append(rid)
        return "done", None, None
    monkeypatch.setattr(MAIN, "_handle_chat_foreground", fake_fg)
    MAIN._RESUMED_JOBS.clear()
    MAIN._resume_times.clear()
    ctx = SimpleNamespace(llm_client=MagicMock(), args=SimpleNamespace(model="m"),
                          sandbox_manager=None, activity_log=None, agent=MagicMock())
    await MAIN._resume_after_job(ctx, {"id": "job-r1", "state": "done", "exit_code": 0,
                                       "command": "sleep 1", "started_by": "replay-7"})
    assert seen and request_kind(seen[0]) == "background"


# ── closing items ──

@pytest.mark.parametrize("rid", ["bench-1", "replay-2", "sched-3", "sub-4", "sim-5", "job-6"])
def test_a_request_that_writes_background_notes_sees_them(rid):
    from ghost_agent.core.agent import _scratch_hidden_namespaces
    from ghost_agent.tools.memory import _scratch_request_kind
    tok = request_id_context.set(rid)
    try:
        assert _scratch_request_kind() == "background"     # it writes into bg …
    finally:
        request_id_context.reset(tok)
    assert _scratch_hidden_namespaces(rid) == ()           # … and its prompt shows bg


@pytest.mark.parametrize("rid", ["39c394ca", "probe-1"])
def test_an_owner_or_probe_prompt_hides_background_notes(rid):
    from ghost_agent.core.agent import _scratch_hidden_namespaces
    assert _scratch_hidden_namespaces(rid) == ("bg",)


async def test_a_probe_notification_says_it_was_not_sent(monkeypatch):
    from unittest.mock import MagicMock
    import ghost_agent.tools.notify_tool as N
    from ghost_agent.core.agent import _notify_delivered
    tok = request_id_context.set("probe-w")
    try:
        out = await N.tool_notify_operator(message="disk full", context=MagicMock())
    finally:
        request_id_context.reset(tok)
    assert str(out).startswith("PROBE — NOT SENT") and "Tell the user" in out
    assert not _notify_delivered({"name": "notify_operator", "content": out})


def test_a_macro_registered_during_a_save_does_not_break_it(tmp_path, monkeypatch, caplog):
    """Deterministic interleaving: while save() snapshots the dict, another
    thread registers a macro. Under the lock it waits; unlocked, the dict
    changes size mid-snapshot and that save is lost."""
    import threading
    from ghost_agent.tools.composed_skills import ComposedSkill, ComposedSkillRegistry
    reg = ComposedSkillRegistry(tmp_path)
    step = [{"tool": "web_search", "params": {"query": "$q"}}]
    reg.compile_from_pattern("seed", step, "t")
    real = ComposedSkill.to_dict
    fired = {"done": False}

    def to_dict_racing(self):
        if not fired["done"]:
            fired["done"] = True
            th = threading.Thread(target=reg.compile_from_pattern, args=("late", step, "t"))
            th.start()
            th.join(0.3)          # let it run if nothing holds it back
        return real(self)
    monkeypatch.setattr(ComposedSkill, "to_dict", to_dict_racing)
    import logging
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        reg.record_usage("seed", True)
    monkeypatch.setattr(ComposedSkill, "to_dict", real)
    # no save failed — a later save heals the file, but until then (or
    # after a crash) the failed one is a lost write
    assert not [r for r in caplog.records if "Failed to save composed skills" in r.getMessage()]
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.name.startswith("Thread"):
            t.join(5)
    on_disk = json.loads((tmp_path / "composed_skills.json").read_text())
    assert on_disk["seed"]["usage_count"] == 1 and "late" in on_disk
