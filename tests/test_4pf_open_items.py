"""§4PF — the §4PE open items closed before autopilot goes to production
(2026-10-15). Every ledger row says WHO did it; a promoted command's real
outcome lands; a declined preview is withdrawn; ingest failures never read
as success; listings are scoped; the remaining file_system minors."""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.tools import memory as KB
from ghost_agent.tools import file_system as FS
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome
from ghost_agent.utils.logging import (conversation_key_context, event_origin_context, request_id_context,
                                       request_origin_label)
from ghost_agent.utils.provenance import note_user_message


def _model(tmp_path):
    from ghost_agent.workspace.model import WorkspaceModel
    m = WorkspaceModel(root=tmp_path / "ws", enabled=True)
    m.sandbox_root = tmp_path / "sb"; (tmp_path / "sb").mkdir(exist_ok=True)
    return m


# ── W5: who did it ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("rid,label", [
    ("web-1", "owner"), ("probe-x", "probe"), ("sched-9", "scheduled"), ("sim-2", "self-play"),
    ("sub-leaf-3", "sub-agent"), ("job-4", "job"), ("bench-5", "test"), ("", "system"),
])
def test_the_origin_label_follows_the_request_kind(rid, label):
    tok = request_id_context.set(rid)
    try:
        assert request_origin_label() == label
    finally:
        request_id_context.reset(tok)


def test_a_pinned_origin_wins_and_every_ledger_row_carries_it(tmp_path):
    m = _model(tmp_path)
    tok = event_origin_context.set("autopilot")
    try:
        m.record_command_outcome(command="pytest -q", exit_code=1)
        m.note("built the parser")
    finally:
        event_origin_context.reset(tok)
    tok = request_id_context.set("web-9")
    try:
        m.record_command_outcome(command="ls", exit_code=0)
    finally:
        request_id_context.reset(tok)
    rows = m.activity.recent(limit=10)
    by = {r.summary: r.origin for r in rows}
    assert by["built the parser"] == "autopilot" and by["ran `pytest -q` exit=1"] == "autopilot"
    assert by["ran `ls` exit=0"] == "owner"


def test_the_prefix_and_the_listings_tag_the_agents_own_work(tmp_path):
    from ghost_agent.workspace.recognition import build_workspace_prefix
    from ghost_agent.tools.workspace import _render_events
    m = _model(tmp_path)
    tok = event_origin_context.set("autopilot")
    try:
        m.record_command_outcome(command="make build", exit_code=2)
    finally:
        event_origin_context.reset(tok)
    m.record_command_outcome(command="git status", exit_code=0)
    listing = _render_events(m, "command", 10, label="command outcomes")
    assert "[autopilot] ran `make build` exit=2" in listing
    assert "[owner]" not in listing and "ran `git status` exit=0" in listing
    prefix = build_workspace_prefix(activity=m.activity, state=m.state, narrative=None, recent_events_n=5)
    assert "[command] [autopilot] ran `make build` exit=2" in prefix


async def test_an_advance_tick_pins_autopilot_and_the_owners_pins_owner(monkeypatch):
    from ghost_agent.core import project_advancer as PA
    seen = []

    async def _body(context, project_id, *a, **kw):
        seen.append(event_origin_context.get()); return "ok"
    monkeypatch.setattr(PA, "_advance_once_body", _body)
    await PA.advance_once(SimpleNamespace(), "abcdefabcdef")
    await PA.advance_once(SimpleNamespace(), "abcdefabcdef", owner_requested=True)
    assert seen == ["autopilot", "owner"] and event_origin_context.get() == ""


def test_project_events_say_who_wrote_them_and_the_digest_skips_the_owners_own(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.core.project_digest import summarize_since
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Digest App")
    tok = request_id_context.set("web-owner")
    try:
        store.log_event(pid, None, "autoadvance_step", {"task": "t1"})
    finally:
        request_id_context.reset(tok)
    tok = event_origin_context.set("autopilot")
    try:
        store.log_event(pid, None, "autoadvance_step", {"task": "t2"})
    finally:
        event_origin_context.reset(tok)
    evs = store.list_events(pid, event_type="autoadvance_step")
    assert {e["payload"]["origin"] for e in evs} == {"owner", "autopilot"}
    res = summarize_since(store, 0)
    assert res.advanced == 1


# ── W7: a promoted command's real outcome ──────────────────────────────────

def test_a_finished_job_supersedes_its_promotion_row(tmp_path):
    from ghost_agent.tools.workspace import _render_events
    m = _model(tmp_path)
    m.record_command_outcome(command="make all", exit_code=0, note="still running — promoted to background job j1",
                             job_id="j1")
    before = _render_events(m, "command", 10, label="command outcomes")
    assert "promoted to background job j1" in before
    m.record_job_finished({"id": "j1", "command": "make all", "exit_code": 2, "state": "done",
                           "started_at": 100.0, "finished_at": 160.0})
    after = _render_events(m, "command", 10, label="command outcomes")
    assert "promoted to background job j1" not in after and "exit=2 — background job j1 finished" in after


def test_the_reaper_fires_the_finish_hook(monkeypatch):
    from ghost_agent.sandbox import jobs as J
    got = []
    J.register_finish_hook(lambda e: got.append(e))
    try:
        J._fire_finish({"id": "j9", "exit_code": 0})
        assert got and got[0]["id"] == "j9"
        J._FINISH_HOOKS.append(lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
        J._fire_finish({"id": "j10"})                      # a failing hook never breaks the reaper
    finally:
        J._FINISH_HOOKS.clear()


def test_the_promotion_row_carries_its_job_id(tmp_path):
    from ghost_agent.tools.execute import _record_promoted_command
    m = _model(tmp_path)
    _record_promoted_command(m, "sleep 999", 90.0, {"id": "job-abc"})
    row = m.activity.recent(limit=1)[0]
    assert row.payload.get("job_id") == "job-abc"


# ── W6, W11, W12, W13 ──────────────────────────────────────────────────────

def test_a_missing_file_is_not_newly_tracked_news():
    from ghost_agent.workspace.state import _diff_snapshots
    from ghost_agent.workspace.schema import FileSnapshot
    assert _diff_snapshots(None, FileSnapshot(path="/x", exists=False)) is None
    assert _diff_snapshots(None, FileSnapshot(path="/x", exists=True, size=1)) == "newly tracked"


def test_a_url_pulled_in_another_project_is_still_recorded_for_this_one(tmp_path):
    m = _model(tmp_path)
    m.current_project_id = "aaaaaaaaaaaa"
    assert m.record_research_artifact(url="https://docs.example/x", title="x") is not None
    assert m.record_research_artifact(url="https://docs.example/x", title="x") is None     # same project: dedup
    m.current_project_id = "bbbbbbbbbbbb"
    assert m.record_research_artifact(url="https://docs.example/x", title="x") is not None  # other project: recorded


def test_listings_are_scoped_to_the_active_project_and_say_so(tmp_path):
    from ghost_agent.tools.workspace import _render_events
    m = _model(tmp_path)
    m.current_project_id = "aaaaaaaaaaaa"; m.record_command_outcome(command="a-cmd", exit_code=0)
    m.current_project_id = "bbbbbbbbbbbb"; m.record_command_outcome(command="b-cmd", exit_code=0)
    out = _render_events(m, "command", 10, label="command outcomes")
    assert "(project bbbbbbbbbbbb)" in out and "b-cmd" in out and "a-cmd" not in out
    m.current_project_id = ""
    assert "a-cmd" in _render_events(m, "command", 10, label="command outcomes")


# ── K10, C2 ────────────────────────────────────────────────────────────────

def _plan(kind, rid, **extra):
    return KB._store_plan({"kind": kind, "rid": rid, "ts": time.time(), **extra})


def test_a_bare_no_withdraws_the_conversations_previews_a_narrowing_does_not():
    t = (request_id_context.set("req-500"), conversation_key_context.set("convNo"))
    try:
        _plan("forget", "req-500", target="x", items=[])
        assert KB.retire_declined_previews("wait, only item 1") == 0
        assert KB.retire_declined_previews("No thanks.") == 1
        assert KB._resolve_plan("yes", "forget")[1] is None
    finally:
        request_id_context.reset(t[0]); conversation_key_context.reset(t[1])


async def test_the_chat_handler_withdraws_previews_on_a_no(monkeypatch):
    from tests.helpers import FakeBgTasks, make_context
    from tests.test_4ji_request_echo_narration import _resp
    from ghost_agent.core import agent as A
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1"); monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context(); agent = A.GhostAgent(ctx)
    from ghost_agent.tools.projects import conversation_fingerprint
    hist = [{"role": "user", "content": "delete project x"}, {"role": "assistant", "content": "Shall I?"}]
    conv = conversation_fingerprint(hist)
    t = (request_id_context.set("req-510"), conversation_key_context.set(conv))
    try:
        tok = _plan("project_delete", "req-510", project_id="abcdefabcdef")
    finally:
        request_id_context.reset(t[0]); conversation_key_context.reset(t[1])
    hist.append({"role": "user", "content": "no"})
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[_resp("ok, left as is.")] * 2)
    await agent.handle_chat({"messages": list(hist)}, FakeBgTasks())
    assert tok not in KB._FORGET_PLANS




# ── K12, K13 ───────────────────────────────────────────────────────────────

async def test_a_second_ingest_of_one_name_is_refused_and_the_claim_is_released(monkeypatch, tmp_path):
    calls = []

    async def _gain(filename, *a, **k):
        calls.append(filename); return "SUCCESS: Ingested"
    monkeypatch.setattr(KB, "tool_gain_knowledge", _gain)
    KB._TOOL_INGEST_IN_FLIGHT.add("notes.txt")
    try:
        out = await KB.tool_knowledge_base(action="ingest_document", filename="notes.txt", sandbox_dir=tmp_path,
                                          memory_system=MagicMock())
        assert getattr(out, "status", None) is OutcomeStatus.REJECTED and not calls
    finally:
        KB._TOOL_INGEST_IN_FLIGHT.discard("notes.txt")
    out2 = await KB.tool_knowledge_base(action="ingest_document", filename="notes.txt", sandbox_dir=tmp_path,
                                       memory_system=MagicMock())
    assert "SUCCESS" in str(out2) and calls == ["notes.txt"] and "notes.txt" not in KB._TOOL_INGEST_IN_FLIGHT


async def test_a_failed_library_index_is_rolled_back_not_reported_as_success(monkeypatch, tmp_path):
    monkeypatch.setattr(KB, "semantic_split_text", lambda text, *a, **k: ["one chunk"])
    rolled = []
    monkeypatch.setattr(KB, "_rollback_partial", lambda ms, name: rolled.append(name))
    ms = MagicMock(); ms.get_library.return_value = []; ms.ingest_document.return_value = (True, "ok")
    ms._update_library_index.side_effect = RuntimeError("index locked")
    (tmp_path / "note.md").write_text("Some text that is long enough to ingest.\n")
    out = str(await KB.tool_gain_knowledge("note.md", tmp_path, ms))
    assert out.startswith("Embedding Error") and "index" in out and rolled == ["note.md"]


# ── P6, outer root, F11, F15, F13 ──────────────────────────────────────────

async def test_update_status_archived_runs_the_archive_action(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Old App")
    ctx = SimpleNamespace(project_store=store, scratchpad=None, graph_memory=None, workspace_model=None,
                          current_project_id=pid, last_user_content="", sandbox_manager=None)
    out = str(await tool_manage_projects_for_model(ctx, action="update", project_id=pid, status="ARCHIVED"))
    assert '"archived": true' in out and store.get_project(pid)["status"].upper() == "ARCHIVED"


def test_the_outer_root_needs_the_project_workspace_shape(tmp_path):
    root = tmp_path / "projects" / "notahexid"; root.mkdir(parents=True)
    assert FS._outer_root(root) == root
    ws = tmp_path / "projects" / "abcdefabcdef"; ws.mkdir()
    assert FS._outer_root(ws) == tmp_path


async def test_a_pdf_page_is_bounded_like_a_text_chunk(tmp_path):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open(); page = doc.new_page()
    for i in range(60):
        page.insert_text((40, 40 + i * 12), "word " * 40)
    doc.save(str(tmp_path / "big.pdf")); doc.close()
    out = str(await FS.tool_file_system(operation="read_chunked", path="big.pdf", chunk_size=1000, sandbox_dir=tmp_path))
    assert "[PDF Data - Page 1 of 1]" in out and "the first 1,000 are shown" in out
    assert len(out) < 1600


async def test_a_directory_swapped_for_a_link_after_the_check_is_refused(tmp_path, monkeypatch):
    """r4 (F15): the containment decision is re-made right before the write;
    the first cut tested the resolved path (never a link). The swap is done
    inside the thread hop, i.e. after `_get_safe_path` and before the write."""
    outside = tmp_path / "outside"; outside.mkdir()
    sb = tmp_path / "sb"; (sb / "d").mkdir(parents=True)
    real_to_thread = asyncio.to_thread
    swapped = []

    async def _swap_then_run(fn, *a, **k):
        if not swapped:
            import shutil
            shutil.rmtree(sb / "d"); os.symlink(outside, sb / "d"); swapped.append(1)
        return await real_to_thread(fn, *a, **k)
    monkeypatch.setattr(FS.asyncio, "to_thread", _swap_then_run)
    import tempfile
    real_mkstemp = tempfile.mkstemp; stemp_dirs = []

    def _spy_mkstemp(*a, **k):
        stemp_dirs.append(str(k.get("dir") or (a[2] if len(a) > 2 else "")))
        return real_mkstemp(*a, **k)
    monkeypatch.setattr(tempfile, "mkstemp", _spy_mkstemp)
    out = await FS.tool_file_system(operation="write", path="d/new.txt", content="x", sandbox_dir=sb)
    assert "Security Error" in str(out) and not (outside / "new.txt").exists(), out
    # the refusal comes BEFORE any temp file is created in the swapped (outside) directory
    assert not [d for d in stemp_dirs if Path(d).resolve() == outside.resolve()], stemp_dirs
    assert not list(outside.iterdir())


async def test_a_replace_result_names_its_own_call(tmp_path):
    (tmp_path / "a.txt").write_text("alpha beta gamma\n")
    tok = request_id_context.set("req-530")
    try:
        await FS.tool_file_system(operation="read", path="a.txt", sandbox_dir=tmp_path)
        out = str(await FS.tool_file_system(operation="replace", path="a.txt", content="beta", replace_with="delta",
                                            sandbox_dir=tmp_path))
    finally:
        request_id_context.reset(tok)
    assert out.startswith("SUCCESS") and "This call's old_string" in out


# ── W9 ─────────────────────────────────────────────────────────────────────

def test_a_portless_service_whose_probe_failed_is_not_reported_running(tmp_path, monkeypatch):
    from tests.test_service_port_leases import _sup
    from ghost_agent.sandbox.services import ServiceSupervisor
    sup, sb = _sup(tmp_path)
    monkeypatch.setattr(ServiceSupervisor, "_pid_state", lambda self, pid: None)
    out = str(sup.start("worker", "python3 worker.py", port=0))      # port=0: a non-listening worker
    assert "liveness NOT confirmed" in out and "RUNNING" not in out.split("\n")[0]


# ── round 4: defects the diff reader found in the first cut ────────────────

def test_a_job_lands_on_its_own_project_and_origin_not_the_observers(tmp_path):
    """r4 MAJOR: the finish row went to whoever observed the landing (the 60 s
    sweeper → system on the chat's project; the owner's next execute →
    owner). A probe's job leaves no row."""
    from ghost_agent.tools.workspace import _render_events
    m = _model(tmp_path)
    m.current_project_id = "bbbbbbbbbbbb"                        # the chat side
    m.record_job_finished({"id": "j1", "command": "make all", "exit_code": 2, "state": "done",
                           "started_at": 1.0, "finished_at": 61.0, "project_id": "aaaaaaaaaaaa", "origin": "autopilot"})
    row = m.activity.recent(limit=1)[0]
    assert row.project_id == "aaaaaaaaaaaa" and row.origin == "autopilot"
    assert m.record_job_finished({"id": "j2", "command": "x", "exit_code": 0, "state": "done",
                                  "project_id": "aaaaaaaaaaaa", "origin": "probe"}) is None
    m.record_job_finished({"id": "j3", "command": "serve", "exit_code": None, "state": "expired",
                           "project_id": "aaaaaaaaaaaa", "started_by": "web-1"})
    assert "EXPIRED" in m.activity.recent(limit=1)[0].summary


def test_expired_and_cancelled_jobs_fire_the_finish_hook():
    from ghost_agent.sandbox import jobs as J
    got = []
    J.register_finish_hook(lambda e: got.append(e.get("state")))
    try:
        for st in ("expired", "cancelled", "done", "lost"):
            J._fire_finish({"id": "j", "state": st})
    finally:
        J._FINISH_HOOKS.clear()
    assert got == ["expired", "cancelled", "done", "lost"]


def test_a_scoped_listing_survives_a_busy_other_project(tmp_path):
    """r4 MAJOR: a window of the newest N rows went blank behind another
    project's rows."""
    from ghost_agent.tools.workspace import _render_events
    m = _model(tmp_path)
    m.current_project_id = "aaaaaaaaaaaa"; m.record_command_outcome(command="mine", exit_code=0)
    m.current_project_id = "bbbbbbbbbbbb"
    for i in range(60):
        m.record_command_outcome(command=f"theirs {i}", exit_code=0)
    m.current_project_id = "aaaaaaaaaaaa"
    out = _render_events(m, "command", 10, label="command outcomes")
    assert "mine" in out and "theirs" not in out


async def test_a_probes_advance_stays_a_probe(monkeypatch):
    from ghost_agent.core import project_advancer as PA
    seen = []

    async def _body(context, project_id, *a, **kw):
        seen.append(event_origin_context.get()); return "ok"
    monkeypatch.setattr(PA, "_advance_once_body", _body)
    tok = request_id_context.set("probe-77")
    try:
        await PA.advance_once(SimpleNamespace(), "abcdefabcdef", owner_requested=True)
    finally:
        request_id_context.reset(tok)
    assert seen == ["probe"]


def test_a_bare_stop_is_not_a_decline():
    t = (request_id_context.set("req-600"), conversation_key_context.set("convStop"))
    try:
        _plan("forget", "req-600", target="x", items=[])
        assert KB.retire_declined_previews("stop") == 0 and KB.retire_declined_previews("abort") == 0
        assert KB.retire_declined_previews("nah") == 1
    finally:
        request_id_context.reset(t[0]); conversation_key_context.reset(t[1])


async def test_an_archived_update_keeps_its_other_fields(tmp_path):
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Old App")
    ctx = SimpleNamespace(project_store=store, scratchpad=None, graph_memory=None, workspace_model=None,
                          current_project_id=pid, last_user_content="", sandbox_manager=None)
    await tool_manage_projects_for_model(ctx, action="update", project_id=pid, status="ARCHIVED", title="Renamed App")
    p = store.get_project(pid)
    assert p["title"] == "Renamed App" and p["status"].upper() == "ARCHIVED"


def test_a_listing_inside_an_autopilot_tick_is_scoped_to_the_pinned_project(tmp_path):
    from ghost_agent.tools.workspace import _render_events
    from ghost_agent.workspace.model import pinned_event_project
    m = _model(tmp_path)
    m.current_project_id = "bbbbbbbbbbbb"
    with pinned_event_project("aaaaaaaaaaaa"):
        m.record_command_outcome(command="tick-cmd", exit_code=0)
        out = _render_events(m, "command", 10, label="command outcomes")
    assert "(project aaaaaaaaaaaa)" in out and "tick-cmd" in out


def test_the_reaper_books_an_expired_job_through_the_hook(tmp_path):
    """r4 MAJOR: the TTL-expired landing (the common one for promoted servers)
    never fired the hook — driven through the real reaper."""
    from tests.test_sandbox_job_promotion import FakeSandbox
    from ghost_agent.sandbox import jobs as J
    sup = J.SandboxJobSupervisor(FakeSandbox(tmp_path))
    jid = "job-0000beef"; sup.host_dir.mkdir(parents=True, exist_ok=True)
    (sup.host_dir / "registry.json").write_text(json.dumps({jid: {
        "id": jid, "state": J.STATE_RUNNING, "pid": 4242, "command": "serve", "project_id": "aaaaaaaaaaaa",
        "deadline_at": time.time() - 1, "promoted_at": time.time(), "container_id": sup._container_generation()}}))
    sup._kill_pgroup = lambda pid: True
    got = []
    J.register_finish_hook(lambda e: got.append((e["id"], e["state"])))
    try:
        sup.reap()
    finally:
        J._FINISH_HOOKS.clear()
    assert got == [(jid, J.STATE_EXPIRED)]


def test_the_reaper_books_a_done_job_through_the_hook(tmp_path):
    """Battery run 2, M11: the sentinel-DONE landing fires the hook (the expiry
    pin covered only the TTL branch)."""
    from tests.test_sandbox_job_promotion import FakeSandbox
    from ghost_agent.sandbox import jobs as J
    sup = J.SandboxJobSupervisor(FakeSandbox(tmp_path))
    jid = "job-0000d01e"; sup.host_dir.mkdir(parents=True, exist_ok=True)
    sup._nonces[jid] = "aabbccdd11223344"
    (sup.host_dir / f"{jid}.exit").write_text("aabbccdd11223344 2\n")
    (sup.host_dir / f"{jid}.log").write_text("built\n")
    (sup.host_dir / "registry.json").write_text(json.dumps({jid: {
        "id": jid, "state": J.STATE_RUNNING, "pid": 4242, "command": "make", "project_id": "aaaaaaaaaaaa",
        "origin": "autopilot", "deadline_at": time.time() + 999, "promoted_at": time.time(), "container_id": None}}))
    got = []
    J.register_finish_hook(lambda e: got.append((e["id"], e["state"], e.get("exit_code"), e.get("origin"))))
    try:
        sup.reap()
    finally:
        J._FINISH_HOOKS.clear()
    assert got == [(jid, J.STATE_DONE, 2, "autopilot")]
