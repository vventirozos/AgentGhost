"""§4LP — projects and workspace: behaviour pins for the review's fixes."""
from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools.projects import tool_manage_projects


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(tmp_path, store):
    from ghost_agent.memory.scratchpad import Scratchpad
    return SimpleNamespace(project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                           graph_memory=None, workspace_model=None, current_project_id=None,
                           last_user_content="")


# ── CRIT: research notes never write through a symlink ──

def test_a_symlinked_research_dir_is_refused(store, tmp_path):
    from ghost_agent.core.project_research import _research_dir, RESEARCH_SUBDIR
    pid = store.create_project("Notes")
    ws = store.ensure_workspace(pid)
    outside = tmp_path / "host_dir"
    outside.mkdir()
    os.symlink(outside, ws / RESEARCH_SUBDIR)
    assert _research_dir(store, pid) is None


def test_a_symlinked_research_file_is_never_written_through(store, tmp_path):
    from ghost_agent.core.project_research import _write_index_md, _research_dir
    pid = store.create_project("Notes")
    rdir = _research_dir(store, pid)
    victim = tmp_path / "victim.txt"
    victim.write_text("original")
    os.symlink(victim, rdir / "INDEX.md")
    _write_index_md(store, pid, rdir)
    assert victim.read_text() == "original"


# ── release never changes a host file's mode through a link ──

def test_release_chmod_skips_links_out_of_the_workspace(store, tmp_path):
    pid = store.create_project("App")
    ws = store.ensure_workspace(pid)
    (ws / "app.py").write_text("print(1)")
    host = tmp_path / "host_secret.txt"
    host.write_text("x")
    os.chmod(host, 0o600)
    os.symlink(host, ws / "link.txt")
    store.set_workspace_readonly(pid, True)
    assert stat.S_IMODE(os.stat(host).st_mode) == 0o600
    store.set_workspace_readonly(pid, False)


# ── the idle advancer never builds a task nobody asked for ──

def test_the_idle_advancer_skips_verifier_followups(store):
    from ghost_agent.core.planning import ProjectPlan
    from ghost_agent.core.project_advancer import _is_unrequested_task
    pid = store.create_project("App")
    store.add_task(pid, "Verifier follow-up: add a retry button")
    plan = ProjectPlan(store, pid)
    assert plan.next_ready_leaf(skip=_is_unrequested_task) is None
    assert plan.next_ready_leaf() is not None          # the owner's own run still sees it
    store.add_task(pid, "Build the settings page")
    assert "settings" in ProjectPlan(store, pid).next_ready_leaf(skip=_is_unrequested_task).description


# ── fork steers only on a user request ──

def test_released_refusals_never_invite_an_unasked_fork(store):
    from ghost_agent.tools.projects import _released_guard, FORK_ONLY_IF_ASKED
    pid = store.create_project("Chess Coach v3")
    store.update_project(pid, status="RELEASED")
    msg = _released_guard(store, pid)
    assert msg and FORK_ONLY_IF_ASKED in msg


# ── the released guard follows titles and task owners ──

async def test_a_title_passed_as_project_id_is_still_guarded(context, store):
    pid = store.create_project("Chess Coach v3")
    store.update_project(pid, status="RELEASED")
    res = await tool_manage_projects(context, action="update", project_id="Chess Coach v3", status="ACTIVE")
    assert res.is_failure and store.get_project(pid)["status"] == "RELEASED"


async def test_a_task_of_a_released_project_is_guarded_from_another_active_project(context, store):
    rel = store.create_project("Released app")
    tid = store.add_task(rel, "old task")
    store.update_project(rel, status="RELEASED")
    other = store.create_project("Other")
    context.current_project_id = other
    res = await tool_manage_projects(context, action="artifact_add", task_id=tid, file_path="x.py")
    assert res.is_failure and "RELEASED" in str(res)


# ── status reports the project it was asked about ──

async def test_status_of_a_named_project_reports_it_without_switching(context, store):
    pid = store.create_project("Jiu Jitsu Calendar")
    res = json.loads(await tool_manage_projects(context, action="status", project_id=pid))
    assert res["project"] == pid and res["current"] is None
    assert context.current_project_id is None


# ── refusals are not successes ──

def test_a_task_update_that_landed_nothing_is_rejected():
    from ghost_agent.tools.projects import _ok
    for key in ("missing", "still_failing", "constraint_violations", "gated_unverified"):
        assert _ok({"updated": [], "count": 0, key: ["t1"]}).is_failure
    assert not _ok({"updated": [{"id": "t1"}], "count": 1}).is_failure


# ── exit never clears another conversation's binding ──

async def test_exit_with_nothing_active_clears_nothing(context, monkeypatch):
    import ghost_agent.tools.projects as PR
    called = []
    monkeypatch.setattr(PR, "_set_current", lambda ctx, pid: called.append(pid))
    res = json.loads(await tool_manage_projects(context, action="exit"))
    assert res["exited"] is None and called == []


# ── workspace: probes do not write; dates move; failed reads are not "pulled" ──

def test_a_probe_turn_writes_nothing_to_the_workspace(tmp_path):
    from ghost_agent.workspace.model import WorkspaceModel
    from ghost_agent.utils.logging import request_origin_context, ORIGIN_PROBE
    wm = WorkspaceModel(tmp_path / "ws")
    tok = request_origin_context.set(ORIGIN_PROBE)
    try:
        assert wm.record_research_artifact(url="https://a.org/x") is None
        assert wm.record_command_outcome(command="raise ValueError('boom')", exit_code=1) is None
    finally:
        request_origin_context.reset(tok)
    assert wm.record_research_artifact(url="https://a.org/x") is not None


def test_the_last_touched_date_moves_on_every_boot(tmp_path):
    import time
    from ghost_agent.workspace.state import WorkspaceStateThread
    root = tmp_path / "ws"
    boots = []
    for _ in range(3):                                 # three agent boots
        st = WorkspaceStateThread(root)
        st.touch_session()
        boots.append((st.state.prior_session_at, st.state.last_session_at))
        time.sleep(0.01)
    # each boot's "previous session" is the boot before it — never frozen
    assert boots[1][0] == boots[0][1] and boots[2][0] == boots[1][1]


def test_an_interact_that_did_not_read_its_page_is_not_pulled():
    from ghost_agent.tools.browser import _interact_read_failed
    assert _interact_read_failed({"aborted": True, "actions": []})
    assert _interact_read_failed({"actions": [{"action": "goto", "ok": True, "url": "https://x/",
                                               "title": "Just a moment...", "status": 403}]})
    assert not _interact_read_failed({"actions": [{"action": "goto", "ok": True, "url": "https://x/",
                                                   "title": "X", "status": 200}]})


# ── battery round: drive the real paths ──

async def test_the_real_idle_tick_never_claims_a_verifier_followup(store):
    from ghost_agent.core.project_advancer import advance_once
    pid = store.create_project("Shipped app")
    store.update_project(pid, status="ACTIVE")
    store.add_task(pid, "Verifier follow-up: add a retry button to the verifier panel")
    ctx = SimpleNamespace(project_store=store, workspace_model=None, current_project_id=None)
    res = await advance_once(ctx, pid)
    assert res.classification == "idle"
    assert all(t.get("status") in ("PENDING", "READY") for t in store.list_tasks(pid))


def test_the_findings_file_is_never_written_through_a_link(store, tmp_path, monkeypatch):
    import ghost_agent.core.project_research as PR
    pid = store.create_project("Notes")
    rdir = PR._research_dir(store, pid)
    victim = tmp_path / "victim2.txt"
    victim.write_text("original")
    os.symlink(victim, rdir / PR.MAIN_LOOP_FINDINGS_FILE)
    monkeypatch.setattr(PR, "request_relevant_to_project", lambda *a, **k: True)
    monkeypatch.setattr(PR, "parse_search_results", lambda out: [{"title": "t", "url": "https://a/", "snippet": "s"}])
    monkeypatch.setattr(PR, "_render_finding", lambda q, r, ts: f"## {q}\n- t\n")
    PR.record_main_loop_findings(store, pid, "query", "output")
    assert victim.read_text() == "original"


async def test_a_research_note_is_never_written_through_a_link(store, tmp_path, monkeypatch):
    import ghost_agent.core.project_research as PR
    pid = store.create_project("Notes")
    rdir = PR._research_dir(store, pid)
    victim = tmp_path / "victim3.txt"
    victim.write_text("original")
    os.symlink(victim, rdir / f"{PR._slugify('Topic Two')}.md")

    async def _summ(*a, **k):
        return "summary"
    monkeypatch.setattr(PR, "_summarize", _summ)
    res = await PR._persist(SimpleNamespace(project_store=store), pid, "Topic Two", "results text")
    assert victim.read_text() == "original" and not res.ok


def test_research_records_only_the_sources_that_loaded():
    from unittest.mock import MagicMock
    from ghost_agent.tools.search import record_loaded_sources
    wm = MagicMock()
    n = record_loaded_sources(wm, ["https://ok/", "https://err/", "https://blocked/"],
                              ["real page text", RuntimeError("timeout"), "BLOCKED page"],
                              lambda c: c.startswith("BLOCKED"), source="deep_research", note="q")
    assert n == 1 and wm.record_research_artifact.call_args.kwargs["url"] == "https://ok/"


async def test_an_aborted_interact_is_not_booked_pulled(tmp_path):
    from unittest.mock import MagicMock
    from tests.test_4ln_browser import _run
    wm = MagicMock(enabled=True)
    wm.record_navigation.return_value = ""
    from ghost_agent.tools import browser as B
    await B.tool_browser(operation="interact", actions=[{"action": "goto", "url": "https://x.org/"}],
                         sandbox_dir=tmp_path, workspace_model=wm,
                         sandbox_manager=__import__("tests.test_4ln_browser", fromlist=["_stub"])._stub(
                             {"final_url": "https://x.org/", "final_title": "x", "aborted": True, "actions": [
                                 {"index": 0, "action": "goto", "ok": False, "url": "https://x.org/", "error": "timeout"}]}))
    wm.record_research_artifact.assert_not_called()


def test_a_dangling_link_at_the_findings_file_creates_nothing(store, tmp_path, monkeypatch):
    import ghost_agent.core.project_research as PR
    pid = store.create_project("Notes")
    rdir = PR._research_dir(store, pid)
    target = tmp_path / "would_be_created.txt"
    os.symlink(target, rdir / PR.MAIN_LOOP_FINDINGS_FILE)          # points at nothing yet
    monkeypatch.setattr(PR, "request_relevant_to_project", lambda *a, **k: True)
    monkeypatch.setattr(PR, "parse_search_results", lambda out: [{"title": "t", "url": "https://a/", "snippet": "s"}])
    monkeypatch.setattr(PR, "_render_finding", lambda q, r, ts: f"## {q}\n- t\n")
    PR.record_main_loop_findings(store, pid, "query", "output")
    assert not target.exists()
    # and without the link, the same call DOES write (the test reaches the write)
    os.unlink(rdir / PR.MAIN_LOOP_FINDINGS_FILE)
    assert PR.record_main_loop_findings(store, pid, "query", "output") is not None
    assert "## query" in (rdir / PR.MAIN_LOOP_FINDINGS_FILE).read_text()      # it really landed


# ── fresh-eye review of the §4LP diff ──

async def test_status_of_an_unknown_project_is_an_error(context):
    res = await tool_manage_projects(context, action="status", title="Nonexistent")
    assert res.is_failure
    res2 = await tool_manage_projects(context, action="status", project_id="deadbeef0000")
    assert res2.is_failure


def test_a_symlinked_project_workspace_gets_no_research_writes(store, tmp_path):
    import shutil
    from ghost_agent.core.project_research import _research_dir
    pid = store.create_project("Notes")
    ws = store.ensure_workspace(pid)
    shutil.rmtree(ws)
    elsewhere = tmp_path / "host_place"
    elsewhere.mkdir()
    os.symlink(elsewhere, ws)
    assert _research_dir(store, pid) is None
    assert not any(elsewhere.iterdir())


# ── §4LP recommendations: release / unrelease / delete are the owner's call ──

async def test_a_model_delete_previews_then_needs_the_users_later_turn(context, store):
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    from ghost_agent.utils.logging import request_id_context
    pid = store.create_project("Keep me")
    tok_rid = request_id_context.set("req-turn-1")
    try:
        prev = json.loads(await tool_manage_projects_for_model(context, action="delete", project_id=pid))
        assert prev.get("confirmation_needed") and store.get_project(pid)
        same = await tool_manage_projects_for_model(
            context, action="delete", project_id=pid, confirm_token=prev["confirm_token"])
        assert "NOT done" in same and store.get_project(pid)   # same turn: refused
    finally:
        request_id_context.reset(tok_rid)
    tok_rid = request_id_context.set("req-turn-2")
    try:
        prev = json.loads(await tool_manage_projects_for_model(context, action="delete", project_id=pid))
        request_id_context.set("req-turn-3")
        await tool_manage_projects_for_model(context, action="delete", project_id=pid,
                                             confirm_token=prev["confirm_token"])
    finally:
        request_id_context.reset(tok_rid)
    assert not store.get_project(pid)


async def test_a_probe_cannot_confirm_but_may_drop_its_own_project(context, store):
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    from ghost_agent.utils.logging import request_id_context
    real = store.create_project("Owner app")
    tok_rid = request_id_context.set("probe-1")
    try:
        prev = json.loads(await tool_manage_projects_for_model(context, action="delete", project_id=real))
        request_id_context.set("probe-2")
        await tool_manage_projects_for_model(context, action="delete", project_id=real,
                                             confirm_token=prev["confirm_token"])
        assert store.get_project(real)
        mine = json.loads(await tool_manage_projects_for_model(context, action="create", title="Probe scratch"))
        mpid = mine["created"]
        context.current_project_id = real          # the delete must not lean on the current project
        await tool_manage_projects_for_model(context, action="delete", project_id=mpid)
        assert store.get_project(real)
        assert not store.get_project(mpid)
    finally:
        request_id_context.reset(tok_rid)


async def test_release_by_the_model_is_only_a_preview(context, store):
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    pid = store.create_project("Demo")
    store.update_project(pid, status="DONE")
    res = json.loads(await tool_manage_projects_for_model(
        context, action="release", project_id=pid,
        directions="Open index.html in a browser; the board loads and you play white."))
    assert res.get("confirmation_needed")
    assert str(store.get_project(pid).get("status")).upper() != "RELEASED"


# ── the idle loop advances only projects the owner put on autopilot ──

async def test_only_the_owners_autoadvance_opts_in_and_counts_as_asked(context, store, monkeypatch):
    import ghost_agent.core.project_advancer as PA
    from ghost_agent.utils.logging import request_id_context
    seen = []

    async def fake_many(ctx, pid, **kw):
        seen.append(kw.get("owner_requested"))
        return SimpleNamespace(steps=[], stopped_reason="done", advanced=0, to_dict=lambda: {})
    monkeypatch.setattr(PA, "advance_many", fake_many)
    a = store.create_project("A")
    others = [store.create_project(n) for n in ("B", "C", "D")]
    tok = request_id_context.set("req-owner")
    try:
        await tool_manage_projects(context, action="autoadvance", project_id=a)
        for rid, pid in zip(("probe-x", "sched-nightly", "sub-worker"), others):
            request_id_context.set(rid)
            await tool_manage_projects(context, action="autoadvance", project_id=pid)
    finally:
        request_id_context.reset(tok)
    assert (store.get_project(a).get("metadata") or {}).get("autopilot")
    assert not any((store.get_project(p).get("metadata") or {}).get("autopilot") for p in others)
    assert seen == [True, False, False, False]


# ── the digest counts only work nobody asked for ──

def test_the_digest_never_counts_the_owners_own_advance(store):
    from ghost_agent.core.project_digest import summarize_since, render_digest
    pid = store.create_project("App")
    store.update_project(pid, status="ACTIVE")
    store.log_event(pid, None, "autoadvance_step", {"tool": "x", "owner_requested": True})
    res = summarize_since(store, 0)
    assert res.advanced == 0 and "advanced 0" not in render_digest(res)
    store.log_event(pid, None, "autoadvance_step", {"tool": "x", "owner_requested": False})
    assert summarize_since(store, 0).advanced == 1


# ── unattended, a verify task or research that did not land is not DONE ──

async def test_an_idle_verify_task_waits_for_the_owner(store):
    from ghost_agent.core.project_advancer import advance_once
    pid = store.create_project("App")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Verify the full pipeline end to end")
    ctx = SimpleNamespace(project_store=store, workspace_model=None, current_project_id=None)

    async def runner(name, args):
        return "wrote index.html"
    res = await advance_once(ctx, pid, tool_runner=runner)
    t = store.get_task(tid if isinstance(tid, str) else tid["id"])
    assert t["status"] == "NEEDS_USER", (res, t)
    assert any(e.get("type") == "autoadvance_needs_user" for e in store.list_events(pid))


async def test_an_owner_run_still_closes_the_same_task(store):
    from ghost_agent.core.project_advancer import advance_once
    pid = store.create_project("App")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Verify the full pipeline end to end")
    ctx = SimpleNamespace(project_store=store, workspace_model=None, current_project_id=None)

    async def runner(name, args):
        return "all 12 checks passed"
    await advance_once(ctx, pid, tool_runner=runner, owner_requested=True)
    t = store.get_task(tid if isinstance(tid, str) else tid["id"])
    assert t["status"] == "DONE", t


def test_unattended_close_rule():
    from ghost_agent.core.planning import TaskStatus
    from ghost_agent.core.project_advancer import _unattended_close
    assert _unattended_close(False, "Build the settings page")[0] == TaskStatus.DONE
    assert _unattended_close(False, "Test the login flow")[0] == TaskStatus.NEEDS_USER
    assert _unattended_close(False, "Research rivals", evidence_ok=False)[0] == TaskStatus.NEEDS_USER
    assert _unattended_close(True, "Test the login flow")[0] == TaskStatus.DONE


async def test_a_user_turn_still_confirms_a_probe_created_project(context, store):
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    from ghost_agent.utils.logging import request_id_context
    pid = store.create_project("Scratch", metadata={"probe_created": True})
    tok = request_id_context.set("req-owner-7")
    try:
        res = json.loads(await tool_manage_projects_for_model(context, action="delete", project_id=pid))
    finally:
        request_id_context.reset(tok)
    assert res.get("confirmation_needed") and store.get_project(pid)


def test_the_idle_loop_sees_only_autopilot_projects(store):
    from ghost_agent.core.project_advancer import idle_candidates
    a = store.create_project("Asked", metadata={"autopilot": True})
    b = store.create_project("Not asked")
    for pid in (a, b):
        store.update_project(pid, status="ACTIVE")
    assert [p["id"] for p in idle_candidates(store)] == [a]


@pytest.mark.parametrize("landed,want", [(False, "NEEDS_USER"), (True, "DONE")])
async def test_idle_research_closes_only_when_the_research_landed(store, monkeypatch, landed, want):
    import ghost_agent.core.project_research as R
    from ghost_agent.core.project_advancer import advance_once

    async def persist(*a, **k):
        return SimpleNamespace(ok=landed, path="research/x.md" if landed else None, summary="")
    monkeypatch.setattr(R, "persist_research_from_output", persist)
    pid = store.create_project("Market")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Survey the pricing pages of three rivals")
    ctx = SimpleNamespace(project_store=store, workspace_model=None, current_project_id=None)

    async def runner(name, args):
        return "### 1. Rival A pricing\nPlans from $9/month\n[Source: https://example.org/a]"
    await advance_once(ctx, pid, tool_runner=runner)
    assert store.get_task(tid)["status"] == want


async def test_an_idle_build_of_a_test_task_waits_for_the_owner(store):
    from ghost_agent.core.coding_executor import CodingResult
    from ghost_agent.core.project_advancer import advance_once
    pid = store.create_project("App", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Test the login flow end to end in test_login.py")

    async def executor(*a, **k):
        return CodingResult(ok=True, summary="wrote test_login.py", files=["test_login.py"])

    async def runner(name, args):
        return "ok"
    ctx = SimpleNamespace(project_store=store, workspace_model=None, current_project_id=None)
    await advance_once(ctx, pid, tool_runner=runner, coding_executor=executor)
    assert store.get_task(tid)["status"] == "NEEDS_USER"
    assert any(e.get("type") == "autoadvance_needs_user" for e in store.list_events(pid))


# ── §4LQ review round: the gate acts on exactly the project it previewed ──

async def test_an_unknown_title_never_falls_through_to_the_current_project(context, store):
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    pid = store.create_project("Shipped")
    store.update_project(pid, status="RELEASED")
    context.current_project_id = pid
    res = await tool_manage_projects_for_model(context, action="unrelease", title="No Such App")
    assert "not found" in res and store.get_project(pid)["status"] == "RELEASED"


async def test_the_confirmed_unrelease_hits_the_previewed_project(context, store):
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    from ghost_agent.utils.logging import request_id_context
    x = store.create_project("App X")
    cur = store.create_project("App Current")
    for pid in (x, cur):
        store.update_project(pid, status="RELEASED")
    context.current_project_id = cur
    tok = request_id_context.set("req-a")
    try:
        prev = json.loads(await tool_manage_projects_for_model(context, action="unrelease", title="App X"))
        assert prev["project"] == x
        request_id_context.set("req-b")
        await tool_manage_projects_for_model(context, action="unrelease", title="App X",
                                             confirm_token=prev["confirm_token"])   # confirmed BY TITLE (M1)
    finally:
        request_id_context.reset(tok)
    assert store.get_project(x)["status"] == "DONE"
    assert store.get_project(cur)["status"] == "RELEASED"


async def test_a_case_mangled_id_still_confirms(context, store):
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    from ghost_agent.utils.logging import request_id_context
    x = store.create_project("App Y")
    store.update_project(x, status="RELEASED")
    tok = request_id_context.set("req-c")
    try:
        prev = json.loads(await tool_manage_projects_for_model(context, action="unrelease", project_id=x))
        request_id_context.set("req-d")
        await tool_manage_projects_for_model(context, action="unrelease", project_id=x.upper(),
                                             confirm_token=prev["confirm_token"])   # m3
    finally:
        request_id_context.reset(tok)
    assert store.get_project(x)["status"] == "DONE"


async def test_the_model_cannot_mark_an_owner_project_as_a_probes(context, store):
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    from ghost_agent.utils.logging import request_id_context
    pid = store.create_project("Owner app")
    tok = request_id_context.set("probe-9")
    try:
        await tool_manage_projects_for_model(context, action="update", project_id=pid,
                                             metadata={"probe_created": True, "autopilot": True})
        await tool_manage_projects_for_model(context, action="update", project_id=pid,
                                             metadata=json.dumps({"probe_created": True}))
        res = json.loads(await tool_manage_projects_for_model(context, action="delete", project_id=pid))
    finally:
        request_id_context.reset(tok)
    meta = store.get_project(pid).get("metadata") or {}
    assert not meta.get("probe_created") and not meta.get("autopilot")
    assert res.get("confirmation_needed") and store.get_project(pid)


async def test_no_preview_for_an_action_that_would_refuse(context, store):
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    pid = store.create_project("Half done")
    res = await tool_manage_projects_for_model(context, action="unrelease", project_id=pid)
    assert "confirmation_needed" not in res and "not RELEASED" in res
    res = await tool_manage_projects_for_model(context, action="release", project_id=pid)
    assert "confirmation_needed" not in res


async def test_the_owner_turns_autopilot_on_and_off_and_a_probe_cannot(context, store):
    from ghost_agent.utils.logging import request_id_context
    pid = store.create_project("App")
    tok = request_id_context.set("req-owner")
    try:
        await tool_manage_projects(context, action="autopilot", project_id=pid, enabled=True)
        assert json.loads(await tool_manage_projects(context, action="status", project_id=pid, title="App"))["autopilot"]
        request_id_context.set("probe-1")
        await tool_manage_projects(context, action="autopilot", project_id=pid, enabled=False)
        assert json.loads(await tool_manage_projects(context, action="autopilot", project_id=pid))["autopilot"]
        request_id_context.set("req-owner-2")
        await tool_manage_projects(context, action="autopilot", project_id=pid, enabled="false")
    finally:
        request_id_context.reset(tok)
    assert not json.loads(await tool_manage_projects(context, action="autopilot", project_id=pid))["autopilot"]


async def test_the_owners_run_takes_back_a_held_task(store):
    from ghost_agent.core.project_advancer import advance_once
    pid = store.create_project("App")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Verify the full pipeline end to end")
    ctx = SimpleNamespace(project_store=store, workspace_model=None, current_project_id=None)

    async def runner(name, args):
        return "all 12 checks passed"
    await advance_once(ctx, pid, tool_runner=runner)
    assert store.get_task(tid)["status"] == "NEEDS_USER"
    res = await advance_once(ctx, pid, tool_runner=runner, owner_requested=True)
    assert res.task_id == tid and store.get_task(tid)["status"] == "DONE"


# ── §4LQ review round 2 ──

async def test_autopilot_by_title_names_that_project_not_the_current(context, store):
    from ghost_agent.utils.logging import request_id_context
    named = store.create_project("Chess Coach")
    cur = store.create_project("Other")
    context.current_project_id = cur
    tok = request_id_context.set("req-owner")
    try:
        await tool_manage_projects(context, action="autopilot", title="Chess Coach", enabled=True)
        bad = await tool_manage_projects(context, action="autopilot", project_id=named, enabled="enable-ish")
    finally:
        request_id_context.reset(tok)
    assert (store.get_project(named).get("metadata") or {}).get("autopilot")
    assert not (store.get_project(cur).get("metadata") or {}).get("autopilot")
    assert "true or false" in bad and (store.get_project(named).get("metadata") or {}).get("autopilot")


async def test_autopilot_through_metadata_is_refused_not_silently_dropped(context, store):
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    pid = store.create_project("App", metadata={"autopilot": True})
    res = await tool_manage_projects_for_model(context, action="update", project_id=pid,
                                               metadata={"autopilot": False})
    assert "action=autopilot" in res and (store.get_project(pid)["metadata"] or {}).get("autopilot")


async def test_a_vanished_current_project_gets_no_preview(context, store):
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    context.current_project_id = "0123456789ab"
    res = await tool_manage_projects_for_model(context, action="delete")
    assert "no longer exists" in res


async def test_one_owner_step_takes_back_exactly_the_task_it_runs(store):
    from ghost_agent.core.project_advancer import advance_once, _HELD_MARK
    pid = store.create_project("App")
    store.update_project(pid, status="ACTIVE")
    build = store.add_task(pid, "Build the settings page")          # ready, and FIRST in order
    v1 = store.add_task(pid, "Verify the login flow")
    v2 = store.add_task(pid, "Verify the signup flow")
    for t in (v1, v2):
        store.update_task(t, status="NEEDS_USER", result_summary=f"wrote x — needs a real check — {_HELD_MARK}")
    ctx = SimpleNamespace(project_store=store, workspace_model=None, current_project_id=None)

    async def runner(name, args):
        return "all checks passed"
    res = await advance_once(ctx, pid, tool_runner=runner, owner_requested=True)
    assert res.task_id == build                                     # held tasks join the normal order
    assert {store.get_task(t)["status"] for t in (v1, v2)} == {"NEEDS_USER"}
    res = await advance_once(ctx, pid, tool_runner=runner, owner_requested=True)
    assert res.task_id == v1 and store.get_task(v1)["status"] == "DONE"
    assert store.get_task(v2)["status"] == "NEEDS_USER"            # not reopened for the idle loop
    await advance_once(ctx, pid, tool_runner=runner)                # the idle loop never touches it
    assert store.get_task(v2)["status"] == "NEEDS_USER"


async def test_a_batch_that_stops_on_held_tasks_says_they_need_the_owner(store):
    from ghost_agent.core.project_advancer import advance_many, _HELD_MARK
    pid = store.create_project("App")
    store.update_project(pid, status="ACTIVE")
    v = store.add_task(pid, "Verify the full pipeline")
    store.update_task(v, status="NEEDS_USER", result_summary=f"wrote x — needs a real check — {_HELD_MARK}")
    store.add_task(pid, "Ship the release notes", depends_on=[v])    # open, but waits on the held task
    store.update_project(pid, status="ACTIVE")
    ctx = SimpleNamespace(project_store=store, workspace_model=None, current_project_id=None)

    async def runner(name, args):
        return "ok"
    res = await advance_many(ctx, pid, max_tasks=None, tool_runner=runner)
    assert res.stop_reason == "needs_user", res


# ── §4LQ review round 3: a taken-back task must be READY ──

def _held(store, pid, desc, **kw):
    from ghost_agent.core.project_advancer import _HELD_MARK
    t = store.add_task(pid, desc, **kw)
    store.update_task(t, status="NEEDS_USER", result_summary=f"wrote x — needs a real check — {_HELD_MARK}")
    return t


async def test_a_held_task_waits_for_its_dependency_on_the_owners_run(store):
    from ghost_agent.core.project_advancer import advance_once
    pid = store.create_project("App")
    store.update_project(pid, status="ACTIVE")
    build = store.add_task(pid, "Build the API")
    v = _held(store, pid, "Verify the API", depends_on=[build])
    ctx = SimpleNamespace(project_store=store, workspace_model=None, current_project_id=None)

    async def runner(name, args):
        return "### 1. Fixture notes\nThe fixture loads three sample rows.\n[Source: https://example.org/a]"
    res = await advance_once(ctx, pid, tool_runner=runner, owner_requested=True)
    assert res.task_id == build and store.get_task(v)["status"] == "NEEDS_USER"


async def test_a_held_parent_is_not_run_over_its_open_subtask(store):
    from ghost_agent.core.project_advancer import advance_once
    pid = store.create_project("App")
    store.update_project(pid, status="ACTIVE")
    parent = _held(store, pid, "Verify the pipeline")
    child = store.add_task(pid, "Research common pipeline fixture formats", parent_id=parent)
    ctx = SimpleNamespace(project_store=store, workspace_model=None, current_project_id=None)

    async def runner(name, args):
        return "### 1. Fixture notes\nThe fixture loads three sample rows.\n[Source: https://example.org/a]"
    res = await advance_once(ctx, pid, tool_runner=runner, owner_requested=True)
    # the subtask runs, not the parent over it (the parent then closes by the
    # tree's normal rollup, inside the owner's own run)
    assert res.task_id == child and store.get_task(child)["status"] == "DONE"


async def test_a_held_task_under_a_paused_phase_stays_put(store):
    from ghost_agent.core.project_advancer import advance_once
    pid = store.create_project("App")
    store.update_project(pid, status="ACTIVE")
    phase = store.add_task(pid, "Phase 2")
    v = _held(store, pid, "Verify the export", parent_id=phase)
    store.update_task(phase, status="PAUSED")
    store.add_task(pid, "Write the README")
    ctx = SimpleNamespace(project_store=store, workspace_model=None, current_project_id=None)

    async def runner(name, args):
        return "### 1. Fixture notes\nThe fixture loads three sample rows.\n[Source: https://example.org/a]"
    res = await advance_once(ctx, pid, tool_runner=runner, owner_requested=True)
    assert res.task_id != v and store.get_task(v)["status"] == "NEEDS_USER"
    assert store.get_task(phase)["status"] == "PAUSED"


async def test_a_needs_user_stop_names_the_waiting_tasks(context, store, monkeypatch):
    import ghost_agent.core.project_advancer as PA
    pid = store.create_project("App")
    t = _held(store, pid, "Verify the pipeline")

    async def fake_many(ctx, p, **kw):
        return PA.AdvanceManyResult([], "needs_user", None)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    res = json.loads(await tool_manage_projects(context, action="autoadvance", project_id=pid))
    assert [x["task_id"] for x in res["needs_user_tasks"]] == [t]


# ── the check rule errs toward holding (§4LQ, after four review rounds) ──

@pytest.mark.parametrize("desc,held", [
    # checks, in every shape the reviews found — held
    ("Integration & Polish: Verify full pipeline", True), ("Test the login flow", True),
    ("Ensure all tests pass", True), ("Verify data integrity", True), ("Test button states", True),
    ("Confirm dialog closes on escape", True), ("Write and run tests for the API", True),
    ("Final check of the deliverables", True), ("Final verification", True), ("QA: everything", True),
    ("Run the test suite", True), ("Smoke-test the API", True), ("Make sure it builds", True),
    ("1. Verify the app", True), ("Re-test the form", True), ("End-to-end testing", True),
    ("Sanity-check the data", True), ("Ensure the export works", True),
    # whole words only — not checks
    ("Checkout page with Stripe", False), ("Testimonials section", False), ("Check-in screen UI", False),
    ("Latest news widget", False), ("Build the settings page", False), ("Run the server", False)])
def test_a_task_that_mentions_a_check_is_held_when_unattended(desc, held):
    from ghost_agent.core.planning import TaskStatus
    from ghost_agent.core.project_advancer import _unattended_close
    assert (_unattended_close(False, desc)[0] == TaskStatus.NEEDS_USER) is held


def test_the_check_rule_is_fast_on_long_text():
    import time
    from ghost_agent.core.project_advancer import _unattended_close
    t = time.monotonic()
    _unattended_close(False, "- " * 100_000)
    assert time.monotonic() - t < 0.5


async def test_one_preview_per_turn_and_a_refused_confirm_says_stop(context, store):
    from ghost_agent.tools.projects import tool_manage_projects_for_model
    from ghost_agent.utils.logging import request_id_context
    pid = store.create_project("Shipped")
    store.update_project(pid, status="RELEASED")
    tok = request_id_context.set("probe-loop")
    try:
        prev = json.loads(await tool_manage_projects_for_model(context, action="unrelease", project_id=pid))
        again = await tool_manage_projects_for_model(context, action="unrelease", project_id=pid)
        conf = await tool_manage_projects_for_model(context, action="unrelease", project_id=pid,
                                                    confirm_token=prev["confirm_token"])
    finally:
        request_id_context.reset(tok)
    assert "already previewed" in again and "STOP" in again
    assert "NOT done" in conf and "STOP" in conf
    assert store.get_project(pid)["status"] == "RELEASED"
