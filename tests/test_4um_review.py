"""§4UM fresh review: the who-may-act wall on a ready project (M1); the agent-owned PROJECT_CHANGES.md, never a
project's own CHANGELOG.md (M2); the folder diff that sees a command's writes, deletes and moves (M3/M4); and
the minors (the whole reply as summary, legacy auto-closed = ready, live marking, task-turn entries, the owner's
ask, the notify promise on a ready project, a killed request's entries, the backfill, the owner's close)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core.prompts import build_project_briefing
from ghost_agent.core.request_scope import RequestScope, install, reset
from ghost_agent.memory import projects as MP
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools import labs as LB
from ghost_agent.tools import projects as TP
from ghost_agent.tools.file_system import tool_file_system
from ghost_agent.utils.logging import request_id_context

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean():
    MP._PENDING_CHANGES.clear()
    MP._REQUEST_ASK.clear()
    LB._KEPT_LAB_SEEN.clear()
    yield
    MP._PENDING_CHANGES.clear()
    MP._REQUEST_ASK.clear()
    LB._KEPT_LAB_SEEN.clear()


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def ctx(tmp_path, store):
    return SimpleNamespace(project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                           graph_memory=None, contradiction_log=None, current_project_id=None,
                           llm_client=None, workspace_model=None, sandbox_dir=str(tmp_path / "sb"),
                           last_user_content="")


def _built(store, title="Site"):
    pid = store.create_project(title)
    ws = Path(store.ensure_workspace(pid))
    (ws / "index.html").write_text("<h1>hi</h1>")
    t = store.add_task(pid, "Build index.html")
    tok = request_id_context.set("req-build")
    try:
        store.update_task(t, status="DONE", result_summary="built")
        store.register_file_artifact(t, "index.html")
    finally:
        request_id_context.reset(tok)
    MP.flush_deferred_sweeps("req-build")
    return pid, ws


class _Owner:
    """An owner request `rid` with the request scope installed (what handle_chat does)."""

    def __init__(self, ctx, rid, ask):
        self.ctx, self.rid, self.ask = ctx, rid, ask

    def __enter__(self):
        self.ctx.last_user_content = self.ask
        self.t1 = request_id_context.set(self.rid)
        self.t2 = install(RequestScope(context=self.ctx))
        TP.begin_request_watch(self.ctx, self.rid, self.ask)
        return self

    def __exit__(self, *exc):
        reset(self.t2)
        request_id_context.reset(self.t1)


def _sh(ws, script):
    subprocess.run(["bash", "-c", script], cwd=ws, check=True)


# ───────────────────────────── M1: the wall on a ready project ─────────────────────────────

@pytest.mark.parametrize("act", ["release", "delete"])
async def test_a_background_turn_cannot_release_or_delete_a_ready_project(ctx, store, act):
    pid, _ = _built(store)
    ctx.current_project_id = pid
    tok = request_id_context.set("sched-autopilot-9")
    try:
        out = await TP.tool_manage_projects_for_model(ctx, action=act, project_id=pid,
                                                      directions="Open index.html in a browser. It shows the page.")
    finally:
        request_id_context.reset(tok)
    assert "only the user" in str(out) and store.get_project(pid)["status"] == "ACTIVE"


async def test_a_request_that_read_outside_content_cannot_release_a_ready_project(ctx, store, monkeypatch):
    import ghost_agent.utils.provenance as PV
    pid, _ = _built(store)
    ctx.current_project_id = pid
    monkeypatch.setattr(PV, "untrusted_seen", lambda rid: ["web page"])
    tok = request_id_context.set("req-owner-web")
    try:
        out = await TP.tool_manage_projects_for_model(ctx, action="release", project_id=pid,
                                                      directions="Open index.html in a browser. It shows the page.")
    finally:
        request_id_context.reset(tok)
    assert "outside content" in str(out) and store.get_project(pid)["status"] == "ACTIVE"


# ───────────────────────────── M2: the agent's own file ─────────────────────────────

def test_a_project_s_own_changelog_is_content_and_the_log_has_its_own_file(ctx, store):
    pid, ws = _built(store)
    (ws / "CHANGELOG.md").write_text("# My project's own changelog\n- v1.0 shipped\n")
    with _Owner(ctx, "req-own-1", "add tetris and note it in the changelog"):
        (ws / "tetris.js").write_text("x")
        TP.note_project_write(ctx, ws / "tetris.js")
        (ws / "CHANGELOG.md").write_text("# My project's own changelog\n- v1.0 shipped\n- tetris\n")
        TP.note_project_write(ctx, ws / "CHANGELOG.md")
    MP.flush_change_log("req-own-1", reply="Added Tetris.")
    assert "v1.0 shipped" in (ws / "CHANGELOG.md").read_text()
    (e,) = store.list_changes(pid)
    assert e["added"] == ["tetris.js"] and e["changed"] == ["CHANGELOG.md"]
    own = (ws / "PROJECT_CHANGES.md").read_text()
    assert own.startswith(MP.CHANGES_FILE_HEADER) and "add tetris" in own
    assert not [p for p in ws.iterdir() if p.name.endswith(".tmp")]


def test_a_foreign_project_changes_file_is_never_overwritten(ctx, store):
    pid, ws = _built(store)
    (ws / "PROJECT_CHANGES.md").write_text("the project's own notes\n")
    with _Owner(ctx, "req-own-2", "add a"):
        (ws / "a.js").write_text("1")
        TP.note_project_write(ctx, ws / "a.js")
    MP.flush_change_log("req-own-2", reply="ok")
    assert (ws / "PROJECT_CHANGES.md").read_text() == "the project's own notes\n"
    assert len(store.list_changes(pid)) == 1                              # the store still has it


def test_the_file_renders_the_last_200_entries_and_a_deploy_leaves_it_out(store):
    pid, ws = _built(store)
    for i in range(205):
        store.add_change(pid, ts=1000.0 + i, ask=f"change {i}", added=[f"f{i}.js"], render=False)
    store.render_changelog(pid)
    txt = (ws / "PROJECT_CHANGES.md").read_text()
    assert "Asked: change 204\n" in txt and "Asked: change 5\n" in txt and "Asked: change 4\n" not in txt
    assert txt.count("\n## #") == 200
    assert "PROJECT_CHANGES.md" in LB.DEPLOY_ARTEFACTS_TOP


# ───────────────────────────── M3/M4: the folder diff ─────────────────────────────

def test_shell_writes_deletes_and_moves_in_a_focused_project_are_one_entry(ctx, store):
    pid, ws = _built(store)
    (ws / "old.js").write_text("old")
    (ws / "gone.js").write_text("bye")
    time.sleep(0.01)
    ctx.current_project_id = pid
    with _Owner(ctx, "req-sh-1", "rework the scripts"):
        TP.prepare_project_write(ctx, project_id=pid)                   # what execute does before it runs
        _sh(ws, "cat > new.js <<'EOF'\nconsole.log(1)\nEOF\nsed -i.bak 's/hi/hello/' index.html && rm index.html.bak\n"
                "rm gone.js\nmv old.js renamed.js")
    assert MP.flush_change_log("req-sh-1", reply="Reworked.") == 1
    (e,) = store.list_changes(pid)
    assert e["added"] == ["new.js", "renamed.js"] and e["changed"] == ["index.html"]
    assert e["removed"] == ["gone.js", "old.js"] and e["ask"] == "rework the scripts"


def test_a_closed_project_reopens_only_at_a_write_tool_s_path_never_from_the_diff(ctx, store):
    pid, ws = _built(store)
    store.update_project(pid, status="DONE")
    ctx.current_project_id = pid
    with _Owner(ctx, "req-sh-2", "fix the title"):
        TP.prepare_project_write(ctx, project_id=pid)
        _sh(ws, "printf '<h1>fixed</h1>' > index.html")
    MP.flush_change_log("req-sh-2", reply="Fixed.")
    assert store.get_project(pid)["status"] == "DONE"                  # the diff alone: logged, not reopened
    assert store.list_changes(pid)[0]["changed"] == ["index.html"]
    with _Owner(ctx, "req-sh-3", "fix it again"):
        (ws / "index.html").write_text("<h1>again</h1>")
        TP.note_project_write(ctx, ws / "index.html")                  # the file tool's noted write
    MP.flush_change_log("req-sh-3", reply="Fixed.")
    assert store.get_project(pid)["status"] == "ACTIVE"
    assert store.list_changes(pid)[0]["meta"]["reopened_from"] == "DONE"


async def test_file_system_delete_and_rename_record_the_source_and_the_destination(ctx, store, tmp_path,
                                                                                   monkeypatch):
    import ghost_agent.tools.file_system as FS
    monkeypatch.setattr(FS, "unread_delete_refusal", lambda *a, **k: None)    # §4TJ: not what this tests
    pid, ws = _built(store)
    (ws / "js").mkdir()
    (ws / "js" / "tetris.js").write_text("t")
    (ws / "a.js").write_text("a")
    time.sleep(0.01)
    sb = tmp_path / "sb"
    with _Owner(ctx, "req-fs-1", "remove tetris and rename a"):         # nothing focused: the hooks watch it
        await tool_file_system(operation="delete", sandbox_dir=sb, path=f"projects/{pid}/js/tetris.js")
        await tool_file_system(operation="rename", sandbox_dir=sb, path=f"projects/{pid}/a.js",
                               destination=f"projects/{pid}/b.js")
    assert not (ws / "js" / "tetris.js").exists() and (ws / "b.js").exists()
    MP.flush_change_log("req-fs-1", reply="Tetris removed, a renamed.")
    (e,) = store.list_changes(pid)
    assert e["removed"] == ["a.js", "js/tetris.js"] and e["added"] == ["b.js"]


def test_execute_snapshots_the_pinned_project_and_named_projects_before_it_runs(ctx, store, monkeypatch):
    from ghost_agent.tools.execute import _prepare_execute_project_writes
    pid, ws = _built(store)
    other, ows = _built(store, "Other")
    seen = []
    monkeypatch.setattr(TP, "prepare_project_write", lambda c, path=None, project_id="", **k: seen.append(project_id))
    with _Owner(ctx, "req-ex-1", "x"):
        _prepare_execute_project_writes({"container_workdir": f"/workspace/projects/{pid}",
                                         "command": f"rm projects/{other}/index.html; echo ok"})
    assert seen == [pid, other]


# ───────────────────────────── minors ─────────────────────────────

def test_a_streamed_reply_fills_the_summary_after_the_entry_was_written(ctx, store):
    pid, ws = _built(store)
    with _Owner(ctx, "req-st-1", "add a"):
        (ws / "a.js").write_text("1")
        TP.note_project_write(ctx, ws / "a.js")
    MP.flush_change_log("req-st-1", reply="")                            # handle_chat's finally: no text yet
    assert store.list_changes(pid)[0]["summary"] == ""
    assert MP.set_change_summary("req-st-1", "## Done\nAdded **a.js** with the counter.") == 1
    assert store.list_changes(pid)[0]["summary"] == "Done Added a.js with the counter."
    assert "Added a.js with the counter" in (ws / "PROJECT_CHANGES.md").read_text()


def _legacy_done(store):
    pid, ws = _built(store)
    with store._lock, store._connect() as conn:          # what the pre-§4UM roll-up did
        conn.execute("UPDATE projects SET status = 'DONE' WHERE id = ?", (pid,))
        conn.commit()
    store.log_event(pid, None, "project_auto_rollup", {"new_status": "DONE", "had_failures": False})
    return pid, ws


async def test_a_legacy_auto_closed_project_reads_ready_and_the_owner_s_close_is_closed(ctx, store):
    pid, ws = _legacy_done(store)
    assert store.project_state(pid) == "ready" and store.planned_work_done(pid)
    b = build_project_briefing(store, pid)
    assert "state READY" in b and "CLOSED by the user" not in b
    ls = json.loads(str(await TP.tool_manage_projects(ctx, action="list")))
    assert "ready" in ls["display"]
    store.update_project(pid, status="ACTIVE")
    tok = request_id_context.set("req-owner-close-2")
    try:
        out = json.loads(str(await TP.tool_manage_projects(ctx, action="update", project_id=pid, status="DONE")))
    finally:
        request_id_context.reset(tok)
    assert "closed by the owner's request" in out["note"]
    assert store.project_state(pid) == "done" and store.list_events(pid, event_type="project_closed")


def test_a_legacy_project_s_next_change_is_logged(ctx, store):
    pid, ws = _legacy_done(store)
    with _Owner(ctx, "req-lg-1", "add tetris"):
        (ws / "t.js").write_text("t")
        TP.note_project_write(ctx, ws / "t.js")
    MP.flush_change_log("req-lg-1", reply="ok")
    assert [e["added"] for e in store.list_changes(pid)] == [["t.js"]]


async def test_a_put_outside_the_app_workdir_or_member_is_not_live(ctx, store, monkeypatch):
    pid, ws = _built(store)

    async def fake_call(context, method, path, **kw):
        return 200, {"owner": f"project:{pid}", "lifetime": "kept",
                     "app": {"name": "site", "member": "web", "workdir": "/app", "url": "https://l.example/"}}
    monkeypatch.setattr(LB, "_call", fake_call)
    with _Owner(ctx, "req-pt-1", "edit index"):
        (ws / "index.html").write_text("new")
        TP.note_project_write(ctx, ws / "index.html")
        await LB._note_project_push(ctx, "owner", "lab_00000011", ws / "index.html", how="put",
                                    member="web", path="/tmp/index.html")
        await LB._note_project_push(ctx, "owner", "lab_00000011", ws / "index.html", how="put",
                                    member="db", path="/app/index.html")
    MP.flush_change_log("req-pt-1", reply="ok")
    assert not store.list_changes(pid, kind="change")[0]["deployed"] and store.live_info(pid) is None


def test_a_subfolder_deploy_marks_every_change_and_a_removal_only_change(ctx, store):
    pid, ws = _built(store)
    (ws / "x.js").write_text("x")
    time.sleep(0.01)
    ctx.current_project_id = pid
    with _Owner(ctx, "req-rm-1", "drop x"):
        TP.prepare_project_write(ctx, project_id=pid)
        _sh(ws, "rm x.js")
    MP.flush_change_log("req-rm-1", reply="ok")
    assert not store.list_changes(pid, kind="change")[0]["deployed"]
    MP.note_change_push(store, pid, None, url="https://l.example/", how="deploy", rid="")
    assert store.list_changes(pid, kind="change")[0]["deployed"]


def test_an_unattended_task_turn_after_ready_is_its_own_task_entry(ctx, store):
    pid, ws = _built(store)
    t = store.add_task(pid, "add the score board")
    rid = "sub-ptask-abc-1"
    MP.set_request_parent(rid, "sched-tick-7")
    assert MP._was_ready_before(store, pid, rid, time.time())
    MP.watch_project(store, pid, key=rid, ask="task: add the score board", source="task", task_id=t)
    tok = request_id_context.set(rid)
    try:
        TP.prepare_project_write(ctx, project_id=pid)                   # the task turn's execute, unattended
        _sh(ws, "echo s > score.js")
        store.update_task(t, status="DONE", result_summary="ok")
    finally:
        request_id_context.reset(tok)
    MP.flush_change_log(rid, reply="Score board added.")
    (e,) = store.list_changes(pid)
    assert e["ask"] == "task: add the score board" and e["meta"]["source"] == "task" and e["added"] == ["score.js"]


def test_the_first_plan_s_unattended_turn_is_not_watched(store):
    pid = store.create_project("Fresh")
    store.add_task(pid, "build")
    assert not MP._was_ready_before(store, pid, "sub-ptask-x-2", time.time())


def test_the_ask_is_the_owner_s_message_not_the_task_turn_s_prompt(ctx, store, monkeypatch):
    pid, ws = _built(store)
    MP.note_request_ask("req-root-9", "please add a scoreboard")
    MP.set_request_parent("sub-ptask-q-1", "req-root-9")
    MP.note_request_ask("sub-ptask-q-1", "TASK PROMPT: build the scoreboard")    # a sub-request: ignored
    monkeypatch.setattr(TP, "_owner_task_turn", lambda context=None: True)
    ctx.last_user_content = "TASK PROMPT: build the scoreboard"
    tok = request_id_context.set("sub-ptask-q-1")
    try:
        (ws / "sb.js").write_text("1")
        TP.note_project_write(ctx, ws / "sb.js")
    finally:
        request_id_context.reset(tok)
    MP.flush_change_log("req-root-9", reply="ok")
    assert store.list_changes(pid)[0]["ask"] == "please add a scoreboard"


def test_a_notify_ask_on_a_ready_project_fires_on_the_change_not_at_once(ctx, store, tmp_path, monkeypatch):
    from ghost_agent.core import notify_promise as NP
    pid, ws = _built(store)
    assert NP.capture_promise(store, pid, "add a and notify me when done", req_id="req-n-1")
    fired = []
    monkeypatch.setattr(NP, "_ready_settled", NP._ready_settled)
    alog = SimpleNamespace(record=lambda *a, **k: fired.append(a) or True)
    import ghost_agent.tools.notify_tool as NT
    monkeypatch.setattr(NT, "_rate_limited", lambda: False)
    monkeypatch.setattr(NT, "_note_sent", lambda *a, **k: None, raising=False)
    assert NP.fire_promise_if_settled(store, alog, pid) is False         # nothing changed since the ask
    with _Owner(ctx, "req-n-1", "add a and notify me when done"):
        (ws / "a.js").write_text("1")
        TP.note_project_write(ctx, ws / "a.js")
    MP.flush_change_log("req-n-1", reply="ok")
    assert NP.fire_promise_if_settled(store, alog, pid) is True and fired


def test_a_killed_request_s_entry_is_written_at_the_next_start(ctx, store, tmp_path):
    pid, ws = _built(store)
    with _Owner(ctx, "req-kill-1", "add k"):
        (ws / "k.js").write_text("1")
        TP.note_project_write(ctx, ws / "k.js")
    assert (tmp_path / "mem" / MP.PENDING_FILE).exists()
    MP._PENDING_CHANGES.clear()                                          # the process died
    fresh = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")
    assert MP.recover_pending_changes(fresh) == 1
    (e,) = fresh.list_changes(pid)
    assert e["added"] == ["k.js"] and e["meta"]["recovered"] and not (tmp_path / "mem" / MP.PENDING_FILE).exists()


def test_the_backfill_is_idempotent_and_an_older_deploy_never_overwrites_a_newer_one(store):
    pid, ws = _built(store)
    MP.note_change_push(store, pid, None, url="https://new.example/", how="deploy", rid="")
    e1 = MP.backfill_change(store, pid, "add a game", added=["g.js"], ts=time.time() - 86400,
                            deployed=True, url="https://old.example/")
    again = MP.backfill_change(store, pid, "add a game", added=["g.js"], ts=float(e1["ts"]) + 5)
    assert again["meta"].get("existing") and len(store.list_changes(pid, kind="change")) == 1
    assert store.live_info(pid)["url"] == "https://new.example/"
    store.add_change(pid, ts=time.time() - 50000, ask="older", added=["o.js"])
    e3 = MP.backfill_change(store, pid, "marked alone", added=["m.js"], ts=time.time() - 40000, deployed=True)
    by = {e["ask"]: e["deployed"] for e in store.list_changes(pid, kind="change")}
    assert by["marked alone"] is True and by["older"] is False and e3["deployed"]


def test_the_backfill_dry_run_changes_nothing_in_the_database(tmp_path, store):
    pid, _ = _built(store)
    db = tmp_path / "mem" / "projects.db"
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "backfill_project_change.py"),
                          "--memory-dir", str(tmp_path / "mem"), "--sandbox-root", str(tmp_path / "sb"),
                          "--project", pid, "--ask", "x", "--added", "a.js", "--ts", "2026-11-12 19:57"],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0 and "dry run" in run.stdout, run.stderr
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_a_defect_report_on_a_ready_project_is_recorded(store):
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.agent import GhostAgent
    pid, _ = _built(store)
    mctx = MagicMock()
    mctx.llm_client.chat_completion = AsyncMock()
    mctx.args.smart_memory = 0.0
    agent = GhostAgent(mctx)
    agent.context = SimpleNamespace(project_store=store, current_project_id=pid)
    assert agent._note_defect_on_done_project("the page is blank after the change") is True
    assert any(t["description"].startswith("FIX (defect):") for t in store.list_tasks(pid))


# ───────────────────────────── round 2: what is not a change ─────────────────────────────

def test_a_screenshot_in_a_request_that_wrote_nothing_is_no_entry_and_no_reopen(ctx, store):
    from ghost_agent.tools.turn_scratch import note_created
    pid, ws = _built(store)
    store.update_project(pid, status="DONE")
    ctx.current_project_id = pid
    with _Owner(ctx, "req-r2-1", "how does the site look?"):
        note_created(ws / "screenshot.png", "screenshot")             # what the browser does before it writes
        (ws / "screenshot.png").write_bytes(b"png")
    assert MP.flush_change_log("req-r2-1", reply="It looks fine.") == 0
    assert store.list_changes(pid) == [] and store.get_project(pid)["status"] == "DONE"


def test_a_screenshot_beside_a_real_edit_is_left_out(ctx, store):
    from ghost_agent.tools.turn_scratch import note_created
    pid, ws = _built(store)
    ctx.current_project_id = pid
    with _Owner(ctx, "req-r2-2", "fix the header and show me"):
        TP.prepare_project_write(ctx, project_id=pid)
        _sh(ws, "printf '<h1>new</h1>' > index.html")
        note_created(ws / "shot.png", "screenshot")
        (ws / "shot.png").write_bytes(b"png")
    MP.flush_change_log("req-r2-2", reply="ok")
    (e,) = store.list_changes(pid)
    assert e["changed"] == ["index.html"] and e["added"] == []


def test_a_sweep_during_a_writing_request_is_not_a_removal(ctx, store, tmp_path):
    from ghost_agent.tools.turn_scratch import _move_to_trash
    pid, ws = _built(store)
    (ws / "screenshot.png").write_bytes(b"png")
    time.sleep(0.01)
    ctx.current_project_id = pid
    with _Owner(ctx, "req-r2-3", "add a"):
        TP.prepare_project_write(ctx, project_id=pid)
        _sh(ws, "echo 1 > a.js")
        assert _move_to_trash(ws / "screenshot.png", tmp_path / "sb", tmp_path / "trash", "req-old")  # 30-min sweep
    MP.flush_change_log("req-r2-3", reply="ok")
    (e,) = store.list_changes(pid)
    assert e["added"] == ["a.js"] and e["removed"] == []


def test_a_sweep_in_a_read_only_request_is_no_entry(ctx, store, tmp_path):
    from ghost_agent.tools.turn_scratch import _move_to_trash
    pid, ws = _built(store)
    (ws / "screenshot.png").write_bytes(b"png")
    ctx.current_project_id = pid
    with _Owner(ctx, "req-r2-4", "what does the footer say?"):
        _move_to_trash(ws / "screenshot.png", tmp_path / "sb", tmp_path / "trash", "req-old")
    assert MP.flush_change_log("req-r2-4", reply="It says X.") == 0 and store.list_changes(pid) == []


def test_a_research_question_is_no_change_and_the_mirror_is_left_out_of_a_real_one(ctx, store, monkeypatch):
    from ghost_agent.core import project_research as PR
    pid, ws = _built(store)
    ctx.current_project_id = pid
    monkeypatch.setattr(PR, "request_relevant_to_project", lambda *a, **k: True)
    out = "### 1. Title\nbody text\n[Source: https://example.com/a]\n"
    with _Owner(ctx, "req-r2-5", "what's the latest flask version?"):
        assert PR.record_main_loop_findings(store, pid, "latest flask version", out)
    assert MP.flush_change_log("req-r2-5", reply="3.1") == 0
    with _Owner(ctx, "req-r2-6", "upgrade flask in requirements"):
        TP.prepare_project_write(ctx, project_id=pid)
        assert PR.record_main_loop_findings(store, pid, "flask 3.1 changes", out)
        _sh(ws, "echo flask==3.1 > requirements.txt")
    MP.flush_change_log("req-r2-6", reply="ok")
    (e,) = store.list_changes(pid)
    assert e["added"] == ["requirements.txt"] and not [p for p in e["added"] + e["changed"] if "research" in p]


def test_a_workspace_sweep_s_removal_is_not_the_owner_s(ctx, store):
    from ghost_agent.core.workspace_cleanup import _note_swept
    pid, ws = _built(store)
    (ws / "debris.log").write_text("x")
    ctx.current_project_id = pid
    with _Owner(ctx, "req-r2-7", "add b"):
        TP.prepare_project_write(ctx, project_id=pid)
        _sh(ws, "echo 1 > b.js")
        _note_swept(ws / "debris.log")
        (ws / "debris.log").unlink()
    MP.flush_change_log("req-r2-7", reply="ok")
    assert store.list_changes(pid)[0]["removed"] == []


def test_a_real_execute_edit_in_a_pinned_project_is_still_logged(ctx, store):
    from ghost_agent.tools.execute import _prepare_execute_project_writes
    pid, ws = _built(store)
    with _Owner(ctx, "req-r2-8", "make the title bigger"):
        _prepare_execute_project_writes({"container_workdir": f"/workspace/projects/{pid}",
                                         "command": "sed -i.bak 's/h1/h2/g' index.html && rm index.html.bak"})
        _sh(ws, "sed -i.bak 's/h1/h2/g' index.html && rm index.html.bak")
    MP.flush_change_log("req-r2-8", reply="Bigger.")
    (e,) = store.list_changes(pid)
    assert e["changed"] == ["index.html"]


# ───────────────────────────── round 2 minors ─────────────────────────────

def test_another_request_s_write_and_an_earlier_change_are_not_this_request_s(ctx, store):
    pid, ws = _built(store)
    ctx.current_project_id = pid
    with _Owner(ctx, "req-r2-9", "add c"):
        _sh(ws, "echo early > early.js")                               # before this request's first write tool
        time.sleep(1.2)
        TP.prepare_project_write(ctx, project_id=pid)
        _sh(ws, "echo 1 > c.js")
        (ws / "other.js").write_text("theirs")                         # a concurrent task turn's file tool
        MP.note_change_write(store, pid, "other.js", rid="req-concurrent-1")
    MP.flush_change_log("req-r2-9", reply="ok")
    (e,) = store.list_changes(pid)
    assert e["added"] == ["c.js"]


def test_a_file_first_added_stays_added_after_later_writes(ctx, store):
    pid, ws = _built(store)
    with _Owner(ctx, "req-r2-10", "add d"):
        (ws / "d.js").write_text("1")
        TP.note_project_write(ctx, ws / "d.js")                        # first touch: snapshot AFTER the write
        TP.prepare_project_write(ctx, project_id=pid)
        _sh(ws, "echo 2 >> d.js")
    MP.flush_change_log("req-r2-10", reply="ok")
    (e,) = store.list_changes(pid)
    assert e["added"] == ["d.js"] and e["changed"] == []


def test_the_snapshot_counts_folders_skips_build_caches_and_remembers_too_large(store, monkeypatch):
    pid, ws = _built(store)
    for d in ("env", ".tox", "target"):
        (ws / d).mkdir()
        (ws / d / "x").write_text("1")
    (ws / "dist").mkdir()
    (ws / "dist" / "app.js").write_text("1")
    snap = MP.snapshot_folder(ws)
    assert "dist/app.js" in snap and "env/x" in snap and "target/x" in snap and ".tox/x" not in snap
    (ws / "env" / "pyvenv.cfg").write_text("home = /usr")                # round 3: a virtualenv, a cargo dir
    (ws / "target" / "CACHEDIR.TAG").write_text("Signature")
    snap = MP.snapshot_folder(ws)
    assert not [r for r in snap if r.split("/")[0] in ("env", ".tox", "target")]
    for i in range(6):
        (ws / f"d{i}").mkdir()
    assert MP.snapshot_folder(ws, max_files=len(snap) + 2) is None    # empty folders count toward the cap
    monkeypatch.setattr(MP, "SNAPSHOT_MAX_FILES", 3)
    calls = []
    real = MP.snapshot_folder
    monkeypatch.setattr(MP, "snapshot_folder", lambda root, *a, **k: calls.append(1) or real(root, 3))
    MP.watch_project(store, pid, key="req-big-1")
    MP.watch_project(store, pid, key="req-big-2")
    assert len(calls) == 1 and MP._PENDING_CHANGES["req-big-2"][f"{id(store)}:{pid}"]["snap"] == "too_large"
    MP._TOO_LARGE.clear()


def test_a_failed_write_stays_pending_and_the_next_flush_writes_it(ctx, store, tmp_path, monkeypatch):
    pid, ws = _built(store)
    with _Owner(ctx, "req-r2-11", "add f"):
        (ws / "f.js").write_text("1")
        TP.note_project_write(ctx, ws / "f.js")
    real, calls = store.add_change, []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("db busy")
        return real(*a, **k)
    monkeypatch.setattr(store, "add_change", flaky)
    assert MP.flush_change_log("req-r2-11", reply="ok") == 0
    assert (tmp_path / "mem" / MP.PENDING_FILE).exists() and "req-r2-11" in MP._PENDING_CHANGES
    assert MP.flush_change_log("req-r2-11", reply="ok") == 1
    assert len(store.list_changes(pid)) == 1 and not (tmp_path / "mem" / MP.PENDING_FILE).exists()


def test_recovery_skips_what_is_logged_and_moves_a_corrupt_file_aside(ctx, store, tmp_path, caplog):
    pid, ws = _built(store)
    with _Owner(ctx, "req-r2-12", "add g"):
        (ws / "g.js").write_text("1")
        TP.note_project_write(ctx, ws / "g.js")
    pending = (tmp_path / "mem" / MP.PENDING_FILE).read_text()
    MP.flush_change_log("req-r2-12", reply="ok")                       # written; then killed before the file went
    (tmp_path / "mem" / MP.PENDING_FILE).write_text(pending)
    assert MP.recover_pending_changes(store) == 0 and len(store.list_changes(pid)) == 1
    assert not (tmp_path / "mem" / MP.PENDING_FILE).exists()
    (tmp_path / "mem" / MP.PENDING_FILE).write_text("{not json")
    assert MP.recover_pending_changes(store) == 0
    assert (tmp_path / "mem" / (MP.PENDING_FILE + ".bad")).exists()
    assert not (tmp_path / "mem" / MP.PENDING_FILE).exists()


def test_a_recovered_row_that_fails_stays_on_disk(store, tmp_path, monkeypatch):
    pid, ws = _built(store)
    (ws / "h.js").write_text("1")
    row = {"pid": pid, "rid": "req-r2-13", "ask": "add h", "first": time.time(), "last": time.time(),
           "files": {"h.js": {"kind": "added", "seq": 1, "t": time.time()}}, "pushes": [], "reopened_from": "",
           "source": "owner", "wrote": True, "wrote_t": time.time()}
    (tmp_path / "mem" / MP.PENDING_FILE).write_text(json.dumps([row]))
    monkeypatch.setattr(store, "add_change", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db busy")))
    assert MP.recover_pending_changes(store) == 0
    assert (tmp_path / "mem" / MP.PENDING_FILE).exists()


def test_a_request_that_ran_no_writing_tool_is_never_diffed(ctx, store):
    """Whatever else moved the folder meanwhile (a dev server's file, the owner's own editor), a request that
    only read or searched writes no entry."""
    pid, ws = _built(store)
    ctx.current_project_id = pid
    with _Owner(ctx, "req-r2-14", "what port does it use?"):
        _sh(ws, "echo '{}' > server-state.json && printf x >> index.html")
    assert MP.flush_change_log("req-r2-14", reply="8080.") == 0 and store.list_changes(pid) == []


# ───────────────────────────── round 3 ─────────────────────────────

async def test_a_lab_get_into_a_project_is_a_change_reported_by_the_tool(ctx, store):
    pid, ws = _built(store)
    store.update_project(pid, status="DONE")
    with _Owner(ctx, "req-r3-1", "pull the config from the lab"):
        await LB._ws_write(ctx, ws / "conf" / "app.ini", b"[app]\n")
    MP.flush_change_log("req-r3-1", reply="Pulled.")
    (e,) = store.list_changes(pid)
    assert e["added"] == ["conf/app.ini"]
    assert store.get_project(pid)["status"] == "ACTIVE"                # reopened: the path came from the tool


async def test_a_rendered_image_then_an_execute_edit_is_one_entry(ctx, store):
    from tests.test_census_open_classes_4qj import _llm, _resp, PNG
    from ghost_agent.tools.execute import _note_execute_project_writes, _prepare_execute_project_writes
    from ghost_agent.tools.image_gen import tool_generate_image
    pid, ws = _built(store)
    store.update_project(pid, status="DONE")
    with _Owner(ctx, "req-r3-2", "make a logo and put it on the page"):
        out = await tool_generate_image(prompt="a logo", llm_client=_llm(renders=[_resp(seed=1, png=PNG)]),
                                        sandbox_dir=ws)
        assert "gen_" in str(out)
        assert store.get_project(pid)["status"] == "DONE"              # round 4: a render alone never reopens
        time.sleep(1.2)
        kw = {"container_workdir": f"/workspace/projects/{pid}", "command": "sed -i.bak 's/hi/logo/' index.html"}
        _prepare_execute_project_writes(kw)
        _sh(ws, "sed -i.bak 's/hi/logo/' index.html && rm index.html.bak")
        _note_execute_project_writes(kw, "ok")
    MP.flush_change_log("req-r3-2", reply="Logo on the page.")
    (e,) = store.list_changes(pid)
    assert len(e["added"]) == 1 and e["added"][0].startswith("gen_") and e["changed"] == ["index.html"]
    assert e["meta"]["reopened_from"] == "DONE"


def _walk_fails_at_the_flush(monkeypatch):
    real, calls = MP.snapshot_folder, []

    def fake(root, *a, **k):
        calls.append(1)
        return real(root) if len(calls) == 1 else None
    monkeypatch.setattr(MP, "snapshot_folder", fake)


def test_a_failed_flush_walk_keeps_the_pinned_command_s_reported_paths(ctx, store, monkeypatch):
    from ghost_agent.tools.execute import _note_execute_project_writes, _prepare_execute_project_writes
    pid, ws = _built(store)
    ctx.current_project_id = pid
    _walk_fails_at_the_flush(monkeypatch)
    kw = {"container_workdir": f"/workspace/projects/{pid}", "command": "sed -i 's/hi/yo/' index.html"}
    with _Owner(ctx, "req-r3-3", "change the heading"):
        _prepare_execute_project_writes(kw)
        (ws / "index.html").write_text("<h1>yo</h1>")
        _note_execute_project_writes(kw, "ok")
    MP.flush_change_log("req-r3-3", reply="Changed.")
    (e,) = store.list_changes(pid)
    assert e["changed"] == ["index.html"] and e["meta"]["folder_too_large"]


def test_a_failed_flush_walk_with_nothing_reported_is_still_an_entry(ctx, store, monkeypatch):
    from ghost_agent.tools.execute import _prepare_execute_project_writes
    pid, ws = _built(store)
    ctx.current_project_id = pid
    _walk_fails_at_the_flush(monkeypatch)
    with _Owner(ctx, "req-r3-4", "rebuild it"):
        _prepare_execute_project_writes({"container_workdir": f"/workspace/projects/{pid}",
                                         "command": "make > build.log"})              # write evidence, nothing noted
        (ws / "index.html").write_text("<h1>rebuilt</h1>")
    MP.flush_change_log("req-r3-4", reply="Rebuilt.")
    (e,) = store.list_changes(pid)
    assert e["meta"]["folder_too_large"] and e["added"] == e["changed"] == e["removed"] == []
    assert "not all listed" in (ws / "PROJECT_CHANGES.md").read_text() and "not all listed" in MP.change_line(e)
    assert time.time() - MP._TOO_LARGE[f"{id(store)}:{pid}"] < 60
    MP._TOO_LARGE.clear()


async def test_a_download_into_a_project_is_noted(ctx, store, monkeypatch):
    import ghost_agent.tools.file_system as FS
    pid, ws = _built(store)

    async def fake_impl(operation=None, sandbox_dir=None, path=None, **kw):
        (Path(sandbox_dir) / "a b.png").write_bytes(b"png")
        return "SUCCESS: downloaded"
    monkeypatch.setattr(FS, "_tool_file_system_impl", fake_impl)
    prepared = []
    real_prep = FS.prepare_project_write_hook
    monkeypatch.setattr(FS, "prepare_project_write_hook",
                        lambda sd, *ps: prepared.extend(ps) or real_prep(sd, *ps))
    with _Owner(ctx, "req-r3-5", "get the logo from the site"):
        await FS.tool_file_system(operation="download", sandbox_dir=ws, url="https://example.com/x/a%20b.png")
    MP.flush_change_log("req-r3-5", reply="ok")
    assert store.list_changes(pid)[0]["added"] == ["a b.png"]
    assert prepared == ["a b.png"]                                      # snapshotted BEFORE the download


def test_an_overwritten_screenshot_is_not_registered_as_the_system_s(ctx, store):
    from ghost_agent.tools.turn_scratch import note_created
    pid, ws = _built(store)
    (ws / "hero.png").write_bytes(b"owner's")
    with _Owner(ctx, "req-r3-6", "x"):
        note_created(ws / "hero.png", "screenshot")
        note_created(ws / "new.png", "screenshot")
    assert MP._abs_key(ws / "hero.png") not in MP._SYSTEM_PATHS
    assert MP._abs_key(ws / "new.png") in MP._SYSTEM_PATHS


def test_recovery_writes_the_part_a_kill_left_out(ctx, store, tmp_path):
    pid, ws = _built(store)
    (ws / "k.js").write_text("1")
    now = time.time()
    store.add_change(pid, ts=now, req_id="req-r3-7", ask="add k", added=["k.js"])   # the change, then killed
    row = {"pid": pid, "rid": "req-r3-7", "ask": "add k", "first": now, "last": now,
           "files": {"k.js": {"kind": "added", "seq": 1, "t": now}},
           "pushes": [{"seq": 2, "t": now + 1, "files": None, "url": "https://l.example/", "lab_id": "lab_00000077",
                       "app": "site", "how": "deploy", "before_t": None}],
           "reopened_from": "", "source": "owner", "wrote": True, "wrote_t": now}
    (tmp_path / "mem" / MP.PENDING_FILE).write_text(json.dumps([row]))
    assert MP.recover_pending_changes(store) == 1
    assert len(store.list_changes(pid, kind="change")) == 1 and len(store.list_changes(pid, kind="deploy")) == 1
    assert store.live_info(pid)["url"] == "https://l.example/"


def test_an_unreadable_folder_is_too_large_never_stuck_taking(store, monkeypatch):
    pid, ws = _built(store)
    monkeypatch.setattr(MP.os, "walk", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert MP.snapshot_folder(ws) is None
    MP.watch_project(store, pid, key="req-r3-8")
    assert MP._PENDING_CHANGES["req-r3-8"][f"{id(store)}:{pid}"]["snap"] == "too_large"
    MP._TOO_LARGE.clear()


def test_a_request_s_own_system_writes_survive_other_requests_flooding_the_registry(ctx, store):
    from ghost_agent.tools.turn_scratch import note_created
    pid, ws = _built(store)
    ctx.current_project_id = pid
    with _Owner(ctx, "req-r3-9", "fix the header and show me"):
        TP.prepare_project_write(ctx, project_id=pid)
        note_created(ws / "shot.png", "screenshot")
        (ws / "shot.png").write_bytes(b"png")
        _sh(ws, "printf '<h1>new</h1>' > index.html")
    for i in range(MP._SYSTEM_PATHS_MAX + 10):                          # other requests' sweeps
        MP.note_system_path(f"/elsewhere/{i}.png")
    MP.flush_change_log("req-r3-9", reply="ok")
    (e,) = store.list_changes(pid)
    assert e["added"] == [] and e["changed"] == ["index.html"] and "req-r3-9" not in MP._SYSTEM_BY_REQ


# ───────────────────────────── round 4 ─────────────────────────────

def test_read_only_commands_on_a_too_large_project_make_no_entry_and_a_write_does(ctx, store):
    from ghost_agent.tools.execute import _prepare_execute_project_writes
    pid, ws = _built(store)
    MP._TOO_LARGE[f"{id(store)}:{pid}"] = time.time()
    ctx.current_project_id = pid
    for i, cmd in enumerate(["python -m pytest -q", "ls -la", "git status"]):
        with _Owner(ctx, f"req-r4-ro-{i}", "run the tests"):
            _prepare_execute_project_writes({"container_workdir": f"/workspace/projects/{pid}", "command": cmd})
        MP.flush_change_log(f"req-r4-ro-{i}", reply="tests pass")
    assert store.list_changes(pid, kind="change") == []
    with _Owner(ctx, "req-r4-w", "fix the heading"):
        _prepare_execute_project_writes({"container_workdir": f"/workspace/projects/{pid}",
                                         "command": "sed -i 's/hi/yo/' index.html"})
    MP.flush_change_log("req-r4-w", reply="Fixed.")
    (e,) = store.list_changes(pid, kind="change")
    assert e["meta"]["folder_too_large"]
    MP._TOO_LARGE.clear()


def test_the_download_hook_and_the_download_name_the_same_file():
    import ghost_agent.tools.file_system as FS
    url = "https://example.com/x/%2Fa%00b%28c%29.png?x=1"
    assert FS._download_target(None, {"url": url}) == FS.download_target_name(None, url) == "a b(c).png".replace(" ", "")
    assert FS._download_target("https://h/y.pdf", {}) == "y.pdf"
    assert FS._download_target("docs/z.pdf", {"url": url}) == "docs/z.pdf"
    assert FS._download_target(None, {"url": "http://[bad/x.pdf"}) == ""           # never raises


def test_a_folder_get_notes_its_batch_once(ctx, store, monkeypatch):
    import ghost_agent.tools.file_system as FS
    pid, ws = _built(store)
    persists, reads = [], []
    real_persist, real_deliv = MP._persist, store.list_deliverables
    monkeypatch.setattr(MP, "_persist", lambda st: persists.append(1) or real_persist(st))
    monkeypatch.setattr(store, "list_deliverables", lambda p: reads.append(1) or real_deliv(p))
    paths = []
    for i in range(50):
        (ws / "site").mkdir(exist_ok=True)
        (ws / "site" / f"f{i}.txt").write_text("x")
        paths.append(ws / "site" / f"f{i}.txt")
    with _Owner(ctx, "req-r4-batch", "get the site folder from the lab"):
        persists.clear(), reads.clear()
        FS.note_project_writes_hook(Path(ctx.sandbox_dir), paths)
        assert len(persists) == 1 and len(reads) == 1
    MP.flush_change_log("req-r4-batch", reply="ok")
    assert len(store.list_changes(pid)[0]["added"]) == 50


async def test_an_unreadable_download_url_is_refused_not_raised(ctx, store):
    import ghost_agent.tools.file_system as FS
    pid, ws = _built(store)
    with _Owner(ctx, "req-r4-badurl", "get it"):
        res = await FS.tool_file_system(operation="download", sandbox_dir=ws, url="http://[bad/x.pdf")
    assert "cannot be read" in str(res) and getattr(getattr(res, "status", None), "value", "") == "rejected"


# ───────────────────────────── round 5 ─────────────────────────────

@pytest.mark.parametrize("cmd", ["rm old.js", "git checkout -- index.html", "black .", "patch -p1 < fix.diff"])
def test_mutating_commands_are_write_evidence_on_a_too_large_project(ctx, store, cmd):
    from ghost_agent.tools.execute import _prepare_execute_project_writes
    pid, ws = _built(store)
    MP._TOO_LARGE[f"{id(store)}:{pid}"] = time.time()
    ctx.current_project_id = pid
    with _Owner(ctx, "req-r5-1", "tidy it"):
        _prepare_execute_project_writes({"container_workdir": f"/workspace/projects/{pid}", "command": cmd})
    MP.flush_change_log("req-r5-1", reply="ok")
    (e,) = store.list_changes(pid, kind="change")
    assert e["meta"]["folder_too_large"]
    MP._TOO_LARGE.clear()


def test_a_build_that_grows_the_folder_past_the_cap_is_still_an_entry(ctx, store, monkeypatch):
    from ghost_agent.tools.execute import _prepare_execute_project_writes
    pid, ws = _built(store)
    ctx.current_project_id = pid
    _walk_fails_at_the_flush(monkeypatch)
    with _Owner(ctx, "req-r5-2", "rebuild"):
        _prepare_execute_project_writes({"container_workdir": f"/workspace/projects/{pid}", "command": "npm run build"})
        (ws / "index.html").write_text("<h1>rebuilt</h1>")
    MP.flush_change_log("req-r5-2", reply="Rebuilt.")
    (e,) = store.list_changes(pid, kind="change")
    assert e["meta"]["folder_too_large"]
    MP._TOO_LARGE.clear()


def test_a_generator_on_an_already_too_large_project_is_said_once_not_logged(ctx, store, caplog):
    import logging
    from ghost_agent.tools.execute import _prepare_execute_project_writes
    pid, ws = _built(store)
    MP._TOO_LARGE[f"{id(store)}:{pid}"] = time.time()
    ctx.current_project_id = pid
    caplog.set_level(logging.INFO, logger="GhostAgent")
    with _Owner(ctx, "req-r5-3", "rebuild"):
        _prepare_execute_project_writes({"container_workdir": f"/workspace/projects/{pid}", "command": "make"})
        _prepare_execute_project_writes({"container_workdir": f"/workspace/projects/{pid}", "command": "make test"})
    MP.flush_change_log("req-r5-3", reply="ok")
    assert store.list_changes(pid, kind="change") == []
    said = [r for r in caplog.records if "too large to compare" in r.getMessage()]
    assert len(said) == 1 and said[0].levelno == logging.INFO
    MP._TOO_LARGE.clear()
