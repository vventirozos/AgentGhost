"""§4UM living projects: only the owner closes a project; a finished plan leaves it OPEN ("ready"); every owner
change is one change-log entry (asked, files added/changed/removed, summary, live or not) mirrored to the
project's PROJECT_CHANGES.md; a push into the project's kept lab (deploy / switch / put) is a deploy entry and marks
the changes it carried; the briefing shows recent changes, the folder as it is on disk, and where it is live;
a backfill script adds a change made before the log existed."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core.prompts import build_project_briefing
from ghost_agent.memory import projects as MP
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools import labs as LB
from ghost_agent.tools.projects import note_project_write, tool_manage_projects
from ghost_agent.utils.logging import request_id_context

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_pending():
    MP._PENDING_CHANGES.clear()
    LB._KEPT_LAB_SEEN.clear()
    yield
    MP._PENDING_CHANGES.clear()
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


def _built(store, title="Site", rid="req-build"):
    """A project whose plan was built (and finished) by request ``rid``."""
    pid = store.create_project(title)
    ws = Path(store.ensure_workspace(pid))
    (ws / "index.html").write_text("<h1>hi</h1>")
    t = store.add_task(pid, "Build index.html")
    tok = request_id_context.set(rid)
    try:
        store.update_task(t, status="DONE", result_summary="built")
        store.register_file_artifact(t, "index.html")
    finally:
        request_id_context.reset(tok)
    MP.flush_deferred_sweeps(rid)
    return pid, ws


def _owner_write(ctx, ws, rel, text, ask, rid):
    ctx.last_user_content = ask
    tok = request_id_context.set(rid)
    try:
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return note_project_write(ctx, p)
    finally:
        request_id_context.reset(tok)


# ───────────────────────────── a finished plan stays open ─────────────────────────────

def test_a_finished_plan_leaves_the_project_open_and_ready_and_fires_once(store):
    fired = []
    store.on_project_done = fired.append
    pid = store.create_project("P", metadata={"constraints": ["one page"]})
    a, b = store.add_task(pid, "a"), store.add_task(pid, "b")
    store.update_task(a, status="DONE", result_summary="x")
    assert store.project_state(pid) == "working" and fired == []
    store.update_task(b, status="DONE", result_summary="x")
    p = store.get_project(pid)
    assert p["status"] == "ACTIVE" and store.project_state(pid) == "ready" and store.planned_work_done(pid)
    assert fired == [pid] and p["metadata"]["constraints"] == []
    ev = store.list_events(pid, event_type="project_auto_rollup")
    assert ev and ev[0]["payload"]["new_status"] == "READY"
    store.update_task(a, status="DONE", result_summary="again")         # no second finish
    assert fired == [pid]
    c = store.add_task(pid, "c")                                         # open work again
    assert store.project_state(pid) == "working" and store.get_project(pid)["metadata"]["constraints"] == ["one page"]
    store.update_task(c, status="DONE", result_summary="x")
    assert fired == [pid, pid]                                           # a new finish fires again


async def test_only_the_owner_closes_a_project(ctx, store):
    pid, _ = _built(store)
    ctx.current_project_id = pid
    tok = request_id_context.set("sched-autopilot-1")
    try:
        out = await tool_manage_projects(ctx, action="update", project_id=pid, status="DONE")
    finally:
        request_id_context.reset(tok)
    assert store.get_project(pid)["status"] == "ACTIVE", str(out)
    tok = request_id_context.set("req-owner-close")
    try:
        await tool_manage_projects(ctx, action="update", project_id=pid, status="DONE")
    finally:
        request_id_context.reset(tok)
    assert store.get_project(pid)["status"] == "DONE" and store.project_state(pid) == "done"


async def test_release_works_on_a_ready_project(ctx, store):
    pid, _ = _built(store)
    ctx.current_project_id = pid
    tok = request_id_context.set("req-owner-rel")
    try:
        out = await tool_manage_projects(ctx, action="release", project_id=pid,
                                         directions="Open index.html in a browser. It shows the page.")
    finally:
        request_id_context.reset(tok)
    assert store.get_project(pid)["status"] == "RELEASED", str(out)


# ───────────────────────────── the change log ─────────────────────────────

def test_an_owner_change_is_one_entry_with_its_files_and_the_reply_summary(ctx, store):
    pid, ws = _built(store)
    _owner_write(ctx, ws, "js/apps/game.js", "game()", "add a game to it", "req-c1")
    _owner_write(ctx, ws, "index.html", "<h1>hi</h1><script src=js/apps/game.js>", "add a game to it", "req-c1")
    assert store.list_changes(pid) == []                                 # written at the request's end
    n = MP.flush_change_log("req-c1", reply="**The game is in.** Open the menu to play.\n\nMore text.")
    assert n == 1
    (e,) = store.list_changes(pid)
    assert e["num"] == 1 and e["ask"] == "add a game to it" and e["req_id"] == "req-c1"
    assert e["added"] == ["js/apps/game.js"] and e["changed"] == ["index.html"] and not e["deployed"]
    assert e["summary"] == "The game is in. Open the menu to play. More text."   # the whole reply, plain
    log = (ws / "PROJECT_CHANGES.md").read_text()
    assert "#1" in log and "js/apps/game.js" in log and "add a game to it" in log and "not deployed" in log
    assert store.project_state(pid) == "ready" and not [t for t in store.list_tasks(pid)
                                                        if t["description"].startswith("Change:")]


def test_a_second_change_is_numbered_and_a_removed_file_is_listed(ctx, store):
    pid, ws = _built(store)
    _owner_write(ctx, ws, "a.js", "1", "add a", "req-1")
    MP.flush_change_log("req-1", reply="Added a.")
    ctx.last_user_content = "drop a"
    (ws / "a.js").unlink()
    tok = request_id_context.set("req-2")
    try:
        note_project_write(ctx, ws / "a.js")
    finally:
        request_id_context.reset(tok)
    MP.flush_change_log("req-2", reply="Removed a.")
    newest, older = store.list_changes(pid)
    assert (newest["num"], newest["removed"], older["num"], older["added"]) == (2, ["a.js"], 1, ["a.js"])


def test_a_write_into_a_closed_project_reopens_it_and_says_so(ctx, store):
    pid, ws = _built(store)
    store.update_project(pid, status="DONE")
    note = _owner_write(ctx, ws, "style.css", "h1{}", "make it blue", "req-r1")
    assert store.get_project(pid)["status"] == "ACTIVE" and "reopened" in note
    MP.flush_change_log("req-r1", reply="Blue now.")
    (e,) = store.list_changes(pid)
    assert e["meta"].get("reopened_from") == "DONE"
    ev = store.list_events(pid, event_type="project_reopened")
    assert ev and "change #1" in ev[0]["payload"]["reason"]


def test_the_request_that_built_the_plan_is_not_a_change(ctx, store):
    pid = store.create_project("P")
    ws = Path(store.ensure_workspace(pid))
    t = store.add_task(pid, "build")
    _owner_write(ctx, ws, "main.py", "x", "build it", "req-plan")
    tok = request_id_context.set("req-plan")
    try:
        store.update_task(t, status="DONE", result_summary="built")
    finally:
        request_id_context.reset(tok)
    MP.flush_change_log("req-plan", reply="Built.")
    assert store.list_changes(pid) == []


def test_a_task_run_change_after_the_plan_was_finished_is_logged(ctx, store):
    pid, ws = _built(store)
    t = store.add_task(pid, "add the score board")
    _owner_write(ctx, ws, "score.js", "s", "add a score board", "req-later")
    tok = request_id_context.set("req-later")
    try:
        store.update_task(t, status="DONE", result_summary="score board")
    finally:
        request_id_context.reset(tok)
    MP.flush_change_log("req-later", reply="Score board added.")
    (e,) = store.list_changes(pid)
    assert e["added"] == ["score.js"]


def test_a_background_turn_logs_nothing(ctx, store):
    pid, ws = _built(store)
    _owner_write(ctx, ws, "x.js", "1", "", "sched-tick-1")
    MP.flush_change_log("sched-tick-1", reply="")
    assert store.list_changes(pid) == []


def test_an_unflushed_request_is_written_by_a_later_flush(ctx, store, monkeypatch):
    pid, ws = _built(store)
    _owner_write(ctx, ws, "late.js", "1", "late", "req-lost")
    monkeypatch.setattr(MP, "DEFERRED_SWEEP_MAX_AGE_S", -1.0)
    MP.flush_change_log("req-other")
    assert [e["added"] for e in store.list_changes(pid)] == [["late.js"]]


def test_the_sweep_keeps_a_changed_file(ctx, store):
    from ghost_agent.core.workspace_cleanup import sweep_project_workspace
    pid, ws = _built(store)
    _owner_write(ctx, ws, "js/tetris.js", "t", "add tetris", "req-keep")
    MP.flush_change_log("req-keep", reply="Done.")
    sweep_project_workspace(store, pid)
    assert (ws / "js/tetris.js").exists() and (ws / "PROJECT_CHANGES.md").exists()


# ───────────────────────────── going live ─────────────────────────────

def test_push_covers_files_and_folders():
    assert MP.push_covers(["a/b.js"], "a/b.js") and MP.push_covers(["a/"], "a/b.js")
    assert MP.push_covers([""], "x.js") and not MP.push_covers(["a/"], "ab.js")
    assert not MP.push_covers([], "x.js") and not MP.push_covers(["a/b.js"], "a/c.js")


def test_a_put_marks_a_change_live_only_when_all_its_files_went(ctx, store):
    pid, ws = _built(store)
    _owner_write(ctx, ws, "js/game.js", "g", "add a game", "req-p1")
    _owner_write(ctx, ws, "index.html", "i", "add a game", "req-p1")
    MP.note_change_push(store, pid, ["js/game.js"], url="https://lab.example/", lab_id="lab_00000001",
                        how="put", rid="req-p1")
    MP.flush_change_log("req-p1", reply="Game added.")
    ch = store.list_changes(pid, kind="change")[0]
    assert not ch["deployed"]
    MP.note_change_push(store, pid, ["index.html"], url="https://lab.example/", lab_id="lab_00000001",
                        how="put", rid="")                                # outside a request: at once
    ch = store.list_changes(pid, kind="change")[0]
    assert ch["deployed"] and ch["url"] == "https://lab.example/"
    live = store.live_info(pid)
    assert live["url"] == "https://lab.example/" and live["behind"] == [] and live["through"] == 1


def test_a_push_after_the_write_in_the_same_request_marks_the_change_live(ctx, store):
    pid, ws = _built(store)
    _owner_write(ctx, ws, "a.js", "1", "a", "req-same")
    MP.note_change_push(store, pid, None, url="https://lab.example/", lab_id="lab_00000001", how="deploy",
                        rid="req-same")
    MP.flush_change_log("req-same", reply="Live.")
    change = store.list_changes(pid, kind="change")[0]
    deploy = store.list_changes(pid, kind="deploy")[0]
    assert change["deployed"] and deploy["meta"]["whole"] and deploy["meta"]["through"] == 1
    assert "Live" in (ws / "PROJECT_CHANGES.md").read_text()


def test_a_later_change_makes_the_live_site_behind(ctx, store):
    pid, ws = _built(store)
    _owner_write(ctx, ws, "a.js", "1", "a", "req-b1")
    MP.note_change_push(store, pid, None, url="https://lab.example/", how="deploy", rid="req-b1")
    MP.flush_change_log("req-b1", reply="ok")
    _owner_write(ctx, ws, "b.js", "1", "b", "req-b2")
    MP.flush_change_log("req-b2", reply="ok")
    assert store.live_info(pid)["behind"] == [2]
    assert "BEHIND" in MP.live_line(store, pid) and "#2" in MP.live_line(store, pid)


def _lab_det(pid, lifetime="kept", url="https://lab.example:20000/"):
    return {"lab_id": "lab_0000000a", "owner": f"project:{pid}", "lifetime": lifetime,
            "app": {"name": "site", "state": "up", "url": url}}


async def test_a_lab_put_into_the_project_s_kept_lab_is_recorded(ctx, store, monkeypatch):
    pid, ws = _built(store)
    calls = []

    async def fake_call(context, method, path, **kw):
        calls.append(path)
        return 200, _lab_det(pid)
    monkeypatch.setattr(LB, "_call", fake_call)
    _owner_write(ctx, ws, "js/apps/tetris.js", "t", "add tetris", "req-lab")
    tok = request_id_context.set("req-lab")
    try:
        await LB._note_project_push(ctx, "owner", "lab_0000000a", ws / "js/apps/tetris.js", how="put",
                                    member="web", path="/app/js/apps/tetris.js")
        await LB._note_project_push(ctx, "owner", "lab_0000000a", ws / "js", how="put",     # a folder put
                                    member="web", path="/app/js")
    finally:
        request_id_context.reset(tok)
    assert calls == ["/v1/labs/lab_0000000a"]                           # one look per kept lab
    MP.flush_change_log("req-lab", reply="Tetris is in and live.")
    change = store.list_changes(pid, kind="change")[0]
    deploy = store.list_changes(pid, kind="deploy")[0]
    assert change["deployed"] and deploy["url"] == "https://lab.example:20000/"
    assert set(deploy["changed"]) == {"js/apps/tetris.js", "js/"} and deploy["meta"]["how"] == "put"


async def test_no_record_for_a_disposable_lab_another_owner_a_probe_or_a_simulation(ctx, store, monkeypatch):
    pid, ws = _built(store)
    det = {"v": _lab_det(pid, lifetime="disposable")}

    async def fake_call(context, method, path, **kw):
        return 200, det["v"]
    monkeypatch.setattr(LB, "_call", fake_call)
    await LB._note_project_push(ctx, "owner", "lab_0000000b", ws / "index.html", how="put",
                                path="/app/index.html")
    det["v"] = {"lifetime": "kept", "owner": "conversation:x", "app": {}}
    await LB._note_project_push(ctx, "owner", "lab_0000000c", ws / "index.html", how="put",
                                path="/app/index.html")
    det["v"] = _lab_det(pid)
    await LB._note_project_push(ctx, "probe", "lab_0000000d", ws / "index.html", how="put",
                                path="/app/index.html")
    ctx.simulated = True
    await LB._note_project_push(ctx, "owner", "lab_0000000e", ws / "index.html", how="put",
                                path="/app/index.html")
    assert store.list_changes(pid) == [] and store.live_info(pid) is None


async def test_a_switch_marks_the_changes_up_to_the_staged_deploy_and_a_deploy_marks_all(ctx, store,
                                                                                           monkeypatch):
    pid, ws = _built(store)

    async def fake_call(context, method, path, **kw):
        return 200, _lab_det(pid)
    monkeypatch.setattr(LB, "_call", fake_call)
    _owner_write(ctx, ws, "a.js", "1", "a", "req-w1")
    MP.flush_change_log("req-w1", reply="ok")
    await LB._note_project_push(ctx, "owner", "lab_0000000f", None, how="switch", whole=True)
    assert not store.list_changes(pid, kind="change")[0]["deployed"]     # no staged deploy known: nothing marked
    await LB._note_project_push(ctx, "owner", "lab_0000000f", ws, how="deploy", whole=True, staged=True)
    assert not store.list_changes(pid, kind="change")[0]["deployed"]     # staged: not live yet
    import time as _t
    _t.sleep(0.02)
    _owner_write(ctx, ws, "late.js", "1", "late", "req-w1b")             # made after the staged deploy
    MP.flush_change_log("req-w1b", reply="ok")
    await LB._note_project_push(ctx, "owner", "lab_0000000f", None, how="switch", whole=True)
    by = {e["num"]: e["deployed"] for e in store.list_changes(pid, kind="change")}
    assert by == {1: True, 2: False}
    _owner_write(ctx, ws, "b.js", "1", "b", "req-w2")
    MP.flush_change_log("req-w2", reply="ok")
    await LB._note_project_push(ctx, "owner", "lab_0000000f", ws, how="deploy", whole=True)
    assert all(e["deployed"] for e in store.list_changes(pid, kind="change"))


# ───────────────────────────── what the model reads ─────────────────────────────

def test_the_briefing_shows_ready_changes_files_and_live(ctx, store):
    pid, ws = _built(store)
    _owner_write(ctx, ws, "js/apps/tetris.js", "t", "add tetris", "req-v1")
    MP.note_change_push(store, pid, None, url="https://lab.example/", how="put", rid="req-v1")
    MP.flush_change_log("req-v1", reply="Tetris is in.")
    (ws / "node_modules").mkdir()
    (ws / "node_modules" / "dep.js").write_text("x")
    b = build_project_briefing(store, pid)
    assert "state READY" in b and "only the user closes it" in b
    assert "RECENT CHANGES" in b and "add tetris" in b and "js/apps/tetris.js" in b
    assert "LIVE AT: Live at https://lab.example/" in b
    files = next(ln for ln in b.splitlines() if ln.startswith("FILES (on disk now)"))
    assert "index.html" in files and "js/apps/tetris.js" in files and "node_modules" not in files


def test_the_briefing_of_a_closed_project_says_a_change_reopens_it(store):
    pid, _ = _built(store)
    store.update_project(pid, status="DONE")
    b = build_project_briefing(store, pid)
    assert "CLOSED by the user" in b and "reopens it" in b and "state READY" not in b


async def test_status_list_and_changes_show_the_state_and_the_log(ctx, store):
    pid, ws = _built(store)
    other = store.create_project("Busy")
    store.add_task(other, "open work")
    for i in range(3):
        _owner_write(ctx, ws, f"f{i}.js", "x", f"change {i}", f"req-s{i}")
        MP.flush_change_log(f"req-s{i}", reply=f"Change {i} done.")
    ctx.current_project_id = pid
    st = json.loads(str(await tool_manage_projects(ctx, action="status")))
    br = st["briefing"]
    assert br["project"]["state"] == "ready" and len(br["recent_changes"]) == 3 and "f2.js" in br["files"]
    assert "only the user closes it" in br["state_note"].lower() or "Only the user closes it" in br["state_note"]
    ls = json.loads(str(await tool_manage_projects(ctx, action="list")))
    rows = ls["display"].splitlines()
    assert rows[0].startswith(other) and "working" in rows[0] and rows[1].startswith(pid) and "ready" in rows[1]
    ch = json.loads(str(await tool_manage_projects(ctx, action="changes", count="2")))
    assert ch["total"] == 3 and len(ch["changes"]) == 2 and ch["changes"][0].startswith("#3") and ch["more"]
    ch2 = json.loads(str(await tool_manage_projects(ctx, action="changes", count="2", metadata={"offset": 2})))
    assert ch2["changes"][0].startswith("#1") and ch2["more"] is None


# ───────────────────────────── the backfill script ─────────────────────────────

def _script(tmp_path, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "backfill_project_change.py"),
                           "--memory-dir", str(tmp_path / "mem"), "--sandbox-root", str(tmp_path / "sb"), *args],
                          capture_output=True, text=True, timeout=60)


def test_the_backfill_adds_a_past_change_and_its_deploy(tmp_path, store):
    pid, ws = _built(store)
    store.update_project(pid, status="DONE")                             # the old roll-up closed it
    args = ["--project", pid, "--ask", "add a game", "--added", "js/apps/game.js", "--changed", "index.html",
            "--summary", "The game is in and live.", "--ts", "2026-11-12 19:57", "--req", "abcd1234",
            "--deployed", "--url", "https://lab.example:20000/", "--lab-id", "lab_00000009", "--reopen"]
    dry = _script(tmp_path, *args)
    assert dry.returncode == 0 and "dry run" in dry.stdout and store.list_changes(pid) == []
    run = _script(tmp_path, *args, "--apply")
    assert run.returncode == 0, run.stderr
    (ch,) = store.list_changes(pid, kind="change")
    (dep,) = store.list_changes(pid, kind="deploy")
    assert ch["num"] == 1 and ch["deployed"] and ch["added"] == ["js/apps/game.js"] and ch["req_id"] == "abcd1234"
    assert dep["url"] == "https://lab.example:20000/" and store.live_info(pid)["through"] == 1
    assert store.get_project(pid)["status"] == "ACTIVE" and store.project_state(pid) == "ready"
    assert "add a game" in (ws / "PROJECT_CHANGES.md").read_text()
    again = _script(tmp_path, *args, "--apply")
    assert again.returncode == 0 and "already" in again.stdout                # idempotent
    assert len(store.list_changes(pid, kind="change")) == 1


def test_a_task_turn_inside_the_owner_s_request_logs_to_the_owner_s_request(ctx, store, monkeypatch):
    import ghost_agent.tools.projects as TP
    pid, ws = _built(store)
    MP.set_request_parent("sub-ptask-x-1", "req-root-1")
    monkeypatch.setattr(TP, "_owner_task_turn", lambda context=None: True)
    _owner_write(ctx, ws, "extra.js", "e", "add extra", "sub-ptask-x-1")
    assert MP.flush_change_log("sub-ptask-x-1", reply="task turn reply") == 0   # the turn's own end: nothing
    assert MP.flush_change_log("req-root-1", reply="Extra added.") == 1
    (e,) = store.list_changes(pid)
    assert e["req_id"] == "req-root-1" and e["added"] == ["extra.js"] and e["summary"] == "Extra added."


def test_a_legacy_auto_closed_project_briefs_as_open_not_done(store):
    """Live probe 2026-11-13: the webOS (closed by the old roll-up) was described as "DONE and 'ready'"
    — the briefing must show it open (ACTIVE / ready); an owner close still shows DONE."""
    from ghost_agent.tools.projects import _briefing
    pid = store.create_project("Legacy", kind="CODING", goal="g")
    t = store.add_task(pid, "Build it → a.py")
    store.update_task(t, status="DONE")
    store.update_project(pid, status="DONE")
    store.log_event(pid, None, "project_auto_rollup", {"new_status": "DONE"})
    b = _briefing(store, pid)["project"]
    assert (b["status"], b["state"]) == ("ACTIVE", "ready")
    store.log_event(pid, None, "project_closed", {"by": "owner"})
    b = _briefing(store, pid)["project"]
    assert (b["status"], b["state"]) == ("DONE", "done")


async def test_get_and_the_project_prompt_show_a_legacy_auto_close_as_open(store, ctx):
    """Live probe 2026-11-13 (r2): `action=get` and the focused-project prompt still said DONE."""
    pid = store.create_project("Legacy2", kind="CODING", goal="g")
    t = store.add_task(pid, "Build it → a.py")
    store.update_task(t, status="DONE")
    store.update_project(pid, status="DONE")
    store.log_event(pid, None, "project_auto_rollup", {"new_status": "DONE"})
    out = json.loads(str(await tool_manage_projects(action="get", project_id=pid, context=ctx)))
    body = out.get("result", out) if isinstance(out, dict) else out
    assert (body.get("status"), body.get("state")) == ("ACTIVE", "ready"), body
    assert "· DONE" not in build_project_briefing(store, pid)
