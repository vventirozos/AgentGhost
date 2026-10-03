"""§4KW — five more independent reviews (API/origin, lessons, scheduler,
sandbox tools, memory). Each test names the world it fails in."""
import ast
import asyncio
import inspect
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.tools import file_system as FS


RELEASED = "aaaaaaaaaaaa"
DEV = "bbbbbbbbbbbb"


class _Store:
    def __init__(self, released=(RELEASED,)):
        self.released = set(released)

    def get_project(self, pid):
        return {"id": pid, "status": "RELEASED" if pid in self.released else "DEVELOPMENT"}

    def list_projects(self, status_filter=None):
        ids = sorted(self.released) if (status_filter or "").upper() == "RELEASED" else []
        return [{"id": i, "status": "RELEASED"} for i in ids]


def _sandbox(tmp_path):
    for pid in (RELEASED, DEV):
        d = tmp_path / "projects" / pid
        d.mkdir(parents=True)
        (d / "app.py").write_text("print(1)\n")
    (tmp_path / "notes.md").write_text("root notes\n")
    return tmp_path


def _fs(tmp_path, store=None, **kw):
    return asyncio.run(FS.tool_file_system(sandbox_dir=tmp_path, project_store=store, **kw))


# ── sandbox: the released lock covers a PARENT folder ───────────────────────
@pytest.mark.parametrize("op,extra", [("delete", {}), ("rename", {"destination": "old_projects"}),
                                      ("move", {"destination": "elsewhere/projects"})])
def test_a_parent_of_a_released_workspace_is_locked(tmp_path, op, extra):
    """Fails in the world where `delete path=projects` wiped every project,
    released ones included (no project id in the path)."""
    sb = _sandbox(tmp_path)
    out = str(_fs(sb, _Store(), operation=op, path="projects", **extra))
    assert "SYSTEM BLOCK" in out and (sb / "projects" / RELEASED / "app.py").exists()


def test_the_projects_folder_is_locked_without_a_released_workspace_too(tmp_path):
    """Fifth review: with no released project, `delete path=projects` removed
    every project."""
    sb = _sandbox(tmp_path)
    out = str(_fs(sb, _Store(released=()), operation="delete", path="projects"))
    assert "SYSTEM BLOCK" in out and (sb / "projects" / DEV / "app.py").exists()


# ── sandbox: a symlink is deleted / moved as itself ─────────────────────────
def test_deleting_a_link_removes_the_link_not_its_target(tmp_path):
    """Fails in the world where delete followed the link and rmtree'd the
    other project's folder, leaving the link dangling."""
    sb = _sandbox(tmp_path)
    os.symlink(sb / "projects" / DEV, sb / "in_link")
    out = str(_fs(sb, _Store(released=()), operation="delete", path="in_link"))
    assert out.startswith("SUCCESS: Deleted 'in_link'.") and "link" in out
    assert not (sb / "in_link").is_symlink() and (sb / "projects" / DEV / "app.py").exists()


def test_renaming_a_link_moves_the_link_not_its_target(tmp_path):
    sb = _sandbox(tmp_path)
    os.symlink(sb / "projects" / DEV, sb / "in_link")
    out = str(_fs(sb, _Store(released=()), operation="rename", path="in_link", destination="moved_link"))
    assert out.startswith("SUCCESS: Renamed/Moved 'in_link' to 'moved_link'.")
    assert (sb / "moved_link").is_symlink() and (sb / "projects" / DEV / "app.py").exists()


def test_copy_still_copies_after_the_dotgit_loop(tmp_path):
    """The `.git` guard now loops over every target; its loop variable must
    not replace the operation's own source path."""
    (tmp_path / "src.txt").write_text("x")
    out = str(_fs(tmp_path, None, operation="copy", path="src.txt", destination="dest.txt"))
    assert "SUCCESS" in out and (tmp_path / "dest.txt").read_text() == "x"


@pytest.mark.parametrize("op", ["copy", "move"])
def test_nothing_is_copied_or_moved_into_dotgit(tmp_path, op):
    (tmp_path / "hook").write_text("#!/bin/sh\n")
    (tmp_path / "repo" / ".git" / "hooks").mkdir(parents=True)
    out = str(_fs(tmp_path, None, operation=op, path="hook", destination="repo/.git/hooks/pre-commit"))
    assert ".git" in out and "SYSTEM BLOCK" in out
    assert not (tmp_path / "repo" / ".git" / "hooks" / "pre-commit").exists()


def test_forget_respects_the_released_lock(tmp_path):
    from ghost_agent.tools.memory import tool_unified_forget
    sb = _sandbox(tmp_path)
    ms = MagicMock()
    ms.get_library.return_value = []
    ms.collection.query.return_value = {"ids": None}
    out = str(asyncio.run(tool_unified_forget(f"projects/{RELEASED}/app.py", sb, ms, project_store=_Store())))
    assert "RELEASED" in out and (sb / "projects" / RELEASED / "app.py").exists()


# ── execute: no destructive re-run, no bulk removal over released work ──────
@pytest.mark.parametrize("cmd,unsafe", [
    ("rm notes.md", True), ("rm a.txt b.txt", True), ("mv a b", True), ("cp a b", True),
    ("find . -name x -delete", True), ("cat x.txt", False), ("python3 run.py", False), ("ls -la", False),
])
def test_a_destructive_final_segment_is_never_rerun_elsewhere(cmd, unsafe):
    """Fails in the world where a failed `rm notes.md` (missing in the
    project) was re-run from the root and deleted the root's file."""
    from ghost_agent.tools.execute import _rerun_unsafe
    assert _rerun_unsafe(cmd) is unsafe


@pytest.mark.parametrize("cmd", ["rm -rf projects", "rm -rf *", "rm -rf ./*", "rm -r projects/",
                                 "find . -mindepth 1 -delete", 'rm -rf "$PWD"/*', "mv projects old"])
def test_bulk_removal_is_refused_while_a_released_project_exists(cmd):
    from ghost_agent.tools.execute import _released_shell_block
    assert "RELEASED" in str(_released_shell_block(_Store(), cmd))
    # fifth review: refused with no released project too — the projects
    # folder holds every project's workspace
    assert "SYSTEM BLOCK" in str(_released_shell_block(_Store(released=()), cmd))


@pytest.mark.parametrize("cmd", ["rm -rf build", "rm -rf ./build", "rm -f *.pyc", "ls projects", "cat projects/x"])
def test_targeted_commands_are_not_bulk(cmd):
    from ghost_agent.tools.execute import _released_shell_block
    assert _released_shell_block(_Store(), cmd) is None


def test_long_option_rm_of_an_absolute_path_is_denied():
    from ghost_agent.tools.validators import validate_shell
    assert validate_shell("rm --recursive --force /workspace/projects")[0] is False
    assert validate_shell("rm -rf ./build")[0] is True


# ── origin: job wakes, scheduled tasks and probes never teach ───────────────
@pytest.mark.parametrize("rid,origin", [("job-job-08952c13", "internal"), ("sched-task_1", "internal"),
                                        ("sub-leaf-1", "internal"), ("probe-1234", "probe"), ("abc12345", "user")])
def test_turn_origin_by_request_id(rid, origin):
    from ghost_agent.utils.logging import request_id_context
    ctx = MagicMock()
    ctx.turn_origin_label = None
    ctx.skill_memory.is_read_only = False
    tok = request_id_context.set(rid)
    try:
        assert A.turn_origin(ctx) == origin
        assert A.turn_may_teach(ctx) is (origin == "user")
    finally:
        request_id_context.reset(tok)


def test_a_probe_turn_writes_no_episode():
    """Fails in the world where `_record_episode_safe` skipped only members:
    79+ probe episodes were recalled ~2,000 times."""
    from ghost_agent.utils.logging import request_id_context
    agent = A.GhostAgent.__new__(A.GhostAgent)
    agent.context = MagicMock()
    agent.context.turn_origin_label = None
    agent.context.skill_memory.is_read_only = False
    tok = request_id_context.set("probe-deadbeef")
    try:
        asyncio.run(agent._record_episode_safe("q", [], "a", req_id="probe-deadbeef"))
    finally:
        request_id_context.reset(tok)
    agent.context.episodic_memory.record_episode.assert_not_called()






# ── scheduler ────────────────────────────────────────────────────────────────
async def test_a_restored_interval_task_keeps_its_cadence(tmp_path, monkeypatch):
    """Fails in the world where each restart put the next run a FULL interval
    after boot (a daily task never fired on a box restarting more often)."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from ghost_agent.tools import tasks as T
    monkeypatch.setattr(T, "task_store_path", tmp_path / "scheduled_tasks.json")

    async def runner(*a):
        return True
    monkeypatch.setattr(T, "run_proactive_task_fn", runner)
    created = time.time() - 3500                    # created 3,500 s ago, every hour
    T._save_task_store({"t1": {"task_name": "hourly", "prompt": "p", "cron_expression": "interval:3600",
                               "kind": "task", "created_at": created}})
    s = AsyncIOScheduler(timezone="UTC")
    s.start()
    try:
        assert T.restore_persisted_tasks(s) == 1
        nxt = s.get_jobs()[0].next_run_time.timestamp() - time.time()
        assert 0 < nxt < 200                         # due in ~100 s, not in an hour
    finally:
        s.shutdown(wait=False)




# ── memory: forget, update_profile, a degraded store ────────────────────────
def _sweep_ms(rows):
    ms = MagicMock()
    ms.get_library.return_value = []
    ms._get_lock.return_value.__enter__ = MagicMock(return_value=None)
    ms._get_lock.return_value.__exit__ = MagicMock(return_value=False)
    ms.collection.query.return_value = {
        "ids": [[r[0] for r in rows]], "distances": [[r[1] for r in rows]],
        "documents": [[r[2] for r in rows]], "metadatas": [[{"type": "auto"} for _ in rows]]}
    return ms


def _deleted_ids(ms):
    return [c.kwargs.get("ids", [None])[0] for c in ms.collection.delete.call_args_list if c.kwargs.get("ids")]


@pytest.mark.parametrize("target", ["postgresql-19-A4.pdf", "yt-8S0FDjFBj8o.captions.en", "notes/report.md"])
def test_forgetting_a_document_keeps_unrelated_owner_facts(tmp_path, target):
    """Fails in the world where `forget <document>` deleted the owner's birth
    date, sons' birthdates and home town (distances 0.53–0.65, live 09-09)."""
    from ghost_agent.tools.memory import tool_unified_forget
    ms = _sweep_ms([("f1", 0.55, "The user's birth date is 1980-01-29"),
                    ("f2", 0.62, "The user lives in Thrakomakedones")])
    asyncio.run(tool_unified_forget(target, tmp_path, ms))
    assert "f1" not in _deleted_ids(ms) and "f2" not in _deleted_ids(ms)


def test_forgetting_an_entity_needs_a_near_match_or_a_literal_mention(tmp_path):
    from ghost_agent.tools.memory import tool_unified_forget
    ms = _sweep_ms([("near", 0.1, "user keeps an iguana"), ("far", 0.55, "The user's birth date is 1980-01-29"),
                    ("lit", 0.9, "the pinball machine is in the garage")])
    asyncio.run(tool_unified_forget("pinball", tmp_path, ms))
    assert set(_deleted_ids(ms)) == {"near", "lit"}


@pytest.mark.parametrize("kw", [{}, {"value": None}, {"value": "   "}])
def test_update_profile_never_deletes_without_an_explicit_empty_string(kw):
    """Fails in the world where any falsy value deleted (the owner's
    root.name was removed this way)."""
    from ghost_agent.tools.memory import tool_update_profile
    prof = MagicMock()
    prof.load.return_value = {"root": {"name": "Vasilis"}}
    out = asyncio.run(tool_update_profile(category="root", key="name", profile_memory=prof, **kw))
    assert "Nothing was changed" in out
    prof.delete.assert_not_called()
    prof.update.assert_not_called()


def test_an_explicit_delete_names_what_it_removed():
    from ghost_agent.tools.memory import tool_update_profile
    prof = MagicMock()
    prof.load.return_value = {"root": {"name": "Vasilis"}}
    prof.delete.return_value = "Removed from Profile: root.name"
    out = asyncio.run(tool_update_profile(category="root", key="name", value="", profile_memory=prof))
    assert "Removed from Profile: root.name" in out and "Vasilis" in out


def test_a_read_degraded_profile_refuses_writes_visibly(tmp_path, monkeypatch):
    """A transient read error (EIO) makes the store write-protected; the
    write must SAY so, not report success over nothing."""
    from ghost_agent.memory.profile import ProfileMemory
    from ghost_agent.tools.memory import tool_update_profile
    p = ProfileMemory(tmp_path)
    _real = Path.read_text

    def _eio(self, *a, **k):
        if self == p.file_path:
            raise OSError(5, "Input/output error")
        return _real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", _eio)
    assert p.update("root", "name", "X").startswith("Error")
    assert p.delete("root", "name").startswith("Error")
    out = asyncio.run(tool_update_profile(category="root", key="city", value="Athens", profile_memory=p))
    assert "SUCCESS" not in str(out) and "could not be read" in str(out)


async def test_the_bus_reports_a_refused_profile_write():
    from ghost_agent.core.bus import MemoryBus
    bus = MemoryBus(profile_memory=MagicMock())
    bus.profile.update.return_value = "Error: the profile store could not be read"
    rep = await bus.publish_fact("update_profile", {"profile_update": {"category": "root", "key": "x", "value": "y"}})
    assert str((rep or {}).get("profile", "")).startswith("error")


# ── lessons: credit, admission, merge ───────────────────────────────────────
def _pb(tmp_path, lessons):
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    sm.save_playbook(lessons)
    return sm


def test_an_unrelated_lesson_is_not_credited_for_being_in_the_prompt(tmp_path):
    """Fails in the world where every lesson in the prompt got helpful +1 on
    every clean turn (membership in `top_triggers` counted as relevance)."""
    from datetime import datetime
    now = datetime.now().isoformat()
    sm = _pb(tmp_path, [{"task": "When querying a database, quote identifiers", "mistake": "x",
                         "solution": "quote identifiers", "last_retrieved_at": now, "helpful_retrievals": 0}])
    assert sm.credit_recent_retrievals(300, query="what is the weather in Athens") == 0
    assert sm.credit_recent_retrievals(300, query="the database query fails on quoted identifiers") == 1




@pytest.mark.parametrize("fix,bad", [
    ("1. file_system(operation=list_files, path=/)\n2. file_system(operation=delete, path=/*)", True),
    ("rm -rf /workspace/*", True), ("rm -rf *", True), ("Use DROP TABLE users to reset", True),
    ("find . -mindepth 1 -delete", True), ("git reset --hard origin/main", True), ("Delete everything in the folder", True),
    ("Use a single operation to clear the entire directory if supported", True), ("purge all cached files", True),
    ("clear the cache entry for that key", False), ("empty the trash file you created", False),
    ("Run rm -rf build/ before rebuilding", False), ("Remove the old log file with rm app.log", False),
    ("Never run rm -rf /", False), ("Avoid DROP TABLE; migrate instead", False),
    ("use `find . -name '*.pyc' -delete`", False), ("verify each file before deleting it", False),
])
def test_a_lesson_may_not_prescribe_bulk_destruction(fix, bad):
    from ghost_agent.memory.lesson_quality import prescribes_destruction
    assert prescribes_destruction(fix) is bad


def test_the_write_point_refuses_a_destructive_fix(tmp_path):
    sm = _pb(tmp_path, [])
    out = sm.learn_lesson("lots of stuff in your sandbox, clean it up", "left files",
                          "1. file_system(operation=delete, path=/*)", source="reflection")
    assert out is None and json.loads(sm.file_path.read_text()) == []




def test_a_twin_from_another_producer_does_not_rewrite_the_lesson(tmp_path):
    """Fails in the world where a dream rule merged into a reflection lesson
    (frequency 1 → 17, fix replaced, verified inherited)."""
    sm = _pb(tmp_path, [{"timestamp": "2025-01-01T00:00:00", "task": "lots of stuff in your sandbox, clean it up",
                         "mistake": "left files", "solution": "list then remove the temp files", "frequency": 1,
                         "source": "reflection"}])
    mem = MagicMock()
    mem.collection.query.return_value = {
        "ids": [["id1"]], "distances": [[0.1]],
        "documents": [["SITUATION: lots of stuff in your sandbox, clean it up\nMISTAKE: left files\n"
                       "SOLUTION: list then remove the temp files"]]}
    sm.learn_lesson("When cleaning the sandbox, verify each file before deleting it", "deleted needed files",
                    "verify each file before deleting it and keep skills and projects", memory_system=mem,
                    source="dream", verified=True)
    pb = json.loads(sm.file_path.read_text())
    # written separately (a refused twin let a stale lesson block a correction
    # forever — second review); the stored lesson is untouched
    old = next(r for r in pb if r.get("source") == "reflection")
    assert len(pb) == 2 and old["solution"] == "list then remove the temp files"
    assert int(old.get("frequency", 1)) == 1 and not old.get("verified")


# ── second fresh review ─────────────────────────────────────────────────────
def test_a_copy_of_a_projects_folder_is_not_a_workspace(tmp_path):
    """backup/projects/<released> could never be cleaned up."""
    sb = _sandbox(tmp_path)
    (sb / "backup" / "projects" / RELEASED).mkdir(parents=True)
    out = str(_fs(sb, _Store(), operation="delete", path="backup"))
    assert out.startswith("SUCCESS") and not (sb / "backup").exists()


def test_copying_the_projects_folder_only_reads_it(tmp_path):
    sb = _sandbox(tmp_path)
    out = str(_fs(sb, _Store(), operation="copy", path="projects", destination="bk2"))
    assert "RELEASED" not in out


def test_deleting_a_link_into_a_released_project_is_allowed(tmp_path):
    sb = _sandbox(tmp_path)
    os.symlink(sb / "projects" / RELEASED, sb / "rel_link")
    out = str(_fs(sb, _Store(), operation="delete", path="rel_link"))
    assert out.startswith("SUCCESS") and (sb / "projects" / RELEASED / "app.py").exists()


@pytest.mark.parametrize("param,dest", [("content", "repo/.git/config"), ("replace_with", "repo/.git/hooks/x")])
def test_healed_destinations_are_checked_for_dotgit(tmp_path, param, dest):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "repo" / ".git" / "hooks").mkdir(parents=True)
    out = str(_fs(tmp_path, None, operation="copy", path="a.txt", **{param: dest}))
    assert "SYSTEM BLOCK" in out and not (tmp_path / dest).exists()


def test_a_healed_destination_into_a_released_project_is_refused(tmp_path):
    sb = _sandbox(tmp_path)
    (sb / "a.txt").write_text("x")
    out = str(_fs(sb, _Store(), operation="rename", path="a.txt", content=f"projects/{RELEASED}/x"))
    assert "RELEASED" in out and (sb / "a.txt").exists()


def test_forgetting_a_memory_is_not_refused_in_a_released_project(tmp_path):
    from ghost_agent.tools.memory import tool_unified_forget
    sb = _sandbox(tmp_path)
    ms = MagicMock()
    ms.get_library.return_value = []
    ms.collection.query.return_value = {"ids": None}
    out = str(asyncio.run(tool_unified_forget("my old address", sb / "projects" / RELEASED, ms, project_store=_Store())))
    assert "RELEASED" not in out


@pytest.mark.parametrize("cmd,unsafe", [
    ("python -m pytest -q tests/test_app.py 2>&1", False), ("ls x 2>/dev/null", False),
    ("python3 run.py >/dev/null 2>&1", False), ("find . -name x", False),
    ("find . -name '*.pyc' -delete", True), ("echo hi > out.txt", True), ("cat a >> b", True),
])
def test_rerun_safety_second_review(cmd, unsafe):
    from ghost_agent.tools.execute import _rerun_unsafe
    assert _rerun_unsafe(cmd) is unsafe


@pytest.mark.parametrize("cmd,wd,blocked", [
    ("mv *.png images/", "/workspace", False), ("mv report.md ./", "/workspace", False),
    ("git rm -r --cached .", "/workspace", False), ("find . -name '*.pyc' -delete", "/workspace", False),
    ("rm -rf -- *", "/workspace/projects/bbbbbbbbbbbb", False),          # inside its own project
    ("rm -rf -- *", "/workspace", True), ("rm -rf ..", "/workspace/projects/bbbbbbbbbbbb", True),
    ("rm -rf ../aaaaaaaaaaaa", "/workspace/projects/bbbbbbbbbbbb", True),
    ("rm -r /workspace/projects", "/workspace/projects/bbbbbbbbbbbb", True),
])
def test_bulk_removal_second_review(cmd, wd, blocked):
    from ghost_agent.tools.execute import _released_shell_block
    assert bool(_released_shell_block(_Store(), cmd, workdir=wd)) is blocked


def test_an_entity_forget_removes_facts_that_share_its_word(tmp_path):
    """0.3 alone removed nothing for "forget my address" (0.57)."""
    from ghost_agent.tools.memory import tool_unified_forget
    ms = _sweep_ms([("addr", 0.57, "The user's home address is 14 Kifisias, Athens"),
                    ("job", 0.41, "The user works as a software engineer at Google"),
                    ("bday", 0.55, "The user's birth date is 1980-01-29")])
    asyncio.run(tool_unified_forget("my address", tmp_path, ms))
    assert _deleted_ids(ms) == ["addr"]


def test_a_refused_profile_write_is_not_reported_as_removed(tmp_path, monkeypatch):
    from ghost_agent.memory.profile import ProfileMemory
    p = ProfileMemory(tmp_path)
    p.update("assets", "pets", "Mortimer the iguana")
    _real = Path.read_text

    def _eio(self, *a, **k):
        if self == p.file_path:
            raise OSError(5, "Input/output error")
        return _real(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", _eio)
    assert p.prune_value("assets", "pets", "mortimer").startswith("Error")
    from ghost_agent.tools.memory import _profile_line
    assert _profile_line("Error: x", "Removed a.b").startswith("⚠️")


@pytest.mark.parametrize("rid", ["probe-1234", "job-job-1", "sched-x"])
def test_probe_and_internal_turns_bump_no_usage_counters(rid, tmp_path):
    from ghost_agent.utils.logging import request_id_context
    from ghost_agent.memory.skills import usage_credit_blocked
    tok = request_id_context.set(rid)
    try:
        assert usage_credit_blocked()
        sm = _pb(tmp_path, [{"task": "a lesson", "solution": "x", "retrievals": 0}])
        assert sm.record_retrievals_bulk(["a lesson"]) == 0
    finally:
        request_id_context.reset(tok)
    tok = request_id_context.set("abc12345")
    try:
        assert not usage_credit_blocked()
    finally:
        request_id_context.reset(tok)


def test_a_sub_agent_stays_sim_and_a_labelled_turn_keeps_its_label():
    from ghost_agent.utils.logging import request_id_context
    ctx = MagicMock()
    ctx.turn_origin_label = None
    ctx.skill_memory.is_read_only = True
    tok = request_id_context.set("sub-agent-1")
    try:
        assert A.turn_origin(ctx) == "sim"
        ctx.turn_origin_label = "bench"
        assert A.turn_origin(ctx) == "bench"
    finally:
        request_id_context.reset(tok)






@pytest.mark.parametrize("fix,bad", [
    ("Remove all debug print statements before committing", False), ("Clear all filters before searching", False),
    ("Use DROP TABLE IF EXISTS in the test fixture", False), ("rm -rf /tmp/build", False),
    ("remove all the files in the sandbox", True), ("Delete everything.", True),
])
def test_the_destruction_screen_second_review(fix, bad):
    from ghost_agent.memory.lesson_quality import prescribes_destruction
    assert prescribes_destruction(fix) is bad


def test_forget_scoped_inside_a_released_project_keeps_its_files(tmp_path):
    from ghost_agent.tools.memory import tool_unified_forget
    sb = _sandbox(tmp_path)
    ms = MagicMock()
    ms.get_library.return_value = []
    ms.collection.query.return_value = {"ids": None}
    asyncio.run(tool_unified_forget("app.py", sb / "projects" / RELEASED, ms, project_store=_Store()))
    assert (sb / "projects" / RELEASED / "app.py").exists()



def test_deleting_a_link_to_the_projects_folder_removes_only_the_link(tmp_path):
    sb = _sandbox(tmp_path)
    os.symlink(sb / "projects", sb / "plink")
    out = str(_fs(sb, _Store(), operation="delete", path="plink"))
    assert out.startswith("SUCCESS") and (sb / "projects" / RELEASED / "app.py").exists()


# ── third review ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kw", [
    {"operation": "write", "file": f"projects/{RELEASED}/app.py", "content": "pwned"},
    {"operation": "delete", "file": f"projects/{RELEASED}/app.py"},
    {"operation": "rename", "path": "a.txt", "new_name": f"projects/{RELEASED}/x"},
    {"operation": "move", "path": "a.txt", "target": f"projects/{RELEASED}/x"},
    {"operation": "move", "path": "a.txt", "new_path": f"projects/{RELEASED}/x"},
    {"operation": "copy", "path": "a.txt", "data": f"projects/{RELEASED}/x"},
    {"operation": "copy", "path": "a.txt", "text": f"projects/{RELEASED}/x"},
])
def test_every_alias_parameter_is_checked_by_the_released_lock(tmp_path, kw):
    sb = _sandbox(tmp_path)
    (sb / "a.txt").write_text("x")
    out = str(_fs(sb, _Store(), **kw))
    assert "RELEASED" in out
    assert (sb / "projects" / RELEASED / "app.py").read_text() == "print(1)\n"
    assert not (sb / "projects" / RELEASED / "x").exists()


def test_a_copy_into_dotgit_through_data_is_refused(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    out = str(_fs(tmp_path, None, operation="copy", path="a.txt", data="repo/.git/config"))
    assert "SYSTEM BLOCK" in out and not (tmp_path / "repo" / ".git" / "config").exists()


@pytest.mark.skipif(not Path("/tmp").samefile("/private/tmp"), reason="macOS case-insensitive volume only")
@pytest.mark.parametrize("variant", ["PROJECTS", "Projects"])
def test_a_case_variant_of_the_projects_folder_is_locked(tmp_path, variant):
    sb = _sandbox(tmp_path)
    if not (sb / variant).exists():
        pytest.skip("case-sensitive filesystem")
    out = str(_fs(sb, _Store(), operation="delete", path=variant))
    assert "SYSTEM BLOCK" in out and (sb / "projects" / RELEASED / "app.py").exists()


def test_forget_never_deletes_through_a_hub_word(tmp_path):
    from ghost_agent.tools.memory import tool_unified_forget
    facts = [(f"f{i}", 0.5, t) for i, t in enumerate([
        "The user's birth date is 1980-01-29", "The user lives in Thrakomakedones", "The user's nickname is Bill",
        "The user works as a software engineer at Google"])]
    ms = _sweep_ms(facts)
    asyncio.run(tool_unified_forget("user nickname", tmp_path, ms))
    assert _deleted_ids(ms) == ["f2"]


def test_an_ambiguous_word_forget_deletes_nothing_and_names_the_candidates(tmp_path):
    from ghost_agent.tools.memory import tool_unified_forget
    ms = _sweep_ms([("home", 0.73, "The user's home address is 14 Kifisias"),
                    ("mail", 0.73, "The user's email address is v@example.com")])
    out = str(asyncio.run(tool_unified_forget("my address", tmp_path, ms)))
    assert _deleted_ids(ms) == [] and "NOT deleted" in out


@pytest.mark.parametrize("cmd,wd", [
    ("rm -rf $(pwd)/*", "/workspace"), ("rm -rf `pwd`", "/workspace"), ("find . -type f -delete", "/workspace"),
    ("find . -exec rm -rf {} +", "/workspace"), ("ls | xargs rm -rf", "/workspace"), ("rm -rf projects/a*", "/workspace"),
    ("rm -rf /workspace/./projects", "/workspace"), ("rm -rf /workspace//projects", "/workspace"),
    ("cd projects; rm -rf aaaaaaaaaaaa", "/workspace"), ("cd / && rm -rf workspace", "/workspace/projects/bbbbbbbbbbbb"),
    ("mv -t /tmp projects", "/workspace"), ("python3 -c 'import shutil; shutil.rmtree(\"projects\")'", "/workspace"),
    ("rm -rf /workspace/projects/AAAAAAAAAAAA", "/workspace"), ("rm -rf PROJECTS", "/workspace"),
])
def test_bulk_removal_bypasses_are_refused(cmd, wd):
    from ghost_agent.tools.execute import _released_shell_block
    assert "RELEASED" in str(_released_shell_block(_Store(), cmd, workdir=wd))


@pytest.mark.parametrize("cmd,unsafe", [
    ("FOO=1 rm a b", True), ("nohup rm x", True), ("bash -c 'rm a b'", True), ("xargs rm < list", True),
    ("echo hi &> out.txt", True), ('echo "a>b"', False), ("python3 -c 'print(1>0)'", False),
])
def test_rerun_safety_third_review(cmd, unsafe):
    from ghost_agent.tools.execute import _rerun_unsafe
    assert _rerun_unsafe(cmd) is unsafe


@pytest.mark.parametrize("fix,bad", [
    ("rm -rf /tmp/../workspace", True), ("rm -rf ./projects", True), ("find . -type f -delete", True),
    ("shutil.rmtree('projects')", True), ("TRUNCATE users", True), ("DELETE FROM users", True), ("git clean -fdx", True),
    ("Clear the workspace first", True), ("delete the projects folder", True), ("delete path=projects/aaaaaaaaaaaa", True),
    ("remove all duplicate rows from the dataframe", False), ("Remove all items from the cart", False),
    ("remove all files matching *.pyc", False), ("DELETE FROM users WHERE id = 3", False),
])
def test_the_destruction_screen_third_review(fix, bad):
    from ghost_agent.memory.lesson_quality import prescribes_destruction
    assert prescribes_destruction(fix) is bad


def test_a_token_less_request_credits_nothing():
    assert not A._query_has_tokens("?") and not A._query_has_tokens("👍") and A._query_has_tokens("restart the service")




def test_only_named_relations_and_birthdates_are_durable():
    """`employer_name` can go stale; `wife_name` and a son's birthdate cannot."""
    from ghost_agent.memory.profile import ProfileMemory
    old = "2020-01-01T00:00:00"
    assert ProfileMemory._staleness_marker("wife_name", old) == ""
    assert ProfileMemory._staleness_marker("son_thodoris_birthdate", old) == ""
    assert ProfileMemory._staleness_marker("employer_name", old) != ""
