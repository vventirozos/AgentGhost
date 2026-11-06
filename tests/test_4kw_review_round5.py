"""§4KW — fourth fresh-eye review (lesson code, tool guards, live data,
verification quality). Each test names the world it fails in."""
import asyncio
import os
import stat
from pathlib import Path

import pytest

from ghost_agent.tools import file_system as FS
from ghost_agent.tools.execute import _bulk_destructive, _released_shell_block, _rerun_unsafe

RELEASED = "aaaaaaaaaaaa"
DEV = "bbbbbbbbbbbb"
_IMM = getattr(stat, "UF_IMMUTABLE", 0) if hasattr(os, "chflags") else 0


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
    return tmp_path


def _fs(sandbox_dir, store=None, **kw):
    return str(asyncio.run(FS.tool_file_system(sandbox_dir=sandbox_dir, project_store=store, **kw)))


# ── file_system lock ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("path", ["/workspace/projects", "/workspace/Projects"])
def test_the_projects_folder_is_locked_from_inside_an_active_project(tmp_path, path):
    """Fails in the world where the parent-folder check looked for
    `<active project>/projects`, did not find it, and `delete
    path=/workspace/projects` from an active project removed every project."""
    sb = _sandbox(tmp_path)
    out = _fs(sb / "projects" / DEV, _Store(), operation="delete", path=path)
    assert "SYSTEM BLOCK" in out
    assert (sb / "projects" / RELEASED / "app.py").exists()


@pytest.mark.parametrize("op,extra", [("delete", {}), ("write", {"content": "pwned"})])
def test_the_url_alias_is_checked_by_the_released_lock(tmp_path, op, extra):
    """Fails in the world where `url=` (healed into the target of any
    non-download operation) skipped the lock: the file was deleted/overwritten."""
    sb = _sandbox(tmp_path)
    out = _fs(sb, _Store(), operation=op, url=f"projects/{RELEASED}/app.py", **extra)
    assert "RELEASED" in out
    assert (sb / "projects" / RELEASED / "app.py").read_text() == "print(1)\n"


def test_a_copy_out_of_a_released_project_is_allowed(tmp_path):
    """Fails in the world where the copy SOURCE was locked like a write."""
    sb = _sandbox(tmp_path)
    out = _fs(sb, _Store(), operation="copy", path=f"projects/{RELEASED}/app.py", destination="copy.py")
    assert out.startswith("SUCCESS") and (sb / "copy.py").read_text() == "print(1)\n"


@pytest.mark.skipif(not _IMM, reason="BSD file flags only")
def test_a_released_workspace_is_immutable_even_after_chmod(tmp_path):
    """Fails in the world where release set only mode bits: the sandbox's
    root (FOWNER) ran `chmod -R u+w . && rm -rf *` and the files were gone."""
    from ghost_agent.memory.projects import ProjectStore
    ws = tmp_path / "ws"
    (ws / "sub").mkdir(parents=True)
    (ws / "sub" / "a.txt").write_text("x")
    ProjectStore._chmod_tree(ws, True)
    with pytest.raises(PermissionError):
        os.chmod(ws / "sub", 0o755)                 # the bits cannot be lifted
    with pytest.raises(PermissionError):
        (ws / "sub" / "a.txt").unlink()
    ProjectStore._chmod_tree(ws, False)
    (ws / "sub" / "a.txt").unlink()
    assert not (ws / "sub" / "a.txt").exists()


@pytest.mark.skipif(not _IMM, reason="BSD file flags only")
def test_a_copy_of_a_released_file_is_not_immutable(tmp_path):
    """Fails in the world where `copy2` carried the immutable flag onto the
    copy, which then could never be edited or deleted."""
    sb = _sandbox(tmp_path)
    from ghost_agent.memory.projects import ProjectStore
    ProjectStore._chmod_tree(sb / "projects" / RELEASED, True)
    out = _fs(sb, _Store(), operation="copy", path=f"projects/{RELEASED}/app.py", destination="copy.py")
    assert out.startswith("SUCCESS")
    assert not os.lstat(sb / "copy.py").st_flags & _IMM
    ProjectStore._chmod_tree(sb / "projects" / RELEASED, False)


# ── execute: bulk removal ────────────────────────────────────────────────────
_REL = [RELEASED]


@pytest.mark.parametrize("cmd", [
    'bash -c "rm -rf projects"', "sh -c 'rm -rf *'", "find . -name '*' -delete", "rm -rf proj*",
    'for d in projects/*; do rm -rf "$d"; done', "rm -rf {projects,tmp}", "cd projects && rm -rf abc*",
    "rm -rf /workspace/projects/cccccccccccc/..", "rm -rf ./projects/../projects",
    "tar --remove-files -czf a.tgz projects", "git -C /workspace clean -fdx", "cd && rm -rf *",
    "cd - && rm -rf *", "pushd /workspace && rm -rf *", "sudo -u root rm -rf projects", "timeout 5 rm -rf *",
    "mv --target-directory=/tmp projects", "find . -type f -delete", "rm -rf $(pwd)/*",
    "python3 -c \"import shutil; shutil.rmtree('projects')\"",
])
def test_bulk_removal_fourth_review_bypasses_are_refused(cmd):
    """Fails in the world where the bulk guard read the command TEXT: these
    passed with a released project present at the workspace root."""
    assert _bulk_destructive(cmd, _REL, "/workspace") is True


@pytest.mark.parametrize("cmd", [
    "grep -rn 'rm -rf' .", "grep -rln shutil.rmtree .", "grep -r truncate *", "python3 train.py --rm .",
    "cd /tmp && rm -rf *", f"rsync -a --delete projects/{RELEASED}/ /tmp/backup/", "rm -rf ..data",
    "find . -name '*.pyc' -delete", "rm -rf node_modules", "mv *.png images/", "git rm -r --cached .",
])
def test_bulk_removal_fourth_review_false_positives_pass(cmd):
    """Fails in the world where the verb regex matched quoted text and
    `--rm`, ignored a `cd` to a safe place, took an rsync SOURCE for a target,
    or took every name that starts with `..` for an escape."""
    assert _bulk_destructive(cmd, _REL, "/workspace") is False


def test_bulk_removal_tracks_the_directory_from_a_project():
    wd = f"/workspace/projects/{DEV}"
    assert _bulk_destructive("rm -rf *", _REL, wd) is False
    assert _bulk_destructive("cd .. && rm -rf *", _REL, wd) is True
    assert _bulk_destructive("rm -rf ..", _REL, wd) is True


# ── execute: released working directory and written paths ──────────────────
_R = f"projects/{RELEASED}"
_WD = f"/workspace/projects/{RELEASED}"


@pytest.mark.parametrize("cmd,wd", [
    ("rm -rf *", _WD), ("rm main.py", _WD), ("mv main.py old.py", _WD), ("echo x > main.py", _WD),
    ("sed -i s/a/b/ main.py", _WD), ("git clean -fdx", _WD), ("chmod -R u+w . && rm -rf *", _WD),
    (f"echo x > {_R}/a.txt", "/workspace"), (f"cd {_R} && rm main.py", "/workspace"),
    (f"tar xzf a.tgz -C {_R}", "/workspace"), (f"cp /tmp/x {_R}/x", "/workspace"),
])
def test_a_write_into_a_released_workspace_is_refused(cmd, wd):
    """Fails in the world where ids were read from the command text only:
    with the released project as the working directory everything passed."""
    assert _released_shell_block(_Store(), cmd, wd) is not None


@pytest.mark.parametrize("cmd,wd", [
    (f"cp -r {_R} /tmp/copy", "/workspace"), (f"tar czf /tmp/b.tgz {_R}", "/workspace"),
    (f"cat {_R}/x > /tmp/x", "/workspace"), ("python3 main.py", _WD), ("cp main.py /tmp/m.py", _WD),
    ("cd /tmp && echo x > a", _WD), (f"grep -rn foo {_R}", "/workspace"),
])
def test_reading_a_released_workspace_is_allowed(cmd, wd):
    """Fails in the world where any mutation token anywhere in a command that
    named a released id refused it — including a copy OUT of it."""
    assert _released_shell_block(_Store(), cmd, wd) is None


# ── execute: the re-run heal ─────────────────────────────────────────────────
@pytest.mark.parametrize("cmd,unsafe", [
    ("find . -name '*.tmp' -print0 | xargs -0 rm", True), ("ls *.log | xargs -r rm", True),
    ("ls | xargs -I{} rm {}", True), ("cat l | xargs -n1 rm -f", True), ("timeout 5 rm x", True),
    ("nice rm x", True), ("exec rm x", True), ("sudo -u root rm x", True),
    ("echo 'a > b'; cat missing", False), ("cmd 2>&1; cat missing", False),
])
def test_the_rerun_guard_sees_verbs_behind_options(cmd, unsafe):
    """Fails in the world where `xargs\\s+(\\w+)` needed the verb right after
    xargs, and a non-final segment's quoted `>` or `2>&1` counted as a write."""
    assert _rerun_unsafe(cmd) is unsafe


# ── forget ───────────────────────────────────────────────────────────────────
def _forget_ms(rows):
    from unittest.mock import MagicMock
    ms = MagicMock()
    docs = {i: (d, t) for i, d, t in rows}
    deleted = []

    def _q(**k):
        ids = list(docs)
        return {"ids": [ids], "distances": [[docs[i][0] for i in ids]],
                "documents": [[docs[i][1] for i in ids]], "metadatas": [[{"type": "auto"} for _ in ids]]}
    ms.collection.query.side_effect = _q
    ms.collection.delete.side_effect = lambda ids=None, **k: deleted.extend(ids or [])
    ms.collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    ms.get_library.return_value = []
    return ms, deleted


_OWNER = [("home", 0.55, "The user's home address is 12 Oak Street"),
          ("email", 0.6, "The user's email address is v@x.com"),
          ("job", 0.7, "The user works at Google"), ("nick", 0.75, "The user's nickname is Vas")]


@pytest.mark.parametrize("target,gone", [("user", set()), ("the user", set()), ("address", set()),
                                         ("home address", {"home"}), ("nickname", {"nick"})])
def test_a_literal_mention_of_a_hub_or_attribute_word_is_a_choice(tmp_path, target, gone):
    """Fails in the world where any LITERAL mention deleted: `forget user`
    removed all four owner facts, `forget address` the home AND the email."""
    from ghost_agent.tools.memory import tool_unified_forget
    ms, deleted = _forget_ms(_OWNER)
    out = asyncio.run(tool_unified_forget(target, tmp_path / "nosb", ms))
    assert set(deleted) == gone
    if not gone:
        assert "NOT deleted" in str(out)


def test_an_entity_named_by_several_facts_is_forgotten_everywhere(tmp_path):
    from ghost_agent.tools.memory import tool_unified_forget
    ms, deleted = _forget_ms([("i1", 1.5, "User previously had an iguana"),
                              ("i2", 1.5, "the iguana was named Mortimer"), ("d", 1.5, "User owns a dog")])
    asyncio.run(tool_unified_forget("iguana", tmp_path / "nosb", ms))
    assert set(deleted) == {"i1", "i2"}


def test_one_full_match_wins_over_partial_ones(tmp_path):
    """Fails in the world where a half-matching fact ("wife is named Maria")
    made "my wife's birthday" ambiguous and nothing was deleted."""
    from ghost_agent.tools.memory import tool_unified_forget
    ms, deleted = _forget_ms([("m", 0.5, "wife is named Maria"), ("b", 0.72, "wife birthday is 3 May")])
    asyncio.run(tool_unified_forget("my wife's birthday", tmp_path / "nosb", ms))
    assert deleted == ["b"]


@pytest.mark.parametrize("a,b,same", [("homework", "home", False), ("workout", "works", False),
                                      ("portfolio", "port", False), ("plan", "planet", False),
                                      ("weigh", "weighs", True), ("address", "addresses", True)])
def test_forget_words_match_by_inflection_not_by_prefix(a, b, same):
    from ghost_agent.tools.memory import _word_matches
    assert _word_matches(a, b) is same


def test_a_job_wake_runs_once_under_its_own_id_and_records_its_conclusion(monkeypatch):
    from unittest.mock import MagicMock
    from ghost_agent import main as M
    import ghost_agent.tools.tasks as T
    import ghost_agent.core.autonomous_activity as AA
    seen, rec = [], []

    async def _fg(ctx, body, rid, project_id=None):
        seen.append(rid)
        return "the job built 3 files", None, None
    monkeypatch.setattr(M, "_handle_chat_foreground", _fg)
    monkeypatch.setattr(T, "should_defer_scheduled_task", lambda *a, **k: False)
    monkeypatch.setattr(AA, "record_scheduled_result", lambda *a, **k: rec.append(k))
    M._RESUMED_JOBS.discard("job-1a2b3c4d")
    ctx = MagicMock(sandbox_manager=None)
    assert asyncio.run(M._resume_after_job(ctx, {"id": "job-1a2b3c4d", "state": "done", "exit_code": 0}))
    assert seen == ["job-1a2b3c4d"]
    assert rec and rec[0]["content"] == "the job built 3 files" and rec[0]["ok"] is True


@pytest.mark.parametrize("origin_id,ctx_kind,expect", [
    ("probe-deadbeef", None, "probe"), ("sub-1234", None, "internal"), ("sub-1234", "leaf", "leaf"),
    ("chatcmpl-1", None, "user_request"), ("chatcmpl-1", "bench", "bench"),
])
def test_the_trajectory_kind_follows_the_origin(origin_id, ctx_kind, expect):
    from types import SimpleNamespace
    from ghost_agent.core.agent import trajectory_task_kind
    from ghost_agent.utils.logging import request_id_context
    ctx = SimpleNamespace()
    if ctx_kind:
        ctx.trajectory_task_kind = ctx_kind
    tok = request_id_context.set(origin_id)
    try:
        assert trajectory_task_kind(ctx) == expect
    finally:
        request_id_context.reset(tok)


@pytest.mark.parametrize("kind,credited", [("user_request", True), ("probe", False), ("internal", False)])
def test_a_thumb_reaches_the_diary_only_for_a_user_row(tmp_path, kind, credited):
    """Fails in the world where the feedback request's own origin ("user") was
    read instead of the LABELLED row's: a thumb on a probe was booked."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from ghost_agent.core.feedback import apply_human_label
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.distill.schema import Trajectory
    from ghost_agent.selfhood import SelfModel
    c = TrajectoryCollector(root=tmp_path, session_id="t")
    t = Trajectory(session_id="r1", user_request="do it", final_response="done", extra={"req_id": "r1"},
                   task_kind=kind)
    c.append(t)
    sm = MagicMock(spec=SelfModel)
    sm.enabled = True
    agent = SimpleNamespace(context=SimpleNamespace(trajectory_collector=c, _recent_trajectories_for_correction={},
                                                    self_model=sm))
    assert apply_human_label(agent, "r1", "positive")["ok"]
    assert sm.record_outcome.called is credited


def _calls_named(fn_node, name):
    import ast
    return [n for n in ast.walk(fn_node) if isinstance(n, ast.Call) and (
        (isinstance(n.func, ast.Attribute) and n.func.attr == name)
        or (isinstance(n.func, ast.Name) and n.func.id == name))]


def test_hydration_judges_scoped_lessons_against_the_users_request():
    """Fails in the world where each sub-query was the scope request."""
    from ghost_agent.core.bus import MemoryBus
    bus = MemoryBus()
    got = []

    async def _fetch(sq, **kw):
        got.append(kw.get("scope_request"))
        return []
    bus._split_query = lambda q: [q, "a derived sub-query"]
    bus._fetch_all_tiers = _fetch
    asyncio.run(bus.hydrate_context("the expanded query", raw_user_text="the user's request"))
    assert got == ["the user's request", "the user's request"]
    got.clear()
    asyncio.run(bus.hydrate_context("the expanded query"))
    assert got == ["the expanded query", "the expanded query"]


# ── mutation survivors (fourth review, verification lens) ───────────────────
class _AngleEmb:
    """Query → e1; each named trigger at an exact cosine distance from it."""
    def __init__(self, dist):
        self.dist = dist

    def __call__(self, texts):
        import numpy as np
        out = []
        for t in texts:
            d = self.dist.get(t, 0.0)
            v = np.zeros(8)
            v[0], v[1] = 1.0 - d, (1.0 - (1.0 - d) ** 2) ** 0.5
            out.append(v)
        return out


def _gate(tmp_path, rows, dist, query="alpha beta gamma"):
    from unittest.mock import MagicMock
    sm = SkillMemory(tmp_path)
    sm.save_playbook([dict(r, mistake="m", solution="s", timestamp="2026-09-01T00:00:00") for r in rows])
    ms = MagicMock()
    ms.embedding_fn = _AngleEmb(dist)
    ms.collection.query.return_value = {
        "documents": [[f"SITUATION: {r['trigger']}\nSOLUTION: s" for r in rows]],
        "distances": [[0.2] * len(rows)], "metadatas": [[{"trigger": r["trigger"]} for r in rows]]}
    return [it["trigger"] for it in sm.get_playbook_items(query, ms)]


class _Coll:
    """A collection that honours `where={"trigger": …}` the way Chroma does."""
    def __init__(self):
        self.rows = {}

    def _match(self, meta, where):
        if not where:
            return True
        if "$and" in where:
            return all(self._match(meta, w) for w in where["$and"])
        return all(meta.get(k) == v for k, v in where.items())

    def get(self, where=None, limit=None, include=None, ids=None):
        hit = [(i, m, d) for i, (m, d) in self.rows.items() if self._match(m, where)]
        return {"ids": [h[0] for h in hit], "metadatas": [h[1] for h in hit], "documents": [h[2] for h in hit]}

    def delete(self, ids=None, where=None):
        for i in list(self.rows):
            if (ids and i in ids) or (where and self._match(self.rows[i][0], where)):
                del self.rows[i]


class _Vec:
    def __init__(self):
        self.collection = _Coll()
        self.n = 0

    def add(self, text, meta):
        self.n += 1
        self.collection.rows[f"id{self.n}"] = (dict(meta), text)
        return "stored"


@pytest.mark.parametrize("rows,gone", [
    ([("f", 0.85, "The colour the user picks as favourite is green")], set()),     # full cover, past 0.8
    ([("f", 0.75, "The colour the user picks as favourite is green")], {"f"}),
    ([("h", 0.55, "favourite band is Muse")], {"h"}),                   # half cover under 0.6
    ([("h", 0.65, "favourite band is Muse")], set()),
])
def test_forget_word_coverage_bounds(tmp_path, rows, gone):
    from ghost_agent.tools.memory import tool_unified_forget
    ms, deleted = _forget_ms(rows)
    asyncio.run(tool_unified_forget("favourite colour", tmp_path / "nosb", ms))
    assert set(deleted) == gone


def test_an_absolute_glob_matching_the_projects_folder_is_bulk():
    assert _bulk_destructive("rm -rf /workspace/proj*", _REL, "/workspace") is True
    assert _bulk_destructive("rm -rf /workspace/build*", _REL, "/workspace") is False


def test_a_hub_word_names_no_fact_even_when_only_one_mentions_it(tmp_path):
    """Fails in the world where a lone literal hit for "user" was deleted."""
    from ghost_agent.tools.memory import tool_unified_forget
    ms, deleted = _forget_ms([("u", 0.7, "The user lives in Athens")])
    out = asyncio.run(tool_unified_forget("user", tmp_path / "nosb", ms))
    assert deleted == [] and "NOT deleted" in str(out)


# ── fifth review: immutability ───────────────────────────────────────────────
@pytest.mark.skipif(not _IMM, reason="BSD file flags only")
def test_a_released_apps_database_stays_writable(tmp_path):
    """Fails in the world where the Jiu Jitsu Calendar's data.db (next to
    app.py) was made immutable and every save failed."""
    import sqlite3
    from ghost_agent.memory.projects import ProjectStore
    ws = tmp_path / "ws"
    (ws / "static").mkdir(parents=True)
    (ws / "app.py").write_text("x")
    (ws / "static" / "i.html").write_text("x")
    sqlite3.connect(ws / "data.db").execute("create table t(x)").connection.commit()
    ProjectStore._chmod_tree(ws, True)
    try:
        assert os.access(ws / "data.db", os.W_OK) and os.access(ws, os.W_OK)   # the bits too, not only the flag
        c = sqlite3.connect(ws / "data.db")
        c.execute("insert into t values (1)")
        c.commit()
        for op in (lambda: (ws / "app.py").unlink(), lambda: (ws / "static" / "i.html").unlink(),
                   lambda: (ws / "static").rename(ws / "s2")):
            with pytest.raises(PermissionError):
                op()
    finally:
        ProjectStore._chmod_tree(ws, False)


def test_a_project_without_a_workspace_path_flags_nothing(tmp_path, monkeypatch):
    """Fails in the world where `Path("")` — the current directory — was
    taken for the workspace and made immutable, recursively."""
    from ghost_agent.memory.projects import ProjectStore
    store = ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sandbox")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("x")
    monkeypatch.setattr(store, "get_project", lambda pid: {"id": pid, "workspace_dir": ""})
    assert store.set_workspace_readonly("aaaaaaaaaaaa", True) == 0
    monkeypatch.setattr(store, "get_project", lambda pid: {"id": pid, "workspace_dir": str(tmp_path)})
    assert store.set_workspace_readonly("aaaaaaaaaaaa", True) == 0        # outside the sandbox
    assert os.access(tmp_path / "f.txt", os.W_OK)


@pytest.mark.skipif(not _IMM, reason="BSD file flags only")
def test_a_copytree_fork_of_a_released_workspace_is_removable(tmp_path):
    from ghost_agent.core import isolation as I
    from ghost_agent.memory.projects import ProjectStore
    src = tmp_path / "src"
    (src / "projects" / RELEASED).mkdir(parents=True)
    (src / "projects" / RELEASED / "app.py").write_text("x")
    ProjectStore._chmod_tree(src / "projects" / RELEASED, True)
    try:
        import shutil
        dest = tmp_path / "fork"
        shutil.copytree(src, dest, symlinks=True)
        I._chmod_writable(dest)
        shutil.rmtree(dest)
        assert not dest.exists()
    finally:
        ProjectStore._chmod_tree(src / "projects" / RELEASED, False)
