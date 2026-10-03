"""§4KW — fourth fresh-eye review (lesson code, tool guards, live data,
verification quality). Each test names the world it fails in."""
import asyncio
import os
import stat
from pathlib import Path

import pytest

from ghost_agent.tools import file_system as FS
from ghost_agent.tools.execute import _bulk_destructive, _released_shell_block, _rerun_unsafe
from ghost_agent.memory.lesson_quality import prescribes_destruction

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


# ── the lesson destruction screen ────────────────────────────────────────────
@pytest.mark.parametrize("fix", [
    "Use execute with command='rm -rf *' to clear old outputs.", 'execute(command="rm -rf ./*")',
    'file_system(operation="delete", path="/workspace/projects")', "bash -c 'rm -rf projects'",
    "find /workspace -mindepth 1 -delete", "Remove every file in /workspace.", "Wipe /workspace clean.",
    "Clean up with rm -rf *", "shutil.rmtree('projects')", "delete path=projects/aaaaaaaaaaaa",
])
def test_a_lesson_written_as_a_tool_call_is_screened(fix):
    """Fails in the world where a command inside a quoted tool argument was
    one opaque token — the original incident's own form."""
    assert prescribes_destruction(fix) is True


@pytest.mark.parametrize("fix", [
    "Remove the node_modules folder and reinstall: rm -rf node_modules && npm install",
    "Clean pycache: find . -type d -name __pycache__ -exec rm -rf {} +",
    "Delete the files you created in /tmp after testing", "Clear the cache directory.",
    "Delete the directory you just created (rm -r out).", "Do not use rm -rf . to clean up; delete files by name.",
])
def test_a_targeted_cleanup_lesson_is_not_screened(fix):
    """Fails in the world where any named folder, a narrowed find that does
    not LEAD the line, or a negation followed by a period was refused."""
    assert prescribes_destruction(fix) is False


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


# ── same request ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("a,b", [
    ("install nginx on the server", "uninstall nginx on the server"),
    ("compress the logs folder", "decompress the logs folder"),
    ("activate the venv for the bot", "deactivate the venv for the bot"),
    ("archive the old projects", "unarchive the old projects"),
    ("are there any services up ?", "is the service up ?"),
    ("Hello, how are you?", "how's the"),
    ("delete this project", "delete this project"), ("redo", "redo"),
])
def test_requests_that_differ_in_meaning_are_not_the_same(a, b):
    """Fails in the world where a prefix that inverts a verb, a plural, or a
    lone shared word passed as a typo/the same request."""
    from ghost_agent.memory.lesson_scope import same_request
    assert same_request(a, b) is False


@pytest.mark.parametrize("a,b", [
    ("show me all projects", "show me all projets"), ("who am I ?", "Who am I?"),
    ("Create a photorealistic image that looks like a cat in Athens",
     "create a photorealistic image that looks like a cat in athens"),
    ("Build me a small tool logslow.py that reads the log", "build me a small tool logslow.py that reads the log"),
])
def test_a_resent_request_is_the_same(a, b):
    """Fails in the world where any "that"/"it" or a single content word made
    a request unmatchable — 13 of 70 live scoped lessons never matched even
    their own request."""
    from ghost_agent.memory.lesson_scope import same_request
    assert same_request(a, b) is True


@pytest.mark.skipif(not os.path.exists("/usr/share/dict/words"), reason="needs the system word list")
def test_a_name_opening_the_trigger_is_specific():
    """Fails in the world where `(?<!^)` exempted the first word: a lesson
    about one team was stored as a general rule."""
    from ghost_agent.memory.lesson_scope import is_general_trigger
    req = "do you know panerithraikos?"
    assert is_general_trigger("Panerithraikos questions: verify the team's league standing", req) is False
    assert is_general_trigger("Verify the league standings before answering sports questions",
                              "verify the panerithraikos league") is True


# ── dedup keeps plans for different requests apart ───────────────────────────
def test_plans_for_different_requests_do_not_merge(tmp_path):
    """Fails in the world where "copy A to B" and "copy B to A" merged into one
    row (same sorted key) and one request was served the other's plan."""
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    a, b = "copy report.txt to backup.txt on the server", "copy backup.txt to report.txt on the server"
    sm.learn_lesson(a, "m", "cp report.txt backup.txt", trigger=a, scope="request", source_request=a,
                    source="reflection")
    out = sm.learn_lesson(b, "m", "cp backup.txt report.txt and verify", trigger=b, scope="request",
                          source_request=b, source="reflection")
    rows = sm._load_playbook()
    row_a = next(r for r in rows if r.get("source_request") == a)
    assert row_a["solution"] == "cp report.txt backup.txt"
    assert out is None or any(r.get("source_request") == b for r in rows)


def test_a_vector_twin_for_another_request_is_its_own_row(tmp_path):
    from unittest.mock import MagicMock
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    a, b = "restart the chess service now", "stop the chess service now"
    sm.save_playbook([{"trigger": a, "task": a, "mistake": "m", "solution": "systemctl restart chess",
                       "scope": "request", "source_request": a, "source": "reflection"}])
    sm._find_duplicate_lesson = lambda *x, **k: {"source": "vector", "trigger": a, "text": "", "distance": 0.08}
    ms = MagicMock()
    sm.learn_lesson(b, "m", "systemctl stop chess and check", ms, trigger=b, scope="request",
                    source_request=b, source="reflection")
    rows = sm._load_playbook()
    assert {r["source_request"] for r in rows} == {a, b}
    assert next(r for r in rows if r["source_request"] == a)["solution"] == "systemctl restart chess"


# ── retrieval ────────────────────────────────────────────────────────────────
def test_an_orphaned_twin_of_a_scoped_plan_is_not_served(tmp_path):
    """Fails in the world where a twin with no row skipped the scope check and
    the raw one-request plan reached every request."""
    from unittest.mock import MagicMock
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    sm.save_playbook([{"trigger": "When asked about a person, search first", "solution": "search",
                       "mistake": "m"}])
    ms = MagicMock()
    ms.embedding_fn = None
    ms.collection.query.return_value = {
        "documents": [["SITUATION: copy the files to the server\nSOLUTION: rsync -a x y"]],
        "distances": [[0.1]], "metadatas": [[{"trigger": "copy the files to the server", "scope": "request"}]]}
    assert sm.get_playbook_items("copy the files to the server please", ms) == []


def test_a_malformed_distance_setting_keeps_the_default(monkeypatch):
    """Fails in the world where `GHOST_LESSON_TRIGGER_DISTANCE=0,3` raised at
    import and the agent could not boot."""
    from ghost_agent.memory import skills
    monkeypatch.setenv("GHOST_LESSON_TRIGGER_DISTANCE", "0,3")
    assert skills._env_distance("GHOST_LESSON_TRIGGER_DISTANCE", 0.30) == 0.30
    monkeypatch.setenv("GHOST_LESSON_TRIGGER_DISTANCE", "0.25")
    assert skills._env_distance("GHOST_LESSON_TRIGGER_DISTANCE", 0.30) == 0.25


# ── behaviour pins replacing source-text pins (fourth review) ───────────────
class _Traj:
    def __init__(self, request="copy report.txt to backup.txt on the server", plan="1. cp report.txt backup.txt",
                 verified=True, note="", general=None):
        self.user_request = request
        self.planning_output = plan
        self.final_response = ""
        self.extra = {"source_failure_reason": "wrong file", "reflected_from": "t1",
                      "plan_verified": verified, "plan_verify_note": note, "general_lesson": general or {}}


_GEN = {"situation": "When copying files between locations on a remote host",
        "mistake": "Copying in the wrong direction", "rule": "State the source and the destination before copying."}


def _sink():
    from unittest.mock import MagicMock
    from ghost_agent.reflection.sink import make_reflection_sink
    sm = MagicMock()
    return make_reflection_sink(MagicMock(), sm, None), sm


def test_the_reflection_sink_writes_the_plan_request_scoped():
    sink, sm = _sink()
    sink(_Traj())
    kw = sm.learn_lesson.call_args_list[0].kwargs
    assert kw["scope"] == "request" and kw["source_request"].startswith("copy report.txt")
    assert kw["verified"] is True


def test_a_confirmed_plan_also_yields_its_general_rule():
    sink, sm = _sink()
    sink(_Traj(general=_GEN))
    assert len(sm.learn_lesson.call_args_list) == 2
    gen = sm.learn_lesson.call_args_list[1].kwargs
    assert gen["task"] == _GEN["situation"] and "scope" not in gen


def test_an_unconfirmed_plan_yields_no_general_rule():
    """Fails in the world where a plan nobody judged (no verdict) still
    produced a rule that reaches every matching request."""
    sink, sm = _sink()
    sink(_Traj(verified=None, general=_GEN))
    assert len(sm.learn_lesson.call_args_list) == 1


@pytest.mark.parametrize("rule", ["Always run `cp report.txt backup.txt` first.",
                                  "Copy report.txt to backup.txt on the server."])
def test_a_general_lesson_needs_a_general_rule_too(rule):
    sink, sm = _sink()
    sink(_Traj(general=dict(_GEN, rule=rule)))
    assert len(sm.learn_lesson.call_args_list) == 1


def test_a_rejected_plan_is_not_a_lesson_but_a_non_answer_is():
    sink, sm = _sink()
    sink(_Traj(verified=False, note="REFUTED: misses the cause"))
    assert sm.learn_lesson.call_count == 0
    sink(_Traj(verified=False, note="no verdict: the judge timed out"))
    assert sm.learn_lesson.call_count == 1


@pytest.mark.parametrize("reply,verified,non_answer", [
    ("VERDICT: CONFIRMED\nit fixes the cause", True, False),
    ("VERDICT: REFUTED\nit repeats the failure", False, False),
    ("**Verdict**: confirmed", True, False),
    ("The plan cannot be considered CONFIRMED — it ignores the cause", False, False),
    ("I am not sure what to say", False, True), ("", False, True),
])
def test_the_plan_verdict_parser(reply, verified, non_answer):
    """Fails in the world where the "VERDICT:" prefix made the first-line
    branches unreachable, or a reply with neither word counted as a rejection."""
    from ghost_agent.reflection.sink import parse_plan_verdict
    v, note = parse_plan_verdict(reply)
    assert v is verified and note.startswith("no verdict") is non_answer


def test_a_job_wake_runs_once_under_its_own_id_and_records_its_conclusion(monkeypatch):
    from unittest.mock import MagicMock
    from ghost_agent import main as M
    import ghost_agent.tools.tasks as T
    import ghost_agent.core.autonomous_activity as AA
    seen, rec = [], []

    async def _fg(ctx, body, rid):
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
    ctx = SimpleNamespace(skill_memory=SimpleNamespace(is_read_only=False))
    if ctx_kind:
        ctx.trajectory_task_kind = ctx_kind
    tok = request_id_context.set(origin_id)
    try:
        assert trajectory_task_kind(ctx) == expect
    finally:
        request_id_context.reset(tok)


@pytest.mark.parametrize("origin_id", ["probe-deadbeef", "sub-1234"])
def test_probe_and_internal_turns_never_calibrate(origin_id):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from ghost_agent.core import agent as A
    from ghost_agent.utils.logging import request_id_context
    agent = A.GhostAgent.__new__(A.GhostAgent)
    agent.context = SimpleNamespace(skill_memory=SimpleNamespace(is_read_only=False),
                                    calibration_tracker=MagicMock(), _calib_pending=None)
    tok = request_id_context.set(origin_id)
    try:
        asyncio.run(agent._record_calibration_safe(req_id=origin_id, tools_run=[], verifier_backfill=None,
                                                    execution_failure_count=0, budget_exhausted=False,
                                                    final_ai_content="ok", user_request="hello"))
    finally:
        request_id_context.reset(tok)
    assert not agent.context.calibration_tracker.method_calls


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
                                                    self_model=sm,
                                                    skill_memory=SimpleNamespace(is_read_only=False)),
                            _flush_stashed_lesson_outcome=lambda *a: None)
    assert apply_human_label(agent, "r1", "positive")["ok"]
    assert sm.record_outcome.called is credited


def test_the_credit_helper_credits_only_a_clean_teaching_turn_with_a_request():
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from ghost_agent.core import agent as A
    from ghost_agent.utils.logging import request_id_context
    agent = A.GhostAgent.__new__(A.GhostAgent)
    sm = MagicMock()
    agent.context = SimpleNamespace(skill_memory=sm)
    tok = request_id_context.set("chatcmpl-1")
    try:
        asyncio.run(agent._credit_turn_lessons(0, "restart the chess service"))
        assert sm.credit_recent_retrievals.call_args.kwargs == {"query": "restart the chess service"}
        sm.reset_mock()
        asyncio.run(agent._credit_turn_lessons(1, "restart the chess service"))
        asyncio.run(agent._credit_turn_lessons(0, "👍"))
        assert not sm.credit_recent_retrievals.called
    finally:
        request_id_context.reset(tok)
    tok = request_id_context.set("probe-deadbeef")
    try:
        asyncio.run(agent._credit_turn_lessons(0, "restart the chess service"))
        assert not sm.credit_recent_retrievals.called
    finally:
        request_id_context.reset(tok)


def _calls_named(fn_node, name):
    import ast
    return [n for n in ast.walk(fn_node) if isinstance(n, ast.Call) and (
        (isinstance(n.func, ast.Attribute) and n.func.attr == name)
        or (isinstance(n.func, ast.Name) and n.func.id == name))]


def test_both_turn_paths_credit_through_the_helper():
    """Both the finalize and the streamed path call the one helper (AST call
    nodes, not source text)."""
    import ast
    import inspect
    from ghost_agent.core import agent as A
    tree = ast.parse(inspect.getsource(A))
    assert len(_calls_named(tree, "_credit_turn_lessons")) == 2
    direct = [c for c in _calls_named(tree, "to_thread")
              if c.args and isinstance(c.args[0], ast.Attribute) and c.args[0].attr == "credit_recent_retrievals"]
    assert len(direct) == 1          # only inside the helper






def test_the_trajectory_record_reads_this_turns_lessons_only():
    """The recorder asks the turn-keyed helper (with the turn id), never the
    unguarded `last_playbook_triggers` attribute (AST nodes)."""
    import ast
    import inspect
    from ghost_agent.core import agent as A
    tree = ast.parse(inspect.getsource(A))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_record_turn_trajectory")
    calls = _calls_named(fn, "_surfaced_lesson_triggers")
    assert calls and all(any(k.arg == "turn_id" for k in c.keywords) for c in calls)
    # neither getattr(…, "last_playbook_triggers") nor ….last_playbook_triggers
    assert not any((isinstance(n, ast.Constant) and n.value == "last_playbook_triggers")
                   or (isinstance(n, ast.Attribute) and n.attr == "last_playbook_triggers") for n in ast.walk(fn))


def test_hydration_judges_scoped_lessons_against_the_users_request():
    """Fails in the world where each sub-query was the scope request."""
    from ghost_agent.core.bus import MemoryBus
    bus = MemoryBus()
    got = []

    async def _dec(q, llm):
        return [q, "a derived sub-query"]

    async def _fetch(sq, **kw):
        got.append(kw.get("scope_request"))
        return []
    bus._decompose_query = _dec
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
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    sm.save_playbook([dict(r, mistake="m", solution="s", timestamp="2026-09-01T00:00:00") for r in rows])
    ms = MagicMock()
    ms.embedding_fn = _AngleEmb(dist)
    ms.collection.query.return_value = {
        "documents": [[f"SITUATION: {r['trigger']}\nSOLUTION: s" for r in rows]],
        "distances": [[0.2] * len(rows)], "metadatas": [[{"trigger": r["trigger"]} for r in rows]]}
    return [it["trigger"] for it in sm.get_playbook_items(query, ms)]


def test_the_trigger_gate_threshold_is_030(tmp_path):
    """Kills 0.30 → 0.6 and 0.30 → 0.05: a no-overlap trigger at 0.2 enters,
    one at 0.4 does not."""
    rows = [{"trigger": "zebra quokka lantern"}, {"trigger": "violet marble harbor"}]
    got = _gate(tmp_path, rows, {"zebra quokka lantern": 0.2, "violet marble harbor": 0.4})
    assert got == ["zebra quokka lantern"]


def test_a_shared_term_admits_a_distant_trigger(tmp_path):
    """Kills "distance only": keyword overlap alone admits."""
    rows = [{"trigger": "gamma ray spectroscopy notes"}]
    assert _gate(tmp_path, rows, {"gamma ray spectroscopy notes": 0.9}) == [rows[0]["trigger"]]


def test_the_gate_measures_against_the_query_not_the_turn_request(tmp_path):
    from ghost_agent.memory.lesson_scope import current_request
    rows = [{"trigger": "zebra quokka lantern"}]
    tok = current_request.set("something else entirely")
    try:
        got = _gate(tmp_path, rows, {"zebra quokka lantern": 0.1, "something else entirely": 0.0})
    finally:
        current_request.reset(tok)
    assert got == ["zebra quokka lantern"]


def test_the_gate_fails_open_with_no_embedder(tmp_path):
    from unittest.mock import MagicMock
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    row = {"trigger": "zebra quokka lantern", "mistake": "m", "solution": "s"}
    sm.save_playbook([row])
    ms = MagicMock()
    ms.embedding_fn = None
    ms.collection.query.return_value = {"documents": [["SITUATION: zebra quokka lantern\nSOLUTION: s"]],
                                        "distances": [[0.2]], "metadatas": [[{"trigger": row["trigger"]}]]}
    assert [it["trigger"] for it in sm.get_playbook_items("alpha beta", ms)] == [row["trigger"]]


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


def test_a_replaced_fix_is_re_embedded_exactly_once(tmp_path):
    """Kills: the refresh not deleting the old twin (the heal then sees it and
    skips), refreshing on every merge, and no refresh on the vector path."""
    from ghost_agent.memory.skills import SkillMemory, lesson_embedding_text, _normalize_lesson
    sm = SkillMemory(tmp_path)
    vec = _Vec()
    sm.learn_lesson("When parsing dates from logs", "guessed", "use ISO", vec, source="reflection")
    assert len(vec.collection.rows) == 1
    sm.learn_lesson("When parsing dates from logs", "guessed", "use ISO 8601 and the log's own timezone", vec,
                    source="reflection")
    docs = [d for _m, d in vec.collection.rows.values()]
    row = sm._load_playbook()[0]
    assert docs == [lesson_embedding_text(_normalize_lesson(row))]
    n = vec.n
    sm.learn_lesson("When parsing dates from logs", "guessed", "short", vec, source="reflection")
    assert vec.n == n                      # nothing replaced: no re-embed


def test_heal_writes_one_twin_for_rows_sharing_a_trigger(tmp_path):
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    sm.save_playbook([{"trigger": "Parallel processing with ordering", "mistake": "a", "solution": "x"},
                      {"trigger": "Parallel processing with ordering", "mistake": "b", "solution": "y"}])
    vec = _Vec()
    assert sm.heal_missing_twins(vec) == 1 and len(vec.collection.rows) == 1


def test_heal_skips_a_row_removed_since_its_snapshot(tmp_path, monkeypatch):
    """Kills the per-lesson re-check under the lock (fourth review m5)."""
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    sm.save_playbook([{"trigger": "When the row is gone", "mistake": "a", "solution": "x"}])
    first = sm._load_playbook()
    calls = {"n": 0}
    real = sm._load_playbook

    def _load():
        calls["n"] += 1
        return first if calls["n"] == 1 else []
    monkeypatch.setattr(sm, "_load_playbook", _load)
    vec = _Vec()
    assert sm.heal_missing_twins(vec) == 0 and not vec.collection.rows


@pytest.mark.parametrize("a,b,same", [
    ("show the logs folder", "show the lgos folder", False),        # 4-letter words: exact only
    ("delete that file", "delete that file", False),                  # short + deictic
    ("restart restart the chess service", "restart the chess service", True),
])
def test_same_request_edges(a, b, same):
    from ghost_agent.memory.lesson_scope import same_request
    assert same_request(a, b) is same


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


def test_a_legacy_row_without_a_producer_merges_as_evidence_only(tmp_path):
    """JSON path: a row with no recorded producer is foreign — its fix and
    verified flag stay; only its frequency moves."""
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    sm.save_playbook([{"trigger": "When copying between hosts", "task": "When copying between hosts",
                       "mistake": "m", "solution": "old fix", "frequency": 1, "verified": False}])
    sm.learn_lesson("When copying between hosts", "m", "a much longer replacement fix text here", source="reflection",
                    verified=True)
    row = sm._load_playbook()[0]
    assert row["solution"] == "old fix" and row.get("verified") is not True and row["frequency"] == 2


def test_a_request_naming_a_home_path_matches_its_redacted_record():
    """Fails in the world where the stored request carried `/Users/<user>`
    (redacted on record) and the live one the real home folder."""
    from ghost_agent.memory.lesson_scope import same_request
    rec = "Using the file system tool, count the lines in /Users/<user>/Data/AI/Agent/PROJECT_JOURNAL.md"
    live = "Using the file system tool, count the lines in /Users/vasilis/Data/AI/Agent/PROJECT_JOURNAL.md"
    assert same_request(rec, live)
    assert not same_request(rec, live.replace("PROJECT_JOURNAL", "README"))


@pytest.mark.parametrize("a,b", [
    ("restart the service now", "restart the services now"),          # one vs many
    ("show any errors in the log", "show errors in the log"),          # a quantity word is content
    ("summarise the present situation report", "summarise the president situation report"),
    ("show the listing of open files", "show the listening of open files"),
])
def test_close_words_that_differ_are_not_typos(a, b):
    """Fails in the world where a plural, a dropped quantity word, or a
    ratio-similar word two edits away passed as the same request."""
    from ghost_agent.memory.lesson_scope import same_request
    assert same_request(a, b) is False


def test_heal_marks_its_own_writes_when_the_recheck_cannot_read(tmp_path):
    """The in-run mark is the fallback when the per-lesson re-check fails:
    two rows with one trigger still get ONE twin."""
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    sm.save_playbook([{"trigger": "Parallel processing with ordering", "mistake": "a", "solution": "x"},
                      {"trigger": "Parallel processing with ordering", "mistake": "b", "solution": "y"}])
    vec = _Vec()
    _get = vec.collection.get

    def _get_or_fail(where=None, **kw):
        if where and "$and" in where:
            raise RuntimeError("store busy")
        return _get(where=where, **kw)
    vec.collection.get = _get_or_fail
    assert sm.heal_missing_twins(vec) == 1 and len(vec.collection.rows) == 1


def test_an_absolute_glob_matching_the_projects_folder_is_bulk():
    assert _bulk_destructive("rm -rf /workspace/proj*", _REL, "/workspace") is True
    assert _bulk_destructive("rm -rf /workspace/build*", _REL, "/workspace") is False


def test_a_hub_word_names_no_fact_even_when_only_one_mentions_it(tmp_path):
    """Fails in the world where a lone literal hit for "user" was deleted."""
    from ghost_agent.tools.memory import tool_unified_forget
    ms, deleted = _forget_ms([("u", 0.7, "The user lives in Athens")])
    out = asyncio.run(tool_unified_forget("user", tmp_path / "nosb", ms))
    assert deleted == [] and "NOT deleted" in str(out)


def test_a_confirmation_that_mentions_refuted_is_still_a_confirmation():
    """The first line's verdict decides; the whole-text fallback is only for a
    reply without one (kills keeping the "VERDICT:" prefix)."""
    from ghost_agent.reflection.sink import parse_plan_verdict
    v, _ = parse_plan_verdict("VERDICT: CONFIRMED\nNothing in the failure log refuted this plan.")
    assert v is True


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
