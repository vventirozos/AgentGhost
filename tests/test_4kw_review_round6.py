"""§4KW — fifth fresh-eye review (shell guards, lessons/forget, live data,
verification). Each test names the world it fails in."""
import asyncio
import os
import stat
import time

import pytest

from ghost_agent.tools.shell_analysis import _bulk_destructive, _released_written_ids, _written_paths, _shell_segments
from ghost_agent.tools.execute import _released_shell_block, _rerun_unsafe
from ghost_agent.memory.lesson_quality import prescribes_destruction

REL = "48e0373aaab3"
DEV = "111111111111"
RW = f"/workspace/projects/{REL}"


class _Store:
    def list_projects(self, st=None):
        return [{"id": REL}]

    def get_project(self, pid):
        return {"status": "RELEASED"} if pid == REL else {"status": "DEVELOPMENT"}


# ── bulk removal ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cmd", [
    "(rm -rf projects)", "( rm -rf projects )", "(cd /workspace/projects; rm -rf *)",
    "true && (rm -rf /workspace/projects/*)", "(rm -rf *)",
    'bash -lc "rm -rf projects"', 'sh -ec "cd /workspace/projects && rm -rf *"', "bash -c -- 'rm -rf projects'",
    "eval rm -rf projects", "echo 'rm -rf projects' | bash", "busybox rm -rf projects", "/usr/bin/env rm -rf projects",
    "doas rm -rf projects", "/bin/rm -rf projects", "rm -rf /*",
    "find projects -maxdepth 1 -mindepth 1 -exec rm -rf {} +", "find /workspace/projects -mindepth 1 -maxdepth 1 -exec rm -rf {} +",
    "find . -mindepth 1 -maxdepth 1 -exec rm -rf {} +", "find /workspace -maxdepth 1 -exec rm -rf {} +",
    "find . -not -name '*.pyc' -delete", "find . ! -name x -delete", "find . -delete -name '*.pyc'",
    "git clean --force -d", "zip -rm /tmp/a.zip projects",
    "D=projects; rm -rf $D", 'P=/workspace/projects; rm -rf "$P"/*', "rm -rf ${WORKSPACE:-/workspace}/projects",
    "rm -rf $(ls -d projects)", "rm -rf `echo projects`",
    "node -e \"require('fs').rmSync('projects',{recursive:true})\"",
    "python3 -c \"import os; os.system('rm -rf projects')\"",
    "python3 -c \"import shutil,os; [shutil.rmtree(p) for p in os.listdir('.')]\"",
    "bash <<'EOF'\ncd /workspace\nrm -rf *\nEOF",
])
def test_fifth_review_bulk_bypasses_are_refused(cmd):
    """Fails in the world where subshell parentheses, combined `-lc` flags,
    eval, a pipe into a shell, wrappers by path, `-maxdepth` sweeps, negated
    find filters, variables, substitutions, interpreter one-liners or a
    shell-read heredoc hid a bulk removal."""
    assert _bulk_destructive(cmd, [REL], "/workspace") is True


@pytest.mark.parametrize("cmd", [
    "rm -rf build  # old artifacts, not projects", "rm -f *.log # clean logs in / as well",
    "cat > /tmp/clean.sh <<'EOF'\ncd /workspace\nrm -rf *\nEOF",
    "rm /tmp/48e0373aaab3-backup.tgz", "rm -rf $TMPDIR/x", "echo a#b", "find . -name '*.pyc' -delete",
])
def test_fifth_review_ordinary_commands_pass(cmd):
    """Fails in the world where a comment became an argument, a heredoc body
    (data) was run as commands, or an id inside a file name counted."""
    assert _bulk_destructive(cmd, [REL], "/workspace") is False


@pytest.mark.parametrize("cmd,bulk", [
    ("cd src && rm -rf ../build", False), ("cd src; cd ..; rm -rf *", False), ("rm -rf *", False),
    ("cd .. && rm -rf *", True), ("rm -rf ..", True),
])
def test_the_directory_is_tracked_inside_a_project(cmd, bulk):
    assert _bulk_destructive(cmd, [REL], f"/workspace/projects/{DEV}") is bulk


def test_the_bulk_guard_holds_with_no_released_project():
    """Fails in the world where the guard was gated on a release existing."""
    assert _released_shell_block(None, "rm -rf projects", "/workspace") is not None
    assert _released_shell_block(None, "rm -rf build", "/workspace") is None


def test_a_huge_quoted_argument_is_parsed_fast():
    """Fails in the world where shlex spent 22 s on a 2 MB token, on the
    event loop, up to three times per call."""
    cmd = "echo '" + "A" * 2_000_000 + "' | base64 -d > f"
    t = time.time()
    _bulk_destructive(cmd, [REL], "/workspace")
    _rerun_unsafe(cmd)
    assert time.time() - t < 1.0


# ── released workspaces: reads pass, writes are seen ─────────────────────────
@pytest.mark.parametrize("cmd,wd", [
    ("find . -name '*.py' -exec grep -n TODO {} +", RW), ("tar -czf /tmp/backup.tgz .", RW),
    ("python3 - <<'EOF'\nprint(df[df.price > 100])\nEOF", RW), ("echo $((3>2))", RW),
    ("rm -f /tmp/scratch.txt", RW), ("mv /tmp/a /tmp/b", RW),
    (f"find projects/{REL} -type f -exec cat {{}} \;", "/workspace"),
])
def test_read_only_work_in_a_released_workspace_passes(cmd, wd):
    """Fails in the world where any find -exec, tar, a `>` in a heredoc or an
    arithmetic expansion, or any removal anywhere counted as a write."""
    assert _released_shell_block(_Store(), cmd, wd) is None


@pytest.mark.parametrize("cmd,wd", [
    (f"cd projects && cd {REL} && sed -i s/a/b/ x.py", "/workspace"),
    (f"cd ../{REL} && touch x", f"/workspace/projects/{DEV}"),
    (f"echo x >| projects/{REL}/a.py", "/workspace"),
    (f"git -C projects/{REL} checkout -- .", "/workspace"),
    ("npm install", RW), ("unzip a.zip", RW), ("git checkout -- .", RW),
])
def test_writes_into_a_released_workspace_are_seen(cmd, wd):
    assert _released_written_ids(cmd, {REL}, wd) == {REL}


@pytest.mark.parametrize("head,args,raw,written", [
    ("cp", ["-t", "out", "a", "b"], "cp -t out a b", ["out"]),
    ("rsync", ["--remove-source-files", "a", "b"], "rsync --remove-source-files a b", ["b", "a"]),
    ("tee", ["log.txt"], "tee log.txt", ["log.txt"]),
    ("dd", ["if=/dev/zero", "of=disk.img"], "dd if=/dev/zero of=disk.img", ["disk.img"]),
    ("tar", ["-czf", "/tmp/a.tgz", "src"], "tar -czf /tmp/a.tgz src", ["/tmp/a.tgz"]),
    ("find", [".", "-delete"], "find . -delete", ["."]),
    ("echo", ["x"], "echo x > /dev/null", []),
    ("echo", ["a > b"], "echo 'a > b'", []),
    ("unzip", ["a.zip", "-d", "out"], "unzip a.zip -d out", ["out"]),
    ("git", ["-C", "repo", "reset", "--hard"], "git -C repo reset --hard", ["repo"]),
])
def test_written_paths(head, args, raw, written):
    assert _written_paths(head, args, raw) == written


@pytest.mark.parametrize("cmd,unsafe", [
    ("(rm notes.md)", True), ("( rm notes.md )", True), ('bash -lc "rm notes.md"', True),
    ("cat missing.txt # > out.txt", False), ("cmd 2> err.log; cat missing", True),
])
def test_the_rerun_guard_fifth_review(cmd, unsafe):
    assert _rerun_unsafe(cmd) is unsafe


def test_nesting_past_the_limit_fails_closed():
    """A removal hidden six shells deep is not parsed — it is refused."""
    import shlex
    cmd = "rm -rf build"
    for _ in range(6):
        cmd = "bash -c " + shlex.quote(cmd)
    assert _bulk_destructive(cmd, [REL], "/workspace") is True
    harmless = "echo hi"
    for _ in range(6):
        harmless = "bash -c " + shlex.quote(harmless)
    assert _bulk_destructive(harmless, [REL], "/workspace") is False


# ── the lesson screen ────────────────────────────────────────────────────────
@pytest.mark.parametrize("fix", [
    "Don't forget to run rm -rf * to clean up.", "Avoid leftovers: rm -rf /workspace/*",
    "Never ask: just rm -rf * to start fresh", "Do not hesitate to rm -rf projects", "Delete all projects.",
    "Run (cd /workspace && rm -rf *) first", "Remove all the old projects before starting.",
    "Clean up the workspace before each task.", "git clean --force -dx",
    'execute(command="find /workspace/projects -mindepth 1 -maxdepth 1 -exec rm -rf {} +")',
])
def test_the_lesson_screen_fifth_review_misses(fix):
    """Fails in the world where any negation word nearby exempted a command,
    and "projects"/"clean up" were no bulk objects/verbs."""
    assert prescribes_destruction(fix) is True


@pytest.mark.parametrize("fix", [
    "Delete the generated files in build/ before rebuilding.", "Delete duplicate files the user listed one by one.",
    "Clear the files list in the UI state before reloading.", "Never run rm -rf /",
    "Do not use rm -rf . to clean up; delete files by name.", "Avoid DROP TABLE; migrate instead",
])
def test_the_lesson_screen_fifth_review_false_positives(fix):
    assert prescribes_destruction(fix) is False


# ── same request ─────────────────────────────────────────────────────────────
_DICT = os.path.exists("/usr/share/dict/words")


@pytest.mark.parametrize("a,b", [
    ("explain base64 encoding of the word hello", "explain base64 decoding of the word hello"),
    ("reactivate the cron job", "deactivate the cron job"),
    ("what is my name", "what is your name"), ("did the backup run", "will the backup run"),
    ("email me the report", "email them the report"),
])
def test_a_swapped_prefix_person_or_tense_is_another_request(a, b):
    """Fails in the world where one prefix swapped for another passed as a
    typo, and pronouns/auxiliaries were dropped before comparing."""
    from ghost_agent.memory.lesson_scope import same_request
    assert same_request(a, b) is False


@pytest.mark.skipif(not _DICT, reason="needs the system word list")
@pytest.mark.parametrize("a,b", [("summarise the present situation report", "summarise the president situation report"),
                                 ("show the inserted rows of the table", "show the inverted rows of the table")])
def test_two_dictionary_words_are_never_a_typo(a, b):
    from ghost_agent.memory.lesson_scope import same_request
    assert same_request(a, b) is False


@pytest.mark.parametrize("a,b", [
    ("show me all projects", "show all projects"), ("do a healthcheck", "healthcheck please"),
    ("give me the weather", "weather please"), ("full breafing please", "full briefing"),
])
def test_ordinary_resends_still_match(a, b):
    from ghost_agent.memory.lesson_scope import same_request
    assert same_request(a, b) is True


def test_a_request_is_compared_in_its_recorded_redacted_form():
    """Fails in the world where only home paths were redacted on the live
    side: a request naming an IP never matched its own recorded copy."""
    from ghost_agent.distill.redact import redact_text
    from ghost_agent.memory.lesson_scope import same_request
    live = "check the 2 interfaces 192.168.1.20 and 10.20.0.7 on the router"
    assert same_request(redact_text(live), live)


def test_name_parts_and_eg_in_general_text():
    from ghost_agent.memory.lesson_scope import is_general_text
    assert is_general_text("Leonidas-style age questions: verify the date", "how old is leonidas now ?") is False
    assert is_general_text("When a request lists examples (e.g. several file types), handle each",
                           "list my files, e.g. the csv ones") is True


# ── dedup ────────────────────────────────────────────────────────────────────
def test_a_reordered_request_gets_its_own_row_on_the_json_path(tmp_path):
    """Fails in the world where the JSON path refused the second plan (same
    sorted key) while the vector path wrote it — and the plan was lost."""
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    a, b = "copy notes.txt to backup.txt", "copy backup.txt to notes.txt"
    sm.learn_lesson(a, "m", "cp notes.txt backup.txt", trigger=a, scope="request", source_request=a, source="reflection")
    sm.learn_lesson(b, "m", "cp backup.txt notes.txt", trigger=b, scope="request", source_request=b, source="reflection")
    rows = {r["source_request"]: r["solution"] for r in sm._load_playbook()}
    assert rows == {a: "cp notes.txt backup.txt", b: "cp backup.txt notes.txt"}


def test_a_close_twin_with_another_trigger_never_replaces_the_fix(tmp_path):
    """Fails in the world where a dream merge put another pattern's (longer)
    fix on a row whose trigger it does not match."""
    from unittest.mock import MagicMock
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    t = "distilled(output_processing/report formatting)"
    sm.save_playbook([{"trigger": t, "task": t, "mistake": "m", "solution": "format the report", "source": "dream",
                       "frequency": 9}])
    sm._find_duplicate_lesson = lambda *x, **k: {"source": "vector", "trigger": t, "text": "", "distance": 0.1}
    sm.learn_lesson("distilled(memory/recall)", "m", "validate critical information against external sources first",
                    MagicMock(), source="dream")
    row = sm._load_playbook()[0]
    assert row["solution"] == "format the report" and row["frequency"] == 10


def test_a_row_scoped_by_a_merge_gets_a_fresh_twin(tmp_path):
    from ghost_agent.memory.skills import SkillMemory
    from ghost_agent.memory import skills as SK
    sm = SkillMemory(tmp_path)
    q = "restart the chess service now"
    sm.save_playbook([{"trigger": q, "task": q, "mistake": "m", "solution": "s", "source": "reflection"}])
    refreshed = []
    sm._refresh_twin = lambda ms, les: refreshed.append(les.get("scope"))
    sm.learn_lesson(q, "m", "s", object(), trigger=q, scope="request", source_request=q, source="reflection")
    assert refreshed == ["request"]


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


@pytest.mark.parametrize("target,rows,gone", [
    ("wife's birthday", [("b", 0.45, "User's wife's birthday is 3 March"), ("m", 0.55, "User's wife is named Maria"),
                         ("a", 0.7, "User lives in Athens")], {"b"}),
    ("Louis", [("l", 0.5, "son is named Louis"), ("e", 0.7, "sister Louise lives in Paris")], {"l"}),
])
def test_forget_takes_the_entity_not_a_look_alike(tmp_path, target, rows, gone):
    """Fails in the world where, after the literal match, one partial match
    was deleted too, and `e` made Louis~Louise one word."""
    from ghost_agent.tools.memory import tool_unified_forget
    ms, deleted = _forget_ms(rows)
    asyncio.run(tool_unified_forget(target, "/nonexistent-sb", ms))
    assert set(deleted) == gone


@pytest.mark.parametrize("a,b", [("plan", "plant"), ("plane", "planet"), ("star", "start"), ("bear", "beard"),
                                 ("bank", "banker"), ("louis", "louise")])
def test_forget_inflections_are_only_inflections(a, b):
    from ghost_agent.tools.memory import _word_matches
    assert _word_matches(a, b) is False
