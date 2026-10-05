"""§4LL (2026-10-04): file handling — the file_system tool, the labels its
results get, and what file failures teach. Each test names the world it
fails in."""
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.test_critic_async import agent, _final  # noqa: F401 — fixture


def _fs(**kw):
    from ghost_agent.tools.file_system import tool_file_system
    return asyncio.run(tool_file_system(**kw))


# ── calls on one file in one message run in order ─────────────────────────────
@pytest.mark.asyncio
async def test_file_calls_in_one_message_run_one_at_a_time_in_order(agent):
    """Fails where one gather ran three edits to the same file at once (one
    landed), and where grouping by path STRING let aliases of one file
    ("app.py", "/workspace/app.py", "/app.py") race."""
    spans = []

    async def fs(**kw):
        t0 = time.monotonic()
        await asyncio.sleep(0.08)
        spans.append((kw.get("new_string"), t0, time.monotonic()))
        return "SUCCESS: edited"
    agent.available_tools["file_system"] = fs
    paths = ["app.py", "/workspace/app.py", "/app.py", "other.py"]
    calls = [{"id": f"t{i}", "function": {"name": "file_system", "arguments": json.dumps(
        {"operation": "edit", "path": p, "old_string": f"a{i}", "new_string": f"b{i}"})}}
        for i, p in enumerate(paths)]
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": calls}}]}, _final("Done.")] + [_final("ok")] * 4)
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat({"messages": [{"role": "user", "content": "edit them"}]},
                                background_tasks=MagicMock())
    assert [n for n, _, _ in spans] == ["b0", "b1", "b2", "b3"]
    assert all(spans[i][2] <= spans[i + 1][1] + 1e-3 for i in range(3))


@pytest.mark.asyncio
async def test_a_read_after_a_write_in_one_message_runs_again(agent):
    """Fails where the batch dedup copied the FIRST read's result into the
    read after the write (stale content, or "does not exist")."""
    ran = []

    async def fs(**kw):
        ran.append(kw.get("operation"))
        return "SUCCESS: Wrote 3 chars to 'x.txt'." if kw.get("operation") == "write" else "--- x.txt CONTENTS ---\nv"
    agent.available_tools["file_system"] = fs
    rd = {"operation": "read", "path": "x.txt"}
    calls = [{"id": "r1", "function": {"name": "file_system", "arguments": json.dumps(rd)}},
             {"id": "w1", "function": {"name": "file_system", "arguments": json.dumps(
                 {"operation": "write", "path": "x.txt", "content": "new"})}},
             {"id": "r2", "function": {"name": "file_system", "arguments": json.dumps(rd)}}]
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": calls}}]}, _final("Done.")] + [_final("ok")] * 4)
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat({"messages": [{"role": "user", "content": "update x"}]},
                                background_tasks=MagicMock())
    assert ran == ["read", "write", "read"]


# ── what a successful file result is labelled ─────────────────────────────────
@pytest.mark.parametrize("text", ["--- app.log CONTENTS ---\nERROR: disk full at 03:12",
                                  "class ValidationException(Exception):\n    pass",
                                  "src/app.py:12:    except Exception as e:",
                                  "notes.md mentions a syntax error we fixed"])
def test_a_read_whose_content_mentions_errors_is_not_a_failed_call(text):
    """Fails where the shared sniffer read "exception"/"error:" in a file's
    CONTENT as the tool failing (claim binding withheld a right summary)."""
    from ghost_agent.distill.outcome_heuristics import looks_like_tool_error
    assert looks_like_tool_error(text, "file_system") is False
    assert looks_like_tool_error("Error: 'x.py' not found.", "file_system") is True
    assert looks_like_tool_error("Security Error: Path '../../x' attempts to access outside sandbox.",
                                 "file_system") is True


@pytest.mark.parametrize("head,flagged", [
    ("SUCCESS: Copied 'edited_photo.jpg' to 'b.jpg'.", False),
    ("SUCCESS: Renamed 'draft_written.md' to 'final.md'.", False),
    ("SUCCESS: Wrote 4000 chars to '.gitignore'.", False),
    ("SUCCESS: Wrote 4000 chars to 'poetry.lock'.", False),
])
def test_a_filename_is_not_a_mutation_word_and_dotfiles_are_inert(head, flagged):
    """Fails where "edited_photo.jpg" or a 2 KB .gitignore booked a correct
    turn failed with "⚠ Unverified: never run"."""
    from ghost_agent.core.agent import _is_unverified_mutation
    assert _is_unverified_mutation({"name": "file_system", "content": head}) is flagged


# ── host paths ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("target", ["~/Desktop/notes.txt", "/Users/someone/Desktop/notes.txt", "/tmp/out.txt",
                                    "/home/u/x.txt"])
def test_a_write_to_the_users_machine_is_refused_not_silently_sandboxed(tmp_path, target):
    """Fails where "write ~/Desktop/notes.txt" landed in <sandbox>/Users/…
    and was reported written to the Desktop."""
    out = _fs(operation="write", sandbox_dir=tmp_path, path=target, content="hello")
    assert "outside your sandbox" in str(out) and not list(tmp_path.rglob("notes.txt"))


def test_writes_inside_the_sandbox_still_work(tmp_path):
    assert "SUCCESS" in str(_fs(operation="write", sandbox_dir=tmp_path, path="/workspace/a.txt", content="x"))
    assert "SUCCESS" in str(_fs(operation="write", sandbox_dir=tmp_path, path=str(tmp_path / "b.txt"),
                                content="y"))
    assert (tmp_path / "a.txt").read_text() == "x" and (tmp_path / "b.txt").read_text() == "y"


def test_reading_a_path_on_the_users_computer_says_ask_for_the_file():
    from ghost_agent.tools.file_system import outside_workspace_message
    m = outside_workspace_message("/Users/someone/Documents/cv.pdf")
    assert "user's own computer" in m and "cat" not in m
    assert "cat" in outside_workspace_message("/usr/lib/python3/x.py")       # container paths keep execute


# ── search ───────────────────────────────────────────────────────────────────
def _sm(*outputs):
    sm = MagicMock()
    sm.execute = MagicMock(side_effect=list(outputs))
    return sm


def test_a_pattern_that_is_not_a_regex_is_searched_as_text(tmp_path):
    """Fails where `print(` came back as a regex parse error booked as a
    successful search."""
    sm = _sm(("regex parse error: unclosed group", 2), ("a.py:3:print(x)", 0))
    out = str(_fs(operation="search", sandbox_dir=tmp_path, pattern="print(", sandbox_manager=sm))
    assert "a.py:3:print(x)" in out and "plain text" in out
    assert " -F " in sm.execute.call_args_list[1].args[0]
    cmd = sm.execute.call_args_list[0].args[0]
    assert "--hidden" in cmd and "--no-ignore-vcs" not in cmd          # whole workspace: .gitignore respected


def test_a_named_path_is_searched_even_if_git_ignored(tmp_path):
    (tmp_path / "dist").mkdir()
    sm = _sm(("dist/a.js:1:x", 0))
    _fs(operation="search", sandbox_dir=tmp_path, pattern="x", path="dist", sandbox_manager=sm)
    assert "--no-ignore-vcs" in sm.execute.call_args.args[0]


def test_matches_survive_one_unreadable_file(tmp_path):
    """Fails where rg's exit 2 (any file errored) threw the matches away."""
    out = str(_fs(operation="search", sandbox_dir=tmp_path, pattern="x",
                  sandbox_manager=_sm(("a.py:3:x = 1\nrg: b.py: Permission denied", 2))))
    assert "a.py:3:x = 1" in out and not out.startswith("Error")


def test_a_real_search_error_is_an_error(tmp_path):
    out = _fs(operation="search", sandbox_dir=tmp_path, pattern="x", sandbox_manager=_sm(("rg: bad flag", 2)))
    assert str(out).startswith("Error") and getattr(out, "status", None).value == "failed"


# ── replace / edit ───────────────────────────────────────────────────────────
def test_an_empty_replacement_deletes_the_text(tmp_path):
    """Fails where new_string="" read as missing and the reply offered to
    overwrite the whole file with the fragment."""
    (tmp_path / "f.txt").write_text("keep REMOVE me\n")
    out = _fs(operation="replace", sandbox_dir=tmp_path, path="f.txt", content="REMOVE ", new_string="")
    assert "<<HELD>>" not in str(out) and (tmp_path / "f.txt").read_text() == "keep me\n"


def test_an_old_string_copied_with_line_numbers_is_named(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\nb = 2\n")
    out = str(_fs(operation="edit", sandbox_dir=tmp_path, path="f.py", old_string="1\ta = 1\n2\tb = 2",
                  new_string="a = 3"))
    assert "copied it from a ranged read" in out and (tmp_path / "f.py").read_text() == "a = 1\nb = 2\n"


def test_a_multi_line_insert_into_a_crlf_file_stays_crlf(tmp_path):
    (tmp_path / "w.bat").write_bytes(b"echo a\r\necho b\r\n")
    _fs(operation="edit", sandbox_dir=tmp_path, path="w.bat", old_string="echo a", new_string="echo a\necho a2")
    data = (tmp_path / "w.bat").read_bytes()
    assert b"echo a\r\necho a2\r\necho b\r\n" == data


def test_the_rollback_advice_names_edit_not_search_replace(tmp_path):
    """Fails where a rolled-back syntax error told the model to "emit a TIGHT
    single-line SEARCH/REPLACE" — which 'edit' refuses."""
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    out = str(_fs(operation="replace", sandbox_dir=tmp_path, path="m.py",
                  content="    return 1", replace_with="  return (1"))
    assert "NOT applied" in out and "operation='edit'" in out and "SEARCH/REPLACE" not in out


# ── small error messages ─────────────────────────────────────────────────────
def test_an_unknown_tool_says_what_to_use(agent):
    from ghost_agent.core.agent import _unknown_tool_message
    assert "execute(command=" in _unknown_tool_message("git", {"execute": 1})
    m = _unknown_tool_message("web_extractor", {"browser": 1, "web_search": 1})
    assert "browser" in m and "web_search" in m


@pytest.mark.parametrize("code,word", [(429, "rate-limiting"), (403, "refused"), (404, "not at that address")])
def test_a_failed_download_says_why_and_not_to_retry(code, word):
    from ghost_agent.tools.file_system import _download_error
    m = _download_error(code, "https://x.org/a.png")
    assert word in m and "Do not retry" in m and "Nothing was saved" in m


def test_a_file_named_http_something_is_read(tmp_path):
    (tmp_path / "http_log.txt").write_text("GET / 200\n")
    assert "GET / 200" in str(_fs(operation="read", sandbox_dir=tmp_path, path="http_log.txt"))


def test_operations_are_case_insensitive_and_the_unknown_list_is_the_schemas(tmp_path):
    from ghost_agent.tools.file_system import ADVERTISED_OPS
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    fs = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "file_system")
    assert tuple(fs["function"]["parameters"]["properties"]["operation"]["enum"]) == ADVERTISED_OPS
    (tmp_path / "a.txt").write_text("hi")
    assert "hi" in str(_fs(operation="Read", sandbox_dir=tmp_path, path="a.txt"))
    out = str(_fs(operation="frobnicate", sandbox_dir=tmp_path, path="a.txt"))
    assert "edit" in out and "batch" not in out


def test_reading_a_directory_points_to_list_files(tmp_path):
    (tmp_path / "docs").mkdir()
    assert "list_files" in str(_fs(operation="read", sandbox_dir=tmp_path, path="docs"))


@pytest.mark.parametrize("name", ["__init__.py", ".gitkeep"])
def test_an_empty_marker_file_can_be_written(tmp_path, name):
    out = _fs(operation="write", sandbox_dir=tmp_path, path=f"pkg/{name}", content="")
    assert "SUCCESS" in str(out) and (tmp_path / "pkg" / name).exists()
    assert "MANDATORY" in str(_fs(operation="write", sandbox_dir=tmp_path, path="app.py", content=""))


# ── what file failures teach ─────────────────────────────────────────────────
@pytest.mark.parametrize("fix,refused", [
    ("Call `git(operation=write, path=\"config\")` to set the remote.", True),
    ("Use file_system(operation=\"batch\") to change several files.", True),
    ("Use file_system(operation='edit', old_string=…, new_string=…) for a part of a file.", False),
])
def test_a_lesson_that_prescribes_a_tool_that_does_not_exist_is_refused(tmp_path, fix, refused):
    """Fails where the reflection on the §4KB checks saved "call git(…)" after
    the git tool was removed — nothing compared lessons with the registry."""
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    sm.learn_lesson("When configuring the repository remote", "it used the wrong tool for the job", fix, None,
                    trigger="When configuring the repository remote", source="reflection", origin="user")
    rows = json.loads(sm.file_path.read_text()) if sm.file_path.exists() else []
    assert (len(rows) == 0) is refused



def test_a_non_empty_replacement_wins_over_an_empty_alias(tmp_path):
    (tmp_path / "f.py").write_text("X = 1\nY = 2\n")
    _fs(operation="replace", sandbox_dir=tmp_path, path="f.py", content="Y = 2", replace_with="", new_string="Y = 42")
    assert (tmp_path / "f.py").read_text() == "X = 1\nY = 42\n"


def test_the_line_ending_follows_the_edited_region(tmp_path):
    (tmp_path / "m.txt").write_bytes(b"c1\r\nc2\r\nc3\r\nn1\nn2\n")
    _fs(operation="edit", sandbox_dir=tmp_path, path="m.txt", old_string="n1", new_string="n1\nn1b")
    assert (tmp_path / "m.txt").read_bytes() == b"c1\r\nc2\r\nc3\r\nn1\nn1b\nn2\n"


def test_the_sandboxs_own_path_in_another_case_is_not_the_users_machine(tmp_path):
    from ghost_agent.tools.file_system import _host_path_block
    import sys
    if sys.platform != "darwin":
        pytest.skip("APFS case-insensitivity")
    p = str(tmp_path.resolve() / "a.txt")
    assert _host_path_block(tmp_path, p.upper()) is None


@pytest.mark.parametrize("text", [
    "Check the page with vision_analysis(action='verify_ui', image='shot.png').",
    "Do NOT call git(operation='commit') — run git through execute.",
    "In the page script, page.evaluate(action='scroll') runs in the browser.",
    "Use file_system(operation='list') to see the folder.",
])
def test_lessons_that_call_real_tools_or_warn_are_kept(text):
    from ghost_agent.memory.lesson_quality import unknown_tool_calls
    assert unknown_tool_calls(text) == []
