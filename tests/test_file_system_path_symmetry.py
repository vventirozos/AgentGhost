"""Regression tests for _get_safe_path / tool_write_file path symmetry.

The 2026-04-19 trace showed a real bug: the agent wrote
'sandbox/test_sample.log' via the file_system tool, the tool stripped
the 'sandbox/' prefix, and the file landed at <sandbox>/test_sample.log
(== container /workspace/test_sample.log). A Python script executing
in the sandbox container then did open("sandbox/test_sample.log"),
which resolved to /workspace/sandbox/test_sample.log and crashed.

These tests pin the new invariant: whatever path the agent writes to
is the path a script can read from.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pathlib import Path

import pytest

from ghost_agent.tools.file_system import (
    _get_safe_path, tool_write_file, tool_read_file,
)


def test_sandbox_prefix_is_honored_literally(tmp_path):
    """Previously stripped; now kept. `sandbox/foo.txt` writes to
    `<sandbox>/sandbox/foo.txt` so container-side `open("sandbox/foo.txt")`
    works from cwd=/workspace."""
    p = _get_safe_path(tmp_path, "sandbox/foo.txt")
    assert p == (tmp_path / "sandbox" / "foo.txt").resolve()


def test_leading_slash_still_flattened(tmp_path):
    """Root-anchored style still resolves to sandbox root — this is
    the one heal we kept (otherwise the security check rejects paths
    outside the sandbox)."""
    p = _get_safe_path(tmp_path, "/foo.txt")
    assert p == (tmp_path / "foo.txt").resolve()


def test_plain_filename_at_root(tmp_path):
    p = _get_safe_path(tmp_path, "foo.txt")
    assert p == (tmp_path / "foo.txt").resolve()


def test_nested_paths_preserved(tmp_path):
    p = _get_safe_path(tmp_path, "a/b/c/foo.txt")
    assert p == (tmp_path / "a" / "b" / "c" / "foo.txt").resolve()


def test_traversal_still_blocked(tmp_path):
    with pytest.raises(ValueError):
        _get_safe_path(tmp_path, "../escape.txt")


def test_traversal_via_sandbox_prefix_also_blocked(tmp_path):
    with pytest.raises(ValueError):
        _get_safe_path(tmp_path, "sandbox/../../escape.txt")


async def test_write_then_read_is_symmetric_with_sandbox_prefix(tmp_path):
    """The exact trace regression: agent writes to 'sandbox/foo.txt'
    via the tool; a Python script running in the sandbox cwd must find
    the file at that same relative path."""
    res = await tool_write_file("sandbox/foo.txt", "hello", tmp_path)
    assert "SUCCESS" in res
    # The write response surfaces the container-side path so the model
    # can reference it consistently in scripts.
    assert "sandbox/foo.txt" in res
    # The file landed at <sandbox>/sandbox/foo.txt — this is what a
    # container-side `open("sandbox/foo.txt")` from cwd=/workspace
    # (bind-mounted to tmp_path) will see.
    assert (tmp_path / "sandbox" / "foo.txt").read_text() == "hello"
    # And the file_system tool read path sees it too
    read_back = await tool_read_file("sandbox/foo.txt", tmp_path)
    assert "hello" in read_back


async def test_write_then_read_plain_filename(tmp_path):
    res = await tool_write_file("foo.txt", "hi", tmp_path)
    assert "SUCCESS" in res
    assert (tmp_path / "foo.txt").read_text() == "hi"
    read_back = await tool_read_file("foo.txt", tmp_path)
    assert "hi" in read_back


async def test_write_response_reports_sandbox_relative_path(tmp_path):
    """The response text must tell the model the container-side path
    so it can pass that exact string to Python's open() in a script."""
    res = await tool_write_file("sandbox/deep/nested/data.json", "{}", tmp_path)
    assert "Script-side path" in res
    assert "sandbox/deep/nested/data.json" in res


# --------------------------------------------------------------------- fixture-count summary

async def test_log_fixture_includes_count(tmp_path):
    """A .log fixture must report its non-empty line count so the next
    turn can cite the number instead of guessing — this is the
    countermeasure to the 2026-04-19 trace where the model wrote
    `assert count == 4` against a 5-line fixture."""
    content = "\n".join([f"line {i}" for i in range(7)])
    res = await tool_write_file("access.log", content, tmp_path)
    assert "FIXTURE-COUNT" in res
    assert "7 non-empty lines" in res


async def test_csv_fixture_includes_count(tmp_path):
    res = await tool_write_file(
        "data.csv", "a,b,c\n1,2,3\n4,5,6\n", tmp_path,
    )
    assert "3 non-empty lines" in res


async def test_jsonl_fixture_includes_record_count(tmp_path):
    res = await tool_write_file(
        "events.jsonl",
        '{"a":1}\n{"a":2}\n{"a":3}\n{"a":4}',
        tmp_path,
    )
    assert "4 JSON records" in res


async def test_python_source_does_not_get_fixture_summary(tmp_path):
    """Source files (.py/.html/.css/.js) shouldn't get count noise."""
    res = await tool_write_file("foo.py", "x = 1\ny = 2\n", tmp_path)
    assert "FIXTURE-COUNT" not in res


async def test_blank_lines_excluded_from_count(tmp_path):
    res = await tool_write_file(
        "x.txt", "a\n\n\nb\n\nc\n", tmp_path,
    )
    assert "3 non-empty lines" in res


def test_system_prompt_carries_test_discipline_directive():
    """If a future refactor drops the directive, the agent regresses
    to writing fixture counts from memory. Pin the wording."""
    from ghost_agent.core.prompts import SYSTEM_PROMPT
    assert "TEST DISCIPLINE" in SYSTEM_PROMPT
    assert "FIXTURE-COUNT" in SYSTEM_PROMPT


# --------------------------------------------------------------------- replace failure feedback

from ghost_agent.tools.file_system import tool_replace_text, _nearest_snippet


def test_nearest_snippet_finds_best_overlap():
    """Verifies the helper picks the line with the most token overlap."""
    content = (
        "def greet(name):\n"
        "    print('hello', name)\n"
        "\n"
        "def farewell(user):\n"
        "    print('bye', user)\n"
    )
    snippet = _nearest_snippet(content, "farewell user print bye")
    # The best-match line should be marked with >>>
    assert ">>>" in snippet
    assert "farewell" in snippet


def test_nearest_snippet_handles_empty_content():
    assert _nearest_snippet("", "whatever") == "(file is empty)"


def test_nearest_snippet_respects_max_len():
    big = "\n".join(f"line {i} with tokens" for i in range(1000))
    out = _nearest_snippet(big, "tokens", max_len=200)
    assert len(out) <= 250  # max_len + some room for truncation marker


async def test_replace_failure_returns_nearest_snippet(tmp_path):
    """The trace regression: model retries replace 4 times because the
    error just said 'copy exactly'. Now the error surfaces the actual
    file content around the closest match.

    The search block must be far enough from every line that the
    fuzzy-block matcher (ratio >= 0.92) does NOT claim it — otherwise a
    near-miss is auto-resolved instead (that path is covered by
    test_file_replace_fuzzy_and_size_guidance). Here the search shares
    tokens with the `count` line so _nearest_snippet anchors there and
    surfaces it, but is structurally too different to fuzzy-match."""
    src = (
        "def test_parser():\n"
        "    if results['/health']['count'] != 11:\n"
        "        print('wrong count')\n"
    )
    (tmp_path / "test_parser.py").write_text(src)
    # Model remembers a wholly different shape of the assertion than what
    # is on disk — token overlap, but no near-match for fuzzy to claim.
    res = await tool_replace_text(
        "test_parser.py",
        "if results['/health']['count'] > 999 and cache['/health']['count'] < 0:",
        "if results['/health']['count'] != 11:",
        tmp_path,
    )
    assert "NOT found" in res
    assert "CLOSEST MATCH" in res
    assert "!= 11" in res  # the actual text is surfaced
    # The directive tells the model not to retry blindly
    assert "do NOT retry" in res or "READ the file" in res


async def test_replace_failure_suggests_alternatives(tmp_path):
    (tmp_path / "x.py").write_text("foo = 1\n")
    res = await tool_replace_text(
        "x.py", "nonexistent block", "whatever", tmp_path,
    )
    assert "READ the file" in res or "operation='write'" in res


async def test_replace_exact_match_still_works(tmp_path):
    """Don't regress the happy path."""
    (tmp_path / "x.py").write_text("foo = 1\nbar = 2\n")
    res = await tool_replace_text(
        "x.py", "foo = 1", "foo = 99", tmp_path,
    )
    assert "SUCCESS" in res
    assert (tmp_path / "x.py").read_text() == "foo = 99\nbar = 2\n"


async def test_replace_flexible_whitespace_still_works(tmp_path):
    """Whitespace-insensitive match still wins before we fall to
    the neighborhood-snippet error. The flexible matcher splits on
    whitespace only, so tokens must align — it collapses internal
    whitespace variations between tokens."""
    (tmp_path / "x.py").write_text("if  foo   ==   1:\n    pass\n")
    res = await tool_replace_text(
        "x.py", "if foo == 1:", "if foo == 2:", tmp_path,
    )
    assert "SUCCESS" in res


# ---------------------------------------------------------------------------
# /workspace/ prefix handling
#
# The sandbox bind-mounts the host sandbox root at /workspace inside the
# container. So /workspace/foo.py (the path the LLM sees in every shell
# command it emits) is the same file as foo.py at the sandbox root on
# the host. Before the fix, _get_safe_path preserved the workspace/
# segment literally, producing a phantom workspace/ directory on disk
# and breaking write-then-read symmetry (the LLM wrote to
# /workspace/skills/x.py, then `python3 /workspace/skills/x.py` in the
# container hit ENOENT because the real host file was at
# <sandbox>/workspace/skills/x.py → container /workspace/workspace/
# skills/x.py). Confirmed in the 2026-04-24 in_gr_news session.
# ---------------------------------------------------------------------------


def test_workspace_prefix_is_stripped(tmp_path):
    """/workspace/foo.py must resolve to <sandbox>/foo.py so the same
    path works from the container (cwd=/workspace)."""
    p = _get_safe_path(tmp_path, "/workspace/foo.py")
    assert p == (tmp_path / "foo.py").resolve()


def test_bare_workspace_prefix_is_stripped(tmp_path):
    """The LLM sometimes drops the leading slash. Same intent, same
    strip."""
    p = _get_safe_path(tmp_path, "workspace/skills/in_gr_news.py")
    assert p == (tmp_path / "skills" / "in_gr_news.py").resolve()


def test_workspace_alone_is_sandbox_root(tmp_path):
    """`/workspace` with no trailing segment resolves to the sandbox
    root itself. Useful when the LLM queries the root dir via the
    file_system list op."""
    p = _get_safe_path(tmp_path, "/workspace")
    assert p == tmp_path.resolve()


def test_workspace_prefix_is_exact_segment_only(tmp_path):
    """`workspaces/` (plural) or `workspace_backup/` (different name)
    must NOT be stripped — it would be a real subdir the LLM created
    deliberately."""
    p1 = _get_safe_path(tmp_path, "/workspaces/foo.py")
    assert p1 == (tmp_path / "workspaces" / "foo.py").resolve()
    p2 = _get_safe_path(tmp_path, "workspace_backup/x.py")
    assert p2 == (tmp_path / "workspace_backup" / "x.py").resolve()


def test_workspace_prefix_double_slashes_collapse(tmp_path):
    """Multiple leading slashes (LLM typo / OS-style path) still
    collapse to the same sandbox-root-relative path."""
    p = _get_safe_path(tmp_path, "//workspace/foo.py")
    assert p == (tmp_path / "foo.py").resolve()


def test_workspace_prefix_traversal_still_blocked(tmp_path):
    """Traversal inside a workspace-prefixed path is still blocked."""
    with pytest.raises(ValueError):
        _get_safe_path(tmp_path, "/workspace/../escape.txt")


async def test_workspace_write_then_read_is_symmetric(tmp_path):
    """The incident regression: LLM writes to /workspace/skills/x.py;
    a script in the container then runs `python3 /workspace/skills/x.py`
    and must find the same file. The host file must live at
    <sandbox>/skills/x.py (NOT <sandbox>/workspace/skills/x.py)."""
    res = await tool_write_file("/workspace/skills/in_gr_news.py", "print('ok')\n", tmp_path)
    assert "SUCCESS" in res
    # Host-side: the skills/ dir lives at sandbox root, not nested
    # under a phantom workspace/.
    assert (tmp_path / "skills" / "in_gr_news.py").exists()
    assert not (tmp_path / "workspace").exists()

    # Reading back with the same /workspace/... path must work.
    read_res = await tool_read_file("/workspace/skills/in_gr_news.py", tmp_path)
    assert "print('ok')" in read_res
