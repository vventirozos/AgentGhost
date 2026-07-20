"""2026-07-20 three-stack review — file_system "wrong content written,
SUCCESS reported" cohort (PROJECT_JOURNAL.md §4B).

Covered here:
  * H7  — streaming replace (>1 MB, single-line old_text) now runs the same
          marker-leak + syntax-regression guard chain as the normal path
          before committing os.replace.
  * MED — SEARCH/REPLACE envelope closer is line-anchored; a `>{4,}` run
          mid-payload (`// >>>> end of patch`) no longer truncates the
          replacement. The glued-closer tolerance survives as a last-resort
          rung reached only when NO line-anchored closer exists.
  * MED — git-conflict marker lines (`<<<<<<< HEAD` / `>>>>>>> ref`) count
          as markers, so a SEARCH text carrying a conflict block is rejected
          instead of mis-split at the conflict's own `=======`.
  * MED — indented markers (`    ====`) no longer evade _MARKER_LINE_RE.
  * MED — tool_write_file: lone-surrogate content returns `Error: ...`
          WITHOUT truncating the existing file (encode-first + tmp/rename).
  * MED — _get_safe_path: an explicit `projects/<other-id>/` path resolves
          to THAT project's dir instead of silently collapsing into the
          active project; the title-guess collapse (no backing dir) stays.
  * MED — tool_rename_file refuses to clobber an existing destination.
  * LOW — file search passes the rg pattern via `-e` so a `-`-leading
          pattern cannot parse as a flag.

C2 (auto-promote / fence strip missing filename=) lives in
test_file_replace_auto_promote.py.
"""
import json

import pytest

from ghost_agent.tools.file_system import (
    _get_safe_path,
    _marker_leak,
    tool_file_search,
    tool_rename_file,
    tool_replace_text,
    tool_write_file,
)


@pytest.fixture
def sandbox(tmp_path):
    d = tmp_path / "sandbox"
    d.mkdir()
    return d


def _big_body(marker_line: str, ext_comment: str = "# pad") -> str:
    """> 1 MB of single-line filler with one unique target line embedded,
    so the streaming (line-by-line) replace path is taken."""
    pad = (ext_comment + " " + "x" * 90 + "\n") * 12000  # ~1.15 MB
    return pad + marker_line + "\n" + pad


# ---------------------------------------------------------------------------
# H7 — streaming replace routed through the guard chain
# ---------------------------------------------------------------------------

async def test_streaming_replace_marker_leak_rejected(sandbox):
    # Merged-args corruption shape: replace_with carries literal envelope
    # markers. The streaming path used to write them verbatim with SUCCESS.
    target = sandbox / "big.txt"
    original = _big_body("UNIQUE_TOKEN_A = old")
    target.write_text(original)
    assert target.stat().st_size > 1024 * 1024

    res = await tool_replace_text(
        "big.txt", "UNIQUE_TOKEN_A = old",
        "UNIQUE_TOKEN_A = new\n<<<< SEARCH\nleak\n====\nmore\n>>>>",
        sandbox)
    assert "REJECTED" in res and "marker" in res
    assert target.read_text() == original
    assert not list(sandbox.glob("*.tmp")), "no orphan tmp files"


async def test_streaming_replace_syntax_regression_rejected(sandbox):
    target = sandbox / "big.py"
    original = _big_body("VALUE = 1")
    target.write_text(original)
    assert target.stat().st_size > 1024 * 1024

    res = await tool_replace_text("big.py", "VALUE = 1", "VALUE = = 1", sandbox)
    assert "REJECTED" in res and "syntax" in res.lower()
    assert target.read_text() == original
    assert not list(sandbox.glob("*.tmp"))


async def test_streaming_replace_clean_edit_still_works(sandbox):
    target = sandbox / "big.py"
    target.write_text(_big_body("VALUE = 1"))

    res = await tool_replace_text("big.py", "VALUE = 1", "VALUE = 2", sandbox)
    assert res.startswith("SUCCESS") and "Streaming replace" in res
    body = target.read_text()
    assert "VALUE = 2" in body and "VALUE = 1" not in body
    assert not list(sandbox.glob("*.tmp"))


# ---------------------------------------------------------------------------
# MED — line-anchored envelope closer
# ---------------------------------------------------------------------------

async def test_midline_closer_run_does_not_truncate_replacement(sandbox):
    # The exact misparse: a `>>>>` run inside a payload line ended the
    # envelope early, silently truncating the replacement.
    target = sandbox / "app.js"
    target.write_text("const a = 1;\nconst b = 2;\n")
    payload = (
        "<<<< SEARCH\n"
        "const a = 1;\n"
        "====\n"
        "const a = 10;\n"
        "// >>>> end of patch\n"
        ">>>>"
    )
    res = await tool_replace_text("app.js", payload, None, sandbox)
    assert res.startswith("SUCCESS"), res
    body = target.read_text()
    assert "const a = 10;" in body
    assert "// >>>> end of patch" in body, "payload after the >>>> run was dropped"


async def test_deletion_envelope_still_parses(sandbox):
    # `====` directly followed by `>>>>` = replace-with-nothing; the
    # anchored closer must not break it.
    target = sandbox / "notes.txt"
    target.write_text("keep me\ndrop me\nkeep too\n")
    payload = "<<<< SEARCH\ndrop me\n====\n>>>>"
    res = await tool_replace_text("notes.txt", payload, None, sandbox)
    assert res.startswith("SUCCESS"), res
    assert "drop me" not in target.read_text()
    assert "keep me" in target.read_text()


async def test_glued_closer_tolerance_still_parses(sandbox):
    # Last-resort rung: an envelope with NO line-anchored closer anywhere
    # (closer glued to content) still applies — same tolerance as before.
    target = sandbox / "t.py"
    target.write_text("def f():\n    print('hello')\n")
    payload = "<<<< SEARCH\ndef f():\n    print('hello')\n====\ndef f():\n    print('world')>>>>"
    res = await tool_replace_text("t.py", payload, None, sandbox)
    assert "SUCCESS" in res, res
    assert "print('world')" in target.read_text()


# ---------------------------------------------------------------------------
# MED — git-conflict content in SEARCH must not hijack the separator
# ---------------------------------------------------------------------------

_CONFLICTED = (
    "line before\n"
    "<<<<<<< HEAD\n"
    "ours()\n"
    "=======\n"
    "theirs()\n"
    ">>>>>>> feature-branch\n"
    "line after\n"
)


async def test_conflict_block_in_search_rejected_not_missplit(sandbox):
    # The conflict's own `=======` used to be read as the envelope
    # separator: SEARCH parsed as `<<<<<<< HEAD\nours()` (present in the
    # file!), so a mangled half-edit applied with SUCCESS.
    target = sandbox / "merged.py"
    target.write_text(_CONFLICTED)
    payload = (
        "<<<< SEARCH\n"
        "<<<<<<< HEAD\n"
        "ours()\n"
        "=======\n"
        "theirs()\n"
        ">>>>>>> feature-branch\n"
        "====\n"
        "resolved()\n"
        ">>>>"
    )
    res = await tool_replace_text("merged.py", payload, None, sandbox)
    assert "REJECTED" in res or "did not match" in res or "None of the" in res
    assert target.read_text() == _CONFLICTED, "file must be untouched"


async def test_conflict_resolution_via_two_arg_replace_still_works(sandbox):
    # The safe path for resolving a conflict: two-arg replace whose
    # old_text IS the conflict block. Marker REMOVAL is count-aware-legal.
    target = sandbox / "merged.py"
    target.write_text(_CONFLICTED)
    conflict = (
        "<<<<<<< HEAD\n"
        "ours()\n"
        "=======\n"
        "theirs()\n"
        ">>>>>>> feature-branch"
    )
    res = await tool_replace_text("merged.py", conflict, "resolved()", sandbox)
    assert res.startswith("SUCCESS"), res
    body = target.read_text()
    assert "resolved()" in body
    assert "<<<<<<<" not in body and "=======" not in body and ">>>>>>>" not in body


def test_marker_line_re_recognizes_git_conflict_lines():
    assert _marker_leak("a\n", "a\n<<<<<<< HEAD\n") != ""
    assert _marker_leak("a\n", "a\n>>>>>>> feature-branch\n") != ""
    assert _marker_leak("a\n", "a\n=======\n") != ""
    # Not markers: run embedded mid-line / glued alnum suffix.
    assert _marker_leak("a\n", "a\nx <<<<<<< y\n") == ""
    assert _marker_leak("a\n", "a\n<<<<<html>\n") == ""


async def test_two_arg_replace_adding_conflict_markers_rejected(sandbox):
    # Write backstop: ADDING conflict-marker lines via replace is refused.
    target = sandbox / "clean.py"
    original = "x = 1\ny = 2\n"
    target.write_text(original)
    res = await tool_replace_text(
        "clean.py", "y = 2", "<<<<<<< HEAD\ny = 2\n=======\ny = 3\n>>>>>>> other", sandbox)
    assert "REJECTED" in res and "marker" in res
    assert target.read_text() == original


# ---------------------------------------------------------------------------
# MED — indented markers no longer evade the leak guard
# ---------------------------------------------------------------------------

def test_marker_line_re_tolerates_leading_whitespace():
    assert _marker_leak("a\n", "a\n    ====\n") != ""
    assert _marker_leak("a\n", "a\n\t<<<< SEARCH\n") != ""
    assert _marker_leak("a\n", "a\n  >>>>\n") != ""


async def test_indented_separator_leak_rejected(sandbox):
    target = sandbox / "app.js"
    original = "const a = 1;\n"
    target.write_text(original)
    res = await tool_replace_text(
        "app.js", "const a = 1;", "const a = 2;\n    ====\nstray", sandbox)
    assert "REJECTED" in res and "marker" in res
    assert target.read_text() == original


# ---------------------------------------------------------------------------
# MED — tool_write_file lone-surrogate content
# ---------------------------------------------------------------------------

async def test_lone_surrogate_write_returns_error_and_preserves_file(sandbox):
    target = sandbox / "keep.txt"
    target.write_text("original content\n")
    # The exact production shape: a \ud800 escape surviving json.loads.
    bad = json.loads('"\\ud800 payload"')
    res = await tool_write_file("keep.txt", bad, sandbox)
    assert res.startswith("Error:"), f"needs the Error: prefix, got: {res[:80]}"
    assert target.read_text() == "original content\n", \
        "the old strict write_text truncated the file to 0 bytes"


async def test_lone_surrogate_write_does_not_create_new_file(sandbox):
    bad = json.loads('"\\udc00abc"')
    res = await tool_write_file("new.txt", bad, sandbox)
    assert res.startswith("Error:")
    assert not (sandbox / "new.txt").exists()
    assert not list(sandbox.glob("*.tmp"))


async def test_normal_unicode_write_unaffected(sandbox):
    res = await tool_write_file("greek.txt", "γειά σου κόσμε\n", sandbox)
    assert res.startswith("SUCCESS")
    assert (sandbox / "greek.txt").read_text(encoding="utf-8") == "γειά σου κόσμε\n"
    assert not list(sandbox.glob("*.tmp"))


# ---------------------------------------------------------------------------
# MED — foreign-project prefix must not collapse into the active project
# ---------------------------------------------------------------------------

ACTIVE = "abc123def456"
OTHER = "fedcba654321"


def _scoped(tmp_path):
    root = tmp_path / "sandbox"
    sb = root / "projects" / ACTIVE
    sb.mkdir(parents=True)
    return sb, root


def test_foreign_project_path_resolves_to_that_project(tmp_path):
    sb, root = _scoped(tmp_path)
    other_dir = root / "projects" / OTHER
    other_dir.mkdir(parents=True)
    (other_dir / "data.txt").write_text("theirs")

    got = _get_safe_path(sb, f"projects/{OTHER}/data.txt")
    assert got == (other_dir / "data.txt").resolve(), \
        "explicit foreign-project path silently rewrote to the active project"


def test_foreign_project_workspace_prefixed_path_resolves_there(tmp_path):
    sb, root = _scoped(tmp_path)
    other_dir = root / "projects" / OTHER
    other_dir.mkdir(parents=True)
    got = _get_safe_path(sb, f"/workspace/projects/{OTHER}/report.md")
    assert got == (other_dir / "report.md").resolve()


def test_title_guess_slug_still_collapses_to_active(tmp_path):
    # No backing dir → the historical heal: title-derived slugs collapse
    # into the active project so store/sweep/file agree on one id.
    sb, root = _scoped(tmp_path)
    got = _get_safe_path(sb, "projects/DeskMiniX3/index.html")
    assert got == (sb / "index.html").resolve()


def test_active_id_prefix_still_strips(tmp_path):
    sb, root = _scoped(tmp_path)
    got = _get_safe_path(sb, f"projects/{ACTIVE}/f.txt")
    assert got == (sb / "f.txt").resolve()


def test_dotdot_slug_never_treated_as_foreign(tmp_path):
    sb, root = _scoped(tmp_path)
    with pytest.raises(ValueError, match="Security"):
        _get_safe_path(sb, "projects/../../evil.txt")


# ---------------------------------------------------------------------------
# MED — tool_rename_file must not clobber an existing destination
# ---------------------------------------------------------------------------

async def test_rename_refuses_existing_file_destination(sandbox):
    (sandbox / "a.txt").write_text("A")
    (sandbox / "b.txt").write_text("B")
    res = await tool_rename_file("a.txt", "b.txt", sandbox)
    assert res.startswith("Error") and "already exists" in res
    assert (sandbox / "a.txt").read_text() == "A"
    assert (sandbox / "b.txt").read_text() == "B", "destination was clobbered"


async def test_rename_into_existing_directory_needs_explicit_path(sandbox):
    (sandbox / "a.txt").write_text("A")
    (sandbox / "dest").mkdir()
    res = await tool_rename_file("a.txt", "dest", sandbox)
    assert res.startswith("Error") and "directory" in res
    assert (sandbox / "a.txt").exists()
    # The explicit form it steers to still works.
    res2 = await tool_rename_file("a.txt", "dest/a.txt", sandbox)
    assert res2.startswith("SUCCESS")
    assert (sandbox / "dest" / "a.txt").read_text() == "A"


async def test_rename_to_fresh_destination_still_works(sandbox):
    (sandbox / "old.txt").write_text("data")
    res = await tool_rename_file("old.txt", "new.txt", sandbox)
    assert res.startswith("SUCCESS")
    assert (sandbox / "new.txt").read_text() == "data"
    assert not (sandbox / "old.txt").exists()


# ---------------------------------------------------------------------------
# LOW — rg pattern passed via -e (a `-`-leading pattern is not a flag)
# ---------------------------------------------------------------------------

class _FakeSandboxManager:
    def __init__(self, output="", exit_code=0):
        self._ret = (output, exit_code)
        self.last_cmd = None

    def execute(self, cmd, timeout=None):
        self.last_cmd = cmd
        return self._ret


async def test_search_pattern_passed_with_e_flag(sandbox):
    (sandbox / "f.txt").write_text("-marker here\n")
    mgr = _FakeSandboxManager("1:-marker here", 0)
    res = await tool_file_search("-marker", sandbox, "f.txt", mgr)
    assert "1:-marker here" in res
    assert " -e " in mgr.last_cmd, mgr.last_cmd
    assert mgr.last_cmd.split(" -e ", 1)[1].startswith("-marker")
