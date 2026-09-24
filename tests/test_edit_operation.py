"""§4KC — `file_system operation='edit'`: exact, unique, or refused with evidence.

Every test names the world it fails in. The RETURN_PATHS table drives each
return statement of `_edit_text_impl` once and asserts exactly one ledger
row AND the reason code that return is meant to carry, on the same tripwire
idiom as `tests/test_edit_ladder_ledger.py`: when a return path is added,
the count assertion fails until the table grows.

Round 2 (fresh-eye review, three lenses) added the pins marked [R2].
"""
import ast
import asyncio
import errno
import inspect
import os
import stat
import textwrap

import pytest

from ghost_agent.tools import file_system as fs
from ghost_agent.tools.file_system import (tool_edit_text, tool_file_system,
                                           tool_replace_text)
from ghost_agent.utils.edit_ledger import read_ledger


@pytest.fixture
def ws(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    return d


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("GHOST_HOME", str(h))
    return h


PY_SRC = "import os\n\ndef f():\n    return 1\n\ndef g():\n    return 1\n"


async def _edit(ws, path, old, new, **kw):
    return await tool_file_system(operation="edit", path=path, sandbox_dir=ws,
                                  old_string=old, new_string=new, **kw)


def _rc(res):
    return getattr(res, "reason_code", None)


# ---------------------------------------------------------------------------
# 1. The contract: exact + unique lands; the file and the ledger agree
# ---------------------------------------------------------------------------

async def test_exact_unique_edit_lands_and_is_recorded(ws, home):
    f = ws / "a.py"
    f.write_text(PY_SRC)
    res = await _edit(ws, "a.py", "def f():\n    return 1\n",
                      "def f():\n    return 2\n")
    assert str(res).startswith("SUCCESS:"), res
    assert "(line 3) in 'a.py'." in str(res)
    assert f.read_text() == PY_SRC.replace("def f():\n    return 1",
                                           "def f():\n    return 2")
    (row,) = read_ledger(home=home)
    assert row["op"] == "edit"
    assert row["applied"] is True
    assert row["strategies"] == ["exact"]
    assert row["blocks_total"] == row["blocks_applied"] == 1
    assert row["reason"] == ""


async def test_empty_new_string_deletes(ws, home):
    """`new_string=""` is a legal deletion, not a missing argument."""
    f = ws / "a.txt"
    f.write_text("keep\ndrop me\nkeep\n")
    res = await _edit(ws, "a.txt", "drop me\n", "")
    assert str(res).startswith("SUCCESS:"), res
    assert f.read_text() == "keep\nkeep\n"


async def test_replace_all_changes_every_occurrence(ws, home):
    f = ws / "a.py"
    f.write_text(PY_SRC)
    res = await _edit(ws, "a.py", "return 1", "return 0", replace_all=True)
    assert str(res).startswith("SUCCESS:"), res
    assert "2 occurrences" in str(res) and "lines 4, 7" in str(res)
    assert f.read_text().count("return 0") == 2
    (row,) = read_ledger(home=home)
    assert row["applied"] and row["blocks_applied"] == 2
    assert row["strategies"] == ["exact", "exact"]


@pytest.mark.parametrize("flag", ["true", "1", "yes", True])
async def test_replace_all_accepts_the_spellings_a_model_sends(ws, home, flag):
    (ws / "a.py").write_text(PY_SRC)
    res = await _edit(ws, "a.py", "return 1", "return 0", replace_all=flag)
    assert str(res).startswith("SUCCESS:"), (flag, res)


async def test_overlapping_occurrences_count_like_replace_does(ws, home):
    """[R2] `x\\nx\\nx\\nx` / `x\\nx\\n`: `str.count` says 2 (non-overlapping) and
    the lines named must be the two that a replace_all touches — 1 and 3 —
    not the overlapping walk's 1, 2, 3. Fails in the overlapping world."""
    f = ws / "o.txt"
    f.write_text("x\nx\nx\nx\n")
    res = await _edit(ws, "o.txt", "x\nx\n", "y\n")
    assert _rc(res) == "old_string_ambiguous", res
    assert "occurs 2 times" in str(res) and "line(s) 1, 3 " in str(res)
    res = await _edit(ws, "o.txt", "x\nx\n", "y\n", replace_all=True)
    assert "lines 1, 3" in str(res), res
    assert f.read_text() == "y\ny\n"


async def test_ambiguity_names_at_most_the_cap_and_says_so(ws, home):
    """[R2] mutation survivor: the cap was unpinned."""
    f = ws / "many.txt"
    f.write_text("".join(f"dup\nline {i}\n" for i in range(9)))
    res = await _edit(ws, "many.txt", "dup\n", "x\n")
    assert _rc(res) == "old_string_ambiguous"
    assert "occurs 9 times" in str(res)
    assert "line(s) 1, 3, 5, 7, 9, 11, 13, 15 — first 8 shown" in str(res), res


# ---------------------------------------------------------------------------
# 2. NO ladder on this path — pinned by CONTRAST with `replace`
# ---------------------------------------------------------------------------

async def test_a_whitespace_drifted_old_string_is_refused_where_replace_would_apply(ws, home):
    """The contrast is the pin. `replace`'s flexible rung applies this call;
    `edit` refuses it and says WHY (indentation), with the region shown.
    Fails in the world where `edit` grew a tolerance rung.
    """
    (ws / "a.py").write_text(PY_SRC)
    (ws / "b.py").write_text(PY_SRC)
    drifted = "def g():\n  return 1\n"           # 2-space indent, file has 4

    r_rep = await tool_replace_text("b.py", drifted, "def g():\n    return 9\n", ws)
    assert "return 9" in (ws / "b.py").read_text(), (
        "precondition: the ladder applies this on a tolerant rung", r_rep)

    r_edit = await _edit(ws, "a.py", drifted, "def g():\n    return 9\n")
    assert _rc(r_edit) == "old_string_not_found", r_edit
    assert "different indentation or spacing" in str(r_edit)
    assert ">>>    6: def g():" in str(r_edit), str(r_edit)
    assert (ws / "a.py").read_text() == PY_SRC

    rows = read_ledger(home=home)
    by_op = {r["op"]: r for r in rows}
    assert by_op["replace"]["applied"] is True
    assert by_op["replace"]["strategies"] and by_op["replace"]["strategies"][0] != "exact"
    assert by_op["edit"]["applied"] is False
    assert by_op["edit"]["strategies"] == []


async def test_a_crlf_file_is_edited_through_the_crlf_inverse(ws, home):
    """[R2] The turn loop strips every `\\r` from tool results, so the model
    can never produce a CRLF old_string; a byte-for-byte rule made every
    CRLF file un-editable. The one invertible normalisation applies, writes
    CRLF, and is recorded as `exact:crlf` so the ledger can see it.
    Fails in the world that refuses (or the one that silently writes LF)."""
    f = ws / "w.txt"
    f.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    res = await _edit(ws, "w.txt", "alpha\nbeta\n", "x\ny\n")
    assert str(res).startswith("SUCCESS:"), res
    assert f.read_bytes() == b"x\r\ny\r\ngamma\r\n"
    (row,) = read_ledger(home=home)
    assert row["strategies"] == ["exact:crlf"] and row["applied"] is True


async def test_line_endings_outside_the_edit_are_preserved(ws, home):
    f = ws / "w.txt"
    f.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    res = await _edit(ws, "w.txt", "beta", "BETA")
    assert str(res).startswith("SUCCESS:"), res
    assert f.read_bytes() == b"alpha\r\nBETA\r\ngamma\r\n"


async def test_miss_diagnosis_names_the_diverging_block(ws, home):
    (ws / "a.py").write_text(PY_SRC)
    res = await _edit(ws, "a.py", "def g():\n    return TWO\n", "x")
    assert _rc(res) == "old_string_not_found"
    assert "first line of old_string occurs at line(s) 6" in str(res)
    assert ">>>    6: def g():" in str(res)


async def test_miss_diagnosis_names_case_on_the_right_line(ws, home):
    """[R2] the line used to be computed from an offset into `.lower()`,
    whose length differs from the original for `İ`."""
    (ws / "a.txt").write_text("İİİİİİİİİİ\nHello\n")
    res = await _edit(ws, "a.txt", "hello", "x")
    assert _rc(res) == "old_string_not_found"
    assert "letter case" in str(res)
    assert ">>>    2: Hello" in str(res), res


async def test_miss_diagnosis_does_not_blame_indentation_falsely(ws, home):
    """[R2] three false "different indentation" verdicts: a diverging tail,
    a missing EOF newline, and the whitespace test not being line-anchored."""
    f = ws / "t.py"
    f.write_text("def f():\n    return 1 + x\n")
    res = await _edit(ws, "t.py", "    return 1\n", "    return 2\n")
    assert _rc(res) == "old_string_not_found"
    assert "different indentation" not in str(res), res

    f.write_text("def f():\n    return 1")                # no EOF newline
    res = await _edit(ws, "t.py", "    return 1\n", "    return 2\n")
    assert "no trailing newline" in str(res), res
    assert "different indentation" not in str(res)


async def test_the_miss_snippet_shows_three_lines_of_context(ws, home):
    """[R2] mutation survivor: `_snippet_at` context was unpinned."""
    (ws / "c.txt").write_text("".join(f"row {i}\n" for i in range(1, 13)))
    res = await _edit(ws, "c.txt", "row 6\nWRONG\n", "x")
    assert _rc(res) == "old_string_not_found"
    body = str(res)
    for k in (3, 4, 5, 7, 8, 9):
        assert f"    {k}: row {k}" in body, (k, body)
    assert ">>>    6: row 6" in body
    assert "   2: row 2" not in body and "  10: row 10" not in body


async def test_ambiguity_is_refused_with_the_lines(ws, home):
    f = ws / "a.py"
    f.write_text(PY_SRC)
    res = await _edit(ws, "a.py", "    return 1\n", "    return 2\n")
    assert _rc(res) == "old_string_ambiguous", res
    assert "at line(s) 4, 7" in str(res)
    assert "replace_all" in str(res)
    assert f.read_text() == PY_SRC
    (row,) = read_ledger(home=home)
    assert row["applied"] is False and row["reason"] == "old_string_ambiguous"


# ---------------------------------------------------------------------------
# 3. Arguments are validated IN THE TOOL, not only by the schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("old,new,code", [
    (None, "x", "missing_old_string"),
    ("", "x", "missing_old_string"),
    ("def f():", None, "missing_new_string"),
    ("def f():", "def f():", "identical_edit"),
    (["def f():"], "x", "non_string_argument"),
    ("def f():", {"a": 1}, "non_string_argument"),
    (0, "x", "non_string_argument"),
    ("def f():", "de\x00f", "nul_in_new_string"),
])
async def test_argument_refusals(ws, home, old, new, code):
    f = ws / "a.py"
    f.write_text(PY_SRC)
    res = await _edit(ws, "a.py", old, new)
    assert _rc(res) == code, (old, new, res)
    assert f.read_text() == PY_SRC
    (row,) = read_ledger(home=home)
    assert row["applied"] is False and row["reason"] == code


async def test_old_names_are_refused_not_healed(ws, home):
    """No aliases: `content`/`replace_with` on `edit` names the right
    parameters instead of being mapped onto them."""
    f = ws / "a.py"
    f.write_text(PY_SRC)
    res = await tool_file_system(operation="edit", path="a.py", sandbox_dir=ws,
                                 content="def f():\n    return 1\n",
                                 replace_with="def f():\n    return 2\n")
    assert _rc(res) == "edit_wrong_params", res
    assert "old_string" in str(res) and "new_string" in str(res)
    assert f.read_text() == PY_SRC


async def test_missing_old_string_names_write_for_creation(ws, home):
    (ws / "a.py").write_text(PY_SRC)
    res = await _edit(ws, "a.py", None, "whole new file")
    assert _rc(res) == "missing_old_string"
    assert "operation='write'" in str(res)


@pytest.mark.parametrize("which", ["old_string", "new_string"])
async def test_an_envelope_in_either_argument_is_refused(ws, home, which):
    f = ws / "a.py"
    f.write_text(PY_SRC)
    env = "<<<< SEARCH\ndef f():\n====\ndef h():\n>>>>"
    kw = {"old": env, "new": "x"} if which == "old_string" else {"old": "def f():", "new": env}
    res = await _edit(ws, "a.py", kw["old"], kw["new"])
    assert _rc(res) == "envelope_on_edit", res
    assert f.read_text() == PY_SRC


async def test_a_setext_underline_is_writable(ws, home):
    """`====` is a Markdown heading underline; the envelope marker-leak
    guard is OFF on this path."""
    f = ws / "README.md"
    f.write_text("Intro\n\ntext\n")
    res = await _edit(ws, "README.md", "Intro\n", "Intro\n=====\n")
    assert str(res).startswith("SUCCESS:"), res
    assert f.read_text().startswith("Intro\n=====\n")


# ---------------------------------------------------------------------------
# 4. A rewrite wearing an edit's clothes is refused; a small file is not
# ---------------------------------------------------------------------------

async def test_whole_file_old_string_is_refused_naming_write(ws, home):
    body = "".join(f"line {i}\n" for i in range(40))
    f = ws / "big.txt"
    f.write_text(body)
    res = await _edit(ws, "big.txt", body, body.replace("line 3\n", "LINE 3\n"))
    assert _rc(res) == "whole_file_on_edit", res
    assert "40 of the 40 lines" in str(res)
    assert "operation='write'" in str(res)
    assert f.read_text() == body
    (row,) = read_ledger(home=home)
    assert row["applied"] is False and row["reason"] == "whole_file_on_edit"


async def test_the_line_count_is_real_lines_not_newlines_plus_one(ws, home):
    """[R2] a 29-line file was counted as 30 and refused one line early."""
    body = "".join(f"l{i}\n" for i in range(29))
    f = ws / "t.txt"
    f.write_text(body)
    res = await _edit(ws, "t.txt", body, body.replace("l3\n", "L3\n"))
    assert str(res).startswith("SUCCESS:"), res


async def test_a_one_line_edit_of_a_long_line_is_not_a_rewrite(ws, home):
    """[R2] the gate measured CHARS only; a minified line carried 98% of the
    file's chars and a one-line edit of it was refused as a rewrite."""
    long = "x" * 5000
    body = "".join(f"s{i}\n" for i in range(35)) + long + "\n"
    f = ws / "min.js"
    f.write_text(body)
    res = await _edit(ws, "min.js", long, "y" * 10)
    assert str(res).startswith("SUCCESS:"), res
    assert f.read_text().endswith("y" * 10 + "\n")


async def test_a_short_file_may_be_edited_whole(ws, home):
    f = ws / "tiny.txt"
    f.write_text("a\nb\nc\n")
    res = await _edit(ws, "tiny.txt", "a\nb\nc\n", "x\ny\nz\n")
    assert str(res).startswith("SUCCESS:"), res
    assert f.read_text() == "x\ny\nz\n"


async def test_no_auto_promote_a_module_body_is_not_written(ws, home):
    f = ws / "mod.py"
    f.write_text(PY_SRC)
    module = "import sys\n\ndef main():\n    return 42\n\nif __name__ == '__main__':\n    main()\n"
    assert fs._looks_like_complete_python_module(module)      # precondition
    res = await _edit(ws, "mod.py", module, None)
    assert _rc(res) == "missing_new_string", res
    assert f.read_text() == PY_SRC
    (row,) = read_ledger(home=home)
    assert row["strategies"] == [] and row["applied"] is False


# ---------------------------------------------------------------------------
# 5. The file-state refusals: typed, recorded, and the file untouched
# ---------------------------------------------------------------------------

async def test_missing_file_is_refused_and_not_created(ws, home):
    res = await _edit(ws, "nope.py", "a", "b")
    assert _rc(res) == "file_not_found", res
    assert "operation='write'" in str(res)
    assert not (ws / "nope.py").exists()


async def test_a_fifo_is_refused_without_blocking(ws, home):
    os.mkfifo(ws / "pipe")
    res = await asyncio.wait_for(_edit(ws, "pipe", "a", "b"), timeout=5)
    assert _rc(res) == "not_a_regular_file"
    (row,) = read_ledger(home=home)
    assert row["applied"] is False and row["reason"] == "not_a_regular_file"


async def test_binary_is_refused_with_a_recorded_reason(ws, home):
    """[R2] plain-string refusals logged an EMPTY ledger reason."""
    (ws / "img.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8)
    res = await _edit(ws, "img.png", "PNG", "JPG")
    assert _rc(res) == "binary"
    (row,) = read_ledger(home=home)
    assert row["applied"] is False and row["reason"] == "binary"


async def test_path_escape_is_a_rejection_outcome(ws, home):
    res = await _edit(ws, "../../etc/passwd", "root", "toor")
    assert getattr(res, "is_rejection", False), res
    assert _rc(res) == "unsafe_path"


async def test_an_overlong_name_is_an_outcome_not_a_traceback(ws, home):
    """[R2] `Path.exists()` on 3.10 lets ENAMETOOLONG/EACCES escape."""
    res = await _edit(ws, "a" * 300 + ".txt", "x", "y")
    assert _rc(res) in ("unsafe_path", "stat_failed", "file_not_found"), res
    (row,) = read_ledger(home=home)
    assert not row["reason"].startswith("exception:"), row


async def test_a_symlink_loop_is_an_outcome_not_a_traceback(ws, home):
    os.symlink("loop", ws / "loop")
    res = await _edit(ws, "loop", "x", "y")
    assert _rc(res) in ("unsafe_path", "unresolvable_path", "stat_failed",
                        "file_not_found", "read_failed", "not_a_regular_file"), res


@pytest.mark.parametrize("spelling", [".git/config", ".GIT/config", ".Git/config",
                                      "sub/.gIt/hooks/pre-commit"])
async def test_dotgit_is_blocked_for_edit_in_any_case(ws, home, spelling):
    """[R2] the block was case-sensitive on a case-insensitive volume."""
    (ws / ".git").mkdir(exist_ok=True)
    (ws / ".git" / "config").write_text("[core]\n")
    res = await _edit(ws, spelling, "[core]", "[core]\n\thooksPath = /x")
    assert _rc(res) == "dotgit_write_blocked", (spelling, res)
    assert (ws / ".git" / "config").read_text() == "[core]\n"


async def test_syntax_regression_rolls_back_and_the_row_says_matched_not_applied(ws, home):
    f = ws / "a.py"
    f.write_text(PY_SRC)
    res = await _edit(ws, "a.py", "def f():\n    return 1\n", "def f(:\n    return 1\n")
    assert _rc(res) == "syntax_regression_rolled_back", res
    assert f.read_text() == PY_SRC
    (row,) = read_ledger(home=home)
    assert row["strategies"] == ["exact"]
    assert row["applied"] is False
    assert row["reason"] == "syntax_regression_rolled_back"


async def test_a_failed_write_is_not_recorded_as_applied(ws, home):
    """EACCES fails the OPEN (no O_TRUNC in the flags), so the file is
    intact — and the outcome says exactly that: FAILED, world_changed=False.
    Read-only means read-only: the writer is deliberately not tmp+rename."""
    f = ws / "ro.txt"
    f.write_text("VALUE = 1\n")
    f.chmod(0o444)
    try:
        res = await _edit(ws, "ro.txt", "VALUE = 1", "VALUE = 2")
        assert _rc(res) == "edit_write_failed", res
        assert not getattr(res, "is_rejection", False), res
        assert getattr(res, "world_changed", None) is False, res
        assert "unchanged on disk" in str(res)
        assert f.read_text() == "VALUE = 1\n"
        (row,) = read_ledger(home=home)
        assert row["applied"] is False
        assert row["reason"] == "edit_write_failed"
    finally:
        f.chmod(0o644)


@pytest.mark.parametrize("keep", [3, 20])
async def test_a_write_that_failed_after_truncating_is_measured_by_bytes(ws, home, monkeypatch, keep):
    """[R2] A failure after the truncate leaves a shorter file — or, with
    `keep=20`, a SAME-SIZE file with different bytes, which a size check
    read as "unchanged". Measured by comparing bytes; world_changed=True."""
    f = ws / "t.txt"
    orig = "VALUE = 1\nVALUE = 3\n"                       # 20 bytes
    f.write_text(orig)

    def _truncate_then_boom(real, data, expect):
        with open(real, "wb") as fh:
            fh.write(data[:keep])
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(fs, "_write_edit_nofollow", _truncate_then_boom)
    res = await _edit(ws, "t.txt", "VALUE = 1", "VALUE = 1000000")
    assert _rc(res) == "edit_write_failed", res
    assert getattr(res, "world_changed", None) is True, res
    assert "TRUNCATED or PARTIALLY" in str(res) and f"{keep} bytes now" in str(res)
    (row,) = read_ledger(home=home)
    assert row["applied"] is False and row["reason"] == "edit_write_failed"


async def test_lone_surrogate_in_new_string_is_refused_before_truncation(ws, home):
    f = ws / "a.txt"
    f.write_text("hello\n")
    res = await _edit(ws, "a.txt", "hello", "hel\ud800lo")
    assert _rc(res) == "lone_surrogate", res
    assert f.read_text() == "hello\n"


async def test_a_concurrent_writer_between_read_and_write_is_refused(ws, home, monkeypatch):
    """[R2] the write re-checks the target's identity+stamp against the
    read BEFORE truncating. Simulated inside the syntax guard: another
    writer appends a line while the edit is being checked."""
    f = ws / "a.py"
    f.write_text(PY_SRC)
    real_guard = fs._syntax_regression

    def _sneak(prev, new, name):
        os.utime(f, (0, 0))                       # a different mtime stamp
        with open(f, "a") as fh:
            fh.write("# someone else\n")
        return real_guard(prev, new, name)
    monkeypatch.setattr(fs, "_syntax_regression", _sneak)
    res = await _edit(ws, "a.py", "def f():\n    return 1\n", "def f():\n    return 2\n")
    assert _rc(res) == "file_changed_underneath", res
    assert f.read_text() == PY_SRC + "# someone else\n"      # theirs, not ours
    (row,) = read_ledger(home=home)
    # matched exact, did NOT land — the rollback shape, not a miss
    assert row["applied"] is False and row["strategies"] == ["exact"]


async def test_a_symlink_swapped_in_after_the_check_never_receives_the_edit(ws, home, monkeypatch, tmp_path):
    """[R2] CRITICAL: the write used to follow whatever `path` was at write
    time. With O_NOFOLLOW + identity check, a symlink planted between the
    safe-path check and the write is refused and the outside file is
    untouched. Fails in the `path.write_bytes` world."""
    outside = tmp_path / "secret.txt"
    outside.write_text("SECRET=1\n")
    f = ws / "race.txt"
    f.write_text("SECRET=1\n")
    real_guard = fs._syntax_regression

    def _swap(prev, new, name):
        f.unlink()
        os.symlink(outside, f)
        return real_guard(prev, new, name)
    monkeypatch.setattr(fs, "_syntax_regression", _swap)
    res = await _edit(ws, "race.txt", "SECRET=1", "PWNED=1")
    assert _rc(res) == "file_changed_underneath", res
    assert outside.read_text() == "SECRET=1\n"
    assert os.path.islink(f)


async def test_post_edit_view_rides_the_treatment_flag_through_the_dispatcher(ws, home):
    """[R2] mutation survivor: the dispatcher's `post_edit=_fs_batch` wiring
    was unpinned (the old test called `tool_edit_text` directly)."""
    (ws / "a.py").write_text(PY_SRC)
    res = await _edit(ws, "a.py", "def f():\n    return 1\n",
                      "def f():\n    return 2\n", fs_batch_enabled=True)
    assert str(res).startswith("SUCCESS:")
    assert "POST-EDIT VIEW" in str(res) and "return 2" in str(res)
    res = await _edit(ws, "a.py", "def f():\n    return 2\n",
                      "def f():\n    return 3\n")
    assert "POST-EDIT VIEW" not in str(res)


# ---------------------------------------------------------------------------
# 6. Every return path, once — the tripwire
# ---------------------------------------------------------------------------

def _impl_return_count():
    tree = ast.parse(textwrap.dedent(inspect.getsource(fs._edit_text_impl)))
    fn = tree.body[0]
    return sum(1 for n in ast.walk(fn) if isinstance(n, ast.Return))


#: label -> the reason code that return statement carries (None = applied).
RETURN_PATHS = {
    "missing_old_string": "missing_old_string",
    "missing_new_string": "missing_new_string",
    "non_string_argument": "non_string_argument",
    "identical_edit": "identical_edit",
    "nul_in_new_string": "nul_in_new_string",
    "envelope_on_edit": "envelope_on_edit",
    "unsafe_path": "unsafe_path",
    "unsafe_path_resolve": "unsafe_path",
    "stat_failed": "stat_failed",
    "root_fallback": "file_not_found",
    "file_not_found": "file_not_found",
    "not_a_regular_file": "not_a_regular_file",
    "not_a_regular_file_fd": "not_a_regular_file",
    "too_large": "too_large",
    "read_failed": "read_failed",
    "binary": "binary",
    "whole_file_on_edit": "whole_file_on_edit",
    "old_string_not_found": "old_string_not_found",
    "old_string_ambiguous": "old_string_ambiguous",
    "edit_write_failed": "edit_write_failed",
    "inode_denied": "hard_linked",
    "applied": None,
}


def test_the_return_path_table_is_not_a_sample():
    """22 `return` statements in `_edit_text_impl` today, one label each
    (`file_changed_underneath` is the guarded writer's outcome, pinned
    separately). Grows → this fails until RETURN_PATHS covers the new one."""
    assert _impl_return_count() == len(RETURN_PATHS) == 22, (
        _impl_return_count(), len(RETURN_PATHS))


async def _drive(label, ws, monkeypatch):
    f = ws / "a.py"
    f.write_text(PY_SRC)
    old, new = "def f():\n    return 1\n", "def f():\n    return 2\n"
    if label == "missing_old_string":
        return await _edit(ws, "a.py", "", new)
    if label == "missing_new_string":
        return await _edit(ws, "a.py", old, None)
    if label == "non_string_argument":
        return await _edit(ws, "a.py", [old], new)
    if label == "identical_edit":
        return await _edit(ws, "a.py", old, old)
    if label == "nul_in_new_string":
        return await _edit(ws, "a.py", old, "x\x00y")
    if label == "envelope_on_edit":
        return await _edit(ws, "a.py", old, "<<<< SEARCH\nx\n====\ny\n>>>>")
    if label == "unsafe_path":
        return await _edit(ws, "../x.py", old, new)
    if label == "unsafe_path_resolve":
        # the dispatcher's own probe denies a loop first (`unresolvable_path`,
        # no ledger row); the impl's branch is reached through the wrapper
        os.symlink("loop", ws / "loop")
        return await tool_edit_text("loop", old, new, ws)
    if label == "stat_failed":
        real_exists = fs.Path.exists

        def _boom(self):
            if str(self).endswith("a.py"):
                raise PermissionError(errno.EACCES, "stat boom")
            return real_exists(self)
        monkeypatch.setattr(fs.Path, "exists", _boom)
        return await _edit(ws, "a.py", old, new)
    if label == "file_not_found":
        return await _edit(ws, "missing.py", old, new)
    if label == "root_fallback":
        root = ws.parent / "root"
        proj = root / "projects" / "p1"
        proj.mkdir(parents=True)
        (root / "a.py").write_text(PY_SRC)
        res = await tool_file_system(operation="edit", path="a.py",
                                     sandbox_dir=proj, old_string=old,
                                     new_string=new)
        assert (root / "a.py").read_text() == PY_SRC, "crossed scope"
        return res
    if label == "not_a_regular_file":
        os.mkfifo(ws / "p")
        return await asyncio.wait_for(_edit(ws, "p", old, new), timeout=5)
    if label == "not_a_regular_file_fd":
        # the path-level check is blinded; the fd-level check must still hold
        monkeypatch.setattr(fs, "_nonregular_refusal", lambda p, n: None)
        os.mkfifo(ws / "p")
        return await asyncio.wait_for(_edit(ws, "p", old, new), timeout=5)
    if label == "too_large":
        monkeypatch.setattr(fs, "_EDIT_MAX_BYTES", 10)
        return await _edit(ws, "a.py", old, new)
    if label == "read_failed":
        def _boom(real, **kw):
            raise OSError(errno.EIO, "read boom")
        monkeypatch.setattr(fs, "_read_regular_nofollow", _boom)
        return await _edit(ws, "a.py", old, new)
    if label == "binary":
        f.write_bytes(b"\x00\x01\x02" * 100)
        return await _edit(ws, "a.py", old, new)
    if label == "whole_file_on_edit":
        body = "".join(f"line {i}\n" for i in range(40))
        f.write_text(body)
        return await _edit(ws, "a.py", body, body + "x\n")
    if label == "old_string_not_found":
        return await _edit(ws, "a.py", "nowhere", new)
    if label == "old_string_ambiguous":
        return await _edit(ws, "a.py", "return 1", "return 0")
    if label == "inode_denied":
        os.link(f, ws / "twin.py")
        return await _edit(ws, "a.py", old, new)
    if label == "edit_write_failed":
        def _boom(real, data, expect):
            raise OSError(errno.EIO, "write boom")
        monkeypatch.setattr(fs, "_write_edit_nofollow", _boom)
        return await _edit(ws, "a.py", old, new)
    if label == "applied":
        return await _edit(ws, "a.py", old, new)
    raise AssertionError(label)


@pytest.mark.parametrize("label", sorted(RETURN_PATHS))
async def test_every_return_path_writes_exactly_one_row(label, ws, home, monkeypatch):
    res = await _drive(label, ws, monkeypatch)
    rows = read_ledger(home=home)
    assert len(rows) == 1, (label, rows, res)
    row = rows[0]
    assert row["op"] == "edit"
    expected = RETURN_PATHS[label]
    if expected is None:
        assert row["applied"] is True and row["reason"] == ""
    else:
        # the label must land on ITS return, not a neighbour's (R2: the old
        # table pinned label↔return only by a reviewer's trace)
        assert _rc(res) == expected, (label, res)
        assert row["applied"] is False, (label, row, res)
        assert row["reason"] == expected, (label, row)


@pytest.mark.parametrize("label", [l for l in sorted(RETURN_PATHS) if l != "applied"])
async def test_every_refusal_leaves_the_file_untouched(label, ws, home, monkeypatch):
    """The one promise every refusal message makes."""
    res = await _drive(label, ws, monkeypatch)
    assert not str(res).startswith("SUCCESS:"), (label, res)
    target = ws / "a.py"
    if label == "binary":
        assert target.read_bytes() == b"\x00\x01\x02" * 100
    elif label == "whole_file_on_edit":
        assert target.read_text() == "".join(f"line {i}\n" for i in range(40))
    elif label in ("unsafe_path", "unsafe_path_resolve", "file_not_found",
                   "not_a_regular_file", "not_a_regular_file_fd", "root_fallback"):
        pass                                     # no regular target file
    else:
        assert target.read_text() == PY_SRC, label


# ---------------------------------------------------------------------------
# 7. Every consumer that enumerates mutating file ops knows `edit`
# ---------------------------------------------------------------------------

def test_edit_is_in_every_mutating_op_enumeration():
    from ghost_agent.core.replay_engine import _PRODUCING_FS_OPS
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    from ghost_agent.utils.constraints import participant_write_violation
    from tests.test_participant_constraint_steer import PARTICIPANT_CONS

    assert "edit" in _PRODUCING_FS_OPS
    assert "edit" not in fs._READ_ONLY_OPS
    schema = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "file_system")
    props = schema["function"]["parameters"]["properties"]
    assert "edit" in props["operation"]["enum"]
    assert {"old_string", "new_string", "replace_all"} <= set(props)
    # the participant guard scans new_string, not just content/replace_with
    assert participant_write_violation(
        PARTICIPANT_CONS, {"operation": "edit", "path": "game.py",
                           "old_string": "pass",
                           "new_string": "def minimax(b, d): pass"}) is not None


def test_the_treatment_arm_schema_does_not_steer_back_to_envelopes():
    """[R2] the `fs_batch` treatment suffix (traffic 1.0 — half of live
    requests) told the model to pack envelopes into operation='replace' and
    omitted `edit` from its path-required list, contradicting the base
    schema's steer and confounding the edit-vs-replace ledger comparison
    by arm."""
    from ghost_agent.tools import registry as REG
    suffix = REG._FS_BATCH_DESC_SUFFIX
    assert "SEARCH" not in suffix and "operation='replace'" not in suffix
    assert "'edit'" in suffix
    import re as _re
    assert _re.search(r"REQUIRED for[^.]*\bedit\b", suffix), suffix


# ---------------------------------------------------------------------------
# 8. The verifier's consumers read the confirmation LINE, not the echo
# ---------------------------------------------------------------------------

def test_the_ledger_parses_the_edit_line_with_the_path_last():
    """[R2] a filename containing `' — replaced 1 occurrence` used to plant a
    phantom second path in the verifier's ledger."""
    from ghost_agent.core.agent import _files_mutated_this_turn
    # `a'.md`: a non-greedy pattern with no `$` anchor captures `a`
    crafted = "a'.md"
    msg = (f"SUCCESS: edited — replaced 1 occurrence of old_string (line 1) "
           f"in '{crafted}'.")
    assert _files_mutated_this_turn(
        [{"role": "tool", "name": "file_system", "content": msg}]) == [crafted]


def test_a_rejected_edit_whose_echo_says_success_is_not_an_unverified_mutation():
    """[R2] `_is_unverified_mutation` substring-scanned the WHOLE result: a
    miss on a README whose nearest-region echo contained "successfully
    edited" booked the turn as failed over a file nothing touched."""
    from ghost_agent.core.agent import (_is_unverified_mutation,
                                        _written_paths_from_confirmation)
    miss = ("REJECTED: old_string was not found in 'notes.md' — 'edit' "
            "matches exactly.\nNearest region of the file (real line numbers):\n"
            "       1: we successfully edited and wrote the file\n")
    assert _is_unverified_mutation(
        {"name": "file_system", "content": miss}) is False
    ok = ("SUCCESS: edited — replaced 1 occurrence of old_string (line 3) in "
          "'app.py'.\n--- POST-EDIT VIEW ---\nSUCCESS: Wrote 3 chars to 'x.md'. "
          "Script-side path (from sandbox cwd): 'x.md'.")
    # a targeted edit is no longer the gate's shape (operator, 2026-09-23);
    # the echo must still not reach the path parser
    assert _is_unverified_mutation({"name": "file_system", "content": ok}) is False
    assert _written_paths_from_confirmation(ok) == ["app.py"]
    whole = "SUCCESS: Wrote 3000 chars to 'app.py'. Script-side path (from sandbox cwd): 'app.py'."
    assert _is_unverified_mutation({"name": "file_system", "content": whole}) is True


# ---------------------------------------------------------------------------
# 9. Release immutability is a property of the RESOLVED path
# ---------------------------------------------------------------------------

class _Store:
    def __init__(self, released):
        self.released = set(released)

    def get_project(self, pid):
        return {"status": "RELEASED" if pid in self.released else "ACTIVE"}


@pytest.mark.parametrize("spelling", [
    "projects/aaaaaaaaaaaa/x.py",
    "Projects/aaaaaaaaaaaa/x.py",
    "projects/AAAAAAAAAAAA/x.py",
    "projects//aaaaaaaaaaaa/x.py",
    "projects/zz/../aaaaaaaaaaaa/x.py",
    "/workspace/projects/aaaaaaaaaaaa/x.py",
])
def test_every_spelling_of_a_released_workspace_is_blocked(tmp_path, spelling):
    """[R2] CRITICAL: the block regex-matched the RAW spelling, first hit
    only, case-sensitively — seven confirmed bypasses on APFS."""
    root = tmp_path / "root"
    (root / "projects" / "aaaaaaaaaaaa").mkdir(parents=True)
    (root / "projects" / "bbbbbbbbbbbb").mkdir(parents=True)
    store = _Store({"aaaaaaaaaaaa"})
    assert fs._released_write_block(store, root, spelling), spelling
    # from an ACTIVE scope, the released id later in the path still blocks
    assert fs._released_write_block(store, root / "projects" / "bbbbbbbbbbbb", spelling), spelling
    # and an ACTIVE target is not blocked
    assert fs._released_write_block(store, root, "projects/bbbbbbbbbbbb/x.py") is None


async def test_replace_gets_the_same_race_closure(ws, home, monkeypatch, tmp_path):
    """[§4KC r3] The security reviewer's TOCTOU was on the SHARED write site:
    `replace` followed a symlink planted after its safe-path check. It now
    writes through the same identity-checked O_NOFOLLOW fd as `edit`, and a
    concurrent writer is refused the same way."""
    outside = tmp_path / "secret.txt"
    outside.write_text("SECRET=1\n")
    f = ws / "race.txt"
    f.write_text("SECRET=1\n")
    real_guard = fs._syntax_regression

    def _swap(prev, new, name):
        f.unlink()
        os.symlink(outside, f)
        return real_guard(prev, new, name)
    monkeypatch.setattr(fs, "_syntax_regression", _swap)
    res = await tool_replace_text("race.txt", "SECRET=1", "PWNED=1", ws)
    assert _rc(res) == "file_changed_underneath", res
    assert outside.read_text() == "SECRET=1\n"
    assert os.path.islink(f)
    (row,) = read_ledger(home=home)
    assert row["op"] == "replace" and row["applied"] is False
    assert row["reason"] == "file_changed_underneath"

    monkeypatch.setattr(fs, "_syntax_regression", real_guard)
    g = ws / "b.py"
    g.write_text(PY_SRC)

    def _sneak(prev, new, name):
        with open(g, "a") as fh:
            fh.write("# other\n")
        os.utime(g, (0, 0))
        return real_guard(prev, new, name)
    monkeypatch.setattr(fs, "_syntax_regression", _sneak)
    res = await tool_replace_text("b.py", "def f():\n    return 1\n",
                                  "def f():\n    return 2\n", ws)
    assert _rc(res) == "file_changed_underneath", res
    assert g.read_text() == PY_SRC + "# other\n"


async def test_the_auto_promote_write_does_not_follow_a_planted_link(ws, home, monkeypatch, tmp_path):
    """[§4KC r3] the promote was the one write left that followed a link."""
    outside = tmp_path / "secret.py"
    outside.write_text("SECRET = 1\n")
    f = ws / "mod.py"
    f.write_text("x = 1\n")
    module = "import sys\n\ndef main():\n    return 42\n\nif __name__ == '__main__':\n    main()\n"
    real_promote = fs._looks_like_complete_python_module

    def _swap_then_judge(text):
        f.unlink()
        os.symlink(outside, f)
        return real_promote(text)
    monkeypatch.setattr(fs, "_looks_like_complete_python_module", _swap_then_judge)
    res = await tool_replace_text("mod.py", module, None, ws)
    assert not str(res).startswith("SUCCESS"), res
    assert outside.read_text() == "SECRET = 1\n"


# ---------------------------------------------------------------------------
# 10. Round 3: the guards decide on the INODE, not the requested spelling
# ---------------------------------------------------------------------------

async def test_a_hardlink_into_dotgit_or_a_release_is_refused(ws, home, tmp_path):
    """[R3] MAJOR ×2: `ln .git/config cfg.ini` carried no `.git` component
    and the in-place fd write mutated the shared inode; same for a hardlink
    into a RELEASED project's file. A shared inode is refused for in-place
    writers, by BOTH `edit` and `replace`."""
    (ws / ".git").mkdir()
    cfg = ws / ".git" / "config"
    cfg.write_text("[core]\n")
    os.link(cfg, ws / "cfg.ini")
    res = await _edit(ws, "cfg.ini", "[core]", "[core]\n\thooksPath = /x")
    assert _rc(res) == "hard_linked", res
    assert cfg.read_text() == "[core]\n"
    res = await tool_replace_text("cfg.ini", "[core]", "[core]\n\thooksPath = /x", ws)
    assert _rc(res) == "hard_linked", res
    assert cfg.read_text() == "[core]\n"
    (row_e, row_r) = read_ledger(home=home)
    assert row_e["reason"] == row_r["reason"] == "hard_linked"


async def test_an_apfs_fold_of_projects_is_still_a_released_write(ws, home, tmp_path):
    """[R3] MAJOR: `project\u017f` (LONG S) opens the real `projects` dir on
    APFS but `.lower()` leaves it alone. The decision is re-derived from the
    kernel's canonical path of the opened fd, so the spelling is irrelevant.
    Skipped where the volume does not fold (the canonical path then names
    a different, non-existent file)."""
    root = tmp_path / "root"
    rel = root / "projects" / "aaaaaaaaaaaa"
    rel.mkdir(parents=True)
    (rel / "x.py").write_text("x = 1\n")
    if not (root / "project\u017f" / "aaaaaaaaaaaa" / "x.py").exists():
        pytest.skip("volume does not fold LONG S to s")
    store = _Store({"aaaaaaaaaaaa"})
    res = await tool_file_system(operation="edit", path="project\u017f/aaaaaaaaaaaa/x.py",
                                 sandbox_dir=root, old_string="x = 1", new_string="x = 2",
                                 project_store=store)
    assert _rc(res) == "released_write_blocked", res
    assert (rel / "x.py").read_text() == "x = 1\n"
    res = await tool_file_system(operation="replace", path="project\u017f/aaaaaaaaaaaa/x.py",
                                 sandbox_dir=root, content="x = 1", replace_with="x = 2",
                                 project_store=store)
    assert _rc(res) == "released_write_blocked", res
    assert (rel / "x.py").read_text() == "x = 1\n"


async def test_a_forged_mtime_does_not_beat_the_identity_check(ws, home, monkeypatch):
    """[R3] a same-size rewrite + `os.utime` with the original nanoseconds
    restored every field but ctime, which userland cannot set."""
    f = ws / "a.py"
    f.write_text(PY_SRC)
    st0 = os.stat(f)
    real_guard = fs._syntax_regression

    def _forge(prev, new, name):
        body = PY_SRC.replace("return 1\n\ndef g", "return 7\n\ndef g")   # same size
        assert len(body) == len(PY_SRC)
        with open(f, "w") as fh:
            fh.write(body)
        os.utime(f, ns=(st0.st_atime_ns, st0.st_mtime_ns))
        return real_guard(prev, new, name)
    monkeypatch.setattr(fs, "_syntax_regression", _forge)
    res = await _edit(ws, "a.py", "def g():\n    return 1\n", "def g():\n    return 2\n")
    assert _rc(res) == "file_changed_underneath", res
    assert "return 7" in f.read_text() and "return 2" not in f.read_text()


async def test_an_unresolvable_path_is_denied_not_skipped(ws, home, tmp_path):
    """[R3] a `.git` symlink pointing out of the sandbox made the probe's
    resolve raise, and the broad except SKIPPED the deny."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    os.symlink("loop", ws / "loop")
    res = await _edit(ws, "loop", "x", "y")
    assert _rc(res) == "unresolvable_path", res


async def test_released_block_outcome_carries_a_reason_code(tmp_path):
    root = tmp_path / "root"
    (root / "projects" / "aaaaaaaaaaaa").mkdir(parents=True)
    out = fs._released_write_block(_Store({"aaaaaaaaaaaa"}), root, "projects/aaaaaaaaaaaa/x.py")
    assert out.reason_code == "released_write_blocked"


async def test_diagnosis_round3_sentences_are_true(ws, home):
    """[R3] a one-line old_string inside a longer line; mixed endings; an
    old_string longer than the file."""
    f = ws / "d.txt"
    f.write_text("xx return 1 yy\n")
    res = await _edit(ws, "d.txt", "return 1\n", "return 2\n")
    assert "only as PART of a longer line" in str(res), res
    assert "misremembered" not in str(res)

    f.write_bytes(b"def f():\r\n    return 1\ndef g():\n    pass\n")
    res = await _edit(ws, "d.txt", "def f():\n    return 1\n", "x")
    assert "mixes CRLF and LF" in str(res), res
    assert "different indentation" not in str(res)

    f.write_text("ab\n")
    res = await _edit(ws, "d.txt", "ab\ncd\nef\n", "x")
    assert "longer than what remains" in str(res), res


def test_a_bannered_inert_edit_keeps_its_exemption_and_a_padded_head_still_parses():
    """[R3] `_is_unverified_mutation` stripped the banner for the marker
    check but handed the RAW content to the path parser, so a bannered
    `.md` edit lost the inert-artifact exemption; and the ledger's head was
    not stripped, so a trailing space defeated the `$` anchor."""
    from ghost_agent.core.agent import (_files_mutated_this_turn,
                                        _is_unverified_mutation)
    bannered = ("[FAILURE BANNER] something\nSUCCESS: edited — replaced 1 "
                "occurrence of old_string (line 2) in 'notes.md'.")
    assert _is_unverified_mutation({"name": "file_system", "content": bannered}) is False
    padded = "SUCCESS: edited — replaced 1 occurrence of old_string (line 2) in 'app.py'.  \n"
    assert _files_mutated_this_turn(
        [{"role": "tool", "name": "file_system", "content": padded}]) == ["app.py"]


# ---------------------------------------------------------------------------
# 11. Round 4
# ---------------------------------------------------------------------------

async def test_replace_binds_the_write_to_the_inode_the_guard_cleared(ws, home, monkeypatch):
    """[R4] MAJOR by trace: `replace` took its identity from an `os.stat`
    BEFORE the inode guard opened its own fd; a name swapped away (guard
    sees a clean inode) and back (write lands on the original) passed the
    guard on one inode and wrote another. The identity is now the guarded
    fd's fstat, and it must agree with the pre-read stat."""
    # Swap by renaming DIRECTORIES: the file inodes keep their ctime, so
    # only the guard/identity binding can catch it (renaming the files
    # themselves bumps ctime and the stamp refuses regardless — a pin
    # written that way survived the mutation).
    (ws / "d").mkdir()
    (ws / "d2").mkdir()
    f = ws / "d" / "a.py"
    f.write_text(PY_SRC)
    other = ws / "d2" / "a.py"
    other.write_text(PY_SRC)
    real_open = os.open
    state = {"swapped": False}

    def _swap_before_guard(p, flags, *a, **k):
        # the guard's O_NOFOLLOW read-open is the first such open after the stat
        if (not state["swapped"] and str(p).endswith("a.py")
                and flags & getattr(os, "O_NOFOLLOW", 0) and not flags & os.O_WRONLY):
            state["swapped"] = True
            os.rename(ws / "d", ws / "tmp")
            os.rename(ws / "d2", ws / "d")
            fd = real_open(p, flags, *a, **k)
            os.rename(ws / "d", ws / "d2")
            os.rename(ws / "tmp", ws / "d")      # …and back: the original is at the name again
            return fd
        return real_open(p, flags, *a, **k)
    monkeypatch.setattr(fs.os, "open", _swap_before_guard)
    res = await tool_replace_text("d/a.py", "def f():\n    return 1\n", "def f():\n    return 2\n", ws)
    assert _rc(res) == "file_changed_underneath", res
    assert f.read_text() == PY_SRC and other.read_text() == PY_SRC


async def test_streaming_replace_keeps_the_mode_and_checks_its_source_inode(ws, home):
    """[R4] the streaming commit carried the tmp file's 0600 onto a 0644
    script, and read whatever inode the name held."""
    big = ws / "big.txt"
    body = "HEAD\n" + ("pad\n" * 300_000) + "TARGET\n"      # > 1 MB, one-line old: streaming
    big.write_text(body)
    big.chmod(0o644)
    res = await tool_replace_text("big.txt", "TARGET", "HIT", ws)
    assert "Streaming" in str(res), res
    assert stat.S_IMODE(os.stat(big).st_mode) == 0o644
    assert big.read_text().endswith("HIT\n")


async def test_streaming_replace_refuses_a_source_swapped_after_the_guard(ws, home, monkeypatch):
    """[R4] the streaming reader opens its own fd; a DIRECTORY swap between
    the guard's open and that open hands it another inode — refused with the
    guard's own verdict, and neither file is touched."""
    (ws / "d").mkdir()
    (ws / "d2").mkdir()
    body = "HEAD\n" + ("pad\n" * 300_000) + "TARGET\n"
    big = ws / "d" / "big.txt"
    big.write_text(body)
    other = ws / "d2" / "big.txt"
    other.write_text(body)
    import sys as _sys
    real_open = os.open
    state = {"swapped": False}

    def _swap_for_the_streaming_reader(p, flags, *a, **k):
        # swap exactly when the STREAMING closure opens its source (by the
        # caller's frame, not by counting opens — a count was wrong once)
        if (not state["swapped"] and str(p).endswith("big.txt")
                and _sys._getframe(1).f_code.co_name == "_streaming_replace"):
            state["swapped"] = True
            os.rename(ws / "d", ws / "tmp")
            os.rename(ws / "d2", ws / "d")
        return real_open(p, flags, *a, **k)
    monkeypatch.setattr(fs.os, "open", _swap_for_the_streaming_reader)
    res = await tool_replace_text("d/big.txt", "TARGET", "HIT", ws)
    assert state["swapped"], "the streaming reader never opened the file"
    assert _rc(res) == "file_changed_underneath", res
    assert (ws / "tmp" / "big.txt").read_text() == body
    assert (ws / "d" / "big.txt").read_text() == body


async def test_round4_diagnosis_sentences_are_true(ws, home):
    """[R4] the PART verdict fired on a line that IS the text (a trailing
    blank line in old_string); the ends-at sentence was gated on the wrong
    comparison."""
    f = ws / "d.txt"
    f.write_text("xx return 1 yy\nreturn 1\nfoo\n")
    res = await _edit(ws, "d.txt", "return 1\n\n", "x")
    assert "only as PART" not in str(res), res
    f.write_text("zz\nab\n")
    res = await _edit(ws, "d.txt", "ab\ncd\n", "x")
    assert "longer than what remains" in str(res), res
    assert "misremembered" not in str(res)


def test_the_canonical_dotgit_test_is_scoped_to_the_sandbox(tmp_path):
    """[R4] a `.git` ABOVE the sandbox root is the host's, not a write into
    repository internals."""
    root = tmp_path / ".git" / "sandbox"          # the sandbox lives under a .git dir
    root.mkdir(parents=True)
    f = root / "a.txt"
    f.write_text("x\n")
    fd = os.open(f, os.O_RDONLY)
    try:
        assert fs._inode_write_refusal(fd, "a.txt", root, None) is None
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# 12. Round 5
# ---------------------------------------------------------------------------

async def test_replace_never_reads_by_path_after_the_guard(ws, home, monkeypatch, tmp_path):
    """[R5] pre-existing MAJOR: `replace` re-read the target BY PATH after the
    guard, so a symlink planted in that window was followed on the READ side
    (an outside file's text copied into the sandbox) and a FIFO wedged the
    loop. Every read is through a guarded fd now."""
    outside = tmp_path / "secret.txt"
    outside.write_text("SECRET=1\nTARGET\n")
    f = ws / "r.txt"
    f.write_text("alpha\nTARGET\n")
    real_open = os.open
    state = {"n": 0}

    def _plant_after_guard(p, flags, *a, **k):
        # the guard's open is the first O_NOFOLLOW read-open; plant the
        # link right after it so the NEXT read sees a symlink
        if (str(p).endswith("r.txt") and flags & getattr(os, "O_NOFOLLOW", 0)
                and not flags & os.O_WRONLY):
            state["n"] += 1
            fd = real_open(p, flags, *a, **k)
            if state["n"] == 1:
                f.unlink()
                os.symlink(outside, f)
            return fd
        return real_open(p, flags, *a, **k)
    monkeypatch.setattr(fs.os, "open", _plant_after_guard)
    res = await tool_replace_text("r.txt", "TARGET", "HIT", ws)
    assert not str(res).startswith("SUCCESS"), res
    assert "SECRET" not in str(res)
    assert outside.read_text() == "SECRET=1\nTARGET\n"


async def test_replace_does_not_wedge_on_a_fifo_planted_after_the_guard(ws, home, monkeypatch):
    f = ws / "r.txt"
    f.write_text("alpha\nTARGET\n")
    real_open = os.open
    state = {"n": 0}

    def _plant_fifo(p, flags, *a, **k):
        if (str(p).endswith("r.txt") and flags & getattr(os, "O_NOFOLLOW", 0)
                and not flags & os.O_WRONLY):
            state["n"] += 1
            fd = real_open(p, flags, *a, **k)
            if state["n"] == 1:
                f.unlink()
                os.mkfifo(f)
            return fd
        return real_open(p, flags, *a, **k)
    monkeypatch.setattr(fs.os, "open", _plant_fifo)
    res = await asyncio.wait_for(tool_replace_text("r.txt", "TARGET", "HIT", ws), timeout=5)
    assert not str(res).startswith("SUCCESS"), res
    # the refusal names the cause; a "not found" over an empty non-blocking
    # read would be the wrong sentence (mutation: S_ISREG check dropped)
    assert "not a regular file" in str(res), res


async def test_replace_read_after_the_guard_is_bound_to_the_guarded_inode(ws, home, monkeypatch):
    """[R5] a DIRECTORY swap after the guard's open (no symlink, so
    O_NOFOLLOW cannot see it) must be refused by the read's inode compare,
    and the other directory's content must never feed the edit."""
    (ws / "d").mkdir()
    (ws / "d2").mkdir()
    f = ws / "d" / "r.txt"
    f.write_text("alpha\nTARGET\n")
    (ws / "d2" / "r.txt").write_text("OTHER\nTARGET\n")
    real_open = os.open
    state = {"n": 0}

    def _swap_after_guard_and_back_before_write(p, flags, *a, **k):
        # swap AFTER the guard's open (the read sees the other inode) and
        # BACK before the write (the guarded inode is at the name again):
        # without the read's own inode compare, OTHER's content would be
        # written over alpha's file with a SUCCESS
        if str(p).endswith("r.txt") and flags & os.O_WRONLY and state.get("swapped"):
            os.rename(ws / "d", ws / "d2")
            os.rename(ws / "tmp", ws / "d")
            state["swapped"] = False
        if (str(p).endswith("r.txt") and flags & getattr(os, "O_NOFOLLOW", 0)
                and not flags & os.O_WRONLY):
            state["n"] += 1
            fd = real_open(p, flags, *a, **k)
            if state["n"] == 1:
                os.rename(ws / "d", ws / "tmp")
                os.rename(ws / "d2", ws / "d")
                state["swapped"] = True
            return fd
        return real_open(p, flags, *a, **k)
    monkeypatch.setattr(fs.os, "open", _swap_after_guard_and_back_before_write)
    res = await tool_replace_text("d/r.txt", "TARGET", "HIT", ws)
    assert not str(res).startswith("SUCCESS"), res
    assert "changed on disk" in str(res), res
    # refused at the READ, so the directories may still be swapped: check
    # both files by content, wherever they sit — neither was touched
    assert sorted(p.read_text() for p in ws.rglob("r.txt")) == \
        ["OTHER\nTARGET\n", "alpha\nTARGET\n"]


async def test_replace_guard_halves_are_pinned_separately(ws, home, monkeypatch):
    """[R5] the r4 pin was satisfied by EITHER half of the fix. (i) a name
    swapped away and NOT back must be refused "while it was being checked"
    (the agreement check); (ii) a same-inode rewrite between the stat and
    the guard must NOT block the edit (the rebind to the guard's fstat)."""
    (ws / "d").mkdir()
    (ws / "d2").mkdir()
    f = ws / "d" / "a.py"
    f.write_text(PY_SRC)
    (ws / "d2" / "a.py").write_text(PY_SRC)
    real_open = os.open
    state = {"done": False}

    def _swap_away(p, flags, *a, **k):
        if (not state["done"] and str(p).endswith("a.py")
                and flags & getattr(os, "O_NOFOLLOW", 0) and not flags & os.O_WRONLY):
            state["done"] = True
            os.rename(ws / "d", ws / "tmp")
            os.rename(ws / "d2", ws / "d")
        return real_open(p, flags, *a, **k)
    monkeypatch.setattr(fs.os, "open", _swap_away)
    res = await tool_replace_text("d/a.py", "def f():\n    return 1\n", "def f():\n    return 2\n", ws)
    assert _rc(res) == "file_changed_underneath" and "being checked" in str(res), res
    monkeypatch.setattr(fs.os, "open", real_open)

    g = ws / "b.py"
    g.write_text(PY_SRC)
    state = {"done": False}

    def _rewrite_same_inode_before_guard(p, flags, *a, **k):
        if (not state["done"] and str(p).endswith("b.py")
                and flags & getattr(os, "O_NOFOLLOW", 0) and not flags & os.O_WRONLY):
            state["done"] = True
            with open(g, "w") as fh:                 # same inode, new stamp
                fh.write(PY_SRC.replace("import os", "import io"))
        return real_open(p, flags, *a, **k)
    monkeypatch.setattr(fs.os, "open", _rewrite_same_inode_before_guard)
    res = await tool_replace_text("b.py", "def f():\n    return 1\n", "def f():\n    return 2\n", ws)
    assert str(res).startswith("SUCCESS"), res
    assert "import io" in g.read_text() and "return 2" in g.read_text()


async def test_streaming_replace_respects_a_read_only_file(ws, home):
    """[R5] tmp+rename rewrote a 0444 file the other two writers refuse."""
    big = ws / "big.txt"
    big.write_text("HEAD\n" + ("pad\n" * 300_000) + "TARGET\n")
    big.chmod(0o444)
    try:
        res = await tool_replace_text("big.txt", "TARGET", "HIT", ws)
        assert not str(res).startswith("SUCCESS"), res
        assert big.read_text().endswith("TARGET\n")
    finally:
        big.chmod(0o644)


async def test_round5_diagnosis_sentences_are_true(ws, home):
    """[R5] popped trailing blanks got their own sentence; a block present
    earlier in the file is not reported as running past EOF."""
    f = ws / "d.txt"
    f.write_text("xx return 1 yy\nreturn 1\nfoo\n")
    res = await _edit(ws, "d.txt", "return 1\n\n", "x")
    assert "ends with 1 blank line(s)" in str(res), res
    assert "different indentation" not in str(res)

    f.write_text("def f():\n    return 1\nfoo\n\tdef f():\n")
    res = await _edit(ws, "d.txt", "def f():\n  return 1\n", "x")
    assert "different indentation" in str(res), res
    assert ">>>    1:" in str(res)
    assert "longer than what remains" not in str(res)


# ---------------------------------------------------------------------------
# 13. Round 6 (final)
# ---------------------------------------------------------------------------

async def test_round6_diagnosis_pins_see_each_change(ws, home):
    """[R6] the r5 pin was blind to two of the changes it claimed: the
    raw-slice gate and the slice-before-ends-at order. Each case here fails
    under exactly one revert."""
    f = ws / "d.txt"
    # (a) block present EARLY with an indentation mismatch AND a trailing
    # blank; its first line recurs near EOF so an ends-at-first order would
    # send the model to the last line instead of reporting indentation
    f.write_text("def f():\n    return 1\nfoo\nbar\ndef f():\n")
    res = await _edit(ws, "d.txt", "def f():\n  return 1\n\n", "x")
    assert "different indentation" in str(res), res
    assert ">>>    1:" in str(res)
    assert "longer than what remains" not in str(res)
    # (b) raw slice differs only by a trailing space → the raw-slice gate
    # must NOT utter the blank-line sentence (it would be false)
    f.write_text("alpha \nbeta\n")
    res = await _edit(ws, "d.txt", "alpha\n\n", "x")
    assert "ends with" not in str(res), res
    assert "different indentation" in str(res)
    # (c) the file DOES have blank lines after the block: the count is stated
    f.write_text("x\n\n\ny\n")
    res = await _edit(ws, "d.txt", "x\n\n\n\n\n", "x2")
    assert "has only 2 after line 1" in str(res), res
    # (d) whitespace-only trailing line in old vs empty ones in the file
    res = await _edit(ws, "d.txt", "x\n  \n", "x2")
    assert "carry whitespace" in str(res), res


async def test_replace_read_identity_mismatch_is_the_guards_verdict(ws, home, monkeypatch):
    """[R6] the non-streaming read's identity mismatch came back as a bare
    "Error: failed to read … [Errno 70]" string with no reason code, while
    the same swap one open earlier was the rejected outcome."""
    import sys as _sys
    (ws / "d").mkdir()
    (ws / "d2").mkdir()
    (ws / "d" / "a.py").write_text(PY_SRC)
    (ws / "d2" / "a.py").write_text(PY_SRC)
    real_open = os.open
    state = {"done": False}

    def _swap_for_reader(p, flags, *a, **k):
        if (not state["done"] and str(p).endswith("a.py")
                and _sys._getframe(1).f_code.co_name == "_read_text_guarded"):
            state["done"] = True
            os.rename(ws / "d", ws / "tmp")
            os.rename(ws / "d2", ws / "d")
        return real_open(p, flags, *a, **k)
    monkeypatch.setattr(fs.os, "open", _swap_for_reader)
    res = await tool_replace_text("d/a.py", "def f():\n    return 1\n", "def f():\n    return 2\n", ws)
    assert state["done"]
    assert _rc(res) == "file_changed_underneath", res
    (row,) = read_ledger(home=home)
    assert row["reason"] == "file_changed_underneath"
