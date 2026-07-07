"""file_system replace bad-byte write-back corruption (PROJECT_JOURNAL §4B).

`tool_replace_text` used to read a mostly-text file with `errors="replace"` then
write the whole thing back — so every stray non-UTF-8 byte was persisted as
U+FFFD, corrupting regions the edit never touched. The fix is an
`errors="surrogateescape"` round-trip through the read + the shared guarded
write + the streaming path, so untouched bad bytes come back byte-identical.
"""
import pytest

from ghost_agent.tools.file_system import tool_replace_text, _syntax_regression

STREAMING_THRESHOLD = 1 * 1024 * 1024  # mirrors the function-local in tool_replace_text


@pytest.fixture
def sandbox(tmp_path):
    d = tmp_path / "sandbox"
    d.mkdir()
    return d


async def test_untouched_bad_bytes_survive_replace(sandbox):
    target = sandbox / "data.txt"
    # A mostly-text file with a real bad byte OUTSIDE the edited region.
    target.write_bytes(b"line one OK\nbad byte here: \x80 keep me\nEDIT THIS line\n")
    res = await tool_replace_text("data.txt", "EDIT THIS", "EDITED", sandbox)
    after = target.read_bytes()
    # Edit applied, bad byte preserved byte-for-byte, no U+FFFD corruption.
    assert b"EDITED" in after
    assert b"\x80" in after, "bad byte was corrupted (regression to errors='replace')"
    assert b"\xef\xbf\xbd" not in after, "U+FFFD appeared — bad byte was lossily replaced"
    assert b"bad byte here: \x80 keep me" in after


async def test_bad_byte_in_edited_line_still_works(sandbox):
    target = sandbox / "d2.txt"
    target.write_bytes(b"prefix \x81 SEARCHME suffix\n")
    res = await tool_replace_text("d2.txt", "SEARCHME", "FOUND", sandbox)
    after = target.read_bytes()
    assert b"FOUND" in after
    assert b"\x81" in after


async def test_streaming_path_preserves_bad_bytes(sandbox):
    # Force the streaming path: file > STREAMING_THRESHOLD, single-line search.
    target = sandbox / "big.txt"
    filler = ("x" * 100 + "\n").encode() * (STREAMING_THRESHOLD // 100 + 50)
    target.write_bytes(b"top \x80 keep\n" + filler + b"REPLACE_TARGET\n")
    assert target.stat().st_size > STREAMING_THRESHOLD
    res = await tool_replace_text("big.txt", "REPLACE_TARGET", "DONE", sandbox)
    after = target.read_bytes()
    assert b"DONE" in after
    assert b"top \x80 keep" in after
    assert b"\xef\xbf\xbd" not in after


async def test_py_file_with_bad_byte_does_not_crash(sandbox):
    # A .py file with a bad byte in a comment → the surrogate makes ast.parse
    # raise UnicodeEncodeError (not SyntaxError); the regression guard must fail
    # open, not crash the replace.
    target = sandbox / "mod.py"
    target.write_bytes(b"x = 1  # note \x80 here\nOLD_NAME = 2\n")
    res = await tool_replace_text("mod.py", "OLD_NAME", "NEW_NAME", sandbox)
    after = target.read_bytes()
    assert b"NEW_NAME" in after
    assert b"\x80" in after
    assert "error" not in res.lower() or "SUCCESS" in res


def test_syntax_regression_fails_open_on_surrogate():
    # new_content carrying a lone surrogate (from a bad byte) must not raise.
    new = "def f():\n    pass  # \udc80\n"
    assert _syntax_regression(new, new, "x.py") == ""
