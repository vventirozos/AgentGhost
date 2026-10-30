"""§4SD — file_system writes retry EAGAIN/EBUSY instead of handing the model
a failure. Req d886d047: an `edit` of index.html on the /workspace bind mount
failed `[Errno 35] Resource temporarily unavailable`; the model spent a turn
retrying it by hand and the same edit landed 24 s later.

Each test names the world it fails in.
"""
import errno
import os

import pytest

from ghost_agent.tools import file_system as fs
from ghost_agent.tools.file_system import tool_file_system


@pytest.fixture
def ws(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("GHOST_HOME", str(h))
    monkeypatch.setattr(fs, "_TRANSIENT_IO_BACKOFF_S", (0, 0, 0, 0, 0))
    return h


def _flaky(monkeypatch, name, *, fails, err=errno.EAGAIN, when=lambda *a: True):
    """Make ``os.<name>`` raise ``err`` for its first ``fails`` matching calls."""
    real = getattr(os, name)
    calls = {"n": 0, "failed": 0}

    def fake(*a, **kw):
        if when(*a):
            calls["n"] += 1
            if calls["failed"] < fails:
                calls["failed"] += 1
                raise OSError(err, os.strerror(err))
        return real(*a, **kw)

    monkeypatch.setattr(os, name, fake)
    return calls


def _opens_for_write(path):
    return lambda p, flags, *a: str(p) == str(path) and flags & os.O_WRONLY


async def test_edit_open_eagain_is_retried_and_lands(ws, monkeypatch):
    # world: the retry is missing — the edit fails with Errno 35 as in d886d047
    f = ws / "index.html"
    f.write_text("<link href=fonts>\n<p>hi</p>\n")
    calls = _flaky(monkeypatch, "open", fails=2, when=_opens_for_write(f))
    res = await tool_file_system(operation="edit", path="index.html", sandbox_dir=ws,
                                 old_string="<link href=fonts>\n", new_string="")
    assert "SUCCESS" in str(res), res
    assert f.read_text() == "<p>hi</p>\n"
    assert calls["n"] == 3


async def test_edit_write_eagain_after_truncate_keeps_whole_content(ws, monkeypatch):
    # world: the WHOLE writer is re-run after its own truncate — the stamp check
    # sees a changed file and refuses, or the file is left empty
    f = ws / "a.py"
    f.write_text("x = 1\ny = 2\n")
    _flaky(monkeypatch, "write", fails=1, when=lambda fd, *a: not isinstance(fd, str) and fd > 2)
    res = await tool_file_system(operation="edit", path="a.py", sandbox_dir=ws,
                                 old_string="y = 2", new_string="y = 3")
    assert "SUCCESS" in str(res), res
    assert f.read_text() == "x = 1\ny = 3\n"


async def test_edit_ftruncate_ebusy_is_retried(ws, monkeypatch):
    # world: EBUSY left out of the retried set
    f = ws / "a.js"
    f.write_text("let a = 1;\n")
    calls = _flaky(monkeypatch, "ftruncate", fails=1, err=errno.EBUSY)
    res = await tool_file_system(operation="edit", path="a.js", sandbox_dir=ws,
                                 old_string="1", new_string="2")
    assert "SUCCESS" in str(res), res
    assert f.read_text() == "let a = 2;\n"
    assert calls["n"] == 2


async def test_write_rename_eagain_is_retried_and_leaves_no_tmp(ws, monkeypatch):
    # world: the write op is not wrapped, or a retried attempt leaks its .tmp
    f = ws / "style.css"                        # new file: a write over an unread one is refused
    calls = _flaky(monkeypatch, "replace", fails=2, when=lambda s, d, *a: str(d) == str(f))
    res = await tool_file_system(operation="write", path="style.css", sandbox_dir=ws,
                                 content="body { color: red; }\n")
    assert "SUCCESS" in str(res), res
    assert f.read_text().startswith("body { color: red; }")
    assert calls["n"] == 3
    assert not [p for p in ws.iterdir() if p.suffix == ".tmp"]


async def test_append_rename_eagain_is_retried_once_not_doubled(ws, monkeypatch):
    # world: append is not wrapped, or a retry appends the part twice
    f = ws / "os.js"
    f.write_text("const a = 1;\n")
    _flaky(monkeypatch, "replace", fails=1, when=lambda s, d, *a: str(d) == str(f))
    res = await tool_file_system(operation="write", path="os.js", sandbox_dir=ws,
                                 content="const b = 2;\n", append=True)
    assert "SUCCESS" in str(res), res
    assert f.read_text() == "const a = 1;\nconst b = 2;\n"
    assert not [p for p in ws.iterdir() if p.suffix == ".tmp"]


async def test_other_errnos_are_not_retried(ws, monkeypatch):
    # world: the helper retries every OSError (EACCES on a 0444 file would be
    # hammered six times and still fail)
    f = ws / "a.py"
    f.write_text("x = 1\n")
    calls = _flaky(monkeypatch, "open", fails=99, err=errno.EACCES, when=_opens_for_write(f))
    res = await tool_file_system(operation="edit", path="a.py", sandbox_dir=ws,
                                 old_string="1", new_string="2")
    assert "SUCCESS" not in str(res)
    assert calls["n"] == 1
    assert f.read_text() == "x = 1\n"


async def test_persistent_eagain_gives_up_bounded_and_file_unchanged(ws, monkeypatch):
    # world: an unbounded retry loop wedges the worker thread forever
    f = ws / "a.py"
    f.write_text("x = 1\n")
    calls = _flaky(monkeypatch, "open", fails=99, when=_opens_for_write(f))
    res = await tool_file_system(operation="edit", path="a.py", sandbox_dir=ws,
                                 old_string="1", new_string="2")
    assert "unchanged on disk" in str(res), res
    assert calls["n"] == len(fs._TRANSIENT_IO_BACKOFF_S) + 1
    assert f.read_text() == "x = 1\n"


def test_retry_helper_returns_value_and_reraises_non_oserror():
    # world: the helper swallows the return value or non-OSError exceptions
    assert fs._retry_transient(lambda a, b=0: a + b, 2, b=3) == 5
    with pytest.raises(fs._FileChangedUnderneath):
        fs._retry_transient(lambda: (_ for _ in ()).throw(fs._FileChangedUnderneath()))


def _big_body(marker):
    pad = ("# pad " + "x" * 90 + "\n") * 12000              # > 1 MB: the streaming replace path
    return pad + marker + "\n" + pad


async def test_streaming_replace_open_eagain_is_retried(ws, monkeypatch):
    # world: the >1 MB line-by-line replace is not wrapped
    from ghost_agent.tools.file_system import tool_replace_text
    f = ws / "big.py"
    f.write_text(_big_body("VALUE = 1"))
    calls = _flaky(monkeypatch, "open", fails=1,
                   when=lambda p, flags, *a: str(p) == str(f) and flags & os.O_RDWR)
    res = await tool_replace_text("big.py", "VALUE = 1", "VALUE = 2", ws)
    assert "Streaming replace" in str(res), res
    assert "VALUE = 2" in f.read_text() and calls["n"] == 2
    assert not [p for p in ws.iterdir() if p.suffix == ".tmp"]


async def test_streaming_replace_rename_eagain_is_retried(ws, monkeypatch):
    # world: the streaming commit's rename is not wrapped
    from ghost_agent.tools.file_system import tool_replace_text
    f = ws / "big.py"
    f.write_text(_big_body("VALUE = 1"))
    calls = _flaky(monkeypatch, "replace", fails=1, when=lambda s, d, *a: str(d) == str(f))
    res = await tool_replace_text("big.py", "VALUE = 1", "VALUE = 2", ws)
    assert "Streaming replace" in str(res), res
    assert "VALUE = 2" in f.read_text() and calls["n"] == 2


# ── §4SD review: O_NONBLOCK is cleared once the fd is a regular file ──────

import ast
import pathlib

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
_NONBLOCK_MODULES = ["tools/file_system.py", "api/routes.py", "core/project_research.py",
                     "core/task_deliverable.py"]


def _nonblock_functions_missing_the_switch(tree):
    """Every function whose own code names O_NONBLOCK (getattr(os,
    "O_NONBLOCK") or os.O_NONBLOCK) but never calls _block_if_regular."""
    missing = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name == "_block_if_regular":
            continue
        names = {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant)} | \
                {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        calls = {getattr(c.func, "id", getattr(c.func, "attr", None))
                 for c in ast.walk(fn) if isinstance(c, ast.Call)}
        if "O_NONBLOCK" in names and "_block_if_regular" not in calls:
            missing.append(fn.name)
    return missing


@pytest.mark.parametrize("mod", _NONBLOCK_MODULES)
def test_every_nonblocking_open_switches_back_to_blocking(mod):
    # world: a buffered read on a virtiofs EAGAIN reads as EOF — the streaming
    # replace committed a truncated file with SUCCESS (review MAJOR)
    tree = ast.parse((_SRC / mod).read_text())
    assert _nonblock_functions_missing_the_switch(tree) == []
    assert any(isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == "_block_if_regular"
               for n in ast.walk(tree)), mod


def test_the_enumeration_fires_on_a_function_without_the_switch():
    bad = ast.parse("import os\ndef f(p):\n    fd = os.open(p, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0))\n"
                    "    return os.read(fd, 9)\n")
    assert _nonblock_functions_missing_the_switch(bad) == ["f"]


def test_the_switch_clears_nonblock_on_a_regular_file_only(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    fd = os.open(str(f), os.O_RDONLY | os.O_NONBLOCK)
    try:
        assert os.get_blocking(fd) is False
        fs._block_if_regular(fd)
        assert os.get_blocking(fd) is True
    finally:
        os.close(fd)
    fifo = tmp_path / "p"
    os.mkfifo(fifo)
    fd = os.open(str(fifo), os.O_RDONLY | os.O_NONBLOCK)
    try:
        fs._block_if_regular(fd)
        assert os.get_blocking(fd) is False                     # a FIFO stays non-blocking
    finally:
        os.close(fd)


# ── §4SD review: a rename the mount performed but reported EAGAIN is done ─

def test_commit_replace_treats_a_performed_rename_as_done(tmp_path, monkeypatch):
    # world: the retry renames again → ENOENT on an applied change
    src, dst = tmp_path / "t.tmp", tmp_path / "d"
    src.write_text("new")
    dst.write_text("old")
    real = os.replace

    def did_it_then_eagain(a, b, *k, **kw):
        real(a, b)
        raise OSError(errno.EAGAIN, "EAGAIN")
    monkeypatch.setattr(os, "replace", did_it_then_eagain)
    fs._commit_replace(src, dst)
    assert dst.read_text() == "new"


def test_commit_replace_retries_a_rename_that_did_not_happen(tmp_path, monkeypatch):
    src, dst = tmp_path / "t.tmp", tmp_path / "d"
    src.write_text("new")
    calls = _flaky(monkeypatch, "replace", fails=2)
    fs._commit_replace(src, dst)
    assert dst.read_text() == "new" and calls["n"] == 3


def test_commit_replace_raises_other_errors(tmp_path, monkeypatch):
    src, dst = tmp_path / "t.tmp", tmp_path / "d"
    src.write_text("new")
    calls = _flaky(monkeypatch, "replace", fails=9, err=errno.EACCES)
    with pytest.raises(PermissionError):
        fs._commit_replace(src, dst)
    assert calls["n"] == 1


async def test_append_whose_rename_reported_eagain_lands_once(ws, monkeypatch):
    # world: the whole append re-runs after a performed rename → the part twice
    # (or refused as a duplicate though it was appended)
    f = ws / "app.js"
    f.write_text("const a = 1;\n")
    real, n = os.replace, {"k": 0}

    def did_it_then_eagain(a, b, *k, **kw):
        real(a, b)
        if str(b) == str(f) and n["k"] == 0:
            n["k"] += 1
            raise OSError(errno.EAGAIN, "EAGAIN")
    monkeypatch.setattr(os, "replace", did_it_then_eagain)
    res = await tool_file_system(operation="write", path="app.js", sandbox_dir=ws,
                                 content="const b = 2;\n", append=True)
    assert "SUCCESS" in str(res), res
    assert f.read_text() == "const a = 1;\nconst b = 2;\n"


async def test_append_stamps_the_file_once_and_never_for_a_refused_duplicate(ws, monkeypatch):
    # world: the mutation stamp moved out of the retried call runs for a refused duplicate
    f = ws / "os.js"
    part = "".join(f"const unique_value_{i} = {i};\n" for i in range(40))   # over APPEND_DUP_MIN_CHARS
    f.write_text("const a = 1;\n")
    stamped = []
    monkeypatch.setattr(fs, "_note_mutation", lambda p: stamped.append(str(p)))
    r1 = await tool_file_system(operation="write", path="os.js", sandbox_dir=ws, content=part, append=True)
    assert "SUCCESS" in str(r1) and stamped and set(stamped) == {str(f)}
    after_first = len(stamped)
    r2 = await tool_file_system(operation="write", path="os.js", sandbox_dir=ws, content=part, append=True)
    assert "already ENDS with this exact part" in str(r2)
    assert len(stamped) == after_first                          # nothing appended, nothing stamped


async def test_streaming_open_closes_its_fd_when_fstat_fails(ws, monkeypatch):
    # world: an fstat failure leaks the streaming fd (one per retry)
    from ghost_agent.tools.file_system import tool_replace_text
    f = ws / "big.py"
    f.write_text(_big_body("VALUE = 1"))
    real_open, real_fstat, rdwr = os.open, os.fstat, []

    def spy_open(p, flags, *a, **k):
        fd = real_open(p, flags, *a, **k)
        if str(p) == str(f) and flags & os.O_RDWR:
            rdwr.append(fd)
        return fd

    def bad_fstat(fd, *a, **k):
        if fd in rdwr:
            raise OSError(errno.EIO, "EIO")
        return real_fstat(fd, *a, **k)
    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "fstat", bad_fstat)
    res = await tool_replace_text("big.py", "VALUE = 1", "VALUE = 2", ws)
    monkeypatch.setattr(os, "fstat", real_fstat)
    assert "SUCCESS" not in str(res)
    assert rdwr, "the streaming path never opened"
    for fd in rdwr:
        with pytest.raises(OSError):
            os.fstat(fd)                                        # EBADF: closed


async def test_append_tmp_build_eagain_is_retried(ws, monkeypatch):
    # world: only the commit is retried — an EAGAIN while the temp copy is
    # built (before the commit) fails the append (battery r3 P5)
    import tempfile
    f = ws / "os.js"
    f.write_text("const a = 1;\n")
    real, n = tempfile.mkstemp, {"k": 0}

    def flaky_mkstemp(*a, **kw):
        if n["k"] == 0:
            n["k"] += 1
            raise OSError(errno.EAGAIN, "EAGAIN")
        return real(*a, **kw)
    monkeypatch.setattr(tempfile, "mkstemp", flaky_mkstemp)
    res = await tool_file_system(operation="write", path="os.js", sandbox_dir=ws,
                                 content="const b = 2;\n", append=True)
    assert n["k"] == 1
    assert "SUCCESS" in str(res), res
    assert f.read_text() == "const a = 1;\nconst b = 2;\n"


# ── review round 2: the commit proves the rename; retries do not nest ──

def test_a_rename_that_did_not_happen_is_not_called_done_when_lstat_fails(tmp_path, monkeypatch):
    # world: `lexists(tmp)` is False on ANY lstat error — EAGAIN on the rename
    # plus a failing lstat reported SUCCESS with the target unchanged
    src, dst = tmp_path / "t.tmp", tmp_path / "d"
    src.write_text("new")
    dst.write_text("old")
    real_lstat, seen = os.lstat, {"n": 0}

    def flaky_lstat(p, *a, **kw):
        seen["n"] += 1
        if seen["n"] > 1 and str(p) == str(src):            # the first is the identity read
            raise OSError(errno.EIO, "EIO")
        return real_lstat(p, *a, **kw)
    monkeypatch.setattr(os, "lstat", flaky_lstat)
    _flaky(monkeypatch, "replace", fails=99)
    with pytest.raises(OSError) as ei:
        fs._commit_replace(src, dst)
    assert ei.value.errno == errno.EAGAIN
    assert dst.read_text() == "old" and src.exists()


def test_a_landed_rename_the_mount_still_showed_is_done_not_enoent(tmp_path, monkeypatch):
    # world: the retry's ENOENT on an applied rename is a false "Error"
    src, dst = tmp_path / "t.tmp", tmp_path / "d"
    src.write_text("new")
    dst.write_text("old")
    real_replace, real_lstat, st = os.replace, os.lstat, {"stale": True}

    def landed_then_eagain(a, b, *k, **kw):
        real_replace(a, b)
        raise OSError(errno.EAGAIN, "EAGAIN")

    def stale_lstat(p, *a, **kw):
        if str(p) == str(src) and st["stale"] and not os.path.exists(src):
            st["stale"] = False                             # the mount shows it once more
            return real_lstat(dst)
        return real_lstat(p, *a, **kw)
    monkeypatch.setattr(os, "replace", landed_then_eagain)
    monkeypatch.setattr(os, "lstat", stale_lstat)
    fs._commit_replace(src, dst)
    assert dst.read_text() == "new"


def test_a_tmp_gone_with_the_old_file_still_in_place_is_not_done(tmp_path, monkeypatch):
    # world: "tmp gone" is taken as "renamed" — the tmp vanished but the target
    # is still the old file, and the write reported SUCCESS over unchanged bytes
    src, dst = tmp_path / "t.tmp", tmp_path / "d"
    src.write_text("new")
    dst.write_text("old")

    def lost_tmp_then_eagain(a, b, *k, **kw):
        if os.path.exists(a):
            os.unlink(a)                                    # gone, never renamed
        raise OSError(errno.EAGAIN, "EAGAIN")
    monkeypatch.setattr(os, "replace", lost_tmp_then_eagain)
    with pytest.raises(OSError):
        fs._commit_replace(src, dst)
    assert dst.read_text() == "old"


async def test_a_failing_commit_is_not_rerun_by_the_writers_retry(ws, monkeypatch):
    # world: nested retries — six whole writes of six renames (~27 s, six tmps)
    f = ws / "os.js"
    calls = _flaky(monkeypatch, "replace", fails=99, when=lambda s, d, *a: str(d) == str(f))
    res = await tool_file_system(operation="write", path="os.js", sandbox_dir=ws, content="const a = 1;\n")
    assert "SUCCESS" not in str(res), res
    assert calls["n"] == len(fs._TRANSIENT_IO_BACKOFF_S) + 1
    assert not [p for p in ws.iterdir() if p.suffix == ".tmp"]


@pytest.mark.parametrize("writer", ["write_text_nofollow", "write_text_nofollow_in_dir"])
def test_the_nofollow_writers_close_their_fd_when_the_switch_fails(tmp_path, monkeypatch, writer):
    # world: an fstat/set_blocking failure after the open leaks the fd
    closed, opened = [], []
    real_open, real_close = os.open, os.close

    def spy_open(*a, **kw):
        fd = real_open(*a, **kw)
        opened.append(fd)
        return fd

    def spy_close(fd):
        closed.append(fd)
        return real_close(fd)
    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "close", spy_close)
    monkeypatch.setattr(os, "set_blocking", lambda fd, b: (_ for _ in ()).throw(OSError(errno.EIO, "EIO")))
    with pytest.raises(OSError):
        if writer == "write_text_nofollow":
            fs.write_text_nofollow(tmp_path / "x.txt", "hi")
        else:
            fs.write_text_nofollow_in_dir(tmp_path, "x.txt", "hi")
    assert opened and set(opened) <= set(closed), (opened, closed)


async def test_an_edit_read_open_eagain_is_retried(ws, monkeypatch):
    # world: only the write side retries — EAGAIN on the edit's read fails it
    f = ws / "index.html"
    f.write_text("<link href=fonts>\n<p>hi</p>\n")
    calls = _flaky(monkeypatch, "open", fails=2,
                   when=lambda p, flags, *a: str(p) == str(f.resolve()) and not flags & (os.O_WRONLY | os.O_RDWR))
    res = await tool_file_system(operation="edit", path="index.html", sandbox_dir=ws,
                                 old_string="<link href=fonts>\n", new_string="")
    assert "SUCCESS" in str(res), res
    assert calls["failed"] == 2
    assert f.read_text() == "<p>hi</p>\n"


# ── second review: the read side's chunk reads and opens retry too ────────────

@pytest.mark.parametrize("reader", ["regular", "guarded"])
def test_a_transient_read_error_is_retried_not_returned_as_a_short_file(tmp_path, monkeypatch, reader):
    # world: EAGAIN on a chunk read fails the edit (or reads it short)
    f = tmp_path / "a.txt"
    f.write_text("hello world\n" * 100)
    calls = _flaky(monkeypatch, "read", fails=2)
    if reader == "regular":
        data, _st = fs._read_regular_nofollow(f)
        assert data == f.read_bytes()
    else:
        assert fs._read_text_guarded(f, os.stat(f)) == f.read_text()
    assert calls["failed"] == 2


def test_the_guarded_reads_open_is_retried(tmp_path, monkeypatch):
    # world: replace's by-path re-read fails on a transient open
    f = tmp_path / "a.txt"
    f.write_text("x\n")
    real = os.path.realpath(f)
    calls = _flaky(monkeypatch, "open", fails=2, when=lambda p, *a: os.path.realpath(p) == real)
    assert fs._read_text_guarded(f, os.stat(f)) == "x\n"
    assert calls["failed"] == 2


def test_the_commits_identity_read_is_retried(tmp_path, monkeypatch):
    # world: the streaming replace's commit (no outer retry) fails the edit on
    # a transient lstat of its own tmp
    src, dst = tmp_path / "t.tmp", tmp_path / "d"
    src.write_text("new")
    dst.write_text("old")
    calls = _flaky(monkeypatch, "lstat", fails=2, when=lambda p, *a: str(p) == str(src))
    fs._commit_replace(src, dst)
    assert dst.read_text() == "new" and calls["failed"] == 2


def test_a_failing_identity_read_is_marked_retried(tmp_path, monkeypatch):
    # world (review round 4): the lstat's give-up escaped the mark, and the
    # writers' whole-step retry nested it (6 × 6 attempts, ~27 s)
    src, dst = tmp_path / "t.tmp", tmp_path / "d"
    src.write_text("new")
    _flaky(monkeypatch, "lstat", fails=99, when=lambda p, *a: str(p) == str(src))
    with pytest.raises(OSError) as ei:
        fs._commit_replace(src, dst)
    assert getattr(ei.value, "_ghost_retried", False) is True
