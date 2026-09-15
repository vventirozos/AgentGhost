"""§4GK round 7 — the defects a read-only reviewer confirmed by execution in
the workspace restore, the shutdown hook and the handheld client.

ONE THEME, again: **the fix was written for the case that was measured.**

* "report, never raise" was implemented for EACCES/EPERM. Every other errno
  a real archive can produce — EEXIST, ENOTDIR, EISDIR, ENAMETOOLONG, ENOSPC
  — still reached the route's generic handler, i.e. a 500 AFTER the wipe. A
  zip holding both `sandbox/a` (a file) and `sandbox/a/b.txt` is a
  one-upload destroy-the-workspace.
* the repair could not repair what it could not OPEN: 0o555 and 0o444 healed,
  0o333/0o111/0o000 did not — and `_unfreeze_chain` skipped the sandbox ROOT,
  which is the only ancestor a top-level member has.
* four `continue` paths dropped a member and answered `unrestored: []`, so
  every client printed "loaded successfully" over files that are not there.
* `threading._register_atexit` bought the hook its ordering and gave up
  `atexit`'s per-callback isolation: `threading._shutdown` runs its callbacks
  with no try/except, so one raise skips `_python_exit` and the whole
  non-daemon join loop under it.
* and the handheld client's LOAD side was never migrated to the fields round
  6 added, so it printed "workspace restored." over an incomplete restore.

Nothing here asserts on source text: every pin drives a real handler, a real
interpreter shutdown, or the client's own function, extracted and executed
(PyQt6 is not installed in this venv — the client runs on the handheld).
"""
import ast
import errno
import io
import json
import os
import stat as _stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ghost_agent.api import routes as R

_ROOT = Path(__file__).resolve().parents[1]
_CLIENT = _ROOT / "interface" / "externals" / "clockwork_ghost" / "client.py"


# ── doubles (the shapes the round-4/6 files drive the routes with) ───────────

class _Req:
    def __init__(self, raw=b""):
        self._raw = raw
        self.headers = {}
        self.app = MagicMock()

    async def body(self):
        return self._raw

    async def json(self):
        return json.loads(self._raw or b"{}")


class _Upload:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    async def read(self, size=-1):
        if size is None or size < 0:
            chunk, self._pos = self._data[self._pos:], len(self._data)
            return chunk
        chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk


def _ws_agent(sandbox: Path):
    agent = MagicMock()
    agent.context.sandbox_dir = sandbox
    agent.context.scratchpad = None
    return agent


async def _save_bytes(sandbox: Path) -> bytes:
    with patch.object(R, "get_agent", return_value=_ws_agent(sandbox)):
        resp = await R.save_workspace(_Req(raw=b""))
    assert isinstance(resp, R.FileResponse), (
        f"the save answered {getattr(resp, 'status_code', resp)}")
    data = Path(resp.path).read_bytes()
    os.unlink(resp.path)
    return data


async def _load(sandbox: Path, zip_bytes: bytes):
    with patch.object(R, "get_agent", return_value=_ws_agent(sandbox)):
        return await R.load_workspace(_Req(raw=b""), _Upload(zip_bytes))


def _zip_of(members) -> bytes:
    """An archive built by hand — the shapes a save cannot produce but an
    uploader can.

    `members` is an ORDERED sequence of (name, payload); a payload of None
    writes a DIRECTORY entry. The order is the archive's order, and the
    archive's order is what decides what already exists when each member
    lands — which is the whole subject of the first test below.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, body in members:
            if body is None:
                zf.writestr(zipfile.ZipInfo(
                    name if name.endswith("/") else name + "/"), b"")
            else:
                zf.writestr(name, body)
    return buf.getvalue()


def _chmod_back(path: Path, mode=0o755):
    """pytest's tmp_path cleanup cannot delete through a frozen directory."""
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_dir() and not p.is_symlink():
            p.chmod(mode)
    path.chmod(mode)


# ═══════════════════════════════════════════════════════════════════════════
# 1. "report, never raise" — for EVERY errno, not the two that were measured
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_zip_carrying_a_file_and_a_directory_of_the_same_name(tmp_path):
    """THE one-upload destroy-the-workspace, driven through the real route.

    `sandbox/a` as a FILE and `sandbox/a/b.txt` under it: `mkdir(parents=
    True, exist_ok=True)` raises EEXIST for `a/` (exist_ok covers an existing
    DIRECTORY, not an existing file) and ENOTDIR one level deeper. Round 6's
    `_mkdir_unfreezing` re-raised every errno but EACCES/EPERM, so this
    reached the route's generic handler: HTTP 500 with an error id, AFTER
    the wipe, with no `unrestored` naming the member that did it. The
    archive is not exotic — one zipper storing a file and a directory under
    one name produces it.

    World where it fails: `if exc.errno not in (EACCES, EPERM): raise`
    (HTTPException 500 out of the route, and the sandbox already empty).
    """
    live = tmp_path / "live"
    live.mkdir()
    (live / "keep-me.txt").write_text("the old workspace")
    archive = _zip_of([
        ("sandbox/a", b"I am a file"),
        ("sandbox/a/b.txt", b"and I am under it"),
        ("sandbox/a/deep/c.txt", b"deeper still"),
        # ...and the DIRECTORY branch of the same call: both arms of
        # `_mkdir_unfreezing` feed `unrestored`, and neither had a pin
        # (§4GK round 7).
        ("sandbox/a/subdir/", None),
        ("sandbox/fine.txt", b"unrelated member"),
    ])

    result = await _load(live, archive)

    assert result["status"] == "success", result
    # the members that CAN land, do — one bad member is not the archive
    assert (live / "a").read_text() == "I am a file"
    assert (live / "fine.txt").read_text() == "unrelated member"
    failed = {u["path"]: u["reason"] for u in result["unrestored"]}
    assert set(failed) == {"a/b.txt", "a/deep/c.txt", "a/subdir/"}, failed
    assert failed["a/b.txt"].startswith("EEXIST"), failed
    assert failed["a/deep/c.txt"].startswith("ENOTDIR"), failed
    assert failed["a/subdir/"].startswith("ENOTDIR"), failed

    # AND AT THE HELPER'S OWN CALL SITE. The route's per-member handler
    # would cover for a `_mkdir_unfreezing` that still raised — masking a
    # defect is not fixing it, and the next caller of this helper will not
    # necessarily sit inside that try. Its signature is the contract: "the
    # reason it could not, or None", for every errno.
    assert R._mkdir_unfreezing(live / "a", live).startswith("EEXIST")
    assert R._mkdir_unfreezing(live / "a" / "deep", live).startswith("ENOTDIR")
    assert R._mkdir_unfreezing(live / "new" / "deep", live) is None
    assert (live / "new" / "deep").is_dir()


@pytest.mark.parametrize("member,before,expect", [
    # a file member whose name the archive ALSO carries as a directory,
    # written FIRST — so the write lands on a directory
    ("sandbox/adir", (("sandbox/adir/", None),), "EISDIR"),
    # ...and a name this filesystem cannot hold at all. This one is raised by
    # the symlink CHECK (`lstat`), not by the write — which is why the
    # per-member body reports rather than only the two calls that had
    # handlers.
    ("sandbox/" + "n" * 300, (), "ENAMETOOLONG"),
])
async def test_a_member_the_filesystem_refuses_is_reported_not_raised(
        tmp_path, member, before, expect):
    """The file half of the same rule. `_restore_file_member` raises EISDIR
    when a member's name is an existing directory, and a 300-byte component
    answers ENAMETOOLONG before the write is even attempted — both re-raised
    past the EACCES/EPERM arm into the generic 500, after the wipe.

    World where it fails: `if exc.errno not in (EACCES, EPERM): raise` and a
    member loop with no handler of its own (HTTPException 500, no
    `unrestored`, sandbox already empty).
    """
    live = tmp_path / "live"
    live.mkdir()
    archive = _zip_of([*before, (member, b"payload"),
                       ("sandbox/ok.txt", b"fine")])

    result = await _load(live, archive)

    assert result["status"] == "success", result
    assert (live / "ok.txt").read_text() == "fine", (
        "one refused member took the rest of the restore with it")
    assert [u["path"] for u in result["unrestored"]] == \
        [member[len("sandbox/"):]], result["unrestored"]
    assert result["unrestored"][0]["reason"].startswith(expect), \
        result["unrestored"]


async def test_the_reason_on_the_wire_carries_no_HOST_path(tmp_path):
    """`str(exc)` on an OSError renders the filename it carries: the
    sandbox's absolute location on the host, its id, and the server's user —
    into `unrestored[].reason`, four lines above the handler whose comment
    says "Don't leak internal exception text to the client".

    The member's own RELATIVE path is already in `unrestored[].path`.

    World where it fails: `{"reason": str(exc)}` — the assertion below finds
    the sandbox's absolute path in the response.
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / "b.txt").write_text("B")
    archive = await _save_bytes(src)

    live = tmp_path / "live"
    live.mkdir()
    real_open = os.open

    def refuse_b(path, flags, *a, **k):
        if str(path).endswith("/b.txt") and (flags & os.O_CREAT):
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_open(path, flags, *a, **k)

    with patch.object(R.os, "open", refuse_b):
        result = await _load(live, archive)

    reason = result["unrestored"][0]["reason"]
    assert result["unrestored"][0]["path"] == "b.txt"
    assert "Permission denied" in reason, reason      # the operator's half
    assert str(live) not in reason and str(tmp_path) not in reason, reason
    assert "/" not in reason, f"a path leaked into the wire format: {reason}"
    assert reason.startswith("EACCES"), reason


# ═══════════════════════════════════════════════════════════════════════════
# 2. the repair reaches what it could not open, and the root it skipped
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_top_level_directory_with_no_READ_bit_is_still_wiped(tmp_path):
    """`_add_owner_access` hangs off `os.open(O_RDONLY)`, so it repaired
    exactly the modes that still grant the owner read: 0o555 and 0o444 heal,
    0o333/0o111/0o000 fail at the `open` and return False before a bit is
    changed. `chmod 111 dist` from inside the sandbox is one command.

    Measured pre-fix: `dist/` survives the wipe with its stale contents and
    is presented as the restored workspace — fault 1 of the round-6 comment,
    unfixed for a whole class of modes.

    World where it fails: an `_add_owner_access` that gives up when `open`
    answers EACCES (`live/dist/stale.js` is still there afterwards).
    """
    live = tmp_path / "live"
    dist = live / "dist"
    dist.mkdir(parents=True)
    (dist / "stale.js").write_text("// from the old workspace\n")
    (live / "notes.txt").write_text("old")
    dist.chmod(0o111)                    # --x, no read: the round-6 blind spot

    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "notes.txt").write_text("new")
    try:
        result = await _load(live, await _save_bytes(clean))

        assert result["status"] == "success", result
        assert result["not_cleared"] == [], result["not_cleared"]
        assert not (live / "dist").exists(), (
            "a 0o111 directory survived the wipe: its stale contents are now "
            "part of a workspace the archive never described")
        assert (live / "notes.txt").read_text() == "new"
    finally:
        _chmod_back(live)


async def test_a_frozen_sandbox_ROOT_is_repaired_for_a_top_level_member(tmp_path):
    """`_unfreeze_chain` walks `rel.parts` — which is EMPTY for a member
    sitting directly in the sandbox, the overwhelmingly common case. So
    `touched` stayed False, the caller's one retry never ran, and the only
    ancestor a top-level member has was the one the repair could not reach.
    `chmod 555 /workspace` from inside the container is the whole recipe.

    The wipe runs BEFORE the repair, so the old file is still there and is
    REPORTED in `not_cleared` — that is the honest outcome, and it is what
    makes this pin readable: the restore is incomplete and says so, instead
    of being incomplete and silent.

    World where it fails: `cur, touched = stop.resolve(), False`
    (`new.txt` is unrestored and never written).
    """
    live = tmp_path / "live"
    live.mkdir()
    (live / "old.txt").write_text("the old workspace")
    live.chmod(0o555)                    # r-x: cannot create or unlink inside

    src = tmp_path / "src"
    src.mkdir()
    (src / "new.txt").write_text("the new workspace")
    try:
        result = await _load(live, await _save_bytes(src))

        assert result["status"] == "success", result
        assert result["unrestored"] == [], result["unrestored"]
        assert (live / "new.txt").read_text() == "the new workspace", (
            "the sandbox root was frozen and the repair skipped it, so a "
            "top-level member could not be written")
        # the wipe ran while the root was still frozen: said out loud
        assert [n.split(" ")[0] for n in result["not_cleared"]] == ["old.txt"], \
            result["not_cleared"]
    finally:
        _chmod_back(live)


async def test_the_unfreeze_walks_the_whole_tree_not_just_its_root(tmp_path):
    """`_unfreeze_tree` is `_add_owner_access(root)` PLUS an `os.walk` that
    repairs every directory under it, and only the first half had a pin: a
    nested frozen directory is what a build tool actually leaves behind
    (`dist/assets/`), and `rmtree` cannot unlink inside it either.

    World where it fails: an `_unfreeze_tree` that stops at `root`
    (`dist/nested/deep/stale.js` survives the wipe).
    """
    live = tmp_path / "live"
    deep = live / "dist" / "nested" / "deep"
    deep.mkdir(parents=True)
    (deep / "stale.js").write_text("// old\n")
    for d in (deep, deep.parent, deep.parent.parent):
        d.chmod(0o555)

    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "notes.txt").write_text("new")
    try:
        result = await _load(live, await _save_bytes(clean))

        assert result["status"] == "success", result
        assert result["not_cleared"] == [], result["not_cleared"]
        assert not (live / "dist").exists(), (
            "a frozen directory NESTED under the top-level one survived the "
            "wipe — only the root was unfrozen")
    finally:
        _chmod_back(live)


def test_the_repair_never_chmods_THROUGH_a_symlink(tmp_path):
    """The security property with no pin: `_add_owner_access` opens
    `O_NOFOLLOW` precisely so that a symlinked component — which the model
    can plant anywhere in its own sandbox — is not chmod'ed through to
    whatever it points at. Dropping that flag turns a workspace restore into
    a permissions rewrite of an arbitrary directory OUTSIDE the sandbox, and
    the round-6 suite had nothing that would notice.

    Both arms: the fast path (the link's target is readable, so `open` would
    succeed and follow) and the round-7 fallback (the target is 0o111, so the
    fast path fails and the parent-fd repair takes over — `lstat` must refuse
    the link there too).

    World where it fails: `os.open(path, O_RDONLY | O_DIRECTORY)` without
    `O_NOFOLLOW`, or a fallback that chmods the path without `lstat`-ing it
    (the outside directory comes back 0o7xx).
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    link = sandbox / "dist"
    link.symlink_to(outside)

    for mode in (0o555, 0o111, 0o000):
        outside.chmod(mode)

        assert R._add_owner_access(link) is False, oct(mode)

        assert _stat.S_IMODE(outside.stat().st_mode) == mode, (
            f"the repair followed a symlink and chmod'ed a directory OUTSIDE "
            f"the sandbox: {oct(mode)} -> "
            f"{oct(_stat.S_IMODE(outside.stat().st_mode))}")
    outside.chmod(0o755)

    # INVERSE: a REAL directory in the same position is repaired, in both
    # arms — this must not become "the repair never does anything".
    for mode in (0o555, 0o111, 0o000):
        real = sandbox / f"real{mode}"
        real.mkdir()
        real.chmod(mode)
        assert R._add_owner_access(real) is True, oct(mode)
        assert _stat.S_IMODE(real.stat().st_mode) & 0o700 == 0o700, oct(mode)


# ═══════════════════════════════════════════════════════════════════════════
# 3. a refused member is a MISSING member
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_refused_member_is_named_not_silently_dropped(tmp_path):
    """Four `continue` paths dropped a member without recording it, so the
    route answered `{"status": "success", "unrestored": []}` over files that
    are NOT on disk — and every client believes that shape (app.js prints
    "Workspace loaded successfully"). The zip-slip arm is the loudest thing
    an archive can contain and it was the quietest thing in the response.

    World where it fails: a bare `continue` (`unrestored == []`, and the
    operator is told the load was clean).
    """
    live = tmp_path / "live"
    live.mkdir()
    archive = _zip_of([
        ("sandbox/../escaped.txt", b"zip slip"),
        ("sandbox/trap.txt", b"a link is sitting at the destination"),
        ("sandbox/fine.txt", b"fine"),
    ])
    # The destination-is-a-link arm guards the window AFTER `.resolve()`, so
    # the only honest way to stand in it is to answer the check the way the
    # filesystem would at that instant.
    real_is_symlink = Path.is_symlink

    def link_at_the_destination(self):
        return self.name == "trap.txt" or real_is_symlink(self)

    with patch.object(Path, "is_symlink", link_at_the_destination):
        result = await _load(live, archive)

    assert result["status"] == "success", result
    assert (live / "fine.txt").read_text() == "fine"
    refused = {u["path"]: u["reason"] for u in result["unrestored"]}
    assert set(refused) == {"../escaped.txt", "trap.txt"}, refused
    assert "zip slip" in refused["../escaped.txt"]
    assert "symlink" in refused["trap.txt"]
    assert not (tmp_path / "escaped.txt").exists()
    assert not (live / "trap.txt").exists(), "the refusal wrote anyway"


async def test_a_member_whose_destination_is_a_link_planted_mid_restore(tmp_path):
    """The ELOOP arms: a link that appears between the check and the open,
    and the one that appears during the RETRY after a chain repair. Both
    refused (correctly) and both silent (not correctly) — the bytes are not
    on disk either way.

    Driven by planting the link inside `os.open`, which is the only place
    the window really exists.

    World where it fails: the two bare `continue`s on ELOOP.
    """
    live = tmp_path / "live"
    live.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    archive = _zip_of([("sandbox/target.txt", b"payload")])
    real_open = os.open

    def loop_on_target(path, flags, *a, **k):
        if str(path).endswith("/target.txt") and (flags & os.O_CREAT):
            raise OSError(errno.ELOOP, "Too many levels of symbolic links",
                          str(path))
        return real_open(path, flags, *a, **k)

    with patch.object(R.os, "open", loop_on_target):
        result = await _load(live, archive)

    assert result["status"] == "success", result
    assert [u["path"] for u in result["unrestored"]] == ["target.txt"], result
    assert "symlink" in result["unrestored"][0]["reason"]
    assert not (live / "target.txt").exists()


async def test_a_released_project_directory_is_reported_as_kept(tmp_path):
    """`not_cleared` had no load-bearing pin: every assertion on it was
    `== []`, so the list could have been emptied and nothing would notice.

    `projects/` is the case where a NON-empty `not_cleared` is the correct
    answer — a released project is human-attested output that a workspace
    restore does not own, and the operator has to be told it is still there
    (its contents are NOT what the archive describes).

    World where it fails: a wipe that reports nothing — the response says
    the restore was clean while the live tree holds a directory the archive
    never mentioned.
    """
    live = tmp_path / "live"
    (live / "projects" / "released").mkdir(parents=True)
    (live / "projects" / "released" / "report.md").write_text("# attested\n")
    (live / "scratch.txt").write_text("old")

    src = tmp_path / "src"
    src.mkdir()
    (src / "notes.txt").write_text("new")

    result = await _load(live, await _save_bytes(src))

    assert result["status"] == "success", result
    assert result["not_cleared"], "a kept directory was not reported at all"
    assert any(n.startswith("projects") for n in result["not_cleared"]), \
        result["not_cleared"]
    assert (live / "projects" / "released" / "report.md").read_text() == \
        "# attested\n"
    assert not (live / "scratch.txt").exists(), "the ordinary wipe stopped"


# ═══════════════════════════════════════════════════════════════════════════
# 4. an unrepresentable FUTURE mtime clamps to the CEILING (the other half)
# ═══════════════════════════════════════════════════════════════════════════

async def test_an_unrepresentable_FUTURE_mtime_clamps_to_the_zip_CEILING(tmp_path):
    """Round 6 made the fallback sign-directed and round 6's pin covered
    only the PAST half, so `dt = _ZIP_DOS_MIN_DATE_TIME` for both ends
    survived it: every far-FUTURE stamp would have been archived as 1980 —
    the same 127-year error, mirrored.

    `time.localtime(1e18)` raises OSError [Errno 22] on this host, which is
    exactly the arm under test; the stamp is injected at `os.stat` because
    the filesystem saturates long before the calendar does.

    World where it fails: a single `dt = _ZIP_DOS_MIN_DATE_TIME` fallback
    (archives `(1980, 1, 1, 0, 0, 0)`).
    """
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "distant.txt").write_text("after the calendar")
    (sandbox / "normal.txt").write_text("today")
    real_stat = os.stat

    class _Unbreakable:
        def __init__(self, st):
            self.st_mode, self.st_size = st.st_mode, st.st_size
            self.st_mtime = 1e18

    def stat_with_a_broken_clock(path, *a, **k):
        st = real_stat(path, *a, **k)
        if str(path).endswith("distant.txt"):
            return _Unbreakable(st)
        return st

    with patch.object(R.os, "stat", stat_with_a_broken_clock):
        data = await _save_bytes(sandbox)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        distant = zf.getinfo("sandbox/distant.txt").date_time
        normal = zf.getinfo("sandbox/normal.txt").date_time
    assert distant == (2107, 12, 31, 23, 59, 58), (
        f"a stamp from after the calendar was archived as {distant} — the "
        f"fallback clamps the wrong way")
    assert normal[0] >= 2020, normal      # control: real stamps untouched


# ═══════════════════════════════════════════════════════════════════════════
# 5. the shutdown hook must not take the interpreter's join loop with it
# ═══════════════════════════════════════════════════════════════════════════

_ISOLATION_SCRIPT = r'''
import os, sys, threading, time
sys.path.insert(0, {src!r})
os.environ.setdefault("GHOST_API_KEY", "x")

out = open({out!r}, "w", buffering=1)

# Registered BEFORE the module's hook, so it runs AFTER it: `threading`
# walks `_threading_atexits` in REVERSE. This stands in for
# `concurrent.futures`' own `_python_exit`, which is registered the same way
# and is the real casualty — it drains the pool and joins the workers.
threading._register_atexit(lambda: out.write("LATER HOOK RAN\n"))

import ghost_agent.api.routes as R


class _DeadLogger:
    """The logger as it is during finalisation: handlers whose streams have
    already been closed."""

    def warning(self, *a, **k):
        raise ValueError("I/O operation on closed file")


R.logger = _DeadLogger()

started = threading.Event()


def wedged():
    started.set()
    time.sleep(0.5)


fut = R._STORE_EXECUTOR.submit(wedged)
R._STORE_INFLIGHT.add(fut)
assert started.wait(5)
out.write("MAIN DONE\n")
'''


def test_the_store_hook_cannot_skip_the_interpreters_join_loop(tmp_path):
    """`atexit._run_exitfuncs` wraps every callback in its own try/except;
    `threading._shutdown` does not. Measured on this interpreter: one raise
    out of a `threading._register_atexit` callback prints "Exception ignored
    in: <module 'threading'...>", **skips every remaining hook** — including
    `concurrent.futures`' `_python_exit` — and skips the non-daemon join loop
    underneath, while the process still exits 0 so nothing looks wrong.

    Round 6 moved this hook onto that channel for its ORDERING and inherited
    the missing isolation with it. The body is not exception-free: it calls
    `logger.warning` at finalisation, where a handler's stream may already be
    closed — the ordinary `ValueError: I/O operation on closed file`.

    World where it fails: the round-6 body with no try/except (the file has
    no "LATER HOOK RAN" line, and stderr carries the ignored exception).
    """
    out = tmp_path / "landed.txt"
    script = tmp_path / "isolation_driver.py"
    script.write_text(_ISOLATION_SCRIPT.format(src=str(_ROOT / "src"),
                                               out=str(out)),
                      encoding="utf-8")

    proc = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True, timeout=120,
                          cwd=str(_ROOT),
                          env={**os.environ, "GHOST_API_KEY": "x",
                               "PYTHONPATH": str(_ROOT / "src")})

    assert proc.returncode == 0, (proc.returncode, proc.stderr[-2000:])
    landed = out.read_text().splitlines()
    assert "MAIN DONE" in landed, (landed, proc.stderr[-2000:])
    assert "LATER HOOK RAN" in landed, (
        "a raise from the store hook skipped every later shutdown hook — on "
        "the real list that is `_python_exit`, i.e. the pool drain and the "
        f"whole non-daemon join loop. File: {landed}, stderr: "
        f"{proc.stderr[-600:]}")
    assert "Exception ignored in" not in proc.stderr, proc.stderr[-600:]
    # the failure is still SAID, on the one channel that survives finalisation
    assert "store-pool shutdown hook failed" in proc.stderr, proc.stderr[-600:]


# ═══════════════════════════════════════════════════════════════════════════
# 6. the handheld client reads the same fields the web client does
# ═══════════════════════════════════════════════════════════════════════════

def _client_function(name: str):
    """Extract ONE top-level function from the uConsole client and execute
    it, with the module constants it needs stubbed.

    PyQt6 is not installed in this venv (the client runs on the handheld
    against its own gui_env), so importing `client.py` is impossible — and a
    source-TEXT assertion is an R4 reject that the pin-quality ratchet
    counts. Parsing the module and compiling the one function is neither: it
    runs the real code, and it breaks on a rename rather than on a comment.
    """
    tree = ast.parse(_CLIENT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"NOTE_OK": "<OK>", "NOTE_WARN": "<WARN>",
                  "NOTE_DIM": "<DIM>", "NOTE_ERR": "<ERR>"}
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         str(_CLIENT), "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name}() is not a top-level function in client.py")


def test_the_handheld_client_reports_an_incomplete_restore():
    """The LOAD side of the handheld client printed "workspace restored."
    unconditionally: the route answers 200 with `unrestored` naming files
    that are NOT on disk and `not_cleared` naming directories whose stale
    contents were carried into the workspace, and round 6 taught `app.js` to
    read both while this client was left on the old shape. Same defect the
    SAVE side here fixed for `X-Ghost-Archive-Omitted`, one endpoint over.

    World where it fails: a note that ignores the response body (every case
    below reads "workspace restored.", including the one where three files
    are missing).
    """
    note = _client_function("_restore_note")

    clean = note({"status": "success", "chat_history": [],
                  "not_cleared": [], "unrestored": []})
    assert clean.startswith("<OK>") and "restored." in clean

    incomplete = note({
        "status": "success", "chat_history": [],
        "not_cleared": ["projects (released projects are never wiped)"],
        "unrestored": [{"path": "dist/vendor.js", "reason": "EACCES: denied"},
                       {"path": "a/b.txt", "reason": "EEXIST: File exists"}]})
    assert incomplete.startswith("<WARN>"), incomplete
    assert "INCOMPLETE" in incomplete, incomplete
    assert "2 file(s) could NOT be written" in incomplete, incomplete
    assert "1 path(s) survived the wipe" in incomplete, incomplete
    # the member NAMES, not `[object Object]`-style repr of the dicts
    assert "dist/vendor.js" in incomplete and "a/b.txt" in incomplete
    assert "reason" not in incomplete, incomplete

    # each field alone is enough to make the restore incomplete
    assert note({"not_cleared": ["dist"], "unrestored": []}).startswith("<WARN>")
    assert note({"not_cleared": [], "unrestored": ["x"]}).startswith("<WARN>")
    # ...and a server that answers neither field is not "incomplete"
    assert note({"status": "success", "chat_history": []}).startswith("<OK>")
    assert note({"not_cleared": None, "unrestored": "boom"}).startswith("<OK>")
