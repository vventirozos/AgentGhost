"""§4GK round 6 — the API-route defects a read-only reviewer confirmed by
execution, pinned by driving the real handlers.

THREE FINDINGS, one theme: a fix that is correct WITHIN one call and wrong
one call later.

* **the restored mode traps the workspace.** Round 5 taught the restore to
  apply archived DIRECTORY modes and reasoned carefully about the hazard —
  inside a single restore. The sandbox is model-writable, so `chmod 555
  dist` is all it takes to make the NEXT restore fail: the wipe cannot
  unlink inside a non-writable directory (`ignore_errors=True` swallowed
  that), the frozen tree was carried into the "restored" workspace, and a
  new member under it answered an opaque 500 from `os.open(O_CREAT)` —
  after the sandbox had already been wiped, for ever, on every later
  archive.
* **the store pool's atexit hook was dead code.** `concurrent.futures`
  registers its own hook through `threading._register_atexit`, which runs
  (and joins) inside `wait_for_thread_shutdown()`, BEFORE `_PyAtExit_Call`.
  So the executor drained its queue first and ours arrived with nothing to
  cancel. The pin round 5 wrote called `_shutdown_store_executor(ex)`
  directly on a private executor: it never entered interpreter shutdown,
  which is the only world where the defect lives, and passed identically
  with the registration line deleted. This one drives a subprocess to exit.
* **an unrepresentable PAST mtime clamped to the MAXIMUM date.** Round 5
  rewrote that fallback for "BOTH ends" and gave both ends the same answer.

Nothing here asserts on source text: every pin drives a real handler or a
real interpreter shutdown.
"""
import io
import json
import os
import stat as _stat
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ghost_agent.api import routes as R

_ROOT = Path(__file__).resolve().parents[1]


# ── doubles (the same shapes the round-5 file drives the routes with) ────────

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


# ═══════════════════════════════════════════════════════════════════════════
# 1. a restored read-only directory must not trap the workspace for ever
# ═══════════════════════════════════════════════════════════════════════════

def _frozen_dist_sandbox(tmp_path, name: str, extra: str | None = None) -> Path:
    """A sandbox holding `dist/` at 0o555 — the shape a build tool leaves
    behind and the model can produce with one `chmod`."""
    sandbox = tmp_path / name
    dist = sandbox / "dist"
    dist.mkdir(parents=True)
    (dist / "bundle.js").write_text("// monday\n")
    if extra:
        (dist / extra).write_text("// tuesday\n")
    (sandbox / "notes.txt").write_text("plain")
    dist.chmod(0o555)
    return sandbox


async def test_a_new_member_under_a_restored_read_only_directory_still_lands(tmp_path):
    """THE 500, reproduced with no crafted archive — two ordinary saves of a
    sandbox whose `dist/` is 0o555, restored in order.

    Measured pre-fix: restore Monday succeeds and leaves a 0o555 `dist/` in
    the live sandbox; restore Tuesday raises
    `PermissionError: .../dist/vendor.js` out of `os.open(O_CREAT)`, which
    `_restore_file_member`'s caller re-raised (only ELOOP was handled) into
    the route's generic 500 — AFTER the wipe, so the workspace is gone and
    the operator gets an error id. And it never heals: every later archive
    with a new member under `dist/` dies at the same point.

    World where it fails: the round-5 restore (`rmtree(ignore_errors=True)`
    + `raise` on EACCES).
    """
    monday = await _save_bytes(_frozen_dist_sandbox(tmp_path, "mon"))
    tuesday = await _save_bytes(
        _frozen_dist_sandbox(tmp_path, "tue", extra="vendor.js"))

    live = tmp_path / "live"
    live.mkdir()
    first = await _load(live, monday)
    assert first["status"] == "success"
    assert _stat.S_IMODE((live / "dist").stat().st_mode) == 0o555, (
        "the archived directory mode was not applied at all — this pin is "
        "not in the world it was written for")

    second = await _load(live, tuesday)

    assert second["status"] == "success"
    assert second["unrestored"] == [], second["unrestored"]
    assert (live / "dist" / "vendor.js").read_text() == "// tuesday\n"
    assert (live / "dist" / "bundle.js").read_text() == "// monday\n"
    # and the archived mode is still honoured after the repair
    assert _stat.S_IMODE((live / "dist").stat().st_mode) == 0o555


async def test_a_restored_read_only_directory_is_actually_wiped_next_time(tmp_path):
    """The quieter half of the same defect, and the one that produces a
    WRONG workspace rather than an error: `shutil.rmtree(item,
    ignore_errors=True)` cannot unlink inside a 0o555 directory, so the
    frozen tree survived the "clean" wipe and its stale contents were
    presented as the restored workspace.

    Restore a sandbox that HAS `dist/`, then one that does not: `dist/` and
    the file in it must be gone.

    World where it fails: the round-5 wipe, which leaves
    `live/dist/bundle.js` sitting in a workspace whose archive never
    mentioned it — with a 200 and no record anywhere.
    """
    with_dist = await _save_bytes(_frozen_dist_sandbox(tmp_path, "mon"))
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "notes.txt").write_text("only this")
    without_dist = await _save_bytes(clean)

    live = tmp_path / "live"
    live.mkdir()
    assert (await _load(live, with_dist))["status"] == "success"
    assert (live / "dist" / "bundle.js").exists()

    result = await _load(live, without_dist)

    assert result["status"] == "success"
    assert result["not_cleared"] == [], result["not_cleared"]
    assert not (live / "dist").exists(), (
        "a 0o555 directory survived the wipe: its stale contents are now "
        "part of a workspace the archive never described")
    assert (live / "notes.txt").read_text() == "only this"


async def test_a_frozen_directory_the_wipe_never_touches_is_repaired(tmp_path):
    """`acquired_skills/` is SKIPPED by the wipe by design, so the wipe fix
    alone cannot reach it — a frozen directory there would still 500 every
    restore. The retry that repairs the chain is what covers it.

    World where it fails: any version that only unfreezes what it wipes.
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / "notes.txt").write_text("x")
    # the SAVE excludes acquired_skills by design, so the member is appended
    # the way a hand-assembled or pre-exclusion archive carries one
    archive_path = tmp_path / "a.zip"
    archive_path.write_bytes(await _save_bytes(src))
    with zipfile.ZipFile(archive_path, "a") as zf:
        zf.writestr("sandbox/acquired_skills/keep/new.py", "# new\n")
        # the DIRECTORY half of the same trap: `mkdir` inside a 0o555
        # ancestor raises EACCES exactly as `os.open(O_CREAT)` does, and it
        # sat OUTSIDE the restore's try, so it reached the generic 500
        # without even the errno check.
        # ...under an ancestor of its OWN, so it is not incidentally
        # unfrozen by the repair the file member above triggers (the first
        # version of this pin put both under `keep/` and the directory
        # mutant survived).
        zf.writestr(zipfile.ZipInfo("sandbox/acquired_skills/frozen/sub/"), b"")

    live = tmp_path / "live"
    (live / "acquired_skills" / "keep").mkdir(parents=True)
    (live / "acquired_skills" / "frozen").mkdir()
    (live / "acquired_skills" / "keep").chmod(0o555)
    (live / "acquired_skills" / "frozen").chmod(0o555)
    (live / "acquired_skills").chmod(0o555)

    result = await _load(live, archive_path.read_bytes())

    assert result["status"] == "success", result
    assert result["unrestored"] == [], result["unrestored"]
    assert (live / "acquired_skills" / "keep" / "new.py").read_text() == "# new\n"
    assert (live / "acquired_skills" / "frozen" / "sub").is_dir()


async def test_an_unwritable_member_is_reported_not_a_500_after_the_wipe(tmp_path):
    """The floor under the repair. If a member STILL cannot be written after
    the chain is unfrozen, the restore says which one — it does not raise
    into the route's generic handler, because by then the sandbox has
    already been wiped and a 500 leaves the operator with an error id and no
    workspace.

    World where it fails: `raise` for every errno but ELOOP.
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("A")
    (src / "b.txt").write_text("B")
    archive = await _save_bytes(src)

    live = tmp_path / "live"
    live.mkdir()
    real_open = os.open

    def refuse_b(path, flags, *a, **k):
        if str(path).endswith("/b.txt") and (flags & os.O_CREAT):
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(path, flags, *a, **k)

    with patch.object(R.os, "open", refuse_b):
        result = await _load(live, archive)

    assert result["status"] == "success"
    assert [u["path"] for u in result["unrestored"]] == ["b.txt"], result
    assert "Permission denied" in result["unrestored"][0]["reason"]
    assert (live / "a.txt").read_text() == "A", (
        "one unwritable member took the rest of the restore with it")


# ═══════════════════════════════════════════════════════════════════════════
# 2. an unrepresentable PAST mtime clamps to 1980, not to 2107
# ═══════════════════════════════════════════════════════════════════════════

async def test_an_unrepresentable_past_mtime_clamps_to_the_zip_EPOCH(tmp_path):
    """`time.localtime(-1e18)` raises OSError [Errno 84] and the round-5
    fallback answered `_ZIP_DOS_MAX_DATE_TIME` for it, so a far-PAST stamp
    was archived as 2107-12-31 — 127 years the wrong way, on the line whose
    own comment says it clamps "BOTH ends".

    The stamp is injected at `os.stat` because APFS saturates st_mtime at
    the int64-nanosecond floor (year 1677), which `localtime` CAN break
    down: the host cannot produce the value, the archive format can, and
    the `except` arm exists precisely for the one the host cannot make.

    World where it fails: a single `dt = _ZIP_DOS_MAX_DATE_TIME` fallback
    (archives `(2107, 12, 31, 23, 59, 58)`).
    """
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "ancient.txt").write_text("before the calendar")
    (sandbox / "normal.txt").write_text("today")
    real_stat = os.stat

    class _Unbreakable:
        """A stat result whose mtime no platform calendar can decompose."""

        def __init__(self, st):
            self.st_mode, self.st_size = st.st_mode, st.st_size
            self.st_mtime = -1e18

    def stat_with_a_broken_clock(path, *a, **k):
        st = real_stat(path, *a, **k)
        if str(path).endswith("ancient.txt"):
            return _Unbreakable(st)
        return st

    with patch.object(R.os, "stat", stat_with_a_broken_clock):
        data = await _save_bytes(sandbox)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        ancient = zf.getinfo("sandbox/ancient.txt").date_time
        normal = zf.getinfo("sandbox/normal.txt").date_time
    assert ancient == (1980, 1, 1, 0, 0, 0), (
        f"a stamp from before the calendar was archived as {ancient} — the "
        f"fallback clamps the wrong way")
    assert normal[0] >= 2020, normal      # control: real stamps untouched


# ═══════════════════════════════════════════════════════════════════════════
# 3. the store pool's shutdown hook must run BEFORE the executor's own
# ═══════════════════════════════════════════════════════════════════════════

_EXIT_SCRIPT = r'''
import os, sys, threading, time, concurrent.futures
sys.path.insert(0, {src!r})
os.environ.setdefault("GHOST_API_KEY", "x")
import ghost_agent.api.routes as R

out = open({out!r}, "w", buffering=1)
tiny = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                             thread_name_prefix="ghost-store")
# the registered hook reads the module global at CALL time
R._STORE_EXECUTOR = tiny
started = threading.Event()


def wedged():
    started.set()
    time.sleep(1.0)


fut = tiny.submit(wedged)
R._STORE_INFLIGHT.add(fut)
assert started.wait(5)
for i in range(3):
    tiny.submit(out.write, "QUEUED MUTATION RAN\n")
out.write("MAIN DONE\n")
'''


def test_the_store_pool_drops_its_queue_at_interpreter_exit(tmp_path):
    """⚠ THE ROUND-5 PIN NEVER ENTERED THE WORLD ITS DOCSTRING NAMED. It
    called `_shutdown_store_executor(ex)` directly on a private executor, so
    it measured what the function does when you call it — not whether
    ANYTHING calls it at shutdown, which is the only place the defect lives.
    It passed identically with the registration line deleted.

    The defect: `atexit.register` puts the hook after
    `wait_for_thread_shutdown()`, and `concurrent.futures.thread` registers
    `_python_exit` through `threading._register_atexit`, which runs INSIDE
    that — draining the work queue and joining the workers. So every store
    mutation still queued behind a wedged one ran during interpreter
    shutdown, against half-finalised modules, and the hook that exists to
    cancel them arrived afterwards with an empty queue. Measured on this
    interpreter driving this module: 3 of 3 queued mutations ran.

    This drives a REAL interpreter to exit with a wedged store call and
    three mutations queued behind it, and reads what landed.

    World where it fails: `atexit.register(_shutdown_store_executor)`
    (the file says `QUEUED MUTATION RAN` three times).
    """
    out = tmp_path / "landed.txt"
    script = tmp_path / "exit_driver.py"
    script.write_text(_EXIT_SCRIPT.format(src=str(_ROOT / "src"), out=str(out)),
                      encoding="utf-8")

    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, str(script)],
                          capture_output=True, text=True, timeout=120,
                          cwd=str(_ROOT),
                          env={**os.environ, "GHOST_API_KEY": "x",
                               "PYTHONPATH": str(_ROOT / "src")})
    elapsed = time.monotonic() - t0

    assert proc.returncode == 0, (proc.returncode, proc.stderr[-2000:])
    landed = out.read_text().splitlines()
    assert "MAIN DONE" in landed, (landed, proc.stderr[-2000:])
    assert [ln for ln in landed if ln == "QUEUED MUTATION RAN"] == [], (
        "queued store mutations ran during interpreter shutdown — the hook "
        f"is registered on the wrong channel. File: {landed}")
    # the wedged call still blocks exit (nothing in-process can change that),
    # but the shutdown must not ADD a wait of its own on top of it
    assert elapsed < 30, f"the interpreter took {elapsed:.1f}s to exit"


def test_the_shutdown_hook_is_registered_where_it_runs_first():
    """The mechanism behind the pin above, so a regression says WHY rather
    than only that three lines appeared in a file: the hook has to be on
    `threading`'s list (run before the join loop), not on `atexit`'s (run
    after it).

    World where it fails: `atexit.register(...)` — the channel reads
    "atexit".
    """
    assert R._STORE_SHUTDOWN_CHANNEL == "threading", R._STORE_SHUTDOWN_CHANNEL
    # `_register_atexit` wraps each hook in a partial, and the list is run in
    # REVERSE — so "runs first" means "registered last". The hook that must
    # lose the race is the `_python_exit` of the module that DEFINES our
    # executor's class (after a reload there can be more than one in the
    # list, and only that one owns our threads).
    registered = [getattr(c, "func", c)
                  for c in getattr(threading, "_threading_atexits", [])]
    assert R._shutdown_store_executor in registered, registered
    owner = type(R._STORE_EXECUTOR).__init__.__globals__["_python_exit"]
    assert registered.index(owner) < registered.index(R._shutdown_store_executor), (
        "the pool's own hook is registered AFTER ours, so it runs FIRST and "
        "drains the queue before we can cancel it")


def test_the_wedged_warning_can_actually_fire(capsys, caplog):
    """`_STORE_INFLIGHT` exists only to tell a RUNNING store call from a
    QUEUED one at exit, and with the hook on the wrong channel it could
    never see one: by the time `atexit` ran, `_python_exit` had joined every
    worker, so `running` was always 0 and the "store is wedged" warning was
    unreachable. It is reachable now; this is the behaviour it reports.
    """
    import concurrent.futures as cf

    ex = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="ghost-store")
    gate, running = threading.Event(), threading.Event()

    def wedged():
        running.set()
        gate.wait(10)

    fut = ex.submit(wedged)
    R._STORE_INFLIGHT.add(fut)
    try:
        assert running.wait(5)
        with caplog.at_level("WARNING"):
            R._shutdown_store_executor(ex)
        assert any("store is wedged" in r.getMessage() for r in caplog.records), \
            [r.getMessage() for r in caplog.records]
    finally:
        gate.set()
        fut.result(5)
        R._STORE_INFLIGHT.discard(fut)
