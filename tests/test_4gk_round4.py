"""§4GK round 4 (2026-09-13): defects found INSIDE §4GJ round 3's fixes.

Round 3 hardened the filesystem, the sandbox and the API. This round drove
those fixes through their real consumers and found that several of them had
introduced new defects, or closed a hole by labelling rather than by acting.
Each pin below names the world it fails in.

Nothing here asserts on source text: every pin drives the real function.
"""
import errno
import os
import shutil
import resource
from pathlib import Path

import pytest

from ghost_agent.sandbox import registry_guard as rg
from ghost_agent.tools.file_system import (copytree_nofollow, read_bytes_nofollow_fd,
                                           read_text_nofollow, walk_nofollow)
from ghost_agent.tools.search import source_block_failed


def _open_fds() -> int:
    return len(os.listdir("/dev/fd"))


# ── walk_nofollow: the descriptor frontier ──────────────────────────────────

def test_leaving_the_walk_early_does_not_leak_the_descriptor_frontier(tmp_path):
    """Sub-directories are opened and QUEUED before the generator suspends,
    so a consumer that breaks out of the loop ran only the inner `finally`
    and orphaned every descriptor still queued — for the life of the process.

    The real consumers all leave early: the idle project reader breaks at a
    12-file cap on every tick, and the workspace ZIP raises when the tree is
    too large. Measured on the pre-fix tree: 240 descriptors leaked across
    five capped walks of a 60-directory tree, monotonic, not reclaimed by gc.

    Fails in any tree where the generator closes only the suspended frame."""
    for i in range(60):
        d = tmp_path / f"d{i:03d}"
        d.mkdir()
        (d / "f.txt").write_text("x")
    before = _open_fds()
    for _ in range(5):
        n = 0
        for _dirpath, files, _dfd in walk_nofollow(tmp_path):
            n += len(files)
            if n >= 12:
                break
    assert _open_fds() == before, (
        f"{_open_fds() - before} descriptors leaked across five capped walks")


def test_a_full_walk_closes_every_descriptor_too(tmp_path):
    """Control: the pre-fix tree was already correct here, which is why the
    leak went unseen — the inner `finally` covers the fully-consumed case."""
    for i in range(20):
        (tmp_path / f"d{i:02d}").mkdir()
        (tmp_path / f"d{i:02d}" / "f.txt").write_text("x")
    before = _open_fds()
    for _ in range(5):
        for _dirpath, _files, _dfd in walk_nofollow(tmp_path):
            pass
    assert _open_fds() == before


def test_descriptor_exhaustion_raises_instead_of_truncating_the_tree(tmp_path):
    """EMFILE/ENFILE are OSError too, so the `except OSError: continue` that
    means "this entry was swapped for a link mid-walk" also swallowed running
    out of descriptors — and the walk returned a SUBSET of the tree with no
    exception and no log. The coding loop then reported "the attempt changed
    no files inside the project workspace" and discarded a good attempt; the
    workspace ZIP shipped incomplete as a 200.

    Fails in any tree that treats exhaustion as a swapped entry."""
    for i in range(400):
        d = tmp_path / f"p{i:03d}"
        d.mkdir()
        (d / "index.js").write_text("x")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, hard))
    try:
        with pytest.raises(OSError) as exc:
            for _dirpath, _files, _dfd in walk_nofollow(tmp_path):
                pass
        assert exc.value.errno in (errno.EMFILE, errno.ENFILE), exc.value
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_a_symlinked_base_is_REFUSED_loudly_not_silently_empty(tmp_path):
    """Two defects, one line, and round 4 fixed the wrong one.

    The original code opened the base O_NOFOLLOW and, when that failed,
    `return`ed — yielding an EMPTY generator that every consumer reads as
    "this tree has no files", indistinguishable from the truth. Round 4
    "fixed" that by resolving the base through `os.path.realpath` first,
    borrowing `copytree_nofollow`'s justification without its PRECONDITION:
    that helper's caller ran the path through `_get_safe_path`, which
    resolves AND contains. These callers do not — `_gather_project_files`
    and `reconcile_research_dir` build `<sandbox>/projects/<pid>` themselves
    and gate only on `is_dir()`, which follows. So the model could delete its
    project directory, replace it with a link to the host's home, and the
    idle project reader would read host files and report their paths as if
    they were inside the project. Reproduced with an `id_rsa`.

    Both halves are needed: do NOT follow, and do NOT go quiet about it.

    Fails in any tree that follows a linked base, and in any tree that
    answers a refused base with an empty walk."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.txt").write_text("x")
    link = tmp_path / "ws"
    link.symlink_to(real)
    with pytest.raises(ValueError) as exc:
        list(walk_nofollow(link))
    assert "symlink" in str(exc.value)
    # …and the real directory is unaffected
    got = [(d, f) for d, f, _ in walk_nofollow(real)]
    assert got and got[0][1] == ["a.txt"]


def test_a_linked_project_directory_cannot_deliver_host_files(tmp_path):
    """The consumer-level statement of the same thing, in the shape the idle
    project reader actually uses: a base built by string concatenation and
    gated on `is_dir()`."""
    host = tmp_path / "host"
    host.mkdir()
    (host / "id_rsa").write_text("PRIVATE KEY")
    sandbox = tmp_path / "sandbox" / "projects"
    sandbox.mkdir(parents=True)
    evil = sandbox / "p1"
    evil.symlink_to(host)
    assert evil.is_dir()                      # the gate the readers use
    with pytest.raises(ValueError):
        list(walk_nofollow(evil))


# ── read_bytes_nofollow_fd: descriptor ownership ────────────────────────────

def test_a_failed_read_does_not_close_the_descriptor_twice(tmp_path, monkeypatch):
    """`os.fdopen(fd, closefd=True)` takes OWNERSHIP of the descriptor. The
    old body wrapped the `with os.fdopen(...)` in
    `except BaseException: os.close(fd)`, so a read that raised mid-stream
    closed the same fd twice: once by the file object on its way out of the
    `with`, once by hand. That is the defect the sibling writer in this module
    documents at length — between the two closes another thread can be handed
    that fd number and the second close shuts down an unrelated file — and
    both consumers run inside `asyncio.to_thread` pools.

    The observable that separates the two worlds is a RAW `os.close` issued
    after ownership has already transferred. The file object's own close is
    C-level and never reaches `os.close`, so in the fixed world this list is
    empty and in the broken one it holds the hand-rolled close.

    Fails in any tree where the manual close covers the post-fdopen window."""
    f = tmp_path / "x.bin"
    f.write_bytes(b"payload")
    raw_closes = []
    real_close = os.close
    real_fdopen = os.fdopen

    def _fdopen(fd, *a, **k):
        fh = real_fdopen(fd, *a, **k)

        class _RaisingHandle:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                fh.close()               # ownership discharged, C-level
                return False

            def read(self_inner, *a2):
                raise OSError(errno.EIO, "EIO simulated mid-read")
        return _RaisingHandle()

    monkeypatch.setattr(os, "fdopen", _fdopen)
    monkeypatch.setattr(os, "close", lambda fd: (raw_closes.append(fd),
                                                 real_close(fd))[1])
    with pytest.raises(OSError):
        read_bytes_nofollow_fd(str(f))
    monkeypatch.undo()
    assert raw_closes == [], (
        f"a descriptor already owned by the file object was closed by hand "
        f"as well: {raw_closes}")


# ── read_text_nofollow: the cap is a HEAD ───────────────────────────────────

def test_the_byte_cap_returns_the_head_not_the_tail(tmp_path):
    """The cap used to `lseek` to `size - max_bytes` and return the END of
    the file. Its only capped caller — the workspace tidy's referenced-media
    scan — uses the head idiom, so for any file over the cap it scanned the
    last 512 KB and never saw an `<img src=…>` in the head: the asset counted
    as unreferenced and the idle sweep DELETED it, against that function's own
    "a false DELETE breaks the build".

    Fails in any tree where the cap reads the tail."""
    f = tmp_path / "index.html"
    f.write_text("HEAD-MARKER\n" + ("x" * 5000) + "\nTAIL-MARKER", encoding="utf-8")
    out = read_text_nofollow(f, max_bytes=64)
    assert out.startswith("HEAD-MARKER")
    assert "TAIL-MARKER" not in out
    assert len(out.encode("utf-8")) <= 64


def test_an_uncapped_read_still_returns_the_whole_file(tmp_path):
    """Control: max_bytes=0 means no cap, and both worlds agree."""
    f = tmp_path / "a.txt"
    f.write_text("one\ntwo\nthree")
    assert read_text_nofollow(f) == "one\ntwo\nthree"


# ── copytree_nofollow: an absolute in-root link names the SOURCE ────────────

def test_an_absolute_in_root_link_cannot_write_back_into_the_source(tmp_path):
    """`_escapes` asks only "does the target resolve inside root", which an
    ABSOLUTE link within the tree satisfies — so it was recreated verbatim and
    still named the source. Reproduced end-to-end: a write through the copied
    alias changed the production file the isolated copy is documented to be
    unable to reach.

    Fails in any tree that recreates an absolute in-root link."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "tool.py").write_text("ORIGINAL\n")
    (src / "alias.py").symlink_to(src / "tool.py")
    dst = tmp_path / "copy"
    skipped = copytree_nofollow(src, dst, src)
    assert any("absolute symlink" in s for s in skipped), skipped
    assert not (dst / "alias.py").exists()
    # nothing inside the copy reaches the source
    for p in dst.rglob("*"):
        if p.is_symlink():
            target = os.readlink(p)
            assert not os.path.isabs(target), (p, target)
    (dst / "tool.py").write_text("WRITTEN IN THE COPY\n")
    assert (src / "tool.py").read_text() == "ORIGINAL\n"


# ── the research status predicate is ANCHORED ───────────────────────────────

def test_a_page_that_quotes_an_error_is_not_a_failed_source():
    """The rule was `"\\nError:" in block` — an unanchored substring over the
    whole block, which contains the PAGE'S OWN TEXT. A successful fetch of a
    page quoting `Error: division by zero` counted as a failed source; with a
    single URL that makes `_n_ok == 0`, so a complete report was booked as a
    FAILED action, fired the strike ledger, and told the model that no source
    could be fetched.

    Fails in any tree where the predicate is unanchored."""
    body = ("### SOURCE: https://example.org/a\n"
            "How to handle exceptions\nError: division by zero happens when…\n")
    assert source_block_failed(body) is False
    real_failure = ("### SOURCE: https://example.org/b\n"
                    "Error: per-URL timeout exceeded (45s)\n")
    assert source_block_failed(real_failure) is True
    assert source_block_failed("### SOURCE: https://example.org/c\nclean text\n") is False
    assert source_block_failed("### SOURCE: https://example.org/d\n") is False


def test_both_research_tools_share_the_one_predicate():
    """Two parsers, one authority: the onion sibling used to spell the loose
    rule while its OWN cache gate, thirty lines away in the same function,
    used the anchored one — so the two disagreed about the same run."""
    from ghost_agent.tools import darkweb_search
    assert darkweb_search._source_block_failed is source_block_failed


# ── registry_guard: a pid is an exact integer, and a kill is verified ───────

def test_a_fractional_pid_is_refused_rather_than_rounded():
    """`int(2.9)` is 2, so a tampered or corrupted `pid: 2.9` row used to
    build a real kill for pid 2 — a neighbour's process.

    Fails in any tree where valid_pid truncates."""
    assert rg.valid_pid(2.9) is None
    assert rg.valid_pid(4242.5) is None
    assert rg.valid_pid(4242.0) == 4242          # integral float is still a pid
    assert rg.valid_pid(4242) == 4242
    assert rg.valid_pid(True) is None
    with pytest.raises(ValueError):
        rg.kill_tree_script(None)                # not TypeError
    with pytest.raises(ValueError):
        rg.kill_tree_script([2])


def test_kill_tree_answers_whether_the_tree_died_not_whether_a_signal_was_sent():
    """It discarded the exec result and returned an unconditional True once
    the pid passed the floor. `services.py` believed it, unlinked the pidfile
    and dropped the registry row while the process kept running and held its
    port. The jobs sibling always re-probed; this one did not.

    Fails in any tree where kill_tree reports "sent"."""
    # §4GK round 5: the kill script prints its own verdict, so the answer
    # comes from the shell that did the killing rather than a second exec.
    assert rg.kill_tree(lambda c, timeout=30: (rg.SURVIVED_MARKER, 0), 4242) is False
    assert rg.kill_tree(lambda c, timeout=30: (rg.KILLED_MARKER, 0), 4242) is True
    # and the script really does ask, after the final KILL
    script = rg.kill_tree_script(4242)
    assert script.rstrip().endswith("true")
    assert rg.SURVIVED_MARKER in script and rg.KILLED_MARKER in script
    assert script.index("sig KILL") < script.index(rg.SURVIVED_MARKER)


# ── survivors of the §4GK battery, turned into pins ─────────────────────────

def test_an_incomplete_reference_scan_does_not_authorise_a_delete(tmp_path):
    """`_referenced_media`'s caller turns "not in the returned set" straight
    into `to_delete`. Every source file the scan could not read — unreadable,
    a symlink the hardened reader refuses, or over the size cap — used to be
    skipped silently, so one unreadable `index.html` was enough to delete the
    assets it points at. This function's own contract is that a false KEEP
    costs kilobytes and a false DELETE breaks the build.

    Fails in any tree where an unscanned source still yields a confident set."""
    from ghost_agent.core import workspace_cleanup as wc
    (tmp_path / "app.js").write_text("nothing references the asset here")
    (tmp_path / "hero.png").write_bytes(b"\x89PNG")
    secret = tmp_path / "index.html"
    secret.write_text("<img src='hero.png'>")
    os.chmod(secret, 0o000)
    try:
        out = wc._referenced_media(tmp_path, ["hero.png"])
        assert "hero.png" in out, (
            "an unreadable source file made the scan report a confident "
            "'unreferenced', which the caller deletes")
    finally:
        os.chmod(secret, 0o644)


def test_a_complete_scan_still_reports_an_unreferenced_asset(tmp_path):
    """Control: with every source readable the answer is exact, so the tidy
    still works. Both worlds agree here — which is why the fail-safe above
    has to be pinned separately."""
    from ghost_agent.core import workspace_cleanup as wc
    (tmp_path / "app.js").write_text("no reference at all")
    (tmp_path / "stray.png").write_bytes(b"\x89PNG")
    assert wc._referenced_media(tmp_path, ["stray.png"]) == set()


def test_a_pending_nonce_survives_many_intervening_commands(tmp_path):
    """§4GK round 4. The pending set was a 64-entry FIFO — a COUNT bound on a
    TIME window. A command has no registry row from `_write_script` until
    `_promote`, 90 s away and up to the whole exec budget; during that window
    the pending mark is the nonce's only protection. Sixty-four ordinary
    `execute` calls evicted it, the next `_save` dropped the nonce,
    `_read_exit` rejected the job's own genuine sentinel, and a ten-minute run
    that exited 0 was reported to the model as EXIT CODE 137 — the regression
    round 3 removed, re-created inside its own fix.

    Fails in any tree that bounds the pending set by count."""
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    sup = SandboxJobSupervisor.__new__(SandboxJobSupervisor)
    import threading
    sup.sandbox = None
    sup._lock = threading.RLock()
    sup._nonces = {}
    sup._nonce_pending = {}
    sup._nonce_store = None

    victim = "job-deadbeef"
    sup._nonces[victim] = "n" * 16
    sup._note_pending_nonce(victim)
    for i in range(300):                     # ordinary commands, none promoted
        sup._nonces[f"job-{i:08x}"] = "x" * 16
        sup._note_pending_nonce(f"job-{i:08x}")
    assert victim in sup._nonce_pending, (
        "the long-running job's pending mark was evicted by later commands")
    # …and the sync therefore keeps its nonce even with an EMPTY registry
    sup._sync_nonces({})
    assert sup._nonces.get(victim) == "n" * 16


def test_an_in_band_job_releases_its_nonce_so_the_store_stays_bounded(tmp_path):
    """`_finish_completed` and `_kill_and_return` write no registry row and
    never call `_save`, so on the commonest path of all — a command that
    completes without ever being promoted — nothing dropped the nonce. 600
    in-band jobs held 600 nonces, and past the cap the warning claimed every
    one belonged to a RUNNING job, which was true of none of them.

    Fails in any tree where the in-band exits skip the release."""
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    import threading
    sup = SandboxJobSupervisor.__new__(SandboxJobSupervisor)
    sup.sandbox = None
    sup._lock = threading.RLock()
    sup._nonces = {"job-aaaaaaaa": "n" * 16}
    sup._nonce_pending = {"job-aaaaaaaa": 0.0}
    sup._nonce_store = None
    sup._release_nonce("job-aaaaaaaa")
    assert sup._nonces == {} and sup._nonce_pending == {}


def test_stopping_a_service_whose_process_survives_reports_failure(tmp_path):
    """§4GK round 4. `_kill_pgroup` discarded `kill_tree`'s answer, so
    `_kill_service` reported a clean stop over a process that survived
    TERM+KILL: the pidfile was unlinked and the row dropped while the process
    kept running and holding its port.

    Fails in any tree where the kill verdict is discarded."""
    from ghost_agent.sandbox.services import ServiceSupervisor
    sup = ServiceSupervisor.__new__(ServiceSupervisor)
    # host_dir is a verified property. Patch it on the CLASS and put the
    # original back — deleting it would strip the attribute for every other
    # test that later runs in this worker process.
    _orig_host_dir = ServiceSupervisor.host_dir
    ServiceSupervisor.host_dir = property(lambda self: tmp_path)
    # every exec exits 0 -> the liveness probe says the pid is STILL ALIVE
    sup._exec = lambda cmd, timeout=30: (rg.SURVIVED_MARKER, 0)
    sup._entry_alive_or_unknown = lambda e: True
    sup._port_listening = lambda port: False
    entry = {"pid": 4242, "name": "web", "port": None, "project_id": None}
    (tmp_path / "web.pid").write_text("4242")
    try:
        # §4GK round 5: the BOOL still means "did anything happen" — making it
        # False on survival was indistinguishable from "there was nothing
        # alive to kill", and `stop()` then reported a process that had just
        # survived TERM+KILL as "was already dead; removed".
        sup._kill_service(entry)
        assert sup._last_kill_survived is True
        assert (tmp_path / "web.pid").exists(), (
            "the pidfile of a process that is still running was deleted")
    finally:
        ServiceSupervisor.host_dir = _orig_host_dir


# ── the cut-off recreate: round 3's addition strictly broke the working path ─

def test_a_cut_off_container_is_actually_recreated_not_re_adopted(tmp_path):
    """§4GK round 4, the whole point of round 3's addition, driven.

    Round 3 called `_recreate_if_cut_off()` BEFORE the readiness check. That
    call nulls `self.container` and stamps `_cut_off_at`, arming the 300 s
    backoff — and the container is then re-adopted by name a few lines below,
    where the SECOND `_recreate_if_cut_off()` (the one positioned after the
    adopt, which can actually reach the remove-and-provision branch) is
    suppressed by the backoff its own earlier twin just wrote. Net effect:
    the container stayed cut off, `_egress_state` was downgraded to "" —
    which reads as "enforcement never attempted" — and that disarmed BOTH the
    tool-side refusal and the in-`_execute_impl` belt. A cut-off sandbox with
    no network and nothing refusing work.

    Fails in any tree where the early call recreates instead of declining
    the TTL short-circuit."""
    from tests.test_sandbox_tor_egress import _stub
    from unittest.mock import MagicMock, patch
    sb = _stub(tmp_path)
    sb._egress_state = "blocked"
    sb._cut_off_at = 0.0
    # THIS generation was already enforced — the class default is False, so
    # without setting it the flag assertion below passes in both worlds.
    sb._tor_attempted = True
    sb.container.attrs = {"HostConfig": {"NetworkMode": "bridge"},
                          "NetworkSettings": {"Networks": {}}}   # cut off
    sb.mark_ready()                       # a FRESH readiness stamp
    assert sb._ready_is_fresh()
    with patch("ghost_agent.sandbox.docker.pretty_log"):
        sb._recreate_if_cut_off()
    # it dropped the container AND reset the per-generation enforcement flag:
    # without the reset the replacement comes up with no Tor rules at all,
    # because `_enforce_egress_once` is a no-op while the flag is set.
    assert sb.container is None
    assert sb._tor_attempted is False, (
        "a recreate is a NEW container generation; leaving _tor_attempted set "
        "means the replacement is never enforced")
    assert not sb._ready_is_fresh()


def test_the_ttl_short_circuit_is_declined_while_the_state_is_blocked(tmp_path):
    """The property the early guard actually needs: a cut-off container must
    not ride the readiness TTL. It must DECLINE the short-circuit, not
    recreate — recreating here arms the backoff that disables the real one."""
    import ast
    import inspect as _inspect
    from ghost_agent.sandbox import docker as docker_mod
    fn = next(n for n in ast.walk(ast.parse(_inspect.getsource(docker_mod)))
              if isinstance(n, ast.FunctionDef) and n.name == "_ensure_running_impl")
    body = ast.unparse(fn)
    calls = body.count("self._recreate_if_cut_off()")
    assert calls == 1, (
        f"_ensure_running_impl calls _recreate_if_cut_off {calls}x — two calls "
        "means the first arms the backoff that suppresses the second")


def test_a_corrupt_library_catalogue_is_quarantined_not_flattened(tmp_path):
    """§4GK round 4, the layer under the reconciler's fix. The reconciler was
    taught this round to refuse to act on an unreadable catalogue — worth
    nothing while the WRITER still reset it to `[]` and carried on, because
    the very next ingest then overwrote the catalogue with a single-entry list
    and every other document became invisible to `list_docs` and undeletable
    by name.

    Fails in any tree where a corrupt catalogue is silently replaced."""
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory.__new__(VectorMemory)
    lib = tmp_path / "library_index.json"
    lib.write_text('["one.pdf", "two.pdf"  <<< truncated')
    vm.library_file = lib
    import threading
    _lock = threading.RLock()
    vm._get_lock = lambda: _lock
    vm._update_library_index("three.pdf", "add")
    # The bad bytes are PRESERVED beside the file, so the lost entries are
    # recoverable. Round 5: the index itself does rebuild from this write
    # onward — refusing the write outright (round 4's first attempt) kept the
    # bytes but blocked every future ingest with no recovery the agent could
    # perform by itself, which is a worse failure than the one it fixed.
    quarantined = tmp_path / "library_index.json.corrupt"
    assert quarantined.exists(), "the corrupt bytes were destroyed"
    assert quarantined.read_text().startswith('["one.pdf", "two.pdf"')
    import json as _json
    assert _json.loads(lib.read_text()) == ["three.pdf"]


def test_a_readable_catalogue_still_takes_the_write(tmp_path):
    """Control: the quarantine must not break the ordinary path."""
    from ghost_agent.memory.vector import VectorMemory
    import json as _json
    import threading
    vm = VectorMemory.__new__(VectorMemory)
    lib = tmp_path / "library_index.json"
    lib.write_text('["one.pdf"]')
    vm.library_file = lib
    _lock = threading.RLock()
    vm._get_lock = lambda: _lock
    vm._update_library_index("two.pdf", "add")
    assert _json.loads(lib.read_text()) == ["one.pdf", "two.pdf"]


async def test_a_cached_research_report_keeps_the_status_of_the_run_it_cached(monkeypatch):
    """§4GK round 4. The cache hit returned the raw report STRING, so the
    partial-coverage reason_code was dropped and `ToolOutcome.coerce` booked a
    plain OK: the same query answered OK from cache five minutes after
    answering partial, and the strike ledger and the corpus saw two different
    verdicts for one result.

    Fails in any tree where the cache hit returns a bare string."""
    from ghost_agent.tools import darkweb_search as dw
    banner = ("[⚠ SOURCE FAILURES: 2 of 3 hidden services could not be fetched…]\n"
              "### REPORT\nthe one that loaded said this")
    monkeypatch.setattr(dw, "_cache_get", lambda k: banner)
    out = await dw.tool_darkweb_research(query="anything", tor_proxy=None)
    assert str(out) == banner
    assert getattr(out, "reason_code", "") == "darkweb_research_sources_partial", (
        getattr(out, "reason_code", None))
    # a clean cached run stays a clean OK
    monkeypatch.setattr(dw, "_cache_get", lambda k: "### REPORT\nall three loaded")
    out2 = await dw.tool_darkweb_research(query="anything", tor_proxy=None)
    assert getattr(out2, "status", "ok") == "ok"
    assert not getattr(out2, "reason_code", "")


def test_the_self_play_snapshot_never_lists_a_symlink(tmp_path):
    """§4GK round 4. `_snapshot_mocks` walked the model-writable self-play
    sandbox with `rglob` and read each file after an `is_symlink()` pre-check
    — two syscalls against a path the sandbox can re-point in between, the
    exact check-then-read window round 3 closed everywhere else. It survived
    because the enumeration that guards the class keyed on a literal
    `os.walk`, and this reader used `rglob`.

    Fails in any tree where the snapshot can capture a linked file."""
    from ghost_agent.core.dream import _snapshot_mocks
    (tmp_path / "a.txt").write_text("A")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.csv").write_text("B")
    outside = tmp_path.parent / "victim.txt"
    outside.write_text("HOST SECRET")
    (tmp_path / "link.txt").symlink_to(outside)
    snap = _snapshot_mocks(tmp_path)
    assert sorted(snap) == ["a.txt", "sub/b.csv"]
    assert all(b"HOST SECRET" not in v for v in snap.values())


def test_one_bad_destination_entry_does_not_abort_the_whole_copy(tmp_path):
    """§4GK round 4, corrected in round 5.

    Round 4 put a per-entry guard around the regular-file write only, so
    `os.makedirs` and the recursive descent still aborted the whole tree on
    the first unexpected OSError — the very thing the fix claimed to stop.
    Round 5 reproduced it with a 120-level deep workspace (`mkdir d; cd d` in
    a loop, trivially model-reachable, and the WALK handles it fine):
    ENAMETOOLONG killed every copy of that workspace at 45 levels,
    `tool_copy_file` returned a bare error over the half-copy, and its own
    overwrite guard then refused the retry.

    Round 4's first pin for this pre-created an unwritable FILE at the
    destination and passed `dirs_exist_ok=True` — a state `tool_copy_file`
    never produces, since it refuses when the destination exists and copies
    with `dirs_exist_ok=False`. It therefore never exercised the branch that
    actually aborts. This drives the production shape.

    Fails in any tree where one unreachable destination path ends the copy."""
    src = tmp_path / "w"
    src.mkdir()
    (src / "keep.txt").write_text("keep")
    (src / "sub").mkdir()
    (src / "sub" / "also.txt").write_text("also")
    # build the deep leg the way the model does: relative, one level at a time
    cwd = os.getcwd()
    os.chdir(src)
    try:
        for _ in range(120):
            os.mkdir("d" * 45)
            os.chdir("d" * 45)
    finally:
        os.chdir(cwd)
    dst = tmp_path / "copy"
    skipped = copytree_nofollow(src, dst, src)
    # everything reachable came across …
    assert (dst / "keep.txt").read_text() == "keep"
    assert (dst / "sub" / "also.txt").read_text() == "also"
    # … and the part that could not be is NAMED, not silently missing
    assert any("could not be created" in s or "subtree could not be copied" in s
               for s in skipped), skipped


def test_a_write_that_dies_mid_file_leaves_no_truncated_stub(tmp_path, monkeypatch):
    """§4GK round 5. The per-entry guard wraps the source read AND the
    destination write, so a failure part-way through left a TRUNCATED file at
    the destination while reporting the entry as NOT copied — and the fork and
    clone paths count what landed with `rglob(...).is_file()`, so the stub was
    counted as a copied file. Reproduced with an EIO on the second chunk: a
    1 MB file where the source had 3 MB.

    Fails in any tree where the report and the destination disagree."""
    src = tmp_path / "s"
    src.mkdir()
    (src / "big.bin").write_bytes(b"A" * (3 << 20))
    (src / "ok.txt").write_text("ok")
    real_read = os.read
    seen = {"n": 0}

    def _flaky(fd, n):
        b = real_read(fd, n)
        if len(b) == (1 << 20):
            seen["n"] += 1
            if seen["n"] == 2:
                raise OSError(errno.EIO, "EIO simulated mid-file")
        return b

    monkeypatch.setattr(os, "read", _flaky)
    skipped = copytree_nofollow(src, tmp_path / "d", src)
    monkeypatch.undo()
    assert not (tmp_path / "d" / "big.bin").exists(), (
        "a truncated stub was left where the report says nothing was copied")
    assert (tmp_path / "d" / "ok.txt").read_text() == "ok"
    assert any("big.bin" in s for s in skipped), skipped


def test_the_copy_preserves_directory_modes(tmp_path):
    """`shutil.copytree` preserved directory mode bits; the dir-fd rewrite
    took the umask default, so a fork of a chmod'd tree came back with
    different permissions than its source."""
    import stat as _s
    src = tmp_path / "src"
    (src / "locked").mkdir(parents=True)
    (src / "locked" / "f.txt").write_text("x")
    (src / "locked" / "sub").mkdir()
    (src / "locked" / "sub" / "g.txt").write_text("y")
    # ⚠ READ-ONLY, NOT 0o700 (§4GK round 6). The first version used an
    # owner-WRITABLE mode, so stamping the mode BEFORE the children — the
    # defect this pin exists for — still wrote them and still ended at 0o700.
    # It passed in both worlds. A read-only source is the only mode that can
    # tell pre-order from post-order.
    os.chmod(src / "locked" / "sub", 0o500)
    os.chmod(src / "locked", 0o500)
    dst = tmp_path / "dst"
    try:
        skipped = copytree_nofollow(src, dst, src)
        assert skipped == [], skipped
        assert (dst / "locked" / "f.txt").read_text() == "x"
        assert (dst / "locked" / "sub" / "g.txt").read_text() == "y"
        assert _s.S_IMODE((dst / "locked").stat().st_mode) == 0o500
        assert _s.S_IMODE((dst / "locked" / "sub").stat().st_mode) == 0o500
    finally:
        for d in (dst / "locked" / "sub", dst / "locked",
                  src / "locked" / "sub", src / "locked"):
            try:
                os.chmod(d, 0o700)
            except OSError:
                pass


def test_an_unknown_verification_shape_is_inconclusive_not_a_leak():
    """§4GK round 4, a defect this round's own fix created. Since the leak
    branch now DISCONNECTS the container, `bool(data.get("IsTor"))` became
    dangerous: valid JSON without that key read as a confirmed direct answer,
    so an endpoint that changed its schema would cut the sandbox off its
    network on every boot. The two unknown shapes — unparseable, and
    parseable but not this schema — must agree.

    Fails in any tree where a missing or non-boolean IsTor reads as False."""
    from ghost_agent.sandbox.tor_egress import parse_tor_check
    assert parse_tor_check('{"IsTor":true,"IP":"1.2.3.4"}')[0] is True
    assert parse_tor_check('{"IsTor":false,"IP":"9.9.9.9"}')[0] is False
    assert parse_tor_check('{"IP":"9.9.9.9"}')[0] is None          # missing key
    assert parse_tor_check('{"IsTor":"yes"}')[0] is None           # wrong type
    assert parse_tor_check('[]')[0] is None                        # not an object
    assert parse_tor_check("<html>challenge</html>")[0] is None    # not JSON


# ── §4GK round 5: defects found INSIDE round 4's fixes ──────────────────────

def test_the_self_play_restore_cannot_write_through_a_planted_link(tmp_path):
    """§4GK round 5. Round 4 hardened the self-play snapshot (the READER) and
    left its writer twin alone twenty lines below: `_restore_mocks` did
    `target.parent.mkdir(parents=True); target.write_bytes(blob)` on a tree
    the SOLVER controls between the snapshot and the restore. Replace
    `data/x.csv` with a link to any host file and the restore overwrites that
    file with bytes the challenge chose. The pre-validator restore runs with
    NO purge, so nothing removes the link first.

    Fails in any tree where the restore follows a link."""
    from ghost_agent.core.dream import _restore_mocks, _snapshot_mocks
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.csv").write_text("pristine mock rows\n")
    snap = _snapshot_mocks(tmp_path)
    victim = tmp_path.parent / "HOST_FILE.txt"
    victim.write_text("HOST CONTENT")
    (tmp_path / "data" / "x.csv").unlink()
    (tmp_path / "data" / "x.csv").symlink_to(victim)
    _restore_mocks(tmp_path, snap)
    assert victim.read_text() == "HOST CONTENT", "the restore wrote through the link"
    # …and the restore still did its job: the link is gone and the mock is back
    assert not (tmp_path / "data" / "x.csv").is_symlink()
    assert (tmp_path / "data" / "x.csv").read_text() == "pristine mock rows\n"


def test_an_ordinary_restore_still_reverts_a_solver_mutation(tmp_path):
    """Control: hardening the write must not break the thing it protects."""
    from ghost_agent.core.dream import _restore_mocks, _snapshot_mocks
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.csv").write_text("pristine\n")
    snap = _snapshot_mocks(tmp_path)
    (tmp_path / "data" / "x.csv").write_text("the solver mutated this")
    _restore_mocks(tmp_path, snap)
    assert (tmp_path / "data" / "x.csv").read_text() == "pristine\n"


def test_a_non_finite_pid_is_refused_rather_than_raised():
    """§4GK round 5, a regression round 4 introduced. The fractional check was
    written OUTSIDE the try, where `int(nan)` raises ValueError and `int(inf)`
    raises OverflowError — and `json.loads` accepts both spellings by default,
    on a registry that lives on the bind mount the sandboxed process writes.
    One planted `{"pid": NaN}` row escaped the row validator, hit the loader's
    outer handler and returned an EMPTY map: every live service row vanished
    from the live map AND from the quarantine, so the next save destroyed them
    permanently — the exact denial the quarantine exists to prevent, caused by
    the guard. Before round 4, `int(nan)` was caught and that one row was
    quarantined while the good rows survived.

    Fails in any tree where the guard raises instead of refusing."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert rg.valid_pid(bad) is None, bad
        assert rg.kill_tree(lambda c, timeout=30: (rg.KILLED_MARKER, 0), bad) is False
        with pytest.raises(ValueError):
            rg.kill_tree_script(bad)
    # a whole row of them is quarantined, not fatal
    assert rg.validate_row({"pid": float("nan")}, require_pid=True) is not None
    assert rg.validate_row({"pid": 4242}, require_pid=True) is None


def test_a_fractional_port_is_refused_like_a_fractional_pid():
    """The sibling one revision behind: round 4 taught `valid_pid` that a
    non-integral value is a malformed row and left `valid_port` truncating.
    A row keeping `8100.9` is invisible to the allocator's `isinstance(int)`
    filter while still comparing unequal to the int port everywhere else."""
    assert rg.valid_port(8100.9) is None
    assert rg.valid_port(8100.0) == 8100
    assert rg.valid_port(8100) == 8100
    assert rg.valid_port(float("nan")) is None


def test_a_snapshot_the_walk_could_not_finish_is_not_an_empty_workspace(tmp_path):
    """§4GK round 5. Round 4 made the walk RAISE on descriptor exhaustion and
    guarded the snapshot with `except ValueError` — which does not cover
    OSError, so the raise went straight through a function the leaf loop
    documents as best-effort. Widening the guard alone is not enough either:
    an empty snapshot is indistinguishable from "the workspace has no files",
    and the leaf loop turns that into "the attempt changed nothing" and
    DISCARDS the work.

    Fails in any tree where an unreadable tree reads as an empty one."""
    import resource
    from ghost_agent.core.coding_loop import (_SNAPSHOT_INCOMPLETE, diff_snapshots,
                                              snapshot_workspace)
    for i in range(400):
        (tmp_path / f"p{i:03d}").mkdir()
        (tmp_path / f"p{i:03d}" / "i.js").write_text("x")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, hard))
    try:
        snap = snapshot_workspace(tmp_path)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert _SNAPSHOT_INCOMPLETE in snap, "the truncation was silent"
    # §4GK round 6: the marker is ASKED FOR, never travelled in the path list.
    # Round 5 returned `[_SNAPSHOT_INCOMPLETE]` from `diff_snapshots`, which
    # put a non-path into a list every consumer treats as workspace paths: the
    # "changed nothing" gate saw a truthy list and passed, the constraint gate
    # ran on an empty file set, and `register_file_artifact` recorded the
    # sentinel as a durable deliverable that blocks the release rehearsal.
    from ghost_agent.core.coding_loop import snapshot_incomplete
    assert snapshot_incomplete(snap) is True
    assert diff_snapshots(snap, snap) == [], (
        "the sentinel escaped into a list of workspace paths")
    for p in diff_snapshots(snap, {"real.txt": "abc"}):
        assert not p.startswith("__ghost"), p
    # control: two real snapshots still diff normally and answer False
    good = snapshot_workspace(tmp_path)
    assert _SNAPSHOT_INCOMPLETE not in good
    assert snapshot_incomplete(good) is False
    assert diff_snapshots(good, good) == []


def test_one_oversized_source_file_does_not_kill_the_media_tidy(tmp_path):
    """§4GK round 5. Round 4 marked any source file over the byte cap
    `unscanned`, which made the whole answer inconclusive — so ONE ordinary
    bundled asset (a 600 KB bundle, a sprite sheet, a minified vendor file)
    permanently disabled the media tidy for that workspace, on every pass,
    forever. That kills the feature for exactly the projects it was built
    for. The head is scanned instead: its hits count, and only candidates
    NOTHING matched fall back to keep-everything.

    Fails in any tree where an oversized file makes every candidate a keep."""
    from ghost_agent.core import workspace_cleanup as wc
    # §4GK round 6: genuinely over the cap. The first version wrote 600 KB
    # against an 8 MB cap, so the branch it exists for was never entered and
    # it passed under round 4's behaviour too — the same mistake its sibling's
    # docstring says was fixed, left standing in this copy.
    from ghost_agent.core import workspace_cleanup as _wc
    (tmp_path / "bundle.js").write_text("x" * (_wc._REFERENCE_SCAN_MAX_BYTES + 4096))
    (tmp_path / "index.html").write_text('<img src="hero.png">')
    (tmp_path / "hero.png").write_bytes(b"\x89PNG")
    (tmp_path / "stray.png").write_bytes(b"\x89PNG")
    ref = wc._referenced_media(tmp_path, ["hero.png", "stray.png"])
    assert "hero.png" in ref
    assert "stray.png" not in ref, (
        "an oversized bundle made every candidate un-deletable, which is the "
        "media tidy switched off rather than made safe")


def test_a_reference_in_the_head_of_an_oversized_file_is_still_found(tmp_path):
    """…and the head really is read, so the common case still resolves."""
    from ghost_agent.core import workspace_cleanup as wc
    (tmp_path / "app.js").write_text('require("./logo.png");\n' + "y" * 600_000)
    (tmp_path / "logo.png").write_bytes(b"\x89PNG")
    assert "logo.png" in wc._referenced_media(tmp_path, ["logo.png"])


def test_a_single_not_tor_answer_does_not_cut_the_container_off(tmp_path):
    """§4GK round 5. Round 4 made the leak branch DESTRUCTIVE — it disconnects
    the container now and the recreate removes and reprovisions it 300 s
    later, taking every in-sandbox service and promoted job with it — while
    the trigger stayed ONE answer from a third-party endpoint. That endpoint
    is measurably unreliable here: the live log carries 6 unusable answers
    against 143 enforcements, and the branch is also reached from the RESUME
    path, whose whole purpose is preserving those services.

    Two independent requests must agree before anything destructive happens.

    Fails in any tree that acts on the first answer alone."""
    from ghost_agent.sandbox import tor_egress as T
    from tests.test_sandbox_tor_egress import _drive, _stub
    answers = iter([b'{"IsTor":false,"IP":"9.9.9.9"}',
                    b'{"IsTor":true,"IP":"5.5.5.5"}'])
    sb = _stub(tmp_path)
    _drive(sb, {T.CHECK_URL: lambda _cmd: (0, next(answers))})
    assert sb.client.networks.get.return_value.disconnect.called is False, (
        "one bad answer from the check endpoint cut the container off")
    assert sb._egress_state == "enforced"
    assert sb._cut_off is False


def test_two_not_tor_answers_DO_cut_the_container_off(tmp_path):
    """Control: corroboration must not disarm the thing it guards. Two
    independent requests both reporting a direct exit is a real leak."""
    from ghost_agent.sandbox import tor_egress as T
    from tests.test_sandbox_tor_egress import _drive, _stub
    sb = _stub(tmp_path)
    _drive(sb, {T.CHECK_URL: (0, b'{"IsTor":false,"IP":"9.9.9.9"}')})
    assert sb.client.networks.get.return_value.disconnect.called is True
    assert sb._cut_off is True


def test_a_large_source_file_is_scanned_WHOLE_not_partly(tmp_path):
    """§4GK round 6, the third version of this fix and the first without a
    trade-off.

    Round 4 REFUSED any source file over the byte cap, which made the whole
    answer inconclusive: one ordinary bundle permanently disabled the media
    tidy for that workspace. Round 5 scanned the HEAD instead — better, but an
    unmatched candidate still fell back to keep-everything, so the tidy still
    could not DELETE anything there. Both were fighting a cap that never
    needed to exist for this question: "does any source file MENTION this
    basename" is a substring search, not a parse. Streamed in bounded chunks
    it is answered COMPLETELY, at one chunk of memory.

    Fails in any tree that stops reading at a cap: the reference below sits 7
    MB into a 9 MB file, past any head, and the unreferenced asset must stay
    deletable at the same time."""
    from ghost_agent.core import workspace_cleanup as wc
    bundle = tmp_path / "bundle.js"
    bundle.write_text("x" * 7_000_000 + 'require("./deep.png");' + "y" * 2_000_000)
    assert bundle.stat().st_size > wc._REFERENCE_SCAN_MAX_BYTES
    (tmp_path / "index.html").write_text('<img src="hero.png">')
    for n in ("hero.png", "deep.png", "stray.png"):
        (tmp_path / n).write_bytes(b"\x89PNG")
    ref = wc._referenced_media(tmp_path, ["hero.png", "deep.png", "stray.png"])
    assert "hero.png" in ref                       # ordinary small source
    assert "deep.png" in ref, (
        "a reference past the old cap was missed — the file was not scanned whole")
    assert "stray.png" not in ref, (
        "a genuinely unreferenced asset was kept, so the tidy is still switched "
        "off rather than made complete")


def test_a_basename_split_across_a_chunk_boundary_is_still_found(tmp_path):
    """The streaming scan carries an overlap of the longest basename, so a
    name straddling a read boundary is not lost. Without it the fix would
    trade one silent miss for another."""
    from ghost_agent.core import workspace_cleanup as wc
    name = "spritesheet-main.png"
    pad = wc._REFERENCE_SCAN_CHUNK_BYTES - 8        # split the name across chunks
    (tmp_path / "app.js").write_text("z" * pad + f'load("{name}");' + "w" * 4096)
    (tmp_path / name).write_bytes(b"\x89PNG")
    assert name in wc._referenced_media(tmp_path, [name])


def test_an_unreadable_source_still_keeps_every_unmatched_candidate(tmp_path):
    """The fail-safe survives: a file the scan cannot read at all is silence
    that is not evidence, so unmatched candidates are kept."""
    from ghost_agent.core import workspace_cleanup as wc
    (tmp_path / "app.js").write_text("nothing here")
    (tmp_path / "stray.png").write_bytes(b"\x89PNG")
    secret = tmp_path / "index.html"
    secret.write_text("<img src='stray.png'>")
    os.chmod(secret, 0o000)
    try:
        assert "stray.png" in wc._referenced_media(tmp_path, ["stray.png"])
    finally:
        os.chmod(secret, 0o644)

def test_the_pending_nonce_window_is_longer_than_any_exec_budget():
    """§4GK round 5. The mark protects the span from `_write_script` to
    `_promote`, and that span is the EXEC BUDGET, not `promote_after_s`: a
    quiet pure-compute command waits out the whole budget with no registry
    row. Round 4's formula, measured at its own documented env floors
    (`JOB_TTL_S=60`, `PROMOTE_AFTER_S=5`), gave a 60 s window against a 600 s
    budget — the victim lost its mark, then its nonce, then landed LOST with a
    fabricated exit code.

    Fails in any tree where the window can fall below an exec budget."""
    import threading
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    sup = SandboxJobSupervisor.__new__(SandboxJobSupervisor)
    sup.sandbox, sup._lock = None, threading.RLock()
    sup._nonces, sup._nonce_pending, sup._nonce_store = {}, {}, None
    import os as _os
    _saved = {k: _os.environ.get(k) for k in
              ("GHOST_SANDBOX_JOB_TTL_S", "GHOST_SANDBOX_JOB_PROMOTE_AFTER_S")}
    try:
        _os.environ["GHOST_SANDBOX_JOB_TTL_S"] = "60"
        _os.environ["GHOST_SANDBOX_JOB_PROMOTE_AFTER_S"] = "5"
        assert sup._nonce_pending_ttl_s() >= 600.0, (
            "at its own documented env floors the pending window is shorter "
            "than an exec budget, which destroys a long job's exit code")
    finally:
        for k, v in _saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v


def test_the_nonce_temp_file_name_is_unique_per_writer(tmp_path):
    """§4GK round 5. A per-pid temp name is shared by every thread, and there
    are three unlocked writers — `_write_script`, `_save`, and the
    `_release_nonce` round 4 added on the commonest path of all. The writer
    opens O_TRUNC, so writer B truncates the temp A is mid-write on and both
    replace it: the file left on disk was corrupt JSON, which makes the store
    load as `{}` on the next restart and every promoted job's genuine sentinel
    is then rejected.

    Fails in any tree where two writers can pick the same temp name."""
    import threading
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    names = set()
    for _ in range(200):
        sup = SandboxJobSupervisor.__new__(SandboxJobSupervisor)
        sup.sandbox, sup._lock = None, threading.RLock()
        sup._nonces, sup._nonce_pending = {"job-aaaaaaaa": "n" * 16}, {}
        sup._nonce_store = tmp_path / "nonces.json"
        seen = []
        _orig = os.replace

        def _spy(a, b, _o=_orig, _s=seen):
            _s.append(str(a))
            return _o(a, b)
        os.replace = _spy
        try:
            sup._save_nonces()
        finally:
            os.replace = _orig
        names.update(seen)
    assert len(names) == 200, (
        f"only {len(names)} distinct temp names across 200 saves — concurrent "
        "writers share one file and truncate each other")


# ── §4GK round 6: defects found INSIDE round 5's fixes ──────────────────────

def test_the_nested_writer_does_not_close_a_descriptor_twice(tmp_path, monkeypatch):
    """§4GK round 6. `write_bytes_nofollow_rel` was born with the exact defect
    its sibling `read_bytes_nofollow_fd` documents twenty lines above:
    `os.fdopen(closefd=True)` takes the descriptor and the `with` closes it on
    the way out even when the body raised, so a manual close in the handler
    closes that descriptor NUMBER again — landing on whatever the kernel has
    since reissued. Measured: four writers doing 400 restores against four
    unrelated readers closed 172 of their open files out from under them.

    Made deterministic by holding the descriptor OPEN: the stand-in file
    object does not close on `__exit__`, so in a correct tree nothing closes
    it and it is still valid afterwards. A handler that closes after
    ownership transferred shows up as a closed descriptor, with no dependence
    on which numbers the kernel happens to reuse.

    Fails in any tree where the manual close covers the post-fdopen window."""
    from ghost_agent.tools.file_system import write_bytes_nofollow_rel
    owned = []
    real_fdopen = os.fdopen

    def _fdopen(fd, *a, **k):
        owned.append(fd)

        class _RaisingHeldOpen:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False             # deliberately does NOT close

            def write(self_inner, _b):
                raise OSError(errno.EIO, "EIO simulated mid-write")
        return _RaisingHeldOpen()

    monkeypatch.setattr(os, "fdopen", _fdopen)
    with pytest.raises(OSError):
        write_bytes_nofollow_rel(tmp_path, "a/b/c.bin", b"payload")
    monkeypatch.undo()
    assert owned, "the writer never reached fdopen — re-point this pin"
    fd = owned[0]
    try:
        os.fstat(fd)                     # still open => nobody closed it by hand
    except OSError as exc:
        raise AssertionError(
            f"the descriptor fdopen took ownership of was closed by hand as "
            f"well ({exc}) — the second close lands on whatever the kernel "
            f"has since reissued") from exc
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

def test_a_planted_DIRECTORY_link_is_removed_not_read_through(tmp_path):
    """§4GK round 6. The unlink-and-retry was applied to the FINAL component
    only. An intermediate component that is a link raised, the entry was
    simply not restored, and the planted link was LEFT IN PLACE — and the
    pre-validator restore runs with no purge, so the validator then read HOST
    bytes the solver chose, presented as the pristine mock. The fix had turned
    a write-through into a read-through and still reported success.

    Fails in any tree that refuses the link without removing it."""
    from ghost_agent.core.dream import _restore_mocks, _snapshot_mocks
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.csv").write_text("pristine mock rows\n")
    snap = _snapshot_mocks(tmp_path)
    host = tmp_path.parent / "hostdir"
    host.mkdir(exist_ok=True)
    (host / "x.csv").write_text("HOST CONTENT")
    shutil.rmtree(tmp_path / "data")
    (tmp_path / "data").symlink_to(host)
    _restore_mocks(tmp_path, snap)
    assert (host / "x.csv").read_text() == "HOST CONTENT"     # never written through
    assert not (tmp_path / "data").is_symlink()               # …and removed
    assert (tmp_path / "data" / "x.csv").read_text() == "pristine mock rows\n"


def test_a_copy_that_produced_no_destination_is_a_failure_not_a_partial(tmp_path):
    """§4GK round 6. Round 5 moved `os.makedirs` inside a per-entry guard so
    one unreachable path could not end the whole copy — correct for a CHILD,
    wrong for the destination ROOT and wrong for a whole-filesystem fault.
    Measured through the real tool: nothing was copied, the destination did
    not exist, and the model was told "Copied 'tree' to 'out/copy'" as a
    PARTIAL carrying world-changed credit and an idempotency record.

    Fails in any tree that reports a total failure as a partial success."""
    import asyncio
    from ghost_agent.tools import file_system as fs
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree" / "a.txt").write_text("a")
    out = tmp_path / "out"
    out.mkdir()
    os.chmod(out, 0o500)
    try:
        res = asyncio.run(fs.tool_copy_file("tree", "out/copy", tmp_path))
        assert str(res).startswith("Error:"), res
        assert getattr(res, "status", None) is None or "partial" not in str(
            getattr(res.status, "value", "")), res
        assert not (out / "copy").exists()
    finally:
        os.chmod(out, 0o700)


def test_the_cut_off_flag_is_cleared_by_a_new_container_generation(tmp_path):
    """§4GK round 6. `_cut_off` was raised by `_block_egress_hard` and lowered
    ONLY by `_recreate_if_cut_off`, which returns early when the container is
    gone, when it is not actually cut off, or while its 300 s backoff stands.
    So a container that died inside that window and was provisioned fresh came
    up healthy and attached with the flag still True — and every later command
    declined the readiness TTL, which is round 4's "TTL disabled for the life
    of the process" defect reached by another route. An arm-on-N with one
    clearer that is not on every path that ends the condition.

    Fails in any tree where the create path leaves the flag raised."""
    import ast
    import inspect as _inspect
    from ghost_agent.sandbox import docker as docker_mod
    src = _inspect.getsource(docker_mod)
    tree = ast.parse(src)
    setters, clearers = [], []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for n in ast.walk(fn):
            if (isinstance(n, ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Attribute)
                    and n.targets[0].attr == "_cut_off"
                    and isinstance(n.value, ast.Constant)):
                (setters if n.value.value is True else clearers).append(fn.name)
    assert set(setters) == {"_block_egress_hard"}, setters
    # every generation boundary must clear it, not just the recreate
    assert "_recreate_if_cut_off" in clearers, clearers
    assert "_ensure_running_impl" in clearers, (
        "the create path provisions a fresh, attached container and does not "
        f"lower _cut_off — clearers are {clearers}")


def test_an_unreadable_self_play_sandbox_is_not_purged(tmp_path):
    """§4GK round 6. `_snapshot_mocks` swallowed every failure and returned
    `{}` — and `_preflight_restore` purges every non-protected file NOT NAMED
    IN THE SNAPSHOT, so an empty snapshot authorises deleting everything the
    setup script created. Round 4 taught the walk to raise on descriptor
    exhaustion and round 5 added the symlinked-root refusal; both land here.
    Round 5 gave the TWIN (`coding_loop.snapshot_workspace`) an explicit
    incomplete marker and left this one.

    Fails in any tree where an unreadable sandbox reads as an empty one."""
    import resource
    from ghost_agent.core.dream import (_SNAPSHOT_INCOMPLETE, _preflight_restore,
                                        _snapshot_mocks)
    for i in range(400):
        (tmp_path / f"d{i:03d}").mkdir()
        (tmp_path / f"d{i:03d}" / "mock.csv").write_text("setup data")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, hard))
    try:
        snap = _snapshot_mocks(tmp_path)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert _SNAPSHOT_INCOMPLETE in snap, "the truncation was silent"
    before = len(list(tmp_path.rglob("*")))
    _preflight_restore(tmp_path, snap)
    assert len(list(tmp_path.rglob("*"))) == before, (
        "the straggler purge ran against a snapshot we never got")


def test_a_complete_self_play_snapshot_still_purges_stragglers(tmp_path):
    """Control: the fail-safe must not disarm the purge it guards. With a
    readable sandbox the cross-attempt leakage cleanup still works."""
    from ghost_agent.core.dream import _preflight_restore, _snapshot_mocks
    (tmp_path / "mock.csv").write_text("setup data")
    snap = _snapshot_mocks(tmp_path)
    (tmp_path / "solution.py").write_text("left over from attempt 1")
    _preflight_restore(tmp_path, snap)
    assert not (tmp_path / "solution.py").exists()
    assert (tmp_path / "mock.csv").read_text() == "setup data"


def test_an_inconclusive_second_probe_does_not_clear_a_measured_leak(tmp_path):
    """§4GK round 6. Round 5 added a re-probe so one bad answer from the check
    endpoint could not destroy the sandbox — but cleared on anything that was
    NOT an explicit second `false`. `parse_tor_check` answers `None` for an
    unparseable body, which includes the sandbox's own infra-error and
    process-failed banners, so an exec that never ran, a timeout or an HTML
    challenge on the SECOND request turned a CONFIRMED direct exit back into
    "enforced" — and logged that the first answer was the bad one. The rule
    the docstring states is "two independent requests both answering not-Tor
    is a leak"; its negation is a second request that AGREES it is Tor, not
    "anything else". The cited base rate (6 unusable answers in 143
    enforcements) is exactly the probability of masking a real leak.

    Fails in any tree where an inconclusive second answer clears a leak."""
    from ghost_agent.sandbox import tor_egress as T
    from tests.test_sandbox_tor_egress import _drive, _stub
    for second in (b"<html>challenge</html>",
                   b"[SANDBOX INFRA ERROR] the daemon is wedged",
                   b""):
        answers = iter([b'{"IsTor":false,"IP":"9.9.9.9"}', second])
        sb = _stub(tmp_path)
        _drive(sb, {T.CHECK_URL: lambda _cmd, _a=answers: (0, next(_a))})
        assert sb.client.networks.get.return_value.disconnect.called is True, (
            f"an inconclusive second answer ({second[:28]!r}) cleared a "
            "measured leak")
        assert sb._cut_off is True


def test_a_failed_second_probe_exec_does_not_clear_a_measured_leak(tmp_path):
    """The same rule for an exec that reports a non-zero code: its body is not
    an answer, whatever it parses to."""
    from ghost_agent.sandbox import tor_egress as T
    from tests.test_sandbox_tor_egress import _drive, _stub
    answers = iter([(0, b'{"IsTor":false,"IP":"9.9.9.9"}'),
                    (124, b'{"IsTor":true,"IP":"5.5.5.5"}')])
    sb = _stub(tmp_path)
    _drive(sb, {T.CHECK_URL: lambda _cmd: next(answers)})
    assert sb.client.networks.get.return_value.disconnect.called is True
    assert sb._cut_off is True


def test_a_surviving_PORT_HOLDER_is_reported_like_a_surviving_pid(tmp_path):
    """§4GK round 6. `_kill_port_holder` called `_kill_pgroup(holder)` and
    threw the answer away, returning True unconditionally — so a reclaim whose
    HOLDER survived TERM+KILL still left the survival channel False, and
    `stop()` reported "stopped", dropped the row and unlinked the pidfile
    while the port was still held. Round 5 fixed exactly this on the sibling
    call site in the same function and left this one.

    Fails in any tree where the port-holder kill discards its verdict."""
    from ghost_agent.sandbox import registry_guard as _rg
    from ghost_agent.sandbox.services import ServiceSupervisor
    sup = ServiceSupervisor.__new__(ServiceSupervisor)
    _orig = ServiceSupervisor.host_dir
    ServiceSupervisor.host_dir = property(lambda self: tmp_path)
    try:
        # every exec answers SURVIVED: the holder outlives TERM+KILL
        sup._exec = lambda cmd, timeout=30: (_rg.SURVIVED_MARKER, 0)
        sup._holder_pid = lambda port: 4242
        sup._pid_ownership = lambda holder, owner: None
        sup._last_kill_survived = False
        assert sup._kill_port_holder(8100, owner_pid=None) is False
        assert sup._last_kill_survived is True, (
            "the port holder survived the kill and nothing said so")
    finally:
        ServiceSupervisor.host_dir = _orig


# ── §4GK round 7: defects found INSIDE round 6's fixes ──────────────────────

def test_a_surviving_port_holder_is_not_overwritten_by_the_pid_verdict(tmp_path):
    """§4GK round 7. Round 6 taught `_kill_port_holder` to record a survival —
    and `_kill_service` overwrote it two statements later with the tracked
    pid's answer alone, so the fix was INERT for every consumer. Measured: a
    service whose tracked pid is dead but whose port is held by a surviving
    orphan still answered "was already dead; removed", dropped the registry row
    and unlinked the pidfile.

    Fails in any tree where either survival can be clobbered by the other."""
    from ghost_agent.sandbox import registry_guard as _rg
    from ghost_agent.sandbox.services import ServiceSupervisor
    sup = ServiceSupervisor.__new__(ServiceSupervisor)
    _orig = ServiceSupervisor.host_dir
    ServiceSupervisor.host_dir = property(lambda self: tmp_path)
    try:
        # the tracked pid is GONE, but the port holder survives TERM+KILL
        def _exec(cmd, timeout=30):
            return (_rg.SURVIVED_MARKER if "S=7777;" in cmd else _rg.KILLED_MARKER, 0)
        sup._exec = _exec
        sup._entry_alive_or_unknown = lambda e: True
        sup._port_listening = lambda port: True
        sup._holder_pid = lambda port: 7777
        sup._pid_ownership = lambda holder, owner: None
        entry = {"pid": 4242, "name": "web", "port": 8100, "project_id": None}
        (tmp_path / "web.pid").write_text("4242")
        sup._last_kill_survived = False
        sup._kill_service(entry)
        assert sup._last_kill_survived is True, (
            "the port holder survived TERM+KILL and the tracked pid's verdict "
            "overwrote that, so the caller was told the service stopped")
        assert (tmp_path / "web.pid").exists()
    finally:
        ServiceSupervisor.host_dir = _orig


def test_an_unreadable_snapshot_retries_instead_of_passing_ungated(tmp_path):
    """§4GK round 7. Round 6 stopped a failed READING being mistaken for "the
    attempt changed nothing" — and did it by falling through with an empty
    written-list, which skips `_run_verify`, `smoke_gate` AND the constraint
    gate (guarded on `written`). The attempt was returned as a SUCCESS with no
    files and no gates, and the task was then marked DONE with no artifacts:
    the task-reaching-DONE-on-evidence-that-no-work-happened class, worse than
    the discard it replaced.

    Fails in any tree that treats an unknown reading as a pass."""
    import ast
    import inspect as _inspect
    from ghost_agent.core import coding_loop as cl
    fn = next(n for n in ast.walk(ast.parse(_inspect.getsource(cl)))
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "build_coding_task_agentic")
    body = ast.unparse(fn)
    # the unknown branch must not fall through into the success path
    i = body.index("_cannot_tell")
    seg = body[i:i + 700]
    assert "continue" in seg, (
        "the unknown-snapshot branch falls through to the gates instead of "
        "retrying — with written == [] every gate is a no-op")


def test_descriptor_pressure_is_not_a_fatal_copy_error():
    """§4GK round 7. Round 6 put EMFILE/ENFILE beside ENOSPC/EROFS/EDQUOT.
    They are not the same kind of thing: a full or read-only filesystem will
    not clear on its own, while running short of descriptors clears the moment
    the walk that consumed them finishes. Treating it as fatal made
    `tool_copy_file` answer a bare error string over a destination tree holding
    0 of 10 files, and the overwrite guard then refused the retry — the
    half-copy-blocks-the-retry failure rounds 4 and 5 closed. It also raises
    BEFORE the stub-unlink, bypassing "NOT COPIED MUST MEAN NOT PRESENT"."""
    from ghost_agent.tools import file_system as fs
    assert errno.ENOSPC in fs._COPY_FATAL_ERRNOS
    assert errno.EROFS in fs._COPY_FATAL_ERRNOS
    assert errno.EMFILE not in fs._COPY_FATAL_ERRNOS, (
        "transient descriptor pressure is treated as a fatal copy error")
    assert errno.ENFILE not in fs._COPY_FATAL_ERRNOS


def test_a_non_ascii_basename_across_a_chunk_boundary_is_found(tmp_path):
    """§4GK round 7. The streaming scan took its overlap in CHARACTERS from
    decoded text while the chunk boundary is a BYTE offset, so a non-ASCII
    basename straddling it was mangled by `errors="replace"` in BOTH windows
    and found in neither. The caller turns that straight into `to_delete` —
    the false DELETE this function's own contract forbids. This box runs a
    Greek locale, so the case is ordinary, not exotic.

    Fails in any tree that measures the overlap in characters."""
    from ghost_agent.core import workspace_cleanup as wc
    for name in ("εικόνα.png", "sprite-sheet.png"):
        d = tmp_path / name.encode("utf-8").hex()[:16]
        d.mkdir()
        # Put the NAME's midpoint exactly on the byte boundary, so a
        # character-measured overlap (shorter than the name's byte length for
        # any non-ASCII name) cannot reach back far enough to rejoin it.
        nb = name.encode("utf-8")
        head = 'load("'.encode("utf-8")
        # Only the LAST two bytes spill past the boundary: the name then
        # starts `len(nb) - 2` bytes before it, which a BYTE-measured overlap
        # reaches and a CHARACTER-measured one (shorter for any non-ASCII
        # name) does not. Placing the midpoint on the boundary is not enough —
        # a 10-character, 16-byte name is still reachable from a 10-byte step
        # back, and the mutant survived that placement.
        pad = wc._REFERENCE_SCAN_CHUNK_BYTES - len(head) - len(nb) + 2
        blob = (b"z" * pad + head + nb + b'");' + b"w" * 8192)
        (d / "app.js").write_bytes(blob)
        (d / name).write_bytes(b"\x89PNG")
        assert name in wc._referenced_media(d, [name]), (
            f"{name} straddling the chunk boundary was reported unreferenced, "
            "which the caller deletes")


def test_the_lesson_verify_restore_cannot_write_through_a_planted_link(tmp_path):
    """§4GK round 7, the sibling one revision behind. `_restore_mocks` was
    migrated onto the nofollow writer in rounds 5 and 6; the two restores
    inside `_verify_lesson_helpful` kept a plain `write_bytes`, which follows
    a link at any component. The purge above them SKIPS names in the snapshot,
    so a link planted at a snapshot-entry name survives to be written through —
    and the second restore runs AFTER the verify solver's turn. The class
    enumeration cannot see it: its own comment says the walk-read rule covers
    no writes at all.

    Fails in any tree where either restore uses a following write."""
    import ast
    import inspect as _inspect
    from ghost_agent.core import dream as dream_mod
    tree = ast.parse(_inspect.getsource(dream_mod))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "_verify_lesson_helpful")
    body = ast.unparse(fn)
    assert "write_bytes_nofollow_rel" in body
    assert ".write_bytes(" not in body, (
        "a restore inside _verify_lesson_helpful still writes through a link")


async def test_a_released_project_is_never_wiped_by_a_workspace_restore(tmp_path):
    """§4GK round 7. Round 6 unfreezes a frozen tree so the wipe can clear a
    directory the ARCHIVE froze — and `projects/` is frozen for a different
    reason: `set_workspace_readonly` makes a RELEASED project 0o555/0o444, and
    the OS half of that immutability was the last thing between a restore and
    the human-attested deliverables inside it. Round 5's `rmtree` could not
    delete through it; round 6 could, silently, reporting success while the
    project rows pointed at a deleted directory.

    Driven through the REAL `load_workspace`. The first version read the
    function's source text (the class the ratchet counts); the second
    re-implemented the wipe loop in the test, which is worse — a change to the
    real route could not reach it, and its mutant survived.

    Fails in any tree where the restore's wipe can reach `projects/`."""
    from tests.test_4gm_round7_routes import _load, _zip_of
    sandbox = tmp_path / "sandbox"
    (sandbox / "projects" / "p1").mkdir(parents=True)
    (sandbox / "projects" / "p1" / "deliverable.md").write_text("attested work")
    (sandbox / "scratch").mkdir()
    (sandbox / "scratch" / "junk.txt").write_text("ordinary workspace file")
    # freeze the released project exactly as `set_workspace_readonly` does
    for q in sorted((sandbox / "projects").rglob("*"), reverse=True):
        os.chmod(q, 0o444 if q.is_file() else 0o555)
    os.chmod(sandbox / "projects", 0o555)
    try:
        await _load(sandbox, _zip_of([("sandbox/fresh.txt", b"from the archive")]))
        assert (sandbox / "projects" / "p1" / "deliverable.md").read_text() == \
            "attested work", "the restore wiped a RELEASED project"
        assert not (sandbox / "scratch").exists(), (
            "an ordinary workspace directory survived the wipe")
        assert (sandbox / "fresh.txt").read_bytes() == b"from the archive"
    finally:
        for q in sorted((sandbox / "projects").rglob("*"), reverse=True):
            try:
                os.chmod(q, 0o700)
            except OSError:
                pass
        try:
            os.chmod(sandbox / "projects", 0o700)
        except OSError:
            pass


def test_the_kill_service_bool_still_means_did_anything_happen(tmp_path):
    """§4GK round 5, pinned properly in round 7 after its mutant survived.

    Round 4 made `_kill_service` return False on survival, which callers could
    not distinguish from "there was nothing alive to kill" — so `stop()`
    reported a process that had just survived TERM+KILL as "was already dead;
    removed", the most misleading answer available. Survival is a THIRD state
    on its own channel, and the bool keeps its original meaning.

    The round-6 pin asserted the channel and never the RETURN, so a mutant
    restoring `if _survived_any: return False` survived it.

    Fails in any tree where survival collapses back into the bool."""
    from ghost_agent.sandbox import registry_guard as _rg
    from ghost_agent.sandbox.services import ServiceSupervisor
    sup = ServiceSupervisor.__new__(ServiceSupervisor)
    _orig = ServiceSupervisor.host_dir
    ServiceSupervisor.host_dir = property(lambda self: tmp_path)
    try:
        sup._exec = lambda cmd, timeout=30: (_rg.SURVIVED_MARKER, 0)
        sup._entry_alive_or_unknown = lambda e: True
        sup._port_listening = lambda port: False
        entry = {"pid": 4242, "name": "web", "port": None, "project_id": None}
        (tmp_path / "web.pid").write_text("4242")
        sup._last_kill_survived = False
        acted = sup._kill_service(entry)
        assert sup._last_kill_survived is True          # the third state
        assert acted is True, (
            "the bool collapsed to False on survival, which the caller cannot "
            "tell from 'there was nothing alive to kill' — that is how a "
            "surviving process gets reported as 'was already dead; removed'")
    finally:
        ServiceSupervisor.host_dir = _orig
