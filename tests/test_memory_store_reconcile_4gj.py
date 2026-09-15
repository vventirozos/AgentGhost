"""§4GJ — the memory substrate's cross-store residuals.

Five defects, one class. Three stores describe ONE document population (the
vector rows, `library_index.json`, `document_outlines.json`) and a fourth
pair spans two stores entirely (episode rows ↔ their vector twins). Every
one of these was a TWO-STEP invariant held together by a comment:

  * `reset_all` deleted rows under the lock and reset the catalogue OUTSIDE
    it, afterwards — an ingest landing in between survived the wipe while
    its catalogue line was erased, so the rows were queryable but invisible
    to `list_docs` and un-re-ingestable (the dedup reads the catalogue);
  * `delete_document_by_name` took the lock three times — death between them
    left a catalogue entry with no rows, which the dedup then refused to
    re-ingest while `outline` reported "no indexed chunks";
  * `frontier._save` skipped the §4M fsync sweep — a power loss after the
    rename publishes an empty file, which `_load` quarantines as corrupt,
    restarting self-play with no cluster mastery;
  * a failed episode twin-delete was swallowed whole, and episodes are
    excluded from `_prune_if_needed`, so the orphan vector outlived its row
    forever;
  * nothing reconciled the pairs after the fact.

Every pin below names the world in which it fails: the pre-fix code.
"""
import json
import os
import sqlite3
import tempfile
import threading
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory.episodes import EpisodicMemory
from ghost_agent.memory.frontier import FrontierTracker
from ghost_agent.memory.vector import VectorMemory


def _mem() -> VectorMemory:
    return VectorMemory(Path(tempfile.mkdtemp()), "http://mock-url")


def _doc_sources(vm) -> set:
    rows = vm.collection.get(where={"type": "document"}, include=["metadatas"])
    return {m.get("source") for m in (rows.get("metadatas") or []) if m}


# ── R1: the class, from the AST ──────────────────────────────────────────────

_SIDECAR_CALLS = {"_update_library_index", "set_document_outline",
                  "drop_document_outline"}
_ROW_MUTATIONS = {"delete", "upsert", "add"}


def _two_step_methods(source: str):
    """(method, unlocked_calls) for every method in `source` that mutates
    BOTH the vector collection and a catalogue sidecar.

    The invariant: such a method must perform those mutations inside ONE
    `with self._get_lock():` block. A method that touches only one store is
    not part of this class and is not reported.
    """
    import ast

    tree = ast.parse(source)
    out = []
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        # Map every node to its enclosing `with self._get_lock()` depth.
        locked_nodes = set()
        for node in ast.walk(fn):
            if not isinstance(node, ast.With):
                continue
            if not any("_get_lock" in ast.dump(item.context_expr)
                       for item in node.items):
                continue
            for inner in ast.walk(node):
                locked_nodes.add(id(inner))

        rows, sidecars, unlocked = [], [], []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not isinstance(f, ast.Attribute):
                continue
            is_row = (f.attr in _ROW_MUTATIONS
                      and isinstance(f.value, ast.Attribute)
                      and f.value.attr == "collection")
            is_sidecar = f.attr in _SIDECAR_CALLS and isinstance(f.value, ast.Name)
            if not (is_row or is_sidecar):
                continue
            (rows if is_row else sidecars).append(f.attr)
            if id(node) not in locked_nodes:
                unlocked.append(f"{f.attr}@{node.lineno}")
        if rows and sidecars:
            out.append((fn.name, unlocked))
    return out


def test_every_two_step_store_mutation_is_one_critical_section():
    """The enumeration that closes the class. `ingest_document` already had
    this shape; `delete_document_by_name` did not (three acquisitions), and
    a future method that pairs a row write with a catalogue write and forgets
    the lock reddens this."""
    import inspect

    import ghost_agent.memory.vector as vector_mod

    methods = _two_step_methods(inspect.getsource(vector_mod))
    assert {m for m, _ in methods} >= {"ingest_document", "delete_document_by_name"}, methods
    offenders = [(m, u) for m, u in methods if u]
    assert offenders == [], f"two-step store mutations outside the lock: {offenders}"


def test_the_enumeration_fires_when_a_sidecar_write_escapes_the_lock():
    """R7-2: the instrument must be shown to fail. Pull the catalogue update
    out of `delete_document_by_name`'s locked block — the pre-fix shape."""
    import inspect

    import ghost_agent.memory.vector as vector_mod

    src = inspect.getsource(vector_mod)
    needle = """        with self._get_lock():
            self.collection.delete(where={"source": filename})
            self._update_library_index(filename, "remove")"""
    assert src.count(needle) == 1
    broken = src.replace(needle, """        with self._get_lock():
            self.collection.delete(where={"source": filename})
        self._update_library_index(filename, "remove")
        if True:""")
    offenders = [(m, u) for m, u in _two_step_methods(broken) if u]
    assert [m for m, _ in offenders] == ["delete_document_by_name"], offenders


# ── delete_document_by_name: one critical section ────────────────────────────

def test_forget_a_document_takes_the_lock_once_and_clears_all_three():
    """Pre-fix: three acquisitions, so a writer could interleave between the
    rows and the catalogue — and a crash there left a catalogue entry whose
    document could be neither queried nor re-ingested."""
    vm = _mem()
    vm.ingest_document("a.pdf", ["alpha one", "alpha two"])
    vm.set_document_outline("a.pdf", {"source": "toc", "entries": [{"title": "I"}]})
    assert vm.get_library() == ["a.pdf"] and vm.get_document_outline("a.pdf")

    # The lock is REENTRANT and the sidecar helpers re-take it, so counting
    # `__enter__` counts re-entries. The property is that the lock is never
    # fully RELEASED between the first store touch and the last catalogue
    # write — i.e. the depth returns to zero exactly once, at the end.
    # Pre-fix the depth hit zero twice mid-way (rows | catalogue | outline).
    spans = []
    real_lock = vm._get_lock()

    class _Recorder:
        def __enter__(self):
            real_lock.acquire()
            spans.append("in")

        def __exit__(self, *a):
            spans.append("out")
            real_lock.release()
            return False

    vm._get_lock = lambda: _Recorder()
    vm.delete_document_by_name("a.pdf")

    depth, zero_crossings = 0, 0
    for ev in spans:
        depth += 1 if ev == "in" else -1
        if depth == 0:
            zero_crossings += 1
    assert spans and depth == 0, spans
    assert zero_crossings == 1, (
        f"the lock was released mid-sequence — a writer can interleave "
        f"between the rows and the catalogue: {spans}")
    assert _doc_sources(vm) == set()
    assert vm.get_library() == []
    assert vm.get_document_outline("a.pdf") == {}


# ── frontier: the store the §4M fsync sweep missed ───────────────────────────

def test_frontier_save_fsyncs_before_the_rename(monkeypatch, tmp_path):
    """Pre-fix: `tmp.write_text()` + `os.replace` — atomic in the namespace,
    but the bytes may still be in the page cache, so a power loss promotes an
    empty file that `_load` then quarantines as corrupt."""
    order = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace",
                        lambda a, b: (order.append("replace"), real_replace(a, b))[1])

    tracker = FrontierTracker(tmp_path)          # __init__ writes the initial state
    assert order[:2] == ["fsync", "replace"], order
    assert json.loads((tmp_path / "self_play_frontier.json").read_text())["runs"] == []


def test_frontier_temp_file_is_pid_unique_and_removed(tmp_path):
    """A FIXED `.tmp` name is the §4BW shape: a second writer sharing the
    directory truncates the temp the first is mid-write on."""
    tracker = FrontierTracker(tmp_path)
    tracker._save({"runs": [], "clusters": {"c": {}}})
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == [], f"temp files left behind: {leftovers}"
    # The name carries this process's pid, so two processes cannot collide.
    captured = {}
    real_replace = os.replace

    def _spy(a, b):
        captured["tmp"] = str(a)
        return real_replace(a, b)

    os.replace = _spy
    try:
        tracker._save({"runs": [], "clusters": {}})
    finally:
        os.replace = real_replace
    assert str(os.getpid()) in captured["tmp"], captured


# ── episodes: a failed twin-delete is reported, not swallowed ────────────────

class _RefusingVectors:
    """A vector store whose episode deletes fail — the world in which the
    orphan is created."""

    def __init__(self, mode):
        self.mode = mode
        self.added = []

    def add(self, text, meta=None):
        self.added.append((text, meta))

    def forget_episode(self, episode_id):
        if self.mode == "raise":
            raise RuntimeError("chroma refused")
        return False


@pytest.mark.parametrize("mode", ["raise", "false"])
def test_a_failed_episode_twin_delete_is_reported(tmp_path, caplog, mode):
    """Pre-fix: `except Exception: pass`. The episode row is committed and
    gone, the vector outlives it, nothing reaps it (episodes are excluded
    from `_prune_if_needed`) — and the log said nothing at all."""
    em = EpisodicMemory(tmp_path)
    em.MAX_EPISODES = 2
    vm = _RefusingVectors(mode)
    with caplog.at_level("WARNING"):
        for i in range(4):
            em.record_episode(f"trigger {i}", lesson=f"lesson {i}",
                              vector_memory=vm)
    assert em.count() <= 2, "the cap did not evict, so nothing was orphaned"
    assert any("kept their vector twin" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]


def test_forget_episode_reports_failure_at_the_PRODUCER(caplog):
    """The other end of the contract `record_episode` depends on.

    Its sibling above drives a STUBBED `forget` that returns False, so it
    pins the caller's reaction and nothing about the real method. With only
    that pin, inverting `return False` → `return True` in the except branch
    leaves the caller silently blind again — the exact §4GJ defect, restored
    from the producer side. This drives the REAL method down its own failure
    path (survived a mutation battery otherwise)."""
    vm = _mem()
    vm.add("episode one", {"type": "episode", "episode_id": 11})

    def _boom(**kw):
        raise RuntimeError("chroma refused the delete")

    real_delete = vm.collection.delete
    vm.collection.delete = _boom
    try:
        with caplog.at_level("WARNING"):
            assert vm.forget_episode(11) is False, (
                "a delete that RAISED reported success — the caller books no "
                "orphan and nothing ever reaps the vector")
    finally:
        vm.collection.delete = real_delete
    assert any("forget_episode(11) failed" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]

    # …and the success path returns True and actually removes the row, so
    # "always False" is not a way to pass the assertion above.
    assert vm.forget_episode(11) is True
    assert vm.collection.get(where={"type": "episode"})["ids"] == []


def test_forget_episode_reports_failure_on_an_unusable_id(caplog):
    """`int(episode_id)` is inside the same try: a non-numeric id is a
    failure to delete, not a silent success."""
    vm = _mem()
    with caplog.at_level("WARNING"):
        assert vm.forget_episode("not-a-number") is False
    assert any("forget_episode(not-a-number) failed" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_live_episode_ids_is_complete_and_none_when_unreadable(tmp_path):
    """The reconciler DELETES the complement of this set, so a truncated
    answer would reap live episodes and an unreadable store must say None,
    never an empty set."""
    em = EpisodicMemory(tmp_path)
    ids = {em.record_episode(f"t{i}", lesson="l") for i in range(3)}
    assert em.live_episode_ids() == ids
    em.db_path = tmp_path / "nope" / "missing.db"
    assert em.live_episode_ids() is None


# ── the reconciler ───────────────────────────────────────────────────────────

def test_reconcile_repairs_all_three_document_drifts():
    vm = _mem()
    vm.ingest_document("real.pdf", ["kept alpha"])
    # (a) catalogue entry whose rows are gone (a crash between the two steps)
    vm._update_library_index("ghost.pdf", "add")
    # (b) rows whose catalogue line is gone (the reset_all race)
    vm.ingest_document("unlisted.pdf", ["orphan beta"])
    vm._update_library_index("unlisted.pdf", "remove")
    # (c) an outline for a document in neither
    vm.set_document_outline("gone.pdf", {"source": "toc", "entries": []})
    # (d) …and an outline for the document arm ADOPTS in this same pass. The
    # outline arm must judge against the catalogue AS REPAIRED, not the
    # snapshot taken before it — otherwise a document re-listed one line
    # earlier has its structure dropped as "unlisted" (§4GJ battery).
    vm.set_document_outline("unlisted.pdf", {"source": "toc", "entries": [{"t": "I"}]})

    report = vm.reconcile_indexes(live_episode_ids=None)

    assert report["catalogue_dropped"] == ["ghost.pdf"]
    assert report["catalogue_adopted"] == ["unlisted.pdf"]
    assert report["outlines_dropped"] == ["gone.pdf"]
    assert sorted(vm.get_library()) == ["real.pdf", "unlisted.pdf"]
    assert vm.get_document_outline("gone.pdf") == {}
    assert vm.get_document_outline("unlisted.pdf"), (
        "the outline of a document adopted in THIS pass was dropped — the "
        "arm judged against the stale pre-repair catalogue")
    # the real document is untouched by all three arms
    assert "real.pdf" in vm.get_library() and _doc_sources(vm) >= {"real.pdf"}


def test_reconcile_reaps_only_orphan_episode_vectors():
    """The reaped id is an OLD one — episode 2 with 1 and 3 alive — because
    that is what an orphan looks like: episodes are evicted from the OLDEST,
    least-used tail, and `forget_episode` already drops the twin of anything
    deleted by name. An id ABOVE everything the live set names is the
    §4GJ-round-5 race (an episode recorded after the set was read) and has
    its own pin next door."""
    vm = _mem()
    for ep in (1, 2, 3):
        vm.add(f"episode trigger {ep}", {"type": "episode", "episode_id": ep})
    vm.add("an ordinary auto memory", {"type": "auto"})
    vm.ingest_document("d.pdf", ["doc chunk"])

    # …and rows whose episode_id is missing or unusable: they cannot be
    # PROVEN orphaned, so the reaper must leave them (§4GJ battery).
    vm.add("episode with no id at all", {"type": "episode"})
    vm.add("episode with a garbage id", {"type": "episode", "episode_id": "n/a"})

    report = vm.reconcile_indexes(live_episode_ids={1, 3})

    assert report["episode_vectors_deleted"] == 1
    eps = vm.collection.get(where={"type": "episode"}, include=["metadatas"])
    kept = [m.get("episode_id") for m in eps["metadatas"]]
    assert sorted(str(k) for k in kept) == ["1", "3", "None", "n/a"], (
        f"a row with no provable episode_id was reaped: {kept}")
    # nothing else was touched
    assert vm.collection.get(where={"type": "auto"})["ids"]
    assert vm.get_library() == ["d.pdf"]


def test_reconcile_is_a_no_op_on_a_healthy_store():
    vm = _mem()
    vm.ingest_document("a.pdf", ["alpha"])
    vm.set_document_outline("a.pdf", {"source": "toc", "entries": []})
    vm.add("episode one", {"type": "episode", "episode_id": 7})

    report = vm.reconcile_indexes(live_episode_ids={7})

    assert report["catalogue_dropped"] == [] and report["catalogue_adopted"] == []
    assert report["outlines_dropped"] == [] and report["episode_vectors_deleted"] == 0
    assert report["skipped"] == []
    assert vm.get_library() == ["a.pdf"] and vm.get_document_outline("a.pdf")


def test_unreadable_episode_ids_skip_the_arm_instead_of_reaping_everything():
    """The trap this guard exists for: reconciling episode vectors against a
    set that failed to load reads EVERY episode vector as an orphan."""
    vm = _mem()
    for ep in (1, 2, 3):
        vm.add(f"episode {ep}", {"type": "episode", "episode_id": ep})

    report = vm.reconcile_indexes(live_episode_ids=None)

    assert report["episode_vectors_deleted"] == 0
    assert any("episode ids unavailable" in s for s in report["skipped"]), report
    assert len(vm.collection.get(where={"type": "episode"})["ids"]) == 3


def test_a_failed_scan_changes_nothing():
    """Fail-SAFE, not fail-clean: without a successful read nothing can be
    PROVEN orphaned, so nothing is repaired."""
    vm = _mem()
    vm.ingest_document("a.pdf", ["alpha"])
    vm._update_library_index("ghost.pdf", "add")

    before = vm.library_file.read_bytes()
    real_get = vm.collection.get
    vm.collection.get = lambda **kw: (_ for _ in ()).throw(RuntimeError("store down"))
    try:
        report = vm.reconcile_indexes(live_episode_ids={1})
    finally:
        vm.collection.get = real_get

    assert report["catalogue_dropped"] == [] and report["catalogue_adopted"] == []
    assert sorted(vm.get_library()) == ["a.pdf", "ghost.pdf"], "it repaired blind"
    assert vm.library_file.read_bytes() == before, "the catalogue file was rewritten"
    # ⚠ The REASON, exactly. A `return` that becomes a fallthrough leaves
    # `rows` unbound, so the arm dies of UnboundLocalError into the OUTER
    # handler: the catalogue survives by accident and the report says
    # "aborted:" instead of naming the invariant that was skipped. Handled
    # skip ≠ crash, and only the reason tells them apart (§4GJ battery).
    assert report["skipped"] == ["document scan failed: store down",
                                 "episode vector scan failed: store down"], report
    assert not any(s.startswith("aborted") for s in report["skipped"]), (
        "a failed scan must be a HANDLED skip, not an abort")


def test_reconcile_is_bounded_per_pass():
    """⚠ The store is shaped like a PRODUCTION one on purpose (§4GJ round
    5). This pin used to hold six catalogue lines and nothing else, and was
    green only because the collection held zero rows of ANY kind: one
    ordinary `auto` memory turned the whole arm off, because the
    "is the empty document sweep real?" corroboration compared a
    document-scoped sweep against `collection.count()` over every row. A
    store that holds a memory and an episode and no documents is the most
    ordinary state there is."""
    vm = _mem()
    vm.add("an ordinary auto memory the agent accreted", {"type": "auto"})
    vm.add("an episode twin", {"type": "episode", "episode_id": 1})
    for i in range(6):
        vm._update_library_index(f"ghost{i}.pdf", "add")

    report = vm.reconcile_indexes(live_episode_ids=None, max_repairs=2)

    assert len(report["catalogue_dropped"]) == 2
    assert report["bounded"] is True
    assert len(vm.get_library()) == 4, "an unbounded pass would empty it in one go"


def test_reconcile_never_raises_on_a_broken_store():
    """§4GJ round 4 rewrote this pin. It used to reach the "catalogue
    unreadable" skip by replacing `vm.get_library` with a raiser — but
    `get_library` catches EVERY exception and answers `[]`, so no real store
    state could enter that branch: the monkeypatch was manufacturing the
    guard it then reported as covered. This drives a real corrupt
    `library_index.json`, which is the state that actually occurs."""
    vm = _mem()
    vm.ingest_document("real.pdf", ["alpha"])
    vm.set_document_outline("real.pdf", {"source": "toc", "entries": [{"t": "I"}]})
    vm.library_file.write_text("{not json at all")

    # The forgiving reader's contract is unchanged — `list_docs` must not
    # die on a corrupt index…
    assert vm.get_library() == []
    # …and the reconciler, the one caller that DELETES against the answer,
    # must not read that `[]` as "nothing is listed".
    before = vm.library_file.read_bytes()
    report = vm.reconcile_indexes(live_episode_ids={1})

    assert isinstance(report, dict)
    assert any("catalogue unreadable" in s for s in report["skipped"]), report
    # ⚠ What this pin may and may not freeze (§4GJ round 5). It used to
    # assert `catalogue_adopted == []` as well, which states "a corrupt
    # catalogue can NEVER be rebuilt from the rows" as the contract — and
    # `_update_library_index` now quarantines the bytes and rebuilds, so a
    # future revision could legitimately let the adopt arm self-heal here.
    # The invariant that must never move is the DESTRUCTIVE one: nothing may
    # be deleted on the word of a catalogue this pass could not read, and
    # the bytes must stay recoverable.
    assert report["catalogue_dropped"] == [] and report["outlines_dropped"] == []
    assert vm.get_document_outline("real.pdf"), "an outline was dropped"
    quarantine = vm.library_file.with_suffix(vm.library_file.suffix + ".corrupt")
    assert (vm.library_file.read_bytes() == before
            or quarantine.read_bytes() == before), (
        "the corrupt catalogue was neither left alone nor preserved")


def test_reconcile_swallows_an_error_the_inner_handlers_do_not_catch():
    """"Never raises" must hold for the OUTER handler too — housekeeping
    runs inside a dream cycle and must not take it down.

    The inner arms catch their own reads, so an inner-arm failure never
    reaches the outer `except`; this drives a step NO inner handler guards
    (the lock itself) with a non-ValueError. Narrowing that `except
    Exception` to `except ValueError` is invisible to every other pin
    (§4GJ battery)."""
    vm = _mem()

    def _boom():
        raise RuntimeError("lock is wedged")

    vm._get_lock = _boom
    report = vm.reconcile_indexes(live_episode_ids={1})     # must not raise
    assert isinstance(report, dict)
    assert any(s.startswith("aborted: ") and "lock is wedged" in s
               for s in report["skipped"]), report


# ── reset_all clears BOTH catalogues ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_reset_all_clears_the_outline_sidecar_too(tmp_path):
    """§4FO: a wiped document that keeps its outline serves that structure to
    the next same-named ingest until the ingest finishes."""
    from ghost_agent.tools.memory import tool_knowledge_base

    mem = MagicMock()
    mem.collection.get = lambda **kw: {"ids": ["1"], "metadatas": [{"type": "document"}]}
    mem.collection.delete = lambda ids=None: None
    lib = tmp_path / "library_index.json"
    lib.write_text(json.dumps(["doc.pdf"]))
    outlines = tmp_path / "document_outlines.json"
    outlines.write_text(json.dumps({"doc.pdf": {"entries": []}}))
    mem.library_file, mem.outlines_file = lib, outlines

    out = await tool_knowledge_base(action="reset_all", memory_system=mem)

    assert "Wiped clean" in out
    assert lib.read_text() == "[]"
    assert outlines.read_text() == "{}", "the outline sidecar survived the wipe"


@pytest.mark.asyncio
async def test_a_failed_wipe_leaves_both_catalogues_alone(tmp_path):
    """CONTROL — both worlds agree (R4). The pre-fix code already guarded the
    library reset on `not failed_batches`; this pins that the guard survived
    the rewrite and now covers the outline sidecar as well. It is the only
    pin in this file that passes on the pre-fix tree, and it is here to catch
    a fix that "simplifies" the guard away."""
    from ghost_agent.tools.memory import tool_knowledge_base

    mem = MagicMock()
    mem.collection.get = lambda **kw: {"ids": ["1"], "metadatas": [{"type": "document"}]}

    def _boom(ids=None):
        raise RuntimeError("delete refused")

    mem.collection.delete = _boom
    lib = tmp_path / "library_index.json"
    lib.write_text(json.dumps(["doc.pdf"]))
    outlines = tmp_path / "document_outlines.json"
    outlines.write_text(json.dumps({"doc.pdf": {"entries": []}}))
    mem.library_file, mem.outlines_file = lib, outlines

    out = await tool_knowledge_base(action="reset_all", memory_system=mem)

    assert "PARTIAL" in out and "left in place" in out
    assert json.loads(lib.read_text()) == ["doc.pdf"]
    assert json.loads(outlines.read_text()) == {"doc.pdf": {"entries": []}}


# ── the reconciler's own wiring (built-but-unwired is a known shape here) ────

def _dream_self(vm, em):
    """The slice of DreamEngine `_reconcile_memory_stores` reads. `vm`/`em`
    are REAL components — `_is_real_component` gates on the module name, so
    a local test double would silently no-op the whole method."""
    from types import SimpleNamespace
    return SimpleNamespace(memory=vm, context=SimpleNamespace(episodic_memory=em))


async def test_the_dream_reconciler_forwards_the_LIVE_episode_ids(tmp_path):
    """Pre-fix/mutant: passing `set()` instead of the live ids reaps EVERY
    episode vector — the reconciler becomes a shredder. Driven through the
    real method with a real store on both sides.

    §4GK round 6: the consumer now forwards the GETTER rather than its answer,
    so the ids are read inside the reconciler's own lock. What this pin
    guarantees is unchanged and is stated as such — the reconciler must end up
    with the REAL live ids, never an empty set — so it asserts the ids the
    reconciler actually sees, not the shape of the argument."""
    from ghost_agent.core.dream import Dreamer

    em = EpisodicMemory(tmp_path)
    evicted_id = em.record_episode("an episode later evicted", lesson="gone")
    live_id = em.record_episode("a live episode", lesson="keep me")
    with closing(sqlite3.connect(em.db_path)) as conn:
        conn.execute("DELETE FROM episodes WHERE id = ?", (evicted_id,))
        conn.commit()
    vm = _mem()
    vm.add("the live episode's twin", {"type": "episode", "episode_id": live_id})
    vm.add("an orphan twin", {"type": "episode", "episode_id": evicted_id})

    forwarded = []
    real_recon = vm.reconcile_indexes
    vm.reconcile_indexes = lambda ids=None, **kw: (forwarded.append(ids),
                                                   real_recon(ids, **kw))[1]

    summary = await Dreamer._reconcile_memory_stores(_dream_self(vm, em))

    _seen = [f() if callable(f) else f for f in forwarded]
    assert _seen == [{live_id}], (
        f"the live id set was not forwarded: {_seen} — an empty set here "
        f"reaps every episode vector")
    kept = [m.get("episode_id") for m in
            vm.collection.get(where={"type": "episode"}, include=["metadatas"])["metadatas"]]
    assert kept == [live_id], f"the live episode's twin was reaped: {kept}"
    assert "orphan episode vector" in summary, summary


async def test_the_dream_reconciler_actually_repairs_and_reports(tmp_path):
    """The method is not a hollow shell: it runs the real reconcile and
    returns a summary the dream can print. An empty return means the REM
    cycle reports nothing even when the store was repaired."""
    from ghost_agent.core.dream import Dreamer

    vm = _mem()
    vm.ingest_document("real.pdf", ["alpha"])
    vm._update_library_index("ghost.pdf", "add")          # a catalogue entry with no rows

    summary = await Dreamer._reconcile_memory_stores(_dream_self(vm, None))

    assert "memory reconcile:" in summary and "catalogue" in summary, summary
    assert vm.get_library() == ["real.pdf"]


async def test_a_healthy_store_reports_nothing_from_the_dream_hook(tmp_path):
    """CONTROL — the hook must be silent when there is nothing to repair, so
    the dream's metrics note does not grow a line every cycle."""
    from ghost_agent.core.dream import Dreamer

    vm = _mem()
    vm.ingest_document("real.pdf", ["alpha"])
    assert await Dreamer._reconcile_memory_stores(_dream_self(vm, None)) == ""


def test_the_rem_cycle_awaits_the_reconciler():
    """Supplement to the two behavioural pins above: the method can be
    perfect and still never run. `built-but-unwired-loops` is a documented
    failure shape in this project — five idle subsystems were inert on the
    live agent for weeks while their own tests passed."""
    import ast
    import inspect

    from ghost_agent.core.dream import Dreamer

    tree = ast.parse(inspect.getsource(Dreamer.dream).lstrip())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_reconcile_memory_stores"]
    assert len(calls) == 1, (
        "the REM cycle does not call _reconcile_memory_stores — the "
        "reconciler is built but unwired")
    awaited = [n for n in ast.walk(tree)
               if isinstance(n, ast.Await) and isinstance(n.value, ast.Call)
               and getattr(n.value.func, "attr", "") == "_reconcile_memory_stores"]
    assert awaited, "the call is not awaited — the coroutine never runs"
