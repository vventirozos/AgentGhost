"""§4GJ round 4 — the reconciler's licence to delete.

The §4GJ reaper closed the cross-store drift. Round 4 read it again, from
the other side: not "does it repair?" but "what does it destroy when one of
its inputs lies?" — because this thing runs UNATTENDED in the dream/REM
cycle, and this project's ledger is full of maintenance passes that ate
memory content (`skill-prune OFF by default`, `self-learning-stack-audit`,
`sweep partial-keepset wipe`).

Five of the six findings share one shape: a failure that comes back EMPTY
rather than raising, and an arm that reads "empty" as "proven absent".

  * a document sweep that returns successfully and empty → every catalogue
    line dropped and every outline with it, while the rows sit there;
  * an adopt arm bounded at the repair cap → the outline arm judging against
    the half of the catalogue adoption never reached, and deleting the rest;
  * `add()` replacing an existing row's metadata WHOLE → a user's fact
    relabelled `type=episode` by an unattended re-ingest, then reaped;
  * `live_episode_ids()` answering `set()` for a store whose schema was just
    created → every episode vector read as an orphan;
  * `get_library()` swallowing a corrupt catalogue into `[]` → the
    reconciler's own "unreadable" skip unreachable, and every live document
    reading as unlisted.

Every pin below names the world in which it fails: the pre-fix code.
"""
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from ghost_agent.memory.episodes import EpisodicMemory
from ghost_agent.memory.vector import VectorMemory


def _mem() -> VectorMemory:
    return VectorMemory(Path(tempfile.mkdtemp()), "http://mock-url")


def _doc_sources(vm) -> set:
    rows = vm.collection.get(where={"type": "document"}, include=["metadatas"])
    return {m.get("source") for m in (rows.get("metadatas") or []) if m}


def _blinkered_sweep(vm, reveal=()):
    """Make the `type == document` sweep return SUCCESSFULLY but report only
    `reveal` — the failure this class is about. Every other query (the
    targeted `source` probe, `count()`, the episode sweep) stays real, which
    is what a broken/renamed/changed `type` filter actually looks like.

    Returns the real `get` so the caller can restore it."""
    real = vm.collection.get
    wanted = set(reveal)

    def _get(**kw):
        if (kw.get("where") or {}) == {"type": "document"}:
            res = real(**kw)
            keep = [i for i, m in enumerate(res.get("metadatas") or [])
                    if (m or {}).get("source") in wanted]
            return {"ids": [(res.get("ids") or [])[i] for i in keep],
                    "metadatas": [(res.get("metadatas") or [])[i] for i in keep]}
        return real(**kw)

    vm.collection.get = _get
    return real


# ── 1. an empty sweep is a failed sweep ──────────────────────────────────────

def test_an_empty_document_sweep_is_a_failed_scan_not_an_empty_library():
    """Pre-fix the arm guarded only the EXCEPTION. A sweep that returned and
    was empty gave `live_sources = set()`, which every later line read as
    "no document anywhere has rows": the whole catalogue was dropped, and
    the outline arm — judging against the now-empty repaired catalogue —
    dropped every outline with it. Measured on the pre-fix tree with one
    real document: `catalogue_dropped: ['keep.pdf'], outlines_dropped:
    ['keep.pdf']`, rows untouched. The live store holds exactly one
    document, the PostgreSQL manual, and `reconcile_indexes` does not
    rebuild an outline it deletes."""
    vm = _mem()
    vm.ingest_document("keep.pdf", ["the only chunk this document has"])
    vm.set_document_outline("keep.pdf", {"source": "toc", "entries": [{"t": "I"}]})

    real = _blinkered_sweep(vm, reveal=())
    try:
        report = vm.reconcile_indexes(live_episode_ids=None)
    finally:
        vm.collection.get = real

    assert report["catalogue_dropped"] == [], report
    assert report["outlines_dropped"] == [], report
    assert vm.get_library() == ["keep.pdf"]
    assert vm.get_document_outline("keep.pdf"), "the outline was reaped"
    assert any("FAILED scan" in s for s in report["skipped"]), report
    # …and the rows were there the whole time, which is the point.
    assert _doc_sources(vm) == {"keep.pdf"}


def test_a_short_sweep_cannot_drop_a_document_whose_rows_are_there():
    """The partial version, and why a deletion needs its own proof: a sweep
    that comes back SHORT (an under-filled page, a filter the store version
    stopped honouring) under-reports the live set, and under-reporting a
    live document is exactly what drops its catalogue line and its outline.
    Here the sweep sees a.pdf and misses b.pdf; b.pdf has rows."""
    vm = _mem()
    vm.ingest_document("a.pdf", ["alpha chunk"])
    vm.ingest_document("b.pdf", ["beta chunk"])
    vm.set_document_outline("b.pdf", {"source": "toc", "entries": [{"t": "II"}]})

    real = _blinkered_sweep(vm, reveal={"a.pdf"})
    try:
        report = vm.reconcile_indexes(live_episode_ids=None)
    finally:
        vm.collection.get = real

    assert report["catalogue_dropped"] == [], report
    assert report["outlines_dropped"] == [], report
    assert sorted(vm.get_library()) == ["a.pdf", "b.pdf"]
    assert vm.get_document_outline("b.pdf"), (
        "b.pdf's outline was dropped on the word of a sweep that never "
        "mentioned it")


def test_a_store_with_no_documents_still_drops_its_ghost_catalogue_entries():
    """CONTROL — the guard must refuse an UNCORROBORATED empty sweep, not
    disable the arm.

    ⚠ And the store is shaped like a real one (§4GJ round 5). This pin used
    to hold ONE catalogue line and zero rows, and passed only because of the
    zero: the corroboration compared a sweep scoped `type == "document"`
    against `collection.count()` over EVERY row, so a single auto memory —
    or an episode twin, or a skill, i.e. any store that has ever been used —
    made "zero live documents" unprovable and skipped BOTH arms forever,
    while `ingest_document`'s dedup refuses to re-ingest the name that
    residue line is holding. Measured on exactly this store: "read as a
    FAILED scan". The corroboration has to ask the same QUESTION by a
    different route, which is the targeted `source` probe each deletion is
    proven by anyway.

    ⚠ THE OUTLINE HALF OF THIS PIN WAS REWRITTEN IN §4GK ROUND 6, because it
    pinned the defect as the contract. "No document anywhere has rows" is
    exactly what a store whose rows were LOST looks like (a partial restore,
    a recreated collection), and the catalogue rebuilds itself from a
    re-ingest while an outline does not — `derive_document_outline` reads the
    chunks that went missing. The catalogue line is still dropped here (it
    blocks re-ingest, and that is the repair this state needs); the outline
    is now KEPT, with a reason."""
    vm = _mem()
    vm.add("an ordinary auto memory the agent accreted", {"type": "auto"})
    vm.add("an episode twin about something", {"type": "episode",
                                               "episode_id": 1})
    vm._update_library_index("ghost.pdf", "add")
    vm.set_document_outline("ghost.pdf", {"source": "toc", "entries": []})

    report = vm.reconcile_indexes(live_episode_ids={1})

    assert report["catalogue_dropped"] == ["ghost.pdf"], report
    assert report["outlines_dropped"] == [], report
    assert vm.get_document_outline("ghost.pdf") == {"source": "toc",
                                                    "entries": []}
    assert any("does not" in s and "outline" in s
               for s in report["skipped"]), report
    assert vm.get_library() == []
    # …and the rows that made the store "non-empty" are untouched.
    assert vm.collection.get(where={"type": "auto"})["ids"]
    assert vm.collection.get(where={"type": "episode"})["ids"]


# ── 2. a bounded pass must not let a later arm act on what it missed ─────────

def test_a_bounded_adoption_does_not_reap_the_outlines_it_did_not_reach():
    """Pre-fix: the adopt loop broke at `cap` and set `bounded=True`, and the
    outline arm then dropped every outline missing from a catalogue that was
    — by construction — missing the live documents adoption never reached.
    `bounded` is reported to the operator as "more next cycle"; those
    outlines were not deferred, they were deleted. Measured: 6 live
    documents, catalogue blanked, `max_repairs=3` → `adopted: [doc00, doc01,
    doc02]`, `outlines_dropped: [doc03, doc04, doc05]`."""
    vm = _mem()
    for i in range(6):
        name = f"doc{i:02d}.pdf"
        vm.ingest_document(name, [f"chunk for {name}"])
        vm.set_document_outline(name, {"source": "toc", "entries": [{"t": name}]})
    vm.library_file.write_text("[]")        # the catalogue lost its contents

    report = vm.reconcile_indexes(live_episode_ids=None, max_repairs=3)

    assert len(report["catalogue_adopted"]) == 3, report
    assert report["bounded"] is True
    assert report["outlines_dropped"] == [], report
    surviving = [i for i in range(6) if vm.get_document_outline(f"doc{i:02d}.pdf")]
    assert surviving == list(range(6)), (
        f"outlines of live documents the bounded pass never reached were "
        f"deleted, not deferred: kept {surviving}")
    assert any("deferred" in s for s in report["skipped"]), report


def test_an_outline_survives_a_catalogue_write_that_silently_failed():
    """Rows outrank the catalogue. `_update_library_index` swallows its own
    write failures (it logs and returns), so "adoption ran" is not the same
    as "adoption landed" — and the pre-fix outline arm asked only whether
    the name was LISTED. A document with rows keeps its structure."""
    vm = _mem()
    vm.ingest_document("a.pdf", ["alpha chunk"])
    vm.set_document_outline("a.pdf", {"source": "toc", "entries": [{"t": "I"}]})
    vm.library_file.write_text("[]")
    vm._update_library_index = lambda *a, **k: None      # the swallowed failure

    report = vm.reconcile_indexes(live_episode_ids=None)

    assert report["outlines_dropped"] == [], report
    assert vm.get_document_outline("a.pdf"), (
        "the outline of a document whose rows are present was dropped "
        "because a catalogue write silently did nothing")


def test_an_outline_needs_both_readers_to_agree_before_it_is_dropped():
    """Two filters, one deletion. The sweep asks `type == document`, the
    targeted probe asks `source == name`, and a drop needs BOTH to say the
    document has no rows — a filter that quietly stops matching is this
    whole round's subject, and it can land on either of them. Here the
    `source` filter is the blind one and the sweep is right.

    (Without this world, the "rows outrank the catalogue" clause is
    unfalsifiable: the §4GJ round-4 battery reverted it and every other pin
    stayed green, which by §R R2 would make it dead code to delete rather
    than a guard to keep.)"""
    vm = _mem()
    vm.ingest_document("b.pdf", ["beta chunk"])
    vm.set_document_outline("b.pdf", {"source": "toc", "entries": [{"t": "II"}]})
    vm.library_file.write_text("[]")
    vm._update_library_index = lambda *a, **k: None      # adoption cannot land

    real = vm.collection.get

    def _get(**kw):
        if (kw.get("where") or {}) == {"source": "b.pdf"}:
            return {"ids": [], "metadatas": []}
        return real(**kw)

    vm.collection.get = _get
    try:
        report = vm.reconcile_indexes(live_episode_ids=None)
    finally:
        vm.collection.get = real

    assert report["outlines_dropped"] == [], report
    assert vm.get_document_outline("b.pdf"), (
        "one blind filter was enough to delete the structure of a document "
        "the other filter had just listed")


# ── 3. no writer may hand another writer's row to a different reaper ─────────

def test_add_refuses_to_reclassify_an_existing_memory(caplog):
    """Ids are md5(text), so two writers with identical text share ONE row —
    and `add` replaced the metadata dict WHOLE on a duplicate, including
    `type`. A type is not a field: it is which population owns the row and
    therefore which reaper may delete it."""
    vm = _mem()
    text = "the sandbox lost egress after a restart :: re-run tor first"
    vm.add(text, {"type": "fact", "timestamp": "2026-01-01T00:00:00Z"})

    with caplog.at_level("WARNING"):
        vm.add(text, {"type": "episode", "episode_id": 7})

    rows = vm.collection.get(where={"type": "fact"}, include=["metadatas"])
    assert rows["ids"], "the user's fact was relabelled out of its own type"
    assert "episode_id" not in (rows["metadatas"][0] or {}), rows["metadatas"]
    assert vm.collection.get(where={"type": "episode"})["ids"] == []
    assert any("REFUSED" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]

    # CONTROL: a SAME-type refresh still lands. The metadata refresh exists
    # for a reason (2026-07-22: a re-learned lesson's twin kept a stale
    # `source_trajectory_id`, so retraction by that id matched nothing) and
    # "refuse every duplicate" would quietly restore that defect.
    vm.add(text, {"type": "fact", "source_trajectory_id": "traj-9"})
    refreshed = vm.collection.get(where={"type": "fact"}, include=["metadatas"])
    assert refreshed["metadatas"][0].get("source_trajectory_id") == "traj-9"


def test_an_unattended_episode_ingest_cannot_get_a_user_memory_reaped(tmp_path):
    """The whole chain, end to end, with no user present.

    Pre-fix, measured on a real store: a `type=fact` memory whose text
    equalled an episode's `trigger :: lesson` became
    `{"type": "episode", "episode_id": 1}` the moment `record_episode` ran;
    the episode was later purged; the reaper deleted the vector as an
    orphan; `collection.get()` came back EMPTY. `reconcile_vector_index`
    re-drives that ingest at every boot."""
    vm = _mem()
    em = EpisodicMemory(tmp_path)
    trigger = "the sandbox lost egress after a restart"
    lesson = "re-run the tor bootstrap before trusting the netns"
    text = f"{trigger} :: {lesson}"
    vm.add(text, {"type": "fact", "timestamp": "2026-01-01T00:00:00Z"})

    em.record_episode(trigger, lesson=lesson, vector_memory=vm)
    # …and the episode is later gone (evicted, consolidated, or purged),
    # which is when the reaper comes for "its" vector.
    with closing(sqlite3.connect(em.db_path)) as conn:
        conn.execute("DELETE FROM episodes")
        conn.commit()

    vm.reconcile_indexes(live_episode_ids=em.live_episode_ids())

    facts = vm.collection.get(where={"type": "fact"}, include=["documents"])
    assert facts["ids"], "the user's memory was relabelled, then reaped"
    assert facts["documents"] == [text]


def test_the_boot_reconcile_reports_a_shadowed_episode_instead_of_retrying(
        tmp_path, caplog):
    """The other end of the same fix: `add` correctly refuses, but
    `reconcile_vector_index` runs at EVERY boot — without asking first it
    would re-drive the refused write forever and keep reporting a hole that
    can never be repaired. It asks `stored_type` and says so once."""
    vm = _mem()
    em = EpisodicMemory(tmp_path)
    trigger = "a colliding trigger for the boot reconcile"
    lesson = "and the lesson that makes its text identical"
    vm.add(f"{trigger} :: {lesson}", {"type": "fact"})
    em.record_episode(trigger, lesson=lesson)       # no vector twin written

    with caplog.at_level("WARNING"):
        assert em.reconcile_vector_index(vm) == 0, (
            "it re-ingested over a memory of another type")
    assert any("another type" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]
    assert vm.collection.get(where={"type": "fact"})["ids"]

    # CONTROL: a genuinely missing twin is still re-ingested, so "return 0
    # always" is not a way to pass the assertion above.
    em.record_episode("a trigger with no twin at all", lesson="a real hole")
    assert em.reconcile_vector_index(vm) == 1


# ── 4. a rebuilt store proves nothing about the vectors ──────────────────────

def test_live_episode_ids_is_None_for_a_store_that_never_held_an_episode(tmp_path):
    """`__init__` CREATES the schema, so the likeliest way this store goes
    empty — the db deleted, relocated, or pointed at a new memory dir —
    produces an empty TABLE, not an error. The documented `except → None`
    guard cannot fire for it, and `set()` is a licence to delete every
    episode vector."""
    em = EpisodicMemory(tmp_path)

    assert em.live_episode_ids() is None, (
        "a store whose schema was just created reported an authoritative "
        "empty id set")

    # CONTROL: a store that HELD episodes and was emptied is a real, provable
    # `set()` — the AUTOINCREMENT high-water mark survives the DELETE — so
    # "always None" would disarm the reaper instead of making it honest.
    em.record_episode("a trigger", lesson="a lesson")
    with closing(sqlite3.connect(em.db_path)) as conn:
        conn.execute("DELETE FROM episodes")
        conn.commit()
    assert em.live_episode_ids() == set()


def test_a_rebuilt_episode_store_does_not_wipe_every_episode_vector(tmp_path):
    """The consequence, driven through the real pair: the vector twins are
    the only semantic-recall copy of an episode's trigger and lesson.
    Pre-fix, measured: fresh store → `set()` → 3 of 3 deleted."""
    vm = _mem()
    for ep in (1, 2, 3):
        vm.add(f"episode trigger number {ep}", {"type": "episode",
                                                "episode_id": ep})
    em = EpisodicMemory(tmp_path)            # the db was deleted / relocated

    report = vm.reconcile_indexes(live_episode_ids=em.live_episode_ids())

    assert report["episode_vectors_deleted"] == 0, report
    assert len(vm.collection.get(where={"type": "episode"})["ids"]) == 3
    assert any("episode ids unavailable" in s for s in report["skipped"]), report


# ── 6. the repairs were bounded; the scans were not ──────────────────────────

def test_the_reconciler_sweeps_in_bounded_pages():
    """`RECONCILE_MAX_REPAIRS` capped the repairs while both arms issued one
    `collection.get(where=…)` and materialised every matching row's metadata
    in a single list — 7,130 document chunks on the live store — on every
    dream cycle. Paged, peak memory is one page, and coverage is unchanged:
    all five documents are still found across the pages.

    ⚠ Measured on the ANSWERS, not on the call shape (§4GJ round 5). This
    pin used to assert the literal `limit=2` / `offset in (0, 2, 4)`
    arguments, so any legitimate change of paging mechanism — a cursor, a
    different page size per arm, an id-keyed walk — reddened it with no
    change in behaviour, while the property that actually matters is that no
    single read materialises more than a page. That one survives any
    mechanism and still fails for the unpaged sweep."""
    vm = _mem()
    for i in range(5):
        vm.ingest_document(f"d{i}.pdf", [f"chunk number {i}"])
        vm.add(f"episode number {i}", {"type": "episode", "episode_id": i})
    vm.library_file.write_text("[]")
    vm.RECONCILE_SCAN_PAGE = 2

    widest = 0
    real = vm.collection.get

    def _get(**kw):
        nonlocal widest
        res = real(**kw)
        widest = max(widest, len((res or {}).get("ids") or []))
        return res

    vm.collection.get = _get
    try:
        report = vm.reconcile_indexes(live_episode_ids=set(range(5)))
    finally:
        vm.collection.get = real

    assert 0 < widest <= vm.RECONCILE_SCAN_PAGE, (
        f"one read materialised {widest} rows with a page size of "
        f"{vm.RECONCILE_SCAN_PAGE} — the sweep is unpaged")

    # Coverage: paging must not cost the sweep a single row.
    assert sorted(report["catalogue_adopted"]) == [f"d{i}.pdf" for i in range(5)]
    assert report["episode_vectors_deleted"] == 0, report
