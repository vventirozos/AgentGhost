"""§4GJ round 5 — what the reaper still could not prove, and what it did anyway.

Round 4 gave this pass its discipline: an input it cannot read is a SKIP
with a reason, and every proposed deletion is justified one at a time by a
targeted probe. Round 5 read the code back against that discipline and found
it applied everywhere except where it deletes.

  * the episode arm — the ONLY arm that deletes rows — took the live id set
    on trust, and the set is a SNAPSHOT read in a different `to_thread` hop:
    an episode recorded in the gap was reaped as an orphan;
  * `set_document_outline` flattened a corrupt outline sidecar with no
    quarantine — character for character the defect fixed for the catalogue
    one screen above it;
  * the "is this empty sweep real?" corroboration compared a
    document-scoped sweep against a count of EVERY row, so it read a correct
    empty sweep as a failed one on any store that holds a single memory;
  * `add()` answered `None` for a refusal and `None` for a success;
  * the boot reconcile compared a stripped index against an unstripped
    document and re-ingested the same episode forever;
  * a zero-byte catalogue file disarmed the reconciler entirely;
  * `catalogue_dropped` was reported without checking the write landed.

Every pin below names the world in which it fails: the pre-fix code.
"""
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory.episodes import EpisodicMemory
from ghost_agent.memory.vector import MemoryWriteRefused, VectorMemory


def _mem() -> VectorMemory:
    return VectorMemory(Path(tempfile.mkdtemp()), "http://mock-url")


def _dream_self(vm, em):
    """The slice of DreamEngine `_reconcile_memory_stores` reads. `vm`/`em`
    are REAL components — `_is_real_component` gates on the module name, so
    a local double would silently no-op the whole method."""
    return SimpleNamespace(memory=vm, context=SimpleNamespace(episodic_memory=em))


def _twin_ids(vm) -> set:
    rows = vm.collection.get(where={"type": "episode"}, include=["metadatas"])
    return {(m or {}).get("episode_id") for m in (rows.get("metadatas") or [])}


# ── 1. the id set is a snapshot, and this arm deletes ────────────────────────

@pytest.mark.asyncio
async def test_an_episode_recorded_after_the_id_snapshot_keeps_its_twin(tmp_path):
    """The race, driven through the REAL consumer.

    `dream._reconcile_memory_stores` reads `live_episode_ids()` in one
    `to_thread` hop and calls the reconciler in the NEXT one. An episode
    recorded in that gap is live, has a twin, and is not in the set — and
    the episode arm, alone among the arms, deleted without a per-victim
    proof. Measured pre-fix with this exact call shape: `deleted: 1`, the
    episode still in SQLite with no twin, and the twin is the only
    semantic-recall copy of its trigger and lesson. It self-heals at the
    next boot, which then logs the "genuinely missing a twin" alarm the
    whole design exists to silence.

    The same pass must still do its job: episode #1 here was evicted long
    ago, its id is BELOW the high-water mark, and its twin IS reaped."""
    from ghost_agent.core.dream import Dreamer

    vm = _mem()
    em = EpisodicMemory(tmp_path)
    evicted = em.record_episode("an episode later evicted",
                                lesson="a lesson that went with it",
                                vector_memory=vm)
    kept_a = em.record_episode("an episode that is still here",
                               lesson="keep me", vector_memory=vm)
    kept_b = em.record_episode("another episode that is still here",
                               lesson="keep me too", vector_memory=vm)
    with closing(sqlite3.connect(em.db_path)) as conn:
        conn.execute("DELETE FROM episodes WHERE id = ?", (evicted,))
        conn.commit()

    recorded = []
    real_getter = em.live_episode_ids

    def _getter_then_a_new_episode():
        ids = real_getter()
        # THE GAP.
        recorded.append(em.record_episode(
            "an episode recorded while the reconcile was starting",
            lesson="written in the gap between the two to_thread hops",
            vector_memory=vm))
        return ids

    em.live_episode_ids = _getter_then_a_new_episode

    await Dreamer._reconcile_memory_stores(_dream_self(vm, em))

    assert _twin_ids(vm) == {kept_a, kept_b, recorded[0]}, (
        "the twin of an episode recorded after the id set was read was "
        "reaped as an orphan")
    # …and the episode is still in SQLite, which is what makes the loss
    # invisible until the next boot's false alarm.
    with closing(sqlite3.connect(em.db_path)) as conn:
        live = {r[0] for r in conn.execute("SELECT id FROM episodes")}
    assert recorded[0] in live


def test_a_deferred_twin_is_reported_not_silently_kept():
    """"Leave it alone" is half the rule; "and say so" is the other half —
    an arm that quietly does nothing is indistinguishable from one that is
    not running (`silent-inoperative-subsystems`)."""
    vm = _mem()
    vm.add("a twin recorded after the snapshot", {"type": "episode",
                                                  "episode_id": 9})

    report = vm.reconcile_indexes(live_episode_ids={1, 2})

    assert report["episode_vectors_deleted"] == 0
    assert any("high-water mark" in s for s in report["skipped"]), report
    assert _twin_ids(vm) == {9}


def test_the_getter_form_is_accepted_and_still_defers_an_unprovable_orphan(tmp_path):
    """§4GK round 6 corrected round 5's version of this.

    Round 5 accepted a CALLABLE and treated its answer as absolutely
    authoritative — no snapshot rule at all — reasoning that reading inside
    this lock leaves no gap, because `record_episode` commits its SQLite row
    before blocking here to write the twin. That reasoning is one ordering
    assumption about another module away from being wrong, and it IS wrong the
    moment a caller reads the ids slightly before handing them over, which is
    what the consumer did.

    Deferring an above-watermark id costs one cycle: it is reaped on the next
    pass if it really is an orphan. Reaping a live episode's twin destroys the
    only semantic-recall copy of its trigger and lesson. On a destructive path
    the cheap guard stays on, whatever form the input took.

    Fails in any tree where the callable form skips the watermark guard."""
    vm = _mem()
    em = EpisodicMemory(tmp_path)
    em.record_episode("an episode that is still here",
                      lesson="keep me", vector_memory=vm)
    orphan = em.record_episode("the newest episode, since deleted",
                               lesson="its twin is now an orphan",
                               vector_memory=vm)
    with closing(sqlite3.connect(em.db_path)) as conn:
        conn.execute("DELETE FROM episodes WHERE id = ?", (orphan,))
        conn.commit()

    # The getter form is ACCEPTED (pre-round-5 this raised
    # `TypeError: 'function' object is not iterable` into the outer handler
    # and reported `aborted:`) …
    report = vm.reconcile_indexes(live_episode_ids=em.live_episode_ids)
    assert not any("aborted" in s for s in report["skipped"]), report
    # … and it still refuses to reap what it cannot prove is an orphan.
    assert report["episode_vectors_deleted"] == 0
    assert any("high-water mark" in s for s in report["skipped"]), report


def test_an_orphan_below_the_watermark_is_still_reaped(tmp_path):
    """Control: the guard defers only the UNPROVABLE case. An orphan whose id
    is below everything the live set names existed when the set was read, so
    it is provably an orphan and the pass still does its work."""
    vm = _mem()
    em = EpisodicMemory(tmp_path)
    old = em.record_episode("an episode later evicted",
                            lesson="gone", vector_memory=vm)
    em.record_episode("a newer episode still here",
                      lesson="keep me", vector_memory=vm)
    with closing(sqlite3.connect(em.db_path)) as conn:
        conn.execute("DELETE FROM episodes WHERE id = ?", (old,))
        conn.commit()
    report = vm.reconcile_indexes(live_episode_ids=em.live_episode_ids)
    assert report["episode_vectors_deleted"] == 1, report

def test_a_corrupt_outline_sidecar_is_preserved_before_it_is_replaced():
    """`_read_outlines` answers `{}` for an unparseable
    `document_outlines.json` and `set_document_outline` then wrote a
    SINGLE-entry dict over it. Measured pre-fix: three documents' outlines
    replaced by one, no quarantine, `get_document_outline` answering `{}`
    for the other two. Nothing rebuilds this file — the live one is 145 KB
    of 4138 entries for the PostgreSQL manual, and deriving it again costs a
    full breadcrumb sweep of ~7000 chunks.

    NOTE (§4GK round 6): these bytes are valid UTF-8 that merely fails to
    parse as JSON — only HALF the corruption class. UNDECODABLE bytes took a
    different route through the same handler (which re-read the file inside
    itself and caught `OSError` only) and lost the quarantine AND every
    future outline write. That half is pinned in
    `test_memory_reconcile_round6_4gk.py`; do not read this pin as covering
    it."""
    vm = _mem()
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        vm.set_document_outline(name, {"source": "toc",
                                       "entries": [[1, name, 0]]})
    raw = vm.outlines_file.read_text()
    vm.outlines_file.write_text(raw[:len(raw) // 2])        # a truncated write
    corrupt = vm.outlines_file.read_bytes()

    vm.set_document_outline("d.pdf", {"source": "toc", "entries": [[1, "IV", 0]]})

    quarantine = vm.outlines_file.with_suffix(vm.outlines_file.suffix + ".corrupt")
    assert quarantine.exists(), (
        "three documents' outlines were flattened by one write with no copy "
        "of the bytes kept")
    assert quarantine.read_bytes() == corrupt
    # …and the write still LANDS: refusing it would block every future
    # ingest's outline with no recovery the agent can perform by itself,
    # which is the other half of the lesson the catalogue already learned.
    assert vm.get_document_outline("d.pdf")["entries"] == [[1, "IV", 0]]
    assert json.loads(vm.outlines_file.read_text())


# ── 3. corroborate the QUESTION, not a different filter ─────────────────────

def test_a_residue_catalogue_entry_is_droppable_on_a_store_that_holds_memories():
    """The reviewer's repro, exactly: one auto memory, one episode twin,
    zero documents and one catalogue line whose document is gone. The
    corroboration compared the document-scoped sweep against
    `collection.count()` over ALL rows; they disagreed for the most innocent
    reason there is, so "zero live documents" became unprovable and BOTH
    arms skipped forever — while `ingest_document`'s dedup refuses to
    re-ingest the name that residue line is holding, permanently."""
    vm = _mem()
    vm.add("the user prefers the worker node for long jobs", {"type": "auto"})
    vm.add("an episode twin about the sandbox", {"type": "episode",
                                                 "episode_id": 1})
    vm._update_library_index("gone.pdf", "add")

    report = vm.reconcile_indexes(live_episode_ids={1})

    assert report["catalogue_dropped"] == ["gone.pdf"], report
    assert not any("FAILED scan" in s for s in report["skipped"]), report
    assert vm.get_library() == []
    assert vm.collection.get(where={"type": "auto"})["ids"]


def test_a_lying_sweep_is_still_caught_on_a_store_that_holds_memories():
    """CONTROL — the replacement corroboration must still REFUSE an
    uncorroborated empty sweep, on the same production-shaped store where
    the old one was disabled. Here the `type == document` filter is the
    blind one and `keep.pdf` has rows."""
    vm = _mem()
    vm.add("an ordinary auto memory", {"type": "auto"})
    vm.ingest_document("keep.pdf", ["the only chunk this document has"])
    vm.set_document_outline("keep.pdf", {"source": "toc", "entries": [[1, "I", 0]]})

    real = vm.collection.get

    def _get(**kw):
        if (kw.get("where") or {}) == {"type": "document"}:
            return {"ids": [], "metadatas": []}
        return real(**kw)

    vm.collection.get = _get
    try:
        report = vm.reconcile_indexes(live_episode_ids=None)
    finally:
        vm.collection.get = real

    assert report["catalogue_dropped"] == [] and report["outlines_dropped"] == []
    assert vm.get_library() == ["keep.pdf"]
    assert vm.get_document_outline("keep.pdf"), "the outline was reaped"
    assert any("FAILED scan" in s for s in report["skipped"]), report


# ── 4. a refusal is not a success ────────────────────────────────────────────

def test_add_answers_which_of_the_four_things_happened():
    """Pre-fix every exit was a bare `return`: `None` for "stored", `None`
    for "refreshed", `None` for "too short to embed" and `None` for
    "refused — this text belongs to another population"."""
    vm = _mem()
    text = "the sandbox lost egress after a restart :: re-run tor first"

    stored = vm.add(text, {"type": "auto", "timestamp": "2026-01-01T00:00:00Z"})
    refreshed = vm.add(text, {"type": "auto", "source_trajectory_id": "T1"})
    refused = vm.add(text, {"type": "identity"})
    too_short = vm.add("tiny")

    assert stored == VectorMemory.ADD_STORED
    assert refreshed == VectorMemory.ADD_REFRESHED
    assert refused == VectorMemory.ADD_REFUSED_TYPE
    assert too_short == VectorMemory.ADD_TOO_SHORT
    assert len({stored, refreshed, refused, too_short}) == 4, (
        "two different outcomes answer the same thing — no caller can act "
        "on that")
    assert stored in VectorMemory.ADD_LANDED
    assert refreshed in VectorMemory.ADD_LANDED
    assert refused not in VectorMemory.ADD_LANDED
    assert too_short not in VectorMemory.ADD_LANDED
    # the refusal itself is unchanged: the row keeps its owner
    assert vm.collection.get(where={"type": "identity"})["ids"] == []
    assert vm.collection.get(where={"type": "auto"})["ids"]


def test_smart_update_raises_and_keeps_the_fact_it_cannot_replace():
    """The store knows; the caller could not. `smart_update` is the identity
    path, and its two callers already report a partial index failure when
    this raises — nothing else told them.

    ⚠ THIS PIN WAS BUILT ON THE WRONG STORE (rewritten §4GK round 6). It held
    NO pre-existing identity row, so the near-neighbour branch never ran and
    the raise was all there was to see — while the real caller,
    `update_profile`, ALWAYS has a predecessor (replacing it is the entire
    reason `smart_update` exists). On that store round 5's raise landed AFTER
    `collection.delete(existing_id)`: the refusal turned a reclassification
    bug into outright destruction of the fact it exists to protect, and the
    identity tier came back EMPTY with the user told only that "retrieval may
    not reflect the change". Ask whether the replacement can land BEFORE
    destroying what it replaces."""
    vm = _mem()
    predecessor = "User car is a Fiat"
    vm.smart_update(predecessor, "identity")          # the fact on file
    text = "User car is a BMW"
    # …and an ordinary dream consolidation of the same sentence already owns
    # the new text, so the replacement cannot land.
    vm.add(text, {"type": "auto", "timestamp": "2026-01-01T00:00:00Z"})

    with pytest.raises(MemoryWriteRefused):
        vm.smart_update(text, "identity")

    surviving = vm.collection.get(where={"type": "identity"},
                                  include=["documents"])
    assert surviving["documents"] == [predecessor], (
        "the identity fact was deleted for a replacement that could not "
        "land — the refusal destroyed what it was protecting")
    assert vm.collection.get(where={"type": "auto"})["ids"]

    # CONTROL: the ordinary path does not raise, does store, and DOES still
    # replace its near neighbour — the guard must not disarm the refinement.
    vm.smart_update("User car is a Citroen", "identity")
    after = vm.collection.get(where={"type": "identity"},
                              include=["documents"])
    assert after["documents"] == ["User car is a Citroen"], after


@pytest.mark.asyncio
async def test_a_refused_identity_write_reaches_the_user_as_a_partial():
    """The whole chain, as the user sees it. Pre-fix, measured: the identity
    fact collided with an existing `auto` row, the write was refused,
    `update_profile` answered "SUCCESS: Profile updated", the fact never
    reached the `identity` tier `inject_identity` queries — and the row that
    survived is `type=auto`, which IS in `_PRUNABLE_TYPES`, so a user
    identity fact was left inside the eviction-eligible population with
    nobody told.

    ⚠ And the store carries the PREDECESSOR the real caller always has
    (§4GK round 6): `update_profile` replaces a fact on file, so the store
    shape without one hid the delete-then-refuse loss entirely. What the user
    keeps is asserted here, not just what they are told."""
    from ghost_agent.tools.memory import tool_update_profile

    vm = _mem()
    vm.smart_update("User car is a Fiat", "identity")
    vm.add("User car is a BMW", {"type": "auto",
                                 "timestamp": "2026-01-01T00:00:00Z"})
    profile = MagicMock()
    profile.update = MagicMock(return_value="JSON updated")

    out = str(await tool_update_profile(
        category="identity", key="car", value="a BMW",
        profile_memory=profile, memory_system=vm, graph_memory=None,
        memory_bus=None,
    ))

    assert "PARTIAL" in out and "vector" in out, out
    # the NEW value never reached the identity tier (that is the refusal)…
    identity = vm.collection.get(where={"type": "identity"},
                                 include=["documents"])
    assert "User car is a BMW" not in identity["documents"], (
        "the fact is in the identity tier after all — then this pin is "
        "testing the wrong world")
    # …and the value that WAS on file is still there: a write that cannot
    # land must not take the predecessor with it.
    assert identity["documents"] == ["User car is a Fiat"], identity
    assert vm.collection.get(where={"type": "auto"})["ids"]


# ── 5. one normaliser on both sides of the dedup question ───────────────────

def test_two_episodes_sharing_a_trigger_stop_ping_ponging_the_shared_row(tmp_path):
    """`indexed_docs` was built with `.strip()` while the episode's own
    document was tested UNSTRIPPED, and `_episode_document` strips only
    `" :"` — so an episode whose lesson ends in "\\n" (ordinary model
    output) never matched the row it had already been dedup'd onto.
    Measured pre-fix: every boot re-ingested one of the two, logged the
    false "genuinely missing a twin" alarm, and flipped the shared row's
    `episode_id` to the other one — forever, alternating. Combined with the
    reaper's snapshot arm, whichever episode did not currently own the row
    lost its twin outright."""
    vm = _mem()
    em = EpisodicMemory(tmp_path)
    trigger = "the sandbox lost egress after a restart"
    lesson = "re-run the tor bootstrap before trusting the netns\n"
    em.record_episode(trigger, lesson=lesson, vector_memory=vm)
    em.record_episode(trigger, lesson=lesson, vector_memory=vm)

    rows = vm.collection.get(where={"type": "episode"}, include=["metadatas"])
    assert len(rows["ids"]) == 1, "the store did not dedup — wrong world"
    owner_before = (rows["metadatas"][0] or {}).get("episode_id")

    assert em.reconcile_vector_index(vm) == 0, (
        "an episode already reachable through the shared entry was reported "
        "as a genuine hole and re-ingested")
    assert em.reconcile_vector_index(vm) == 0, "…and again at the next boot"

    after = vm.collection.get(where={"type": "episode"}, include=["metadatas"])
    assert len(after["ids"]) == 1
    assert (after["metadatas"][0] or {}).get("episode_id") == owner_before, (
        "the shared row's episode_id flipped between the two episodes")


def test_a_genuinely_missing_twin_is_still_re_ingested(tmp_path):
    """CONTROL: "call everything dedup-covered" would pass the pin above and
    disarm the repair this method exists for."""
    vm = _mem()
    em = EpisodicMemory(tmp_path)
    em.record_episode("an episode with no twin at all",
                      lesson="a real hole in the index")

    assert em.reconcile_vector_index(vm) == 1
    assert len(vm.collection.get(where={"type": "episode"})["ids"]) == 1


# ── 6. both readers must agree on a truncated catalogue ─────────────────────

def test_a_zero_byte_catalogue_does_not_disarm_the_reconciler():
    """`_update_library_index` has always read a zero-byte
    `library_index.json` as "[]" while `_load_library` let `json` raise on
    it, so a truncated write disarmed the reconciler ("catalogue
    unreadable") until some UNRELATED ingest happened to rewrite the file —
    the guard that never runs, on the store that needs it. An empty
    catalogue proposes no deletion; the adopt arm rebuilds it from the rows
    that are actually there, which is the repair this state needs."""
    vm = _mem()
    vm.ingest_document("real.pdf", ["alpha chunk"])
    vm.set_document_outline("real.pdf", {"source": "toc", "entries": [[1, "I", 0]]})
    vm.library_file.write_text("")

    report = vm.reconcile_indexes(live_episode_ids=None)

    assert report["catalogue_adopted"] == ["real.pdf"], report
    assert not any("unreadable" in s for s in report["skipped"]), report
    assert vm.get_library() == ["real.pdf"]
    assert vm.get_document_outline("real.pdf"), "the outline went with it"


# ── 8. "the drop ran" is not "the drop landed" ──────────────────────────────

def test_a_catalogue_drop_that_did_not_land_is_not_reported_as_dropped():
    """`_update_library_index` swallows its own write failures, and the drop
    arm appended the name unconditionally — so an operator reading
    `catalogue_dropped` was told the residue was gone while the file still
    listed it, every cycle, one repair off the cap each time."""
    vm = _mem()
    vm.add("an ordinary auto memory", {"type": "auto"})
    vm._update_library_index("ghost.pdf", "add")
    vm._update_library_index = lambda *a, **k: None      # the swallowed failure

    report = vm.reconcile_indexes(live_episode_ids=None)

    assert report["catalogue_dropped"] == [], report
    assert any("did not land" in s for s in report["skipped"]), report
    assert vm.get_library() == ["ghost.pdf"], "the world the report described"


def test_a_non_string_catalogue_entry_is_actually_dropped():
    """The state that produced the false report forever: `data.remove` was
    asked for the string "123" while the list held the number `123`, matched
    nothing, and changed no bytes. Reported dropped every cycle, still
    blocking re-ingest of that name every cycle."""
    vm = _mem()
    vm.add("an ordinary auto memory", {"type": "auto"})
    vm.library_file.write_text(json.dumps([123, "gone.pdf"]))

    report = vm.reconcile_indexes(live_episode_ids=None)

    assert sorted(report["catalogue_dropped"]) == ["123", "gone.pdf"], report
    assert vm.get_library() == [], (
        f"reported dropped, still listed: {vm.get_library()}")
