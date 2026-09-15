"""§4GK round 6 — what the reaper's "two readers" could not see, and what
the refusal vocabulary never reached.

Round 5 gave the reconcile pass its discipline (an input it cannot read is a
SKIP with a reason; every deletion is proven one at a time) and taught
`add()` to SAY that a write was refused. Round 6 read both back against the
stores they actually run on:

  * the "two readers that must agree" differ only in their filter KEY — they
    are one reader asked twice, and the state that matters (rows gone,
    plain-file sidecars alive) gets a well-formed "no rows" from both, so ONE
    unattended dream cycle deleted the catalogue line AND the irrecoverable
    outline;
  * the `.corrupt` quarantine was write-once: any earlier event permanently
    disarmed it, and the SECOND corruption was flattened with no copy while
    the log said the bytes were preserved;
  * `_read_outlines_for_write` re-read the file inside its own handler and
    caught only `OSError`, so undecodable bytes lost the quarantine AND every
    future outline write, forever;
  * the bus's new `refused: …` status was invisible to its only classifier,
    so a refused canonical write reported SUCCESS;
  * `ADD_REFUSALS` was wired into the REPAIR call sites and not into the
    PRIMARY ones — including the one that DELETES first;
  * after a rebuild of the episode store, re-used ids let an old twin
    masquerade as a live episode's twin forever;
  * the empty-sweep corroboration probed only the first `cap` names at stake.

Every pin below names the world in which it fails: the pre-fix code.
"""
import json
import logging
import tempfile
from pathlib import Path

import pytest

from ghost_agent.memory.episodes import EpisodicMemory
from ghost_agent.memory.skills import SkillMemory, lesson_embedding_text
from ghost_agent.memory.vector import VectorMemory
from ghost_agent.tools.acquired_skills import AcquiredSkillManager


def _mem() -> VectorMemory:
    return VectorMemory(Path(tempfile.mkdtemp()), "http://mock-url")


def _corrupt_copies(vm, name: str):
    return [p.read_bytes() for p in sorted(vm.chroma_dir.glob(f"{name}.corrupt*"))]


# ── 1. the two readers are one reader, and the outline is the half that
#      cannot be rebuilt ───────────────────────────────────────────────────

def test_wholesale_row_loss_does_not_take_the_outline_with_the_catalogue():
    """The state that actually matters: the Chroma rows are gone while the
    plain-file sidecars survive (a partial restore, a reset/recreated
    collection, a re-embed that dropped rows). The `type == document` sweep
    and the `source == name` probe are the same reader asked twice, so both
    answer a well-formed, non-raising "no rows" and the corroboration
    passes. Measured pre-fix on this exact shape: catalogue
    ['postgres_manual.pdf'] + 2 outline entries + 5 rows, rows removed with
    no exception → `catalogue_dropped: ['postgres_manual.pdf']`,
    `outlines_dropped` BOTH names, `document_outlines.json` == `{}` — in one
    unattended dream cycle.

    The catalogue line is recoverable (re-ingest, and the adopt arm re-lists
    it); `derive_document_outline` reads the chunks that are exactly what
    went missing, so the outline is not."""
    vm = _mem()
    vm.ingest_document("postgres_manual.pdf",
                       [f"chapter {i} of the manual" for i in range(5)])
    vm.set_document_outline("postgres_manual.pdf",
                            {"source": "toc", "entries": [[1, "Tutorial", 0]]})
    vm.set_document_outline("second.pdf", {"source": "toc", "entries": []})
    assert vm.get_library() == ["postgres_manual.pdf"]

    # the rows vanish; nothing else does, and nothing raises
    vm.collection.delete(where={"type": "document"})
    assert vm.collection.get(where={"type": "document"})["ids"] == []

    report = vm.reconcile_indexes(live_episode_ids=None)

    # the cheap half is still repaired — that residue line blocks re-ingest
    assert report["catalogue_dropped"] == ["postgres_manual.pdf"], report
    # …and the half that cannot be rebuilt is kept, with a reason
    assert report["outlines_dropped"] == [], report
    assert vm.get_document_outline("postgres_manual.pdf")["entries"] == [
        [1, "Tutorial", 0]]
    assert set(json.loads(vm.outlines_file.read_text())) == {
        "postgres_manual.pdf", "second.pdf"}
    assert any("rows" in s and "outline" in s for s in report["skipped"]), report


def test_a_single_document_losing_its_rows_keeps_its_outline_for_a_cycle():
    """The same asymmetry where the store is otherwise healthy: `live.pdf`
    has rows, `lost.pdf`'s are gone. Pre-fix BOTH of `lost.pdf`'s sidecars
    went in the same pass — the pass believing its own conclusion one step
    later. The recoverable half goes first; the outline is deferred."""
    vm = _mem()
    for name in ("live.pdf", "lost.pdf"):
        vm.ingest_document(name, [f"the only chunk {name} has"])
        vm.set_document_outline(name, {"source": "toc", "entries": [[1, name, 0]]})
    vm.collection.delete(where={"source": "lost.pdf"})

    first = vm.reconcile_indexes(live_episode_ids=None)
    assert first["catalogue_dropped"] == ["lost.pdf"], first
    assert first["outlines_dropped"] == [], first
    assert vm.get_document_outline("lost.pdf")["entries"] == [[1, "lost.pdf", 0]]
    assert any("THIS pass" in s for s in first["skipped"]), first

    # …and the deferral is ONE CYCLE, not forever: the next pass, with the
    # catalogue already repaired, does drop it.
    second = vm.reconcile_indexes(live_episode_ids=None)
    assert second["outlines_dropped"] == ["lost.pdf"], second
    assert vm.get_document_outline("live.pdf")["entries"] == [[1, "live.pdf", 0]]


# ── 8. the corroboration must ask about EVERY name at stake ─────────────────

def test_a_live_document_past_the_cap_still_vetoes_an_empty_sweep():
    """`_at_stake[:cap]` read the first `cap` names, found them absent, and
    believed the empty sweep — while the live document sat past the cap. The
    cap bounds REPAIRS; this loop performs none, it only asks whether the
    sweep lied, and a name past the cap that still has rows is exactly the
    proof that it did."""
    vm = _mem()
    for name in ("ghost1.pdf", "ghost2.pdf"):
        vm._update_library_index(name, "add")
    vm.ingest_document("live.pdf", ["the only chunk this document has"])

    real = vm.collection.get

    def _blind(**kw):
        if (kw.get("where") or {}) == {"type": "document"}:
            return {"ids": [], "metadatas": []}
        return real(**kw)

    vm.collection.get = _blind
    try:
        report = vm.reconcile_indexes(live_episode_ids=None, max_repairs=2)
    finally:
        vm.collection.get = real

    assert report["catalogue_dropped"] == [], report
    assert any("FAILED scan" in s for s in report["skipped"]), report
    assert set(vm.get_library()) == {"ghost1.pdf", "ghost2.pdf", "live.pdf"}


# ── 2. the quarantine was write-once ────────────────────────────────────────

def test_a_second_catalogue_corruption_is_preserved_too():
    """`if not _quarantine.exists(): write` — so the FIRST corruption, from
    any earlier unrelated event, permanently disarmed the guard. Measured: a
    second corruption destroyed three real document names with no copy while
    the log still said "its bytes are preserved at …"."""
    vm = _mem()
    earlier = vm.library_file.with_suffix(vm.library_file.suffix + ".corrupt")
    earlier.write_bytes(b'["something from an earlier, unrelated event"]')

    vm.library_file.write_bytes(
        b'["alpha.pdf", "beta.pdf", "gamma.pdf"')      # truncated write
    lost = vm.library_file.read_bytes()

    vm._update_library_index("delta.pdf", "add")

    copies = _corrupt_copies(vm, "library_index.json")
    assert lost in copies, (
        "three real document names were replaced with no copy of the bytes")
    assert earlier.read_bytes() == b'["something from an earlier, unrelated event"]'
    assert vm.get_library() == ["delta.pdf"]


def test_a_second_outline_corruption_is_preserved_too():
    """The same guard, the same disarm, on the sidecar nothing rebuilds."""
    vm = _mem()
    earlier = vm.outlines_file.with_suffix(vm.outlines_file.suffix + ".corrupt")
    earlier.write_bytes(b'{"from-an-earlier-event": {}}')

    for name in ("a.pdf", "b.pdf", "c.pdf"):
        vm.set_document_outline(name, {"source": "toc", "entries": [[1, name, 0]]})
    raw = vm.outlines_file.read_bytes()
    vm.outlines_file.write_bytes(raw[:len(raw) // 2])
    lost = vm.outlines_file.read_bytes()

    vm.set_document_outline("d.pdf", {"source": "toc", "entries": [[1, "IV", 0]]})

    assert lost in _corrupt_copies(vm, "document_outlines.json")
    assert earlier.read_bytes() == b'{"from-an-earlier-event": {}}'
    assert vm.get_document_outline("d.pdf")["entries"] == [[1, "IV", 0]]


def test_the_same_corrupt_bytes_do_not_mint_a_sidecar_per_read():
    """CONTROL — one sidecar per DISTINCT corrupt content, not one per
    attempt: the outline writer meets the same bad file on every write until
    someone fixes it."""
    vm = _mem()
    vm.outlines_file.write_bytes(b"{not json at all")
    for i in range(3):
        vm.outlines_file.write_bytes(b"{not json at all")
        vm.set_document_outline(f"x{i}.pdf", {"source": "toc", "entries": []})
    assert _corrupt_copies(vm, "document_outlines.json") == [b"{not json at all"]


# ── 3. the handler re-read the file that had just failed ────────────────────

def test_undecodable_outline_bytes_are_preserved_and_the_write_still_lands():
    """`quarantine.write_text(self.outlines_file.read_text())` — a SECOND
    read of the file that just failed. For undecodable bytes the first read
    raises UnicodeDecodeError, the second raises it again, and `except
    OSError` does not catch a ValueError. Measured pre-fix: no quarantine, no
    write, and EVERY future outline write failing identically forever at
    `logger.error` only — including `derive_document_outline`, the recovery
    the log line points at."""
    vm = _mem()
    vm.set_document_outline("a.pdf", {"source": "toc", "entries": [[1, "I", 0]]})
    bad = b"\xff\xfe\x00not utf-8 at all \xc3\x28"
    vm.outlines_file.write_bytes(bad)

    vm.set_document_outline("d.pdf", {"source": "toc", "entries": [[1, "IV", 0]]})

    assert vm.get_document_outline("d.pdf")["entries"] == [[1, "IV", 0]], (
        "the write did not land — the sidecar is stuck undecodable forever")
    assert bad in _corrupt_copies(vm, "document_outlines.json")
    # …and the store is out of the hole: the next write works too.
    vm.set_document_outline("e.pdf", {"source": "toc", "entries": [[1, "V", 0]]})
    assert set(json.loads(vm.outlines_file.read_text())) == {"d.pdf", "e.pdf"}


def test_undecodable_catalogue_bytes_are_preserved_and_the_write_still_lands():
    """The sibling. `read_text()` raised OUTSIDE the inner try, so the bytes
    never reached the quarantine at all — they fell to the outer handler as a
    plain "Library index error" and every future ingest failed the same way
    forever."""
    vm = _mem()
    bad = b"\xff\xfe\x00library index, unreadable \xc3\x28"
    vm.library_file.write_bytes(bad)

    assert vm._update_library_index("delta.pdf", "add") is True
    assert vm.get_library() == ["delta.pdf"]
    assert bad in _corrupt_copies(vm, "library_index.json")


# ── 4. the bus says "refused"; its only classifier heard nothing ────────────

@pytest.mark.asyncio
async def test_a_refused_vector_leg_is_a_canonical_failure():
    """Producer and consumer read TOGETHER, through the real bus. `add()`
    refuses a write that would reclassify an existing row's type;
    `MemoryBus._vector` reports `refused: …`; both classifiers in
    tools/memory keyed on `startswith("error")`. Measured pre-fix:
    `_bus_write_failures` → `[]` and `_bus_canonical_failed` → False, and for
    `insert_fact` the vector leg IS the canonical store, so a fact that was
    never stored reported SUCCESS."""
    from ghost_agent.core.bus import MemoryBus
    from ghost_agent.tools.memory import (_bus_canonical_failed,
                                          _bus_write_failures)

    vm = _mem()
    text = "the worker node keeps the long jobs off the laptop"
    vm.add(text, {"type": "auto", "timestamp": "2026-01-01T00:00:00Z"})
    bus = MemoryBus(vector_memory=vm)

    report = await bus.publish_fact("insert_fact", {
        "text": text,
        "metadata": {"timestamp": "2026-09-14T00:00:00Z", "type": "manual"},
        "triplets": [],
    })

    assert _bus_write_failures(report), report
    assert _bus_canonical_failed(report, "insert_fact") is True, report
    # CONTROL: the vector leg is a secondary index for a profile write, and
    # a secondary failure is still a PARTIAL, not a FAILED.
    assert _bus_canonical_failed(report, "update_profile") is False, report
    # CONTROL: an ordinary write is not a failure.
    ok = await bus.publish_fact("insert_fact", {
        "text": "an entirely new fact about the tailnet",
        "metadata": {"timestamp": "2026-09-14T00:00:00Z", "type": "manual"},
        "triplets": [],
    })
    assert _bus_write_failures(ok) == [], ok
    assert _bus_canonical_failed(ok, "insert_fact") is False, ok


def test_a_stubbed_store_is_not_an_owner():
    """`smart_update` now DECIDES WHETHER TO DELETE on `stored_type`, so an
    invented answer destroys a fact. Every truthiness test in that probe is
    satisfied by a `MagicMock` collection, and `str(meta["type"])` then
    answers "<MagicMock name=…>" — measured: a plain `smart_update` against a
    stubbed store refused its own write and kept the row it was supposed to
    replace (3 existing pins went red). The store's own rule, already written
    into `ADD_REFUSALS`: a stub's answer is not the store's answer."""
    from unittest.mock import MagicMock

    vm = _mem()
    vm.collection = MagicMock()
    vm.collection.query.return_value = {
        "ids": [["existing_abc"]],
        "distances": [[0.45]],
        "documents": [["paraphrase one"]],
        "metadatas": [[{"timestamp": "old"}]],
    }
    # …driven exactly as the store's existing pins drive it: `add` stubbed,
    # so this asserts the DECISION, not the embedding.
    vm.add = MagicMock(return_value=VectorMemory.ADD_STORED)

    assert vm.stored_type("paraphrase two") is None

    vm.smart_update("paraphrase two")
    vm.collection.delete.assert_called_with(ids=["existing_abc"])
    vm.add.assert_called()


# ── 5. ADD_REFUSALS reached the repair sites, not the primary ones ──────────

def test_a_refused_skill_description_keeps_the_embedding_it_could_not_replace(tmp_path):
    """`collection.delete(where={name, acquired_skill})` ran FIRST, then
    `add(description)` was refused — the same delete-then-refuse shape as
    `smart_update`, on a path that then logs "SKILL ACQUIRED — Permanently
    learned new tool" and returns True. Measured pre-fix: zero
    `acquired_skill` rows for the name afterwards, i.e. the skill vanished
    from semantic routing while the registry said it was learned."""
    vm = _mem()
    mgr = AcquiredSkillManager(tmp_path, memory_system=vm)
    assert mgr.save_skill("tail_net_ping", "Ping a host on the tailnet by name",
                          {"type": "object", "properties": {}},
                          "def run():\n    return 1\n")
    assert vm.collection.get(
        where={"$and": [{"name": "tail_net_ping"},
                        {"type": "acquired_skill"}]})["ids"]

    # a dream consolidation stored the NEW description's exact sentence as an
    # ordinary memory — so the replacement embedding cannot land
    new_desc = "Ping any host on the tailnet and report the round trip"
    vm.add(new_desc, {"type": "auto", "timestamp": "2026-01-01T00:00:00Z"})

    assert mgr.save_skill("tail_net_ping", new_desc,
                          {"type": "object", "properties": {}},
                          "def run():\n    return 2\n")

    rows = vm.collection.get(
        where={"$and": [{"name": "tail_net_ping"},
                        {"type": "acquired_skill"}]},
        include=["documents"])
    assert rows["ids"], (
        "the skill is gone from semantic routing — the old embedding was "
        "deleted for a replacement that could not land")
    assert rows["documents"] == ["Ping a host on the tailnet by name"]
    # the registry and the file are the canonical stores and still updated
    assert mgr.get_all_skills()["tail_net_ping"]["description"] == new_desc
    # …and the colliding row keeps its owner
    assert vm.collection.get(where={"type": "auto"})["ids"]


def test_a_refused_lesson_twin_is_said_out_loud(tmp_path, caplog):
    """`add_lesson`'s own twin write ignored the answer. The playbook row
    lands (it is the canonical store) while the vector copy the playbook's
    semantic path reads never exists — the lesson is reachable only by the
    substring fallback, and "SKILL ACQUIRED" said nothing about it."""
    vm = _mem()
    sm = SkillMemory(tmp_path)
    trigger = "the sandbox lost egress after a restart"
    anti = "assuming the container kept its netns"
    correct = "re-run the transparent tor setup before any fetch"
    text = lesson_embedding_text({"trigger": trigger, "anti_pattern": anti,
                                  "correct_pattern": correct})
    vm.add(text, {"type": "auto", "timestamp": "2026-01-01T00:00:00Z"})

    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        out = sm.learn_lesson("", "", "", memory_system=vm, trigger=trigger,
                              anti_pattern=anti, correct_pattern=correct,
                              origin="test")

    assert out == "written"
    assert any(l.get("trigger") == trigger
               for l in json.loads(sm.file_path.read_text()))
    assert vm.collection.get(where={"type": "skill"})["ids"] == [], (
        "the twin landed after all — then this pin is testing the wrong world")
    assert any("dark to semantic recall" in r.getMessage()
               for r in caplog.records if r.levelno >= logging.WARNING), (
        "the lesson is invisible to semantic recall and nobody was told")


# ── 6. an episode store REBUILD re-issues ids the old twins still carry ─────

def test_a_twin_from_a_previous_generation_stops_masquerading(tmp_path):
    """Episode ids are `INTEGER PRIMARY KEY AUTOINCREMENT`: a rebuilt store
    re-issues #1. The vector reaper decides by id alone, so old #1's twin is
    never deferred (its id IS live) and never reaped — it masquerades as new
    #1's twin forever, and a semantic hit on the old text resolves to the
    wrong episode. This arm had no way to notice; the id↔text pairing is the
    proof, and it needs no second store."""
    vm = _mem()
    em = EpisodicMemory(tmp_path)
    old = em.record_episode("the tor circuit died mid-search",
                            lesson="race one circuit per engine",
                            vector_memory=vm)
    Path(em.db_path).unlink()                       # the store is rebuilt
    em2 = EpisodicMemory(tmp_path)
    new = em2.record_episode("the sandbox lost egress after a restart",
                             lesson="re-run tor before any fetch",
                             vector_memory=vm)
    assert old == new == 1
    assert len(vm.collection.get(where={"type": "episode"})["ids"]) == 2

    em2.reconcile_vector_index(vm)

    rows = vm.collection.get(where={"type": "episode"},
                             include=["metadatas", "documents"])
    assert len(rows["ids"]) == 1, (
        "two rows still claim episode #1 — one of them can only answer with "
        "the wrong episode")
    assert rows["documents"][0] == em2._episode_document(
        "the sandbox lost egress after a restart", "re-run tor before any fetch")
    assert int(rows["metadatas"][0]["episode_id"]) == new


def test_a_live_twin_wearing_stale_metadata_is_not_deleted(tmp_path):
    """CONTROL — the dangerous half. A row whose text IS a live episode's
    document is that episode's twin with a stale label (the store dedups on
    text, so the label follows the last writer). It is reported and LEFT;
    re-ingesting its real owner relabels it."""
    vm = _mem()
    em = EpisodicMemory(tmp_path)
    shared = "hello ghost what is new on the worker node today"
    ep1 = em.record_episode(shared, lesson="", vector_memory=vm)
    ep2 = em.record_episode(shared, lesson="", vector_memory=vm)
    assert ep1 != ep2
    before = vm.collection.get(where={"type": "episode"})["ids"]
    assert len(before) == 1                 # dedup'd onto one row

    em.reconcile_vector_index(vm)

    assert vm.collection.get(where={"type": "episode"})["ids"] == before


def test_a_truncated_episode_read_deletes_nothing(tmp_path):
    """CONTROL — "no live episode has this text" is only proof when the
    episode read was not cut short by its LIMIT. A truncated sweep reports
    and deletes nothing, the standing rule for this whole pass."""
    vm = _mem()
    em = EpisodicMemory(tmp_path)
    em.record_episode("the tor circuit died mid-search", lesson="race engines",
                      vector_memory=vm)
    Path(em.db_path).unlink()
    em2 = EpisodicMemory(tmp_path)
    em2.record_episode("the sandbox lost egress after a restart",
                       lesson="re-run tor first", vector_memory=vm)

    em2.reconcile_vector_index(vm, limit=1)

    assert len(vm.collection.get(where={"type": "episode"})["ids"]) == 2
