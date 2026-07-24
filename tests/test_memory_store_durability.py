"""Durability of the small JSON-backed memory stores (2026-07-22).

Five stores keep their state in a single JSON file that is read on
startup / on every write and rewritten with an atomic rename. Three of
them — ``CompetenceProfile``, ``ContradictionLog``, ``AdaptiveThreshold``
— collapsed "the file is ABSENT" and "the file is PRESENT but I could not
read it" into the same outcome (start empty), so the very next write
atomically overwrote the intact file: a total, silent wipe, logged at
debug level at best. ``MemoryJournal`` and ``ProfileMemory`` had already
been hardened against exactly this (propagate on a transient OSError,
sidecar the bytes on corruption); these three had not.

This module also covers:

* the journal's crash-safe TAKE/ACK lifecycle — ``pop_all`` stages the
  batch to an in-flight file so a deploy (``kill``) landing mid-drain no
  longer destroys up to 50 buffered facts;
* the widened ``is_upstream_transient`` classifier (``core.llm`` re-raises
  some upstream failures as a plain ``RuntimeError`` / ``Exception``);
* profile singleton replacement (a correction must not accumulate) and
  the bounded ``notes.info`` sink;
* journal capacity overflow being visible, and re-queued items surviving
  an overflow push_front.
"""

import json
import logging
import os

import httpx
import pytest

from ghost_agent.memory.adaptive_threshold import AdaptiveThreshold
from ghost_agent.memory.competence import CompetenceProfile
from ghost_agent.memory.contradiction_log import ContradictionLog
from ghost_agent.memory.journal import MemoryJournal, is_upstream_transient
from ghost_agent.memory.profile import ProfileMemory


# chmod-based unreadability does not apply to root.
root_skip = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses file permissions",
)


class _Unreadable:
    """Make a file present-but-unreadable for the duration of the block."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self._mode = self.path.stat().st_mode
        os.chmod(self.path, 0o000)
        return self.path

    def __exit__(self, *exc):
        os.chmod(self.path, 0o600)
        return False


# =========================================================================
# 1. HIGH — an unreadable store must never be overwritten
# =========================================================================


@root_skip
def test_competence_refuses_to_overwrite_unreadable_profile(tmp_path):
    cp = CompetenceProfile(tmp_path)
    for _ in range(50):
        cp.record("shell", "ls", success=True)
    path = tmp_path / "competence_profile.json"
    before = path.read_text()

    with _Unreadable(path):
        cp2 = CompetenceProfile(tmp_path)      # load fails → degraded
        cp2.record("sql", "select", success=False)   # must NOT write

    # The 50-observation history is still on disk, untouched.
    assert path.read_text() == before
    assert CompetenceProfile(tmp_path).observations("shell", "ls") == 50
    assert list(tmp_path.glob("*.corrupt-*")) == []


@root_skip
def test_contradiction_log_refuses_to_overwrite_unreadable_file(tmp_path):
    log = ContradictionLog(tmp_path)
    for i in range(5):
        log.record(new_fact=f"fact {i}", old_facts=[], deleted_ids=[])
    path = tmp_path / "contradiction_log.json"
    before = path.read_text()

    with _Unreadable(path):
        log2 = ContradictionLog(tmp_path)
        log2.record(new_fact="new", old_facts=[], deleted_ids=[])
        # An unreadable log reads as empty (it cannot be served), but it
        # must not have destroyed anything.
        assert log2.get_recent() == []

    assert path.read_text() == before
    assert len(ContradictionLog(tmp_path).get_recent(limit=10)) == 5


@root_skip
def test_adaptive_threshold_refuses_to_overwrite_unreadable_state(tmp_path):
    at = AdaptiveThreshold(tmp_path, initial=0.7)
    for _ in range(25):
        at.record(0.6, True)
    path = tmp_path / "adaptive_threshold.json"
    before = json.loads(path.read_text())
    assert len(before["window"]) == 25

    with _Unreadable(path):
        at2 = AdaptiveThreshold(tmp_path, initial=0.7)
        assert at2.get_threshold() == 0.7          # fell back to initial
        for _ in range(25):
            at2.record(0.99, True)                 # must NOT write

    after = json.loads(path.read_text())
    assert after == before


@root_skip
def test_unreadable_load_is_logged_loudly_not_at_debug(tmp_path, caplog):
    cp = CompetenceProfile(tmp_path)
    cp.record("shell", "ls", success=True)
    path = tmp_path / "competence_profile.json"
    with _Unreadable(path):
        with caplog.at_level(logging.DEBUG, logger="GhostAgent"):
            CompetenceProfile(tmp_path)
    loud = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert loud, "an unreadable competence profile must not be a debug line"
    assert "unreadable" in " ".join(r.getMessage() for r in loud).lower()


def test_competence_corrupt_file_is_sidecarred_then_restarts(tmp_path):
    path = tmp_path / "competence_profile.json"
    path.write_text('{"shell|ls": {"alpha": 5.0, "be')     # truncated write
    cp = CompetenceProfile(tmp_path)
    cp.record("shell", "ls", success=True)

    sidecars = list(tmp_path.glob("competence_profile.corrupt-*"))
    assert len(sidecars) == 1
    assert sidecars[0].read_text().startswith('{"shell|ls"')
    assert cp.observations("shell", "ls") == 1


def test_contradiction_corrupt_file_is_sidecarred_then_restarts(tmp_path):
    path = tmp_path / "contradiction_log.json"
    path.write_text('[{"new_fact": "half writ')
    log = ContradictionLog(tmp_path)
    log.record(new_fact="fresh", old_facts=[], deleted_ids=[])

    sidecars = list(tmp_path.glob("contradiction_log.corrupt-*"))
    assert len(sidecars) == 1
    assert len(log.get_recent()) == 1


def test_adaptive_corrupt_file_is_sidecarred_then_restarts(tmp_path):
    path = tmp_path / "adaptive_threshold.json"
    path.write_text("{not json at all")
    at = AdaptiveThreshold(tmp_path, initial=0.7)
    assert at.get_threshold() == 0.7
    at.record(0.8, True)

    sidecars = list(tmp_path.glob("adaptive_threshold.corrupt-*"))
    assert len(sidecars) == 1
    assert json.loads(path.read_text())["window"]


def test_wrong_type_json_is_treated_as_corrupt(tmp_path):
    # Valid JSON of the WRONG shape breaks every caller; sidecar it too.
    (tmp_path / "competence_profile.json").write_text("[1, 2, 3]")
    (tmp_path / "contradiction_log.json").write_text('{"a": 1}')
    (tmp_path / "adaptive_threshold.json").write_text("[]")

    CompetenceProfile(tmp_path)
    ContradictionLog(tmp_path)
    AdaptiveThreshold(tmp_path)

    assert len(list(tmp_path.glob("*.corrupt-*"))) == 3


def test_absent_file_is_still_a_silent_cold_start(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        cp = CompetenceProfile(tmp_path)
        log = ContradictionLog(tmp_path)
        at = AdaptiveThreshold(tmp_path)
    assert cp.estimate("shell") == 0.5
    assert log.get_recent() == []
    assert at.get_threshold() == 0.7
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# =========================================================================
# 2. HIGH — journal take/ack crash safety
# =========================================================================


def _items(journal):
    return [e["type"] for e in journal.load()]


def test_pop_all_still_returns_and_clears(tmp_path):
    """Signature/behaviour for the un-updated drain is unchanged."""
    j = MemoryJournal(tmp_path, max_capacity=10)
    j.append("smart_memory", {"text": "a"})
    j.append("post_mortem", {"text": "b"})

    popped = j.pop_all()
    assert [e["type"] for e in popped] == ["smart_memory", "post_mortem"]
    assert j.load() == []
    assert json.loads((tmp_path / "memory_journal.json").read_text()) == []


def test_inflight_batch_survives_a_crash_mid_drain(tmp_path):
    j = MemoryJournal(tmp_path, max_capacity=10)
    for i in range(3):
        j.append("smart_memory", {"text": f"fact {i}"})

    taken = j.pop_all()          # drain takes ownership…
    assert len(taken) == 3
    assert (tmp_path / "memory_journal.inflight.json").exists()

    # …and the process is killed here (deploy): a brand-new instance is
    # the next boot. Recovered work folds to the overflow HEAD (drained
    # first), so it surfaces via pending_count()/pop_all(), not len(load()).
    reborn = MemoryJournal(tmp_path, max_capacity=10)
    assert reborn.pending_count() == 3
    # Recovery is one-shot: the staging file is consumed.
    assert not (tmp_path / "memory_journal.inflight.json").exists()
    assert [e["data"]["text"] for e in reborn.pop_all()] == \
        ["fact 0", "fact 1", "fact 2"]


def test_ack_prevents_replay(tmp_path):
    j = MemoryJournal(tmp_path, max_capacity=10)
    j.append("smart_memory", {"text": "done"})
    j.pop_all()
    j.ack()                       # consolidation persisted downstream

    assert MemoryJournal(tmp_path, max_capacity=10).load() == []


def test_partial_ack_replays_only_the_unfinished_items(tmp_path):
    j = MemoryJournal(tmp_path, max_capacity=10)
    j.append("smart_memory", {"text": "one"})
    j.append("smart_memory", {"text": "two"})
    taken = j.pop_all()
    j.ack([taken[0]])             # first item consolidated, then crash

    recovered = MemoryJournal(tmp_path, max_capacity=10).pop_all()
    assert [e["data"]["text"] for e in recovered] == ["two"]


def test_next_take_rotates_the_previous_batch_away(tmp_path):
    """The un-updated drain never acks; a fresh take is the implicit ack
    (the drain is awaited serially, so a new take means the last one
    returned). Otherwise every restart would replay stale work."""
    j = MemoryJournal(tmp_path, max_capacity=10)
    j.append("smart_memory", {"text": "old"})
    j.pop_all()
    j.append("smart_memory", {"text": "new"})
    j.pop_all()

    recovered = MemoryJournal(tmp_path, max_capacity=10).pop_all()
    assert [e["data"]["text"] for e in recovered] == ["new"]


def test_empty_take_acks_the_previous_batch(tmp_path):
    j = MemoryJournal(tmp_path, max_capacity=10)
    j.append("smart_memory", {"text": "x"})
    j.pop_all()
    assert j.pop_all() == []      # nothing left to do → previous batch done
    assert MemoryJournal(tmp_path, max_capacity=10).load() == []


def test_recovery_does_not_duplicate_when_crash_was_before_the_clear(tmp_path):
    # Crash in the ~1 ms window between staging and clearing the queue:
    # the item is in BOTH files and must come back exactly once.
    j = MemoryJournal(tmp_path, max_capacity=10)
    j.append("smart_memory", {"text": "both"})
    staged = j.load()
    (tmp_path / "memory_journal.inflight.json").write_text(json.dumps(staged))

    recovered = MemoryJournal(tmp_path, max_capacity=10).load()
    assert len(recovered) == 1


def test_recovered_items_go_to_the_front_of_the_queue(tmp_path):
    j = MemoryJournal(tmp_path, max_capacity=10)
    j.append("smart_memory", {"text": "in flight"})
    j.pop_all()
    # A new turn journals something before the restart happens.
    j2 = MemoryJournal(tmp_path, max_capacity=10)   # recovers on construction
    order = [e["data"]["text"] for e in j2.pop_all()]   # overflow head drains first
    assert order[0] == "in flight"


def test_inflight_corruption_does_not_wedge_the_journal(tmp_path):
    j = MemoryJournal(tmp_path, max_capacity=10)
    j.append("smart_memory", {"text": "queued"})
    (tmp_path / "memory_journal.inflight.json").write_text("{trunc")

    reborn = MemoryJournal(tmp_path, max_capacity=10)
    assert [e["data"]["text"] for e in reborn.load()] == ["queued"]
    assert list(tmp_path.glob("memory_journal.inflight.corrupt-*"))


def test_drain_shares_the_inflight_lifecycle(tmp_path):
    j = MemoryJournal(tmp_path, max_capacity=10)
    j.append("post_mortem", {"text": "dream work"})
    assert len(j.drain()) == 1
    # recovered on the next boot to the overflow head → counted by pending_count
    assert MemoryJournal(tmp_path, max_capacity=10).pending_count() == 1


@root_skip
def test_transient_read_error_still_propagates(tmp_path):
    """Unchanged fail-loud policy: an unreadable QUEUE must not read as
    empty (recovery must not paper over it either)."""
    j = MemoryJournal(tmp_path, max_capacity=10)
    j.append("smart_memory", {"text": "keep me"})
    with _Unreadable(j.file_path):
        with pytest.raises(OSError):
            MemoryJournal(tmp_path, max_capacity=10).load()
    assert len(j.load()) == 1


# =========================================================================
# 3. MED — widened transient classifier
# =========================================================================


def _http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://127.0.0.1:8088/v1/chat/completions")
    return httpx.HTTPStatusError(
        f"HTTP {status}", request=req, response=httpx.Response(status, request=req))


# The literal message core.llm raises after its own retries are exhausted.
_LLM_EMPTY_BODY = (
    "Upstream returned an empty/non-JSON response (HTTP 200, 0 bytes) after "
    "retry. This typically follows a context overflow or an upstream restart; "
    "the request did not complete."
)


@pytest.mark.parametrize("exc", [
    RuntimeError(_LLM_EMPTY_BODY),
    Exception("Max retries exceeded"),
    Exception("HTTP 503 Service Unavailable"),
    RuntimeError("upstream server disconnected without sending a response"),
    TimeoutError("read timed out"),
    ConnectionError("connection refused"),
    httpx.ReadTimeout("slow"),
    httpx.ConnectError("refused"),
    _http_error(500),
    _http_error(503),
])
def test_transient_errors_are_requeued(exc):
    assert is_upstream_transient(exc) is True


@pytest.mark.parametrize("exc", [
    _http_error(400),
    _http_error(404),
    _http_error(422),
    ValueError("bad json"),
    json.JSONDecodeError("Expecting value", "", 0),
    ValueError("request timed out"),      # type wins over the message
    TypeError("'str' object has no attribute 'get'"),
    KeyError("choices"),
    Exception("HTTP 400 Bad Request"),
    Exception("model not found"),
    Exception(""),
])
def test_definitive_errors_are_not_requeued(exc):
    assert is_upstream_transient(exc) is False


def test_classifier_never_raises_on_odd_input():
    class _Weird(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    assert is_upstream_transient(_Weird()) is False
    assert is_upstream_transient(None) is False


# =========================================================================
# 4. MED — profile singletons (corrections replace, not accumulate)
# =========================================================================


@pytest.mark.parametrize("key,first,second", [
    ("wife", "Anna", "Maria"),
    ("husband", "Nikos", "Petros"),
    ("son", "Thodoris", "Leonidas"),
    ("daughter", "Eleni", "Sofia"),
    ("car", "Tesla", "Volvo"),
])
def test_singular_profile_keys_replace_on_correction(tmp_path, key, first, second):
    pm = ProfileMemory(tmp_path)
    pm.update("relationships", key, first)
    pm.update("relationships", key, second)

    data = pm.load()
    stored = [v for cat in data.values() if isinstance(cat, dict)
              for k, v in cat.items() if k == ("car" if key == "car" else key)]
    assert stored == [second], f"{key} accumulated instead of being corrected"
    ctx = pm.get_context_string()
    assert first not in ctx and second in ctx


def test_plural_keys_still_merge(tmp_path):
    # The merge branch is deliberate for genuinely multi-valued facts.
    pm = ProfileMemory(tmp_path)
    pm.update("assets", "pets", "Hanzo the dog")
    pm.update("assets", "pets", "Mortimer the iguana")
    assert pm.load()["assets"]["pets"] == ["Hanzo the dog", "Mortimer the iguana"]

    pm.update("interests", "language", "python")
    pm.update("interests", "language", "rust")
    assert set(pm.load()["interests"]["language"]) == {"python", "rust"}


def test_canonicalised_vehicle_lands_on_the_singleton_car(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("assets", "vehicle", "Tesla")
    msg = pm.update("assets", "vehicle", "Volvo")
    assert "normalised from" in msg
    assert pm.load()["assets"]["car"] == "Volvo"


# =========================================================================
# 5. MED — notes.info (the malformed-profile_update sink) is bounded
# =========================================================================


def test_notes_info_sink_is_bounded(tmp_path):
    # core.agent does profile_up.get("category","notes"), .get("key","info"),
    # so every malformed profile_update lands here with the whole fact.
    pm = ProfileMemory(tmp_path)
    for i in range(40):
        pm.update("notes", "info", f"stray fact number {i}")

    stored = pm.load()["notes"]["info"]
    assert isinstance(stored, list)
    assert len(stored) <= 3
    # Newest survive, oldest are dropped.
    assert "stray fact number 39" in stored
    assert "stray fact number 0" not in stored


def test_notes_info_values_are_truncated(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("notes", "info", "seed")
    pm.update("notes", "info", "x" * 5000)
    rendered = pm.get_context_string()
    assert len(rendered) < 1000


def test_context_string_stays_bounded_under_sustained_junk(tmp_path):
    pm = ProfileMemory(tmp_path)
    for i in range(200):
        pm.update("notes", "info", f"junk {i} " + "y" * 300)
    assert len(pm.get_context_string()) < 1200


def test_generic_merge_keys_are_capped_too(tmp_path):
    pm = ProfileMemory(tmp_path)
    for i in range(30):
        pm.update("interests", "topics", f"topic {i}")
    topics = pm.load()["interests"]["topics"]
    assert len(topics) <= 8
    assert "topic 29" in topics


# =========================================================================
# 6. MED — capacity overflow is visible; requeued items survive it
# =========================================================================


def test_capacity_overflow_spills_losslessly(tmp_path, caplog):
    # 2026-07-23: overflow past the hot cap now SPILLS to the overflow file
    # (drained oldest-first) instead of dropping the oldest. No WARNING — it
    # is a lossless INFO-level spill, not data loss.
    j = MemoryJournal(tmp_path, max_capacity=3)
    for i in range(3):
        j.append("smart_memory", {"text": f"f{i}"})
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        j.append("smart_memory", {"text": "overflow"})

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    # hot buffer holds the newest cap; f0 spilled but is NOT lost.
    assert [e["data"]["text"] for e in j.load()] == ["f1", "f2", "overflow"]
    assert j.pending_count() == 4
    assert [e["data"]["text"] for e in j.pop_all()] == ["f0", "f1", "f2", "overflow"]


def test_no_warning_when_under_capacity(tmp_path, caplog):
    j = MemoryJournal(tmp_path, max_capacity=5)
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        for i in range(5):
            j.append("smart_memory", {"text": f"f{i}"})
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_push_front_overflow_preserves_retried_items(tmp_path):
    """core.agent pushes back ``requeue + items[i:]`` — the transient retries
    sit at the HEAD. Since 2026-07-23 push_front folds to the overflow HEAD and
    is LOSSLESS: everything is retained (past the cap too) and the requeued
    retries drain FIRST."""
    j = MemoryJournal(tmp_path, max_capacity=3)
    requeue = [
        {"type": "smart_memory", "data": {"text": "retry a"}, "retries": 1},
        {"type": "post_mortem", "data": {"text": "retry b"}, "retries": 2},
    ]
    unprocessed = [
        {"type": "smart_memory", "data": {"text": f"pending {i}"}}
        for i in range(4)
    ]
    j.push_front(requeue + unprocessed)

    assert j.pending_count() == 6                      # nothing dropped
    drained = [e["data"]["text"] for e in j.pop_all()]
    assert drained[:2] == ["retry a", "retry b"]       # retries drain first
    assert set(drained) == {"retry a", "retry b",
                            "pending 0", "pending 1", "pending 2", "pending 3"}


def test_push_front_without_retry_stamps_is_lossless(tmp_path):
    j = MemoryJournal(tmp_path, max_capacity=3)
    items = [{"type": "note", "data": {"n": i}} for i in range(5)]
    j.push_front(items)
    # Past the hot cap, but nothing dropped — drains in order (overflow head).
    assert [e["data"]["n"] for e in j.pop_all()] == [0, 1, 2, 3, 4]


def test_push_front_under_capacity_is_drained_front_first(tmp_path):
    j = MemoryJournal(tmp_path, max_capacity=5)
    j.append("tail", {"n": 1})
    j.push_front([{"type": "head", "data": {"n": 0}}])
    # push_front goes to the overflow head (invisible to load(), drained first).
    assert [e["type"] for e in j.load()] == ["tail"]
    assert [e["type"] for e in j.pop_all()] == ["head", "tail"]


# =========================================================================
# 7. improvements — durable write, integer sample count
# =========================================================================


def test_journal_save_fsyncs_before_replace(tmp_path, monkeypatch):
    calls = {"fsync": 0}
    real_fsync = os.fsync

    def _counting_fsync(fd):
        calls["fsync"] += 1
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _counting_fsync)
    j = MemoryJournal(tmp_path, max_capacity=5)
    j.append("smart_memory", {"text": "durable"})
    assert calls["fsync"] >= 1
    assert not list(tmp_path.glob("*.tmp"))


def test_competence_counts_low_weight_records(tmp_path):
    """`n` used to be int((alpha-1)+(beta-1)), so weighted records below
    1.0 were invisible to the ``n >= 1`` gate in estimate()."""
    cp = CompetenceProfile(tmp_path)
    for _ in range(4):
        cp.record("vision", "describe", success=True, weight=0.05)

    assert cp.observations("vision", "describe") == 4
    # …and the cell is now actually consulted instead of the neutral prior.
    assert cp.estimate("vision", "describe") != 0.5
    assert "n=4" in cp.get_context_string()

    reopened = CompetenceProfile(tmp_path)
    assert reopened.observations("vision", "describe") == 4


def test_competence_legacy_file_without_n_still_counts(tmp_path):
    (tmp_path / "competence_profile.json").write_text(
        json.dumps({"shell|ls": {"alpha": 4.0, "beta": 2.0}}))
    cp = CompetenceProfile(tmp_path)
    assert cp.observations("shell", "ls") == 4      # (4-1)+(2-1)


def test_competence_flush_interval_is_opt_in(tmp_path):
    # Default = write-through (the durability contract the drain relies on).
    cp = CompetenceProfile(tmp_path)
    cp.record("shell", "ls", success=True)
    assert CompetenceProfile(tmp_path).observations("shell", "ls") == 1

    # Opt-in debounce holds writes back, and flush() forces them out.
    debounced = CompetenceProfile(tmp_path, flush_interval=60.0)
    for _ in range(3):
        debounced.record("sql", "select", success=True)
    assert CompetenceProfile(tmp_path).observations("sql", "select") == 0
    debounced.flush()
    assert CompetenceProfile(tmp_path).observations("sql", "select") == 3
