import json
import logging
import re
import threading
import time
import os
from pathlib import Path

logger = logging.getLogger("GhostAgent")

# How many times a journal item that failed on an upstream-transient
# error gets re-queued before it is dropped for good. Bounded so a
# permanently-broken item can't pin the drain loop forever.
JOURNAL_MAX_RETRIES = 2


class RetryableConsolidationError(Exception):
    """An upstream-transient failure (5xx / timeout / connection error)
    inside a journal-drain task, raised AFTER the in-client retries
    (worker-node failover + one 5xx retry) are exhausted and BEFORE any
    memory write happened. The drain loop catches this and re-queues the
    journal item (bounded by ``JOURNAL_MAX_RETRIES``) instead of losing
    the consolidation — the item was already popped, so an ordinary
    log-and-swallow made the drop permanent and invisible."""


# --- transient classification -------------------------------------------
# Phrases that only ever appear in a failure whose *cause* is upstream
# state (a restarted / busy / disconnected llama-server), i.e. re-running
# the identical request later can succeed. These exist because
# ``core.llm`` deliberately re-raises some upstream failures as PLAIN
# ``RuntimeError`` / ``Exception`` with an explanatory message instead of
# the original httpx type, so a type-only classifier called them
# definitive and the consolidation was dropped:
#   * llm.py `raise RuntimeError("Upstream returned an empty/non-JSON
#     response … the request did not complete.")` (post context-overflow
#     / upstream restart)
#   * llm.py `raise Exception("Max retries exceeded")` at the end of each
#     retry loop.
_TRANSIENT_MESSAGE_MARKERS = (
    "max retries exceeded",
    "the request did not complete",
    "upstream restart",
    "empty/non-json response",
    "timed out",
    "timeout",
    "connection reset",
    "connection refused",
    "connection aborted",
    "connection error",
    "server disconnected",
    "remote end closed",
    "broken pipe",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "internal server error",
    "server is busy",
    "slot unavailable",
)

# An HTTP status rendered INTO a message string. 5xx = retry, 4xx = never
# (the request was rejected on its own merits and will be rejected again).
_HTTP_5XX_IN_MESSAGE = re.compile(r"\bhttp[ /]?5\d\d\b|\b5\d\d\s+(?:server error|internal|bad gateway|service unavailable)\b")
_HTTP_4XX_IN_MESSAGE = re.compile(r"\bhttp[ /]?4\d\d\b")


def is_upstream_transient(exc: BaseException) -> bool:
    """True for failures worth re-running the same task against later:
    timeouts, connection-level errors, and HTTP 5xx. A 4xx or a parsing
    error would fail identically on retry, so it is NOT transient.

    Boundary (deliberately conservative — a too-broad classifier makes a
    permanently-broken item retry until ``JOURNAL_MAX_RETRIES`` burns out
    on every drain):

    1. TYPE first, and *definitively* so. ``ValueError`` (which
       ``json.JSONDecodeError`` subclasses), ``TypeError``, ``KeyError``,
       ``AttributeError``, ``IndexError`` are bugs or malformed payloads:
       always False, and short-circuited BEFORE any message matching so a
       decoder error whose text happens to contain "timeout" can't sneak
       through.
    2. httpx types keep their existing authoritative meaning: 5xx →
       transient, anything else (4xx included) → definitive, with no
       fall-through to the message heuristics.
    3. Builtin transport-ish types (``TimeoutError`` — which
       ``asyncio.TimeoutError`` aliases on 3.11+ — and ``ConnectionError``).
    4. ONLY THEN a message allow-list, for the plain
       ``RuntimeError``/``Exception`` values ``core.llm`` raises after it
       has exhausted its own retries.
    """
    if exc is None:
        return False

    # (1) Structurally definitive: parse / programming errors.
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError,
                        IndexError, NotImplementedError)):
        return False

    # (2) httpx — authoritative, never falls through to text matching.
    try:
        import httpx
    except Exception:  # pragma: no cover - httpx is a hard dep in prod
        httpx = None
    if httpx is not None:
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                return int(exc.response.status_code) >= 500
            except Exception:
                return False
        # TimeoutException subclasses TransportError in httpx, but name both:
        # the timeout case is the one the in-client retry loop does NOT cover.
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True

    # (3) Builtin transport failures (asyncio.TimeoutError is TimeoutError).
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    # (4) Message allow-list for the re-raised-as-plain-Exception cases.
    try:
        msg = str(exc).lower()
    except Exception:
        return False
    if not msg:
        return False
    if _HTTP_4XX_IN_MESSAGE.search(msg):
        return False
    if _HTTP_5XX_IN_MESSAGE.search(msg):
        return True
    return any(m in msg for m in _TRANSIENT_MESSAGE_MARKERS)


def _redact_journal_data(data):
    """Best-effort redaction of secret-shaped strings in a journal entry
    before it hits disk. Tool-output dumps (e.g. an ``env`` listing, a DB
    URI) can carry secrets; the local journal is a softer target than the
    already-redacted trajectory corpus, so we run the same scrubber here.
    Never raises — redaction failure must not drop the journal entry."""
    try:
        from ..distill.redact import redact_text
    except Exception:
        return data

    def _walk(v):
        if isinstance(v, str):
            return redact_text(v)
        if isinstance(v, dict):
            return {k: _walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_walk(x) for x in v]
        return v

    try:
        return _walk(data)
    except Exception:
        return data


class MemoryJournal:
    """Crash-safe buffer of consolidations waiting for an idle window.

    TAKE / ACK lifecycle (2026-07-22)
    ---------------------------------
    ``pop_all()`` used to empty the on-disk queue in one atomic write and
    hand the items to the drain loop, which then spends up to 90 s of LLM
    time PER ITEM. ``asyncio.CancelledError`` is a ``BaseException``, so
    the drain's ``except Exception`` does not catch it and it has no
    ``finally`` — the documented deploy procedure (a plain ``kill``)
    landing mid-drain therefore destroyed up to ``max_capacity`` buffered
    facts and lessons, silently: the file was already ``[]``, so nothing
    on restart could tell that anything had been lost.

    The queue now hands work over in two steps:

    ``pop_all()``   copies the batch to an on-disk IN-FLIGHT staging file
                    (``memory_journal.inflight.json``) BEFORE clearing the
                    queue, then returns the items exactly as before —
                    signature and semantics unchanged, so the un-updated
                    drain in ``core.agent`` keeps working untouched.
    ``ack()``       clears (or partially clears) the staging file once the
                    caller has durably consumed the batch. Optional: a
                    caller that never acks still gets at-least-once
                    delivery, see below.
    ``recover_inflight()``
                    folds an orphaned staging batch back to the FRONT of
                    the queue. Runs automatically on construction and on
                    the first ``load()`` of the process, so recovery needs
                    no cooperation from any caller.

    Because the current drain cannot ack (it is owned by another module),
    the staging file is also ROTATED on the next ``pop_all()``: a fresh
    take implies the previous drain returned, since the drain is awaited
    serially. Net effect:

      * crash / cancel mid-drain  → items come back on next boot
        (at-least-once: an item whose consolidation had already been
        written may be re-consolidated once — the memory layer dedups
        facts, and re-doing a consolidation is cheap next to losing 50);
      * normal operation          → staging file is rotated away, no
        replay, no growth.

    A crash in the ~1 ms window between staging and clearing the queue
    would leave the item in BOTH files; recovery de-duplicates against the
    queue contents, so that window cannot double-deliver either.
    """

    def __init__(self, path: Path, max_capacity: int = 50):
        self.file_path = path / "memory_journal.json"
        self.inflight_path = path / "memory_journal.inflight.json"
        self.max_capacity = max_capacity
        self._lock = threading.RLock()
        self._recovered = False
        if not self.file_path.exists():
            self._save([])
        try:
            self.recover_inflight()
        except Exception as exc:
            # Never let a boot-time recovery problem take the agent down;
            # `load()` retries it (recover_inflight resets the flag).
            logger.warning("Journal in-flight recovery deferred: %s", exc)

    # ------------------------------------------------------------ storage

    def _atomic_write(self, target: Path, data) -> None:
        """Write ``data`` as JSON to ``target`` durably.

        ``os.replace`` is atomic w.r.t. the DIRECTORY entry, but without an
        fsync the temp file's CONTENT may still be in the page cache when
        the machine loses power — the rename then publishes a zero-length
        or torn file. The journal is small and written rarely (once per
        append / take), so the fsync is free in practice.
        """
        temp_path = target.with_suffix('.tmp')
        payload = json.dumps(data, indent=2)
        with open(temp_path, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, target)

    def _save(self, data):
        self._atomic_write(self.file_path, data)

    def _quarantine_corrupt(self) -> list:
        """Corruption: PRESERVE the raw bytes in a timestamped sidecar
        BEFORE any subsequent _save() overwrites them, then start clean.
        Without this, a partial write silently discarded every queued
        post_mortem / smart_memory entry (the dream consolidator's work
        queue). Matches the SkillMemory / FrontierTracker recovery policy."""
        try:
            sidecar = self.file_path.with_suffix(f".corrupt-{int(time.time())}.json")
            os.replace(self.file_path, sidecar)
            logger.warning(
                "memory_journal.json was corrupt; preserved to %s and "
                "started a fresh journal.", sidecar.name,
            )
        except Exception:
            pass
        return []

    def load(self):
        with self._lock:
            # First load of the process folds any orphaned in-flight batch
            # (a drain killed mid-flight) back into the queue.
            self._maybe_recover_inflight()
            return self._read_queue()

    def _read_queue(self):
        with self._lock:
            try:
                content = self.file_path.read_text()
            except FileNotFoundError:
                return []
            except UnicodeDecodeError:
                # Undecodable bytes are corruption, not disk sickness —
                # same preserve-and-restart path as bad JSON below.
                return self._quarantine_corrupt()
            # Any OTHER OSError (EIO, EACCES, disk full…) propagates:
            # treating a transient read failure as "empty journal" let the
            # next append() atomically overwrite the intact on-disk queue
            # with a single-entry list — silent data loss. Callers' append
            # wrappers log a warning (same fail-loud policy as
            # SkillMemory._load_playbook).
            if not content.strip():
                return []
            try:
                data = json.loads(content)
                # A valid-JSON-but-wrong-TYPE file (dict/scalar) would break
                # append()/pop_all() (which expect a list). Treat as corrupt.
                if not isinstance(data, list):
                    raise ValueError(f"journal is a {type(data).__name__}, expected list")
                return data
            except Exception:
                return self._quarantine_corrupt()

    # ----------------------------------------------------- in-flight queue

    def _read_inflight(self) -> list:
        """Contents of the staging file, or [] when there is none.

        Corruption is quarantined like the queue's; a transient OSError
        propagates (same fail-loud rule — pretending the staging file is
        empty is exactly the silent-loss bug this mechanism exists to
        prevent)."""
        try:
            content = self.inflight_path.read_text()
        except FileNotFoundError:
            return []
        except UnicodeDecodeError:
            content = ""
        if not content.strip():
            return []
        try:
            data = json.loads(content)
            if not isinstance(data, list):
                raise ValueError(
                    f"in-flight is a {type(data).__name__}, expected list")
            return data
        except Exception as exc:
            try:
                sidecar = self.inflight_path.with_suffix(
                    f".corrupt-{int(time.time())}.json")
                os.replace(self.inflight_path, sidecar)
                logger.warning(
                    "memory_journal.inflight.json was corrupt (%s); preserved "
                    "to %s.", exc, sidecar.name,
                )
            except Exception:
                pass
            return []

    def _clear_inflight(self) -> None:
        try:
            self.inflight_path.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not clear journal in-flight file: %s", exc)

    def _maybe_recover_inflight(self) -> int:
        if self._recovered:
            return 0
        return self.recover_inflight()

    def recover_inflight(self) -> int:
        """Fold an orphaned in-flight batch back into the FRONT of the
        queue and clear the staging file. Returns how many items came
        back. Idempotent, and automatic (construction + first ``load()``),
        so a restart after a mid-drain kill loses nothing."""
        with self._lock:
            self._recovered = True  # set first: _read_queue() calls back in
            try:
                staged = self._read_inflight()
                if not staged:
                    self._clear_inflight()
                    return 0
                queue = self._read_queue()
                # De-dup against the queue: a crash in the window between
                # staging and clearing leaves an item in BOTH files.
                seen = set()
                for entry in queue:
                    try:
                        seen.add(json.dumps(entry, sort_keys=True))
                    except Exception:
                        pass
                fresh = []
                for entry in staged:
                    try:
                        key = json.dumps(entry, sort_keys=True)
                    except Exception:
                        key = None
                    if key is not None and key in seen:
                        continue
                    fresh.append(entry)
                if fresh:
                    self._save(self._merge_front(fresh, queue))
                self._clear_inflight()
                logger.warning(
                    "Recovered %d in-flight journal item(s) from an "
                    "interrupted drain (%d already back in the queue).",
                    len(fresh), len(staged) - len(fresh),
                )
                return len(fresh)
            except Exception:
                # Let the next load() try again rather than skipping
                # recovery for the whole process lifetime.
                self._recovered = False
                raise

    def ack(self, items: list = None) -> None:
        """Confirm that a batch handed out by ``pop_all``/``drain`` has
        been durably consumed, so it will NOT be replayed on restart.

        ``ack()`` clears the whole staging batch. ``ack(items)`` clears
        only those entries (partial ack for a drain that processes item by
        item). Callers that never ack still get at-least-once delivery —
        the batch is rotated away by the next take."""
        with self._lock:
            if items is None:
                self._clear_inflight()
                return
            staged = self._read_inflight()
            if not staged:
                return
            done = set()
            for entry in items or []:
                try:
                    done.add(json.dumps(entry, sort_keys=True))
                except Exception:
                    pass
            remaining = []
            for entry in staged:
                try:
                    key = json.dumps(entry, sort_keys=True)
                except Exception:
                    key = None
                if key is not None and key in done:
                    continue
                remaining.append(entry)
            if remaining:
                self._atomic_write(self.inflight_path, remaining)
            else:
                self._clear_inflight()

    # Alias: "commit" reads better at a transactional call site.
    commit = ack

    def inflight(self) -> list:
        """Items currently handed out but not yet acked (diagnostics)."""
        with self._lock:
            return self._read_inflight()

    # ------------------------------------------------------------- public

    def append(self, item_type: str, data: dict):
        data = _redact_journal_data(data)
        with self._lock:
            journal = self.load()
            journal.append({"type": item_type, "data": data})
            if len(journal) > self.max_capacity:
                # Silent until 2026-07-22: ~25 back-to-back turns overflow
                # the default cap of 50 before the drain's idle window ever
                # opens, and the OLDEST buffered consolidations were dropped
                # with no log at any level — while the drain loop logs a
                # WARNING for dropping a single item. Make the loss visible.
                dropped = len(journal) - self.max_capacity
                journal = journal[-self.max_capacity:]
                logger.warning(
                    "Memory journal is full (capacity %d): discarded %d oldest "
                    "buffered item(s) to make room for a new %s. The drain "
                    "(~2 min idle) is not keeping up.",
                    self.max_capacity, dropped, item_type,
                )
            self._save(journal)

    def _take_all(self) -> list:
        """Shared body of pop_all/drain: stage the batch as in-flight,
        clear the queue, return the items."""
        with self._lock:
            journal = self.load()
            if not journal:
                # Nothing to take → the previous batch (if any) is done.
                self._clear_inflight()
                return journal
            try:
                # Stage BEFORE clearing. Rotating the file here is also the
                # implicit ack of the previous batch: the drain is awaited
                # serially, so a fresh take means the last one returned.
                self._atomic_write(self.inflight_path, journal)
            except Exception as exc:
                # Staging is an availability win, not a precondition — do
                # not block the drain on it, but say so loudly.
                logger.warning(
                    "Journal in-flight staging failed (%s); this batch of %d "
                    "is not crash-recoverable.", exc, len(journal),
                )
            self._save([])
            return journal

    def pop_all(self):
        return self._take_all()

    def push_front(self, items: list):
        if not items: return
        with self._lock:
            journal = self.load()
            self._save(self._merge_front(items, journal))

    def _merge_front(self, items: list, journal: list) -> list:
        """Put ``items`` at the head of ``journal``, honouring capacity."""
        combined = items + journal
        if len(combined) <= self.max_capacity:
            return combined
        if len(items) <= self.max_capacity:
            # Preserve the re-queued items at the head. push_front is
            # called to requeue work the consolidator could not finish
            # because the user returned — dropping those items would
            # silently erase history we were explicitly trying to
            # save, so we drop the tail (most-recent appends, which
            # will be re-captured by the next journaling cycle).
            logger.warning(
                "Memory journal is full (capacity %d): re-queueing %d item(s) "
                "discarded %d newest buffered item(s).",
                self.max_capacity, len(items), len(combined) - self.max_capacity,
            )
            return combined[:self.max_capacity]
        # Pathological: more items than the journal can hold. The caller
        # (core.agent's drain) passes `requeue + items[i:]`, i.e. the
        # transient-failure RETRIES sit at the HEAD — and `items[-cap:]`
        # sliced exactly those off first, preferentially destroying the
        # entries this mechanism exists to protect. Retries (stamped with
        # a `retries` count by the drain) now win the capacity fight; the
        # remaining room goes to the most-recent of the rest.
        def _is_retry(entry) -> bool:
            return isinstance(entry, dict) and bool(entry.get("retries"))

        retried = [e for e in items if _is_retry(e)]
        logger.warning(
            "Memory journal overflow on re-queue: %d item(s) for %d slots — "
            "keeping %d retried item(s) and the most recent of the rest.",
            len(items), self.max_capacity, min(len(retried), self.max_capacity),
        )
        if not retried:
            # No retry stamps to protect: keep the most recent (unchanged
            # legacy behaviour).
            return items[-self.max_capacity:]
        keep = retried[-self.max_capacity:]
        room = self.max_capacity - len(keep)
        if room <= 0:
            return keep
        others = [e for e in items if not _is_retry(e)]
        return keep + others[-room:] if others else keep

    def drain(self) -> list:
        """Atomically return and clear all journal entries.

        Used by the dream consolidator to take ownership of the journal
        contents in a single critical section. Equivalent to `pop_all`
        (same in-flight staging / ack lifecycle) but named to make the
        lifecycle (drain → consolidate → discard) explicit at the call
        site.
        """
        return self._take_all()
