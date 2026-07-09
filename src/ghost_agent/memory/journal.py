import json
import logging
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


def is_upstream_transient(exc: BaseException) -> bool:
    """True for failures worth re-running the same task against later:
    timeouts, connection-level errors, and HTTP 5xx. A 4xx or a parsing
    error would fail identically on retry, so it is NOT transient."""
    try:
        import httpx
    except Exception:  # pragma: no cover - httpx is a hard dep in prod
        return False
    # TimeoutException subclasses TransportError in httpx, but name both:
    # the timeout case is the one the in-client retry loop does NOT cover.
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return int(exc.response.status_code) >= 500
        except Exception:
            return False
    return False


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
    def __init__(self, path: Path, max_capacity: int = 50):
        self.file_path = path / "memory_journal.json"
        self.max_capacity = max_capacity
        self._lock = threading.RLock()
        if not self.file_path.exists():
            self._save([])

    def _save(self, data):
        temp_path = self.file_path.with_suffix('.tmp')
        temp_path.write_text(json.dumps(data, indent=2))
        os.replace(temp_path, self.file_path)

    def load(self):
        with self._lock:
            try:
                content = self.file_path.read_text()
            except FileNotFoundError:
                return []
            except Exception:
                return []
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
                # Corruption: PRESERVE the raw bytes in a timestamped
                # sidecar BEFORE any subsequent _save() overwrites them,
                # then start clean. Without this, a partial write silently
                # discarded every queued post_mortem / smart_memory entry
                # (the dream consolidator's work queue). Matches the
                # SkillMemory / FrontierTracker recovery policy.
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

    def append(self, item_type: str, data: dict):
        data = _redact_journal_data(data)
        with self._lock:
            journal = self.load()
            journal.append({"type": item_type, "data": data})
            if len(journal) > self.max_capacity:
                journal = journal[-self.max_capacity:]
            self._save(journal)

    def pop_all(self):
        with self._lock:
            journal = self.load()
            if journal:
                self._save([])
            return journal

    def push_front(self, items: list):
        if not items: return
        with self._lock:
            journal = self.load()
            combined = items + journal
            if len(combined) <= self.max_capacity:
                new_journal = combined
            elif len(items) <= self.max_capacity:
                # Preserve the re-queued items at the head. push_front is
                # called to requeue work the consolidator could not finish
                # because the user returned — dropping those items would
                # silently erase history we were explicitly trying to
                # save, so we drop the tail (most-recent appends, which
                # will be re-captured by the next journaling cycle).
                new_journal = combined[:self.max_capacity]
            else:
                # Pathological: more requeued items than the journal can
                # hold. Keep the most-recent of `items` (the tail of the
                # list, which is the latest append). We cannot honour both
                # invariants simultaneously, so we honour "no recency loss"
                # over "preserve every requeued entry".
                new_journal = items[-self.max_capacity:]
            self._save(new_journal)

    def drain(self) -> list:
        """Atomically return and clear all journal entries.

        Used by the dream consolidator to take ownership of the journal
        contents in a single critical section. Equivalent to `pop_all`
        but named to make the lifecycle (drain → consolidate → discard)
        explicit at the call site.
        """
        with self._lock:
            journal = self.load()
            if journal:
                self._save([])
            return journal
