import json
import threading
import os
from pathlib import Path

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
            try: return json.loads(self.file_path.read_text())
            except: return []

    def append(self, item_type: str, data: dict):
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
