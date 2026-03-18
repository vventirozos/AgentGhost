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
            new_journal = items + journal
            if len(new_journal) > self.max_capacity:
                new_journal = new_journal[:self.max_capacity]
            self._save(new_journal)
