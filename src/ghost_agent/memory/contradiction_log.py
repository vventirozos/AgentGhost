"""Contradiction resolution log.

Records every belief revision performed by the smart-memory contradiction
engine so the agent can explain *why* it changed its mind:

    "I previously thought X, but updated to Y because you said Z."

Persisted as a JSON file alongside the other memory stores.
"""

import json
import logging
import os
import threading
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("GhostAgent")


class ContradictionLog:
    MAX_ENTRIES = 200

    def __init__(self, memory_dir: Path):
        self.file_path = memory_dir / "contradiction_log.json"
        self._lock = threading.RLock()
        if not self.file_path.exists():
            self._save([])

    def _save(self, entries: list):
        tmp = self.file_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, indent=2))
        os.replace(tmp, self.file_path)

    def _load(self) -> list:
        try:
            data = json.loads(self.file_path.read_text())
            # Wrong-type (dict/scalar) would break record()/get_recent which
            # expect a list — treat as empty.
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def record(self, new_fact: str, old_facts: list, deleted_ids: list, reason: str = ""):
        """Record a belief revision event.

        Parameters
        ----------
        new_fact : str
            The incoming fact that triggered the revision.
        old_facts : list[dict]
            The old facts that were superseded (each has 'id' and 'text').
        deleted_ids : list[str]
            IDs of the memories that were actually deleted.
        reason : str, optional
            Free-form reason from the contradiction engine.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "new_fact": new_fact,
            "superseded": [
                {"id": f.get("id", "?"), "text": f.get("text", "?")}
                for f in old_facts
                if f.get("id") in deleted_ids or str(f.get("id", "")).replace("ID:", "").strip() in deleted_ids
            ],
            "deleted_ids": deleted_ids,
            "reason": reason,
        }
        with self._lock:
            entries = self._load()
            entries.insert(0, entry)
            if len(entries) > self.MAX_ENTRIES:
                entries = entries[:self.MAX_ENTRIES]
            self._save(entries)
        logger.debug(f"Contradiction logged: '{new_fact[:60]}' superseded {len(deleted_ids)} old facts")

    def get_recent(self, limit: int = 10) -> list:
        """Return the most recent contradiction events."""
        with self._lock:
            entries = self._load()
        return entries[:limit]

    def explain_belief_change(self, query: str) -> str:
        """Search the log for contradictions related to a query term.

        Returns a human-readable explanation if found, empty string otherwise."""
        if not query:
            return ""
        query_lower = query.lower()
        with self._lock:
            entries = self._load()
        matches = []
        for entry in entries[:50]:  # Only search recent entries
            new_fact = (entry.get("new_fact") or "").lower()
            old_texts = " ".join(
                s.get("text", "").lower() for s in entry.get("superseded", [])
            )
            if query_lower in new_fact or query_lower in old_texts:
                matches.append(entry)
        if not matches:
            return ""

        lines = ["## BELIEF REVISION HISTORY:"]
        for m in matches[:5]:
            old_strs = "; ".join(
                s.get("text", "?")[:80] for s in m.get("superseded", [])
            )
            lines.append(
                f"- [{m.get('timestamp', '?')}] Updated to: \"{m.get('new_fact', '?')[:100]}\""
                f" (superseded: \"{old_strs}\")"
            )
        return "\n".join(lines)

    def clear(self):
        """Wipe the contradiction log."""
        with self._lock:
            self._save([])
