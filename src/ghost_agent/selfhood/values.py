"""Operating-principles substrate — the NORMATIVE layer of selfhood.

Selfhood previously tracked only *episodic* continuity (what I did, what
I'm still chewing on, my mood). It had no values / identity / goals: the
functional-test finding was that mood and open-questions feed *only* the
wake-up prefix string and steer nothing, and that the agent's qualitative
self-reports are paraphrase-unstable (the +0.40 confabulation gap)
precisely because there is no stable internal state to anchor to.

``ValuesThread`` adds a small, bounded, agent-authored list of operating
principles ("I verify before asserting", "I prefer reversible actions").
Unlike mood, principles are:

  * **persistent** across sessions (JSON, atomic temp+rename write,
    corrupt-recovery — the same discipline as ``SelfStateThread``);
  * **surfaced in the wake-up prefix every turn**, so they are in-context
    and shape generation — the move from cosmetic to behaviour-influencing;
  * **explicit and checkable**, so they can back a self-critique gate
    (``core.agent`` pre-final-response) and give confabulation-prone
    self-reports a stable thing to anchor to.

Bounded (most-recent-wins) so the prefix stays compact.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .schema import _utcnow_iso

logger = logging.getLogger("GhostSelfhood")

VALUES_FILENAME = "values.json"
MAX_PRINCIPLES = 12


@dataclass
class Principle:
    text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    added_at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "added_at": self.added_at}

    @classmethod
    def from_dict(cls, d: dict) -> "Principle":
        return cls(
            text=str(d.get("text", "")),
            id=str(d.get("id") or uuid.uuid4().hex),
            added_at=str(d.get("added_at") or _utcnow_iso()),
        )


class ValuesThread:
    """Single-file persisted operating principles.

    Read-on-construct, write-on-every-mutation (the file is tiny). A
    corrupt / partial file is treated as "no principles" rather than
    crashing — same fail-safe posture as the rest of selfhood."""

    def __init__(self, root: Path, *, enabled: bool = True):
        self.root = Path(root)
        self.path = self.root / VALUES_FILENAME
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self._principles: List[Principle] = self._read_or_empty()

    # ------------------------------------------------------- persistence

    def _read_or_empty(self) -> List[Principle]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("values read failed (%s); starting empty", e)
            return []
        if not isinstance(data, list):
            return []
        out: List[Principle] = []
        for d in data:
            if isinstance(d, dict) and str(d.get("text", "")).strip():
                out.append(Principle.from_dict(d))
        return out

    def _flush(self) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps([p.to_dict() for p in self._principles],
                               ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp.replace(self.path)
        except Exception as e:
            logger.warning("values flush failed: %s", e)

    # ------------------------------------------------------- read / write

    def principles(self) -> List[Principle]:
        with self._lock:
            return list(self._principles)

    def note_principle(self, text: str) -> Optional[Principle]:
        text = (text or "").strip()
        if not text:
            return None
        with self._lock:
            # Dedup by case-insensitive text — re-noting is a no-op.
            for p in self._principles:
                if p.text.lower() == text.lower():
                    return p
            p = Principle(text=text)
            self._principles.append(p)
            overflow = len(self._principles) - MAX_PRINCIPLES
            if overflow > 0:
                del self._principles[:overflow]  # most-recent-wins
            self._flush()
            return p

    def remove_principle(self, principle_id: str) -> bool:
        with self._lock:
            before = len(self._principles)
            self._principles = [p for p in self._principles if p.id != principle_id]
            if len(self._principles) != before:
                self._flush()
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._principles = []
            self._flush()

    # ------------------------------------------------------- rendering

    def as_text(self) -> str:
        """Plain bulleted list of principle texts (for a self-critique
        gate prompt). Empty string when there are none."""
        with self._lock:
            if not self._principles:
                return ""
            return "\n".join(f"- {p.text}" for p in self._principles)

    def format_as_prefix(self, *, max_chars: int = 800) -> str:
        """First-person principles block for the wake-up prefix. Empty
        when there are no principles (a hollow header is noise)."""
        with self._lock:
            if not self._principles:
                return ""
            lines = ["My operating principles (how I choose to work):"]
            for p in self._principles:
                lines.append(f"  - {p.text}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        return text


__all__ = ["ValuesThread", "Principle", "MAX_PRINCIPLES"]
