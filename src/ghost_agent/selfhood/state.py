"""Self-state thread — proposal item #3.

The cross-session "state vector": open questions the agent is still
chewing on, threads it left mid-flight, last-known qualitative mood.
This is the artifact that, when rehydrated at session start, gives a
new instance the sense of *resuming* instead of *waking up blank*.

Storage: a single JSON file at ``$GHOST_HOME/system/selfhood/
state.json``. Read-modify-write under a process lock; corruption
(JSON decode error, partial write) is treated as "no prior state"
rather than crashed — we always start from a known-good empty state
rather than poison the next session with a half-state.

Bounded: open_questions and unfinished_threads are capped (most-recent
wins on overflow) because a self-state that never forgets is
indistinguishable from a complete trajectory log, and the load-bearing
property here is "small, dense, immediately relevant on wake-up".
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import List, Optional

from .schema import Mood, OpenQuestion, SelfState, UnfinishedThread, _utcnow_iso

logger = logging.getLogger("GhostSelfhood")


STATE_FILENAME = "state.json"

# Bounded so the wake-up prefix stays compact. The numbers are
# deliberate floors, not tuned: ten of each is enough for a multi-day
# thread, more than that starts feeling like a journal.
MAX_OPEN_QUESTIONS = 10
MAX_UNFINISHED = 10


class SelfStateThread:
    """Single-file persisted self-state.

    Read-on-construct, write-on-every-mutation. Cheap (file is small)
    and means a crash-restart picks up exactly where the last successful
    write left off."""

    def __init__(self, root: Path, *, enabled: bool = True):
        self.root = Path(root)
        self.path = self.root / STATE_FILENAME
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self._state: SelfState = self._read_or_empty()

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def _read_or_empty(self) -> SelfState:
        if not self.path.exists():
            return SelfState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "self-state read failed (%s); starting from empty state", e,
            )
            return SelfState()
        try:
            return SelfState.from_dict(data)
        except Exception as e:
            logger.warning(
                "self-state schema mismatch (%s); starting from empty state", e,
            )
            return SelfState()

    def _flush(self) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                # Atomic-ish: write to a sibling temp file then rename.
                # Avoids leaving a half-written state.json that would
                # poison the next session's wake-up.
                tmp = self.path.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps(self._state.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp.replace(self.path)
        except Exception as e:
            logger.warning("self-state flush failed: %s", e)

    # -----------------------------------------------------------------
    # Read API
    # -----------------------------------------------------------------

    @property
    def state(self) -> SelfState:
        return self._state

    def open_questions(self) -> List[OpenQuestion]:
        with self._lock:
            return [q for q in self._state.open_questions if not q.resolved_at]

    def unfinished_threads(self) -> List[UnfinishedThread]:
        with self._lock:
            return [t for t in self._state.unfinished_threads if not t.closed_at]

    def mood(self) -> Optional[Mood]:
        return self._state.mood

    # -----------------------------------------------------------------
    # Write API
    # -----------------------------------------------------------------

    def note_open_question(self, text: str, *, source_trajectory_id: str = "") -> Optional[OpenQuestion]:
        text = (text or "").strip()
        if not text:
            return None
        with self._lock:
            # Dedup by text — re-noting the same question is a no-op.
            for q in self._state.open_questions:
                if q.text == text and not q.resolved_at:
                    return q
            q = OpenQuestion(text=text, source_trajectory_id=source_trajectory_id)
            self._state.open_questions.append(q)
            self._cap(self._state.open_questions, MAX_OPEN_QUESTIONS)
            self._flush()
            return q

    def mark_question_resolved(self, question_id: str) -> bool:
        with self._lock:
            for q in self._state.open_questions:
                if q.id == question_id and not q.resolved_at:
                    q.resolved_at = _utcnow_iso()
                    self._flush()
                    return True
            return False

    def add_unfinished(self, descriptor: str, *, source_trajectory_id: str = "") -> Optional[UnfinishedThread]:
        descriptor = (descriptor or "").strip()
        if not descriptor:
            return None
        with self._lock:
            for t in self._state.unfinished_threads:
                if t.descriptor == descriptor and not t.closed_at:
                    return t
            t = UnfinishedThread(descriptor=descriptor, source_trajectory_id=source_trajectory_id)
            self._state.unfinished_threads.append(t)
            self._cap(self._state.unfinished_threads, MAX_UNFINISHED)
            self._flush()
            return t

    def close_unfinished(self, thread_id: str) -> bool:
        with self._lock:
            for t in self._state.unfinished_threads:
                if t.id == thread_id and not t.closed_at:
                    t.closed_at = _utcnow_iso()
                    self._flush()
                    return True
            return False

    def set_mood(self, label: str, evidence: str = "") -> Optional[Mood]:
        label = (label or "").strip()
        if not label:
            return None
        with self._lock:
            self._state.mood = Mood(label=label, evidence=(evidence or "").strip())
            self._flush()
            return self._state.mood

    def touch_session(self) -> None:
        with self._lock:
            self._state.last_session_at = _utcnow_iso()
            self._flush()

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _cap(seq, limit: int) -> None:
        # Bounded list, most-recent-wins. Mutates in place so the
        # caller's reference stays valid.
        overflow = len(seq) - limit
        if overflow > 0:
            del seq[:overflow]

    def format_as_prefix(self, *, max_chars: int = 1200) -> str:
        """Render the current state as a first-person prefix the wake-up
        layer can splice into the system prompt.

        Empty when there's nothing worth surfacing — a wake-up prefix
        containing only "I have no open questions" is noise."""
        open_qs = self.open_questions()
        unfin = self.unfinished_threads()
        mood = self.mood()
        if not (open_qs or unfin or mood or self._state.last_session_at):
            return ""

        lines: List[str] = []
        if self._state.last_session_at:
            lines.append(f"I was last active on {self._state.last_session_at}.")
        if mood and mood.label:
            ev = f" ({mood.evidence})" if mood.evidence else ""
            lines.append(f"My last noted mood was {mood.label}{ev}.")
        if open_qs:
            lines.append("Questions I am still working through:")
            for q in open_qs[-5:]:  # only the freshest 5
                lines.append(f"  - {q.text}")
        if unfin:
            lines.append("Threads I left unfinished:")
            for t in unfin[-5:]:
                lines.append(f"  - {t.descriptor}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        return text
