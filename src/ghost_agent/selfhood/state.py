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

import datetime
import json
import logging
import threading
from pathlib import Path
from typing import List, Optional

from .mood import (
    age_seconds,
    describe_mood_provenance,
    mood_is_stale,
    provenance_phrase,
)
from .schema import Mood, OpenQuestion, SelfState, UnfinishedThread, _utcnow_iso

logger = logging.getLogger("GhostSelfhood")


STATE_FILENAME = "state.json"
MOOD_HISTORY_FILENAME = "mood.history.jsonl"

# Bounded so the wake-up prefix stays compact. The numbers are
# deliberate floors, not tuned: ten of each is enough for a multi-day
# thread, more than that starts feeling like a journal.
MAX_OPEN_QUESTIONS = 10
MAX_UNFINISHED = 10

# Bounded growth for the mood audit trail (mirrors autobiographical's
# _COMPACT_MAX_BYTES pattern). The file is append-only and
# ``mood_history()`` does a full read on every narrative regeneration,
# so an uncapped file grows both disk and read cost without bound.
_MOOD_COMPACT_MAX_BYTES = 512 * 1024
_MOOD_COMPACT_KEEP_LINES = 1000

# Input caps for set_mood (review R2): keep one mood record small
# relative to both the 1200-char prefix budget and the history
# compaction's line-count cap.
_MOOD_LABEL_MAX_CHARS = 64
_MOOD_EVIDENCE_MAX_CHARS = 300


def _cap_text(s: str, cap: int) -> str:
    """Strip + cap with an ellipsis when truncated (total length ≤ cap)."""
    s = (s or "").strip()
    if len(s) <= cap:
        return s
    return s[: cap - 1].rstrip() + "…"


class SelfStateThread:
    """Single-file persisted self-state.

    Read-on-construct, write-on-every-mutation. Cheap (file is small)
    and means a crash-restart picks up exactly where the last successful
    write left off."""

    def __init__(self, root: Path, *, enabled: bool = True):
        self.root = Path(root)
        self.path = self.root / STATE_FILENAME
        self.mood_history_path = self.root / MOOD_HISTORY_FILENAME
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

    @property
    def lock(self) -> threading.RLock:
        """Public handle on the state RLock, for callers that need a
        read-modify-write to be atomic against concurrent mutators
        (e.g. SelfModel.update_derived_mood's grace-window check).
        Re-entrant, so holding it across a set_mood/clear_mood call is
        safe. Exposed deliberately: reaching into ``_lock`` from
        another module would turn a future rename into a silent no-op
        inside that caller's blanket exception guard."""
        return self._lock

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

    def set_mood(
        self, label: str, evidence: str = "", *, source: str = "self",
    ) -> Optional[Mood]:
        # Bounded inputs (review R2): the tool path caps nothing, and a
        # single oversized evidence line would defeat the history
        # compaction's LINE cap (it can never bring the file under the
        # byte cap → full rewrite on every transition forever) and
        # crowd everything else out of the 1200-char prefix. Truncation
        # carries an ellipsis so a cut never reads as a complete
        # sentence (derived writers stay well under the caps; only
        # tool-authored text can hit them).
        label = _cap_text(label, _MOOD_LABEL_MAX_CHARS)
        if not label:
            return None
        evidence = _cap_text(evidence, _MOOD_EVIDENCE_MAX_CHARS)
        # Constrain provenance to the two known values so a caller typo
        # can't mint a third population in the history file.
        source = "derived" if source == "derived" else "self"
        with self._lock:
            prior = self._state.mood
            if (prior is not None and prior.label == label
                    and getattr(prior, "source", "self") == source):
                # Same functional state re-confirmed: refresh the
                # staleness clock (and evidence, when fresh evidence is
                # supplied) WITHOUT a history append — the history
                # records transitions, not heartbeats, so the "mood arc"
                # stays an arc and the file doesn't fill with identical
                # lines at one-per-turn rate. Swap in a NEW Mood rather
                # than mutating the prior one field-by-field: readers
                # (mood(), format_as_prefix) don't take the lock, and a
                # single reference assignment can't be observed torn.
                refreshed = Mood(
                    label=label,
                    evidence=evidence or prior.evidence,
                    source=source,
                )
                self._state.mood = refreshed
                self._flush()
                return refreshed
            mood = Mood(label=label, evidence=evidence, source=source)
            self._state.mood = mood
            self._flush()
            # Mood history — append-only audit trail so the narrative
            # can describe arcs ("I shifted from stuck to satisfied")
            # rather than just the latest slot. Failures are swallowed
            # — the latest mood is already persisted in state.json.
            try:
                self.mood_history_path.parent.mkdir(parents=True, exist_ok=True)
                rec = {
                    "label": mood.label,
                    "evidence": mood.evidence,
                    "set_at": mood.set_at,
                    "source": mood.source,
                }
                with self.mood_history_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False))
                    f.write("\n")
                self._maybe_compact_mood_history_locked()
            except Exception as e:
                logger.warning("mood history append failed: %s", e)
            return mood

    def _maybe_compact_mood_history_locked(self) -> None:
        """Rewrite the mood history keeping the newest
        ``_MOOD_COMPACT_KEEP_LINES`` records once it passes the byte cap.
        Caller holds ``self._lock``. Best-effort — a failed compaction
        must never fail the mood write that triggered it."""
        try:
            if self.mood_history_path.stat().st_size <= _MOOD_COMPACT_MAX_BYTES:
                return
            from collections import deque
            with self.mood_history_path.open("r", encoding="utf-8") as f:
                tail = deque(f, maxlen=_MOOD_COMPACT_KEEP_LINES)
            tmp = self.mood_history_path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                f.writelines(tail)
            tmp.replace(self.mood_history_path)
            logger.info(
                "mood history compacted to newest %d records", len(tail),
            )
        except Exception as e:
            logger.warning("mood history compaction failed: %s", e)

    def clear_mood(self) -> bool:
        """Retire the current mood slot (review R2: acute derived states
        whose basis has been falsified — "idle" while the operator is
        actively chatting, "overloaded" after a clean turn — must not
        stand as current for the full 48h TTL). Clears the latest slot
        only; the history keeps the state the mood transitioned INTO
        (the arc records states held, and a retirement is the absence
        of a state, not a new one). Returns True when something was
        cleared."""
        with self._lock:
            if self._state.mood is None:
                return False
            self._state.mood = None
            self._flush()
            return True

    def mood_history(self, *, limit: int = 20) -> List[Mood]:
        """Tail of the mood history, oldest first within the returned
        window. Returns empty list when the file is missing or limit
        <= 0."""
        if limit <= 0 or not self.mood_history_path.exists():
            return []
        try:
            lines = self.mood_history_path.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            logger.warning("mood history read failed: %s", e)
            return []
        out: List[Mood] = []
        for line in lines[-limit:]:
            s = line.strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                continue
            out.append(Mood(
                label=str(d.get("label") or ""),
                evidence=str(d.get("evidence") or ""),
                set_at=str(d.get("set_at") or ""),
                # Lines predating the provenance field were tool-authored.
                source=str(d.get("source") or "self"),
            ))
        return out

    def stale_open_questions(self, *, max_age_days: float = 3.0) -> List[OpenQuestion]:
        """Return open questions whose ``opened_at`` is older than
        ``max_age_days``. Used by an idle gardener hook so the
        open-question list stays alive — a question that sits for a
        week with no engagement should either be re-engaged or
        resolved, not silently accumulated."""
        threshold = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=max(0.0, float(max_age_days))
        )
        out: List[OpenQuestion] = []
        with self._lock:
            for q in self._state.open_questions:
                if q.resolved_at:
                    continue
                try:
                    # Tolerate either "...Z" or full-iso timestamps.
                    raw = (q.opened_at or "").rstrip("Z")
                    opened = datetime.datetime.fromisoformat(raw).replace(
                        tzinfo=datetime.timezone.utc
                    )
                except Exception:
                    continue
                if opened < threshold:
                    out.append(q)
        return out

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
        # caller's reference stays valid. Resolved/closed entries are
        # evicted first — they count toward the cap but carry no live
        # value, and a blind head-first eviction could drop the only
        # still-open entry while retaining dead ones.
        overflow = len(seq) - limit
        if overflow <= 0:
            return
        i = 0
        while overflow > 0 and i < len(seq):
            item = seq[i]
            if getattr(item, "resolved_at", None) or getattr(item, "closed_at", None):
                del seq[i]
                overflow -= 1
            else:
                i += 1
        if overflow > 0:
            del seq[:overflow]

    def format_as_prefix(
        self,
        *,
        max_chars: int = 1200,
        now: Optional[datetime.datetime] = None,
        include_mood_age: bool = True,
    ) -> str:
        """Render the current state as a first-person prefix the wake-up
        layer can splice into the system prompt.

        Empty when there's nothing worth surfacing — a wake-up prefix
        containing only "I have no open questions" is noise.

        The mood line carries its provenance and age, and is dropped
        entirely once past ``mood.MOOD_STALE_AFTER_HOURS`` (or when its
        timestamp doesn't parse): a 23-day-old label presented as
        current is exactly how "curious" became the eternal answer.
        ``now`` is injectable for tests; None means wall-clock.

        ``include_mood_age=False`` renders the mood line as label +
        provenance ONLY — no age phrase and no evidence (staleness drop
        still applies) — for the narrative consumer, whose prompt sha1
        is the regenerate skip-guard: a wall-clock-varying "2h ago"
        (R2) or an evidence count draining as verdicts cross the 7-day
        age bound (R6) would defeat that guard and bake clock-derived
        text into a diary that persists for days."""
        open_qs = self.open_questions()
        unfin = self.unfinished_threads()
        mood = self.mood()
        if not (open_qs or unfin or mood or self._state.last_session_at):
            return ""

        lines: List[str] = []
        if self._state.last_session_at:
            lines.append(f"I was last active on {self._state.last_session_at}.")
        if mood and mood.label and not mood_is_stale(mood.set_at, now=now):
            if include_mood_age:
                prov = describe_mood_provenance(
                    getattr(mood, "source", "self"),
                    age_seconds(mood.set_at, now=now),
                )
                ev = f" ({mood.evidence})" if mood.evidence else ""
                lines.append(f"My mood, {prov}: {mood.label}{ev}.")
            else:
                # Narrative-facing rendering: label + provenance ONLY —
                # no age AND no evidence. Evidence strings legitimately
                # vary with the clock (streak counts drain as verdicts
                # cross the 7-day age bound with zero new events, R6),
                # and this block feeds the regenerate skip-guard's
                # sha1: stability here must hold by CONSTRUCTION, not
                # by policing every evidence template. New-verdict
                # regenerations are not lost — a new verdict changes
                # the experiences block of the same prompt anyway.
                # Shared wording helper so this rendering can't drift
                # from the tool/prefix surface (review R7).
                prov = provenance_phrase(getattr(mood, "source", "self"))
                lines.append(f"My mood, {prov}: {mood.label}.")
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
