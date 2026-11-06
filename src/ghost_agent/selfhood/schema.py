"""Dataclasses for the selfhood module.

The five components of the proposed "unified self" all share these
records. Keeping schema flat and JSON-serialisable (no pydantic) so
the on-disk format diffs cleanly and migrations are explicit.

Naming convention: every record that represents the agent's own
first-person experience carries ``subject="self"``. The recognition
layer keys off that tag to treat retrieved records as autobiographical
("I did this") rather than external knowledge ("the system did this").
"""

from __future__ import annotations

import datetime
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "v1"


def _utcnow_iso() -> str:
    # datetime.utcnow() is deprecated in 3.12 and slated for removal.
    # We keep the trailing "Z" so existing on-disk records stay
    # parse-compatible with the new ones.
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(tzinfo=None)
        .isoformat()
        + "Z"
    )


@dataclass
class Experience:
    """One first-person experiential record. Cross-referenced to the
    trajectory it summarises by sharing the trajectory's id."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trajectory_id: str = ""
    timestamp: str = field(default_factory=_utcnow_iso)
    subject: str = "self"  # the tag the recognition layer keys off
    summary: str = ""      # first-person prose
    user_handle: str = ""  # who I was talking to (best-effort)
    user_first_words: str = ""  # short prefix of the user's request, helps recall
    tools_used: List[str] = field(default_factory=list)
    outcome: str = "unknown"  # passed | failed | unknown
    cluster: Optional[str] = None
    # What I actually ANSWERED — a short, redacted, single-line excerpt of
    # the final reply (2026-09-05). The template summary records how the
    # turn went ("reached for X and the answer landed"), never what was
    # said, so recall could surface the question and never the answer:
    # live, "Kosmas age" returned five records of the question and none
    # of the birth date. Empty on rows written before the field existed
    # (`from_dict` fills the default) and on turns with no reply text.
    answer_gist: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experience":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class Mood:
    """The stored mood (§4TC): two numbers and what was last read off them.

    ``p`` (pleasantness) and ``e`` (energy) are in [-1, 1] and were written
    at ``updated_at``; they decay from there (``selfhood/mood.py``). ``label``
    and ``intensity`` are what was last read off the pair — kept so a label
    or an intensity is only given up clearly past a boundary — with when
    (``set_at``), by whom (``source``: "auto" | "owner") and why (``cause``).
    ``novelty`` is the tag that makes the P+ E+ region "curious". ``recent``
    holds the last few events that moved the pair (``{"at","cause","kind"}``).

    The MANNER is separate: a way of talking the owner set, with its own
    intensity. It does not decay.

    Before §4TC a mood was ``{"label", "evidence", "set_at", "source":
    "self"|"derived"}``. Such a record has no pair: ``SelfState.from_dict``
    ignores it and the mood starts neutral."""

    p: float = 0.0
    e: float = 0.0
    updated_at: str = ""
    label: str = "neutral"
    intensity: int = 0
    novelty: bool = False
    source: str = "auto"
    set_at: str = ""
    cause: str = ""
    recent: List[Dict[str, Any]] = field(default_factory=list)
    manner: str = ""
    manner_intensity: int = 0
    manner_set_at: str = ""

    @classmethod
    def from_dict(cls, d: Any) -> Optional["Mood"]:
        """A stored mood, or None when ``d`` is not one (an old label-only
        record, garbage). Every field is coerced: a hand-edited file costs
        the mood, never the state."""
        if not isinstance(d, dict) or "p" not in d or "e" not in d:
            return None
        from .mood import LABELS, NEUTRAL, SOURCE_AUTO, SOURCE_OWNER, clamp, resolve_manner

        def _text(key: str, cap: int = 400) -> str:
            v = d.get(key)
            return v[:cap] if isinstance(v, str) else ""

        def _level(key: str) -> int:
            v = d.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return 0
            try:
                return max(0, min(3, int(v)))
            except (TypeError, ValueError, OverflowError):
                return 0

        label = _text("label", 32)
        intensity = _level("intensity")
        if label not in LABELS or intensity == 0:
            label, intensity = NEUTRAL, 0
        recent = [{"at": str(r.get("at") or "")[:40], "cause": str(r.get("cause") or "")[:400],
                   "kind": str(r.get("kind") or "")[:24]}
                  for r in (d.get("recent") if isinstance(d.get("recent"), list) else [])
                  if isinstance(r, dict)][-MOOD_RECENT_CAUSES:]
        manner = resolve_manner(d.get("manner")) if isinstance(d.get("manner"), str) else ""
        return cls(
            p=clamp(d.get("p")), e=clamp(d.get("e")),
            updated_at=_text("updated_at", 40),
            label=label, intensity=intensity,
            novelty=d.get("novelty") is True,
            source=SOURCE_OWNER if d.get("source") == SOURCE_OWNER else SOURCE_AUTO,
            set_at=_text("set_at", 40), cause=_text("cause"),
            recent=recent,
            manner=manner,
            manner_intensity=(_level("manner_intensity") or 2) if manner else 0,
            manner_set_at=_text("manner_set_at", 40) if manner else "",
        )


#: how many of the latest events ``Mood.recent`` keeps
MOOD_RECENT_CAUSES = 5


@dataclass
class SelfState:
    """The cross-session state thread. Persisted as one JSON file.

    §4TB (2026-11-05): `open_questions` and `unfinished_threads` are no
    longer fields. A file written before that carries both keys; they are
    ignored on read and absent from the next write."""

    schema_version: str = SCHEMA_VERSION
    mood: Optional[Mood] = None
    last_session_at: str = ""
    #: §4TC (review): when the OWNER was last here — stamped when an owner's
    #: request arrives and when it ends, through the owner-only gate. The
    #: length of a quiet is measured from it, so a restart does not change
    #: it. Distinct from `last_session_at`, which every captured turn stamps.
    last_owner_turn_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.mood is None:
            d["mood"] = None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SelfState":
        # Tolerant by construction: unknown keys are ignored (a pre-§4TB
        # file with its two removed lists loads), and a mood that is not the
        # §4TC shape — the old label record, a hand-edited value — is no
        # mood (`Mood.from_dict`), never a TypeError that `_read_or_empty`
        # would swallow into an empty state the next write then puts over
        # the good file.
        return cls(
            schema_version=str(d.get("schema_version") or SCHEMA_VERSION),
            mood=Mood.from_dict(d.get("mood")),
            last_session_at=str(d.get("last_session_at") or ""),
            last_owner_turn_at=(d.get("last_owner_turn_at")
                                if isinstance(d.get("last_owner_turn_at"), str) else ""),
        )
