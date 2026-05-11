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
    return datetime.datetime.utcnow().isoformat() + "Z"


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experience":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


@dataclass
class OpenQuestion:
    """Something the agent (or the user, via the agent) is still
    trying to figure out across sessions."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    text: str = ""
    source_trajectory_id: str = ""
    opened_at: str = field(default_factory=_utcnow_iso)
    resolved_at: str = ""  # empty when still open


@dataclass
class UnfinishedThread:
    """A task / topic the agent left mid-flight in a prior session."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    descriptor: str = ""           # short label
    source_trajectory_id: str = ""
    opened_at: str = field(default_factory=_utcnow_iso)
    closed_at: str = ""


@dataclass
class Mood:
    """Latest qualitative state. Bounded by recency, not by sentiment
    analysis — the agent reports its own functional state."""

    label: str = ""                # short tag (e.g. "curious", "stuck", "satisfied")
    evidence: str = ""             # one sentence why
    set_at: str = field(default_factory=_utcnow_iso)


@dataclass
class SelfState:
    """The cross-session state thread. Persisted as one JSON file."""

    schema_version: str = SCHEMA_VERSION
    open_questions: List[OpenQuestion] = field(default_factory=list)
    unfinished_threads: List[UnfinishedThread] = field(default_factory=list)
    mood: Optional[Mood] = None
    last_session_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.mood is None:
            d["mood"] = None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SelfState":
        oq = [OpenQuestion(**q) for q in (d.get("open_questions") or [])]
        ut = [UnfinishedThread(**t) for t in (d.get("unfinished_threads") or [])]
        mood_raw = d.get("mood")
        mood = Mood(**mood_raw) if isinstance(mood_raw, dict) else None
        return cls(
            schema_version=str(d.get("schema_version") or SCHEMA_VERSION),
            open_questions=oq,
            unfinished_threads=ut,
            mood=mood,
            last_session_at=str(d.get("last_session_at") or ""),
        )
