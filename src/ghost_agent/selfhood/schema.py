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
    intensity. It lasts until the end of the day it was set on, stepping
    down on the way (§4TS, read-time: ``mood.manner_level``).

    §4TX: ``held_until`` — a mood the owner SET does not decay and is not
    moved by events until then (the next 04:00, like a manner); ``owner_at``
    — when the owner last set, reset or changed the manner (an event about a
    message older than that is not applied).

    §4UD: the part of the pair the owner's own message tone put there is
    FOUR non-negative numbers — ``private_up_p`` / ``private_up_e`` (what
    upbeat messages added) and ``private_dn_p`` / ``private_dn_e`` (what
    unhappy ones took off). They decay with the pair and are left out of
    the mood other people hear. Kept apart so one side can never be netted
    against the other (review F1: one signed number hid −0.18 of tone behind
    a +0.08 that was never in the pair). The pair without them is
    ``p − private_up_p + private_dn_p`` (same for ``e``): what the mood would
    be had no message tone ever been read. ``private_p`` / ``private_e`` are
    the NET, read-only; a file written before §4UD carries them as its only
    share and loads onto the side of their sign. ``tone_touched`` — a tone
    had a part in this pair since it was last neutral or set by the owner
    (then other people get a fresh reading of the level, never the held one).
    ``prev`` — what the owner's last set mood replaced (``set_mood`` undo).

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
    held_until: str = ""
    owner_at: str = ""
    private_up_p: float = 0.0
    private_up_e: float = 0.0
    private_dn_p: float = 0.0
    private_dn_e: float = 0.0
    tone_touched: bool = False
    prev: Dict[str, Any] = field(default_factory=dict)

    @property
    def private_p(self) -> float:
        """The owner's tone in the pleasantness, NET (up − down). Read-only."""
        return self.private_up_p - self.private_dn_p

    @property
    def private_e(self) -> float:
        return self.private_up_e - self.private_dn_e

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
        shares = cls._shares_from(d)
        # a share an older version stored OVER the cap is cut on its way in — with the pair, so that what the
        # tones hold in the pair is what the shares say (the cut part leaves both)
        over_p = clamp(d.get("private_p")) - (shares["private_up_p"] - shares["private_dn_p"]) \
            if "private_up_p" not in d else 0.0
        over_e = clamp(d.get("private_e")) - (shares["private_up_e"] - shares["private_dn_e"]) \
            if "private_up_e" not in d else 0.0
        return cls(
            p=clamp(clamp(d.get("p")) - over_p), e=clamp(clamp(d.get("e")) - over_e),
            updated_at=_text("updated_at", 40),
            label=label, intensity=intensity,
            novelty=d.get("novelty") is True,
            source=SOURCE_OWNER if d.get("source") == SOURCE_OWNER else SOURCE_AUTO,
            set_at=_text("set_at", 40), cause=_text("cause"),
            recent=recent,
            manner=manner,
            manner_intensity=(_level("manner_intensity") or 2) if manner else 0,
            manner_set_at=_text("manner_set_at", 40) if manner else "",
            held_until=_text("held_until", 40) if d.get("source") == SOURCE_OWNER else "",
            owner_at=_text("owner_at", 40),
            **shares,
            prev=cls._prev_from(d.get("prev")),
        )

    @staticmethod
    def _shares_from(d: Dict[str, Any]) -> Dict[str, Any]:
        """The four tone shares of a stored record, each in [0, its cap]. A
        record from before §4UD holds one signed number per axis: it goes to
        the side of its sign, so the cap that held it still does."""
        from .mood import SENTIMENT_SHARE_MAX, SENTIMENT_SHARE_MAX_E, clamp

        def _side(key: str, legacy: str, sign: float, cap: float) -> float:
            v = clamp(d.get(key)) if key in d else max(0.0, sign * clamp(d.get(legacy)))
            return max(0.0, min(cap, v))
        out = dict(
            private_up_p=_side("private_up_p", "private_p", 1.0, SENTIMENT_SHARE_MAX),
            private_up_e=_side("private_up_e", "private_e", 1.0, SENTIMENT_SHARE_MAX_E),
            private_dn_p=_side("private_dn_p", "private_p", -1.0, SENTIMENT_SHARE_MAX),
            private_dn_e=_side("private_dn_e", "private_e", -1.0, SENTIMENT_SHARE_MAX_E),
        )
        touched = d.get("tone_touched")
        out["tone_touched"] = (touched is True) if "tone_touched" in d else any(out.values())
        return out

    @staticmethod
    def _prev_from(v: Any) -> Dict[str, Any]:
        """What the owner's last set mood replaced, or {} (anything else)."""
        from .mood import LABELS, clamp
        if not isinstance(v, dict) or not isinstance(v.get("at"), str) or not v["at"].strip():
            return {}
        prev: Dict[str, Any] = {"p": clamp(v.get("p")), "e": clamp(v.get("e")), "at": v["at"][:40]}
        n = v.get("intensity")
        # the label and level of what was replaced are kept only for a mood the owner had set and still held
        if (v.get("owner") is True and v.get("label") in LABELS and isinstance(n, int) and not isinstance(n, bool)
                and 1 <= n <= 3 and isinstance(v.get("held_until"), str) and v["held_until"].strip()):
            prev.update(label=v["label"], intensity=n, owner=True, held_until=v["held_until"][:40])
        return prev


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
    #: §4UD (F9): when a background job the OWNER started last did work — the
    #: quiet is measured from the later of the two (the face read "bored"
    #: while the agent worked for the owner all day)
    last_owner_job_at: str = ""

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
            last_owner_job_at=(d.get("last_owner_job_at")
                               if isinstance(d.get("last_owner_job_at"), str) else ""),
        )


def read_jsonl_rows(path, *, what: str = "") -> List[Dict[str, Any]]:
    """§4UD (F5): THE reader of the selfhood jsonl files (the character's log,
    the sentiment journal, the mood journals): every line that is a JSON
    object, oldest first. A byte that is not UTF-8 costs its own line, never
    the file (it is replaced, and the line then parses or is skipped); a
    torn or hand-edited line is skipped. The number of skipped lines is
    logged once per file state. [] when the file is absent or unreadable.
    Never raises."""
    import logging
    from pathlib import Path
    p = Path(path)
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        return []
    except OSError as e:
        logging.getLogger("GhostSelfhood").warning("%s unreadable: %s", what or p.name, e)
        return []
    out: List[Dict[str, Any]] = []
    bad = 0
    for line in raw.decode("utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except (ValueError, RecursionError):
            bad += 1
            continue
        if isinstance(row, dict):
            out.append(row)
        else:
            bad += 1
    if bad:
        key = (str(p), len(raw), bad)
        if key not in _BAD_LINES_LOGGED:
            if len(_BAD_LINES_LOGGED) > 64:
                _BAD_LINES_LOGGED.clear()
            _BAD_LINES_LOGGED.add(key)
            logging.getLogger("GhostSelfhood").warning(
                "%s: %d line(s) could not be read and were skipped (%d kept)", what or p.name, bad, len(out))
    return out


_BAD_LINES_LOGGED: set = set()
