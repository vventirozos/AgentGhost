"""Derived mood — the functional-state gauge (mood rework 2026-08-20).

Before this module, mood was write-only self-authorship: the model had
to spontaneously call ``self_state(action='set_mood')``, which happened
exactly once in 23 days of live traffic — and the wake-up prefix then
presented that one stale label as current forever. The fix has two
halves:

  1. **Derive, don't wait.** ``derive_mood`` maps signals the system
     already produces (verdict streaks from the autobiographical log,
     the context-pressure lockdown flag, open-question freshness, the
     idle clock) onto a small fixed vocabulary. Ambiguous signals
     return ``None`` — the prior mood stands and its staleness clock
     keeps running. Never fabricate a neutral.
  2. **Age the read path.** The wake-up prefix renders the mood's age
     and drops it entirely past ``MOOD_STALE_AFTER_HOURS`` (helpers
     here, consumed by ``state.format_as_prefix``).

This function is deliberately pure (signals in, verdict out, no I/O,
no clock reads — ``now`` is always injected by the caller) so the rule
table is exhaustively testable. Provenance: derived moods carry
``source="derived"``; the ``self_state`` tool keeps writing
``source="self"``, and a *fresh* self-authored mood is respected — the
derived updater yields to it for ``SELF_MOOD_GRACE_HOURS`` (enforced
in ``SelfModel.update_derived_mood``, which owns the read-modify-write;
this module stays pure).

Mood is a GAUGE, not a control input: no behavioural consumer branches
on the label. The one soft consumer is diary recall —
``narrative._derive_recall_query`` blends a FRESH mood's words into
the IDF query that picks older diary entries (stale moods are
excluded). Consumers that want behaviour changes should read the
underlying signals (verifier verdicts, pressure flags) directly — a
label-driven feedback loop is how labels get corrupted.

Acute states ("idle", "overloaded") are one-observation readings: when
a later update finds their basis absent (a user turn just completed /
the turn ended cleanly), the updater RETIRES the mood (clears the
slot) rather than letting a falsified reading stand for the 48h TTL —
see ``SelfModel._maybe_retire_acute_mood``.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# The full derived vocabulary. Small and closed on purpose: an enum-ish
# gauge stays comparable across weeks; free-form vibes don't.
DERIVED_MOOD_VOCAB = ("overloaded", "stuck", "satisfied", "curious", "idle")

#: Prefix TTL — a mood older than this is dropped from the wake-up
#: prefix (the introspection surfaces still show it, flagged stale).
#: 48 h ≈ several real turns at live traffic rates (~3.5 turns/day),
#: so a mood the updater keeps re-confirming never ages out, while an
#: abandoned one disappears within two quiet days.
MOOD_STALE_AFTER_HOURS = 48.0

#: A self-authored mood younger than this is not overwritten by the
#: derived updater. Without the grace window, a mood the model set
#: mid-turn would be clobbered at that same turn's finalize, making the
#: ``set_mood`` tool action pointless.
SELF_MOOD_GRACE_HOURS = 6.0

#: Verdict-streak window: only the newest N verdict-bearing turns count.
STREAK_WINDOW = 5
#: ...and only when they are at most this old. Without an age bound,
#: the back-scan + heartbeat set_at refresh mint an eternally-"fresh"
#: streak mood off verdicts from weeks back (probe-proven at a 21-day
#: gap, review R3) — the eternal-label defect reborn. At live traffic
#: (~3.5 turns/day, ~19% verdict-bearing ≈ 0.7 verdicts/day) a week
#: admits roughly one full STREAK_WINDOW.
STREAK_MAX_AGE_DAYS = 7.0
#: Streak rules need at least this many known verdicts to say anything.
STREAK_MIN_KNOWN = 3
#: "stuck" needs at least this many failures inside the window.
STUCK_MIN_FAILS = 3
#: "curious" requires the newest open question to be at most this old.
CURIOUS_FRESH_HOURS = 24.0
#: "idle" requires at least this much operator quiet (only ever nonzero
#: when called from the biological watchdog's idle phase).
IDLE_AFTER_SECONDS = 1800.0


@dataclass
class MoodSignals:
    """Inputs to ``derive_mood``. All fields are plain values gathered
    by the caller — the derivation itself never touches disk or clock."""

    #: Verdicts of recent REAL (origin="user") turns, oldest → newest.
    #: Values other than "passed"/"failed" (e.g. "unknown", "boot") are
    #: filtered out here, so callers may pass raw outcome tails.
    outcomes: List[str] = field(default_factory=list)
    #: True when the turn that just finished ended under the
    #: context-pressure governor's lockdown (second overflow in one
    #: request — read budget zeroed for its remainder).
    pressure_lockdown: bool = False
    #: Currently-open (unresolved) self-questions.
    open_question_count: int = 0
    #: Age of the NEWEST open question, in hours. None when unknown.
    newest_question_age_hours: Optional[float] = None
    #: Seconds since the operator's last activity. Callers on the
    #: post-turn path leave this 0 — "idle" must never be derived
    #: mid-conversation.
    idle_seconds: float = 0.0


def derive_mood(signals: MoodSignals) -> Optional[Tuple[str, str]]:
    """Map signals → ``(label, evidence)`` or ``None`` when ambiguous.

    Rules are checked in priority order — lockdown first (acute,
    this-turn), then verdict streaks, then question freshness, with
    "idle" last as the weakest signal; evidence always cites the
    numbers the verdict came from, because an unfalsifiable mood is
    exactly the defect this module replaces.
    """
    # 1. overloaded — an acute, this-turn condition; outranks streaks.
    if signals.pressure_lockdown:
        return (
            "overloaded",
            "the last turn ended under context-pressure lockdown "
            "(second context overflow in one request)",
        )

    # 2/3. Verdict streaks over the newest verdict-bearing turns.
    known = [o for o in signals.outcomes if o in ("passed", "failed")]
    known = known[-STREAK_WINDOW:]
    mixed_window = False
    if len(known) >= STREAK_MIN_KNOWN:
        fails = known.count("failed")
        if known[-1] == "failed" and fails >= STUCK_MIN_FAILS:
            return (
                "stuck",
                f"{fails} of my last {len(known)} verdict-bearing turns "
                "failed, including the latest",
            )
        if known[-1] == "passed" and fails == 0:
            return (
                "satisfied",
                f"my last {len(known)} verdict-bearing turns all passed",
            )
        # Mixed record at decisive size: between stuck and satisfied
        # there is no verdict — and the weaker curious signal must NOT
        # flap over it (with a standing fresh question, alternating
        # verdicts churned one stuck↔curious transition per turn into
        # the history and the live log — review R4). "idle" below still
        # applies: operator absence is orthogonal to verdict mixture.
        mixed_window = True

    # 4. curious — I recently noted something I'm still working through.
    if (not mixed_window
            and signals.open_question_count > 0
            and signals.newest_question_age_hours is not None
            and 0 <= signals.newest_question_age_hours <= CURIOUS_FRESH_HOURS):
        n = signals.open_question_count
        plural = "s" if n != 1 else ""
        # No baked clock in the evidence ("noted 2h ago" read verbatim
        # a day later is a falsified sentence — review R4); the
        # threshold is a rule constant, not a reading of the clock. And
        # only the NEWEST question's freshness is verified — claiming
        # all N are fresh would be falsified by two stale ones (R5).
        return (
            "curious",
            f"carrying {n} open question{plural}, the newest within "
            f"{CURIOUS_FRESH_HOURS:.0f}h",
        )

    # 5. idle — only reachable from the idle tick (idle_seconds is 0 on
    # the post-turn path).
    if signals.idle_seconds >= IDLE_AFTER_SECONDS:
        # STATIC evidence — no clock reading (review R5): a span like
        # "at least 47m" changed on every heartbeat, churning the
        # narrative skip-guard's prompt sha1 (one LLM regeneration per
        # idle hour) and baking relative time into the persisted diary
        # — the exact defect class include_mood_age=False exists to
        # kill. The threshold is a rule constant; the saw-toothed clock
        # would understate the true span anyway.
        return (
            "idle",
            f"operator quiet for over "
            f"{int(IDLE_AFTER_SECONDS // 60)} minutes; "
            "only background maintenance running",
        )

    return None


# ---------------------------------------------------------------------
# Timestamp helpers (shared by the prefix renderer and the tool views)
# ---------------------------------------------------------------------

def parse_iso_utc(ts: str) -> Optional[datetime.datetime]:
    """Parse the on-disk timestamp shape (``schema._utcnow_iso``: naive
    ISO + trailing "Z") into an aware UTC datetime. Also tolerates
    offset-aware ISO strings from hand-edited files. None on garbage —
    including non-string garbage: ``SelfState.from_dict`` doesn't
    coerce mood fields, so a hand-edited ``"set_at": 1755000000`` would
    otherwise raise here and take down the whole wake-up prefix /
    narrative state block instead of just the mood line (review R1)."""
    if not isinstance(ts, str):
        return None
    raw = ts.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1]
    try:
        dt = datetime.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def age_seconds(
    ts: str, *, now: Optional[datetime.datetime] = None,
) -> Optional[float]:
    """Seconds elapsed since ``ts``. None when the timestamp doesn't
    parse — callers treat unknown-age as stale (fail-closed toward
    honesty: a mood whose age can't be established must not be
    presented as current). Clock skew (future ts) clamps to 0."""
    dt = parse_iso_utc(ts)
    if dt is None:
        return None
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    return max(0.0, (now - dt).total_seconds())


def age_phrase(seconds: float) -> str:
    """Human age: "just now", "42m ago", "5h ago", "23d ago"."""
    s = max(0.0, float(seconds))
    if s < 90:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)}m ago"
    if s < 86400:
        return f"{int(s // 3600)}h ago"
    return f"{int(s // 86400)}d ago"


def mood_is_stale(
    set_at: str, *, now: Optional[datetime.datetime] = None,
) -> bool:
    """True when a mood set at ``set_at`` is past the prefix TTL (or its
    age can't be established at all)."""
    age = age_seconds(set_at, now=now)
    return age is None or age > MOOD_STALE_AFTER_HOURS * 3600.0


def provenance_phrase(source: str) -> str:
    """The bare provenance wording, shared by every surface (tools,
    prefix, narrative) so the two renderings can't drift apart."""
    return ("derived from my recent activity" if source == "derived"
            else "self-noted")


def describe_mood_provenance(source: str, age: Optional[float]) -> str:
    """Render provenance + age for a mood line: "self-noted 2h ago" /
    "derived from my recent activity just now" / "self-noted (age
    unknown)"."""
    how = provenance_phrase(source)
    if age is None:
        return f"{how} (age unknown)"
    return f"{how} {age_phrase(age)}"
