"""Temporal anchoring — store the invariant, derive the quantity.

A stated age is a MEASUREMENT, not a fact. "Leonidas is 4 months old" is
true only on the day it is said; stored verbatim it is wrong a month later
and nothing downstream can tell. The profile block carries no timestamp at
all (``ProfileMemory.get_context_string`` renders bare ``- key: value``
lines into ``### USER PROFILE ###`` on every turn), so a snapshot reads to
the model as PRESENT TENSE for ever.

Observed live: the fact was stated 2026-07-07 and was still being recalled
verbatim as "Leonidas is 4 months old" on 2026-09-04, ~2 months later.
The vector store does stamp its rows and the prompt does carry CURRENT
TIME, so the arithmetic was *possible* — but it required the model to
notice an ISO stamp, subtract, and then overrule an unstamped profile line
that flatly asserted otherwise. It never did.

The invariant behind an age is a BIRTH DATE, which is constant. This
module converts one into the other:

    anchor("Leonidas is 4 months old", said_at=2026-07-07)
        -> "Leonidas born ~2026-02-20"
    derive("Leonidas born ~2026-02-20", now=2026-09-04)
        -> "Leonidas born ~2026-02-20 -> ~6 months old"

``anchor()`` runs at WRITE time, ``derive()`` at RENDER time. What is on
disk is therefore always a constant, and what the model reads is always
current — the model is never asked to do the subtraction, which is the
step it demonstrably skips.

DESIGN NOTES

* Midpoint anchoring. "9 years old" means the true age is somewhere in
  [9, 10), so the best point estimate of the birth date is ``said_at``
  minus 9.5 years — not minus 9 years. This matters: naive subtraction
  systematically biases every derived age one unit too HIGH, which is
  exactly the class of error the module exists to remove. The residual
  uncertainty (±6 months for a year-stated age) is why every derived age
  is rendered with a ``~``.

* Idempotence. ``anchor()`` output contains no age phrase, so re-running
  it is a no-op — required, because the value passes through more than one
  writer (the update_profile tool composes text/triplet/profile payloads
  from it, and ``ProfileMemory.update`` anchors again at the boundary).
  ``derive()`` strips any pre-existing gloss before adding one, so a
  derived string that leaks back into a store cannot compound.

* KV-cache stability. The derived gloss is a rounded age ("~6 months
  old"), NOT a date or a day count, so the rendered profile block changes
  only when the derived age changes — roughly monthly for an infant,
  yearly for a child. ``{{PROFILE}}`` is inside the byte-stable system
  prefix that ``core/agent.py`` deliberately keeps identical across turns
  for upstream prefix-cache hits; rendering "today 2026-09-04" here would
  have invalidated that prefix every single day.

* Conservative matching. An age rewrite requires an explicit age CUE
  ("old", "age", "aged", "yo", "mo"). A bare "4 months" is left alone — it
  is far more often a duration than an age, and a wrong rewrite is worse
  than a miss.

* "born" as the marker verb reads oddly for a non-person ("the project
  born ~2026-02-20"). Accepted deliberately: profile facts carrying ages
  are overwhelmingly about people and pets, and the marker's legibility to
  both a regex and an LLM is worth more than the rare awkward phrasing.
"""

from __future__ import annotations

import datetime
import re
from typing import Optional

__all__ = ["anchor", "derive", "signature", "has_anchor"]


# ── Anchor surface form ─────────────────────────────────────────────────
# ``born ~YYYY-MM-DD`` / ``born ~YYYY-MM`` / ``born ~YYYY``. The optional
# ``on``/``in`` filler lets derive() also gloss a birth date the user
# stated directly ("Leonidas was born on 2026-03-07"), which is the same
# fact class and needs the same arithmetic.
_ANCHOR_RE = re.compile(
    r"\bborn\s+(?:on\s+|in\s+)?(~?)(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?",
    re.IGNORECASE,
)

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}
_MONTH_ALT = "|".join(list(_MONTHS) + [m[:3] for m in _MONTHS])

# The month-NAME form a person actually types: "born March 12, 2026",
# "born on 12 March 2026", "born in March 2026". anchor() normalises this
# to ISO at write time, but the form still has to be READ: the live
# profile already held two of these (the operator stated both children's
# exact birth dates in a chat turn), written by the pre-change code and
# therefore never normalised. Without this they carry no derived age at
# all — and an exact date the model has to subtract from by hand is the
# original defect, just one step further along.
_ANCHOR_TEXT_RE = re.compile(
    r"\bborn\s+(?:on\s+|in\s+)?(?:"
    r"(?P<m1>" + _MONTH_ALT + r")\.?\s+(?P<d1>\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(?P<y1>\d{4})"
    r"|(?P<d2>\d{1,2})(?:st|nd|rd|th)?\s+(?P<m2>" + _MONTH_ALT + r")\.?\s*,?\s*(?P<y2>\d{4})"
    r"|(?P<m3>" + _MONTH_ALT + r")\.?\s+(?P<y3>\d{4})"
    r")",
    re.IGNORECASE,
)

# A previously-derived gloss, stripped before a fresh one is appended so
# derive() is idempotent and self-correcting.
_GLOSS_RE = re.compile(
    r"\s*(?:→|->)\s*~?\d+\s+(?:year|month|week|day)s?\s+old",
    re.IGNORECASE,
)

# An absolutised "N units ago" anchor, recognised by signature() only.
_AGO_ANCHOR_RE = re.compile(r"\bin\s+~\d{4}-\d{2}-\d{2}", re.IGNORECASE)

_UNIT_ALIASES = {
    "year": "year", "years": "year", "yr": "year", "yrs": "year",
    "y": "year", "yo": "year", "y/o": "year",
    "month": "month", "months": "month", "mo": "month", "mos": "month",
    "week": "week", "weeks": "week", "wk": "week", "wks": "week",
    "day": "day", "days": "day",
}

# Ages beyond this are almost certainly a misparse (a year number, a count
# of items), not a person's age. Left untouched rather than rewritten.
_MAX_PLAUSIBLE_AGE = {"year": 130, "month": 1560, "week": 6800, "day": 47500}


def _norm_unit(raw: str) -> Optional[str]:
    return _UNIT_ALIASES.get(str(raw or "").strip().lower().rstrip("."))


def _shift_months(d: datetime.date, months: int) -> datetime.date:
    """``d`` shifted by ``months`` (may be negative), clamping the day to
    the target month's length so 31 Mar - 1 month is 28/29 Feb rather than
    a ValueError."""
    total = (d.year * 12 + (d.month - 1)) + months
    year, month = divmod(total, 12)
    month += 1
    # Days in the target month, without importing calendar for one value.
    if month == 12:
        last = 31
    else:
        last = (datetime.date(year + (month // 12), (month % 12) + 1, 1)
                - datetime.timedelta(days=1)).day
    return datetime.date(year, month, min(d.day, last))


def _months_between(start: datetime.date, end: datetime.date) -> int:
    """Whole months elapsed from ``start`` to ``end`` (negative if end is
    before start)."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return months


def _as_date(value) -> Optional[datetime.date]:
    """Coerce date / datetime / ISO-ish string to a ``date``; None on
    anything unparseable — callers must degrade to a no-op, never guess."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str):
        try:
            from ..utils.helpers import parse_utc_timestamp
            return parse_utc_timestamp(value).date()
        except Exception:
            pass
        try:
            return datetime.date.fromisoformat(value.strip()[:10])
        except Exception:
            return None
    return None


def _today() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).date()


def _birth_from_age(said_at: datetime.date, count: int, unit: str) -> datetime.date:
    """Midpoint estimate of the birth date behind an age stated at
    ``said_at``. See the DESIGN NOTES on why the half-unit offset is not
    optional."""
    if unit == "year":
        return _shift_months(said_at, -(count * 12 + 6))
    if unit == "month":
        return _shift_months(said_at, -count) - datetime.timedelta(days=15)
    if unit == "week":
        return said_at - datetime.timedelta(days=int(round(count * 7 + 3.5)))
    return said_at - datetime.timedelta(days=count)


def _fmt_anchor(d: datetime.date, unit: str) -> str:
    """Format an anchor at the GRANULARITY the statement supports.

    An age given in years is only known to ±6 months, so it anchors to the
    month (``born ~2017-01``); an age given in months or finer anchors to
    the day (``born ~2026-02-20``). The granularity is not cosmetic — it is
    how derive() knows not to render "1 year old" back as "~18 months
    old", which would present the midpoint estimate as if it were a
    measurement. Precision has to survive the round trip or the anchor is
    lying about what is known.
    """
    if unit == "year":
        return f"born ~{d.year:04d}-{d.month:02d}"
    return f"born ~{d.isoformat()}"


# ── Write-time: age / relative phrases -> absolute anchors ──────────────
#
# Each pattern captures (count, unit-or-None). A leading copula is swept
# into the match and dropped, so "Leonidas is 4 months old" becomes
# "Leonidas born ~2026-02-20" rather than "Leonidas is born ~…".
_COPULA = r"(?:\b(?:is|was|are|were|turns|turned)\s+)?"

_UNIT_ALT = (r"year|years|yr|yrs|month|months|mo|mos|"
             r"week|weeks|wk|wks|day|days")

# PREDICATIVE / appositive forms — the age stands on its own ("is 4 months
# old", "(9 years old)", ", aged 9,"). Here the phrase can be REPLACED by
# the anchor and the sentence still reads.
#
# Note both spaced variants require whitespace on at least one side of the
# compound, which is what keeps the fully-hyphenated ATTRIBUTIVE form
# ("9-year-old son") out of this list — see _ATTRIB_AGE_RE below.
_AGE_PATTERNS = [
    # "9 years old", "9 years-old"
    (re.compile(_COPULA + r"\b(\d{1,3})\s+(" + _UNIT_ALT + r")\s*[-\s]\s*old\b",
                re.IGNORECASE), True),
    # "9-year old", "9 year old"
    (re.compile(_COPULA + r"\b(\d{1,3})\s*[-\s]\s*(" + _UNIT_ALT + r")\s+old\b",
                re.IGNORECASE), True),
    # "aged 4 months", "age 4 months"
    (re.compile(r"\bage[d]?\s+(\d{1,3})\s+(" + _UNIT_ALT + r")\b",
                re.IGNORECASE), True),
    # "9yo", "9 y/o"
    (re.compile(_COPULA + r"\b(\d{1,3})\s*(yo|y/o)\b", re.IGNORECASE), True),
    # "4mo" — the compact form an extractor emits ("Leonidas, 4mo")
    (re.compile(_COPULA + r"\b(\d{1,3})\s*(mo|mos)\b(?!\w)", re.IGNORECASE), True),
    # "age 9" / "aged 9" — a bare number after an age cue means YEARS
    (re.compile(r"\bage[d]?\s+(\d{1,3})\b(?!\s*[-\s]?\s*(?:year|month|week|day))",
                re.IGNORECASE), False),
]

# ATTRIBUTIVE form: a fully-hyphenated compound modifying (or standing in
# for) a noun — "his 9-year-old son Thodoris", "a 4-month-old named
# Leonidas". Replacing it in place produces garbage — "equipment for a born
# ~2017-02-20 named Thodoris" was the live result on a real stored fact —
# because the compound is filling an adjective/noun slot, not a predicate.
#
# So this one form ANNOTATES instead of replacing: "9-year-old (born
# ~2017-01-07)". The sentence stays grammatical, and derive() renders the
# authoritative current value immediately after the frozen one
# ("9-year-old (born ~2017-01-07 → ~10 years old)"). Keeping the stale
# token is the deliberate trade: it is the only position where removing it
# costs grammaticality, and the derived value sits adjacent to it.
#
# The lookahead keeps this idempotent — an annotated compound is not
# re-annotated on a second pass.
_ATTRIB_AGE_RE = re.compile(
    r"\b(\d{1,3})-(" + _UNIT_ALT + r")-old\b(?!\s*\(?\s*born\b)",
    re.IGNORECASE)

# Verbatim spans — a quoted string is a RECORD (a search query the agent
# ran, a command, a code literal), not a claim about the user, and
# rewriting one falsifies the record. Live proof: a stored skill lesson
# held `web_search(query="Thodoris basketball Panellinios age 9")`, and
# without this guard the anchor pass rewrote the query text itself.
_QUOTED_RE = re.compile(r"\"[^\"\n]*\"|'[^'\n]*'|`[^`\n]*`")

# "3 years ago" -> an absolute date. Same defect class as an age: the
# phrase silently re-anchors to whenever it is next read.
_AGO_RE = re.compile(
    r"\b(\d{1,3})\s+(year|years|month|months|week|weeks|day|days)\s+ago\b",
    re.IGNORECASE)


def _protected_spans(text: str) -> list:
    """Spans of ``text`` that anchor() must not rewrite.

    Two kinds, for the same reason — neither is a claim the store owns:

    * a QUOTED string is a record (a search query the agent ran, a
      command, a code literal), and rewriting one falsifies the record.
      Live proof: a stored skill lesson held
      ``web_search(query="Thodoris basketball Panellinios age 9")`` and an
      unguarded pass rewrote the query text itself.
    * a DERIVED gloss ("→ ~3 weeks old") is render output. If one ever
      flows back into a writer, anchoring it would freeze a rendered value
      into the store — reintroducing exactly the snapshot this module
      exists to eliminate.
    """
    spans = [(m.start(), m.end()) for m in _QUOTED_RE.finditer(text)]
    spans += [(m.start(), m.end()) for m in _GLOSS_RE.finditer(text)]
    return spans


def _sub_unprotected(pattern: "re.Pattern", text: str, repl) -> str:
    """``pattern.sub(repl, text)`` that leaves protected spans alone.

    Spans are computed against the ORIGINAL text and matched by offset, so
    a substitution earlier in the string cannot shift a later match out of
    (or into) a protected region mid-pass.
    """
    spans = _protected_spans(text)
    if not spans:
        return pattern.sub(repl, text)
    out, last = [], 0
    for m in pattern.finditer(text):
        out.append(text[last:m.start()])
        protected = any(m.start() < e and s_ < m.end() for s_, e in spans)
        out.append(m.group(0) if protected else repl(m))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def anchor(text, said_at=None) -> str:
    """Rewrite time-derived phrases in ``text`` into absolute anchors.

    ``said_at`` is when the statement was made (date, datetime or ISO
    string); it defaults to now. Idempotent: anchored output contains no
    rewritable age phrase, so re-application is a no-op. Non-string input,
    quoted spans and implausible ages are returned untouched — this runs
    on every profile write and must never be able to destroy a value.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    when = _as_date(said_at) or _today()

    def _anchor_for(m: "re.Match", has_unit: bool):
        """The anchor text behind a match, or None when it should be left
        alone (unparseable count, unknown unit, implausible age)."""
        try:
            count = int(m.group(1))
        except (TypeError, ValueError):
            return None
        unit = _norm_unit(m.group(2)) if has_unit else "year"
        if unit is None or count > _MAX_PLAUSIBLE_AGE.get(unit, 0):
            return None
        return _fmt_anchor(_birth_from_age(when, count, unit), unit)

    # Attributive compounds are ANNOTATED, and this runs FIRST: the
    # annotation it appends ends in "…)" and contains no rewritable age
    # phrase, so the predicative patterns below cannot reach into it.
    def _sub_attrib(m: "re.Match") -> str:
        anc = _anchor_for(m, True)
        return m.group(0) if anc is None else f"{m.group(0)} ({anc})"

    out = _sub_unprotected(_ATTRIB_AGE_RE, text, _sub_attrib)

    for pattern, has_unit in _AGE_PATTERNS:
        def _sub(m: "re.Match", _hu=has_unit) -> str:
            anc = _anchor_for(m, _hu)
            return m.group(0) if anc is None else anc
        out = _sub_unprotected(pattern, out, _sub)

    def _sub_ago(m: "re.Match") -> str:
        try:
            count = int(m.group(1))
        except (TypeError, ValueError):
            return m.group(0)
        unit = _norm_unit(m.group(2))
        if unit is None:
            return m.group(0)
        if unit == "year":
            when2 = _shift_months(when, -count * 12)
        elif unit == "month":
            when2 = _shift_months(when, -count)
        elif unit == "week":
            when2 = when - datetime.timedelta(days=count * 7)
        else:
            when2 = when - datetime.timedelta(days=count)
        return f"in ~{when2.isoformat()}"

    out = _sub_unprotected(_AGO_RE, out, _sub_ago)

    # Canonicalise a month-NAME birth date to ISO, so every stored anchor
    # has ONE shape for derive() to read. This is not a decay fix (an exact
    # date does not decay) — it is a parser fix: two spellings of the same
    # anchor meant the reader had to know both, and the one it did not know
    # rendered no age at all.
    def _sub_named(m: "re.Match") -> str:
        parsed = _named_anchor_date(m)
        if parsed is None:
            return m.group(0)
        d, coarse = parsed
        return f"born {d.year:04d}-{d.month:02d}" if coarse else f"born {d.isoformat()}"

    return _sub_unprotected(_ANCHOR_TEXT_RE, out, _sub_named)


# ── Read-time: anchors -> the value that is true today ──────────────────

def _named_anchor_date(m: "re.Match"):
    """``(date, coarse)`` for a month-NAME anchor match, or None when the
    parts do not form a real calendar date. A month-only match ("born in
    March 2026") is coarse — same rule as an ISO month-precision anchor."""
    g = m.groupdict()
    try:
        if g.get("m1"):
            mon, day, year = g["m1"], int(g["d1"]), int(g["y1"])
            coarse = False
        elif g.get("m2"):
            mon, day, year = g["m2"], int(g["d2"]), int(g["y2"])
            coarse = False
        else:
            mon, day, year = g["m3"], 15, int(g["y3"])
            coarse = True
        key = str(mon).lower().rstrip(".")
        month = _MONTHS.get(key) or _MONTHS.get(
            next((k for k in _MONTHS if k.startswith(key)), ""), None)
        if not month:
            return None
        return datetime.date(year, month, day), coarse
    except (TypeError, ValueError, KeyError):
        return None


def _anchor_date(m: "re.Match"):
    """``(date, approximate, coarse)`` for an anchor match, or a None date
    when it is not a real calendar date.

    ``coarse`` marks an anchor known only to the month or the year — which
    is what a year-stated age produces. It is the precision signal derive()
    needs: see _age_phrase.
    """
    approx = bool(m.group(1))
    try:
        year = int(m.group(2))
        month = m.group(3)
        day = m.group(4)
        if month is None:
            # Year only — nothing finer is known either way.
            return datetime.date(year, 7, 1), True, True
        if day is None:
            # Month granularity means the DAY is unknown, so the age is
            # approximate — but "coarse" (render in years) is decided by
            # the ESTIMATE marker, not by granularity. `born ~2017-01` came
            # from "9 years old" and is good to ±6 months, so months would
            # be false precision; `born 2026-03` came from someone SAYING
            # "born in March 2026" and is good to ±15 days, so forcing
            # years on it renders an infant as "~0 years old".
            return datetime.date(year, int(month), 15), True, approx
        return datetime.date(year, int(month), int(day)), approx, False
    except (TypeError, ValueError):
        return None, approx, False


def _age_phrase(born: datetime.date, now: datetime.date, approx: bool,
                coarse: bool = False) -> Optional[str]:
    """The age at ``now``, rounded to the unit a human would use. None when
    the anchor is in the future — bad data or clock skew must not render
    "-3 months old".

    A ``coarse`` anchor always renders in YEARS. Rendering one in months
    would dress up a ±6-month estimate as a measurement: "my son is 1 year
    old" would come back as "~18 months old", which is both more precise
    than anything known and not what the user said.

    The single ``months < 0`` test is the whole future guard. An earlier
    revision also tested ``born > now`` up front; a mutation run showed
    that branch could not be reached — ``_months_between`` returns a
    negative for EVERY future date, including one later the same month
    (the day-clamp takes it to -1) — so it was dead code that no pin could
    distinguish from a live guard.
    """
    months = _months_between(born, now)
    tilde = "~" if approx else ""
    if months < 0:
        return None
    if coarse:
        years = months // 12
        return f"{tilde}{years} year{'s' if years != 1 else ''} old"
    if months == 0:
        days = (now - born).days
        # Weeks are the unit people actually use for a newborn past the
        # first fortnight, and rendering them keeps the round trip honest:
        # "3 weeks old" has to come back as weeks, not as "~24 days old".
        if days >= 14:
            weeks = days // 7
            return f"{tilde}{weeks} week{'s' if weeks != 1 else ''} old"
        return f"{tilde}{days} day{'s' if days != 1 else ''} old"
    if months < 24:
        return f"{tilde}{months} month{'s' if months != 1 else ''} old"
    years = months // 12
    return f"{tilde}{years} year{'s' if years != 1 else ''} old"


def derive(text, now=None) -> str:
    """Expand every anchor in ``text`` with the value true at ``now``.

    ``"Leonidas born ~2026-02-20"`` -> ``"Leonidas born ~2026-02-20 -> ~6
    months old"``. Render-time only — the result is never persisted, so
    the store keeps holding constants. Any pre-existing gloss is stripped
    first, which makes this idempotent and self-correcting if a derived
    string ever does leak back into a store.
    """
    if not isinstance(text, str) or "born" not in text.lower():
        return text
    today = _as_date(now) or _today()
    stripped = _GLOSS_RE.sub("", text)

    def _sub(m: "re.Match") -> str:
        born, approx, coarse = _anchor_date(m)
        if born is None:
            return m.group(0)
        phrase = _age_phrase(born, today, approx, coarse)
        if phrase is None:
            return m.group(0)
        return f"{m.group(0)} → {phrase}"

    out = _ANCHOR_RE.sub(_sub, stripped)

    def _sub_named(m: "re.Match") -> str:
        parsed = _named_anchor_date(m)
        if parsed is None:
            return m.group(0)
        born, month_only = parsed
        phrase = _age_phrase(born, today, month_only, False)
        return m.group(0) if phrase is None else f"{m.group(0)} → {phrase}"

    return _ANCHOR_TEXT_RE.sub(_sub_named, out)


def has_anchor(text) -> bool:
    """True when ``text`` carries at least one absolute anchor."""
    return isinstance(text, str) and bool(
        _ANCHOR_RE.search(text) or _ANCHOR_TEXT_RE.search(text))


def signature(text) -> str:
    """An anchor-blind form of ``text``, for equality checks that must
    treat two statements of the SAME fact as identical.

    Without this, anchoring would turn a harmless restatement into an
    accumulation: "Leonidas is 4 months old" and, six weeks later, "…is 5
    months old" produce different anchor dates, so the profile's
    exact-string dedup would keep BOTH and inject two contradictory birth
    dates into every prompt. Comparing signatures instead lets the newer
    statement REFINE the anchor in place.
    """
    if not isinstance(text, str):
        return ""
    blinded = _GLOSS_RE.sub("", text)
    blinded = _ANCHOR_TEXT_RE.sub("born <anchor>", blinded)
    blinded = _ANCHOR_RE.sub("born <anchor>", blinded)
    blinded = _AGO_ANCHOR_RE.sub("in <anchor>", blinded)
    return " ".join(blinded.lower().split())
