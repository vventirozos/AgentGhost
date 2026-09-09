"""Arithmetic refutation of age claims against anchored memory (§4EQ).

WHY THIS EXISTS. 46% of user turns run no tools, so the evidence-grounded
verifier cannot rule on them and they enter the calibration corpus as the
`_UNVERIFIED_PRIOR` placeholder (§4EO). §4EP then measured what filling that
gap is worth: growing the corpus with CONFIRM-only labels shrinks the observed
Brier delta by 2-16x across seeds, because every added row is one the base-rate
predictor already gets right. **Coverage that cannot refute is worse than no
coverage.**

So this route is REFUTE-ONLY, by construction and not by accident:

  * A contradiction returns an issue. A match returns NOTHING — not a
    CONFIRMED, not an UNCERTAIN. The turn stays a placeholder exactly as it is
    today, and every label this route can ever add is a NEGATIVE, the scarce
    class (57 of 402 verdict rows).
  * Absence returns nothing either. The check fires only where BOTH a claimed
    value and a stored comparand exist, so "the store does not mention it" can
    never become "the answer is wrong" — the refute-on-absence trap the
    verifier already carries a truncation guard for, made structurally
    impossible here rather than guarded after the fact.

WHY ARITHMETIC AND NOT A JUDGE. An age against a stored birth date is a
computation, so it needs no model, has no prompt to be talked out of, costs
nothing, and cannot hallucinate a contradiction. §4EL anchored those birth
dates precisely so the value could be recomputed instead of remembered; this
is the first consumer that recomputes one in order to CHECK something.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..memory.temporal import (_AGE_PATTERNS, _ANCHOR_RE, _ANCHOR_TEXT_RE,
                               _COMPOUND_AGE_RE, _NAME_TOKEN, _shift_months,
                               compound_age_months, conjoined_subject,
                               tense_marked, _anchor_date, _months_between,
                               _named_anchor_date, _norm_unit, _today)

# ── WHICH NAME IS THE SUBJECT OF AN AGE PHRASE ─────────────────────────
#
# Not the nearest one. Three review rounds of "nearest name within N
# characters" each moved the defect instead of removing it (§4FN C1 →
# C1-R2 → round 3 C2/M1): a nearby name captured any age at all
# ("Thodoris's laptop is 3 years old", "Leonidas has a cot that is 20
# years old" — REFUTED), while one word between a name and its own age
# ("Thodoris — now 9 years old") handed the age to the OTHER child. A
# distance is a proxy for the grammar; the grammar itself is short enough
# to read.
#
# The subject is the name the phrase is PREDICATED of: the closest name
# before it on the line, when everything between them is a LINK — a
# copula, an age cue, a hedge, punctuation — and nothing else. One word
# outside that list ("'s laptop", "goes to a school that", "and Maria")
# and the name is not the subject, so the phrase is skipped: an unbound
# phrase costs a label we never had, a mis-bound one writes a 0.0 on a
# correct turn. A name is also never the subject when it closes a LIST
# ("Thodoris and Leonidas: 9 years and 5 months old" is two ages). The
# one backward form is the agent's own disambiguation, a bracketed name
# right after the phrase: "9 years old (Thodoris) and 5 months old
# (Leonidas)".
#
# A DATE is a link too: the agent's usual arithmetic shape is "Leonidas:
# born March 12, 2026 → today (Sep 4, 2026) is about 6 months old", and a
# table row is "| Leonidas | 2026-03-12 | 5 months old |" — the birth date
# standing between the name and the age is the computation, not another
# subject. A date SHAPE, not bare digits: "| Thodoris | 2 | 3 years old |"
# is a count of bikes (round 4 M4).
#
# What is deliberately NOT a link (round 4 C1–C4): tense and time words
# ("was", "turned", "turns", "at", "in", "on" — a past or future age is
# not today's, and `tense_marked` skips the whole line anyway), "of" (the
# age belongs to the noun BEFORE "of Thodoris"), a bare possessive
# ("Thodoris's 5 month old brother" — only "'s age" links), and nouns of
# any kind. Every word on the list is pinned in both directions.
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|"
          r"july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
          r"dec(?:ember)?)")
_DATE = (r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{2,4}|"
         + _MONTH + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|"
         r"\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTH + r"\.?(?:,?\s+\d{4})?|\d{4})")
_LINK_WORDS = (r"is|'s\s+age|’s\s+age|'\s+age|age|aged|now|currently|today|"
               r"just|already|still|only|about|around|roughly|approximately|"
               r"nearly|almost|exactly|over|under|who|born|b")
_SUBJECT_LINK_RE = re.compile(
    r"(?:[\s:—–\-,=>→()*|.]|\b(?:" + _LINK_WORDS + r")\b|(?<!\w)" + _DATE + r"(?!\w))*\Z",
    re.IGNORECASE)
_ANNOT_OPEN_RE = re.compile(r"\s*[(\[]\s*")
#: …and "[Thodoris](https://…)" is a markdown link, not an annotation
_ANNOT_CLOSE_RE = re.compile(r"\s*[)\]](?!\()")
#: "the father of Thodoris", "the brother of Leonidas": the name after "of"
#: is a modifier of the noun before it — never the subject of the age, in a
#: reply or in a stored value (round 4 C2)
_OF_RE = re.compile(r"\bof\s*\Z", re.IGNORECASE)
#: a stored "Thodoris's born ~2026-03-20 brother" names the possessor, not
#: the person the date belongs to (round 4 C3)
_POSSESSIVE_RE = re.compile(r"['’]s?\Z", re.IGNORECASE)
#: a bracketed name is an ANNOTATION of the phrase before it, never the
#: subject of the phrase after it: "9 years old (Thodoris), 5 months old
#: (Leonidas)" (round 4 C5)
_BRACKETED_BEFORE_RE = re.compile(r"[(\[]\s*\Z")
_BRACKETED_AFTER_RE = re.compile(r"\s*[)\]]")

#: ⚠ AND THE BINDING IS CONFINED TO ONE LINE. Distance alone was measured
#: against the real reply history (1977 replies) and produced THREE hits, all
#: of them FALSE — every one a markdown list or table where subjects and ages
#: interleave and the nearest name by character count belongs to the previous
#: row:
#:
#:   "- **Leonidas:** born March 12, 2026 → … about 6 months old
#:    - **Thodoris:** …"        -> bound the infant's age to the 9-year-old
#:   "| Vasilis | 1980-01-29 | 44 years |"  -> bound Vasilis's age to Thodoris
#:
#: A line is the unit these replies are actually organised in — one list item,
#: one table row, one sentence. Binding across lines is what turned a correct
#: answer into a 0.0 in the one class the corpus cannot afford noise in.
#: Measured after the fix: zero refutations across the same 1977 replies.


def _line_span(text: str, pos: int) -> Tuple[int, int]:
    """The line containing ``pos``, as ``(start, end)``."""
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return start, (len(text) if end == -1 else end)

#: Months per unit, for putting a claim and a stored fact on one scale.
_UNIT_MONTHS = {"year": 12.0, "month": 1.0, "week": 12.0 / 52.0,
                "day": 12.0 / 365.25}


def _plausible_units(count: float) -> Tuple[float, float]:
    """The band a stated count admits, in its own unit: one below to two
    above, or five percent either way for a large count ("180 days" for a
    child of 176 — days are rounded to fives and tens, round 4 C7) — and
    never below HALF the count, or "1 year old" admits a newborn (N−1 = 0;
    round 4 M5)."""
    slack = 0.05 * count
    return (max(count - max(1.0, slack), count * 0.5, 0.0),
            count + max(2.0, slack))


def _plausible_months(count: float, unit: str) -> Optional[Tuple[float, float]]:
    """The true-age interval a claim of ``count unit`` is consistent with.

    ⚠ GENEROUS ON PURPOSE. A colloquially rounded but CORRECT answer must
    never be refuted. The live case that motivated this: a child of 5 months
    23 days, where the store's own `_age_phrase` renders "5 months" and the
    user called it "about 6 months" — both are right, and a pedantic check
    would have written a 0.0 on one of them.

    So a claim of N units admits anything from one unit below to two above:
    "6 months" accepts a true 5.0-8.0 months, "9 years" accepts 8-11 years.
    That still refutes what it exists to refute — "9 years old" against a
    true 5.8 months is off by a factor of 18.
    """
    per = _UNIT_MONTHS.get(unit or "")
    if per is None or count < 0:
        return None
    lo, hi = _plausible_units(count)
    return (lo * per, hi * per)


def _true_months(born, now) -> float:
    """Exact age in months, day remainder included — the day part is what
    makes the tolerance above honest rather than a fudge. §4FN: the
    remainder is the days past the last whole month (`(days % 30)/30` was
    off by up to a month — 117.0 for a true 117.33 — which bounded how
    tight the compound window could be)."""
    whole = max(0, _months_between(born, now))
    anchor = _shift_months(born, whole)
    frac = max(0.0, min(0.999, (now - anchor).days / 30.4375))
    return float(whole) + frac


def anchored_subjects(profile: Any) -> List[Tuple[str, Any]]:
    """``(subject name, birth date)`` for every anchored fact in the profile.

    Walks values rather than reading known keys: the anchor lands wherever
    `temporal.anchor` found an age phrase, and a key list would go stale the
    first time a fact is stored somewhere new (the whole-reader-set rule).

    The NAME is the word immediately before the anchor, which is how these
    read in the live store: ``"Thodoris (born 2016-11-25) and Leonidas (born
    2026-03-12)"``. A value with no name in front of the anchor yields
    nothing — an unattributable birth date cannot refute a claim about anyone.
    Nor does a name that closes a LIST: ``"Thodoris and Leonidas: born
    ~2017-04-04"`` is one date glued onto two people (a writer defect this
    module must survive in data already stored — round 3 C1), and reading it
    as Leonidas's would refute the correct answer about him forever. Names
    are Unicode ("Θοδωρής (born 2016-11-25)" is a subject — round 3 m4).
    """
    out: List[Tuple[str, Any]] = []

    def _walk(node):
        if isinstance(node, dict):
            for v in node.values():
                _walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _walk(v)
        elif isinstance(node, str):
            _scan(node)

    def _scan(text: str):
        for rx, parse in ((_ANCHOR_RE, _anchor_date),
                          (_ANCHOR_TEXT_RE, _named_anchor_date)):
            for m in rx.finditer(text):
                try:
                    parsed = parse(m)
                except Exception:  # noqa: BLE001 — a bad row is not a fact
                    continue
                born = parsed[0] if isinstance(parsed, tuple) else parsed
                if born is None:
                    continue
                # The name sits before the anchor, possibly through an
                # opening bracket: "Leonidas (born 2026-03-12".
                head = text[:m.start()].rstrip(" ([-—,:")
                nm = re.search(r"(?<![\w'’-])(" + _NAME_TOKEN + r")\s*$", head)
                if not nm or not nm.group(1)[:1].isupper():
                    continue
                if conjoined_subject(head[:nm.start()] + nm.group(1), comma_names=None):
                    continue
                if _POSSESSIVE_RE.search(nm.group(1)) or _OF_RE.search(head, 0, nm.start()):
                    continue
                out.append((nm.group(1), born))

    try:
        _walk(profile)
    except Exception:  # noqa: BLE001 — a checker must never break a turn
        return []
    return out


def _age_claims(reply: str) -> List[Tuple[float, str, int, int]]:
    """``(count, unit, position)`` for every age phrase in the reply.

    Reuses `temporal._AGE_PATTERNS` AND `temporal._COMPOUND_AGE_RE` — the
    same patterns that ANCHOR a stored age — so the writer and this checker
    cannot disagree about what an age phrase is. A bare "age 9" (no unit) is
    years, matching that module. Returns ``(count, unit, start, end)``: a
    compound ("5 months and 23 days old") is ONE claim, in MONTHS (the
    unit whose tolerance is honest for it — one month below, two above —
    where its largest unit's would admit a 3-year band for "1 year and 2
    months", review §4FN M1); ``start``/``end`` are the phrase span the
    subject binding reads its link from.
    """
    # ⚠ THE PATTERNS OVERLAP BY DESIGN. "9 years old" matches both the
    # "N years old" and the "N year old" forms, so a naive walk reports the
    # same claim twice — and this route writes NEGATIVES, where a duplicated
    # issue is a duplicated accusation. Spans that overlap are one claim.
    spans: List[Tuple[int, int]] = []
    found: List[Tuple[float, str, int, int]] = []
    # §4FN: a COMPOUND age ("5 months and 23 days old", "9 years, 9 months
    # and 10 days old") is ONE claim whose value is the sum, not the last
    # component. The single-unit patterns below match "23 days old" inside
    # it and refuted three correct replies live (2026-09-04: "Leonidas is
    # stated as 23 day(s) old, but the stored birth date … makes Leonidas
    # 5.9 months old"). Compounds are taken first, as whole-day totals so
    # the generous window below still applies, and their spans shadow the
    # single-unit matches inside them.
    for m in _COMPOUND_AGE_RE.finditer(reply or ""):
        # the span shadows the single-unit patterns EVEN when the compound
        # yields no claim (implausible total, review §4FN M-A): reading its
        # tail component is the original defect. A plural compound
        # ("Thodoris and Leonidas are 9 years and 5 months old") is not
        # decided here: its subject is a list, and the binding below never
        # takes a list or a plural copula as a link (round 3 C1).
        spans.append((m.start(), m.end()))
        total_months = compound_age_months(m)
        if total_months is None:
            continue
        found.append((round(total_months, 2), "month", m.start(), m.end()))
    for rx, has_unit in _AGE_PATTERNS:
        for m in rx.finditer(reply or ""):
            try:
                # a decimal is one number ("9.5 years old" is 9.5, never 5)
                count = float(str(m.group(1)).replace(",", "."))
                count = int(count) if count.is_integer() else count
            except (TypeError, ValueError):
                continue
            unit = _norm_unit(m.group(2)) if has_unit else "year"
            if unit not in _UNIT_MONTHS:
                continue
            if any(m.start() < e and s_ < m.end() for s_, e in spans):
                continue
            spans.append((m.start(), m.end()))
            found.append((count, unit, m.start(), m.end()))
    return found


def refute_age_claims(*, reply: str, profile: Any, now=None) -> List[str]:
    """Contradictions between age claims in ``reply`` and anchored memory.

    ⚠ AN EMPTY LIST MEANS "NOTHING TO SAY", NEVER "THE REPLY IS FINE". Every
    caller must treat it as no-verdict; reading it as a pass would turn this
    into the confirm-only route §4EP measured as worthless.

    Total: never raises. A checker that can break a turn would be traded away
    the first time it did.
    """
    try:
        if not isinstance(reply, str) or not reply.strip():
            return []
        subjects = anchored_subjects(profile)
        if not subjects:
            return []
        today = _today() if now is None else now
        claims = _age_claims(reply)
        if not claims:
            return []
        # one name, two birth dates = bad data, not two chances to refute
        # (round 4 C2: the outcome depended on dict order)
        dates_by_name: Dict[str, set] = {}
        for _n, _b in subjects:
            dates_by_name.setdefault(_n.lower(), set()).add(_b)
        names = frozenset(dates_by_name)
        issues: List[str] = []
        occ_cache: Dict[Tuple[int, int], list] = {}
        for count, unit, start, end in claims:
            if unit not in _UNIT_MONTHS or count < 0:
                continue
            line_lo, line_hi = _line_span(reply, start)
            # a past or future age is not today's (round 4 C1): "Thodoris
            # was 4 years old when you moved", "Leonidas turns 7 months old
            # on October 12", "By March Leonidas is 12 months old"
            if tense_marked(reply[line_lo:line_hi]):
                continue
            if (line_lo, line_hi) not in occ_cache:
                occ_cache[(line_lo, line_hi)] = _name_occurrences(
                    reply, line_lo, line_hi, subjects)
            bound = _bind_subject(reply, start, end, line_lo,
                                  occ_cache[(line_lo, line_hi)], names)
            if bound is None:
                continue          # unbound phrase — not our business
            name, born = bound
            if len(dates_by_name.get(name.lower(), ())) > 1:
                continue
            if born > today:
                continue          # a future birth date is bad data, not evidence
            lo_u, hi_u = _plausible_units(count)
            if unit in ("week", "day"):
                # compared on the claim's OWN scale: a calendar-month age
                # and a 12/52-month week disagree by a day, and "26 weeks"
                # for a child of 25 weeks 1 day was refuted (round 4 C7)
                per_days = 7 if unit == "week" else 1
                true_days = (today - born).days
                if lo_u * per_days <= true_days <= hi_u * per_days:
                    continue
                issues.append(
                    f"{name} is stated as {count:g} {unit}(s) old, but the stored "
                    f"birth date {born.isoformat()} makes {name} "
                    f"{true_days} days old today")
                continue
            true_m = _true_months(born, today)
            per = _UNIT_MONTHS[unit]
            if lo_u * per <= true_m <= hi_u * per:
                continue          # consistent — and consistency says NOTHING
            issues.append(
                f"{name} is stated as {count:g} {unit}(s) old, but the stored "
                f"birth date {born.isoformat()} makes {name} "
                f"{true_m:.1f} months old today")
        return issues
    except Exception:  # noqa: BLE001 — a checker must never break a turn
        return []


def _name_occurrences(reply: str, line_lo: int, line_hi: int,
                      subjects: List[Tuple[str, Any]]) -> list:
    """``(start, end, name, born)`` for every anchored name on the line —
    computed once per line, not once per claim (round 4 m4)."""
    line = reply[line_lo:line_hi]
    occ = []
    for name, born in subjects:
        for m in re.finditer(r"\b" + re.escape(name) + r"\b", line, re.IGNORECASE):
            occ.append((line_lo + m.start(), line_lo + m.end(), name, born))
    return occ


def _bind_subject(reply: str, start: int, end: int, line_lo: int,
                  occ: list, names=()) -> Optional[Tuple[str, Any]]:
    """The anchored ``(name, born)`` the phrase at ``[start, end)`` is
    predicated of, or None (see the module note above the link pattern).

    Forward first: the closest anchored name BEFORE the phrase on its line,
    accepted only when the text between them is all link, the name does not
    close a list, does not follow "of", and is not itself bracketed (an
    annotation of the PREVIOUS phrase). Otherwise the one backward form, a
    bracketed name right after the phrase. Anything else is unbound.
    """
    before = [o for o in occ if o[1] <= start]
    if before:
        q, q_end, name, born = max(before, key=lambda o: o[1])
        bracketed = (_BRACKETED_BEFORE_RE.search(reply, line_lo, q)
                     and _BRACKETED_AFTER_RE.match(reply, q_end))
        if (not bracketed
                and _SUBJECT_LINK_RE.match(reply, q_end, start)
                and not conjoined_subject(reply[line_lo:q_end], comma_names=names)
                and not _OF_RE.search(reply, line_lo, q)):
            return name, born
    after = [o for o in occ if o[0] >= end]
    if after:
        q, q_end, name, born = min(after, key=lambda o: o[0])
        if (_ANNOT_OPEN_RE.fullmatch(reply, end, q)
                and _ANNOT_CLOSE_RE.match(reply, q_end)):
            return name, born
    return None


__all__ = ["refute_age_claims", "anchored_subjects"]
