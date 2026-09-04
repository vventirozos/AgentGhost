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
                               _anchor_date, _months_between,
                               _named_anchor_date, _norm_unit, _today)

#: How far from an age phrase a subject NAME may sit and still be read as its
#: subject. "Your sons are 9 (Thodoris) and 5 months old (Leonidas)" binds at
#: ~10 chars; a name two sentences away is not the subject of this phrase.
#: Deliberately tight: an unbound phrase is skipped, and skipping costs a
#: label we never had, while mis-binding writes a 0.0 on a correct turn.
_BIND_WINDOW = 60

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


def _plausible_months(count: int, unit: str) -> Optional[Tuple[float, float]]:
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
    return (max(0.0, (count - 1) * per), (count + 2) * per)


def _true_months(born, now) -> float:
    """Exact age in months, day remainder included — the day part is what
    makes the tolerance above honest rather than a fudge."""
    whole = _months_between(born, now)
    return max(0.0, float(whole)) + ((now - born).days % 30) / 30.0


def anchored_subjects(profile: Any) -> List[Tuple[str, Any]]:
    """``(subject name, birth date)`` for every anchored fact in the profile.

    Walks values rather than reading known keys: the anchor lands wherever
    `temporal.anchor` found an age phrase, and a key list would go stale the
    first time a fact is stored somewhere new (the whole-reader-set rule).

    The NAME is the word immediately before the anchor, which is how these
    read in the live store: ``"Thodoris (born 2016-11-25) and Leonidas (born
    2026-03-12)"``. A value with no name in front of the anchor yields
    nothing — an unattributable birth date cannot refute a claim about anyone.
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
                nm = re.search(r"([A-Z][\w'-]+)\s*$", head)
                if nm:
                    out.append((nm.group(1), born))

    try:
        _walk(profile)
    except Exception:  # noqa: BLE001 — a checker must never break a turn
        return []
    return out


def _age_claims(reply: str) -> List[Tuple[int, str, int]]:
    """``(count, unit, position)`` for every age phrase in the reply.

    Reuses `temporal._AGE_PATTERNS` — the same patterns that ANCHOR a stored
    age — so the writer and this checker cannot disagree about what an age
    phrase is. A bare "age 9" (no unit) is years, matching that module.
    """
    # ⚠ THE PATTERNS OVERLAP BY DESIGN. "9 years old" matches both the
    # "N years old" and the "N year old" forms, so a naive walk reports the
    # same claim twice — and this route writes NEGATIVES, where a duplicated
    # issue is a duplicated accusation. Spans that overlap are one claim.
    spans: List[Tuple[int, int]] = []
    found: List[Tuple[int, str, int]] = []
    for rx, has_unit in _AGE_PATTERNS:
        for m in rx.finditer(reply or ""):
            try:
                count = int(m.group(1))
            except (TypeError, ValueError):
                continue
            unit = _norm_unit(m.group(2)) if has_unit else "year"
            if unit not in _UNIT_MONTHS:
                continue
            if any(m.start() < e and s_ < m.end() for s_, e in spans):
                continue
            spans.append((m.start(), m.end()))
            found.append((count, unit, m.start()))
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
        issues: List[str] = []
        for count, unit, pos in claims:
            window = _plausible_months(count, unit)
            if window is None:
                continue
            # Bind this phrase to the NEAREST anchored name within the
            # window — and only if ONE name is nearest.
            #
            # ⚠ A TIE IS AMBIGUITY, AND AMBIGUITY MUST NOT REFUTE. The live
            # reply "Your sons are 9 (Thodoris) and 5 months old (Leonidas)"
            # puts both names exactly 14 characters from "5 months old", and
            # first-wins bound the infant's age to the nine-year-old and
            # refuted a CORRECT answer. Skipping the tie costs a label we
            # never had; taking it writes a 0.0 on a turn that was right.
            line_lo, line_hi = _line_span(reply, pos)
            ranked = []
            for name, born in subjects:
                for m in re.finditer(r"\b" + re.escape(name) + r"\b",
                                     reply, re.IGNORECASE):
                    # Same line first, then near enough on it.
                    if not (line_lo <= m.start() < line_hi):
                        continue
                    dist = abs(m.start() - pos)
                    if dist <= _BIND_WINDOW:
                        ranked.append((dist, name, born))
            if not ranked:
                continue          # unbound phrase — not our business
            ranked.sort(key=lambda r: r[0])
            distinct = {r[1] for r in ranked if r[0] == ranked[0][0]}
            if len(distinct) > 1:
                continue          # two subjects equally close: say nothing
            _dist, name, born = ranked[0]
            true_m = _true_months(born, today)
            lo, hi = window
            if lo <= true_m <= hi:
                continue          # consistent — and consistency says NOTHING
            issues.append(
                f"{name} is stated as {count} {unit}(s) old, but the stored "
                f"birth date {born.isoformat()} makes {name} "
                f"{true_m:.1f} months old today")
        return issues
    except Exception:  # noqa: BLE001 — a checker must never break a turn
        return []


__all__ = ["refute_age_claims", "anchored_subjects"]
