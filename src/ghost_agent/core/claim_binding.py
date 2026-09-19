"""§4IM — the claim-binding verifier: verdicts computed from validated quotes.

WHY (measured, §4IJ, six bench arms on seed + live-derived cases). The
incumbent judge asks two models for an OPINION and returns one. The cheap
model false-refutes a third of clean replies ("not supported" on vibes); the
strong model — the agent's own — confirms contradictions it can plainly see
("a minor issue"). Every re-weighting of the two opinions (rules, objections
block, concession downgrade, rebuttal burden, forced suspects, judge swaps)
moved at most a tenth of the false-confirm rate and always at a clean-side
cost. The structure was the defect.

WHAT. One cheap-leg call does QUOTING work, which small models do reliably
and which code can check:

    reply  ──►  checkable claims, each a VERBATIM quote of the reply
    evidence ─► for each claim, the evidence span that supports or
                contradicts it, VERBATIM

Code then validates every quote by normalized containment (a claim that is
not in the reply, a span that is not in the evidence, is dropped — never a
verdict), compares claim and span (numbers normalised for units, rounding
and hedges), scans the evidence for a second line with the same skeleton
and a different value (CONFLICTING EVIDENCE — the omitted-contradiction
class), flags labelled values that cannot be (a latitude of 128°), and
computes the verdict:

    REFUTED    ⇐ at least one VALIDATED contradiction / conflict / implausible
                 value on a checkable claim (the issue names both quotes)
    CONFIRMED  ⇐ every checkable claim bound to an agreeing span
    UNCERTAIN  ⇐ otherwise (unbound load-bearing claims are counted, not
                 refuted, in phase 1; truncated evidence is never a refute)

A subjective gloss or a derived summary is not a checkable claim, so it
cannot be refuted: the "beautiful Saturday afternoon" class of fake refute
is impossible by construction. A validated disagreement refutes whatever
the strong model's mood: the "34°C beside 35°C", "5 PNGs beside seven",
"RECOVERED beside missing" class of laundered confirm is impossible by
construction. The failure mode of everything else is UNCERTAIN.

Pure functions throughout (the LLM call is injected) so every rule has a
table test; the orchestration lives in `Verifier._verify_claim_binding`.
"""
from __future__ import annotations

import functools
import json
import math
import datetime
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── contract ────────────────────────────────────────────────────────────

CLAIM_KINDS = ("number", "count", "date", "name", "file", "status", "other")
#: Kinds whose absence from the evidence is load-bearing (counted in phase
#: 1, a candidate refute in phase 2). "other" is prose the model chose to
#: list; it is checked when bound and ignored when not.
LOAD_BEARING_KINDS = ("number", "count", "date", "name", "file", "status")
RELATIONS = ("support", "contradict", "absent")

MAX_CLAIMS = 8
MIN_QUOTE_CHARS = 4
#: Prompt guidance for the model; the parser never CUTS a quote to these —
#: a span sliced at 160 chars lost the "Tonight: low 24°C" that carried the
#: claim's figure and left the current temperature to disagree with (seed
#: long-weather-1, a fake refute). Oversized quotes are validated whole and
#: only dropped beyond the hard sanity caps below.
MAX_QUOTE_CHARS = 240
MAX_SPAN_CHARS = 160
HARD_QUOTE_CAP = 1200
HARD_SPAN_CAP = 2000

#: Hedge words that widen the numeric tolerance from "rounds to the same
#: figure at the claim's precision" to ±`HEDGE_REL_TOL`.
HEDGE_WORDS = ("~", "about", "approximately", "approx", "around", "roughly",
               "nearly", "almost", "close to", "circa", "≈")
HEDGE_REL_TOL = 0.05

CLAIM_BINDING_PROMPT = """You are a claim binder. You do NOT judge the reply. You quote.

REPLY (the agent's answer to the user):
{claim}

EVIDENCE (the tool outputs the reply was built from, each prefixed with [tool_name]):
{evidence}

USER REQUEST:
{context}

Step 1 — list the reply's CHECKABLE claims: every specific number, count, date, name, file/path, status ("succeeded", "started", "recovered", "all tasks done") or other verifiable statement. Each claim is an EXACT, VERBATIM fragment of the REPLY (copy it character for character, 4–240 characters). Do not list subjective phrasing ("beautiful", "fast", "clean"), advice, or plans. At most {max_claims} claims, most load-bearing first.

Step 2 — for EACH claim, quote the EVIDENCE span (exact, verbatim, at most 160 characters, from one place) that most directly SUPPORTS or CONTRADICTS it, and say which. If the evidence states two different values for the same thing, quote the one that DIFFERS from the claim and mark it "contradict". If nothing in the evidence bears on the claim, write an empty evidence_quote and relation "absent". Never paraphrase a quote; never invent one.

Respond ONLY with a MINIFIED single-line JSON object — no code fences, no prose before or after, no extra keys. Your response MUST start with {{ and contain no newlines:
{{"claims":[{{"quote":"exact reply fragment","kind":"number|count|date|name|file|status|other","evidence_quote":"exact evidence fragment or empty","relation":"support|contradict|absent"}}]}}"""


@dataclass
class Binding:
    quote: str
    kind: str
    evidence_quote: str
    relation: str                 # the model's stated relation (a hint only)
    valid_claim: bool = False     # quote ⊂ reply
    valid_span: bool = False      # evidence_quote ⊂ evidence
    outcome: str = "unbound"      # agree | disagree | conflict | implausible | unbound | unchecked
    detail: str = ""              # human-readable reason for disagree/conflict/implausible
    residual: bool = False        # §4IN: outcome set by the residual judge's validated quote

    def to_dict(self) -> Dict[str, Any]:
        return {"quote": self.quote, "kind": self.kind, "evidence_quote": self.evidence_quote,
                "relation": self.relation, "valid_claim": self.valid_claim,
                "valid_span": self.valid_span, "outcome": self.outcome, "detail": self.detail,
                "residual": self.residual}


@dataclass
class ClaimBindingResult:
    verdict: str                  # CONFIRMED | REFUTED | UNCERTAIN
    confidence: float
    issues: List[str]
    bindings: List[Binding] = field(default_factory=list)
    dropped: int = 0              # model rows whose claim quote was not in the reply
    reasoning: str = ""
    audit: List[Any] = field(default_factory=list)   # AuditFigure rows (the number audit)
    entities: List[Any] = field(default_factory=list)   # AuditEntity rows (the named-entity audit)
    findings: List[Any] = field(default_factory=list)   # ClassFinding rows (artifact / constraint / evidence / topic)

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for b in self.bindings:
            c[b.outcome] = c.get(b.outcome, 0) + 1
            if b.residual:
                c[f"residual_{b.outcome}"] = c.get(f"residual_{b.outcome}", 0) + 1
        c["dropped"] = self.dropped
        for f in self.audit:
            key = f"audit_{f.status}"
            c[key] = c.get(key, 0) + 1
        for e in self.entities:
            key = f"entity_{e.status}"
            c[key] = c.get(key, 0) + 1
        for g in self.findings:
            key = f"class_{g.kind}"
            c[key] = c.get(key, 0) + 1
        return c

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "confidence": self.confidence, "issues": list(self.issues),
                "bindings": [b.to_dict() for b in self.bindings], "dropped": self.dropped,
                "audit": [f.to_dict() for f in self.audit],
                "entities": [e.to_dict() for e in self.entities],
                "findings": [g.to_dict() for g in self.findings],
                "counts": self.counts(), "reasoning": self.reasoning}


# ── quote validation ────────────────────────────────────────────────────

def normalize_for_containment(s: str) -> str:
    """Case + whitespace + Unicode folding (NFKC, zero-widths stripped, curly
    quotes/dashes straightened) — the same folding the rebuttal-burden
    validator uses, so "verbatim" tolerates typographic punctuation."""
    # NFC, not NFKC: NFKC turns "2⁵" into "25" and "½" into "1⁄2" — figures
    # that were never written (review §4IN m6). Typographic quotes, dashes
    # and no-break spaces are folded explicitly.
    s = unicodedata.normalize("NFC", str(s or ""))
    s = s.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                                   "\u2212": "-", "\u2010": "-", "\u2011": "-",       # the true minus and hyphens: "−5°C" is negative (review §4IY)
                                   "\u00a0": " ", "\u202f": " ", "\u2009": " "}))
    s = re.sub("[\\u200b\\u200c\\u200d\\ufeff]", "", s)
    return re.sub(r"\s+", " ", s.strip().lower())


def _fold_spaces(text: str) -> str:
    """Length-preserving: no-break / thin spaces become spaces, so a RAW
    line and its normalized snapped window read the same figures."""
    return str(text or "").translate(str.maketrans({"\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u2212": "-", "\u2010": "-", "\u2011": "-"}))


def _whole_token_find(needle: str, hay: str) -> int:
    """Index of an occurrence of `needle` in `hay` that does not split a
    token — "the count is 12" is NOT in "the count is 120 files", "port
    810" is not in "port 8100" (review §4IN M3). -1 when none."""
    start = 0
    while True:
        i = hay.find(needle, start)
        if i < 0:
            return -1
        j = i + len(needle)
        left_ok = (i == 0 or not (hay[i - 1].isalnum() and needle[:1].isalnum())) and not (
            needle[:1].isdigit() and i >= 2 and hay[i - 1] in ".," and hay[i - 2].isdigit())   # "284 orders" is not in "1,284 orders" (review §4IY)
        right_ok = (j >= len(hay) or not (hay[j].isalnum() and needle[-1:].isalnum())
                    and not (needle[-1:].isdigit() and hay[j] in ".," and j + 1 < len(hay) and hay[j + 1].isdigit()))
        if left_ok and right_ok:
            return i
        start = i + 1


def quote_in(quote: str, text: str, *, min_chars: int = MIN_QUOTE_CHARS) -> bool:
    """Normalized containment on whole tokens with a minimum length — a
    fragment shorter than `min_chars` cannot anchor anything."""
    nq = normalize_for_containment(quote)
    if len(nq) < min_chars:
        return False
    return _whole_token_find(nq, normalize_for_containment(text)) >= 0


#: Snapping: the model's quote is a LOCATOR; the validated text is the
#: source's own. A quote that is not a verbatim substring but matches a
#: window of the text at this ratio or better (a dropped word, a changed
#: article, a trimmed clause end) is replaced by that window, so every rule
#: downstream still runs on text that exists. Below the ratio: unbound.
#: Mined pool, 60 clean replies: 56 "found no evidence span" rows before
#: snapping — the E4B finds the passage and does not copy it exactly.
SNAP_MIN_RATIO = 0.9
SNAP_MIN_CHARS = 12          # a short quote matches too many windows to be a locator


SNAP_MAX_WINDOWS = 60        # bound the work: a saturated quote on a 12k evidence blocked the loop for seconds


def _snap_tokens_match(nq: str, cand: str) -> bool:
    """The window is the quote's own field, not a neighbour's: the same
    alphabetic tokens (labels may not drift — "channel x" is not "channel
    y", review §4IN m7) and at most one differing numeric token (a misquoted
    figure: the real text wins)."""
    strip = lambda x: re.sub(r"°[a-z]\b", "°", x)          # "36°c": the unit letter is part of the figure
    ta, tb = re.findall(r"[a-z]+", strip(nq)), re.findall(r"[a-z]+", strip(cand))
    if sorted(ta) != sorted(tb):
        return False
    na, nb = re.findall(r"\d+(?:\.\d+)?", nq), re.findall(r"\d+(?:\.\d+)?", cand)
    if len(na) != len(nb):
        return False
    diff = [(x, y) for x, y in zip(na, nb) if x != y]
    return len(diff) == 0 or (len(diff) == 1 and len(diff[0][0]) == len(diff[0][1]))   # 29→28, never 12→120


def snap_quote(quote: str, text: str) -> Optional[str]:
    """The normalized window of `text` the quote denotes: the quote itself
    (widened to whole tokens) when it is a verbatim substring, else the
    best window around the quote's rarest long token when it matches at
    ≥ SNAP_MIN_RATIO with the same labels; None otherwise."""
    import difflib
    nq = normalize_for_containment(quote)
    nt = normalize_for_containment(text)
    if len(nq) < MIN_QUOTE_CHARS:
        return None
    i = _whole_token_find(nq, nt)
    if i >= 0:
        a, b = _token_bounds(nt, i, i + len(nq))
        return nt[a:b].strip()
    if len(nq) < SNAP_MIN_CHARS:
        return None
    toks = re.findall(r"[a-z0-9][\w.-]{3,}", nq)
    if not toks:
        return None
    # the rarest present token anchors; its positions are capped, rarest first
    tok = min(set(toks), key=lambda t: (nt.count(t) or 10 ** 6, -len(t)))
    if nt.count(tok) == 0:
        return None
    off = nq.find(tok)
    L = len(nq)
    positions: List[int] = []
    start = 0
    while len(positions) < SNAP_MAX_WINDOWS // 3:
        j = nt.find(tok, start)
        if j < 0:
            break
        positions.append(j)
        start = j + 1
    best, best_ratio = None, 0.0
    for j in positions:
        for lo in (j - off - 4, j - off, j - off + 4):
            lo = max(0, lo)
            for span_len in (L, int(L * 0.9), int(L * 1.1)):
                a, b = _token_bounds(nt, lo, min(len(nt), lo + span_len))
                cand = nt[a:b]
                if not cand or not _snap_tokens_match(nq, cand):
                    continue
                sm = difflib.SequenceMatcher(None, nq, cand, autojunk=False)
                if sm.quick_ratio() < SNAP_MIN_RATIO:
                    continue
                ratio = sm.ratio()
                if ratio > best_ratio:
                    best, best_ratio = cand, ratio
                    if ratio >= 0.999:
                        return best.strip()
    if best is None or best_ratio < SNAP_MIN_RATIO:
        return None
    return best.strip()


def _token_bounds(text: str, lo: int, hi: int) -> Tuple[int, int]:
    """Widen [lo, hi) to whole tokens: a window that cuts "36°c" to "3",
    "12.75" to "12", "1,000" to "1" or "-5" to "5" would mint a figure that
    was never written."""
    while lo > 0 and text[lo - 1].isalnum() and lo < len(text) and text[lo].isalnum():
        lo -= 1
    if lo > 0 and text[lo - 1] in "-+€$£" and lo < len(text) and text[lo].isdigit():
        lo -= 1
    while 0 < hi < len(text):
        if text[hi - 1].isalnum() and text[hi].isalnum():
            hi += 1
        elif text[hi - 1].isdigit() and text[hi] in ".," and hi + 1 < len(text) and text[hi + 1].isdigit():
            hi += 2                                            # cross the separator AND take the digit
        else:
            break
    while hi < len(text) and (text[hi] in "%°" or (text[hi - 1] == "°" and text[hi].isalpha())):
        hi += 1                                                # a unit glued to its figure: "36°c", "28%"
    return lo, hi


# ── numbers and units ───────────────────────────────────────────────────

# Three shapes, in this order: thousands-grouped ("1,284", "12,345.6"), a
# DECIMAL COMMA ("4,3", "23,60" — the Greek and continental spelling; fresh-eye
# review §4IY: "4,3" lexed as the figure 3 and refuted a correct Greek reply),
# then a plain figure. The thousands alternative is written as an atomic
# group (a lookahead captured, then re-matched by backreference) — Python
# 3.10 has no possessive quantifier and `\d{1,3}(?:,\d{3})+` re-tried every
# group on a comma-run of mixed widths: a 2,900-element JSON array cost the
# binder 6 s per digest.
def _num_core(tag: str) -> str:
    return (rf"[-+]?\d{{1,3}}(?=(?P<{tag}g>(?:,\d{{3}})+))(?P={tag}g)(?:\.\d+)?"
            rf"|[-+]?\d+,\d{{1,2}}(?![\d,])"
            rf"|[-+]?\d+(?:\.\d+)?")
_NUM_CORE = _num_core("n")   # "100 200 300" is three figures, not one
_NUM_CORE_LO, _NUM_CORE_HI = _num_core("lo"), _num_core("hi")
_UNIT_ALT = (r"%|°c|°f|°|tb|gb|mb|kb|kib|mib|gib|bytes?|ms|s|sec|secs|seconds?|min|mins|minutes?|h|hr|hrs|hours?"
             r"|km|m|cm|mm|kg|mg|g|k|million|billion|thousand|bn|mn")
#: `(?![.,]?\d)` after the figure: a number is never the truncated prefix of
#: a longer one. Glued page text ("orbitalperiodof 29.45years", the space
#: collapsed by the extractor) made the engine back off "29.45" (blocked by
#: the "y") to "29" — a figure that was never written, refuting a correct
#: "26"→ see the date rule (live row 2026-09-18, the Saturn probe).
#: `(?:(?P<cur>[€$£])\s*)?` — NOT `(?P<cur>[€$£])?\s*`: an unconditional `\s*`
#: before the figure backtracks O(n²) over the space run a masked code fence
#: leaves behind (14,000 spaces → 7.7 s per call; a 14 KB fenced reply cost
#: 26 s in the turn — corpus replay §4IN). Same match set.
_NUM_RE = re.compile(
    rf"(?:(?P<cur>[€$£])\s*)?(?<![\w.])(?<!\d,)(?P<num>{_NUM_CORE})(?![.,]?\d)\s*(?P<unit>{_UNIT_ALT})?(?![\w])", re.IGNORECASE)   # `(?<!\d,)`: never the tail of "4,3"
#: A RANGE is one quantity, not two: "spans x=360–380" bound to `ballX: 370`
#: is agreement (the value lies inside), not a 360-vs-370 contradiction
#: (mined pool rec-3adeaf27e8, a clean reply refuted). Connectors: a dash,
#: "to", or "between A and B"; the unit may sit on either endpoint.
_RANGE_RE = re.compile(
    rf"(?P<between>\bbetween\s+)?(?:(?P<cur>[€$£])\s*)?(?<![\w.])(?<!\d,)(?P<lo>{_NUM_CORE_LO})(?:\s*(?P<unit1>{_UNIT_ALT}))?"
    rf"(?P<conn>(?<!\s)-(?!\s)|\s*[–—]\s*|\s+to\s+|\s+and\s+)(?:(?P<cur2>[€$£])\s*)?(?P<hi>{_NUM_CORE_HI})(?![.,]?\d)\s*(?P<unit>{_UNIT_ALT})?(?![\w])",
    re.IGNORECASE)
#: Word/letter multipliers that scale a CURRENCY figure ("€3.4 million" =
#: "€3.4M" = 3,400,000 money). Without a currency sign "m" stays metres.
_MONEY_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6,
               "bn": 1e9, "billion": 1e9}

#: unit → (family, factor to the family's base)
_UNITS: Dict[str, Tuple[str, float]] = {
    "b": ("bytes", 1.0), "byte": ("bytes", 1.0), "bytes": ("bytes", 1.0),
    "kb": ("bytes", 1024.0), "kib": ("bytes", 1024.0),
    "mb": ("bytes", 1024.0 ** 2), "mib": ("bytes", 1024.0 ** 2),
    "gb": ("bytes", 1024.0 ** 3), "gib": ("bytes", 1024.0 ** 3),
    "tb": ("bytes", 1024.0 ** 4),
    "ms": ("time", 0.001), "s": ("time", 1.0), "sec": ("time", 1.0), "secs": ("time", 1.0),
    "second": ("time", 1.0), "seconds": ("time", 1.0),
    "min": ("time", 60.0), "mins": ("time", 60.0), "minute": ("time", 60.0), "minutes": ("time", 60.0),
    "h": ("time", 3600.0), "hr": ("time", 3600.0), "hrs": ("time", 3600.0),
    "hour": ("time", 3600.0), "hours": ("time", 3600.0),
    "mm": ("length", 0.001), "cm": ("length", 0.01), "m": ("length", 1.0), "km": ("length", 1000.0),
    "%": ("percent", 1.0), "°c": ("temp_c", 1.0), "°": ("degrees", 1.0), "°f": ("temp_f", 1.0),
    "mg": ("mass", 0.001), "g": ("mass", 1.0), "kg": ("mass", 1000.0),
    "k": ("thousand", 1000.0),
}


@dataclass(frozen=True)
class Quantity:
    value: float          # in the family's base unit (or raw when unitless); a range's LOW end
    family: str           # "" when unitless
    decimals: int         # decimals the text carried (for rounding comparisons)
    text: str
    unit: str = ""        # the unit token as written, lowercased ("kb", "min", "million")
    hi: Optional[float] = None   # a range's HIGH end (same base unit); None for a scalar
    bound: str = ""       # "lower" ("over 160", "160+", "at least"), "upper" ("under", "up to", "at most"), "" exact

    @property
    def is_range(self) -> bool:
        return self.hi is not None


#: Date and clock tokens are not quantities: `2026-07-07` would otherwise
#: read as 2026, −7, −7 and `14:20` as 14 and 20 — both produced fake
#: disagreements on the seed set (pg-orders, weather).
#: Finite month spellings — a `[a-z]*` wildcard read "3 separate", "2 octets",
#: "12 decimal", "market 2026" as dates and erased the figures (review §4IN M4).
#: Greek months too — nominative ("Αύγουστος 2026"), the genitive every
#: written date uses ("15 Αυγούστου"), colloquial "Μάη", the three-letter
#: abbreviations — with and without accents: "στις 15 Αυγούστου" left a
#: bare 15 that a gazzetta dateline "16 Αυγούστου 2026 - 22:17" then
#: "misreported" (corpus turn 2b753f78, §4IY). Lexical guards speak Greek.
_GREEK_MONTH = (r"(?:[ιί]αν(?:ου[αά]ρ(?:ιος|[ιί]ου))?|φεβ(?:ρου[αά]ρ(?:ιος|[ιί]ου))?|μ[αά]ρ(?:τ(?:ιος|[ιί]ου))?"
                r"|απρ(?:[ιί]λ(?:ιος|[ιί]ου))?|μ[αά][ιίϊΐ](?:ος|ου)?|μ[αά]η|ιο[υύ]ν(?:ιος|[ιί]ου)?|ιο[υύ]λ(?:ιος|[ιί]ου)?"
                r"|α[υύ]γ(?:ο[υύ]στ(?:ος|ου))?|σεπτ?(?:[εέ]μβρ(?:ιος|[ιί]ου))?|οκτ(?:[ωώ]βρ(?:ιος|[ιί]ου))?"
                r"|νο[εέ](?:μβρ(?:ιος|[ιί]ου))?|δεκ(?:[εέ]μβρ(?:ιος|[ιί]ου))?)\.?(?![^\W\d_])")
_MONTH = (r"(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
          r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?(?![a-z])|" + _GREEK_MONTH + ")")
_DAY = r"\d{1,2}(?!\d)(?:st|nd|rd|th)?"      # "June 2026" has no day: "20" is not one
_DATE_TIME_RE = re.compile(
    r"(?<!\d)\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?\b|\b\d{1,2}:\d{2}(?::\d{2})?\b"   # (?<!\d): glued "on2026-05-14" is still a date
    r"|\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b"
    # month-name dates and ranges: "Feb 27-28, 2025", "March 14, 2026",
    # "September 8th, 2040", "28 February 2025", "Feb 28 - Mar 1, 2026"
    rf"|\b{_MONTH}\s+{_DAY}(?:\s*[-–]\s*(?:{_MONTH}\s+)?{_DAY})?(?:,?\s*\d{{4}})?"
    rf"|\b{_DAY}\s+{_MONTH}(?:,?\s*\d{{4}})?"
    # month + year, no day ("as of June 2026", glued "June2026[update]")
    rf"|\b{_MONTH}\s*\d{{4}}\b"
    # a decade ("the 2000s", "the 1990s") is neither a year nor 2000 seconds
    r"|\b(?:1[89]|20)\d0s\b"
    # relative time ("OTD 20 years ago", "posted 3 hours ago") is a date
    # anchored to the source's own unknown "now" — never "the" value a reply
    # figure misreads (corpus replay §4IP R6: "a 24-year span" vs "20 years
    # ago", protected until then only by a list number in the sentence)
    r"|\b\d+(?:[.,]\d+)?\s*(?:sec(?:ond)?s?|min(?:ute)?s?|h(?:ou)?rs?|days?|weeks?|wks?|months?|years?|yrs?|decades?)\s+ago\b",
    re.IGNORECASE)
def mask_dates(text: str) -> str:
    return _DATE_TIME_RE.sub(lambda m: " " * len(m.group(0)), str(text or ""))


#: Identifiers are tokens, not quantities: an IPv4 address, a dotted version
#: of three or more parts, a hex id carrying both letters and digits (task
#: ids, hashes), a UUID. `127.0.0.2` read as 127.0 hid a swapped last octet
#: (mined rec-ca993114e7); as a token it is looked up verbatim.
#: A standards citation ("IEEE P2851", "ISO 9001", "RFC 7231", "IEEE 802.11",
#: "CVE-2024-1234") is an identifier, not a name and not a figure: looked up
#: verbatim, UNSUPPORTED withholds a confirm, a same-skeleton twin refutes.
#: A closed list of citation prefixes — the acronym-body fabrication the
#: §4IQ bench class injects was the binder's one remaining false confirm
#: (6/27 mined), and the entity audit must not admit acronyms (labels).
_STANDARD_PREFIX = r"(?-i:IEEE|ISO|IEC|RFC|ANSI|ASTM|ITU|ETSI|NIST|DIN|CVE|EN|BS|JIS|PEP)"   # capitals only ("en 13" is English)
_IDENT_RE = re.compile(
    r"\b\d{1,3}(?:\.\d{1,3}){3}\b"
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    r"|\bv?\d+(?:\.\d+){2,}\b"
    rf"|\b{_STANDARD_PREFIX}[ \t-]?[A-Z]?\d{{2,}}(?:[.:-]\d+)*(?!\w)(?!\.\w)"     # a sentence-final stop may follow
    r"|\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{8,}\b", re.IGNORECASE)


def mask_identifiers(text: str) -> str:
    return _IDENT_RE.sub(lambda m: " " * len(m.group(0)), str(text or ""))


#: `grep -n` / `cat -n` line numbers ("173:const channel = …", "  42\tfoo",
#: "12| bar") are evidence FORMAT, not figures.
_LINE_NUMBER_RE = re.compile(r"(?m)^[ \t]*\d{1,6}(?:[:|→]|\t| {2,})")


def mask_line_numbers(text: str) -> str:
    return _LINE_NUMBER_RE.sub(lambda m: " " * len(m.group(0)), str(text or ""))


def mask_non_quantities(text: str) -> str:
    # dates and clocks first: "12:30 meeting" is a clock, not line 12
    return mask_identifiers(mask_line_numbers(mask_dates(text)))


def _is_bare_year(v: float, unit: str, decimals: int, raw: str = "") -> bool:
    """A unitless four-digit integer in the calendar range ("May 6, 2492",
    "in 2026") is a year, not a quantity; "2048 bytes" keeps its unit and
    stays a quantity, and a comma-grouped "2,000" is a count, never a year."""
    # NOT 1800 (§4IS-b tried it): "total_orders 1847 (prev 1649)" lost its
    # 1847 to the year rule and the reply's 1,847 was refuted against 1649 —
    # a bare 1800s number is a count as often as a year, so it stays a
    # figure here AND is looked up by `audit_years` (1800–2999) as a year.
    return not unit and decimals == 0 and 1900 <= v <= 2999 and "," not in raw


def _scale(raw: str, unit: str, cur: str) -> Optional[Tuple[float, str, int]]:
    """(value in the family base, family, decimals) for one written figure;
    None when it is not a number or is a bare year."""
    if re.fullmatch(r"[-+]?\d+,\d{1,2}", str(raw or "")):
        num_txt = raw.replace(",", ".")      # "4,3" / "23,60": a decimal comma
    else:
        num_txt = raw.replace(",", "")
    if len(num_txt.lstrip("-+")) > 1 and num_txt.lstrip("-+").startswith("0") and "." not in num_txt:
        return None                      # "000": a thousands-group remnant left by a mask, never a written figure
    try:
        v = float(num_txt)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None                      # a 309-digit print(2**1024) is not a figure (review §4IY: OverflowError took the binder out for the turn)
    decimals = len(num_txt.split(".")[1]) if "." in num_txt else 0
    if _is_bare_year(v, unit, decimals, raw) and not cur:
        return None
    if cur:
        return v * _MONEY_MULT.get(unit, 1.0), "money", decimals
    fam, factor = _UNITS.get(unit, ("", 1.0))
    if unit in ("million", "billion", "thousand", "bn", "mn"):
        return v * _MONEY_MULT[unit], "", decimals
    if fam == "thousand":        # "12k" is a unitless 12000
        return v * factor, "", decimals
    return v * factor, fam, decimals


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


_LOWER_BOUND_RE = re.compile(r"(?:\bover|\bmore than|\bat least|\babove|\bexceed(?:s|ing)?|\bupwards of|>=?|≥)\s*$", re.I)
_UPPER_BOUND_RE = re.compile(r"(?:\bunder|\bless than|\bfewer than|\bat most|\bbelow|\bup to|\bno more than|\bwithin|<=?|≤)\s*$", re.I)


def _bound_at(text: str, start: int, end: int) -> str:
    """A figure's bound marker from its immediate context: the words before
    it ("over 160", "at least 3") or a "+" right after it ("160+"). A bound
    is not a measurement — "Over 160+ reviews" against "164 reviews" is
    agreement, never a misreport (corpus replay)."""
    before = text[max(0, start - 16):start]
    after = text[end:end + 1]
    if after == "+" or _LOWER_BOUND_RE.search(before):
        return "lower"
    if _UPPER_BOUND_RE.search(before):
        return "upper"
    return ""


def extract_quantities_with_pos(text: str) -> List[Tuple[Quantity, int]]:
    """Every quantity in `text` with its character offset, ranges first
    (a range's endpoints are never also reported as two scalars)."""
    masked = mask_non_quantities(_fold_spaces(text))
    found: List[Tuple[Quantity, int]] = []
    for m in _RANGE_RE.finditer(masked):
        conn = m.group("conn").strip().lower()
        if conn == "and" and not m.group("between"):
            continue                                   # "5 and 7 tasks" — two figures, not a range
        u1, u2 = (m.group("unit1") or "").lower(), (m.group("unit") or "").lower()
        if u1 and u2 and u1 != u2:
            continue
        unit, cur = (u2 or u1), (m.group("cur") or m.group("cur2") or "")
        lo, hi = _scale(m.group("lo"), unit, cur), _scale(m.group("hi"), unit, cur)
        if lo is None or hi is None or lo[1] != hi[1] or lo[0] > hi[0]:
            continue
        found.append((Quantity(lo[0], lo[1], max(lo[2], hi[2]), m.group(0).strip(), unit, hi[0]),
                      m.start()))
        masked = _blank(masked, m.start(), m.end())
    scalars: List[Tuple[Quantity, int, int]] = []
    for m in _NUM_RE.finditer(masked):
        unit, cur = (m.group("unit") or "").lower(), m.group("cur") or ""
        # "1h 30m": an "m" right after an hour figure is minutes, not metres (review §4IY)
        if unit == "m" and scalars and scalars[-1][0].family == "time" and masked[scalars[-1][2]:m.start()].strip() in ("", "and"):
            unit = "min"
        sc = _scale(m.group("num"), unit, cur)
        if sc is None:
            continue
        q = Quantity(sc[0], sc[1], sc[2], m.group(0).strip(), unit, bound=_bound_at(masked, m.start(), m.end()))
        # a compound duration ("2 hours 30 min") is ONE quantity: merge adjacent time figures
        # separated by whitespace or "and" when the second's unit is finer (review §4IY)
        if (scalars and q.family == "time" and scalars[-1][0].family == "time" and q.unit and scalars[-1][0].unit
                and masked[scalars[-1][2]:m.start()].strip() in ("", "and")
                and _UNITS.get(q.unit, ("", 0))[1] < _UNITS.get(scalars[-1][0].unit, ("", 0))[1]):
            prev, pstart, _pend = scalars[-1]
            merged = Quantity(prev.value + q.value, "time", max(prev.decimals, q.decimals),
                              masked[pstart:m.end()].strip(), prev.unit, bound=prev.bound)
            scalars[-1] = (merged, pstart, m.end())
            continue
        scalars.append((q, m.start(), m.end()))
    found.extend((q, st) for q, st, _e in scalars)
    found.sort(key=lambda p: p[1])
    return found


def extract_quantities(text: str) -> List[Quantity]:
    return [q for q, _ in extract_quantities_with_pos(text)]


_HEDGE_RE = re.compile(r"(?<![\w/])~\s*\d|≈\s*\d|\b(?:about|approximately|approx|around|roughly|nearly|almost|circa)\b|\bclose to\b", re.I)


def _hedged(text: str) -> bool:
    """A hedge WORD, or "~"/"≈" right before a figure — "~/Data" and
    "roundabout" are not hedges."""
    return bool(_HEDGE_RE.search(str(text or "")))


def _compare_factor(claim_q: Quantity, span_q: Quantity) -> float:
    """Figures are compared in the CLAIM's own unit at the claim's decimals:
    "48 KB" vs 49152 bytes → 48.0 vs 48.0; "0.04s" vs "0.041s" → 0.04 vs
    0.04. Unitless and non-convertible families compare raw."""
    if claim_q.family and claim_q.family == span_q.family:      # mass too: "2 kg" vs "1.95 kg" rounds in kg, not grams (review §4IY)
        return _claim_unit_factor(claim_q)
    return 1.0


def _scalars_agree(a: float, b: float, *, decimals: int, factor: float, hedged: bool) -> bool:
    if hedged:
        base = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / base <= HEDGE_REL_TOL
    return _round_half_up(b / factor, decimals) == _round_half_up(a / factor, decimals)


def _round_half_up(x: float, decimals: int) -> float:
    if not math.isfinite(x):
        return x
    q = 10 ** decimals
    return math.floor(abs(x) * q + 0.5) / q * (1 if x >= 0 else -1)


def _within(lo: float, hi: float, v: float, *, decimals: int, factor: float, hedged: bool) -> bool:
    """`v` lies inside [lo, hi] at the given precision (a hedged claim widens
    the range by HEDGE_REL_TOL of its magnitude)."""
    if hedged:
        tol = HEDGE_REL_TOL * max(abs(lo), abs(hi), 1e-9)
        return lo - tol <= v <= hi + tol
    rv = _round_half_up(v / factor, decimals)
    return _round_half_up(lo / factor, decimals) <= rv <= _round_half_up(hi / factor, decimals)


_DECIMAL_BYTE_UNITS = {"kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12}


def _si_alias(q: Quantity) -> Optional[Quantity]:
    """The same figure read with decimal byte units ("1.5 MB" = 1,500,000
    bytes as well as 1,572,864); kib/mib/gib stay binary (review §4IY)."""
    if q.family == "bytes" and q.unit in _DECIMAL_BYTE_UNITS:
        base = _UNITS[q.unit][1]
        f = _DECIMAL_BYTE_UNITS[q.unit] / base
        return Quantity(q.value * f, q.family, q.decimals, q.text, q.unit, None if q.hi is None else q.hi * f, q.bound)
    return None


def quantities_agree(claim_q: Quantity, span_q: Quantity, *, hedged: bool) -> bool:
    """Same family (or both unitless) and the span's value ROUNDS to the
    claim's figure at the claim's precision; a hedged claim tolerates
    ±HEDGE_REL_TOL instead. A range agrees with a value inside it, and with
    a range whose two endpoints agree. Decimal byte units agree under 1000ⁿ
    as well as 1024ⁿ."""
    if _quantities_agree(claim_q, span_q, hedged=hedged):
        return True
    for a, b in ((_si_alias(claim_q), span_q), (claim_q, _si_alias(span_q))):
        if a is not None and b is not None and _quantities_agree(a, b, hedged=hedged):
            return True
    return False


def _quantities_agree(claim_q: Quantity, span_q: Quantity, *, hedged: bool) -> bool:
    if claim_q.family != span_q.family and claim_q.family and span_q.family:
        return False
    f, d = _compare_factor(claim_q, span_q), claim_q.decimals
    if claim_q.bound == "lower" and not span_q.is_range:
        return span_q.value >= claim_q.value
    if claim_q.bound == "upper" and not span_q.is_range:
        return span_q.value <= claim_q.value
    if claim_q.is_range and span_q.is_range:
        return (_scalars_agree(claim_q.value, span_q.value, decimals=d, factor=f, hedged=hedged)
                and _scalars_agree(claim_q.hi, span_q.hi, decimals=d, factor=f, hedged=hedged))
    if claim_q.is_range:
        return _within(claim_q.value, claim_q.hi, span_q.value, decimals=d, factor=f, hedged=hedged)
    if span_q.is_range:
        return _within(span_q.value, span_q.hi, claim_q.value, decimals=d, factor=f, hedged=hedged)
    return _scalars_agree(claim_q.value, span_q.value, decimals=d, factor=f, hedged=hedged)


def _claim_unit_factor(q: Quantity) -> float:
    if q.family == "money":
        return _MONEY_MULT.get(q.unit, 1.0)
    return _UNITS.get(q.unit, ("", 1.0))[1] or 1.0


_NAME_STOP = frozenset("""
the a an of and or to in for on with by from at is are was were be been being as that this these
those it its into about over under how what when where which who why not no do does did can could
should would will may might must all any some each every has have had than then there here
και του της των τον την τους τις στο στη στην στον στα στις στους για από με που είναι ήταν
έχει έχουν αυτό αυτή αυτά ένα μια μία δεν όχι θα να ως πως ότι όπως επίσης ενώ αλλά μετά πριν
κατά προς υπό ακόμη ακόμα πολύ εδώ εκεί μέσω όταν όπου οποία οποίο οποίος αυτού αυτής
""".split())
#: The anchor test needs more function words than a NAME does ("Thomas More",
#: "Bill Such" keep their surnames — review §4IY): the extra ones live here.
_ANCHOR_STOP = _NAME_STOP | frozenset("""
they their them theirs only also more most very such same other others after before because during
while where until since about above below between through against without within around across
είχε είχαν μόνο κάθε όλοι όλες όλα όλο όλη τότε τώρα άλλο άλλη άλλος άλλοι αυτές αυτοί γιατί χωρίς
μέσα πάνω κάτω ώστε έτσι εκείνος εκείνη εκείνο μετά πριν επειδή ενώ όμως αφού μέχρι σχεδόν περίπου
""".split())
# ↑ review §4IX: "They reported 34 cases" anchored on "They found 35 issues"
# ↑ Greek function words (§4IU, req 2ef4f0a2): the anchor matched "στην" between
# a reply sentence about Σπήλιος and a search line about an army officer, and
# the misreport rule then refuted his birth year against the officer's.


_NEGATION_RE = re.compile(r"\b(?:not|no|never|none|failed|failure|fails|error|errors|cannot|can't|unable|missing|denied|refused|rejected|timed out|timeout|exception|traceback"
                          r"|δεν|όχι|απέτυχ\w*|αποτυχία|σφάλμα|λάθος|αδύνατ\w*|ανεπιτυχ\w*)\b", re.I)


def _polarity_clash(claim_quote: str, span: str) -> bool:
    """A status claim and a span that shares its subject but carries a
    failure/negation word the claim does not ("All tests passed" against
    "3 failed, 0 passed", review §4IN m1) — code cannot call that agreement."""
    return bool(_NEGATION_RE.search(span)) and not _NEGATION_RE.search(claim_quote)


def lexical_anchor(claim_quote: str, span: str) -> bool:
    """A non-numeric claim is anchored to its span when one of its content
    words (≥4 chars, not a stopword) occurs in the span — "RECOVERED" is not
    anchored by "required file missing"; "server restarted" is anchored by
    "restarted ghost-agent"."""
    words = [w for w in re.findall(r"[^\W_]+", normalize_for_containment(claim_quote))
             if len(w) >= 4 and w not in _ANCHOR_STOP]
    if not words:
        return False
    hay = normalize_for_containment(span)
    # a WORD or a ≥5-letter stem at a word start — "ready" is not in "already", "test" is not in
    # "latest" (review §4IY: both confirmed a status claim against an unrelated line)
    hay_words = set(re.findall(r"[^\W_]+", hay))          # "total_orders" carries the word "orders"
    return any(w in hay_words or (len(w) >= 6 and any(h.startswith(w[:5]) for h in hay_words)) for w in words)


def compare_claim_span(claim_quote: str, span: str) -> Tuple[str, str]:
    """-> (outcome, detail) for a bound pair. A claim carrying quantities:
    "agree" when every claim quantity finds an agreeing span quantity,
    "disagree" when one matches a span quantity's family but not its value.
    A claim without quantities: "agree" when lexically anchored to the span,
    else "unchecked" — code cannot grade a status word; the model's opinion
    is not a verdict."""
    outcome, detail, _q, _s = _compare(claim_quote, span)
    return outcome, detail


def _compare(claim_quote: str, span: str) -> Tuple[str, str, Optional[Quantity], Optional[Quantity]]:
    """`compare_claim_span` plus the claim figure that disagreed and the span
    figure it was measured against (None otherwise), for the guards in
    `bind`."""
    cq = extract_quantities(claim_quote)
    if not cq:
        if lexical_anchor(claim_quote, span) and not _polarity_clash(claim_quote, span):
            return "agree", "", None, None
        return "unchecked", "no comparable figure", None, None
    sq = extract_quantities(span)
    if not sq:
        return "unchecked", "span carries no quantity", None, None
    hedged = _hedged(claim_quote)
    supported = 0
    for q in cq:
        # comparable = the same unit family; a unit-bearing claim figure is
        # never compared with a bare number ("4TB" is not "289.90")
        comparable = [s for s in sq if s.family == q.family]
        if not comparable:
            continue                                   # unknown, not a contradiction
        if any(quantities_agree(q, s, hedged=hedged) for s in comparable):
            supported += 1
            continue
        near = [s for s in comparable if _near_miss(q, s)]
        if not near:
            continue                                   # a different figure of the same family (a bound, a core count)
        best = min(near, key=lambda s: abs(s.value - q.value))
        return "disagree", f"claim says {q.text!r}, evidence says {best.text!r}", q, best
    if supported:
        return "agree", "", None, None
    return "unchecked", "no claim figure had a comparable evidence figure", None, None


def _claim_states(claim_quote: str, span_fig: Optional[Quantity]) -> bool:
    """The span figure the disagreement was measured against is one the
    CLAIM states itself ("the ball at x=365 … left of the wall at x=360"
    bound to `channel = { x: 360 }`): the claim knows both numbers, so its
    other figure is another quantity, not a misreading (mined
    rec-ebc239c1ca, clean). Only THAT figure — "100" near "76" in a list of
    bumper values must not trip it."""
    if span_fig is None:
        return False
    return _slot_text(span_fig) in {_slot_text(c) for c in extract_quantities(claim_quote)}


def _dense(q: Optional[Quantity], text: str) -> bool:
    """Three or more figures of the claim figure's family in one text — a
    record (a struct literal, coordinates, a spec line)."""
    if q is None:
        return False
    return sum(1 for s in extract_quantities(text) if s.family == q.family) >= 3


def _aligned(claim_quote: str, span: str) -> bool:
    """The claim's non-numeric skeleton is the span's, contains it, or is
    contained in it (a near-verbatim repetition — the vision description
    the reply copied with one figure swapped; "meta=335" against "Topic
    clusters: meta=334, coding=241, …"). In a dense span only an aligned
    claim's one-digit slip singles out a quantity: "ball: x=365, y=560,
    r=8" against `const plunger = { x: 375, y: 560, w: 12, h: 40, … }` is
    two records, not one misread (mined rec-ebc239c1ca, clean)."""
    import difflib
    a, b = _skeleton(claim_quote), _skeleton(span)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.8


def _single_comparable(q: Optional[Quantity], span: str) -> bool:
    """A shared subject word settles WHICH figure the claim's corresponds
    to only when the span carries exactly one figure of that family:
    "center the ball in the channel (x=372)" bound to `const channel = { x:
    360, w: 20 }` shares "channel" and still compares a centre with a left
    edge (mined rec-ebc239c1ca, a clean reply refuted). Several comparable
    figures need a typo-shaped pair instead."""
    if q is None:
        return False
    return sum(1 for s in extract_quantities(span) if s.family == q.family) == 1


def figure_elsewhere(q: Quantity, evidence: str, *, hedged: bool) -> Optional[str]:
    """The evidence line, other than the bound span, that carries a figure
    AGREEING with the claim's — the binder bound the wrong reading ("35°C"
    bound to "temperature 34°C" while the same tool printed "feels like
    35°C"). None when no line agrees."""
    for ln in str(evidence or "").splitlines():
        if any(quantities_agree(q, s, hedged=hedged) for s in extract_quantities(_strip_label(ln))):
            return ln.strip()
    return None


def _typo_shaped_disagreement(claim_quote: str, span: str) -> bool:
    """True when some claim figure and a same-family span figure, written
    in the claim's unit at the claim's precision, have the same length and
    differ in exactly one digit — the shape of a swapped or misread digit
    rather than of two different quantities."""
    def one_digit_apart(x: float, y: float, f: float, dec: int) -> bool:
        a, b = f"{abs(x) / f:.{dec}f}", f"{abs(y) / f:.{dec}f}"
        if sum(ch.isdigit() for ch in a) < 2:
            return False                   # a single digit is one digit away from every other digit
        return len(a) == len(b) and a != b and sum(p != r for p, r in zip(a, b)) == 1

    for q in extract_quantities(claim_quote):
        f = _claim_unit_factor(q)
        for sq in extract_quantities(span):
            if sq.family != q.family or sq.is_range != q.is_range:
                continue                   # a scalar is never a misread range
            if sq.decimals != q.decimals and q.family not in ("bytes", "time", "length", "money", "mass"):
                continue                   # a misread digit keeps the written shape: "19" is not a slip of "17.10" (corpus replay)
            if q.is_range:
                # one endpoint equal, the other a digit off: "20–25" vs "20–26"
                same_lo = f"{q.value / f:.{q.decimals}f}" == f"{sq.value / f:.{q.decimals}f}"
                same_hi = f"{q.hi / f:.{q.decimals}f}" == f"{sq.hi / f:.{q.decimals}f}"
                if ((same_lo and one_digit_apart(q.hi, sq.hi, f, q.decimals))
                        or (same_hi and one_digit_apart(q.value, sq.value, f, q.decimals))):
                    return True
                continue
            if one_digit_apart(q.value, sq.value, f, q.decimals):
                return True
    return False


def _near_miss(q: Quantity, s: Quantity) -> bool:
    """A span figure close enough to the claim's to be the SAME quantity
    misreported (34 vs 35, 9,692 vs 9,592, 19 KB vs 18,433 B) rather than a
    different quantity that happens to share a family (a bound of 100,000
    beside a count of 9,592; 10 cores beside a load of 1.61)."""
    if q.is_range != s.is_range:
        return False                       # a value outside a range is unknown, never a misreport of it
    def close(x: float, y: float) -> bool:
        x, y = abs(x), abs(y)
        if x == 0 or y == 0:
            return x == y
        return max(x, y) / min(x, y) <= 1.5
    if q.is_range:
        return close(q.value, s.value) and close(q.hi, s.hi)
    return close(q.value, s.value)


# ── conflicting evidence (skeleton scan) ────────────────────────────────

_PACKER_LABEL_RE = re.compile(r"^\s*\[[\w .\-]{1,40}\]\s*")


def _strip_label(line: str) -> str:
    """The evidence packer opens a tool's first line with `[tool_name] `; a
    second row of the same output carries no label. Compare bodies."""
    return _PACKER_LABEL_RE.sub("", str(line or ""), count=1)


def _skeleton(line: str) -> str:
    body = normalize_for_containment(_strip_label(line))
    body = _LINE_NUMBER_RE.sub("#", _DATE_TIME_RE.sub("#", _IDENT_RE.sub("#", body)))
    return re.sub(r"\s+", " ", _NUM_RE.sub("#", body)).strip()


def _slot_text(q: Quantity) -> str:
    return f"{q.value:.10g}" + (f"-{q.hi:.10g}" if q.is_range else "")


def _slots(text: str) -> List[str]:
    """The line's values in order — identifiers, dates, clocks and line
    numbers as tokens, quantities as normalized numbers — so two readings
    of one field compare slot by slot. Every token the skeleton wildcards
    is a slot: two rows of an hourly forecast ("14:00 31°C" / "15:00 32°C")
    differ in TWO slots and are two records, not one field read twice
    (review §4IN M1 — a live false refute class)."""
    body = _fold_spaces(_strip_label(text))
    found: List[Tuple[int, str]] = [(m.start(), m.group(0).lower()) for m in _IDENT_RE.finditer(body)]
    no_ids = mask_identifiers(body)
    found += [(m.start(), m.group(0).lower()) for m in _DATE_TIME_RE.finditer(no_ids)]
    found += [(m.start(), m.group(0).strip().lower()) for m in _LINE_NUMBER_RE.finditer(mask_dates(no_ids))]
    found += [(pos, _slot_text(q)) for q, pos in extract_quantities_with_pos(body)]
    return [t for _, t in sorted(found)]


def find_conflicting_line(evidence: str, span: str, claim_quote: str, *,
                          reply_slots: Optional[set] = None) -> Optional[str]:
    """Another evidence line with the SAME non-numeric skeleton as the
    bound span's line and a DIFFERENT value where the claim's figure sits —
    two readings of one field. None when the evidence has no such line, and
    None when the reply states the twin's value too (`reply_slots`): a reply
    listing both project ids reports two records, it omits nothing."""
    ev = str(evidence or "")
    nspan = normalize_for_containment(span)
    if not nspan:
        return None
    lines = [ln for ln in ev.splitlines() if ln.strip()]
    home = next((ln for ln in lines if nspan in normalize_for_containment(ln)), None)
    if home is None:
        return None
    home_sk = _skeleton(home)
    home_slots = _slots(home)
    if not home_slots or home_sk.count("#") == 0:
        return None
    if not re.search(r"[a-z]{3}", home_sk):
        return None                    # a bare-number line (a chart's DOM dump) names no field
    if _META_LINE_RE.match(_strip_label(home)):
        return None                    # a per-block packer/tool meta line repeats across blocks
    if sum(1 for ln in lines if _skeleton(ln) == home_sk) >= 3:
        return None                    # an enumeration (table rows, "tick 1"… "tick 40", LTS releases): records, not readings
    claim_slots = set(_slots(claim_quote))
    home_parent = _parent_line(lines, home)
    for ln in lines:
        if ln is home or normalize_for_containment(_strip_label(ln)) == normalize_for_containment(_strip_label(home)):
            continue
        if _skeleton(ln) != home_sk:
            continue
        if home_parent is not None and _parent_line(lines, ln) != home_parent:
            continue                   # `port:` under `api:` and under `db:` are two records (review §4IY)
        slots = _slots(ln)
        if len(slots) != len(home_slots):
            continue
        diff = [i for i, (x, y) in enumerate(zip(home_slots, slots)) if x != y]
        # ONE slot differs and it is the slot the claim reported: two readings
        # of one field. Rows of a table differ in most slots — not a conflict.
        if len(diff) == 1 and home_slots[diff[0]] in claim_slots:
            if reply_slots is not None and (slots[diff[0]] in reply_slots or _stated_at_precision(slots[diff[0]], reply_slots)):
                continue
            return ln.strip()
    return None


_META_LINE_RE = re.compile(r"^\s*(?:LENGTH|EXIT CODE|HTTP_STATUS|STATUS|TRUNCATED|ELAPSED)\s*:", re.I)


def _parent_line(lines: List[str], line: str) -> Optional[str]:
    """The nearest preceding line with a smaller indentation (a YAML / JSON
    / TOML parent key), or None when `line` is not indented — then every
    same-skeleton line is a candidate reading."""
    indent = len(line) - len(line.lstrip(" \t"))
    if indent == 0:
        return None
    try:
        idx = next(i for i, ln in enumerate(lines) if ln is line)
    except StopIteration:
        return None
    for prev in reversed(lines[:idx]):
        if prev.strip() and len(prev) - len(prev.lstrip(" \t")) < indent:
            return prev.strip()
    return None


def _stated_at_precision(slot: str, reply_slots: set) -> bool:
    """The twin's value is one the reply states at ITS precision: 1.414 for
    1.4142135623730951 (corpus replay: the phi row)."""
    try:
        v = float(slot)
    except ValueError:
        return False
    for r in reply_slots:
        try:
            rv = float(r)
        except ValueError:
            continue
        dec = len(r.split(".")[1]) if "." in r else 0
        if _round_half_up(v, dec) == _round_half_up(rv, dec):
            return True
    return False


def reply_slots_of(reply: str) -> set:
    """Every value the reply states (identifiers and figures; a range counts
    as itself AND as each end), for the reports-both guard of the conflict
    scan."""
    out: set = set()
    for text in (reply, _mask_non_prose(reply)):
        for t in _slots(text):
            out.add(t)
            if "-" in t.lstrip("-"):
                lo, hi = t.lstrip("-").split("-", 1)
                out.update({("-" if t.startswith("-") else "") + lo, hi})
    return out


# ── implausible labelled values ─────────────────────────────────────────

_LABELLED_RANGES = (
    (re.compile(r"\b(?:lat|latitude)\b\s*[=:]?\s*([-+]?\d{1,3}(?:\.\d+)?)(?!\d)(?!\s*(?:ms|us|µs|s|sec|secs|seconds?|min|minutes?)\b)", re.I), -90.0, 90.0, "latitude"),   # "p99 lat 250 ms" is latency (review §4IY)
    (re.compile(r"\b(?:lon|lng|longitude)\b\s*[=:]?\s*([-+]?\d{1,3}(?:\.\d+)?)(?!\d)", re.I), -180.0, 360.0, "longitude"),
)


def implausible_value(text: str) -> Optional[str]:
    for rx, lo, hi, label in _LABELLED_RANGES:
        for m in rx.finditer(str(text or "")):
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            if v < lo or v > hi:
                return f"{label} {m.group(1)} is outside [{lo:g}, {hi:g}]"
    return None


# ── parsing the model's rows ────────────────────────────────────────────

_ROW_RE = re.compile(r"\{[^{}]*\}")


def _salvage_rows(text: str) -> List[Dict[str, Any]]:
    """A payload cut mid-list (token cap) still yields every COMPLETE row:
    decode an object at each `{` with the JSON decoder itself, so a quote
    that contains braces (`cfg = {a: 1}`) survives (review §4IN nit)."""
    out: List[Dict[str, Any]] = []
    t = str(text or "")
    dec = json.JSONDecoder()
    i = t.find("{")
    while i >= 0:
        try:
            obj, end = dec.raw_decode(t, i)
        except Exception:  # noqa: BLE001
            i = t.find("{", i + 1)
            continue
        if isinstance(obj, dict) and obj.get("quote"):
            out.append(obj)
            i = t.find("{", i + max(1, end - i))
        elif isinstance(obj, dict) and isinstance(obj.get("claims"), list):
            out.extend(r for r in obj["claims"] if isinstance(r, dict) and r.get("quote"))
            i = t.find("{", i + max(1, end - i))
        else:
            i = t.find("{", i + 1)
    return out


def parse_binder_output(data: Any) -> List[Dict[str, str]]:
    """Coerce the binder's JSON into rows; tolerate a str payload (whole or
    truncated), a list payload, missing fields and unknown kinds/relations."""
    if isinstance(data, str):
        s = data.strip()
        try:
            data = json.loads(s[s.index("{"):s.rindex("}") + 1])
        except Exception:  # noqa: BLE001 — truncated or fenced: salvage complete rows
            data = {"claims": _salvage_rows(s)}
    rows = data.get("claims") if isinstance(data, dict) else data
    out: List[Dict[str, str]] = []
    if not isinstance(rows, list):
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        q = str(r.get("quote") or "").strip()
        if not q:
            continue
        kind = str(r.get("kind") or "other").strip().lower()
        rel = str(r.get("relation") or "absent").strip().lower()
        span = str(r.get("evidence_quote") or "").strip()
        if len(q) > HARD_QUOTE_CAP or len(span) > HARD_SPAN_CAP:
            continue                                   # not a quote, a dump
        out.append({"quote": q,
                    "kind": kind if kind in CLAIM_KINDS else "other",
                    "evidence_quote": span,
                    "relation": rel if rel in RELATIONS else "absent"})
        if len(out) >= MAX_CLAIMS:
            break
    return out


# ── binding + verdict ───────────────────────────────────────────────────

_LEAD_WORD_RE = re.compile(r"(?:\b(?:over|under|above|below|at\s+least|at\s+most|more\s+than|less\s+than|up\s+to|about|approximately|approx\.?|around|roughly|nearly|almost|circa|~|≈)\s*)$", re.I)


def _with_reply_lead(quote: str, reply: str) -> str:
    """The quote with the bound/hedge word that immediately precedes it in
    the reply, when the quote itself starts with the figure."""
    q = str(quote or "")
    if not q or not re.match(r"[-+~≈€$£]?\d", q):
        return q
    nr = normalize_for_containment(reply)
    i = nr.find(normalize_for_containment(q))
    if i <= 0:
        return q
    m = _LEAD_WORD_RE.search(nr[max(0, i - 16):i])
    return (m.group(0) + q) if m else q


def bind(reply: str, evidence: str, rows: List[Dict[str, str]]) -> Tuple[List[Binding], int]:
    """Validate every row against the texts and grade each bound pair.
    Returns (bindings, dropped) — dropped = rows whose claim quote was not
    in the reply (a hallucinated claim is not a claim)."""
    bindings: List[Binding] = []
    dropped = 0
    rslots = reply_slots_of(reply)
    for r in rows:
        b = Binding(r["quote"], r["kind"], r["evidence_quote"], r["relation"])
        snapped_claim = snap_quote(b.quote, reply)
        b.valid_claim = snapped_claim is not None
        if not b.valid_claim:
            dropped += 1
            continue
        if snapped_claim != normalize_for_containment(b.quote):
            b.quote = snapped_claim                    # the reply's own words, not the model's rendering
        imp = implausible_value(b.quote)
        if imp:
            b.outcome, b.detail = "implausible", imp
            bindings.append(b)
            continue
        snapped_span = snap_quote(b.evidence_quote, evidence) if b.evidence_quote else None
        b.valid_span = snapped_span is not None
        if not b.valid_span:
            b.outcome = "unbound"
            bindings.append(b)
            continue
        if snapped_span != normalize_for_containment(b.evidence_quote):
            b.evidence_quote = snapped_span            # the evidence's own words
        # "over 160 reviews" quoted as "160 reviews", "around 28%" as "28% today": the bound or
        # hedge sits one word before the quote in the REPLY (review §4IY)
        outcome, detail, dq, ds = _compare(_with_reply_lead(b.quote, reply), b.evidence_quote)
        compared = outcome
        elsewhere = figure_elsewhere(dq, evidence, hedged=_hedged(b.quote)) if dq is not None else None
        if outcome == "disagree" and elsewhere:
            # the claim's own figure stands in another evidence line: the
            # reply may be reporting THAT reading and the binder bound the
            # wrong one — not a contradiction, not a confirmation
            outcome, detail = "unchecked", f"figures differ ({detail}) but the evidence also states {elsewhere[:120]!r}"
        elif (outcome == "disagree"
                and not (lexical_anchor(b.quote, b.evidence_quote) and _single_comparable(dq, b.evidence_quote))
                and not (_typo_shaped_disagreement(b.quote, b.evidence_quote)
                         and (not _dense(dq, b.evidence_quote) or _aligned(b.quote, b.evidence_quote))
                         and not _claim_states(b.quote, ds))):
            # The figures differ but the two quotes share no subject word and
            # the figures are not a one-digit slip of each other — most
            # likely two different quantities ("tonight drops to 24°C" bound
            # to "current conditions: temperature 31°C", seed long-weather-1;
            # "All 7 tasks finished" bound to "batch ran 5 task(s)" while the
            # ledger listed seven, mined rec-a6e2b9b9b6). The model's
            # `relation` label is an opinion and never turns this into a
            # verdict: a disagreement needs a shared subject or a typo-shaped
            # pair (9,692 vs 9,592; 2027 vs 2026; 22 GB vs 21 GB).
            outcome, detail = "unchecked", f"figures differ ({detail}) but the quotes share no subject, or the span carries several comparable figures"
        if outcome == "agree" and b.relation == "contradict":
            # code and model disagree (the figures match, the model says
            # they conflict) — not a verdict either way in phase 1
            outcome, detail = "unchecked", "model says contradict, figures agree"
        elif outcome == "agree":
            conflict = find_conflicting_line(evidence, b.evidence_quote, b.quote, reply_slots=rslots)
            if conflict:
                outcome, detail = "conflict", f"evidence also states: {conflict[:160]!r}"
        elif compared == "unchecked" and b.relation == "contradict":
            # the model says the span contradicts but neither side carries a
            # figure code can compare (a status word) — recorded for the
            # shadow read-through, never a verdict in phase 1
            outcome, detail = "unchecked", "model-stated contradiction without a comparable figure"
        b.outcome, b.detail = outcome, detail
        bindings.append(b)
    return bindings, dropped


def verdict_from_bindings(bindings: List[Binding], *, evidence_truncated: bool,
                          dropped: int = 0, audit: Optional[List["AuditFigure"]] = None,
                          entities: Optional[List["AuditEntity"]] = None,
                          strict_figures: bool = False,
                          findings: Optional[List["ClassFinding"]] = None
                          ) -> ClaimBindingResult:
    """The mechanical verdict (see the module docstring). `audit` rows from
    `audit_numbers`: a MISREPORTED figure is a validated contradiction (it
    refutes); UNSUPPORTED figures are counted in the reasoning only —
    derived counts and conversions live there (phase 2 decides). `entities`
    rows from `audit_entities`: a named entity the evidence and context
    never mention WITHHOLDS a confirm (the evidence does not cover the
    reply) and never refutes (the reply may know it); so does an unsupported
    identifier, and — with `strict_figures` — an unsupported figure."""
    audit = list(audit or [])
    entities = list(entities or [])
    findings = list(findings or [])
    unsupported_entities = [e.text for e in entities if e.status == "unsupported"]
    ent_note = (f"; {len(unsupported_entities)} named entit{'y' if len(unsupported_entities) == 1 else 'ies'} "
                f"not in the evidence: {', '.join(repr(t) for t in unsupported_entities[:3])}") if unsupported_entities else ""
    issues: List[str] = []
    for b in bindings:
        if b.outcome == "disagree" and b.residual:
            issues.append(f"claim {b.quote!r} is contradicted by the evidence: {b.evidence_quote!r}")
        elif b.outcome == "disagree":
            issues.append(f"claim {b.quote!r} vs evidence {b.evidence_quote!r}: {b.detail}")
        elif b.outcome == "conflict":
            issues.append(f"claim {b.quote!r} reports one of two evidence values — {b.detail}")
        elif b.outcome == "implausible":
            issues.append(f"claim {b.quote!r}: {b.detail}")
    for g in findings:
        if g.status == "refute":
            # a constraint issue is spelled exactly as the turn loop's own
            # mechanical tier spells it ("word_cap: …"), so every reader that
            # recognises a delivery-shape refute (never a project task, never
            # a correction banner, the reshape directive) recognises this one
            # (review §4IN consumer M2)
            issues.append(g.detail if g.kind == "constraint" else f"{g.kind}: {g.detail}")
    seen_texts = {b.quote for b in bindings if b.outcome in ("disagree", "conflict")}
    seen_twins = {b.detail for b in bindings if b.outcome == "conflict"}
    for f in audit:
        if f.status == "misreported" and not any(f.text in q for q in seen_texts):
            issues.append(f"figure {f.text!r} in {f.sentence[:100]!r} — the evidence says {f.evidence_text!r} "
                          f"({f.evidence_line[:100]!r})")
        elif f.status == "conflicted" and not any(f.evidence_line[:80] in d for d in seen_twins):
            seen_twins.add(f.evidence_line)
            issues.append(f"figure {f.text!r} in {f.sentence[:100]!r} reports one of two evidence values — "
                          f"the evidence also states {f.evidence_line[:120]!r}")
    n_agree = sum(1 for b in bindings if b.outcome == "agree")
    n_unbound = sum(1 for b in bindings if b.outcome == "unbound")
    n_unchecked = sum(1 for b in bindings if b.outcome == "unchecked")
    n_unsupported = sum(1 for f in audit if f.status == "unsupported" and f.family not in ("identifier", "year"))
    unsupported_ids = [f.text for f in audit if f.status == "unsupported" and f.family == "identifier"]
    if unsupported_ids:
        ent_note += (f"; {len(unsupported_ids)} identifier(s) not in the evidence: "
                     f"{', '.join(repr(t) for t in unsupported_ids[:3])}")
    unsupported_years = [f.text for f in audit if f.status == "unsupported" and f.family == "year"]
    if unsupported_years:
        ent_note += (f"; {len(unsupported_years)} year(s) not in the evidence: "
                     f"{', '.join(unsupported_years[:4])}")
    withholds = [g for g in findings if g.status == "withhold"]
    if withholds:
        ent_note += "; " + "; ".join(f"{g.kind}: {g.detail}" for g in withholds)
    echo_rows = list(dict.fromkeys([f.text for f in audit if f.status == "echo"]
                                   + [e.text for e in entities if e.status == "echo"]))   # a year is both a figure and a year row
    if echo_rows:
        ent_note += (f"; {len(echo_rows)} fact(s) rest only on the agent's own earlier words (memory is not a source): "
                     f"{', '.join(repr(t) for t in echo_rows[:3])}")
    if dropped >= 2 and dropped > n_agree:
        ent_note += f"; {dropped} claim quote(s) were not in the reply"      # a binder that invented most rows
    withheld = bool(unsupported_entities or unsupported_ids or unsupported_years or withholds or echo_rows
                    or (strict_figures and n_unsupported) or (dropped >= 2 and dropped > n_agree))
    if issues:
        conf = min(0.95, 0.8 + 0.05 * len(issues))
        return ClaimBindingResult("REFUTED", conf, issues, bindings, dropped,
                                  reasoning=f"{len(issues)} validated contradiction(s) between the reply's own words and the evidence",
                                  audit=audit, entities=entities, findings=findings)
    if not bindings:
        return ClaimBindingResult("UNCERTAIN", 0.4, [], bindings, dropped,
                                  reasoning="no checkable claim survived validation"
                                  + (f"; {n_unsupported} figure(s) not found in the evidence" if n_unsupported else "")
                                  + ent_note, audit=audit, entities=entities, findings=findings)
    if n_unbound == 0 and n_unchecked == 0 and n_agree >= 1 and not withheld:
        # CONFIRMED means every checkable claim was bound to an agreeing
        # span. An "unchecked" row (a status word code cannot grade, or the
        # model and the figures disagreeing) is exactly what a residual
        # judge exists for — it keeps the verdict at UNCERTAIN, never a
        # confirm on the model's word. An entity the evidence never names
        # (mined pool: 11/60 fabrications confirmed — "Dr. Elin Vasquez
        # verified…", "won the Meridian Prize", the appended sentence the
        # binder never listed) withholds the confirm the same way.
        return ClaimBindingResult("CONFIRMED", 0.9, [], bindings, dropped,
                                  reasoning=f"all {n_agree} checkable claim(s) bound to agreeing evidence"
                                  + (f"; {n_unsupported} figure(s) not found in the evidence" if n_unsupported else ""),
                                  audit=audit, entities=entities, findings=findings)
    parts = []
    if n_unbound:
        parts.append(f"{n_unbound} claim(s) found no evidence span"
                     + (" (evidence truncated)" if evidence_truncated else ""))
    if n_unchecked:
        parts.append(f"{n_unchecked} claim(s) code could not grade")
    if n_unsupported:
        parts.append(f"{n_unsupported} figure(s) not found in the evidence")
    if withheld and n_agree >= 1 and not n_unbound and not n_unchecked:
        parts.append(f"all {n_agree} checkable claim(s) agree")
    return ClaimBindingResult("UNCERTAIN", 0.5, [], bindings, dropped,
                              reasoning="; ".join(parts) + ent_note, audit=audit, entities=entities,
                              findings=findings)


def render_prompt(claim: str, evidence: str, context: str) -> str:
    return CLAIM_BINDING_PROMPT.format(claim=claim, evidence=evidence, context=context or "(none)",
                                       max_claims=MAX_CLAIMS)


# ── §4IN: the residual judge with a quote burden ────────────────────────
# What code could not grade — a status word, a claim whose span the binder
# never found — goes to a model ONCE, with the burden reversed: it may not
# say "unsupported"; it must copy the evidence fragment that bears on each
# claim, and code accepts a relation only when that fragment is really in
# the evidence AND shares a subject word with the claim. A "contradict" that
# passes both is a refute with both quotes in the issue; a "support" that
# passes both is an agree; anything else stays unchecked.

RESIDUAL_PROMPT = """You are a claim binder for RESIDUAL claims. You do NOT judge the reply. You quote.

USER REQUEST:
{context}

REPLY (the agent's answer):
{claim}

EVIDENCE (tool outputs):
{evidence}

RESIDUAL CLAIMS (fragments of the reply that code could not check):
{claims}

For EACH residual claim, copy the ONE evidence fragment that most directly SUPPORTS or CONTRADICTS it — verbatim, no paraphrase, no ellipsis, at most {max_span} characters. "contradict" only when the fragment states the opposite about the SAME thing (the same file, task, service, quantity). "support" only when the fragment states what the claim states. If the evidence says nothing about the claim: relation "absent" and an empty evidence_quote. Never invent a fragment.

Return MINIFIED single-line JSON only (no newlines, no code fences):
{{"claims":[{{"quote":"<the residual claim, verbatim>","evidence_quote":"<exact evidence fragment or empty>","relation":"support|contradict|absent"}}]}}"""


def render_residual_prompt(claim: str, evidence: str, context: str, residual_quotes: List[str]) -> str:
    listed = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(residual_quotes))
    return RESIDUAL_PROMPT.format(claim=claim, evidence=evidence, context=context or "(none)",
                                  claims=listed, max_span=MAX_SPAN_CHARS)


def residual_bindings(res: ClaimBindingResult) -> List[Binding]:
    return [b for b in res.bindings if b.outcome in ("unchecked", "unbound") and not b.residual]


def shares_subject(a: str, b: str) -> bool:
    """A content word (≥4 letters, not a stopword) common to both texts,
    exactly or by a shared five-letter stem ("restarted"/"restart",
    "moons"/"moon"). The subject link the residual judge's quote must have."""
    wa = [w for w in re.findall(r"\w+", normalize_for_containment(a)) if len(w) >= 4 and w not in _ANCHOR_STOP]
    wb = {w for w in re.findall(r"\w+", normalize_for_containment(b)) if len(w) >= 4 and w not in _ANCHOR_STOP}
    for w in wa:
        if w in wb:
            return True
        if len(w) >= 6 and any(len(v) >= 5 and v[:5] == w[:5] for v in wb):
            return True
    return False


def _disagreement_holds(claim_quote: str, span: str, dq: Optional[Quantity], ds: Optional[Quantity], evidence: str) -> bool:
    """`bind`'s guard chain for a computed disagreement, in one place: the
    claim's figure stands nowhere else in the evidence, and the pair shares
    a subject with a single comparable figure or is typo-shaped (dense
    records aside)."""
    if dq is not None and figure_elsewhere(dq, evidence, hedged=_hedged(claim_quote)):
        return False
    if lexical_anchor(claim_quote, span) and _single_comparable(dq, span):
        return True
    return bool(_typo_shaped_disagreement(claim_quote, span)
                and (not _dense(dq, span) or _aligned(claim_quote, span))
                and not _claim_states(claim_quote, ds))


def apply_residual(res: ClaimBindingResult, reply: str, evidence: str, raw_model_output: Any, *,
                   evidence_truncated: bool = False, strict_figures: bool = False) -> ClaimBindingResult:
    """Fold the residual judge's validated quotes into the bindings and
    recompute the verdict. A row is accepted only for a residual claim it
    names, with a fragment that is in the evidence and shares the claim's
    subject; `absent` and everything unvalidated leave the row unchecked."""
    rows = parse_binder_output(raw_model_output)
    evidence = mask_self_echo(evidence)              # §4IV/§4IX: the agent's own earlier words validate nothing here either
    pending = residual_bindings(res)
    for r in rows:
        q = normalize_for_containment(r["quote"])
        target = next((b for b in pending if normalize_for_containment(b.quote) == q
                       or (len(q) >= SNAP_MIN_CHARS and (q in normalize_for_containment(b.quote)
                                                         or normalize_for_containment(b.quote) in q))), None)
        if target is None:
            continue
        rel, span = r["relation"], r["evidence_quote"]
        span = snap_quote(span, evidence) if (rel in ("support", "contradict") and span) else None
        if span is None:
            target.detail = target.detail or "residual judge: no validated fragment"
            continue
        if not shares_subject(target.quote, span):
            target.detail = "residual judge: fragment shares no subject with the claim"
            continue
        # figures on either side are compared by code and override the
        # model's relation ("Copied 5 files" supported by "7 files copied"
        # is a disagreement, review §4IN m2); a contradiction of a status
        # claim needs a polarity clash the model cannot manufacture
        if extract_quantities(target.quote):
            outcome, _d, dq, ds = _compare(target.quote, span)
            if outcome not in ("agree", "disagree"):
                target.detail = "residual judge: the claim's figure has no comparable figure in the fragment"
                continue
            if outcome == "disagree" and not _disagreement_holds(target.quote, span, dq, ds, evidence):
                # the same guards `bind` applies (review §4IY: a residual row re-opened the
                # seed long-weather-1 refute — "24°C tonight" against "current 31°C")
                target.detail = "residual judge: figures differ but the quotes share no subject, or the claim's figure stands elsewhere"
                continue
        elif rel == "contradict":
            # a span that contradicts "no errors" must itself carry an error word — the
            # reversed arm was satisfied by the claim's OWN negation (review §4IY)
            if not _polarity_clash(target.quote, span):
                target.detail = "residual judge: contradiction without a polarity clash"
                continue
            outcome = "disagree"
        else:
            if _polarity_clash(target.quote, span):
                target.detail = "residual judge: support with a polarity clash"
                continue
            outcome = "agree"
        target.evidence_quote, target.valid_span, target.residual = span, True, True
        target.outcome = outcome
        target.detail = f"residual judge: {rel}, validated"
        pending = [b for b in pending if b is not target]
    return verdict_from_bindings(res.bindings, evidence_truncated=evidence_truncated, dropped=res.dropped,
                                 audit=res.audit, entities=res.entities, strict_figures=strict_figures,
                                 findings=res.findings)


# ── §4IV: the agent's own words are not evidence ─────────────────────────
# Probe-4c (§4IT close): asked for the ΧΡΩΠΕΙ founders' dates, the agent ran
# `recall`, expanded episode ep:434 and restated its OWN earlier reply —
# "(1854–1935)", "(1866–1912)", both fabricated — and both tiers CONFIRMED
# the restatement against the episode's `OUTCOME (SUCCESS): …` line, which
# is that earlier reply verbatim. The judge saw the echo (§4HJ) one store
# further away. What the agent said before is a claim, not a source: a
# region of the evidence that is the agent's own earlier words can bind
# nothing and support nothing; a fact that rests only there is an ECHO — a
# code-validated withhold (it caps a cheap CONFIRMED like §4IR's name
# withhold and reaches the user's caveat) and never a refute (restating
# one's own past is not an invention of this turn: the objection tier reads
# the evidence unmasked). Shapes: an episode record's OUTCOME body and
# LESSON line (`knowledge_base(action='expand', ref='ep:N')`), a session
# expand's `assistant:` lines, a memory arc's `AI:` lines under a `USER:`
# line, and an earlier assistant reply of this conversation as
# `_prior_turn_evidence` labels it (`[assistant] … [/assistant]`).
# a record's header may share its line with the packer's block label ("[knowledge_base] EPISODE 434 [fetch]")
_EPISODE_HEAD_RE = re.compile(r"(?m)^(?:\[[a-z][\w .\-]{0,39}\] )*EPISODE \d+ \[")
_SESSION_HEAD_RE = re.compile(r"(?m)^(?:\[[a-z][\w .\-]{0,39}\] )*SESSION \S+ — ")
_ECHO_OUTCOME_RE = re.compile(
    r"(?ms)^OUTCOME \((?:SUCCESS|FAILURE)\): .*?"
    r"(?=\n(?:LESSON:|[ \t]*\d+\. \w+\(|EPISODE \d+ \[|TRIGGER:|CONTEXT:|\[[\w .\-]{1,40}\] )|\Z)")
_ECHO_LESSON_RE = re.compile(r"(?m)^LESSON: .*$")
#: Inside an EPISODE record only these lines are NOT the agent's words: the
#: header, the user's TRIGGER, the CONTEXT line, the numbered tool excerpts
#: and the packer's own marks. Everything else — the OUTCOME body with or
#: without its header, the LESSON, an orphan fragment after a "…[gap]…" —
#: is echo (fresh-eye review §4IX: the packer's claim window keeps exactly
#: the part of the OUTCOME that overlaps the claim, header dropped).
_EPISODE_KEEP_LINE_RE = re.compile(
    r"^(?:\[[a-z][\w .\-]{0,39}\] )*(?:EPISODE \d+ \[|TRIGGER:|CONTEXT:|[ \t]*\d+\. \w+\(|…\[gap\]…|…\[PACKER CUT|\s*$)")
#: a session expand's assistant message keeps its newlines: mask to the next role line
_ECHO_ASSISTANT_LINE_RE = re.compile(r"(?ms)^assistant: .*?(?=\n(?:user|assistant|system|tool): |\n\[[a-z][\w .\-]{0,39}\] |\Z)")
_ARC_USER_RE = re.compile(r"(?m)^(?:CONTENT: )?USER: ")
_ECHO_AI_RE = re.compile(r"(?ms)^(?:AI|ASSISTANT): .*?(?=\n(?:CONTENT: )?USER: |\nSOURCE: |\n\[[a-z][\w .\-]{0,39}\] |\Z)")   # multi-paragraph replies too (review §4IY)
#: a `[assistant]` block cut by the prior-evidence cap loses its closer: mask to the end (the safe direction)
_ECHO_PRIOR_ASSISTANT_RE = re.compile(r"(?ms)^\[assistant\] .*?(?:^\[/assistant\]$|\Z)")


def _regions(text: str, head_re: "re.Pattern") -> List[Tuple[int, int]]:
    """[start, end) of each record opened by `head_re`, closed by the next
    such header, the next packer block label, or the end."""
    heads = list(head_re.finditer(text))
    out: List[Tuple[int, int]] = []
    for i, h in enumerate(heads):
        nxt = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        lbl = _BLOCK_LABEL_RE.search(text, h.end(), nxt)
        out.append((h.start(), lbl.start() if lbl else nxt))
    return out


def self_echo_spans(text: str) -> List[Tuple[int, int]]:
    """[start, end) of every region of `text` that is the agent's own earlier
    words (see the section comment). Empty when there is none."""
    t = str(text or "")
    if not t:
        return []
    spans: List[Tuple[int, int]] = []
    for a, b in _regions(t, _EPISODE_HEAD_RE):
        pos = a
        for line in t[a:b].splitlines(keepends=True):
            end = pos + len(line)
            if not _EPISODE_KEEP_LINE_RE.match(line):
                spans.append((pos, end - (1 if line.endswith("\n") else 0)))
            pos = end
    for a, b in _regions(t, _SESSION_HEAD_RE):
        spans += [(m.start(), m.end()) for m in _ECHO_ASSISTANT_LINE_RE.finditer(t, a, b)]
    if _ARC_USER_RE.search(t):
        spans += [(m.start(), m.end()) for m in _ECHO_AI_RE.finditer(t)]
    spans += [(m.start(), m.end()) for m in _ECHO_PRIOR_ASSISTANT_RE.finditer(t)]
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def mask_self_echo(text: str) -> str:
    """Length-preserving: every echo region blanked, so a binder quote cannot
    snap there and no audit finds support there."""
    t = str(text or "")
    spans = self_echo_spans(t)
    if not spans:
        return t
    out = []
    pos = 0
    for a, b in spans:
        out.append(t[pos:a]); out.append(" " * (b - a)); pos = b
    out.append(t[pos:])
    return "".join(out)


def self_echo_text(text: str) -> str:
    t = str(text or "")
    return "\n".join(t[a:b] for a, b in self_echo_spans(t))


def _regrade_echo(audit: List["AuditFigure"], entities: List["AuditEntity"], reply: str, echo: str) -> None:
    """Rows the masked evidence left UNSUPPORTED that the echo text alone
    supports become status "echo" — in place."""
    if not echo:
        return
    fig_ok = {a.text for a in audit_numbers(reply, echo) + audit_years(reply, echo) + audit_identifiers(reply, echo)
              if a.status == "supported"}
    ent_ok = {e.text for e in audit_entities(reply, echo) if e.status == "supported"}
    for a in audit:
        if a.status == "unsupported" and a.text in fig_ok:
            a.status = "echo"
    for e in entities:
        if e.status == "unsupported" and e.text in ent_ok:
            e.status = "echo"


#: Of the packer's external tools, the two that may hand the agent its OWN
#: words back (`execute`: a `cat` of the draft it just wrote; `recall`: its
#: earlier reply). Kept here so the binder and the objection tier share it.
SOURCE_EXCLUDED_TOOLS = frozenset({"execute", "recall"})


def _source_tool(name: str) -> bool:
    try:
        from .agent import _evidence_is_external
    except Exception:  # noqa: BLE001
        return False
    key = str(name or "").lower().strip().replace("-", "_").replace(" ", "_")
    return key not in SOURCE_EXCLUDED_TOOLS and bool(_evidence_is_external({"name": name}))


def source_text(raw_sources: str) -> str:
    """The part of a turn's raw tool output that can vouch for a fact: the
    EXTERNAL tools' blocks (web, browser, documents, databases…) minus
    `execute`/`recall`, with the agent's own earlier words masked. Fresh-eye
    review §4IX: the §4IR cap, the §4IT caveat and the life-span audit read
    the whole raw string, so an `[execute] cat report.md` or a
    `[file_system]` write receipt echoing the agent's own draft vouched for
    a fabricated name — the cap and the caveat vanished. Unlabelled raw text
    (no packer blocks) is kept as it is: a caller that built it chose it."""
    raw = str(raw_sources or "")
    if not raw:
        return ""
    blocks = evidence_blocks(raw)
    if not blocks or all(name == "" for name, _ in blocks):
        return mask_self_echo(raw)
    kept = [f"[{name}] {body}" for name, body in blocks if name and _source_tool(name)]
    return mask_self_echo("\n".join(kept))


def echo_facts(res: Optional["ClaimBindingResult"]) -> List[str]:
    """The facts of a result that rest only on the agent's own earlier words."""
    if res is None:
        return []
    out = [str(a.text) for a in (res.audit or []) if getattr(a, "status", "") == "echo"]
    out += [str(e.text) for e in (res.entities or []) if getattr(e, "status", "") == "echo"]
    seen: set = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def run_binding(reply: str, evidence: str, raw_model_output: Any, *, raw_sources: str = "",
                evidence_truncated: bool = False, context: str = "",
                strict_figures: bool = False) -> ClaimBindingResult:
    """Pure pipeline from the binder's raw output to the verdict. `context`
    is the ask / conversation the reply answers — an entity or figure named
    there is not the reply's invention."""
    rows = parse_binder_output(raw_model_output)
    # §4IV: the agent's own earlier words (an expanded episode's OUTCOME, a
    # session's `assistant:` lines, an earlier reply) can bind nothing and
    # support nothing; what rests only there is an ECHO withhold
    ev_bind = _strip_marks(mask_self_echo(evidence))      # the packer's own marks are blanked too (§4IY)
    echo = self_echo_text(evidence) if self_echo_spans(evidence) else ""
    raw_bind = source_text(raw_sources) if raw_sources else ""
    bindings, dropped = bind(reply, ev_bind, rows)
    audit = (audit_numbers(reply, ev_bind, context) + audit_identifiers(reply, ev_bind, context)
             + audit_years(reply, ev_bind, context))
    entities = audit_entities(reply, ev_bind, context)
    _regrade_echo(audit, entities, reply, echo)
    findings = class_checks(reply, evidence, context)
    # §4IU: life spans — a range attached to the wrong person REFUTES (both
    # quotes below); a range the sources never carry WITHHOLDS like a name
    for ls in audit_life_spans(reply, ev_bind, context, raw_sources=raw_bind):
        if ls.status == "misattributed":
            findings.append(ClassFinding(
                "attribution", "refute",
                f"the reply attaches the life span ({ls.span}) to {ls.name!r}, but the evidence attaches it to "
                f"{ls.evidence_name!r} ({ls.evidence_line[:120]!r})"))
        elif ls.status == "unsupported":
            findings.append(ClassFinding("attribution", "withhold",
                                         f"life span ({ls.span}) of {ls.name!r} not in the evidence"))
    return verdict_from_bindings(bindings, evidence_truncated=evidence_truncated, dropped=dropped,
                                 audit=audit, entities=entities, strict_figures=strict_figures,
                                 findings=findings)


# ── §4IM phase 1.5: the deterministic number audit ───────────────────────
# The binder lists at most MAX_CLAIMS quotes; on a 2,000-character reply
# the perturbed figure is often not among them (mined pool: fact_swap 2/9
# caught vs the judge's 4/9). Numbers do not need a model to be found: every
# figure in the reply is extracted here (code fences, inline code, URLs and
# paths masked; dates masked; bare years dropped) and checked against every
# figure in the evidence with the same unit / rounding / hedge rules. A
# figure with an agreeing evidence figure is SUPPORTED; one with only a
# near-miss whose evidence line shares a subject word with the reply
# sentence (or a typo-shaped pair) is MISREPORTED — a validated
# contradiction; anything else is UNSUPPORTED (counted, never a verdict in
# phase 1: derived counts and conversions live there).

_CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_HEADING_ORDINAL_RE = re.compile(r"(?m)^(#{1,6}\s+)\d+\.")          # "### 7. Title" numbers a heading, states nothing
_SOURCE_LINE_RE = re.compile(r"(?m)^\s*\[Source:[^\]\n]*\]?\s*$")   # a citation line's URL carries path dates
MAX_ANCHOR_LINE_CHARS = 600                                          # a 4 KB single-line JSON blob anchors everything
_URL_PATH_RE = re.compile(r"(?:https?://|file://|/api/|/workspace/|~/)[^\s)\]>]*|(?<![\w.,/-])(?:[\w.-]+/)+[\w.-]+")   # "1,200/day" keeps its 1,200; `(?<!-)` keeps a hyphen run linear (review §4IX/§4IY: 13 s on 180 KB)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;\n])\s+")


@dataclass
class AuditFigure:
    text: str
    value: float
    family: str
    sentence: str
    status: str            # supported | misreported | conflicted | unsupported
    evidence_text: str = ""
    evidence_line: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "status": self.status, "evidence_text": self.evidence_text,
                "evidence_line": self.evidence_line[:160], "sentence": self.sentence[:160]}


def _mask_non_prose(text: str) -> str:
    t = _CODE_FENCE_RE.sub(lambda m: " " * len(m.group(0)), str(text or ""))
    t = _INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), t)
    t = _HEADING_ORDINAL_RE.sub(lambda m: m.group(1) + " " * (len(m.group(0)) - len(m.group(1))), t)
    return _URL_PATH_RE.sub(lambda m: " " * len(m.group(0)), t)


def _evidence_line_prose(ln: str) -> str:
    """An evidence line with URLs/paths masked and citation lines blanked —
    the reply side was masked, the evidence side was not, so a URL path date
    ("…/2013/08/28/when-do…") refuted "25 centuries" (corpus replay)."""
    if _SOURCE_LINE_RE.match(ln):
        return ""
    return _URL_PATH_RE.sub(lambda m: " " * len(m.group(0)), ln)


# a '.' is a decimal point only BETWEEN digits ("3.5"); a sentence that ends
# in a figure ("below 100,000. The sieve…") still ends (review §4IP R6: the
# old `(?<!\d)` merged it with the next sentence, so every sentence-level
# rule read two statements as one)
_SENT_END_RE = re.compile(r"[.!?;](?!\d)|\n")


#: An abbreviation's stop is not a sentence end ("approx. 29 users" keeps its
#: hedge; "Dr. Vasquez", "e.g. 5", "v. 2"). Length-preserving mask, applied
#: before the split (review §4IP R7 m2).
_ABBREV_RE = re.compile(r"\b(?:approx|ca|cf|e\.g|i\.e|etc|vs|v|dr|mr|mrs|ms|prof|st|no|fig|inc|ltd|jr|sr|min|max|avg|est)\.(?=\s)", re.IGNORECASE)


def _sentence_span(text: str, pos: int) -> Tuple[int, int]:
    """[start, end) of the sentence around `pos`; a '.' between digits is a
    decimal point and an abbreviation's stop is not a sentence end."""
    text = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", " "), text)   # every stop of "e.g." / "i.e."
    start = 0
    for m in _SENT_END_RE.finditer(text, 0, pos):
        start = m.end()
    m_end = _SENT_END_RE.search(text, pos)
    return start, (m_end.start() if m_end else len(text))


def _sentence_at(text: str, pos: int) -> str:
    start, end = _sentence_span(text, pos)
    return text[start:end].strip()


_LIST_BEFORE_RE = re.compile(r"\d\s*(?:,|/|\bor\b|\band\b)\s*$")
_LIST_AFTER_RE = re.compile(r"^\s*(?:,|/|\bor\b|\band\b)\s*\d")


def _in_inline_list(s: Quantity, line: str) -> bool:
    """The evidence figure is one item of an in-line list ("version 13 or
    15", "100, 150, 75"): one of several, never "the" field a reply figure
    misreads (corpus replay: "PostgreSQL 19" against "…version13 or 15")."""
    body = normalize_for_containment(line)
    tok = normalize_for_containment(s.text)
    i = body.find(tok)
    while i >= 0:
        before, after = body[max(0, i - 12):i], body[i + len(tok):i + len(tok) + 12]
        if _LIST_BEFORE_RE.search(before) or _LIST_AFTER_RE.match(after):
            return True
        i = body.find(tok, i + 1)
    return False


def _reply_states_value(s: Quantity, reply_vals: set) -> bool:
    """The reply states the evidence figure — exactly, or at the evidence
    figure's own precision ("18.4" states the "18" of "PostgreSQL 18"; the
    corpus replay refuted "PostgreSQL 19" against that heading)."""
    if s.value in reply_vals:
        return True
    f = _claim_unit_factor(s)          # compare in the figure's WRITTEN unit: 508 ms vs 509 ms, not 0.508 vs 0.509 at 0 decimals
    return any(_round_half_up(rv / f, s.decimals) == _round_half_up(s.value / f, s.decimals) for rv in reply_vals)


def _glued_occurrence(fig_text: str, evidence: str) -> bool:
    """The written figure appears in the evidence with a letter glued to
    either side (a snippet join, not a different number)."""
    digits = re.sub(r"[^\d.,]", "", str(fig_text or ""))
    if not digits or not re.search(r"\d", digits):
        return False
    esc = re.escape(digits)
    return re.search(rf"(?<=[^\W\d_]){esc}(?![\d])|(?<![\d]){esc}(?=[^\W\d_])", str(evidence or "")) is not None


def _url_occurrence(fig_text: str, evidence: str) -> Optional[Tuple[str, str]]:
    """The written figure stands as a whole token inside a URL or path of
    the evidence — the region `_evidence_line_prose` masks. -> ("port",
    line) when it is the port of a URL authority ("http://127.0.0.1:8101/"
    states the port 8101), ("token", line) for any other whole-token
    occurrence ("/2015/", "/v2/34/"), None otherwise. The masking hid the
    browser's own `URL: http://127.0.0.1:8101/` from the figure lookup, and
    an unrelated `const PORT = 8100;` then "misreported" the port (corpus
    turn 8b779b6b, §4IY)."""
    digits = re.sub(r"[^\d]", "", str(fig_text or ""))
    if not digits or digits != re.sub(r"[^\d.,]", "", str(fig_text or "")):
        return None                                  # a decimal or grouped figure never lives in a URL as itself
    esc = re.escape(digits)
    port_re = re.compile(r"^https?://[^/?#\s]*:" + esc + r"(?=[/?#]|$)", re.I)
    tok_re = re.compile(r"(?<![\w.,])" + esc + r"(?![\w.,])")
    for ln in str(evidence or "").splitlines():
        for m in _URL_PATH_RE.finditer(_strip_label(ln)):
            if port_re.match(m.group(0)):
                return "port", ln
            if tok_re.search(m.group(0)):
                return "token", ln
    return None


_LABEL_TAIL_RE = re.compile(r"(?:[:=]|\b(?:is|are|was|were|of|at|to|equals?|είναι|ήταν|σε|στα|στις|στους)\b)\s*$", re.I)


def _content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[^\W_]+", normalize_for_containment(text)) if len(w) >= 4 and w not in _ANCHOR_STOP]


def _same_subject_word(w: str, k: str) -> bool:
    """`lexical_anchor`'s word test — equal, or a ≥6-letter word sharing a
    5-letter stem — plus a short word and its plural ("line"/"lines",
    "port"/"ports", "box"/"boxes"). NOT the name-inflection `_same_word`
    further down: a first draft of this rule shadowed that function by name
    (§4IY R8), and its stem test calls "data" and "date" one word."""
    if w == k or (len(w) >= 6 and len(k) >= 5 and w[:5] == k[:5]):
        return True
    short, long_ = sorted((w, k), key=len)
    return len(short) >= 3 and long_ in (short + "s", short + "es", short[:-1] + "ies" if short.endswith("y") else "")


def _neighbour_words(before_txt: str, after_txt: str) -> List[str]:
    """The nearest content word on each side of a figure; one stopword or
    short token may stand between ("port 8103" → port; "18 release
    research" → release, never research)."""
    def near(seq: List[str]) -> List[str]:
        skipped = 0
        for w in seq:
            if w.isdigit():
                break
            if w in _ANCHOR_STOP or len(w) < 4:
                skipped += 1
                if skipped > 1:
                    break
                continue
            return [w]
        return []
    return (near(re.findall(r"[^\W_]+", normalize_for_containment(before_txt))[::-1])
            + near(re.findall(r"[^\W_]+", normalize_for_containment(after_txt))))


def _figure_subject_words(sentence: str, off: int, fig_text: str) -> List[str]:
    """The words that label a figure in its sentence: its immediate
    neighbours, or every word of the label phrase when the figure follows a
    colon, an equals sign or a copula ("Research events: 18", "the research
    count is 18", `research=17`)."""
    before_txt, after_txt = sentence[:off], sentence[off + len(fig_text):]
    words = _neighbour_words(before_txt, after_txt)
    if _LABEL_TAIL_RE.search(before_txt):
        label = re.split(r"[,;.!?\n(|]", before_txt)[-1]
        words = _content_words(label) + words
    return words


def _evidence_figure_subject(s: Quantity, line: str) -> List[str]:
    body = _strip_label(line)
    m = re.search(r"(?<![\d.,])" + re.escape(s.text) + r"(?!\d|[.,]\d)", body)   # "research=17, data=16": a list comma is not a decimal
    if m is None:
        return []
    return _figure_subject_words(body, m.start(), s.text)


def _figure_anchor(sentence: str, off: int, fig_text: str, s: Quantity, line: str) -> bool:
    """The shared subject word `lexical_anchor` found is the FIGURE's
    subject: a word labelling the reply figure labels the evidence figure
    too, or the sentence and the line share two distinct content words. One
    incidental word is not a subject — "Recent focus: PostgreSQL 18 release
    research" against `…debugging=29, research=17, data=16` shares
    "research", but that 18 is a version bound to "PostgreSQL" and the
    cluster count two words later is another quantity (corpus turn
    ad8c43ca, a correct briefing refuted, §4IY). "port 8103" against `PORT
    = 8102` and "a 13-line sample fixture" against "14 non-empty lines in
    sample fixture" still hold."""
    ev_words = _evidence_figure_subject(s, line)
    cl_words = _figure_subject_words(sentence, off, fig_text)
    if any(_same_subject_word(w, k) for k in ev_words for w in cl_words):
        return True
    hay = set(re.findall(r"[^\W_]+", normalize_for_containment(_strip_label(line))))
    shared = {w for w in _content_words(sentence) if any(_same_subject_word(w, h) for h in hay)}
    return len(shared) >= 2


def audit_numbers(reply: str, evidence: str, context: str = "") -> List[AuditFigure]:
    """Every prose figure of the reply, graded against the evidence. A figure
    the CONTEXT states (the user's own numbers restated) is supported by it;
    a supported figure whose evidence line has a same-skeleton twin differing
    only in that slot is CONFLICTED — the omitted-contradiction class caught
    without the binder having listed the claim (mined Pinball cases: the
    binder quoted `ballX` while the twin row changed `ballY`)."""
    prose = _mask_non_prose(reply)
    masked = mask_non_quantities(prose)
    ev_lines = [ln for ln in str(evidence or "").splitlines() if ln.strip()]
    ev_figs: List[Tuple[Quantity, str]] = []
    for ln in ev_lines:
        for q in extract_quantities(_evidence_line_prose(_strip_label(ln))):
            ev_figs.append((q, ln))
    ctx_figs: List[Tuple[Quantity, str]] = []
    for ln in str(context or "").splitlines():
        for q in extract_quantities(ln):
            ctx_figs.append((q, ln))
    reply_figs = extract_quantities_with_pos(masked)
    reply_vals = {q.value for q, _ in reply_figs} | {q.hi for q, _ in reply_figs if q.is_range}
    rslots = reply_slots_of(reply)
    # a line whose skeleton repeats is a ROW of a table or series: one of
    # many records, never "the" field a reply figure misreads (review §4IN
    # M2: an honest average over a 30-row latency table was refuted by the
    # one row a digit away, 154/300 random tables)
    sk_counts: Dict[str, int] = {}
    for ln in ev_lines:
        sk = _skeleton(ln)
        sk_counts[sk] = sk_counts.get(sk, 0) + 1
    is_row = {ln: sk_counts.get(_skeleton(ln), 0) >= 2 for ln in ev_lines}
    out: List[AuditFigure] = []
    for q, pos in reply_figs:
        sentence = _sentence_at(prose, pos)
        hedged = _hedged(sentence)
        same = [(s, ln) for s, ln in ev_figs if s.family == q.family]
        agreeing = [(s, ln) for s, ln in same if quantities_agree(q, s, hedged=hedged)]
        if agreeing:
            # the twin scan needs a figure that can single out a field: a bare
            # "1" agrees with "EXIT CODE: 1" and conflicts with the next
            # command's "EXIT CODE: 0" (mined rec-1b3e3e95e1, clean). Every
            # agreeing line is scanned — the twin of the SECOND agreeing line
            # went unseen (mined rec-e93e72ff85: 64,378 beside 64,377).
            twin_pair = None
            if sum(ch.isdigit() for ch in q.text) >= 2:
                for s, ln in agreeing:
                    twin = find_conflicting_line(evidence, _strip_label(ln), q.text, reply_slots=rslots)
                    if twin:
                        twin_pair = (s, twin)
                        break
            if twin_pair:
                out.append(AuditFigure(q.text, q.value, q.family, sentence, "conflicted", twin_pair[0].text, twin_pair[1]))
            else:
                s, ln = agreeing[0]
                out.append(AuditFigure(q.text, q.value, q.family, sentence, "supported", s.text, ln))
            continue
        in_ctx = [(s, ln) for s, ln in ctx_figs if s.family == q.family and quantities_agree(q, s, hedged=hedged)]
        if in_ctx:
            s, ln = in_ctx[0]
            out.append(AuditFigure(q.text, q.value, q.family, sentence, "supported", s.text, "[context] " + ln))
            continue
        # MISREPORTED needs all three: a typo-shaped pair (one digit apart at
        # the reply's precision), a subject word shared by the reply sentence
        # and the evidence line, and an evidence figure the reply never
        # states itself (a reply that says both 400 and 440 is talking about
        # two things). Measured on 60 live replies: the looser near-miss +
        # subject rule flagged 12 clean replies; this rule keeps the port
        # 8103/8102 catch and drops the canvas/ball/list-number noise.
        dense = len(extract_quantities(sentence)) >= 3   # coordinates, tables, specs
        # §4IU: a year-shaped figure (a bare 1800s integer; 1900+ never reach
        # here) is looked up, never "misreported" — a birth year one digit
        # from a stranger's is a different year, not a typo (req 2ef4f0a2:
        # "1854 vs 1852", two different men)
        if not q.unit and not q.family and q.decimals == 0 and "," not in q.text and 1800 <= q.value <= 1899:
            out.append(AuditFigure(q.text, q.value, q.family, sentence, "unsupported"))   # same shape test as `_is_bare_year`: "€1850" and "1,850" stay figures
            continue
        # §4IT: the figure's own digits GLUED to letters in the evidence
        # ("Πάρνηθος203" — the pre-§4IO ddgs join) is where the reply took
        # it from; calling it absent and then a misreport of a pagination
        # "1 - 200" two lines down refuted a correct address (corpus turn
        # 45360357). A glued occurrence supports nothing for a binding
        # (§4IM) but it does rule out "misreported".
        if _glued_occurrence(q.text, evidence):
            out.append(AuditFigure(q.text, q.value, q.family, sentence, "unsupported"))
            continue
        if "." in q.text and re.search(r"(?<![\d.])" + re.escape(q.text) + r"\.\d", evidence):
            out.append(AuditFigure(q.text, q.value, q.family, sentence, "unsupported"))     # "3.12" is the head of "3.12.4" (review §4IY)
            continue
        in_url = _url_occurrence(q.text, evidence) if not q.unit else None
        if in_url:
            kind, ln = in_url
            out.append(AuditFigure(q.text, q.value, q.family, sentence, "supported" if kind == "port" else "unsupported",
                                   q.text if kind == "port" else "", ln if kind == "port" else ""))
            continue
        s_start, _s_end = _sentence_span(prose, pos)
        off = pos - s_start - (len(prose[s_start:pos]) - len(prose[s_start:pos].lstrip()))   # `sentence` is stripped
        near = [] if dense else [(s, ln) for s, ln in same
                                 if _typo_shaped_disagreement(q.text, s.text) and _near_miss(q, s) and not s.bound
                                 and s.decimals - q.decimals <= 2      # a 14-decimal float is not a misread 380
                                 and len(ln) <= MAX_ANCHOR_LINE_CHARS and lexical_anchor(sentence, ln)
                                 and not _reply_states_value(s, reply_vals) and not is_row.get(ln, False)
                                 and not _in_inline_list(s, ln)
                                 and _figure_anchor(sentence, off, q.text, s, ln)]
        if near:
            s, ln = min(near, key=lambda p: abs(p[0].value - q.value))
            out.append(AuditFigure(q.text, q.value, q.family, sentence, "misreported", s.text, ln))
            continue
        out.append(AuditFigure(q.text, q.value, q.family, sentence, "unsupported"))
    return out


#: A bare year in prose: four digits in the calendar range, a whole token
#: (not part of an id, a dimension "1920x1080", a version, a ratio or a
#: path), outside code and tables. The number audit drops these on purpose
#: (a year is not a quantity to misreport by rounding); this audit asks a
#: different question — was the year in the evidence at all?
_YEAR_TOKEN_RE = re.compile(r"(?<![\w.,/:%#@-])(?:1[89]\d{2}|2[0-9]\d{2})(?!\w)(?![.,/:%#@-]\w)")   # "το 1870." keeps its stop


def _year_in(tok: str, hay: str) -> bool:
    """`tok` occurs in the (normalized) haystack as a year — not as the digits
    of a decimal ("1.2010"), a thousands group ("2024,000") or a four-digit
    slash pair ("Νόμος 1848/1989" supported a birth year 1848 on the first
    live probe). A URL path date ("/2015/05/18/"), a season ("2024/25"), a
    line:column ("app.js:2564:25" restated as "line 2564") and a dash range
    ("1848-1932") are the same number the reply took, and count."""
    if re.search(r"(?<![\d.])(?<!\d/)" + re.escape(tok) + r"(?!\d)(?![.,]\d)(?!/\d{4})", hay) is not None:
        return True
    # "2500 employees" against "2,500 people": the same count, grouped (review §4IX)
    return len(tok) == 4 and re.search(r"(?<![\d.,])" + re.escape(tok[0]) + "," + re.escape(tok[1:]) + r"(?![\d,])", hay) is not None


def _clock_years() -> set:
    """This year and its neighbours: the agent's own calendar, not a fact
    the evidence needs to carry ("as of 2026", "next year")."""
    y = datetime.date.today().year
    return {str(v) for v in (y - 1, y, y + 1)}


_PACKER_MARK_RE = re.compile(r"…\[PACKER CUT#[0-9a-f]+:\s*\d+\s+of\s+\d+\s+chars shown\]|…\[gap\]…")


def _strip_marks(text: str) -> str:
    """The packer's truncation and gap marks, blanked length-preservingly —
    their digits ("1854 of 1975 chars shown") supported a year (review §4IY)."""
    return _PACKER_MARK_RE.sub(lambda m: " " * len(m.group(0)), str(text or ""))


def audit_years(reply: str, evidence: str, context: str = "") -> List[AuditFigure]:
    """§4IS-b (req probe-012ca9ec, the Αλκιβιάδου retry): the reply gave the
    founders life spans — "(1850–1925)", "(1885–1975)" — that appeared in
    none of 26 tool outputs, and nothing noticed: the number audit drops
    bare years by design and the cheap judge looked elsewhere. Every bare
    year in the reply's prose is looked up verbatim in evidence ∪ context;
    an UNSUPPORTED one withholds a confirm exactly as an unsupported
    identifier does — never a refute (a year from memory is often right),
    and never the §4IR cap trigger. Corpus cost before shipping: 7 of 475
    good turns carried a year the digest lacked, several of them dimensions
    this token rule excludes."""
    prose = _mask_non_prose(str(reply or ""))
    prose = re.sub(r"(?<=\b(?:1[5-9]|20)\d{2})-(?=(?:1[5-9]|20)\d{2}(?!\d))", "–", prose)   # "1848-1894" is a range; a phone "2101-2345" is not (review §4IX/§4IY)
    hay = normalize_for_containment(_strip_marks(str(evidence or "")) + "\n" + str(context or ""))
    clock = _clock_years()
    out: List[AuditFigure] = []
    seen: set = set()
    for m in _YEAR_TOKEN_RE.finditer(prose):
        tok = m.group(0)
        if tok in seen or tok in clock:
            continue
        seen.add(tok)
        sentence = _sentence_at(prose, m.start())
        status = "supported" if _year_in(tok, hay) else "unsupported"
        out.append(AuditFigure(tok, float(tok), "year", sentence, status))
    return out


def audit_identifiers(reply: str, evidence: str, context: str = "") -> List[AuditFigure]:
    """Every identifier token in the reply (IPs, dotted versions, hex ids,
    UUIDs — inline code included, since that is where ids live), looked up
    verbatim in evidence ∪ context; a supported one is checked for a
    same-skeleton twin. Rows carry family "identifier"; an UNSUPPORTED
    identifier withholds a confirm, a CONFLICTED one refutes."""
    # code fences masked, URLs and inline code KEPT: `http://127.0.0.1:8100`
    # and `/api/download/<id>.png` are where identifiers live (mined
    # rec-83da54be2d: the IP twin went unseen because the URL was masked)
    prose = _CODE_FENCE_RE.sub(lambda m: " " * len(m.group(0)), str(reply or ""))
    ev = str(evidence or "")
    hay_ev = normalize_for_containment(ev)
    hay_ctx = normalize_for_containment(str(context or ""))
    out: List[AuditFigure] = []
    seen: set = set()
    rslots = reply_slots_of(reply)
    for m in _IDENT_RE.finditer(prose):
        tok = m.group(0).lower()
        if tok in seen:
            continue
        seen.add(tok)
        sentence = _sentence_at(prose, m.start())
        bare = tok[1:] if tok.startswith("v") and tok[1:2].isdigit() else tok      # "v29.4.0" is "Version 29.4.0"
        std = re.fullmatch(rf"({_STANDARD_PREFIX})[ \t-]?([a-z]?\d.*)", m.group(0))
        if std:                                     # "ISO 8601" / "ISO-8601" / "ISO8601" are one citation (review §4IX)
            std_re = re.compile(r"(?<!\w)" + re.escape(std.group(1).lower()) + r"[ \t-]?" + re.escape(std.group(2).lower()) + r"(?!\w)")   # "piso 8601" is not ISO 8601 (review §4IY)
            if std_re.search(hay_ev):
                home = next((ln for ln in ev.splitlines() if std_re.search(ln.lower())), "")
                out.append(AuditFigure(m.group(0), 0.0, "identifier", sentence, "supported", tok, home))
                continue
            if std_re.search(hay_ctx):
                out.append(AuditFigure(m.group(0), 0.0, "identifier", sentence, "supported", tok, "[context]"))
                continue
            out.append(AuditFigure(m.group(0), 0.0, "identifier", sentence, "unsupported"))   # never the substring test: "piso 8601"
            continue
        if tok in hay_ev or bare in hay_ev:
            home = next((ln for ln in ev.splitlines() if tok in ln.lower()), "")
            twin = find_conflicting_line(ev, _strip_label(home), tok, reply_slots=rslots) if home else None
            out.append(AuditFigure(m.group(0), 0.0, "identifier", sentence,
                                   "conflicted" if twin else "supported", tok, twin or home))
        elif tok in hay_ctx or bare in hay_ctx:
            out.append(AuditFigure(m.group(0), 0.0, "identifier", sentence, "supported", tok, "[context]"))
        else:
            out.append(AuditFigure(m.group(0), 0.0, "identifier", sentence, "unsupported"))
    return out


# ── §4IM phase 2a: the named-entity audit ───────────────────────────────
# A fabricated sentence usually carries no figure: "The lead maintainer,
# Dr. Elin Vasquez, verified the result", "It also won the Meridian Prize",
# "independently confirmed by the Karlsen Institute". The binder lists at
# most MAX_CLAIMS quotes and rarely picks the appended sentence, so every
# listed claim agrees and the reply is CONFIRMED (mined pool: 11/60). A
# proper name does not need a model to be found either: every Title-Case
# run of two or more words (connectors allowed) and every honorific-led
# name in the reply's prose — headings, table rows, code, URLs excluded —
# is looked up in the evidence and the context. Absent from both, it is
# UNSUPPORTED: CONFIRMED is withheld (the evidence does not cover the
# reply), never a refute (the reply may know it; phase 2 decides).

#: A Title-Case token: Latin, Greek or Cyrillic capital + lowercase (§4IP R7:
#: the audit was blind to every non-Latin name). All-caps tokens stay out
#: ("VALUE line", "JSON object" are labels), as do single tokens.
_UC = r"A-ZΑ-ΩΆ-ΏА-ЯЁ"
_LC = r"a-zα-ωά-ώϊϋΐΰа-яё"
_TC = rf"[{_UC}][{_LC}][\w'’-]*"
_CONNECT = r"(?:of|the|for|and|de|von|van|da|di|du|la|le|del|der|των|του|της)"
_HONORIFIC = r"(?:Dr|Prof|Mr|Mrs|Ms|Sir|Dame|Lord|Lady|Δρ|Καθ|κ|κα)\.?"
_INITIALS = rf"(?:[{_UC}]\.[ \t]){{1,3}}"       # "J. K. Thornwood", "A. Lindqvist" — a space after each initial ("U.S. Tensions" is not a name)
_SP = r"[ \t]{1,2}"            # masked code/URLs leave long runs of spaces: never bridge them
_ENTITY_RE = re.compile(
    rf"\b(?:{_HONORIFIC}{_SP}{_TC}(?:{_SP}{_TC})*"
    rf"|(?<![.\w]){_INITIALS}{_TC}"                      # not the tail of "U.S."
    rf"|{_TC}(?:{_SP}(?:{_CONNECT}{_SP})*{_TC})+)")       # a RUN of connectors: "Anneli van der Berg"
_HONORIFIC_RE = re.compile(rf"^{_HONORIFIC}\s")
_LEADING_RE = re.compile(r"^(?:the|dr|prof|mr|mrs|ms|sir|dame|lord|lady|δρ|καθ|κ|κα|η|ο|το|τον|την)\.?\s+", re.I)
_SKIP_LINE_RE = re.compile(r"^\s*(?:#|\||\*\*[^*]+\*\*\s*$|>)")
_BOLD_LEADIN_RE = re.compile(r"(?m)^\s*(?:[-*•]\s*)?\*\*[^*\n]{1,60}\*\*:?")   # "**All Systems Online** — …": a label, not a name
_FIRST_TOKEN_RE = re.compile(rf"^{_TC}(?:{_SP}{_CONNECT})?{_SP}")


@functools.lru_cache(maxsize=1)
def _dictionary() -> frozenset:
    try:
        with open("/usr/share/dict/words", encoding="utf-8", errors="ignore") as fh:
            return frozenset(w.strip().lower() for w in fh if w.strip())
    except OSError:
        return frozenset()


def _trim_sentence_initial(text: str) -> str:
    """"Ask Elin Vasquez's team", "Per the Karlsen Institute": a capitalised
    common word opening a sentence is not part of the name that follows.
    Without a word list every opener is dropped (the safe direction: a
    missed entity withholds nothing, a joined one mislabels)."""
    if _HONORIFIC_RE.match(text):
        return text
    m = _FIRST_TOKEN_RE.match(text)
    if not m:
        return text
    fm = re.match(r"[A-Za-z'’-]+", text)
    if fm is None:
        return text[m.end():]              # a non-Latin opener: no word list can vouch for it — drop it (the safe direction)
    first = fm.group(0).lower()
    words = _dictionary()
    if words and first not in words:
        return text
    return text[m.end():]


@dataclass
class AuditEntity:
    text: str
    sentence: str
    status: str            # supported | unsupported

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "status": self.status, "sentence": self.sentence[:160]}


def entity_key(text: str) -> str:
    """The comparable form of a name: folded, honorific / leading article
    dropped, possessive trimmed."""
    k = _LEADING_RE.sub("", normalize_for_containment(text))
    return re.sub(r"'s$", "", k).strip()


def _fold_accents(text: str) -> str:
    """Combining marks dropped: a Greek headline in capitals carries no tonos
    ("ΕΘΝΙΚΟ ΛΕΞΙΚΟ" lower-cases to "εθνικο", the reply writes "εθνικό")."""
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")


#: Greek → Latin, the way a judge or an English source writes a Greek name
#: ("Δρ. Ελένη Βασκέζ" → "dr. eleni vaskez", "Μητσοτάκης" → "mitsotakis").
#: Digraphs first, then letters; accents folded before. Used only to ADD a
#: match (a name found under transliteration is supported / present) —
#: never to deny one.
_GREEK_DIGRAPHS = (("ου", "ou"), ("αυ", "av"), ("ευ", "ev"), ("ηυ", "iv"), ("μπ", "b"), ("ντ", "d"),
                   ("γκ", "g"), ("γγ", "ng"), ("γχ", "nch"), ("τσ", "ts"), ("τζ", "tz"))
_GREEK_LETTERS = str.maketrans({
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i", "θ": "th", "ι": "i", "κ": "k",
    "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t",
    "υ": "y", "φ": "f", "χ": "ch", "ψ": "ps", "ω": "o"})


def translit_greek(text: str) -> str:
    """Lower-cased, accent-folded, Greek letters mapped to their usual Latin
    spelling; Latin text passes through unchanged."""
    t = _fold_accents(str(text or "").lower())
    if not re.search(r"[α-ω]", t):
        return t
    for src, dst in _GREEK_DIGRAPHS:
        t = t.replace(src, dst)
    return t.translate(_GREEK_LETTERS)


def _tok_supported(t: str, hay: str, hay_folded: str) -> bool:
    """One name token in the haystack. A Greek/Cyrillic token INFLECTS
    ("Δημήτριο Κουφοντίνα" in the reply, "Δημήτρης Κουφοντίνας" in the
    source) and loses its accents in capitals, so a non-Latin token is
    compared accent-folded and, at six or more letters, on its stem — the
    token minus its last two letters, at a word start (corpus replay §4IP
    R7: eight Greek turns gained a withhold, and the objection tier would
    have convicted the inflected spelling as an absent name)."""
    # a WORD, not a substring (fresh-eye review §4IX: "Mark Stone" was "present"
    # in "stock MARKet … mileSTONE"); a possessive or plural is the same word
    if _word_in(t, hay):
        return True
    if re.fullmatch(r"[a-z0-9\-]+", t):
        return False
    tf = _fold_accents(t)
    if _word_in(tf, hay_folded):
        return True
    if len(tf) >= 6 and _stem_in(tf, hay_folded):
        return True
    # a Greek name in the reply, an English source ("Μητσοτάκης" / "Mitsotakis")
    tl = translit_greek(tf)
    if tl != tf and (_word_in(tl, hay_folded) or (len(tl) >= 6 and _stem_in(tl, hay_folded))):
        return True
    return False


def _word_in(tok: str, hay: str) -> bool:
    return re.search(r"(?<![^\W_])" + re.escape(tok) + r"(?:['’]s|s|es)?(?![^\W_])", hay) is not None


def _stem_in(tok: str, hay: str) -> bool:
    """The token minus its last two letters, at a word start, followed by an
    INFLECTION (at most three letters) and a word end — "Δημήτρη-ς" matches
    "Δημήτρη", "Κουφοντίν-ας" matches "Κουφοντίνα"; "δημητρ-ιακών" (cereals)
    does not match "Δημήτρης" (review §4IX: the open-ended stem did)."""
    return re.search(r"(?<![^\W_])" + re.escape(tok[:-2]) + r"[^\W\d_]{0,3}(?![^\W_])", hay) is not None


def _entity_supported(key: str, hay: str) -> bool:
    if key and re.search(r"(?<![^\W_])" + re.escape(key) + r"(?![^\W_])", hay):    # "li wei" is not in "Eli Weiss" (review §4IY)
        return True
    toks = [t for t in re.findall(r"[^\W_][\w-]{2,}", key) if t not in _NAME_STOP]   # Greek/Cyrillic tokens too
    if not toks:
        return False
    hay_folded = _fold_accents(hay) if any(not re.fullmatch(r"[a-z0-9\-]+", t) for t in toks) else hay
    if all(_tok_supported(t, hay, hay_folded) for t in toks):      # "Vasquez, Elin" still covers "Elin Vasquez"
        return True
    # a Latin-script reply against a Greek source ("Kyriakos Mitsotakis" / "Κυριάκος
    # Μητσοτάκης", review §4IX): the bridge ran one way only
    if re.search(r"[α-ω]", hay) and all(re.fullmatch(r"[a-z0-9\-]+", t) for t in toks):
        hays = [translit_greek(hay), translit_greek_elot(hay),
                translit_greek(_fold_accents(re.sub(r"(?<=[εα])υ(?=[θκξπστφχ])", "f", hay)))]
        return all(any(_word_in(t, h) or (len(t) >= 6 and _stem_in(t, h)) for h in hays) for t in toks)
    return False


def _any_name_word_supported(key: str, hay: str, hay_folded: str) -> bool:
    toks = [t for t in re.findall(r"[^\W_][\w-]{2,}", key) if t not in _NAME_STOP]
    if not toks:
        return True
    if any(_tok_supported(t, hay, hay_folded) for t in toks):
        return True
    if re.search(r"[α-ω]", hay) and all(re.fullmatch(r"[a-z0-9\-]+", t) for t in toks):
        hays = [translit_greek(hay), translit_greek_elot(hay), translit_greek(_fold_accents(re.sub(r"(?<=[εα])υ(?=[θκξπστφχ])", "f", hay)))]
        return any(_word_in(t, h) or (len(t) >= 6 and _stem_in(t, h)) for t in toks for h in hays)
    return False


def audit_entities(reply: str, evidence: str, context: str = "") -> List[AuditEntity]:
    """Every named entity in the reply's prose, graded against evidence ∪
    context."""
    prose = _BOLD_LEADIN_RE.sub(lambda m: " " * len(m.group(0)), _mask_non_prose(reply))
    hay = normalize_for_containment(str(evidence or "") + "\n" + str(context or ""))
    out: List[AuditEntity] = []
    seen: set = set()
    pos = 0
    for line in prose.splitlines(keepends=True):
        start = pos
        pos += len(line)
        if _SKIP_LINE_RE.match(line) or not line.strip():
            continue
        for m in _ENTITY_RE.finditer(line):
            text = m.group(0)
            abs_pos = start + m.start()
            s_start, s_end = _sentence_span(prose, abs_pos)
            # opens its sentence — or its clause after a bold label / code span and a
            # dash or colon ("**Acquired Skills** — Two Python scripts", "`x.py` — Related
            # Mars script"): the opener's capital is punctuation, not a name (§4IR)
            if prose[s_start:abs_pos].strip(" \t*_—–-:•") == "":
                text = _trim_sentence_initial(text)
                if not _ENTITY_RE.fullmatch(text):
                    continue
            key = entity_key(text)
            if len(key) < 5 or key in seen:
                continue
            seen.add(key)
            status = "supported" if _entity_supported(key, hay) else "unsupported"
            out.append(AuditEntity(text.strip(), prose[s_start:s_end].strip(), status))
    return out


# ── §4IR: a validated name withhold outranks a cheap CONFIRMED ──────────
# The binder never refutes on an unsupported name (it might be a name the
# agent knows), but its withhold is CODE-validated — the name is in neither
# the evidence nor the request/project note — while the incumbent's CONFIRMED
# on the same reply is an opinion. Measured on the paired pools (§4IR): the
# cheap judge confirmed every fabricated-name trial the binder withheld on
# (15/15 mined, 10/10 seed; the §4IQ class 15/15 + 6/6) and 8 more
# fact-swap/omission trials; the price was 5/49 mined and 0/34 seed good
# confirms, 9/165 live ones. The rule caps such a CONFIRMED at the withheld
# confidence — UNCERTAIN to every consumer, never a refute.
_LOOPBACK_ID_RE = re.compile(r"^(?:127\.\d{1,3}\.\d{1,3}\.\d{1,3}|0\.0\.0\.0|localhost)$", re.I)


def unsupported_names(res: "ClaimBindingResult", prior_evidence: str = "") -> List[str]:
    """The names and identifiers the binder found in NEITHER evidence nor
    context — minus any an earlier turn's evidence carried (§4HZ: a proof of
    invention is against the whole session) and minus loopback addresses
    (the agent's own URL convention, not a fact about the world)."""
    names = [e.text for e in (res.entities or []) if getattr(e, "status", "") == "unsupported"]
    names += [a.text for a in (res.audit or [])
              if getattr(a, "family", "") == "identifier" and getattr(a, "status", "") == "unsupported"
              and not _LOOPBACK_ID_RE.match(str(a.text or ""))]
    if prior_evidence and names:
        hay = normalize_for_containment(mask_self_echo(prior_evidence))     # an earlier reply of ours vouches for nothing
        names = [n for n in names if not _entity_supported(entity_key(n), hay)]
    return names


def name_withhold_caps_confirm(res: Optional["ClaimBindingResult"], *, truncation_severity: float,
                               truncation_floor: float, prior_evidence: str = "", raw_sources: str = "") -> List[str]:
    """The names that justify capping a cheap CONFIRMED, or [] when nothing
    does: the binder withheld (UNCERTAIN, no contradiction), at least one
    name/identifier is unsupported across the session, and the digest was
    not cut past the floor (an absence from a cut digest proves little).
    With the turn's RAW tool outputs (§4IU self-review) the absence is
    tested against the whole sources — a name the packer left out is not a
    name the sources lacked — and the digest's floor does not apply."""
    if res is None or res.verdict != "UNCERTAIN":
        return []
    src = source_text(raw_sources)
    if not src.strip() and truncation_severity >= truncation_floor:
        return []                          # a `[task_list]` row is not "the whole sources" (review §4IY)
    names = unsupported_names(res, (mask_self_echo(str(prior_evidence or "")) + "\n" + src).strip())
    # §4IV: a fact that rests only on the agent's own earlier words is the
    # same validated withhold on a different ground — it caps too
    return names + [f for f in echo_facts(res) if f not in names]


# ── §4IU: a life span attached to the wrong person ───────────────────────
# "Name (YYYY–YYYY)" is a fixed convention. Req 2ef4f0a2 wrote "Σπήλιος
# (Σπυρίδων) Οικονομίδης (1854–1933)"; the sources carried that exact range
# once — "Γεώργιος Οικονομίδης του Ιωάννη (1854-1933) ήταν πολιτικός" — a
# namesake. Every lookup-based audit is blind to this: the years exist. The
# attribution is checkable: the same range in the evidence, attached to a
# name that shares the family name but not the given name, is a validated
# contradiction with both quotes. A range attached to a matching name
# anywhere in the evidence is support; a range found nowhere is left to the
# year audit (unsupported → withhold).
_LIFE_SPAN_RE = re.compile(
    rf"(?P<name>{_TC}(?:[ \t]\({_TC}\))?(?:[ \t]{_TC}){{0,3}})[*_]*[ \t]*[*_]*\((?P<lo>1[5-9]\d{{2}}|20\d{{2}})[ \t]*[–—-][ \t]*(?P<hi>1[5-9]\d{{2}}|20\d{{2}})\)")   # bold markers may sit between the name and the span


#: Title and role words that precede a surname in prose ("President
#: Papandreou (1888–1968)", "Πρωθυπουργός Παπανδρέου") — not part of the name.
_NAME_ROLE_WORDS = frozenset("""
president professor prof dr mr mrs ms sir dame lord lady king queen prince saint st general colonel
captain major admiral bishop father senator governor mayor minister chancellor judge rabbi imam
πρόεδρος πρωθυπουργός καθηγητής καθηγητή καθ δρ στρατηγός βασιλιάς βασιλιά άγιος αγίου επίσκοπος πατήρ
υπουργός υπουργού δήμαρχος δημάρχου βουλευτής ναύαρχος ραβίνος ραβίνου
""".split())
#: Greek given names come as a formal form, a diminutive and an English
#: rendering — one person (fresh-eye review §4IX: "Γιώργος" vs "Γεώργιος
#: Παπανδρέου" was a namesake and a REFUTE). Keys are transliterated forms,
#: including the ones `translit_greek`'s digraph rule produces ("konstadinos").
_GIVEN_NAME_FAMILIES = (
    ("georgios", "giorgos", "george", "yorgos", "yiorgos"),
    ("ioannis", "giannis", "yiannis", "yannis", "john", "ioanni", "ianni"),
    ("nikolaos", "nikos", "nicholas", "nick", "nikolas"),
    ("konstantinos", "konstadinos", "kostas", "costas", "constantine", "konstantin"),
    ("emmanouil", "manolis", "emmanuel", "manos"),
    ("dimitrios", "dimitris", "demetrios", "demetris", "mimis"),
    ("michail", "michalis", "michael"),
    ("vasileios", "vasilis", "basil", "vassilis"),
    ("athanasios", "thanasis", "sakis", "nasos"),
    ("eleftherios", "lefteris"),
    ("stylianos", "stelios"),
    ("spyridon", "spilios", "spyros", "spiros"),
    ("alexandros", "alekos", "alexander", "alex", "alexis"),
    ("panagiotis", "takis", "panos", "panayiotis"),
    ("gerasimos", "makis"),
    ("christos", "chris", "christodoulos"),
    ("petros", "peter"),
    ("pavlos", "paul"),
    ("andreas", "andrew"),
    ("antonios", "antonis", "anthony", "adonis"),
    ("theodoros", "thodoris", "theodore", "thodoros", "theo"),
    ("stathis", "efstathios", "eustathios"), ("babis", "charalambos", "charalampos"),
    ("aristotelis", "aristotle", "aris"), ("odysseas", "odysseus", "ulysses"), ("elytis", "elytis"),
    # English nicknames (review §4IY: "Jimmy Carter" vs "James Carter" was a namesake)
    ("james", "jim", "jimmy"), ("robert", "bob", "bobby", "rob", "robbie"), ("edward", "ted", "teddy", "ed", "eddie"),
    ("william", "bill", "billy", "will", "liam"), ("lev", "leo"), ("elizabeth", "liz", "beth", "betty", "eliza"),
    ("margaret", "maggie", "peggy", "meg"), ("richard", "dick", "rick", "richie"), ("charles", "charlie", "chuck"),
    ("thomas", "tom", "tommy"), ("joseph", "joe", "joey"), ("michael", "mike", "mick"), ("daniel", "dan", "danny"),
    ("anthony", "tony"), ("christopher", "chris", "kit"), ("katherine", "kate", "katie", "kathy", "catherine"),
    ("john", "jack", "johnny"), ("alexander", "alex", "sasha", "xander"), ("benjamin", "ben", "benny"),
    ("samuel", "sam", "sammy"), ("henry", "harry", "hal"), ("frederick", "fred", "freddie"), ("stephen", "steve", "steven"),
    ("dostoevsky", "dostoyevsky", "dostoievski"),
    ("sotirios", "sotiris"),
    ("evangelos", "vangelis", "evagelos", "vagelis"),
    ("ilias", "elias"),
    ("eleni", "helen", "helena"),
    ("maria", "mary"),
    ("aikaterini", "katerina", "catherine", "katina"),
    ("sofia", "sophia"),
    ("anastasios", "tasos", "anastasis"),
    ("apostolos", "tolis"),
)
_GIVEN_NAME_CANON = {v: fam[0] for fam in _GIVEN_NAME_FAMILIES for v in fam}
#: A second romanisation — the ELOT one: ντ→nt, μπ→mp, γκ→gk, αυ→au, ευ→eu —
#: because an English source writes "Kazantzakis" where `translit_greek`
#: writes "kazadzakis" (review §4IX exhibit 1).
_GREEK_DIGRAPHS_ELOT = (("ου", "ou"), ("αυ", "au"), ("ευ", "eu"), ("ηυ", "iu"), ("μπ", "mp"), ("ντ", "nt"),
                        ("γκ", "gk"), ("γγ", "ng"), ("γχ", "nch"), ("τσ", "ts"), ("τζ", "tz"))


def translit_greek_elot(text: str) -> str:
    t = _fold_accents(str(text or "").lower())
    if not re.search(r"[α-ω]", t):
        return t
    for src, dst in _GREEK_DIGRAPHS_ELOT:
        t = t.replace(src, dst)
    return t.translate(_GREEK_LETTERS)


def _romanisations(word: str) -> List[str]:
    """Every spelling a source may use for one Greek word: the plain
    romanisation, the ELOT one, and the phonetic one where αυ/ευ before a
    voiceless consonant read af/ef ("Ελευθέριος" → eleftherios); a Latin
    word is itself. A known given name adds its family's canonical form."""
    w = str(word or "").lower()
    phon = re.sub(r"(?<=[εα])υ(?=[θκξπστφχ])", "f", w)
    # mid-word μπ/ντ/γκ read mb/nd/ng in English renderings ("Lambros", "Andonis")
    mid = re.sub(r"(?<=[α-ωά-ώ])μπ", "mb", re.sub(r"(?<=[α-ωά-ώ])ντ", "nd", re.sub(r"(?<=[α-ωά-ώ])γκ", "ng", phon)))
    forms = [translit_greek(_fold_accents(w)), translit_greek_elot(w),
             translit_greek(_fold_accents(phon)), translit_greek(_fold_accents(mid))]
    out: List[str] = []
    for f in forms:
        for v in (f, _GIVEN_NAME_CANON.get(f)):
            if v and v not in out:
                out.append(v)
    return out


def _name_tokens(text: str) -> set:
    """Name words with function words, titles and role words dropped; each
    word carries all its romanisations (and its given-name family) joined
    by "|", so any spelling on either side matches."""
    return {"|".join(_romanisations(t)) for t in re.findall(r"[^\W\d_][\w'’]{2,}", str(text or "").lower())   # "Jean-Paul" is two words
            if t not in _NAME_STOP and t not in _NAME_ROLE_WORDS}


def _same_word(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) >= 6 and len(b) >= 6:
        sa, sb = a[:-2], b[:-2]
    elif len(a) >= 4 and len(b) >= 4:
        sa, sb = a[:-1], b[:-1]
    elif min(len(a), len(b)) >= 3:         # "Ίων" / "Ίωνα": a short name and its one-letter inflection
        return (a.startswith(b) or b.startswith(a)) and abs(len(a) - len(b)) <= 1
    else:
        return False
    return (b.startswith(sa) and len(b) - len(sa) <= 3) or (a.startswith(sb) and len(a) - len(sb) <= 3)


def _same_name_tok(a: str, b: str) -> bool:
    """Two name tokens are the same word when equal under either romanisation
    or when one carries the other's stem with an inflection of at most three
    letters — Greek names INFLECT ("Σπήλιου Οικονομίδη" is "Σπήλιος
    Οικονομίδης"), and short names drop one letter ("Νίκου" / "Νίκος")."""
    return any(_same_word(x, y) for x in a.split("|") for y in b.split("|"))


def _names_relation(rtoks: set, etoks: set, cross_script: bool = False) -> str:
    """"same" — one name is contained in the other (a surname-only or a fuller
    spelling), they share two words, or the one unshared word on each side is
    a near-spelling of the other; "namesake" — exactly one shared word (the
    family name) beside clearly different given names, in ONE script: the
    misattribution shape; "other" — nothing shared, or a pair written in two
    scripts (a romanisation the tables do not know — "Μαρία Κάλλας" / "Maria
    Callas", "Αϊνστάιν" / "Einstein", fresh-eye review §4IY): a translation,
    an organisation's other name, or a stranger — code cannot tell which, so
    it is neither support nor a contradiction."""
    if not rtoks or not etoks:
        return "other"                     # a name of stopwords only vouches for nothing
    r_hit = {a for a in rtoks if any(_same_name_tok(a, b) for b in etoks)}
    e_hit = {b for b in etoks if any(_same_name_tok(a, b) for a in rtoks)}
    if r_hit == rtoks or e_hit == etoks or len(r_hit) >= 2:
        return "same"
    if len(r_hit) != 1:
        return "other"
    if cross_script:
        return "other"
    # the same script, one shared word: are the unshared words near-spellings of each other?
    import difflib
    ru, eu = [t for t in rtoks if t not in r_hit], [t for t in etoks if t not in e_hit]
    best = max((difflib.SequenceMatcher(None, x, y).ratio() for a in ru for b in eu
                for x in a.split("|") for y in b.split("|")), default=0.0)
    return "same" if best >= 0.7 else "namesake"


@dataclass
class LifeSpanFinding:
    name: str
    span: str
    status: str                 # supported | misattributed | unsupported
    evidence_name: str = ""
    evidence_line: str = ""
    sentence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "span": self.span, "status": self.status,
                "evidence_name": self.evidence_name, "evidence_line": self.evidence_line[:160]}


def audit_life_spans(reply: str, evidence: str, context: str = "", raw_sources: str = "") -> List[LifeSpanFinding]:
    """`raw_sources` — the turn's whole tool outputs when the caller has them:
    the attribution test is about what the SOURCES say, and the digest is a
    selection (req 2ef4f0a2: the namesake's line was in a tool output the
    packer left out)."""
    prose = _mask_non_prose(str(reply or ""))
    ev = str(evidence or "") + "\n" + str(context or "") + "\n" + str(raw_sources or "")
    out: List[LifeSpanFinding] = []
    seen: set = set()
    for m in _LIFE_SPAN_RE.finditer(prose):
        name, lo, hi = m.group("name").strip(), m.group("lo"), m.group("hi")
        if not (15 <= int(hi) - int(lo) <= 110):
            continue                       # "Top Breakthroughs (2025–2026)" is a period, not a life
        span = f"{lo}–{hi}"
        if span in seen:
            continue                       # one finding per span: the reply spelling the name twice is one claim
        seen.add(span)
        sentence = _sentence_at(prose, m.start())
        rtoks = _name_tokens(name)             # an alias in parentheses ("(Σπυρίδων)") is one more token of the same person
        pat = re.compile(rf"(?<!\d){lo}[ \t]*[–—-][ \t]*{hi}(?!\d)")
        occurrences = list(pat.finditer(ev))
        if not occurrences:
            out.append(LifeSpanFinding(name, span, "unsupported", sentence=sentence))
            continue
        status, ev_name, ev_line = "unsupported", "", ""
        for o in occurrences:
            before = ev[max(0, o.start() - 90):o.start()]
            line_start = ev.rfind("\n", 0, o.start()) + 1
            line = ev[line_start:ev.find("\n", o.end()) if ev.find("\n", o.end()) >= 0 else len(ev)]
            # the name the source attaches to the range: the Title-Case run right before it
            # (emphasis markers dropped — a bold name in a source is still the name)
            nm = re.search(rf"({_TC}(?:[ \t](?:{_CONNECT}[ \t])*{_TC}){{0,4}})[ \t]*\(?$",
                           re.sub(r"[*_]+", " ", before).rstrip(" \t(").rstrip())
            if not nm:
                continue                   # a range attached to nobody ("De Geyter, Pierre, 1848-1932") vouches for no one
            cross = bool(re.search(r"[α-ωΑ-Ωά-ώ]", name)) != bool(re.search(r"[α-ωΑ-Ωά-ώ]", nm.group(1)))
            rel = _names_relation(rtoks, _name_tokens(nm.group(1)), cross_script=cross)
            if rel == "same":
                status = "supported"; break
            if rel == "namesake" and status != "misattributed":
                status, ev_name, ev_line = "misattributed", nm.group(1), line
        out.append(LifeSpanFinding(name, span, status, ev_name, ev_line, sentence))
    return out


# ── §4IT: what the sources did not say, for the USER ───────────────────────
# The withholds above protect the labels; nothing so far told the person
# reading the reply. The Αλκιβιάδου retry shipped "(1850–1925)" and
# "(1885–1975)" as facts. `unverified_facts` is the short, defensible list a
# caveat line can carry: only things the reply states EXACTLY and the whole
# session's evidence carries NOWHERE — bare years, standards citations, and
# named people/organisations written with an honorific or in Latin
# Title-Case (a Greek capitalised phrase is too often a common noun in the
# genitive to put in front of the user). Never on a digest cut past the
# floor; never on a REFUTED (that verdict has its own banner).
def unverified_facts(res: Optional["ClaimBindingResult"], *, evidence: str, prior_evidence: str = "",
                     truncation_severity: float = 0.0, truncation_floor: float = 0.25, limit: int = 5,
                     raw_sources: str = "") -> List[str]:
    if res is None or res.verdict == "REFUTED":
        return []
    src = source_text(raw_sources)
    if not src.strip() and truncation_severity >= truncation_floor:
        return []                          # a cut DIGEST proves little; the sources decide when we have them
    # §4IV: our own earlier words are not a source — each text masked on its own, so an
    # unterminated OUTCOME at the end of one cannot swallow the start of the next; the
    # packer's own marks carry digits ("1854 of 1975 chars shown") and are not evidence (review §4IY)
    hay = normalize_for_containment(_strip_marks(mask_self_echo(str(evidence or ""))) + "\n" + mask_self_echo(str(prior_evidence or ""))
                                    + "\n" + src)
    hay_folded = _fold_accents(hay)
    out: List[str] = []
    for a in (res.audit or []):
        if getattr(a, "status", "") not in ("unsupported", "echo"):
            continue
        fam, text = getattr(a, "family", ""), str(getattr(a, "text", "") or "")
        if fam == "year" and not _year_in(text, hay):
            out.append(text)
        elif fam == "identifier" and re.match(rf"^{_STANDARD_PREFIX}", text) and normalize_for_containment(text) not in hay:
            out.append(text)
    for g in (res.findings or []):
        if getattr(g, "kind", "") == "attribution" and getattr(g, "status", "") == "withhold":
            m = re.search(r"\((\d{4}–\d{4})\) of '([^']+)'", str(getattr(g, "detail", "") or ""))
            if m and not re.search(r"(?<!\d)" + m.group(1).replace("–", r"[ \t]*[–—-][ \t]*") + r"(?!\d)", hay):
                out.append(f"{m.group(1)} ({m.group(2)})")
    for e in (res.entities or []):
        if getattr(e, "status", "") not in ("unsupported", "echo"):
            continue
        text = str(getattr(e, "text", "") or "")
        named = bool(_HONORIFIC_RE.match(text)) or bool(re.fullmatch(r"[A-Z][a-z][\w'’-]*(?:[ \t](?:(?:of|the|for|and|de|von|van|da|di|du|la|le|del|der)[ \t])?[A-Z][a-z][\w'’-]*)+", text))
        if not named:
            continue
        key = entity_key(text)
        # listed only when NO word of the name is anywhere in the sources ("Karlsen Institute"
        # beside "Karlsen Road" is a judgement, not a defensible absence) — the per-word test
        # now carries the Latin→Greek bridge too (review §4IY: a false caveat)
        if not _any_name_word_supported(key, hay, hay_folded):
            out.append(text)
    spans = [x for x in out if "–" in x]
    out = [x for x in out if "–" in x or not any(x in sp.split(" ")[0] for sp in spans)]   # a span subsumes its own years
    seen: set = set()
    uniq = [x for x in out if not (x in seen or seen.add(x))]
    return uniq[:limit]


# ── §4IN phase 2: the class checks ──────────────────────────────────────
# Four failure classes the binder cannot see through quotes, decided the
# same way — by code, from things that are checkable — and reusing the
# detectors the turn loop already trusts. Two REFUTE (a validated defect
# in the reply's own text), two WITHHOLD a confirm (the reply may honestly
# report a failed tool or paraphrase the topic; neither is a contradiction).

@dataclass
class ClassFinding:
    kind: str              # artifact | constraint | evidence | topic
    status: str            # refute | withhold
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "status": self.status, "detail": self.detail[:200]}


# A packer label: `[tool_name] ` — starts with a letter, snake/space/dot/dash,
# and a SPACE after it. Fresh-eye review §4IY: `[0] OK goto`, `[1] Bien…`,
# `[edit]` and `[...truncated...]` lines inside browser / wiki / research
# bodies split a block and dropped everything after them as a non-tool
# "block" — the cap, the caveat and the appeal lost the page that carried
# the name. Producers also neutralise label-shaped body lines (§4IY).
_BLOCK_LABEL_RE = re.compile(r"(?m)^\s*\[(?P<name>[a-z][\w .\-]{0,39})\] (?=\S|\n|$)")


def evidence_blocks(evidence: str) -> List[Tuple[str, str]]:
    """The packer's `[tool] body` blocks as (tool name, body); a body with
    no label is one block named ""."""
    text = str(evidence or "")
    marks = list(_BLOCK_LABEL_RE.finditer(text))
    if not marks:
        return [("", text)] if text.strip() else []
    out: List[Tuple[str, str]] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()
        if body:
            out.append((m.group("name").strip().lower(), body))
    return out


def _block_failed_or_empty(name: str, body: str) -> bool:
    """One block is a failure (the shared tool-error sniffer — status when
    the body carries one, prose rules otherwise) or an empty retrieval (the
    turn's evidence gate, for the tools it knows). Never a third vocabulary."""
    from ..distill.outcome_heuristics import _looks_like_tool_error
    if _looks_like_tool_error(body):
        return True
    if not body.strip() or body.strip().lower() in ("(empty output)", "(no output)", "no results", "no results found."):
        return True
    try:
        from .evidence_gate import assess_turn_evidence
        a = assess_turn_evidence([{"name": name, "content": body}])
        return a.consulted > 0 and a.substantive == 0
    except Exception:  # noqa: BLE001
        return False


def evidence_all_failed(evidence: str) -> bool:
    """Every evidence block is a tool failure or an empty retrieval —
    nothing a success claim could rest on. Delegates per block to the shared
    sniffer and the evidence gate."""
    blocks = evidence_blocks(evidence)
    if not blocks:
        return False
    return all(_block_failed_or_empty(name, body) for name, body in blocks)


_ASK_FRAMING = frozenset("""
please tell show give find list answer question explain describe help want need know like would could
should make write check look search user assistant think just also really very much many
whats yourself reply nothing else your quick ghost hello thanks
""".split())


def ask_content_words(context: str) -> List[str]:
    """The ask's subject words: content words of four letters or more that
    are neither stopwords nor question framing ("please tell me"), in
    order, deduplicated."""
    seen: set = set()
    out: List[str] = []
    for w in re.findall(r"\w+", normalize_for_containment(context)):
        if len(w) < 4 or w in _ANCHOR_STOP or w in _ASK_FRAMING or w.isdigit() or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def _script_of(text: str) -> str:
    """"latin", "greek", "cyrillic" — the script most of the letters are in —
    or "" when there are no letters."""
    counts = {"latin": 0, "greek": 0, "cyrillic": 0}
    for ch in str(text or ""):
        if not ch.isalpha():
            continue
        o = ord(ch)
        if o < 0x250:
            counts["latin"] += 1
        elif 0x370 <= o < 0x400 or 0x1F00 <= o < 0x2000:
            counts["greek"] += 1
        elif 0x400 <= o < 0x530:
            counts["cyrillic"] += 1
    best = max(counts, key=counts.get)
    return best if counts[best] else ""


def _any_ask_word_in(words: List[str], hay: str) -> bool:
    hay_words = set(re.findall(r"\w+", hay))
    for w in words:
        if w in hay:
            return True
        if len(w) >= 6 and any(h[:5] == w[:5] for h in hay_words if len(h) >= 5):
            return True
    return False


def reply_off_topic(reply: str, context: str) -> Optional[str]:
    """None of the ask's subject words — nor a word sharing its first five
    letters ("moons"/"moon", "restarted"/"restart") — appears in the reply.
    None when the ask has no subject word or the reply carries one; else
    the words missed. A withhold, never a refute: a reply may paraphrase.

    §4IV, across languages: an English ask about "Spilios Oikonomidis" is
    answered by a Greek reply about "Σπήλιος Οικονομίδης" — the words are
    compared transliterated as well (the binder's `translit_greek`), and
    when the ask and the reply are written in different scripts and no
    word bridges them, a lexical check cannot tell paraphrase from drift
    and ABSTAINS (None) rather than withhold every cross-language answer
    (probe-5b, §4IT close: an English ask, a Greek reply, "none of the
    ask's subject words appears")."""
    words = ask_content_words(context)
    if not words:
        return None
    hay = normalize_for_containment(reply)
    if _any_ask_word_in(words, hay):
        return None
    t_words = [translit_greek(w) for w in words]
    if _any_ask_word_in(t_words, translit_greek(hay)):
        return None
    ask_script, reply_script = _script_of(" ".join(words)), _script_of(hay)
    if ask_script and reply_script and ask_script != reply_script:
        return None                        # different scripts, no bridge: not decidable by a word test
    return "none of the ask's subject words " + ", ".join(repr(w) for w in words[:4]) + " appears in the reply"


_ASK_MARKER = "|| USER REQUEST: "


def ask_of(context: str) -> str:
    """The CURRENT request inside the verifier's context. The turn loop hands
    `verify_claim` `constraint_note + request` ("ACTIVE PROJECT CONSTRAINTS
    (user-mandated, MUST hold): … || USER REQUEST: <text>"); the mechanical
    constraint and topic checks must read ONLY the request — reading the
    note is the §4FD constraint-bleed the turn-loop tier deliberately avoids
    (review §4IN consumer M1). The full context stays for the entity and
    figure audits, where a value named in the note is not the reply's
    invention."""
    c = str(context or "")
    i = c.rfind(_ASK_MARKER)
    return c[i + len(_ASK_MARKER):] if i >= 0 else c


def class_checks(reply: str, evidence: str, context: str = "") -> List[ClassFinding]:
    ask = ask_of(context)
    out: List[ClassFinding] = []
    try:
        from .objection import _claim_noise_markers
        marks = _claim_noise_markers(reply)
    except Exception:  # noqa: BLE001
        marks = []
    if marks:
        out.append(ClassFinding("artifact", "refute",
                                "machine noise in the reply: " + ", ".join(repr(m) for m in marks[:3])))
    try:
        from .turn_state_check import refute_turn_state
        for rule, msg in refute_turn_state(request=ask, reply=str(reply or "")):
            if rule != "empty_evidence":                 # tools_run is not visible here; see `evidence_all_failed`
                out.append(ClassFinding("constraint", "refute", f"{rule}: {msg}"))
    except Exception:  # noqa: BLE001
        pass
    if evidence_all_failed(evidence):
        out.append(ClassFinding("evidence", "withhold", "every evidence block is a tool failure"))
    off = reply_off_topic(reply, ask)
    if off:
        out.append(ClassFinding("topic", "withhold", off))
    return out

