"""Mechanical adjudication of a judge's objections — arithmetic before opinion.

Three rounds of escalation policy (rebuttal burden, tier routing,
truncation guard) all traded false alarms against real catches, because
all three were *arbitrating opinions*: the cheap judge says "wrong", the
main model says "fine", and any policy that trusts one loses what the
other was right about.

The measured way out: **a large share of the disagreement is not a matter
of opinion at all.** Across every completed bench arm, 145 distinct
false-alarm objections were collected and classified — **62 (43%) are
decidable by arithmetic or string search**:

    "The largest file size is 18,433 bytes, not exactly 18 KB."
        → 18433 / 1024 = 18.0 — a unit conversion, not a contradiction.
    "Iceland's population is stated as 396,000, but the evidence
     provides 396,960."
        → 0.24% apart — a rounding, not a fabrication.
    "Humidity around 28% is not in the evidence."
        → search the evidence: if "28" is there, the judge simply
          missed it.

Every one of those is a systematic class the small judge gets wrong and
that no prompt round fixed (two GEPA attempts refuted by the gate). They
need no model at all — they need `float()` and `in`.

So this module resolves what is decidable and returns UNRESOLVED for
what genuinely is not, leaving the expensive main-model call for real
ambiguity (subjective gloss, semantic drift, domain judgement).

DESIGN RULES, all conservative by construction — each clause below was
earned by a measured wrong verdict (2026-08-06/07 adversarial reviews):

* **Dismiss only on proof.** A numeric dismissal needs the relation to be
  provable from what is WRITTEN: a unit conversion requires the unit
  tokens to be present and to convert (`3 errors, not 21 errors` is not a
  ×7 "conversion" — nothing says weeks); a rounding must round exactly
  AND stay inside a graded relative-error budget (`1,000` is round(1,440)
  at one significant figure, and 44% wrong — that is not a rounding
  anyone meant). An absence dismissal requires EVERY cited atom present
  under boundary-anchored matching (`180` is not "present" inside
  `1800`).
* **Uphold only on proof.** A numeric contradiction is upheld only in an
  explicit contrast shape, with at least one strictly contradictory pair
  (all-equal numbers contradict nothing), no innocent-related pair, and
  the contradicting pair ANCHORED — one side found in the claim, the
  other in the evidence — so a judge-hallucinated figure cannot convict.
  Never in a version/date-labelled context.
* **Mixed signals are not proof.** One pair innocent and another pair
  contradictory, or some cited atoms present and others absent →
  UNRESOLVED, escalate.
* **Fail toward the existing pipeline.** UNRESOLVED means "carry on as
  before" — escalate. Whatever this module cannot prove is handled
  exactly as it was before it existed.

⚠ IT DOES CHANGE VERDICTS — an earlier version of this docstring claimed
it "can only remove work", which is false in both directions and worth
stating plainly, because it is the whole risk surface:

* a DISMISS returns CONFIRMED *without* the main-model call the old path
  would have made, at a confidence floor of 0.7 (see `_escalate_refute`).
  That is deliberate — a mechanical dismissal is PROVEN, not a soft
  overturn, so it is not subject to `_CONFIRM_WITHHELD_CONF_CAP` — but it
  is a verdict the legacy path never produced;
* an UPHOLD keeps the cheap judge's REFUTED and *suppresses* the
  escalation entirely, immunising it from the overturner.

Both are the point of the module, and both are why every rule below
decides in one direction only unless it can prove the other.

Kill switch: GHOST_VERIFY_OBJECTION_CHECK=0 (default ON — this ships
enabled, unlike the companion `GHOST_VERIFY_OVERTURN_QUOTE` /
`GHOST_VERIFY_TIER_ROUTING` policies, which lost to legacy and are OFF).
Note it is also skipped entirely when `retry` is set, which is the code
path (`verify_code_output`), and it sits behind
`GHOST_VERIFY_ESCALATE_REFUTE`.
"""

from __future__ import annotations

import itertools
import logging
import os
import re
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger("GhostAgent")

DISMISS = "dismiss"      # proven a false alarm
UPHOLD = "uphold"        # proven a real defect
UNRESOLVED = "unresolved"  # genuinely needs judgement

# Pair-relation grades used by the numeric rule (internal).
_EQUAL = "equal"          # same value — contradicts nothing
_RELATED = "related"      # proven rounding/conversion — innocent
_GRAY = "gray"            # round-shaped but too lossy to prove — escalate
_CONTRA = "contra"        # provable difference — candidate contradiction
_INCOMPARABLE = "incomparable"  # different dimensions — no arithmetic


def enabled() -> bool:
    return os.getenv("GHOST_VERIFY_OBJECTION_CHECK", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def dismiss_enabled() -> bool:
    """The DISMISS direction alone (UPHOLD keeps working regardless).

    ⚠ Motivated by the 2026-08-07 re-bench counters: UPHOLD measured 40
    protects / 2 damage (the workhorse), while DISMISS measured 3
    rescues / 9 DAMAGE — net negative. A corrupted claim whose judge
    objection happens to be phrased in mechanically-dismissible shape
    gets erased with no model call. The splice experiment
    (20260807T102629Z_upholdonly) then measured dismiss-OFF as strictly
    dominant: balanced 0.829 vs 0.819, 5 of the 9 damaged catches
    RECOVERED by the escalation, and all 3 rescues survived anyway
    (the overturner independently rescued the same false alarms) — at
    the cost of 12 extra escalation calls per 340 trials.

    ⚠ DEFAULT OFF since 2026-08-07 on that measurement: uphold-only is
    the shipping configuration. GHOST_VERIFY_OBJECTION_DISMISS=1
    re-enables the dismiss direction (the bench sweeps it).
    """
    return os.getenv("GHOST_VERIFY_OBJECTION_DISMISS", "0").strip().lower() \
        in ("1", "true", "yes", "on")


# Marker carried by rule 4's `why` string. The whole-verdict caller uses it
# to tell a NON-ASSERTIVE dismissal apart from the factual ones the
# 2026-08-07 measurement switched off.
_NONASSERTIVE_WHY = "objection targets a non-assertive fragment"


# Set once the opt-in warning has fired, so a per-call check does not spam
# the log (this predicate runs on every objection).
_NONASSERTIVE_OPT_IN_WARNED = False

# KNOWN-UNSAFE INPUT CLASSES, enumerated by adversarial review after v3.
# Kept in code, not just the journal, because this is what an operator
# flipping the switch is actually buying.
_NONASSERTIVE_KNOWN_HOLES = (
    "presupposition inside a pure intention ('restart the CRASHED service' "
    "asserts a crash) — unreachable by any lexical veto",
    "coordinators joining a second predication ('… and the token stays "
    "unrotated')",
    "unvetoed clause joiners: ASCII hyphen, parentheses, slash, and the "
    "unicode comma/enumeration marks",
    "copula-like verbs (stays/appears/seems/holds/contains) and the "
    "contracted 's",
    "a plan that is itself the defect (an unrequested destructive action, "
    "or a plan offered INSTEAD of the work)",
)


def _warn_nonassertive_opt_in() -> None:
    """Say plainly what enabling this rule buys. A flag that silently arms a
    rule with known laundering paths is a trap; a loud one is a choice."""
    global _NONASSERTIVE_OPT_IN_WARNED
    _NONASSERTIVE_OPT_IN_WARNED = True
    logger.warning(
        "GHOST_VERIFY_OBJECTION_NONASSERTIVE=1: objection rule 4 is ARMED. "
        "It suppresses a REFUTE with no main-model call, and adversarial "
        "review found it still launders %d known input classes: %s. Three "
        "hardening rounds did not converge (yield 3 -> 32 -> 34), so this "
        "is an EXPERIMENT, not a safety feature.",
        len(_NONASSERTIVE_KNOWN_HOLES), "; ".join(_NONASSERTIVE_KNOWN_HOLES))


def nonassertive_enabled() -> bool:
    """Rule 4's dismissal. **SHIPS DEFAULT-OFF** —
    `GHOST_VERIFY_OBJECTION_NONASSERTIVE=1` opts in.

    ⚠ WHY OFF, recorded so nobody re-argues it from the rule's appeal. The
    rule is a LEXICAL PROXY for a SEMANTIC property ("this fragment asserts
    nothing"). Two adversarial rounds each found the proxy leaking in a
    shape the author had not modelled — v1 fell to an objection that cited a
    real contradiction while also quoting the next step; v2 fell to `;` and
    `:` as clause joins, to inflection (`passed` vetoed, `passes` not), to
    relative clauses ("I'll fix the bug that broke prod"), and to framing
    words that spell an alignment complaint. v3 closes all 32 reported
    inputs, and the reviewer's judgement — which this default accepts — is
    that the next escape is one word cleverer, not that none exists.

    The asymmetry decides it: a false NEGATIVE costs one main-model call on
    a turn that was making one anyway; a false POSITIVE erases a real catch
    with NO model call and NO visible trace (`CONFIRMED@0.9`,
    `objection_dismissed=True` — indistinguishable in the ledger from a
    correct rescue). Two measured rescues do not buy that risk.

    ⚠ THE ORIGINAL JUSTIFICATION WAS RETRACTED. An LLM probe scored "0
    damage over 31 corrupted trials"; review showed the number was
    STRUCTURALLY incapable of being nonzero — the bench's `fault_fabrication`
    strings carry digits and result verbs (vetoed before rule 4 can act) and
    `fault_silent_failure` keeps the claim, so its objections are
    absence-shaped and settle at rule 2. The damage surface now lives in
    `tests/test_objection_nonassertive_damage.py`: 12 adversarial fixtures
    asserted at the per-objection AND whole-verdict levels, plus mutations
    (punctuation, tense, subordinator) so the suite pins the MECHANISM
    rather than the strings that motivated it. Its own limit is stated
    there: it proves no laundering on shapes we thought of.

    Before flipping this ON, sweep it in `scripts/verify_bench.py` beside
    `GHOST_VERIFY_OBJECTION_DISMISS` and let a measurement decide.
    """
    on = os.getenv("GHOST_VERIFY_OBJECTION_NONASSERTIVE", "0").strip(
        ).lower() in ("1", "true", "yes", "on")
    if on and not _NONASSERTIVE_OPT_IN_WARNED:
        _warn_nonassertive_opt_in()
    return on


def nonassertive_dismissal(reasons: Sequence[str]) -> bool:
    """True when EVERY reason behind a DISMISS came from rule 4.

    Deliberately unanimous: one factual dismissal mixed in and the whole
    verdict goes back through the gate the 2026-08-07 measurement closed."""
    items = [str(r or "") for r in (reasons or [])]
    # ⚠ Match the WHY, not the whole line. `resolve_refute` formats each
    # reason as "{issue} → {why}", and a substring test over the whole line
    # let an ISSUE TEXT containing this marker phrase spoof the exemption —
    # a factual rule-2 dismissal then reached production with the general
    # DISMISS gate closed (review M4, reproduced end-to-end; self-reachable
    # here, since any turn that discusses objection.py puts the literal
    # string into the claim). The why is appended last, so rsplit isolates
    # it even when the issue itself contains the separator.
    return bool(items) and all(
        r.rsplit(" \u2192 ", 1)[-1].startswith(_NONASSERTIVE_WHY)
        for r in items)


# A number with optional thousands separators / decimals.
# A COMPARABLE QUANTITY, not merely a digit string. The distinction is
# load-bearing and was found by measurement: the first version compared
# `SN851X` with `SN850X` (0.1% apart → "rounding") and `2025` with
# `2026`, dismissing two genuine contradictions. A number is a quantity
# only when it stands alone as a token — identifiers, years and dates
# are excluded below.
# ⚠ The trailing lookahead must exclude "." only when a DIGIT follows it.
# Blanket-excluding "." let the engine backtrack off a sentence-final
# period: "the evidence provides 396,960." matched as `396`, which then
# read as a ×1000 UNIT CONVERSION against a claimed 396,000 and DISMISSED
# a real catch. `(?!\.\d)` still rejects a truncated decimal ("18.4.3"
# yields nothing, as before) while letting terminal punctuation end a
# number normally.
_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9._/-])(-?\d[\d,]*(?:\.\d+)?)(?![A-Za-z0-9_/-])(?!\.\d)")

# The unit token immediately after a number, when there is one.
_UNIT_AFTER_RE = re.compile(r"[ \t]*(°?[A-Za-z%][A-Za-z%/°]{0,14})")

# Known units: token → (family, tuple of to-base factors). A tuple holds
# every convention in live use (KB is 1024 to an OS and 1000 to a disk
# vendor — either counts as the same quantity). Families with an empty
# factor tuple (temperature) are NOT linearly convertible: a pair mixing
# two of them is INCOMPARABLE, never dismissed and never upheld.
#
# ⚠ THE MAP IS THE FIX for the worst measured defect in this module's
# first version: `_UNIT_FACTORS` was a bare list of numbers (7, 24, 60,
# 100, 365, 1024…) applied to ANY pair — so "3 errors, not 21 errors"
# was dismissed as a ×7 "unit conversion", "2 hours, not 48 hours" as
# ×24 (both already in hours!), "5, not 500" as ×100. A conversion can
# only be claimed when the units are WRITTEN and actually convert.
_UNIT_MAP = {}
for _family, _entries in (
    ("bytes", {("byte", "bytes"): (1.0,),
               ("kb", "kib", "kilobyte", "kilobytes"): (1024.0, 1000.0),
               ("mb", "mib", "megabyte", "megabytes"): (1024.0 ** 2, 1e6),
               ("gb", "gib", "gigabyte", "gigabytes"): (1024.0 ** 3, 1e9),
               ("tb", "tib", "terabyte", "terabytes"): (1024.0 ** 4, 1e12)}),
    ("time", {("ms", "millisecond", "milliseconds"): (0.001,),
              ("s", "sec", "secs", "second", "seconds"): (1.0,),
              ("min", "mins", "minute", "minutes"): (60.0,),
              ("h", "hr", "hrs", "hour", "hours"): (3600.0,),
              ("day", "days"): (86400.0,),
              ("week", "weeks"): (604800.0,),
              ("year", "years"): (31536000.0,)}),
    ("distance", {("mm",): (0.001,), ("cm",): (0.01,),
                  ("m", "meter", "meters", "metre", "metres"): (1.0,),
                  ("km", "kilometer", "kilometers", "kilometre",
                   "kilometres"): (1000.0,),
                  ("mi", "mile", "miles"): (1609.344,),
                  ("ft", "feet", "foot"): (0.3048,)}),
    ("mass", {("mg",): (0.001,), ("g", "gram", "grams"): (1.0,),
              ("kg",): (1000.0,), ("lb", "lbs", "pound", "pounds"): (453.592,),
              ("oz",): (28.3495,)}),
    ("percent", {("%", "percent", "pct"): (1.0,)}),
    ("temp", {("°c", "°f", "celsius", "fahrenheit", "kelvin",
               "degree", "degrees", "deg"): ()}),
    # ⚠ Round-2 review: units MISSING from this map were silently
    # discarded, so "13 km/h vs 8 mph" (equal in reality) was convicted
    # as a numeric contradiction. Cover the units this agent's tool
    # output actually carries; a truly exotic unit still risks a
    # direct-compare conviction, which is why the map errs long.
    ("speed", {("km/h", "kmh", "kph"): (1 / 3.6,),
               ("mph",): (0.44704,), ("knot", "knots", "kn"): (0.514444,),
               ("m/s",): (1.0,)}),
    ("datarate", {("bps", "bit/s"): (1.0,), ("kbps", "kbit/s"): (1e3,),
                  ("mbps", "mbit/s"): (1e6,), ("gbps", "gbit/s"): (1e9,),
                  ("kb/s",): (8e3,), ("mb/s",): (8e6,), ("gb/s",): (8e9,)}),
    ("frequency", {("hz",): (1.0,), ("khz",): (1e3,), ("mhz",): (1e6,),
                   ("ghz",): (1e9,), ("rpm",): (1 / 60.0,)}),
    ("pressure", {("psi", "mmhg", "bar", "hpa", "kpa", "atm"): ()}),
):
    for _tokens, _factors in _entries.items():
        for _t in _tokens:
            _UNIT_MAP[_t] = (_family, _factors)

# Contexts where digits are LABELS, never quantities: years, ISO dates,
# d/m/y, times, identifiers (already excluded lexically above).
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_DATE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b|"
    r"\b\d{1,2}:\d{2}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\.?\s+\d{1,2}\b", re.I)

# Contrast shapes — the only frame in which a numeric mismatch is an
# assertion of contradiction rather than incidental figures.
_CONTRAST_RE = re.compile(
    r"\bnot\b|\bbut\b|\bvs\.?\b|instead of|rather than|whereas|differs|"
    r"disagree|mismatch|contradicts", re.I)

# Version-ish context: "18.4 vs 19 Beta 2" is a genuine difference, so
# numeric proximity must NOT dismiss it.
_VERSION_RE = re.compile(
    r"\bversion\b|\bv\d|\bbeta\b|\brc\b|\balpha\b|\brelease\b|\bbuild\b|"
    r"\bsemver\b|\bport\b|\bpid\b|\bcve\b|\bid\b|\b\d+\.\d+\.\d+\b", re.I)

# Absence complaints. ⚠ Every synonym missing from this list is a hole
# through which an absence complaint falls into the NUMERIC rule (whose
# bare "not"/"but" contrast test it will usually pass) and gets convicted
# as a "contradiction" without anyone looking at the evidence — that
# exact bug shipped twice (2026-08-06 with "not in", 2026-08-07 with
# "omits"/"never mentioned"/"fails to mention"). Err on the side of
# matching: a false absence-match routes to rule 2, which can only
# dismiss by FINDING the atoms or uphold by proving them absent — both
# strictly safer than rule 1's arithmetic on an omission complaint.
_ABSENCE_RE = re.compile(
    r"not (?:in|present|found|mentioned|supported|stated|listed|shown|"
    r"included|provided|reflected|given)"
    r"|unsupported|unsubstantiated"
    r"|no (?:mention|evidence|record|trace|reference)"
    r"|absent from|missing from|\b(?:is|are|was|were) missing"
    r"|does not (?:appear|include|contain|mention|state|show|list|"
    r"provide|give)"
    r"|doesn't (?:appear|include|contain|mention|state|show|list|"
    r"provide|give)"
    r"|\bomits?\b|\bomitted\b|\bomission\b"
    r"|fails? to (?:mention|state|include|note|report|provide|show)"
    r"|never (?:mentions?|mentioned|states?|stated|gives?|given|"
    r"provides?|provided|shows?|shown|appears?|lists?|listed|"
    r"includes?|included)"
    r"|\bnowhere\b|leaves? out|left out|\blacks\b|\blacking\b"
    r"|makes no mention", re.I)

# Machine-noise markers — checked for LITERAL presence in the claim,
# never taken on the judge's word (mirrors the anchor test in
# verifier.py, deliberately: one definition of "the claim contains
# machine noise", used by both).
#
# ⚠ "+++" / "---" are only diff noise when they look like diff HEADER
# lines. A bare "---" line is a markdown horizontal rule, and any marker
# inside a ```fenced``` block is FLAGGED presentation, not leaked noise —
# `_claim_noise_markers` strips fences first and requires the header
# shape for the +++/--- pair.
_ARTIFACT_MARKERS = ("@@", "<<<<", ">>>>", "\x1b[",
                     "<tool_call", "<parameter=", "<arg_key>")
_DIFF_HEADER_RE = re.compile(r"(?m)^(\+\+\+|---)[ \t]\S")
_FENCED_BLOCK_RE = re.compile(r"```.*?```", re.S)
# Inline code spans are FLAGGED presentation exactly like fences
# (round-2 M3: "the hunk header `@@ -1,3 +1,3 @@` marks…" was convicted
# for the @@ inside its own backticks).
_INLINE_CODE_RE = re.compile(r"`[^`\n]{1,200}`")

# The noise-allegation shapes that arm rule 3. ⚠ Anchored and
# context-gated: the first version used the bare fragment `artifact` and
# unanchored `ansi`, so "the wrong build artifact" (a content objection
# about a filename) and "the claim's exp-ANSI-on of scope" (semantic
# drift) both fired rule 3 and were DISMISSED with no model call.
# "artifact"/"marker" alone are English; they only mean machine noise
# next to a noise word.
_NOISE_TERM_RE = re.compile(
    r"diff markers?|\bansi\b|escape (?:code|sequence)s?|"
    r"control characters?|tool[- ]call framing|merge conflict markers?",
    re.I)
_ARTIFACT_WORD_RE = re.compile(r"\b(?:artifacts?|markers?)\b", re.I)
_NOISE_CONTEXT_RE = re.compile(
    r"\bleak(?:s|ed|ing)?\b|\braw\b|\bunflagged\b|\bstray\b|\bmachine\b|"
    r"\bnoise\b|\bdiff\b|\bansi\b|\bescape\b|\bcontrol\b|\bconflict\b",
    re.I)

# ⚠ WHO omitted it decides WHERE to look (round-2 C5). "The reply
# omits the 28% humidity that the evidence provides" is a CLAIM-side
# complaint: searching the EVIDENCE for 28% and dismissing "the judge
# missed it" read the proof BACKWARDS — the atom being in the evidence
# is exactly what makes the judge right. When the omission verb's
# subject is the reply/claim, rule 2 searches the CLAIM: atoms present
# there prove the complaint false (dismiss); atoms absent leave a
# materiality question (is the omission refute-worthy?) that only the
# strong model can answer — UNRESOLVED, never a mechanical uphold.
# ⚠ The gap is TEMPERED (round-3 F1): a plain `[^.;,\n]{0,40}` bridged
# "the claim states 55% but the evidence omits it" — the claim-noun
# paired with the EVIDENCE's verb across the intervening subject, the
# complaint routed to the claim-side search, and the claim "proved" the
# omission complaint false by containing its own figure. Circular:
# CONFIRMED@0.9 on a correct refute, zero calls. The gap refuses to
# cross an evidence-noun, and the verb must not hand off to an
# EVIDENCE-NOUN AGENT in any prepositional form.
#
# ⚠ The agent test names the evidence-noun, not the preposition
# (round-4): the first version blocked a bare `\bby\b`, which (a) missed
# "omitted FROM the tool output" / "left out OF the evidence" — the
# same circular dismiss through a different preposition — and (b) fired
# on "the 3 risks listed by THE USER", misrouting a genuine claimward
# completeness complaint to the evidence-side search where the risks
# are naturally present → "judge missed it". Both directions were
# reproduced end-to-end.
# ⚠ ACTIVE and PASSIVE are different grammars (round-5, resolving the
# round-4 window tension for good). A single verb list with one agent
# lookahead could not be windowed correctly: an ACTIVE verb after a
# claim-noun subject takes a SOURCE preposition ("the reply omits the
# figure from the evidence digest" — still claimward), while only a
# PASSIVE participle takes an evidence AGENT ("was omitted from the
# evidence" — evidence-side). Any single window either let adverb
# padding revive the circular dismiss ("omitted deliberately from the
# evidence") or misrouted active complaints by paraphrase length
# ("omits the figure of 28% that the evidence provides"). Three
# branches:
#   A. claim-noun … ACTIVE verb — claimward, NO agent lookahead
#      ("omitted"/"left out" join only when NOT aux-preceded, i.e. the
#      active past "the claim omitted X", not "is omitted");
#   B. claim-noun … AUX + participle — claimward only without an
#      evidence-agent PP (commas allowed inside the lookahead: "omitted,
#      without explanation, from the evidence" is the same agent);
#   C. participle + from/of + CLAIM-noun ("the 28% figure was omitted
#      from the reply") — claimward with no leading claim-noun at all.
_CLAIM_NOUNS_P = r"(?:reply|claim|response|answer|summary|report)"
_EV_NOUNS_P = (r"(?:evidence|tool|outputs?|digest|logs?|results?|"
               r"snippet|excerpt)")
_NO_AUX = (r"(?<!is )(?<!are )(?<!was )(?<!were )(?<!been )(?<!being )"
           r"(?<!get )(?<!gets )(?<!got )")
_CLAIMWARD_RE = re.compile(
    # A. active
    r"\b" + _CLAIM_NOUNS_P + r"\b"
    r"(?:(?!\b" + _EV_NOUNS_P + r"\b)[^.;,\n]){0,40}?"
    r"\b(?:" + _NO_AUX + r"omits?|" + _NO_AUX + r"omitted|"
    r"lacks|lacking|" + _NO_AUX + r"leaves? out|" + _NO_AUX + r"left out|"
    r"fails? to (?:mention|state|include|note|report|provide|show)|"
    r"makes no mention|"
    r"does not (?:mention|include|contain|state|show|list|provide|give)|"
    r"doesn't (?:mention|include|contain|state|show|list|provide|give)|"
    r"never (?:mentions?|states?|shows?|lists?|includes?|provides?|"
    r"gives?))\b"
    # B. passive with pre-verb claim-noun
    r"|\b" + _CLAIM_NOUNS_P + r"\b"
    r"(?:(?!\b" + _EV_NOUNS_P + r"\b)[^.;,\n]){0,40}?"
    r"\b(?:is|are|was|were|been|being|gets?|got)\s+"
    r"(?:omitted|left out)\b"
    r"(?![^.;\n]{0,28}?\b(?:by|from|of|in)\b[^.;\n]{0,20}?"
    r"\b" + _EV_NOUNS_P + r"\b)"
    # C. participle targeting a claim-noun
    r"|\b(?:omitted|left out)\b[^.;,\n]{0,12}?\b(?:from|of)\b"
    r"[^.;,\n]{0,20}?\b" + _CLAIM_NOUNS_P + r"\b",
    re.I)

# Quoted fragments the objection points at.
_QUOTED_RE = re.compile(r"['\"“”‘’]([^'\"“”‘’]{3,80})['\"“”‘’]")

# Relative tolerance for "this is the same number, rounded" without any
# structural rounding match, and for unit-converted comparison.
_ROUND_TOL = 0.02
# A rounding that discards decimals (2.9 → 3) may cost more relative
# error than one that zeroes trailing digits (396,960 → 400,000), because
# integer-rounding small quantities is normal writing. Beyond these
# budgets a round-shaped match proves nothing → GRAY, escalate.
_DECIMAL_ROUND_MAX_ERR = 0.20
_MAGNITUDE_ROUND_MAX_ERR = 0.05
_CONVERTED_GRAY_ERR = 0.10
# O(n²) pair scan cap — an 80KB pathological issue held ~20k numbers and
# took 3.7s; a real judge objection holds fewer than ten.
_MAX_NUMS = 25


def _truncation_floor() -> float:
    """How much of the evidence must have been cut before an absence
    complaint stops being mechanically decidable.

    ⚠ SINGLE SOURCE OF TRUTH, shared with the verifier's truncation guard
    (`_truncation_min_severity`). This was a bare 0.25 literal here while
    the guard's copy was env-tunable, so setting
    GHOST_VERIFY_TRUNCATION_MIN_SEVERITY moved one threshold and silently
    left the other behind — the two would then disagree about the same
    evidence. Read per call, so a test or a bench sweep can move it.
    """
    try:
        v = float(os.getenv("GHOST_VERIFY_TRUNCATION_MIN_SEVERITY", "0.25"))
    except ValueError:
        return 0.25
    # ⚠ Clamp to [0, 1]: severity is a fraction. A -1 made the guard
    # downgrade all-absence refutes over fully INTACT evidence; a 2.0
    # made the absence rule UPHOLD over an 85%-cut digest with the
    # reason "absent from intact evidence". Misconfig-only, but the var
    # is documented as bench-sweepable, so a sweep typo must not invert
    # the semantics.
    return max(0.0, min(1.0, v))


# Glued number+unit ("18KB", "26.6°C") is invisible to `_NUM_RE` (its
# trailing lookahead rejects a following letter), which halved rule 1's
# coverage — judges glue units often. Pre-spacing is limited to
# MULTI-CHAR known units: single letters would turn "the 80s" into
# "80 s" (seconds).
_GLUED_UNIT_RE = re.compile(
    r"(\d)(kb|kib|mb|mib|gb|gib|tb|tib|ms|km/h|kmh|kph|mph|m/s|km|cm|mm|"
    r"kg|mg|hz|khz|mhz|ghz|rpm|bps|kbps|mbps|gbps|°c|°f)\b", re.I)


def _numbers_with_units(text: str) -> List[Tuple[float, str, Optional[str]]]:
    """Standalone COMPARABLE quantities as (value, raw, unit-or-None).

    The raw spelling is kept because PRECISION decides rounding vs
    contradiction; the unit token (the word right after the number, when
    it is a known unit) is what licenses a conversion dismissal.
    """
    out: List[Tuple[float, str, Optional[str]]] = []
    text = _GLUED_UNIT_RE.sub(r"\1 \2", str(text or ""))
    for m in _NUM_RE.finditer(text):
        raw = m.group(1).rstrip(",")
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        # A bare 4-digit year is a label, not a magnitude.
        if 1900 <= v <= 2100 and "." not in raw and "," not in raw:
            continue
        unit: Optional[str] = None
        um = _UNIT_AFTER_RE.match(text, m.end())
        if um:
            token = um.group(1).lower().rstrip(".,;:")
            if token in _UNIT_MAP:
                unit = token
        out.append((v, raw, unit))
        if len(out) >= _MAX_NUMS:
            break
    return out


def _numbers(text: str) -> List[Tuple[float, str]]:
    """Back-compat view of `_numbers_with_units` without the units."""
    return [(v, raw) for v, raw, _u in _numbers_with_units(text)]


def _has_label_digits(text: str) -> bool:
    """True when the objection's numbers live in a LABEL context —
    years, dates, times, versions, ports, ids. Numeric proximity is
    meaningless there ("2025 vs 2026" is a different year, not a
    rounding)."""
    return bool(_YEAR_RE.search(text) or _DATE_RE.search(text)
                or _VERSION_RE.search(text))


def _sig_figs(raw: str) -> int:
    """Significant figures in a number AS WRITTEN."""
    s = str(raw).strip().lstrip("+-").replace(",", "")
    if "." in s:
        return len(s.replace(".", "").lstrip("0")) or 1
    return len(s.rstrip("0").lstrip("0")) or 1


def _grade_direct(a: float, ra: str, b: float, rb: str) -> str:
    """Grade two same-dimension values: _EQUAL / _RELATED / _GRAY /
    _CONTRA.

    ⚠ PRECISION IS THE DISCRIMINATOR, not proximity — a bare 2% tolerance
    dismissed "26.7°C vs evidence 26.6°C", a fact SWAP. A genuine
    rounding LOSES precision, so equal significant figures + any gap is a
    contradiction, however small.

    ⚠ PRECISION, NOT MAGNITUDE, picks which value is the rounded one —
    deriving coarse/fine from `sorted()` recognised only roundings that
    round DOWN ("27, not 26.6" was upheld as a contradiction).

    ⚠ AN EXACT round() MATCH IS NOT PROOF EITHER: round(1440, -3) == 1000
    and round(2499, -3) == 2000, yet "1,000 minutes" for 1,440 (a day!)
    is a 44% misstatement nobody calls a rounding. Round-shaped matches
    carry graded relative-error budgets; beyond them the pair is _GRAY —
    round-shaped but unproven, so it escalates rather than dismisses.
    """
    if a == b:
        return _EQUAL
    fa, fb = abs(a), abs(b)
    if fa == 0.0 or fb == 0.0:
        return _GRAY            # relative error undefined — not provable
    # ⚠ Checked BEFORE the precision gate (its first placement sat after
    # the equal-sig-figs CONTRA return, which fired first on "3" vs
    # "3,000,000" — both one significant figure): a pair an exact power
    # of 1000 apart is likelier a magnitude-suffix abbreviation
    # ("3 m users" = 3,000,000) than a thousand-fold misstatement —
    # GRAY, escalate (round-2 M2).
    lo_v, hi_v = sorted((fa, fb))
    if lo_v > 0:
        ratio = hi_v / lo_v
        # Binary powers included (round-3 F2): a KB/MB judge objection
        # with the unit word dropped sits ×1024/×1024² apart — 2.4% and
        # 4.9% off the decimal powers, outside the 2% net.
        for mag in (1e3, 1024.0, 1e6, 1024.0 ** 2, 1e9, 1024.0 ** 3):
            if abs(ratio - mag) / mag <= _ROUND_TOL:
                return _GRAY
    sf_a, sf_b = _sig_figs(ra), _sig_figs(rb)
    if sf_a == sf_b:
        return _CONTRA          # equal precision ⇒ a real difference
    coarse_raw, fine_raw = (ra, rb) if sf_a < sf_b else (rb, ra)
    coarse = abs(float(coarse_raw.replace(",", "")))
    fine = abs(float(fine_raw.replace(",", "")))
    rel = abs(fine - coarse) / fine
    if rel <= _ROUND_TOL:
        return _RELATED
    # ⚠ HALF-STEP DISTANCE, not round() equality (round-2 review):
    # Python's round() is banker's rounding over binary floats, so
    # "1,500 for 1,450" (round(1450,-2)=1400) and "0.4 for 0.35"
    # (round(0.35,1)=0.3) were convicted as contradictions while
    # "3 for 2.9" dismissed — an indefensible boundary. `coarse` is a
    # legitimate rounding of `fine` at step `s` iff it is a multiple of
    # `s` within half a step of `fine`, whichever way the half falls.
    decimal_match = False
    magnitude_match = False
    diff = abs(fine - coarse)
    for places in range(0, 6):
        step = 10.0 ** -places
        eps = step * 1e-9
        if diff <= step / 2 + eps:
            decimal_match = True
            break
    if not decimal_match:
        try:
            width = len(str(int(fine)))
        except (ValueError, OverflowError):
            width = 0
        for digits in range(1, 6):
            k = width - digits
            if k <= 0:
                break
            step = 10.0 ** k
            if (diff <= step / 2 + step * 1e-9
                    and abs(coarse / step - round(coarse / step)) < 1e-9):
                magnitude_match = True
                break
    if decimal_match:
        return _RELATED if rel <= _DECIMAL_ROUND_MAX_ERR else _GRAY
    if magnitude_match:
        return _RELATED if rel <= _MAGNITUDE_ROUND_MAX_ERR else _GRAY
    return _CONTRA


def _grade_pair(a: float, ra: str, ua: Optional[str],
                b: float, rb: str, ub: Optional[str]) -> str:
    """Grade a pair of quantities, honouring their written units.

    * both bare, or only one carries a unit → direct comparison (the
      judge is comparing them; assume the same dimension);
    * same unit token → direct comparison;
    * different units, same family, factors known → convert to the base
      unit and compare by relative error (written precision is
      meaningless across a conversion);
    * different units otherwise (different family, or a factorless
      family like temperature) → _INCOMPARABLE: no arithmetic can
      dismiss OR convict across that pair.
    """
    if ua is None or ub is None or ua == ub:
        grade = _grade_direct(a, ra, b, rb)
        # ⚠ ONE-SIDED unit rescue (round-3 F2): "18,433, not exactly
        # 18 KB" — the judge dropped the word "bytes", the pair
        # direct-compared, ratio 1024 slipped past the decimal
        # power-of-1000 net, and the module's own FLAGSHIP false-alarm
        # class was mechanically UPHELD. When exactly one side carries a
        # unit and the direct grade says contradiction, try reading the
        # bare number as the family's BASE unit: if the conversion lands
        # (18 × 1024 = 18432 ≈ 18433), the pair is RELATED after all.
        # Fires on GRAY as well as CONTRA: the binary-power net grays a
        # ×1024 pair before this rescue can look at it, and gray means
        # unproven — a conversion that actually LANDS is proof.
        if grade in (_CONTRA, _GRAY) and (ua is None) != (ub is None):
            unit = ua or ub
            _fam, factors = _UNIT_MAP[unit]
            united, bare = (abs(a), abs(b)) if ua else (abs(b), abs(a))
            for f in factors:
                conv = united * f
                hi = max(conv, bare)
                if hi > 0 and abs(conv - bare) / hi <= _ROUND_TOL:
                    return _RELATED
        return grade
    fam_a, factors_a = _UNIT_MAP[ua]
    fam_b, factors_b = _UNIT_MAP[ub]
    if fam_a != fam_b or not factors_a or not factors_b:
        return _INCOMPARABLE
    if a == b == 0:
        return _EQUAL
    best = None                     # (rel, f_a, f_b)
    for f_a in factors_a:
        for f_b in factors_b:
            base_a, base_b = abs(a) * f_a, abs(b) * f_b
            hi = max(base_a, base_b)
            if hi == 0:
                continue
            rel = abs(base_a - base_b) / hi
            if best is None or rel < best[0]:
                best = (rel, f_a, f_b)
    if best is None:
        return _GRAY
    best_rel, f_a, f_b = best
    if best_rel == 0.0 and f_a == f_b:
        # ⚠ Equal at the SAME scale is a spelling variant, not a
        # conversion — "stated as 5%, but the evidence says 5 percent"
        # names one quantity twice, and if the objection is wrong it is
        # wrong about something NON-numeric (the Paris-vs-Athens drift
        # class), which is the strong model's job. Equal ACROSS scales
        # ("48 MB vs 50,331,648 bytes", f 1024² vs 1) is a genuine
        # conversion the judge misread as a contradiction — that IS the
        # proof, and it stays dismissable below.
        return _EQUAL
    if best_rel <= _ROUND_TOL:
        return _RELATED
    if best_rel <= _CONVERTED_GRAY_ERR:
        return _GRAY
    return _CONTRA


# The packer's own truncation marker, which is METADATA, not evidence.
# Fallback shape only — the authoritative pattern is compiled in agent.py
# and imported lazily below (importing it eagerly would be circular:
# agent → verifier → objection).
_PACKER_MARK_FALLBACK_RE = re.compile(
    r"…\[PACKER CUT#[0-9a-f]+:\s*\d+\s+of\s+\d+\s+chars shown\]")


def _strip_packer_marks(evidence: str) -> str:
    """Remove the packer's truncation markers before searching evidence.

    ⚠ The marker carries DIGITS — a random 8-hex nonce and the two byte
    counts ("…[PACKER CUT#ed2d35ed: 400 of 5400 chars shown]"). Searching
    it as evidence meant a cited number could be "found" inside the
    packer's own bookkeeping and a proven-real absence DISMISSED with no
    model call. The byte counts make that DETERMINISTIC for any objection
    citing them; the nonce added a ~3%-per-process-start random component
    that surfaced as a flaky test rather than as an obvious bug.
    """
    text = str(evidence or "")
    try:                                    # authoritative pattern
        from .agent import _TRUNCATION_MARK_RE
        return _TRUNCATION_MARK_RE.sub(" ", text)
    except Exception:                       # pragma: no cover - import guard
        return _PACKER_MARK_FALLBACK_RE.sub(" ", text)


def _canon(s: str) -> str:
    """One canonical text form used for BOTH atoms and evidence.

    ⚠ Asymmetric normalization was a measured bug: atoms had their digit
    commas STRIPPED ("396,960" → "396960") while evidence had commas
    turned into SPACES ("396,960" → "396 960"), so the absence rule could
    never find a ≥4-digit number in comma-grouped evidence and upheld the
    module's own flagship example as a "real defect". Digit-grouping
    commas are deleted on both sides; curly quotes straighten; hyphens
    and whitespace collapse to single spaces; a space between a digit and
    a unit letter is removed so "18 KB" and "18KB" meet in the middle.
    """
    s = str(s or "").strip().lower()
    s = s.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    s = re.sub(r"(?<=\d),(?=\d)", "", s)     # 396,960 → 396960
    # ⚠ Hyphens collapse to spaces only BETWEEN LETTERS (round-2 M1):
    # "partly-cloudy" must meet "partly cloudy", but a blanket collapse
    # turned "SHA-256" into "sha 256" and the absence rule then "found"
    # a cited 256 inside a checksum name — a manufactured presence that
    # dismissed a real catch. Digit-adjacent hyphens are identifier
    # glue (SHA-256, UTF-16, ISO-8601) and stay put; `_number_present`
    # refuses digits reached through them.
    s = re.sub(r"(?<=[a-z])[\-–—]+(?=[a-z])", " ", s)
    s = re.sub(r"[\s,;]+", " ", s)
    s = re.sub(r"(?<=\d) (?=[a-z%°])", "", s)  # 18 kb → 18kb
    return s


def _cited_atoms(issue: str) -> List[Tuple[str, bool]]:
    """The concrete things an absence complaint says are missing, as
    (atom, is_number): quoted fragments and bare numbers."""
    atoms: List[Tuple[str, bool]] = [
        (q.strip(), False) for q in _QUOTED_RE.findall(issue)
        if q.strip() and len(q.strip()) >= 2]
    # ⚠ Numeric atoms are exempt from the length filter (round-2 m4):
    # boundary matching makes a single digit safe, and dropping "5" made
    # '"5 nodes" is not in the evidence' undecidable — or worse, C4's
    # enabler: the bare number that would have forced MIXED was
    # filtered, leaving only the substring-matched quote.
    atoms += [(raw.replace(",", ""), True)
              for _v, raw, _u in _numbers_with_units(issue) if raw.strip()]
    return atoms[:25]


def _number_present(atom: str, canon_text: str) -> bool:
    """Boundary-anchored presence for a NUMERIC atom.

    ⚠ Raw substring matching was a measured bug: "180" was "found" inside
    "1800 rpm", "12" inside "4128", and comma-normalization manufactured
    "800" out of "1,800" — each one a real absence dismissed and a true
    catch erased. A cited number is present only where it stands as a
    whole number: not preceded by a digit or decimal point, not followed
    by more digits or a decimal continuation ("28" is absent from
    "28.5").

    ⚠ THE ATOM IS CANONICALISED TOO (2026-08-09). `_canon` documents
    itself as "one canonical text form used for BOTH atoms and evidence",
    and `_atom_present`'s string branch honours that — but this numeric
    branch escaped the atom RAW, so the haystack had its digit-grouping
    commas stripped while the needle kept them. `'18,433'` was therefore
    NOT FOUND in evidence that literally reads `manage.py 18,433 bytes`,
    while `'18433'` was. Every comma-grouped figure a judge cited failed
    to anchor, which silently disarmed the numeric UPHOLD rule (whose
    proof requires one side traceable to the claim and the other to the
    evidence) on exactly the ≥4-digit numbers it most needed.

    This is the SAME asymmetry `_canon`'s own docstring records as a
    previously-measured bug, surviving in the one path that did not call
    it. `_canon` is idempotent on numerals, so canonical atoms are
    unaffected, and the boundary guards above still apply to the
    canonical form — `'800'` is still refused inside `'1800'`.
    """
    a = re.escape(_canon(atom).lstrip("+-"))
    return bool(re.search(
        rf"(?<![\d.])(?<![A-Za-z][\-–])(?<![A-Za-z][\-–] ){a}(?!\.?\d)",
        canon_text))


def _atom_present(atom: str, is_number: bool, canon_text: str) -> bool:
    if is_number:
        return _number_present(atom, canon_text)
    a = _canon(atom)
    if not a:
        return False
    # ⚠ A quoted atom with digit EDGES gets the same boundary guards as
    # a bare number (round-2 C4): plain substring found "8 gb" inside
    # "18 gb" (via the digit-unit glue) and '"3 users"' inside "13
    # users" — quoted claim fragments are how judges cite figures, and
    # each false presence dismissed a real absence catch.
    pre = r"(?<![\d.])" if a[0].isdigit() else ""
    suf = r"(?!\.?\d)" if a[-1].isdigit() else ""
    return bool(re.search(pre + re.escape(a) + suf, canon_text))


def _claim_noise_markers(claim: str) -> List[str]:
    """Machine-noise markers LITERALLY present in the claim, with the
    flagged shapes excluded.

    ⚠ Two measured false convictions shaped this: a claim presenting a
    diff inside a ```fenced``` block had its "+++" counted as leaked
    noise (a fenced diff is exactly how a claim SHOULD show a diff), and
    a bare "---" — the markdown horizontal rule — read as a diff header.
    Fences are stripped first; +++/--- count only in diff-header shape
    ("--- a/file"), which a horizontal rule never has.
    """
    body = _FENCED_BLOCK_RE.sub(" ", str(claim or ""))
    # An UNCLOSED trailing fence (claim truncated mid-fence) is still
    # flagged presentation — strip from the dangling ``` to the end
    # (round-2 m6).
    if body.count("```") % 2 == 1:
        body = body[:body.rfind("```")]
    body = _INLINE_CODE_RE.sub(" ", body)
    present = [m for m in _ARTIFACT_MARKERS if m in body]
    hm = _DIFF_HEADER_RE.search(body)
    if hm:
        present.append(hm.group(1))
    return present


def _is_noise_allegation(text: str) -> bool:
    """Does the objection allege MACHINE NOISE in the claim (rule 3)?

    Context-gated: "artifact" and "marker" are ordinary English ("the
    wrong build artifact" is a content objection) and only mean noise
    next to a noise word. An unanchored `artifact|ansi` regex fired on
    both — and on "exp-ansi-on" — and dismissed real objections.
    """
    if _NOISE_TERM_RE.search(text):
        return True
    return bool(_ARTIFACT_WORD_RE.search(text)
                and _NOISE_CONTEXT_RE.search(text))


# ── Non-assertive fragments (rule 4, §4BD) ──────────────────────────
# A stated NEXT STEP or a QUESTION back to the user asserts nothing about
# the world, so no evidence can contradict it.
#
# ⚠ v2 (adversarial review, same day). v1 asked only "does SOME span in the
# objection look like an intention?" and then dismissed the WHOLE objection.
# Three reproduced laundering paths killed it: an objection citing a real
# contradiction that ALSO quoted the next step; a multi-sentence span whose
# first sentence was a fabrication ("The endpoint is reachable … I'll add
# the retry next"); and absence complaints whose synonym is missing from
# `_ABSENCE_RE` ("not corroborated by the evidence") reaching rule 4 and
# turning an UPHOLD into a DISMISS. v2 therefore requires the objection to
# reduce ENTIRELY to one non-assertive CLAUSE:
#   (1) the fragment is a single clause that STARTS with the marker and
#       contains no internal sentence break;
#   (2) it is present verbatim in the claim, digit-free, result-verb-free;
#   (3) everything else in the objection is pure FRAMING — every remaining
#       word drawn from a small allowlist. Any content word (evidence,
#       support, contradiction, a number, …) means the objection is saying
#       something more than "this is a plan", and it escalates.

# (1) whole-fragment shapes. The marker must OPEN the clause, and `[^.!?]*`
# forbids an internal sentence break — so an assertion cannot precede or
# follow the intention inside one fragment.
# Anything that can START A SECOND PREDICATION inside the fragment. v2
# assumed `[^.!?]*` bounded a clause; it does not — `;` `:` `,` a dash, a
# newline, or a bare relativizer all carry a full assertion ("I'll retry
# the fetch; every exit node is blocked", "I'll fix the bug that broke
# prod"). These three closed FUNCTION-WORD classes are enumerable in a way
# the open class of content verbs is not.
_CLAUSE_BREAK_RE = re.compile(r"[;:,\u2014\u2013\n\r\u2026]|\.\.\.")
_SUBORDINATOR_RE = re.compile(
    r"\b(?:that|which|who|whom|whose|because|since|although|though|while|"
    r"whereas|when|whenever|after|before|until|unless|if|as|so\s+that|"
    r"now\s+that|given\s+that|due\s+to|owing\s+to)\b", re.I)
# A copula/auxiliary is the spine of a predicative assertion ("the tunnel
# IS up", "the credentials WERE unchanged"). An intention clause needs none.
_COPULA_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|am|has|have|had|does|do|did|"
    r"isn't|aren't|wasn't|weren't|hasn't|haven't|didn't|doesn't)\b", re.I)

_NONASSERTIVE_CLAUSE_RE = re.compile(
    r"^(?:"
    r"I(?:'|\u2019)?(?:ll|\s+will|\s+shall|\s+am\s+going\s+to|\s+plan\s+to|"
    r"\s+intend\s+to|\s+can\s+(?:retry|rerun|re-run|look|check|open|fix))"
    r"|Let\s+me"
    r"|Next(?:,)?\s+I(?:'|\u2019)?(?:ll|\s+will)"
    r"|Want\s+me\s+to|Shall\s+I|Should\s+I|Do\s+you\s+want\s+me\s+to"
    r"|Would\s+you\s+like\s+me\s+to"
    r")\b[^.!?]*[.!?]?$", re.I)

# (2) a fragment asserting a RESULT is not a pure intention. Extended after
# the review: v1 covered past-tense verbs of DOING but not the predicative
# assertions where fabrications actually live ("the config is valid").
_RESULT_ASSERTION_RE = re.compile(
    # ⚠ INFLECTED forms only. After a modal ("I'll re-run the test") the
    # verb is a BARE INFINITIVE describing a future action and asserts
    # nothing; the -ed / -s / irregular-past forms are what assert a
    # result ("review passes", "the check ran"). v3 vetoed both and killed
    # the rule's own target shape — the distinction is the point.
    r"\b(?:passes|passed|passing|fails|failed|failing|"
    r"succeeds|succeeded|works|worked|working|"
    r"verifies|verified|validates|validated|confirms|confirmed|"
    r"tests|tested|completes|completed|finishes|finished|"
    r"fixes|fixed|resolves|resolved|runs|ran|"
    r"ensures|ensured|deploys|deployed|installs|installed|"
    r"creates|created|writes|wrote|saves|saved|returns|returned|"
    r"produces|produced|finds|found|shows|shown|showed|"
    r"reproduces|reproduced|exists|existed|matches|matched|"
    r"detects|detected|broke|broken|"
    r"reachable|valid|invalid|healthy|green|empty|clean|correct|incorrect|"
    r"already|unchanged|successful|successfully)\b", re.I)

# (3) framing vocabulary. Every word left in the objection once the
# fragment is removed must appear here, or the objection is asserting
# something beyond "this is a plan/question" and must escalate.
# ⚠ DELIBERATELY EXCLUDES evidence / support / corroborated / verifiable /
# substantiated: those make it a CONTENT complaint, which is rule 2's
# business (and where `_ABSENCE_RE`'s synonym gaps live — see the C3 path
# in the review). Excluding them is what keeps a missing synonym escalating
# instead of silently dismissing.
_FRAMING_WORDS = frozenset("""
a an the this that these those it its
is are was were be being been
statement sentence claim reply response text fragment part line phrase
agent assistant model user
states state stated asserts assert asserted says say said proposes propose
proposed promises promise promised implies imply implied includes include
included ends end ending offers offer offered adds add added mentions
mention mentioned contains contain
future next plan plans step steps action actions intention intentions
intent promise question offer suggestion
not no non never merely only just simply also and or but of to for in on
with as at from by about
which what who whom whose when where why how
""".split())

# ⚠ The allowlist is DELIBERATELY INCOMPLETE, and that is the safe
# direction: an unlisted word makes the objection escalate (a false
# NEGATIVE — one extra main-model call), whereas a permissive proxy would
# dismiss real complaints (a false POSITIVE — a laundered defect). This
# project's own rule: when the distinction is semantic, use an explicit
# allowlist and accept the gaps; a clever structural proxy fails silently
# and broadly. Grow it only from objections observed in the wild.

_WORD_RE = re.compile(r"[A-Za-z]+")

# ASCII `isdigit()` misses non-ASCII numerals and spelled-out numbers, both
# of which can carry a fabricated quantity into a "plan" (review N5).
_NUMBER_WORD_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"dozen|hundred|thousand|million|billion|half|twice|first|second|third)\b",
    re.I)


def _has_numeral(text: str) -> bool:
    """Any digit (ASCII or not), numeric character, or number word."""
    for ch in text or "":
        if ch.isdigit() or ch.isnumeric():
            return True
    return bool(_NUMBER_WORD_RE.search(text or ""))

# Quoted spans inside an objection: 'like this', "like this", `like this`,
# and the curly variants a judge emits.
#
# ⚠ CONTRACTIONS AND POSSESSIVES, both found by review. A naive
# `'([^']+)'` splits on the apostrophe inside the quoted text — the very
# fragments this rule targets are contractions ("'I'll open …'" captured as
# `ll open …`, which no longer matches the intent pattern). And a
# POSSESSIVE before the quote ("The reply's statement 'I'll …'") consumed
# the opening delimiter. So: an apostrophe followed by a LETTER is a
# contraction, and an opening delimiter may not be preceded by one.
_QUOTED_SPAN_RE = re.compile(
    r"(?<![A-Za-z0-9])'((?:[^']|'(?=[A-Za-z])){6,200}?)'"
    r"|\"([^\"]{6,200})\""
    r"|`([^`]{6,200})`"
    r"|(?<![A-Za-z0-9])\u2018((?:[^\u2019]|\u2019(?=[A-Za-z])){6,200}?)\u2019"
    r"|\u201c([^\u201d]{6,200})\u201d")

# Bounds. A "fragment" longer than this is a paragraph, not a clause.
_FRAGMENT_MIN_CHARS = 12
_FRAGMENT_MAX_CHARS = 160


def _fragment_is_nonassertive(frag: str, claim_canon: str) -> bool:
    """One clause, verbatim in the reply, asserting nothing."""
    f = (frag or "").strip()
    if not (_FRAGMENT_MIN_CHARS <= len(f) <= _FRAGMENT_MAX_CHARS):
        return False
    if _has_numeral(f):
        return False              # a number could be fabricated
    if _RESULT_ASSERTION_RE.search(f):
        return False              # asserts an outcome, not just an intent
    if (_CLAUSE_BREAK_RE.search(f) or _SUBORDINATOR_RE.search(f)
            or _COPULA_RE.search(f)):
        return False              # a second predication can hide there
    if not _NONASSERTIVE_CLAUSE_RE.match(f):
        return False              # must OPEN with the marker, one clause
    return _atom_present(f, False, claim_canon)


def _remainder_is_framing(issue: str, frag: str) -> bool:
    """True when the objection says nothing beyond 'this is a plan'.

    This is the guard that stops a REAL complaint riding along with a
    quoted next step (review C1): every word left after removing the
    fragment must be framing vocabulary, and any digit disqualifies.
    """
    rest = str(issue or "")
    i = rest.find(frag)
    if i >= 0:
        rest = rest[:i] + " " + rest[i + len(frag):]
    if _has_numeral(rest):
        return False
    # Single letters are the debris of possessives and contractions
    # ("the reply's statement" -> reply, s) and carry no complaint
    # semantics, so they are not treated as content words.
    return all(w.lower() in _FRAMING_WORDS
               for w in _WORD_RE.findall(rest) if len(w) > 1)


def _nonassertive_fragment(issue: str, claim: str) -> Optional[str]:
    """The claim fragment this objection targets, IF the objection reduces
    ENTIRELY to that fragment and the fragment asserts nothing.

    Proof discipline (this module dismisses only by looking): the fragment
    must appear VERBATIM in the claim — quoted inside the objection, or the
    objection text itself being that quote. A judge's paraphrase is never
    accepted, because a paraphrase cannot be checked.
    """
    text = str(issue or "").strip()
    claim_s = str(claim or "")
    if not text or not claim_s:
        return None
    c_canon = _canon(claim_s)

    # (a) fragments the objection quotes explicitly — the rest of the
    # objection must be framing only.
    for m in _QUOTED_SPAN_RE.finditer(text):
        frag = next((g for g in m.groups() if g), "")
        if (_fragment_is_nonassertive(frag, c_canon)
                and _remainder_is_framing(text, frag)):
            return frag
    # (b) the objection IS a bare quote of the reply (the judge's "issues"
    # list often just echoes the offending sentence). Nothing remains to
    # check, but the clause shape still must hold.
    if _fragment_is_nonassertive(text, c_canon):
        return text
    return None


def resolve_issue(issue: str, claim: str, evidence: str,
                  truncation_severity: float = 0.0) -> Tuple[str, str]:
    """Adjudicate ONE objection mechanically. Returns (decision, why)."""
    text = str(issue or "").strip()
    if not text:
        return (UNRESOLVED, "empty issue")

    is_absence = bool(_ABSENCE_RE.search(text))
    # ⚠ An ABSENCE complaint is never a numeric dispute, and the exclusion
    # is load-bearing rather than tidy. `_CONTRAST_RE` matches a bare
    # "not", which every absence complaint contains ("… is not in the
    # evidence"), so without this rule 1 adjudicated them as numeric
    # contradictions and UPHELD them — resurrecting, in the protected
    # no-model-call path, exactly the deleted rule 4 whose measured harm
    # (false alarms 2 → 7) is recorded below. "34°C and humidity 28% are
    # not in the evidence" asserts a MISSING fact, not a wrong one; only
    # rule 2 can settle it, by looking. The synonym list in `_ABSENCE_RE`
    # is what keeps this exclusion honest — see the warning there.
    has_contrast = bool(_CONTRAST_RE.search(text)) and not is_absence
    nums = _numbers_with_units(text)

    # ── 1. Numeric dispute stated inside the objection itself.
    if len(nums) >= 2 and has_contrast:
        if _has_label_digits(text):
            return (UNRESOLVED, "version-labelled numbers need judgement")
        related_pair = None
        contra_pairs = []
        gray = 0
        for (a, ra, ua), (b, rb, ub) in itertools.combinations(nums, 2):
            grade = _grade_pair(a, ra, ua, b, rb, ub)
            if grade == _RELATED and related_pair is None:
                related_pair = (ra, rb)
            elif grade == _CONTRA:
                contra_pairs.append((a, ra, b, rb))
            elif grade == _GRAY:
                gray += 1
        if related_pair and contra_pairs:
            # ⚠ Mixed signals: the old code dismissed on the FIRST related
            # pair, so "100 warnings ≈ 96" shielded a genuine 42-vs-100
            # dispute cited in the same breath. Innocence of one pair is
            # not innocence of the objection.
            return (UNRESOLVED,
                    "mixed numeric signals (one pair consistent, another "
                    "contradictory) need judgement")
        if related_pair and gray:
            # ⚠ GRAY blocks dismissal exactly as CONTRA does (round-2
            # C1): gray means "round-shaped but unproven suspicion", and
            # one innocent unit conversion cited beside a 20%-off
            # latency figure was dismissing the WHOLE objection — the
            # round-1 "first related pair shields the dispute" bug, one
            # grade over. Innocence of one pair is not innocence of the
            # objection.
            return (UNRESOLVED,
                    "mixed numeric signals (one pair consistent, another "
                    "unproven) need judgement")
        if related_pair:
            ra, rb = related_pair
            return (DISMISS, f"rounding/precision/unit: {ra} ≈ {rb}")
        if contra_pairs:
            # ⚠ ANCHOR before convicting: the judge's own sentence is not
            # evidence. "claim states 500 users, whereas the evidence
            # shows 3" was upheld even when claim AND evidence both said
            # 3 — the 500 was hallucinated. A contradiction counts only
            # when one side is found in the claim and the other in the
            # evidence.
            c_claim = _canon(claim)
            c_ev = _canon(_strip_packer_marks(evidence))
            for a, ra, b, rb in contra_pairs[:10]:
                atom_a, atom_b = ra.replace(",", ""), rb.replace(",", "")
                in_cl = {atom_a: _number_present(atom_a, c_claim),
                         atom_b: _number_present(atom_b, c_claim)}
                in_ev = {atom_a: _number_present(atom_a, c_ev),
                         atom_b: _number_present(atom_b, c_ev)}
                # ⚠ ASYMMETRIC anchoring (round-2 refinement of the
                # exclusive-anchoring fix): the CLAIM-side figure must be
                # claim-EXCLUSIVE (in the claim, NOT in the evidence) —
                # that is what makes it a fabrication — while the
                # evidence-side figure need only be present in the
                # evidence. Requiring the evidence figure to be absent
                # from the claim blocked the most realistic fact_swap of
                # all: a claim that QUOTES the evidence number and then
                # contradicts it ("the evidence says 3 users, so with
                # projections we have 500"). The hallucination shield
                # survives: a judge-invented figure is in NEITHER text,
                # so no assignment satisfies claim-exclusivity, and a
                # figure both sides agree on can never be the convicting
                # one.
                for x, y in ((atom_a, atom_b), (atom_b, atom_a)):
                    if in_cl[x] and not in_ev[x] and in_ev[y]:
                        return (UPHOLD,
                                f"numeric contradiction beyond rounding "
                                f"({ra} vs {rb}, claim-side figure "
                                f"unsupported)")
            return (UNRESOLVED,
                    "numeric mismatch, but the figures are not traceable "
                    "to the claim and evidence")
        # All pairs equal, gray, or incomparable — nothing was PROVEN
        # either way. ⚠ The old code upheld here: an objection whose
        # numbers all AGREED ("that 28% refers to Paris, not Athens" —
        # textbook semantic drift) was convicted as a "numeric
        # contradiction" because the loop skipped equal pairs and fell
        # through to UPHOLD.
        return (UNRESOLVED, "no proven numeric contradiction")

    # ── 2. Absence complaint — is the cited atom actually there?
    if is_absence:
        atoms = _cited_atoms(text)
        if not atoms:
            return (UNRESOLVED, "absence complaint with nothing citable")
        if _CLAIMWARD_RE.search(text):
            c_cl = _canon(claim)
            found_cl = sum(1 for a, n in atoms
                           if _atom_present(a, n, c_cl))
            if found_cl == len(atoms):
                return (DISMISS,
                        "the reply DOES contain the cited fact "
                        "(omission complaint is factually false)")
            return (UNRESOLVED,
                    "claim-side omission — whether it matters needs "
                    "judgement")
        c_ev = _canon(_strip_packer_marks(evidence))
        found = sum(1 for a, n in atoms if _atom_present(a, n, c_ev))
        if found == len(atoms):
            # ⚠ ACCEPTED RISK, stated plainly (2026-08-07 review): the
            # evidence string is partially attacker-controllable (web
            # page text), and a page embedding the cited atom makes this
            # dismissal fire with no model in the loop. This is the
            # verifier's existing trust model — evidence is ground truth,
            # and a page that "supports" the claim also convinces the
            # judge and the strong model — so the mechanical form adds
            # SPEED of compromise, not a new authority. Injection
            # resistance is an egress/ingress-layer property; a
            # confidence cap here would only re-open the false-alarm
            # class this rule exists to close.
            return (DISMISS,
                    "cited fact IS present in the evidence (judge missed it)")
        if found:
            # ⚠ Mixed presence proves neither side: '"28 %" is not in the
            # evidence' with "28%" present used to UPHOLD because the
            # quoted spelling missed while the number hit — a formatting
            # variant convicted as a real defect.
            return (UNRESOLVED,
                    "some cited facts present, some absent — needs judgement")
        if truncation_severity >= _truncation_floor():
            return (UNRESOLVED,
                    f"absent, but {truncation_severity:.0%} of the evidence "
                    f"was cut")
        return (UPHOLD, "cited fact absent from intact evidence")

    # ── 3. Machine-noise allegation — is the noise literally there?
    # Added after the v5 arm measured artifact_leak catches at 0/4: the
    # objection ("the claim contains unflagged diff markers") is neither
    # numeric nor absence-shaped, so it escalated and the overturner
    # destroyed it — the exact class that took the heaviest measured
    # damage (14 kills in one arm). Presence in the CLAIM is a literal
    # string test; nobody's opinion is required.
    # ⚠ THE CONTEXT GATE IS BYPASSED WHEN THE MARKERS ARE DEMONSTRABLY THERE
    # (2026-08-09, measured on the 433-trial re-bench). `_is_noise_allegation`
    # requires a noise WORD next to "artifact"/"marker", because those are
    # ordinary English ("the wrong build artifact" is a content objection).
    # Sound in general — but the cheap judge's issue text for this fault class
    # is frequently the bare word `'artifact'`, which fails the gate, so rule 3
    # never armed, the objection escalated unresolved, and the strong model
    # dismissed a complaint that carried no argument. Measured cost: 23 of
    # artifact_leak's 58 trials overturned, every one wrongly, dragging its
    # catch rate to 0.431 from the 0.828 the cheap judge actually achieved.
    #
    # The ambiguity the gate protects against CANNOT ARISE once the claim is
    # known to contain the markers: "the wrong build artifact" is a content
    # objection about a claim that has no diff markers in it. So when
    # `_claim_noise_markers` finds them, a bare artifact word is accepted as
    # the allegation. `_claim_noise_markers` is itself the conservative half
    # (fences stripped, diff-header shape required) and is unchanged.
    _present = _claim_noise_markers(claim)
    if _is_noise_allegation(text) or (_present and _ARTIFACT_WORD_RE.search(text)):
        if _present:
            return (UPHOLD,
                    f"machine noise literally present in the claim: "
                    f"{_present[0]!r}")
        return (DISMISS, "alleged machine noise is NOT in the claim")

    # ── 4. Non-assertive fragment — the objection targets a part of the
    # reply that makes no claim about the world, so no evidence can
    # contradict it (§4BD-b 2026-08-12). Two shapes: a STATED NEXT STEP
    # ("I'll open the parser next") and a QUESTION back to the user
    # ("Want me to look at the service log?").
    #
    # Motivation: after retiring the GEPA adjudicate artifact, the residual
    # false-refutes on the honest-failure FP-trap set were dominated by
    # future-plan complaints, and prompt text does not fix it (the E4B judge
    # reads the rule and does not follow it, measured p=1.0) — so the fix
    # had to be mechanical.
    #
    # PROOF REQUIRED, in this module's tradition — it dismisses only by
    # LOOKING: the targeted fragment must be found VERBATIM in the claim
    # (quoted inside the objection, or the objection IS the quote), and the
    # objection must reduce ENTIRELY to it (`_remainder_is_framing`).
    #
    # ⚠ WHAT THE GUARDS ARE, AND WHAT THEY ARE NOT. `_fragment_is_nonassertive`
    # rejects numerals, inflected result verbs, clause breaks, subordinators
    # and copulas — every one of them earned by a laundering input an
    # adversarial review actually produced (an objection citing a real
    # contradiction beside a quoted plan; "I'll retry the fetch; every exit
    # node is blocked"; "I'll fix the bug that broke prod"; `passed` vetoed
    # while `passes` was not). They are a LEXICAL proxy for a SEMANTIC
    # property, which is why this rule ships DEFAULT-OFF — see
    # `nonassertive_enabled` for the decision and what would reverse it.
    # And it decides ONE objection: `resolve_refute` still needs EVERY
    # objection dismissed before a refute is dropped.
    # ⚠ CALLERS MUST GATE. This returns DISMISS regardless of
    # `nonassertive_enabled()` — the flag is enforced at the two verifier
    # call sites, not here, so a future third caller would inherit the rule
    # silently (the silent-inoperative-subsystem shape, in reverse). Any new
    # consumer of a rule-4 DISMISS must check `nonassertive_enabled()` and
    # `nonassertive_dismissal()` first.
    _frag = _nonassertive_fragment(text, claim)
    if _frag:
        _shape = "question" if _frag.rstrip().endswith("?") else "stated next step"
        return (DISMISS,
                f"{_NONASSERTIVE_WHY} ({_shape}) — it asserts nothing the "
                f"evidence could contradict")

    # ── (5) TRIED AND REJECTED ON MEASUREMENT, kept as a warning:
    # upholding whenever an objection merely mentions two DIFFERENT
    # numbers — no contrast phrase required — raised protection from 37
    # to 59 catches but hardened false alarms from 2 to 7, because an
    # issue that simply quotes "34°C and humidity 28%" contains two
    # differing numbers and says nothing about a contradiction. Net wash
    # on the balanced score, worse on the axis the operator cares about.
    # The CONTRAST FRAME is what makes a numeric mismatch an assertion;
    # without it there is no proof, and this module dismisses and upholds
    # only on proof.

    return (UNRESOLVED, "not mechanically decidable")


def resolve_refute(issues: Sequence[str], claim: str, evidence: str,
                   truncation_severity: float = 0.0
                   ) -> Tuple[Optional[str], List[str], List[str]]:
    """Adjudicate a whole REFUTED verdict's issue list.

    Returns ``(decision, reasons, unresolved_issues)`` where decision is:
      * UPHOLD   — at least one objection is PROVEN real; the refute
                   stands and needs no main-model call;
      * DISMISS  — every objection is PROVEN a false alarm; the refute
                   can be dropped without a main-model call;
      * None     — some objection needs judgement; escalate as before
                   (``unresolved_issues`` is what actually needs it).
    """
    items = [str(i or "").strip() for i in (issues or []) if str(i or "").strip()]
    if not items:
        return (None, [], [])
    reasons: List[str] = []
    unresolved: List[str] = []
    dismissed = 0
    for issue in items:
        decision, why = resolve_issue(issue, claim, evidence,
                                      truncation_severity)
        if decision == UPHOLD:
            return (UPHOLD, [f"{issue} → {why}"], [])
        if decision == DISMISS:
            dismissed += 1
            reasons.append(f"{issue} → {why}")
        else:
            unresolved.append(issue)
    if dismissed and not unresolved:
        return (DISMISS, reasons, [])
    return (None, reasons, unresolved)


__all__ = ["DISMISS", "UPHOLD", "UNRESOLVED", "enabled",
           "dismiss_enabled", "nonassertive_enabled", "nonassertive_dismissal",
           "resolve_issue", "resolve_refute"]
